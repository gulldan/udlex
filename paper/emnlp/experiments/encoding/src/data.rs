//! Real key sets and query streams: corpus forms from `train.jsonl`, the
//! dictionary union from the inventory's sources, queries from `test.jsonl`.

use std::io::{self, BufRead};
use std::path::Path;

use hashbrown::{HashMap, HashTable};
use rayon::prelude::*;
use serde_json::Value;

use crate::hashing::{SplitMix, key_hash};
use crate::strs::Strs;

/// Forms sorted bytewise, each with the id of its lemma in `lemmas`.
pub struct KeySet {
    pub keys: Strs,
    pub lemma: Vec<u32>,
    pub lemmas: Strs,
}

#[derive(Default)]
struct Interner {
    strs: Strs,
    table: HashTable<u32>,
}

impl Interner {
    fn new() -> Self {
        Self {
            strs: Strs::new(),
            table: HashTable::new(),
        }
    }

    fn find(&self, s: &[u8]) -> Option<u32> {
        let strs = &self.strs;
        self.table
            .find(key_hash(s, 0), |&i| strs.get(i as usize) == s)
            .copied()
    }

    fn intern(&mut self, s: &[u8]) -> u32 {
        if let Some(id) = self.find(s) {
            return id;
        }
        let id = u32::try_from(self.strs.len()).expect("more than 2^32 strings");
        self.strs.push(s);
        let strs = &self.strs;
        self.table
            .insert_unique(key_hash(s, 0), id, |&i| key_hash(strs.get(i as usize), 0));
        id
    }
}

/// yada ends keys with NUL and entries carry `u16` lengths.
fn usable(s: &[u8]) -> bool {
    !s.is_empty() && s.len() <= usize::from(u16::MAX) && !s.contains(&0)
}

fn sorted_ids(strs: &Strs) -> Vec<u32> {
    let mut order: Vec<u32> = (0..strs.len() as u32).collect();
    order.par_sort_unstable_by(|&a, &b| strs.get(a as usize).cmp(strs.get(b as usize)));
    order
}

fn finalize(forms: Interner, lemma_of: Vec<u32>, lemmas: Interner) -> KeySet {
    let lemma_order = sorted_ids(&lemmas.strs);
    let mut lemma_rank = vec![0_u32; lemma_order.len()];
    let mut sorted_lemmas = Strs::new();
    for (rank, &old) in lemma_order.iter().enumerate() {
        lemma_rank[old as usize] = rank as u32;
        sorted_lemmas.push(lemmas.strs.get(old as usize));
    }
    let mut keys = Strs::new();
    let mut lemma = Vec::with_capacity(lemma_of.len());
    for old in sorted_ids(&forms.strs) {
        keys.push(forms.strs.get(old as usize));
        lemma.push(lemma_rank[lemma_of[old as usize] as usize]);
    }
    KeySet {
        keys,
        lemma,
        lemmas: sorted_lemmas,
    }
}

fn each_sentence(path: &Path, mut f: impl FnMut(&Value)) -> io::Result<()> {
    for line in io::BufReader::new(std::fs::File::open(path)?).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(&line)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{path:?}: {e}")))?;
        f(&row);
    }
    Ok(())
}

/// The corpus table: every training form with its most frequent lemma
/// (ties go to the bytewise smaller lemma).
pub fn corpus_keyset(train: &Path) -> io::Result<KeySet> {
    let mut forms = Interner::new();
    let mut lemmas = Interner::new();
    let mut votes: HashMap<(u32, u32), u32> = HashMap::new();
    each_sentence(train, |row| {
        let (Some(tokens), Some(gold)) = (row["tokens"].as_array(), row["lemmas"].as_array())
        else {
            return;
        };
        for (token, lemma) in tokens.iter().zip(gold) {
            let (Some(token), Some(lemma)) = (token.as_str(), lemma.as_str()) else {
                continue;
            };
            if usable(token.as_bytes()) && usable(lemma.as_bytes()) {
                let pair = (
                    forms.intern(token.as_bytes()),
                    lemmas.intern(lemma.as_bytes()),
                );
                *votes.entry(pair).or_default() += 1;
            }
        }
    })?;
    let mut best: Vec<Option<(u32, u32)>> = vec![None; forms.strs.len()];
    for (&(form, lemma), &count) in &votes {
        let slot = &mut best[form as usize];
        let better = slot.is_none_or(|(held, held_count)| {
            count > held_count
                || (count == held_count
                    && lemmas.strs.get(lemma as usize) < lemmas.strs.get(held as usize))
        });
        if better {
            *slot = Some((lemma, count));
        }
    }
    let lemma_of = best.into_iter().map(|b| b.expect("voted").0).collect();
    Ok(finalize(forms, lemma_of, lemmas))
}

/// One dictionary file of the inventory.
pub struct Source {
    pub path: String,
    pub lemma_first: bool,
    pub priority: i64,
}

pub fn sources(inventory: &Path, iso: &str) -> io::Result<Vec<Source>> {
    let text = std::fs::read_to_string(inventory)?;
    let root: Value = serde_json::from_str(&text)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    let mut out: Vec<Source> = root["languages"][iso]["sources"]
        .as_array()
        .map(|list| {
            list.iter()
                .filter_map(|s| {
                    Some(Source {
                        path: s["path"].as_str()?.to_owned(),
                        lemma_first: s["lemma_first"].as_bool().unwrap_or(false),
                        priority: s["priority"].as_i64().unwrap_or(i64::MAX),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    // Stable: equal priorities keep the inventory's order.
    out.sort_by_key(|s| s.priority);
    Ok(out)
}

/// A compiled bundle table as it ships: every form with its lemma, read
/// back from `<lang dir>/.generations/<current>/<table>.<codec>`. Returns
/// `None` when the generation has no such table, plus where it was read.
pub fn bundle_keyset(lang_dir: &Path, table: &str) -> io::Result<Option<(KeySet, String)>> {
    let bad = |what: String| io::Error::new(io::ErrorKind::InvalidData, what);
    let current = std::fs::read_to_string(lang_dir.join("current"))?;
    let generation = lang_dir.join(".generations").join(current.trim());
    let meta: Value = serde_json::from_str(&std::fs::read_to_string(generation.join("meta.json"))?)
        .map_err(|e| bad(format!("meta.json: {e}")))?;
    let Some(codec) = meta["codecs"][table].as_str() else {
        return Ok(None);
    };
    let declared = meta["entries"][table].as_u64().unwrap_or(0);
    let blob = std::fs::read(generation.join("strings.txt"))?;
    let strings: Vec<&[u8]> = blob.split(|&b| b == b'\n').collect();
    let bytes = std::fs::read(generation.join(format!("{table}.{codec}")))?;
    let mut rows: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    let id_row = |key: &[u8], id: u64| -> io::Result<(Vec<u8>, Vec<u8>)> {
        let lemma = strings
            .get(usize::try_from(id).unwrap_or(usize::MAX))
            .ok_or_else(|| bad(format!("string id {id} out of range")))?;
        Ok((key.to_vec(), lemma.to_vec()))
    };
    match codec {
        "map" => {
            use fst::Streamer;
            let map = fst::Map::new(bytes).map_err(|e| bad(e.to_string()))?;
            let mut stream = map.stream();
            while let Some((key, id)) = stream.next() {
                rows.push(id_row(key, id)?);
            }
        }
        "inline" => {
            use fst::Streamer;
            let fsa = fst::raw::Fst::new(bytes).map_err(|e| bad(e.to_string()))?;
            let mut stream = fsa.stream();
            while let Some((key, _)) = stream.next() {
                // form NUL cut(LEB128) suffix: lemma = form minus cut bytes, plus suffix.
                let at = key
                    .iter()
                    .position(|&b| b == 0)
                    .ok_or_else(|| bad("inline key without separator".into()))?;
                let (form, rest) = (&key[..at], &key[at + 1..]);
                let (mut cut, mut shift, mut used) = (0_usize, 0_u32, 0_usize);
                for &byte in rest {
                    cut |= usize::from(byte & 0x7F) << shift;
                    used += 1;
                    shift += 7;
                    if byte & 0x80 == 0 {
                        break;
                    }
                }
                let stem = form
                    .len()
                    .checked_sub(cut)
                    .ok_or_else(|| bad("inline cut exceeds the form".into()))?;
                rows.push((form.to_vec(), [&form[..stem], &rest[used..]].concat()));
            }
        }
        "da" => {
            let units: Vec<u32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes(*c))
                .collect();
            let offset = |unit: u32| (unit >> 10) << ((unit & (1 << 9)) >> 6);
            // Depth-first over (position, unit, key so far), as the core walks it.
            let mut stack = vec![(0_u32, units[0], Vec::new())];
            while let Some((position, unit, key)) = stack.pop() {
                let base = offset(unit) ^ position;
                if unit & 0x100 != 0 {
                    let leaf = units[base as usize];
                    rows.push(id_row(&key, u64::from(leaf & 0x7FFF_FFFF))?);
                }
                for byte in 1..=255_u32 {
                    let child = base ^ byte;
                    if let Some(&next) = units.get(child as usize)
                        && next & 0x8000_00FF == byte
                    {
                        let mut longer = key.clone();
                        longer.push(byte as u8);
                        stack.push((child, next, longer));
                    }
                }
            }
        }
        other => {
            return Err(bad(format!(
                "{table}: codec {other} is not a dictionary table"
            )));
        }
    }
    let mut forms = Interner::new();
    let mut lemmas = Interner::new();
    let mut lemma_of: Vec<u32> = Vec::new();
    for (form, lemma) in &rows {
        if !usable(form) || !usable(lemma) {
            continue;
        }
        let id = forms.intern(form);
        if id as usize == lemma_of.len() {
            lemma_of.push(lemmas.intern(lemma));
        }
    }
    if lemma_of.len() as u64 != declared {
        return Err(bad(format!(
            "{table}: read {} forms, meta.json declares {declared}",
            lemma_of.len()
        )));
    }
    let origin = format!("{} {}.{codec}", generation.display(), table);
    Ok(Some((finalize(forms, lemma_of, lemmas), origin)))
}

/// The dictionary union: sources in priority order, the first lemma seen for
/// a form wins. This reproduces the key set, not the core's merge policy.
pub fn external_keyset(repo: &Path, sources: &[Source]) -> io::Result<KeySet> {
    let mut forms = Interner::new();
    let mut lemmas = Interner::new();
    let mut lemma_of: Vec<u32> = Vec::new();
    for source in sources {
        let mut reader = io::BufReader::new(std::fs::File::open(repo.join(&source.path))?);
        let mut line = Vec::new();
        loop {
            line.clear();
            if reader.read_until(b'\n', &mut line)? == 0 {
                break;
            }
            // crawdad walks chars, so a row must be text.
            let Ok(text) = std::str::from_utf8(&line) else {
                continue;
            };
            let mut cols = text.trim_end_matches(['\n', '\r']).split('\t');
            let (Some(first), Some(second)) = (cols.next(), cols.next()) else {
                continue;
            };
            let (form, lemma) = if source.lemma_first {
                (second, first)
            } else {
                (first, second)
            };
            if !usable(form.as_bytes()) || !usable(lemma.as_bytes()) {
                continue;
            }
            let id = forms.intern(form.as_bytes());
            if id as usize == lemma_of.len() {
                lemma_of.push(lemmas.intern(lemma.as_bytes()));
            }
        }
    }
    Ok(finalize(forms, lemma_of, lemmas))
}

pub fn test_tokens(test: &Path) -> io::Result<Strs> {
    let mut out = Strs::new();
    each_sentence(test, |row| {
        for token in row["tokens"].as_array().into_iter().flatten() {
            if let Some(token) = token.as_str().filter(|t| usable(t.as_bytes())) {
                out.push(token.as_bytes());
            }
        }
    })?;
    Ok(out)
}

/// The reference answer for a key set.
pub struct Oracle<'k> {
    map: HashMap<&'k [u8], u32>,
    set: &'k KeySet,
}

impl<'k> Oracle<'k> {
    pub fn new(set: &'k KeySet) -> Self {
        let mut map = HashMap::with_capacity(set.keys.len());
        for (i, key) in set.keys.iter().enumerate() {
            map.insert(key, set.lemma[i]);
        }
        Self { map, set }
    }

    pub fn lemma(&self, key: &[u8]) -> Option<&'k [u8]> {
        self.map
            .get(key)
            .map(|&id| self.set.lemmas.get(id as usize))
    }
}

/// What a lookup must fold into the loop's accumulator; computable from the
/// oracle's lemma, so a pass checks itself without the oracle in memory.
#[inline]
pub fn sink_bytes(lemma: Option<&[u8]>) -> u64 {
    match lemma {
        None => 0,
        Some(l) => crate::dict::Lemma { head: l, tail: &[] }.sink(),
    }
}

pub struct Workload {
    pub name: String,
    pub queries: Strs,
    pub checksum: u64,
    pub hits: usize,
}

fn workload(name: &str, queries: Strs, oracle: &Oracle) -> Workload {
    let mut checksum = 0_u64;
    let mut hits = 0;
    for q in queries.iter() {
        let lemma = oracle.lemma(q);
        hits += usize::from(lemma.is_some());
        checksum = checksum.wrapping_add(sink_bytes(lemma));
    }
    Workload {
        name: name.to_owned(),
        queries,
        checksum,
        hits,
    }
}

fn sample(from: &Strs, n: usize, rng: &mut SplitMix) -> Strs {
    let mut out = Strs::new();
    for _ in 0..if from.is_empty() { 0 } else { n } {
        out.push(from.get(rng.below(from.len())));
    }
    out
}

/// A word-shaped non-key: a key whose last character comes from another key.
fn near_miss(set: &KeySet, oracle: &Oracle, rng: &mut SplitMix) -> Vec<u8> {
    let last_char = |s: &[u8]| {
        let mut at = s.len() - 1;
        while at > 0 && s[at] & 0xC0 == 0x80 {
            at -= 1;
        }
        at
    };
    for _ in 0..16 {
        let a = set.keys.get(rng.below(set.keys.len()));
        let b = set.keys.get(rng.below(set.keys.len()));
        let mut made = a[..last_char(a)].to_vec();
        made.extend_from_slice(&b[last_char(b)..]);
        if oracle.lemma(&made).is_none() {
            return made;
        }
    }
    let mut made = set.keys.get(rng.below(set.keys.len())).to_vec();
    made.push(1);
    while oracle.lemma(&made).is_some() {
        made.push(1);
    }
    made
}

pub const SWEEP: [u32; 5] = [10, 25, 50, 75, 90];
const LENGTHS: usize = 32;

/// How many queries the synthetic workloads draw.
#[derive(Clone, Copy)]
pub struct Sizes {
    pub uniform: usize,
    pub mixed: usize,
}

/// Every workload of one key set, in a fixed order.
///
/// A UD test set replayed verbatim is short enough for a branch predictor
/// to learn by heart (measured: zero mispredictions per miss on a 3 000
/// token stream), so the streams that carry the conclusions are redrawn:
/// * `iid` — independent draws from the test tokens: their Zipf law and
///   natural miss rate, in an order that cannot be memorised;
/// * `iid_hit`, `iid_miss` — the same from the stream's hits and misses;
/// * `sweep_pNN` — the two mixed at a set miss rate;
/// * `len_NN` — draws from the stream's hits of exactly `NN` bytes;
/// * `uni_hit`, `uni_miss` — keys drawn uniformly and word-shaped non-keys,
///   the way the hashing literature measures;
/// * `real` — the verbatim replay, kept as the reference for that artefact.
///
/// `extra` adds streams that only make sense for one table, redrawn too.
pub fn workloads(
    set: &KeySet,
    oracle: &Oracle,
    real: Strs,
    extra: Vec<(String, Strs)>,
    sizes: Sizes,
) -> Vec<Workload> {
    let mut rng = SplitMix(0x5EED);
    let (mut hit, mut miss) = (Strs::new(), Strs::new());
    for q in real.iter() {
        if oracle.lemma(q).is_some() {
            hit.push(q);
        } else {
            miss.push(q);
        }
    }
    let uni_hit = sample(&set.keys, sizes.uniform, &mut rng);
    let mut uni_miss = Strs::new();
    for _ in 0..sizes.uniform {
        uni_miss.push(&near_miss(set, oracle, &mut rng));
    }
    // A test with no hits (or no misses) borrows the uniform stream instead.
    let hit_pool = if hit.is_empty() { &uni_hit } else { &hit };
    let miss_pool = if miss.is_empty() { &uni_miss } else { &miss };
    let mut out = Vec::new();
    for p in SWEEP {
        let mut mixed = Strs::new();
        for _ in 0..sizes.mixed {
            let pool = if rng.below(100) < p as usize {
                miss_pool
            } else {
                hit_pool
            };
            mixed.push(pool.get(rng.below(pool.len())));
        }
        out.push(workload(&format!("sweep_p{p:03}"), mixed, oracle));
    }
    // Where a per-byte walk and a fixed-price hash cross is read off these.
    for len in 1..=LENGTHS {
        let mut bucket = Strs::new();
        hit.iter()
            .filter(|q| q.len() == len)
            .for_each(|q| bucket.push(q));
        if bucket.len() >= 200 {
            let drawn = sample(&bucket, sizes.mixed / 4, &mut rng);
            out.push(workload(&format!("len_{len:02}"), drawn, oracle));
        }
    }
    let iid_hit = sample(hit_pool, sizes.mixed, &mut rng);
    let iid_miss = sample(miss_pool, sizes.mixed, &mut rng);
    out.push(workload("iid_hit", iid_hit, oracle));
    out.push(workload("iid_miss", iid_miss, oracle));
    out.push(workload("uni_hit", uni_hit, oracle));
    out.push(workload("uni_miss", uni_miss, oracle));
    let iid = sample(&real, sizes.uniform, &mut rng);
    out.push(workload("iid", iid, oracle));
    out.push(workload("real", real, oracle));
    for (name, queries) in extra {
        let drawn = sample(&queries, sizes.mixed, &mut rng);
        out.push(workload(&name, drawn, oracle));
    }
    out.retain(|w| !w.queries.is_empty());
    out
}

/// The probes the cascade sends to the dictionary table: tokens the corpus
/// table did not answer, each tried as written and then lowercased.
pub fn cascade_probes(tokens: &Strs, corpus: impl Fn(&[u8]) -> bool, external: &Oracle) -> Strs {
    let mut out = Strs::new();
    for token in tokens.iter() {
        let text = std::str::from_utf8(token).expect("JSON strings are UTF-8");
        let lower = text.to_lowercase();
        if corpus(token) || corpus(lower.as_bytes()) {
            continue;
        }
        out.push(token);
        if external.lemma(token).is_none() && lower.as_bytes() != token {
            out.push(lower.as_bytes());
        }
    }
    out
}
