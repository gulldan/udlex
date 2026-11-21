//! The one interface every candidate answers through, and the value layers
//! that turn an index payload into a lemma.
//!
//! Every lab lookup is `#[inline(always)]`, so each candidate's loop is one
//! function and none wins or loses by an inlining heuristic. The core's own
//! `HybridTable::get` keeps its `#[inline]` hint: that is the shipped code.

use std::path::Path;

use hashbrown::HashMap;

use crate::data::KeySet;
use crate::img::{Img, as_bytes, write_sections};

/// A lemma as two borrowed pieces: `head ++ tail`. String layers answer with
/// an empty tail; the rule layer answers a prefix of the query plus a suffix.
#[derive(Clone, Copy)]
pub struct Lemma<'a> {
    pub head: &'a [u8],
    pub tail: &'a [u8],
}

impl Lemma<'_> {
    /// Length, first and last byte: forces the loads that deliver the lemma.
    #[inline(always)]
    pub fn sink(&self) -> u64 {
        let len = (self.head.len() + self.tail.len()) as u64;
        let first = self
            .head
            .first()
            .or(self.tail.first())
            .copied()
            .unwrap_or(0);
        let last = self.tail.last().or(self.head.last()).copied().unwrap_or(0);
        1 + len * 31 + u64::from(first) * 7 + u64::from(last) * 13
    }

    /// Whether the two pieces spell exactly `want`.
    pub fn is(&self, want: &[u8]) -> bool {
        want.len() == self.head.len() + self.tail.len()
            && want.starts_with(self.head)
            && want.ends_with(self.tail)
    }
}

/// `get(form) -> lemma`: the contract all candidates share.
pub trait Dict {
    fn get<'a>(&'a self, key: &'a [u8]) -> Option<Lemma<'a>>;
}

/// The same dictionary behind a real call: what a candidate costs when its
/// lookup is not inlined into the caller, as the core's `Table::get` is not.
pub struct Outlined<D>(pub D);

impl<D: Dict> Dict for Outlined<D> {
    #[inline(never)]
    fn get<'a>(&'a self, key: &'a [u8]) -> Option<Lemma<'a>> {
        self.0.get(key)
    }
}

/// A key structure alone: form to a 31-bit payload.
pub trait Index {
    fn find(&self, key: &[u8]) -> Option<u32>;
}

/// A value layer: payload (and the query) to the lemma.
pub trait Values {
    fn lemma<'a>(&'a self, payload: u32, key: &'a [u8]) -> Lemma<'a>;
}

pub struct With<I, V>(pub I, pub V);

impl<I: Index, V: Values> Dict for With<I, V> {
    #[inline(always)]
    fn get<'a>(&'a self, key: &'a [u8]) -> Option<Lemma<'a>> {
        let payload = self.0.find(key)?;
        Some(self.1.lemma(payload, key))
    }
}

/// Payload = string id, resolved as the core does: `strings.idx` for a pair
/// of boundaries, then `strings.txt` — two dependent loads.
pub struct Ids {
    idx: &'static [u32],
    txt: &'static [u8],
    _img: Img,
}

impl Values for Ids {
    #[inline(always)]
    fn lemma<'a>(&'a self, payload: u32, _key: &'a [u8]) -> Lemma<'a> {
        // SAFETY: payloads are ids below the entry count; boundaries ascend
        // inside `txt` and every entry ends with a newline.
        let head = unsafe {
            let start = *self.idx.get_unchecked(payload as usize) as usize;
            let end = *self.idx.get_unchecked(payload as usize + 1) as usize;
            self.txt.get_unchecked(start..end - 1)
        };
        Lemma { head, tail: &[] }
    }
}

/// Payload = byte offset of `[len u16][lemma]`: one dependent load.
pub struct Direct {
    blob: &'static [u8],
    _img: Option<Img>,
}

impl Direct {
    /// A layer over a blob another structure owns: the hash side's entries.
    pub fn over(blob: &'static [u8]) -> Self {
        Self { blob, _img: None }
    }
}

impl Values for Direct {
    #[inline(always)]
    fn lemma<'a>(&'a self, payload: u32, _key: &'a [u8]) -> Lemma<'a> {
        // SAFETY: payloads are offsets of length-prefixed entries in `blob`.
        let head = unsafe {
            let at = self.blob.as_ptr().add(payload as usize);
            let len = usize::from(at.cast::<u16>().read_unaligned());
            std::slice::from_raw_parts(at.add(2), len)
        };
        Lemma { head, tail: &[] }
    }
}

/// `Direct` with an identity flag: a payload with `FLAG` set means the
/// lemma is the form itself, and nothing loads at all.
pub struct Flagged(pub Direct);

/// yada keeps 31 value bits, so the identity flag is bit 30.
pub const FLAG: u32 = 1 << 30;

impl Values for Flagged {
    #[inline(always)]
    fn lemma<'a>(&'a self, payload: u32, key: &'a [u8]) -> Lemma<'a> {
        if payload & FLAG != 0 {
            return Lemma {
                head: key,
                tail: &[],
            };
        }
        self.0.lemma(payload, key)
    }
}

/// Payload = rule id: cut `k` bytes off the form, append a suffix. The rule
/// table is small enough to stay in cache; the stem is the query itself.
pub struct Rules {
    recs: &'static [u64],
    sfx: &'static [u8],
    _img: Img,
}

impl Values for Rules {
    #[inline(always)]
    fn lemma<'a>(&'a self, payload: u32, key: &'a [u8]) -> Lemma<'a> {
        // SAFETY: payloads are rule ids and a rule's suffix lies in `sfx`.
        // The cut saturates: behind a fingerprint-only index a non-key may
        // be handed the rule of a longer form.
        unsafe {
            let rec = *self.recs.get_unchecked(payload as usize);
            let cut = (rec & 0xFFFF) as usize;
            let len = ((rec >> 16) & 0xFFFF) as usize;
            let at = (rec >> 32) as usize;
            Lemma {
                head: key.get_unchecked(..key.len().saturating_sub(cut)),
                tail: self.sfx.get_unchecked(at..at + len),
            }
        }
    }
}

/// The payload of every key under each layer, and what the layers weigh.
pub struct Payloads {
    pub ids: Vec<u32>,
    pub off: Vec<u32>,
    pub offid: Vec<u32>,
    pub rule: Vec<u32>,
    pub ids_bytes: u64,
    pub off_bytes: u64,
    pub rule_bytes: u64,
    pub rules: usize,
    pub identity_keys: usize,
}

/// The payloads of a key set under every layer; with `dir`, also writes the
/// three value images there.
pub fn build_values(set: &KeySet, dir: Option<&Path>) -> std::io::Result<Payloads> {
    let n = set.keys.len();
    // ids: the core's strings.txt / strings.idx.
    let mut txt = Vec::with_capacity(set.lemmas.blob_len() + set.lemmas.len());
    let mut idx = vec![0_u32];
    // off: [len u16][lemma], deduplicated.
    let mut direct = Vec::with_capacity(set.lemmas.blob_len() + 2 * set.lemmas.len());
    let mut direct_at = Vec::with_capacity(set.lemmas.len());
    for lemma in set.lemmas.iter() {
        txt.extend_from_slice(lemma);
        txt.push(b'\n');
        idx.push(u32::try_from(txt.len()).expect("strings.txt exceeds 4 GiB"));
        direct_at.push(u32::try_from(direct.len()).expect("lemma blob exceeds 4 GiB"));
        direct.extend_from_slice(&(lemma.len() as u16).to_le_bytes());
        direct.extend_from_slice(lemma);
    }
    assert!(
        direct.len() < FLAG as usize,
        "lemma blob collides with the identity flag"
    );
    assert!(set.lemmas.len() < FLAG as usize);
    let (mut ids_bytes, mut off_bytes) = (0, 0);
    if let Some(dir) = dir {
        ids_bytes = write_sections(&dir.join("values.ids.img"), &[as_bytes(&idx), &txt])?;
        off_bytes = write_sections(&dir.join("values.off.img"), &[&direct])?;
    }

    let mut rule_ids: HashMap<(u16, &[u8]), u32> = HashMap::new();
    let mut recs: Vec<u64> = Vec::new();
    let mut sfx: Vec<u8> = Vec::new();
    let mut out = Payloads {
        ids: set.lemma.clone(),
        off: Vec::with_capacity(n),
        offid: Vec::with_capacity(n),
        rule: Vec::with_capacity(n),
        ids_bytes,
        off_bytes,
        rule_bytes: 0,
        rules: 0,
        identity_keys: 0,
    };
    for (i, form) in set.keys.iter().enumerate() {
        let lemma = set.lemmas.get(set.lemma[i] as usize);
        let at = direct_at[set.lemma[i] as usize];
        out.off.push(at);
        let same = form == lemma;
        out.identity_keys += usize::from(same);
        out.offid.push(if same { FLAG } else { at });
        let shared = form.iter().zip(lemma).take_while(|(a, b)| a == b).count();
        let rule = ((form.len() - shared) as u16, &lemma[shared..]);
        let next = recs.len() as u32;
        let id = *rule_ids.entry(rule).or_insert_with(|| {
            let rec = u64::from(rule.0) | (rule.1.len() as u64) << 16 | (sfx.len() as u64) << 32;
            recs.push(rec);
            sfx.extend_from_slice(rule.1);
            next
        });
        out.rule.push(id);
    }
    out.rules = recs.len();
    if let Some(dir) = dir {
        out.rule_bytes = write_sections(&dir.join("values.rule.img"), &[as_bytes(&recs), &sfx])?;
    }
    Ok(out)
}

pub fn open_ids(dir: &Path) -> std::io::Result<Ids> {
    let img = Img::open(&dir.join("values.ids.img"))?;
    Ok(Ids {
        idx: img.words(0),
        txt: img.bytes(1),
        _img: img,
    })
}

pub fn open_direct(dir: &Path) -> std::io::Result<Direct> {
    let img = Img::open(&dir.join("values.off.img"))?;
    Ok(Direct {
        blob: img.bytes(0),
        _img: Some(img),
    })
}

pub fn open_rules(dir: &Path) -> std::io::Result<Rules> {
    let img = Img::open(&dir.join("values.rule.img"))?;
    Ok(Rules {
        recs: img.words(0),
        sfx: img.bytes(1),
        _img: img,
    })
}
