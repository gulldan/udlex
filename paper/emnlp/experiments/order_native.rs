//! Full-cascade paired predictions using the 1.0.0 reader, builder and inference probe.
use super::sources;
use crate::normalization::{self, Casing};
use crate::probe::{self, Options, Source};
use crate::train::{self, Pair, Thresholds, Variant, View};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::Path;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
type Words = IndexMap<String, String>;
type Votes = HashMap<String, HashMap<String, i64>>;
const N: usize = 5;
const ARMS: [&str; N] = [
    "declared",
    "learned",
    "single",
    "greedy_subset",
    "exact_subset",
];

#[derive(Deserialize)]
struct Src {
    id: String,
    path: String,
    sha256: String,
    lemma_first: bool,
    unique: bool,
}
#[derive(Deserialize)]
struct Lang {
    sources: Vec<Src>,
    status: Option<String>,
}
#[derive(Deserialize)]
struct Inventory {
    languages: HashMap<String, Lang>,
}
#[derive(Deserialize)]
struct Line {
    tokens: Vec<String>,
    lemmas: Vec<Option<String>>,
    block: String,
}

fn read_lines(path: &Path, mut f: impl FnMut(Line)) -> Result<()> {
    for s in BufReader::new(File::open(path)?).lines() {
        let line: Line = serde_json::from_str(&s?)?;
        if line.tokens.len() != line.lemmas.len() {
            return Err("Unaligned word and gold arrays".into());
        }
        f(line);
    }
    Ok(())
}

fn read_map(root: &Path, s: &Src, iso: &str, wanted: &HashSet<String>) -> Result<Words> {
    let mut words = Words::new();
    let path = root.join(&s.path);
    let integrity = if s.unique {
        sources::stream_normalized_unique(&mut words, &path, s.lemma_first, iso)?
    } else {
        sources::stream_resolved_preferred(&mut words, &path, s.lemma_first, iso)?
    };
    if integrity.sha256 != s.sha256 {
        return Err(format!("Hash changed: {}", s.path).into());
    }
    words.retain(|x, _| wanted.contains(x));
    words.shrink_to_fit();
    Ok(words)
}

#[derive(Clone, Default)]
struct Policy {
    gain: i64,
    inversions: usize,
    order: Vec<usize>,
}

fn preferred(candidate: &Policy, current: &Policy) -> bool {
    candidate.gain > current.gain
        || (candidate.gain == current.gain
            && (candidate.inversions, candidate.order.as_slice())
                < (current.inversions, current.order.as_slice()))
}

fn learn(maps: &[Words], gold: &Votes) -> ([Vec<usize>; N], [i64; N]) {
    let count = maps.len();
    let end = (1_usize << count) - 1;
    // Sum gains by the exact set of dictionaries covering each training form.
    let mut gains = vec![vec![0_i64; end + 1]; count];
    let mut coverage = vec![vec![0_i64; end + 1]; count];
    let mut copy = 0;
    for (form, labels) in gold {
        let identity = labels.get(form).copied().unwrap_or(0);
        copy += identity;
        let mask = maps
            .iter()
            .enumerate()
            .filter(|(_, m)| m.contains_key(form))
            .fold(0, |m, (j, _)| m | 1 << j);
        for (j, source) in maps.iter().enumerate() {
            if let Some(lemma) = source.get(form) {
                gains[j][mask] += labels.get(lemma).copied().unwrap_or(0) - identity;
                coverage[j][mask] += labels.values().sum::<i64>();
            }
        }
    }
    // A subset sum at the complement of a prefix counts only unclaimed forms.
    for bit in 0..count {
        for mask in 0..=end {
            if mask & (1 << bit) != 0 {
                for j in 0..count {
                    gains[j][mask] += gains[j][mask ^ (1 << bit)];
                    coverage[j][mask] += coverage[j][mask ^ (1 << bit)];
                }
            }
        }
    }
    let mut best = vec![None; end + 1];
    best[0] = Some(Policy::default());
    for mask in 0..end {
        let prior = best[mask].as_ref().unwrap().clone();
        for (j, gain) in gains.iter().enumerate() {
            if mask & (1 << j) != 0 {
                continue;
            }
            let next = mask | (1 << j);
            let mut cand = prior.clone();
            cand.gain += gain[end ^ mask];
            cand.inversions += prior.order.iter().filter(|v| **v > j).count();
            cand.order.push(j);
            if best[next].as_ref().is_none_or(|v| preferred(&cand, v)) {
                best[next] = Some(cand);
            }
        }
    }
    let declared: Vec<_> = (0..count).collect();
    let full = best[end].as_ref().unwrap().order.clone();
    let mut subset = &best[0].as_ref().unwrap().order;
    let mut subset_gain = 0;
    for p in best.iter().flatten() {
        if p.gain > subset_gain
            || (p.gain == subset_gain && (p.order.len(), &p.order) < (subset.len(), subset))
        {
            subset = &p.order;
            subset_gain = p.gain;
        }
    }
    let mut greedy = vec![];
    let mut single = vec![];
    let mut mask = 0_usize;
    while mask != end {
        let j = (0..count)
            .filter(|j| mask & (1 << j) == 0)
            .max_by_key(|&j| {
                (
                    gains[j][end ^ mask],
                    coverage[j][end ^ mask],
                    std::cmp::Reverse(j),
                )
            })
            .unwrap();
        if mask == 0 {
            single.push(j);
        }
        if gains[j][end ^ mask] < 0 || coverage[j][end ^ mask] == 0 {
            break;
        }
        greedy.push(j);
        mask |= 1 << j;
    }
    let policies = [declared, full, single, greedy, subset.clone()];
    let scores = std::array::from_fn(|a| {
        let mut mask = 0;
        let mut n = copy;
        for &j in &policies[a] {
            n += gains[j][end ^ mask];
            mask |= 1 << j;
        }
        n
    });
    assert!(scores[1] >= scores[0] && scores[4] >= scores[1] && scores[4] >= scores[3]);
    (policies, scores)
}

#[derive(Clone)]
struct Prediction {
    text: String,
    stage: Source,
}
#[derive(Default, Clone, Serialize)]
struct Metric {
    tokens: u64,
    correct: u64,
    wins: u64,
    losses: u64,
    changed: u64,
}
impl Metric {
    fn add(&mut self, y: &str, baseline: &str, answer: &str) {
        self.tokens += 1;
        self.correct += u64::from(answer == y);
        self.wins += u64::from(answer == y && baseline != y);
        self.losses += u64::from(answer != y && baseline == y);
        self.changed += u64::from(answer != baseline);
    }
}

fn grade(
    data: &[Line],
    forms: &IndexMap<String, usize>,
    pred: &[Vec<[Prediction; 2]>],
) -> serde_json::Value {
    let mut totals: [Metric; N] = Default::default();
    let mut hits: [Metric; N] = Default::default();
    let mut blocks: BTreeMap<&str, [i64; N + 1]> = BTreeMap::new();
    let mut examples = vec![];
    let mut n_examples = [0; 3];
    for s in data {
        let b = blocks.entry(&s.block).or_default();
        for (i, (form, lemma)) in s.tokens.iter().zip(&s.lemmas).enumerate() {
            let Some(y) = lemma else { continue };
            let id = forms[form];
            let initial = usize::from(i == 0);
            let reference = &pred[0][id][initial];
            let ext = matches!(reference.stage, Source::External | Source::ExternalLower);
            b[0] += 1;
            for j in 0..N {
                let answer = &pred[j][id][initial].text;
                totals[j].add(y, &reference.text, answer);
                if ext {
                    hits[j].add(y, &reference.text, answer);
                }
                b[j + 1] += i64::from(answer == y) - i64::from(reference.text == *y);
            }
            let changed = &pred[1][id][initial].text;
            if changed != &reference.text {
                assert!(
                    ext && matches!(
                        pred[1][id][initial].stage,
                        Source::External | Source::ExternalLower
                    )
                );
                let which = if changed == y {
                    0
                } else if &reference.text == y {
                    1
                } else {
                    2
                };
                if n_examples[which] < 5 {
                    examples.push(
                        json!({"form":form,"gold":y,"declared":reference.text,"learned":changed}),
                    );
                    n_examples[which] += 1;
                }
            }
        }
    }
    let mut hist: BTreeMap<[i64; N + 1], usize> = BTreeMap::new();
    let mut block_types = BTreeMap::new();
    for (id, b) in blocks {
        if b[0] > 0 {
            *hist.entry(b).or_default() += 1;
            *block_types
                .entry(if id.starts_with("d:") {
                    "document"
                } else {
                    "sentence"
                })
                .or_insert(0) += 1;
        }
    }
    json!({"counts":totals,"external_hits":hits,"examples":examples,"block_types":block_types,
           "hist":hist.into_iter().collect::<Vec<_>>()})
}

pub fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err("usage: paper_order ROOT INPUTS ISO OUTPUT".into());
    }
    let root = Path::new(&args[1]);
    let input = Path::new(&args[2]);
    let iso = &args[3];
    let output = Path::new(&args[4]);
    let inv: Inventory =
        serde_json::from_reader(BufReader::new(File::open(input.join("inventory.json"))?))?;
    if env!("CARGO_PKG_VERSION") != "1.0.0" {
        return Err("Expected 1.0.0 engine".into());
    }
    let lang = &inv.languages[iso];
    if lang.status.as_deref() != Some("paired") || lang.sources.len() < 2 {
        return Err("No independent pair for this language".into());
    }
    let dir = input.join(iso);
    let mut pairs = Vec::new();
    let mut gold = Votes::new();
    read_lines(&dir.join("train.jsonl"), |s| {
        for (i, (form, lemma)) in s.tokens.into_iter().zip(s.lemmas).enumerate() {
            if let Some(lemma) = lemma {
                *gold
                    .entry(normalization::nfc(&form).into_owned())
                    .or_default()
                    .entry(normalization::nfc(&lemma).into_owned())
                    .or_default() += 1;
                pairs.push(Pair {
                    form,
                    lemma,
                    sentence_initial: i == 0,
                });
            }
        }
    })?;
    let mut dev = vec![];
    let mut test = vec![];
    read_lines(&dir.join("dev.jsonl"), |s| dev.push(s))?;
    read_lines(&dir.join("test.jsonl"), |s| test.push(s))?;
    let casing = Casing::for_language(iso);
    let mut forms = IndexMap::new();
    let mut target = HashSet::new();
    for s in dev.iter().chain(&test) {
        for form in &s.tokens {
            if !forms.contains_key(form) {
                forms.insert(form.clone(), forms.len());
                let n = normalization::nfc(form);
                target.insert(casing.lowercase(&n).into_owned());
                target.insert(n.into_owned());
            }
        }
    }
    let mut wanted = target.clone();
    wanted.extend(gold.keys().cloned());
    let mut maps = vec![];
    for src in &lang.sources {
        maps.push(read_map(root, src, iso, &wanted)?);
    }
    let (policies, train_scores) = learn(&maps, &gold);
    let names: Vec<_> = policies
        .iter()
        .map(|p| p.iter().map(|&j| &lang.sources[j].id).collect::<Vec<_>>())
        .collect();
    let union = target
        .iter()
        .filter_map(|key| {
            maps.iter()
                .find_map(|m| m.get(key))
                .map(|v| (key.clone(), v.clone()))
        })
        .collect();
    let mut tables = train::build_for_language_with_resolved_external(&pairs, union, vec![], iso);
    let eligible: Vec<_> = tables.external.keys().cloned().collect();
    let mut pred = Vec::new();
    for order in &policies {
        tables.external = eligible
            .iter()
            .filter_map(|f| {
                order
                    .iter()
                    .find_map(|&j| maps[j].get(f))
                    .map(|v| (f.clone(), v.clone()))
            })
            .collect();
        let view = View::new(&tables, Thresholds::default(), Variant::Full);
        let answers = forms
            .keys()
            .map(|form| {
                std::array::from_fn(|i| {
                    let a = probe::lemma_with_info(&view, form, i != 0, Options::default());
                    Prediction {
                        text: a.lemma.into_owned(),
                        stage: a.source,
                    }
                })
            })
            .collect();
        pred.push(answers);
    }
    let result = json!({"iso":iso,"engine":env!("CARGO_PKG_VERSION"),"arms":ARMS,"order":names,"train_correct":train_scores,
                      "train_tokens":pairs.len(),"dev":grade(&dev,&forms,&pred),"test":grade(&test,&forms,&pred)});
    let tmp = output.with_extension("partial");
    fs::write(&tmp, serde_json::to_vec(&result)?)?;
    fs::rename(tmp, output)?;
    Ok(())
}
