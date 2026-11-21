//! A lab for form-to-lemma dictionary encodings on real key sets.
//!
//! ```text
//! encoding-lab list
//! encoding-lab prepare --inputs DIR [--dicts DIR | --repo DIR] --work DIR --lang ISO
//!                      [--keysets exact,external] [--cands a,b,..] [--discard]
//!                      [--uniform-queries N] [--mixed-queries N]
//! encoding-lab bench   --work DIR --lang ISO --keyset exact --cand NAME
//!                      [--sample-ms 30] [--samples 5] [--counters] [--cold] [--note TEXT]
//!                      [--lengths]
//! ```
//!
//! `prepare` builds every candidate's image and proves it against the
//! oracle (a `hashbrown` map) on every key and every query; `bench` then
//! times one candidate per process, so no two structures share a cache.

mod alloc;
mod bench;
mod cands;
mod data;
mod dict;
mod hashing;
mod img;
mod index;
mod lexicon;
mod perf;
mod strs;

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::Ordering::Relaxed;
use std::time::{Duration, Instant};

use serde_json::{Map, Value, json};

use cands::{Costs, Cx, Spec, Visitor, default_names};
use data::{KeySet, Oracle, Workload};
use dict::{Dict, Payloads, build_values};
use strs::Strs;

#[global_allocator]
static GLOBAL: alloc::Counting = alloc::Counting;

type Opts = HashMap<String, String>;

fn parse_opts(args: &[String]) -> Opts {
    let mut out = Opts::new();
    let mut i = 0;
    while i < args.len() {
        let key = args[i].trim_start_matches("--").to_owned();
        match args.get(i + 1).filter(|next| !next.starts_with("--")) {
            Some(value) => {
                out.insert(key, value.clone());
                i += 2;
            }
            None => {
                out.insert(key, "1".to_owned());
                i += 1;
            }
        }
    }
    out
}

fn need<'a>(opts: &'a Opts, key: &str) -> io::Result<&'a str> {
    opts.get(key)
        .map(String::as_str)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, format!("missing --{key}")))
}

fn other(message: impl Into<String>) -> io::Error {
    io::Error::other(message.into())
}

/// Anonymous memory gets huge pages under `THP=always`, file mappings do
/// not; without this, heap-held structures would win on TLB alone.
fn disable_thp() -> bool {
    #[cfg(target_os = "linux")]
    {
        const PR_SET_THP_DISABLE: libc::c_int = 41;
        // SAFETY: a process-wide flag, set before any worker thread exists.
        unsafe { libc::prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0) == 0 }
    }
    #[cfg(not(target_os = "linux"))]
    false
}

fn main() -> ExitCode {
    // `ENCODING_LAB_THP=1` leaves the system's THP policy in force.
    let thp_off = std::env::var_os("ENCODING_LAB_THP").is_none() && disable_thp();
    index::quiet_phast_panics();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opts = parse_opts(args.get(1..).unwrap_or_default());
    let done = match args.first().map(String::as_str) {
        Some("list") => {
            default_names().iter().for_each(|name| println!("{name}"));
            Ok(())
        }
        Some("prepare") => prepare(&opts),
        Some("bench") => run_bench(&opts, thp_off),
        _ => Err(other(
            "usage: encoding-lab list | prepare | bench (see src/main.rs)",
        )),
    };
    match done {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("encoding-lab: {error}");
            ExitCode::FAILURE
        }
    }
}

// ---------------------------------------------------------------- prepare

fn lengths_json(lengths: impl Iterator<Item = (usize, usize)>) -> Value {
    // (byte length, weight): mean, and the share at or below each threshold.
    let mut hist = [0_usize; 33];
    let (mut total, mut bytes) = (0_usize, 0_usize);
    for (len, weight) in lengths {
        hist[len.min(32)] += weight;
        total += weight;
        bytes += len * weight;
    }
    let mut share = Map::new();
    for k in [4, 6, 8, 10, 12, 16] {
        let below: usize = hist[..=k].iter().sum();
        share.insert(format!("le{k}"), json!(below as f64 / total.max(1) as f64));
    }
    json!({ "mean_bytes": bytes as f64 / total.max(1) as f64, "share": share, "hist": hist.to_vec() })
}

fn stats(set: &KeySet, pay: &Payloads, oracle: &Oracle, real: &Strs) -> Value {
    let chars: usize = set
        .keys
        .iter()
        .map(|k| String::from_utf8_lossy(k).chars().count())
        .sum();
    let (mut hits, mut identity_hits) = (0_usize, 0_usize);
    for q in real.iter() {
        if let Some(lemma) = oracle.lemma(q) {
            hits += 1;
            identity_hits += usize::from(lemma == q);
        }
    }
    json!({
        "keys": set.keys.len(),
        "lemmas": set.lemmas.len(),
        "rules": pay.rules,
        "identity_keys": pay.identity_keys as f64 / set.keys.len().max(1) as f64,
        "bytes_per_char": set.keys.blob_len() as f64 / chars.max(1) as f64,
        "key_len": lengths_json(set.keys.iter().map(|k| (k.len(), 1))),
        "stream": {
            "queries": real.len(),
            "hit_rate": hits as f64 / real.len().max(1) as f64,
            "identity_hits": identity_hits as f64 / hits.max(1) as f64,
            "query_len": lengths_json(real.iter().map(|q| (q.len(), 1))),
            "hit_len": lengths_json(real.iter().filter(|q| oracle.lemma(q).is_some()).map(|q| (q.len(), 1))),
        },
    })
}

/// Proves a candidate against the oracle: every key, then every query.
struct Verify<'a> {
    set: &'a KeySet,
    oracle: &'a Oracle<'a>,
    workloads: &'a [Workload],
    exact: bool,
    live_before: usize,
    heap_bytes: usize,
    false_positives: u64,
    error: Option<String>,
}

impl Visitor for Verify<'_> {
    fn visit<D: Dict>(&mut self, dict: &D) {
        self.heap_bytes = alloc::live().saturating_sub(self.live_before);
        for (i, key) in self.set.keys.iter().enumerate() {
            let want = self.set.lemmas.get(self.set.lemma[i] as usize);
            if !dict.get(key).is_some_and(|l| l.is(want)) {
                self.error = Some(format!(
                    "key {:?} answered wrongly",
                    String::from_utf8_lossy(key)
                ));
                return;
            }
        }
        for workload in self.workloads {
            let mut acc = 0_u64;
            for q in workload.queries.iter() {
                let got = dict.get(q);
                acc = acc.wrapping_add(got.map_or(0, |l| l.sink()));
                let want = self.oracle.lemma(q);
                let agrees = match (got, want) {
                    (Some(lemma), Some(want)) => lemma.is(want),
                    (None, None) => true,
                    _ => false,
                };
                if agrees {
                    continue;
                }
                if !self.exact && want.is_none() {
                    self.false_positives += 1;
                } else {
                    let q = String::from_utf8_lossy(q);
                    self.error = Some(format!("{}: query {q:?} answered wrongly", workload.name));
                    return;
                }
            }
            if self.exact && acc != workload.checksum {
                self.error = Some(format!(
                    "{}: checksum differs from the oracle's",
                    workload.name
                ));
                return;
            }
        }
    }
}

fn file_bytes(files: &[PathBuf]) -> u64 {
    files
        .iter()
        .filter_map(|f| std::fs::metadata(f).ok())
        .map(|m| m.len())
        .sum()
}

fn save_u32s(path: &Path, words: &[u32]) -> io::Result<()> {
    std::fs::write(path, img::as_bytes(words))
}

fn load_u32s(path: &Path) -> io::Result<Vec<u32>> {
    Ok(std::fs::read(path)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect())
}

/// A bundle's `exact` and `lower` tables, read by the production codec.
fn corpus_tables(lang_dir: &Path) -> io::Result<Vec<lexicon::CoreHybrid>> {
    let current = std::fs::read_to_string(lang_dir.join("current"))?;
    let generation = lang_dir.join(".generations").join(current.trim());
    let mut out = Vec::new();
    for name in ["exact", "lower"] {
        let path = generation.join(format!("{name}.hybrid"));
        if path.exists() {
            out.push(lexicon::CoreHybrid::open(img::map_file(&path)?).map_err(other)?);
        }
    }
    Ok(out)
}

fn prepare(opts: &Opts) -> io::Result<()> {
    let inputs = Path::new(need(opts, "inputs")?);
    let lang = need(opts, "lang")?;
    let work = Path::new(need(opts, "work")?).join(lang);
    let wanted = opts.get("keysets").map_or("exact,external", String::as_str);
    let names: Vec<String> = match opts.get("cands") {
        Some(list) => list.split(',').map(str::to_owned).collect(),
        None => default_names(),
    };
    let specs = names
        .iter()
        .map(|n| Spec::parse(n))
        .collect::<io::Result<Vec<_>>>()?;
    let size = |key: &str, default: usize| {
        opts.get(key)
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let sizes = data::Sizes {
        uniform: size("uniform-queries", 1 << 20),
        mixed: size("mixed-queries", 1 << 18),
    };
    let single = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .map_err(|e| other(e.to_string()))?;

    let tokens = data::test_tokens(&inputs.join(lang).join("test.jsonl"))?;
    let corpus = data::corpus_keyset(&inputs.join(lang).join("train.jsonl"))?;
    let corpus_oracle = Oracle::new(&corpus);
    let mut report = Map::new();
    for keyset in wanted.split(',') {
        let external;
        let mut origin = format!(
            "forms of {}",
            inputs.join(lang).join("train.jsonl").display()
        );
        let (set, extra) = match keyset {
            "exact" => (&corpus, Vec::new()),
            "external" => {
                if let Some(dicts) = opts.get("dicts") {
                    let Some((read, from)) =
                        data::bundle_keyset(&Path::new(dicts).join(lang), "external")?
                    else {
                        eprintln!("{lang}: the bundle has no external table, skipping");
                        continue;
                    };
                    external = read;
                    origin = from;
                } else {
                    let sources = data::sources(&inputs.join("inventory.json"), lang)?;
                    if sources.is_empty() {
                        eprintln!("{lang}: no dictionary sources, skipping external");
                        continue;
                    }
                    external = data::external_keyset(Path::new(need(opts, "repo")?), &sources)?;
                    origin = format!("inventory sources under {}", need(opts, "repo")?);
                }
                let oracle = Oracle::new(&external);
                // The bundle's own corpus tables decide what reaches its
                // dictionary table; without a bundle, the training forms do.
                let shipped = match opts.get("dicts") {
                    Some(dicts) => corpus_tables(&Path::new(dicts).join(lang))?,
                    None => Vec::new(),
                };
                let probes = if shipped.is_empty() {
                    data::cascade_probes(&tokens, |t| corpus_oracle.lemma(t).is_some(), &oracle)
                } else {
                    data::cascade_probes(
                        &tokens,
                        |t| shipped.iter().any(|table| table.get(t).is_some()),
                        &oracle,
                    )
                };
                (&external, vec![("cascade".to_owned(), probes)])
            }
            unknown => return Err(other(format!("unknown keyset {unknown}"))),
        };
        if set.keys.is_empty() {
            eprintln!("{lang}/{keyset}: empty key set, skipping");
            continue;
        }
        let dir = work.join(keyset);
        std::fs::create_dir_all(&dir)?;
        let oracle = Oracle::new(set);
        let mut real = Strs::new();
        tokens.iter().for_each(|t| real.push(t));
        let workloads = data::workloads(set, &oracle, real, extra, sizes);
        set.keys.save(&dir.join("keys.bin"))?;
        set.lemmas.save(&dir.join("lemmas.bin"))?;
        save_u32s(&dir.join("lemma.ids"), &set.lemma)?;
        let mut expect = Map::new();
        for w in &workloads {
            w.queries.save(&dir.join(format!("q.{}.bin", w.name)))?;
            expect.insert(
                w.name.clone(),
                json!({ "queries": w.queries.len(), "hits": w.hits, "checksum": w.checksum.to_string() }),
            );
        }
        let pay = build_values(set, Some(&dir))?;
        let cx = Cx {
            dir: dir.clone(),
            data: Some((set, &pay)),
        };
        let mut costs = Costs::default();
        let mut cands = Map::new();
        for spec in &specs {
            let started = Instant::now();
            let files = spec.files(&dir);
            let mut entry = json!({ "heap": spec.heap(), "exact": spec.exact() });
            let panics_before = index::PHAST_PANICS.load(Relaxed);
            let mut verify = Verify {
                set,
                oracle: &oracle,
                workloads: &workloads,
                exact: spec.exact(),
                live_before: alloc::live(),
                heap_bytes: 0,
                false_positives: 0,
                error: None,
            };
            // A crate that panics on a key set it cannot hold (too many
            // keys, too long a suffix) loses its own row, not the language.
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                cands::build(spec, &cx, &mut costs, &single)
                    .and_then(|()| cands::open(spec, &cx, &mut verify))
            }));
            let failure = match outcome {
                Ok(done) => done.err().map(|e| e.to_string()).or(verify.error.take()),
                Err(_) => Some("the builder or reader panicked".to_owned()),
            };
            let cost = costs.of(&spec.cost_keys(&dir));
            let value_bytes = match spec.value_layer() {
                Some("ids") => pay.ids_bytes,
                Some("off") => pay.off_bytes,
                Some("rule") => pay.rule_bytes,
                _ => 0,
            };
            let disk = file_bytes(&files);
            let index_bytes = if spec.keeps_image() { disk } else { 0 } + verify.heap_bytes as u64;
            entry["verified"] = json!(failure.is_none());
            entry["error"] = json!(failure);
            entry["disk_bytes"] = json!(disk);
            entry["heap_bytes"] = json!(verify.heap_bytes);
            entry["index_bytes"] = json!(index_bytes);
            entry["value_bytes"] = json!(value_bytes);
            entry["bytes_per_key"] =
                json!((index_bytes + value_bytes) as f64 / set.keys.len() as f64);
            entry["build_ns"] = json!(cost.ns as u64);
            entry["build_peak_heap"] = json!(cost.peak_heap);
            entry["false_positives"] = json!(verify.false_positives);
            entry["caught_panics"] = json!(index::PHAST_PANICS.load(Relaxed) - panics_before);
            eprintln!(
                "{lang}/{keyset} {:<24} {} {:>7.2} B/key build {:>8.1} ms  ({:.1}s)",
                spec.name,
                if entry["verified"] == json!(true) {
                    "ok  "
                } else {
                    "FAIL"
                },
                entry["bytes_per_key"].as_f64().unwrap_or(0.0),
                cost.ns as f64 / 1e6,
                started.elapsed().as_secs_f64(),
            );
            if let Some(error) = entry["error"].as_str() {
                eprintln!("    {error}");
            }
            cands.insert(spec.name.clone(), entry);
        }
        let real = &workloads
            .iter()
            .find(|w| w.name == "real")
            .ok_or_else(|| other("empty test"))?
            .queries;
        report.insert(
            keyset.to_owned(),
            json!({ "origin": origin, "stats": stats(set, &pay, &oracle, real), "workloads": expect, "candidates": cands }),
        );
        if opts.contains_key("discard") {
            std::fs::remove_dir_all(&dir)?;
        }
    }
    std::fs::create_dir_all(&work)?;
    let text = serde_json::to_string_pretty(&json!({ "lang": lang, "keysets": report }))?;
    std::fs::write(work.join("prepare.json"), text)
}

// ------------------------------------------------------------------ bench

struct Noop;

impl Visitor for Noop {
    fn visit<D: Dict>(&mut self, _dict: &D) {}
}

/// Time from before `open` to the K-th answer of the real stream.
struct FirstAnswers<'a> {
    queries: &'a Strs,
    started: Instant,
    took: Duration,
}

impl Visitor for FirstAnswers<'_> {
    fn visit<D: Dict>(&mut self, dict: &D) {
        let mut acc = 0_u64;
        for q in self.queries.iter().take(COLD_QUERIES) {
            acc = acc.wrapping_add(dict.get(q).map_or(0, |l| l.sink()));
        }
        std::hint::black_box(acc);
        self.took = self.started.elapsed();
    }
}

const COLD_QUERIES: usize = 64;

struct Bench<'a> {
    started: Instant,
    opened_after: Duration,
    workloads: &'a [(String, Strs, u64)],
    protocol: bench::Protocol,
    results: Map<String, Value>,
}

impl Visitor for Bench<'_> {
    fn visit<D: Dict>(&mut self, dict: &D) {
        self.opened_after = self.started.elapsed();
        for (name, queries, expect) in self.workloads {
            let mut modes = Map::new();
            modes.insert(
                "thr".to_owned(),
                bench::measure(dict, queries, *expect, false, &self.protocol),
            );
            // The sweep only ranks structures by miss rate; one mode is enough.
            if !name.starts_with("sweep") {
                modes.insert(
                    "lat".to_owned(),
                    bench::measure(dict, queries, *expect, true, &self.protocol),
                );
            }
            self.results.insert(name.clone(), Value::Object(modes));
        }
    }
}

/// Drops the candidate's own files from the page cache, and no others: a
/// neighbour evicted here would page back in at random from a hard disk.
#[cfg(target_os = "linux")]
fn evict(files: &[PathBuf]) -> io::Result<()> {
    use std::os::fd::AsRawFd;
    for path in files {
        let file = std::fs::File::open(path)?;
        // SAFETY: advice on a descriptor we hold; clean pages leave the cache.
        unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
    }
    Ok(())
}

/// Reads the candidate's files front to back, so the timed passes never
/// wait for a disk: sequential here, where a cold lookup would seek.
fn prewarm(files: &[PathBuf]) -> io::Result<()> {
    for path in files {
        io::copy(&mut std::fs::File::open(path)?, &mut io::sink())?;
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn major_faults() -> i64 {
    // SAFETY: getrusage fills the zeroed struct it is given.
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &raw mut usage);
        usage.ru_majflt
    }
}

#[cfg(target_os = "linux")]
fn current_cpu() -> i32 {
    // SAFETY: no arguments, no side effects.
    unsafe { libc::sched_getcpu() }
}

#[cfg(not(target_os = "linux"))]
fn evict(_files: &[PathBuf]) -> io::Result<()> {
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn major_faults() -> i64 {
    0
}

#[cfg(not(target_os = "linux"))]
fn current_cpu() -> i32 {
    -1
}

fn run_bench(opts: &Opts, thp_off: bool) -> io::Result<()> {
    let lang = need(opts, "lang")?;
    let keyset = need(opts, "keyset")?;
    let work = Path::new(need(opts, "work")?).join(lang);
    let dir = work.join(keyset);
    let spec = Spec::parse(need(opts, "cand")?)?;
    let number = |key: &str, default: u64| {
        opts.get(key)
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let prepared: Value =
        serde_json::from_str(&std::fs::read_to_string(work.join("prepare.json"))?)?;
    let prepared = &prepared["keysets"][keyset];
    if prepared["candidates"][&spec.name]["verified"] != json!(true) {
        return Err(other(format!("{} was not verified by prepare", spec.name)));
    }
    let mut workloads = Vec::new();
    for (name, expect) in prepared["workloads"]
        .as_object()
        .ok_or_else(|| other("no workloads"))?
    {
        // `--lengths` runs the per-length profile instead of the main set.
        if name.starts_with("len_") != opts.contains_key("lengths") {
            continue;
        }
        let checksum = expect["checksum"]
            .as_str()
            .and_then(|c| c.parse().ok())
            .ok_or_else(|| other("checksum"))?;
        workloads.push((
            name.clone(),
            Strs::load(&dir.join(format!("q.{name}.bin")))?,
            checksum,
        ));
    }
    let loaded;
    let data = if spec.needs_rows() {
        let set = KeySet {
            keys: Strs::load(&dir.join("keys.bin"))?,
            lemma: load_u32s(&dir.join("lemma.ids"))?,
            lemmas: Strs::load(&dir.join("lemmas.bin"))?,
        };
        let pay = build_values(&set, None)?;
        loaded = (set, pay);
        Some((&loaded.0, &loaded.1))
    } else {
        None
    };
    let cx = Cx {
        dir: dir.clone(),
        data,
    };

    let mut own = spec.files(&dir);
    own.extend(
        spec.value_layer()
            .map(|layer| dir.join(format!("values.{layer}.img"))),
    );
    own.retain(|path| path.exists());
    // A candidate that rebuilds itself at open is opened once: its cold
    // start is that rebuild, and repeating it would only repeat the work.
    let rebuilds = spec.needs_rows();
    let mut cold = Value::Null;
    if opts.contains_key("cold") && !rebuilds {
        let real = &workloads
            .iter()
            .find(|w| w.0 == "real")
            .ok_or_else(|| other("no real stream"))?
            .1;
        evict(&own)?;
        let faults = major_faults();
        let mut first = FirstAnswers {
            queries: real,
            started: Instant::now(),
            took: Duration::ZERO,
        };
        cands::open(&spec, &cx, &mut first)?;
        cold = json!({
            "open_and_first_answers_ns": first.took.as_nanos() as u64,
            "answers": COLD_QUERIES.min(real.len()),
            "major_faults": major_faults() - faults,
        });
    }
    prewarm(&own)?;
    let mut open_ns = 0;
    if !rebuilds {
        cands::open(&spec, &cx, &mut Noop)?;
        let started = Instant::now();
        cands::open(&spec, &cx, &mut Noop)?;
        open_ns = started.elapsed().as_nanos() as u64;
    }

    let mut run = Bench {
        started: Instant::now(),
        opened_after: Duration::ZERO,
        workloads: &workloads,
        protocol: bench::Protocol {
            sample: Duration::from_millis(number("sample-ms", 30)),
            samples: number("samples", 5) as usize,
            counters: opts.contains_key("counters"),
        },
        results: Map::new(),
    };
    cands::open(&spec, &cx, &mut run)?;
    if rebuilds {
        open_ns = run.opened_after.as_nanos() as u64;
    }
    let exact = spec.exact();
    if exact
        && run
            .results
            .values()
            .any(|modes| modes["thr"]["checksum_ok"] != json!(true))
    {
        return Err(other(format!(
            "{}: a pass disagreed with the oracle's checksum",
            spec.name
        )));
    }
    let line = json!({
        "lang": lang,
        "keyset": keyset,
        "cand": spec.name,
        "note": opts.get("note"),
        "round": opts.get("round"),
        "cpu": current_cpu(),
        "loadavg": std::fs::read_to_string("/proc/loadavg").ok().map(|s| s.trim().to_owned()),
        "thp_disabled": thp_off,
        "heap_images": img::heap_mode(),
        "open_warm_ns": open_ns,
        "open_rebuilds": rebuilds,
        "cold": cold,
        "caught_panics": index::PHAST_PANICS.load(Relaxed),
        "results": run.results,
    });
    println!("{line}");
    Ok(())
}

#[cfg(test)]
mod tests;
