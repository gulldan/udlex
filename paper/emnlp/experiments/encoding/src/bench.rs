//! The measured loops. Throughput: independent lookups the CPU may overlap.
//! Latency: each lookup's index waits for the previous answer.

use std::hint::black_box;
use std::time::{Duration, Instant};

use serde_json::{Map, Value, json};

use crate::dict::Dict;
use crate::perf::{Counter, GROUPS};
use crate::strs::Strs;

#[inline(never)]
pub fn pass_throughput<D: Dict>(dict: &D, queries: &Strs) -> u64 {
    let mut acc = 0_u64;
    for i in 0..queries.len() {
        // SAFETY: `i < len`.
        let key = unsafe { queries.get_unchecked(i) };
        acc = acc.wrapping_add(dict.get(key).map_or(0, |lemma| lemma.sink()));
    }
    acc
}

#[inline(never)]
pub fn pass_latency<D: Dict>(dict: &D, queries: &Strs) -> u64 {
    let (mut acc, mut i) = (0_u64, 0_usize);
    while i < queries.len() {
        // SAFETY: `i < len`.
        let key = unsafe { queries.get_unchecked(i) };
        let sunk = dict.get(key).map_or(0, |lemma| lemma.sink());
        acc = acc.wrapping_add(sunk);
        // The top bit is never set, but only the answer can prove it: the
        // next key cannot be fetched before this lookup retires.
        i += 1 + (black_box(sunk) >> 63) as usize;
    }
    acc
}

pub struct Protocol {
    pub sample: Duration,
    pub samples: usize,
    pub counters: bool,
}

/// CPU time of this thread: on a quiet pinned core it equals wall time, on
/// a shared one it leaves out the slices other processes took.
#[cfg(target_os = "linux")]
fn thread_cpu_ns() -> u128 {
    // SAFETY: clock_gettime fills the zeroed timespec it is given.
    unsafe {
        let mut now: libc::timespec = std::mem::zeroed();
        libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &raw mut now);
        now.tv_sec as u128 * 1_000_000_000 + now.tv_nsec as u128
    }
}

#[cfg(not(target_os = "linux"))]
fn thread_cpu_ns() -> u128 {
    0
}

fn median(sorted: &[f64]) -> f64 {
    sorted[sorted.len() / 2]
}

/// Times one workload in one mode; `expect` is the oracle's checksum.
pub fn measure<D: Dict>(
    dict: &D,
    queries: &Strs,
    expect: u64,
    latency: bool,
    protocol: &Protocol,
) -> Value {
    let pass = |d: &D, q: &Strs| {
        if latency {
            pass_latency(d, q)
        } else {
            pass_throughput(d, q)
        }
    };
    let n = queries.len() as f64;
    let got = pass(dict, queries);
    let mut samples = Vec::with_capacity(protocol.samples);
    let mut cpu_samples = Vec::with_capacity(protocol.samples);
    let mut passes_per_sample = 1_u64;
    for _ in 0..protocol.samples {
        let (start, cpu_start) = (Instant::now(), thread_cpu_ns());
        let mut passes = 0_u64;
        loop {
            black_box(pass(dict, queries));
            passes += 1;
            if start.elapsed() >= protocol.sample {
                break;
            }
        }
        cpu_samples.push((thread_cpu_ns() - cpu_start) as f64 / (passes as f64 * n));
        samples.push(start.elapsed().as_nanos() as f64 / (passes as f64 * n));
        passes_per_sample = passes;
    }
    samples.sort_by(f64::total_cmp);
    cpu_samples.sort_by(f64::total_cmp);
    let mut out = json!({
        "ns_min": samples[0],
        "ns_median": median(&samples),
        "cpu_ns_min": cpu_samples[0],
        "cpu_ns_median": median(&cpu_samples),
        "samples": samples,
        "queries": queries.len(),
        "checksum_ok": got == expect,
    });
    if protocol.counters && !latency {
        let mut counts = Map::new();
        for group in GROUPS {
            let open: Vec<_> = group
                .iter()
                .filter_map(|&(name, kind, config)| Some((name, Counter::open(kind, config)?)))
                .collect();
            open.iter().for_each(|(_, c)| c.start());
            for _ in 0..passes_per_sample {
                black_box(pass(dict, queries));
            }
            for (name, counter) in &open {
                if let Some(count) = counter.stop() {
                    counts.insert(
                        (*name).to_owned(),
                        json!(count as f64 / (passes_per_sample as f64 * n)),
                    );
                }
            }
        }
        out["per_query"] = Value::Object(counts);
    }
    out
}
