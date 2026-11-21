//! Native udlex worker: the protocol of `worker.py` over stdin/stdout, one JSON object per line.
//!
//! Usage: `udlex-speed BUNDLE ISO TEST.jsonl REFERENCE.json THREADS`.
//! Each window is an independent sample of test sentences drawn with replacement; every
//! sentence goes through `Lexicon::lemmatize`, the native counterpart of the Python call.

use std::borrow::Cow;
use std::io::{BufRead, Write};
use std::time::Instant;

use serde_json::{Value, json};

fn send(value: &Value) {
    let mut out = std::io::stdout().lock();
    writeln!(out, "{value}").expect("stdout");
    out.flush().expect("stdout");
}

/// One field of /proc/self/status in KiB.
fn status_kb(key: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .unwrap_or_default()
        .lines()
        .find_map(|line| line.strip_prefix(key))
        .and_then(|rest| rest.trim().trim_end_matches(" kB").trim().parse().ok())
        .unwrap_or(0)
}

/// User plus system CPU seconds of the process; Linux reports them in 1/100 s ticks.
fn cpu_seconds() -> f64 {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let fields: Vec<&str> = stat.rsplit_once(") ").map_or(vec![], |(_, rest)| rest.split(' ').collect());
    let tick = |i: usize| fields.get(i).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
    (tick(11) + tick(12)) / 100.0
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn sample(seed: u64, n: usize, population: usize) -> Vec<usize> {
    let mut state = seed;
    (0..n).map(|_| (splitmix(&mut state) % population as u64) as usize).collect()
}

type Kept<'a> = Vec<(usize, Vec<Cow<'a, str>>)>;

/// Lemmatises the sample on `threads` threads; keeps every `stride`-th answer for checking.
fn window<'a>(
    lexicon: &'a udlex_rs::Lexicon,
    sentences: &'a [Vec<&'a str>],
    picks: &[usize],
    threads: usize,
    stride: usize,
) -> (f64, usize, Kept<'a>) {
    let run = |part: &[usize], offset: usize| {
        let mut bytes = 0_usize;
        let mut kept = Vec::new();
        for (k, &i) in part.iter().enumerate() {
            let out = lexicon.lemmatize(&sentences[i]);
            bytes += out.iter().map(|lemma| lemma.len()).sum::<usize>();
            if (offset + k) % stride == 0 {
                kept.push((i, out));
            }
        }
        (bytes, kept)
    };
    let start = Instant::now();
    let parts: Vec<(usize, Kept<'a>)> = if threads == 1 {
        vec![run(picks, 0)]
    } else {
        let size = picks.len().div_ceil(threads).max(1);
        std::thread::scope(|scope| {
            let handles: Vec<_> = picks
                .chunks(size)
                .enumerate()
                .map(|(c, part)| scope.spawn(move || run(part, c * size)))
                .collect();
            handles.into_iter().map(|h| h.join().expect("worker thread")).collect()
        })
    };
    let seconds = start.elapsed().as_secs_f64();
    let bytes = std::hint::black_box(parts.iter().map(|(b, _)| *b).sum::<usize>());
    (seconds, bytes, parts.into_iter().flat_map(|(_, kept)| kept).collect())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let [_, bundle, iso, test, reference, threads] = args.as_slice() else {
        panic!("usage: udlex-speed BUNDLE ISO TEST.jsonl REFERENCE.json THREADS");
    };
    let threads: usize = threads.parse().expect("THREADS");
    let owned: Vec<Vec<String>> = std::fs::read_to_string(test)
        .expect("test")
        .lines()
        .map(|line| {
            let row: Value = serde_json::from_str(line).expect("test row");
            row["tokens"]
                .as_array()
                .expect("tokens")
                .iter()
                .map(|t| t.as_str().expect("token string").to_owned())
                .collect()
        })
        .collect();
    let sentences: Vec<Vec<&str>> = owned.iter().map(|s| s.iter().map(String::as_str).collect()).collect();
    let expected: Vec<Vec<String>> =
        serde_json::from_reader(std::fs::File::open(reference).expect("reference")).expect("reference json");
    assert_eq!(expected.len(), sentences.len(), "reference and test differ in length");
    let lengths: Vec<usize> = sentences.iter().map(Vec::len).collect();
    let expected_bytes: Vec<usize> = expected.iter().map(|s| s.iter().map(String::len).sum()).collect();

    let start = Instant::now();
    let lexicon = udlex_rs::load(bundle, iso).expect("load");
    let load_s = start.elapsed().as_secs_f64();
    let start = Instant::now();
    std::hint::black_box(lexicon.lemmatize(&sentences[0]));
    let first_s = start.elapsed().as_secs_f64();
    send(&json!({"event": "ready", "import_s": 0.0, "load_s": load_s, "first_s": first_s,
                 "rss_load_kb": status_kb("VmRSS:"), "model": format!("udlex-rs {iso}"), "files": [bundle]}));

    let mut picks: Vec<usize> = Vec::new();
    for line in std::io::stdin().lock().lines() {
        let command: Value = serde_json::from_str(&line.expect("stdin")).expect("command");
        match command["cmd"].as_str().expect("cmd") {
            "warm" => {
                let budget = command["seconds"].as_f64().expect("seconds");
                let (mut n, mut spent, mut rate, mut round) = (16_usize, 0.0, 0.0, 0_u64);
                while spent < budget || round < 2 {
                    let draw = sample(0xC9 ^ round, n, sentences.len());
                    let (seconds, _, _) = window(&lexicon, &sentences, &draw, threads, usize::MAX);
                    rate = draw.iter().map(|&i| lengths[i]).sum::<usize>() as f64 / seconds;
                    spent += seconds;
                    round += 1;
                    if seconds < 0.25 {
                        n *= 4;
                    }
                }
                send(&json!({"rate": rate, "windows": round}));
            }
            "prep" => {
                let n = command["sentences"].as_u64().expect("sentences") as usize;
                picks = sample(command["seed"].as_u64().expect("seed"), n, sentences.len());
                send(&json!({"tokens": picks.iter().map(|&i| lengths[i]).sum::<usize>()}));
            }
            "go" => {
                let stride = (picks.len() / 5000).max(1);
                let cpu = cpu_seconds();
                let (seconds, bytes, kept) = window(&lexicon, &sentences, &picks, threads, stride);
                let cpu = cpu_seconds() - cpu;
                let mut mismatch = 0_usize;
                let mut checked = 0_usize;
                for (i, out) in &kept {
                    checked += out.len();
                    mismatch += out.iter().zip(&expected[*i]).filter(|(a, b)| a.as_ref() != b.as_str()).count();
                    mismatch += out.len().abs_diff(expected[*i].len());
                }
                let want: usize = picks.iter().map(|&i| expected_bytes[i]).sum();
                send(&json!({"tokens": picks.iter().map(|&i| lengths[i]).sum::<usize>(), "seconds": seconds,
                             "cpu_s": cpu, "checked": checked, "mismatch": mismatch, "bytes_ok": bytes == want}));
            }
            "exit" => {
                send(&json!({"vmhwm_kb": status_kb("VmHWM:"), "vmrss_kb": status_kb("VmRSS:")}));
                return;
            }
            other => panic!("unknown command {other}"),
        }
    }
}
