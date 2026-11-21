//! The lab's own proof: the restated hash side places keys exactly as the
//! core does, and every candidate answers a small mixed-script dictionary
//! byte for byte like the oracle, end to end through `prepare` and `bench`.

use std::path::PathBuf;

use serde_json::Value;

use crate::cands::default_names;
use crate::index::place;
use crate::lexicon::{SPLIT_BYTES, build_hybrid};
use crate::{parse_opts, prepare, run_bench};

fn forms() -> Vec<String> {
    let stems = [
        "кот",
        "собак",
        "read",
        "Haus",
        "κόσμ",
        "كتاب",
        "학교",
        "東京",
        "talo",
        "ev",
    ];
    let endings = [
        "", "а", "ами", "s", "ing", "es", "ος", "ات", "에서", "に", "ssa", "lerinden", "ünüz",
    ];
    let mut out = Vec::new();
    for (i, stem) in stems.iter().enumerate() {
        for (j, ending) in endings.iter().enumerate() {
            out.push(format!("{stem}{ending}"));
            out.push(format!("{stem}{ending}{}", "x".repeat((i * 7 + j) % 23)));
        }
    }
    out.sort();
    out.dedup();
    out
}

#[test]
fn pilot_matches_core() {
    let rows: Vec<(String, u64)> = forms().into_iter().zip(0..).collect();
    let file = build_hybrid(&rows, |id| Some(format!("lemma{id}"))).unwrap();
    let word = |at: usize| u64::from_le_bytes(file[at..at + 8].try_into().unwrap()) as usize;
    let (seed, n_buckets, n_slots, da_len) = (word(32), word(40), word(48), word(56));
    let pilots_at = (128 + da_len).next_multiple_of(16);
    let core_pilots: Vec<u16> = file[pilots_at..pilots_at + 2 * n_buckets]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .collect();
    let long: Vec<&[u8]> = rows
        .iter()
        .map(|(f, _)| f.as_bytes())
        .filter(|f| f.len() > SPLIT_BYTES)
        .collect();
    assert!(long.len() > 50);
    let lab = place(&long).unwrap();
    assert_eq!((lab.seed as usize, lab.n_slots), (seed, n_slots));
    assert_eq!(lab.pilots, core_pilots);
}

fn scratch() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("encoding-lab-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("inputs/xxx")).unwrap();
    std::fs::create_dir_all(dir.join("repo")).unwrap();
    dir
}

#[test]
fn every_candidate_matches_the_oracle() {
    let dir = scratch();
    let forms = forms();
    let lemma = |form: &str| -> String {
        // A third keep their form, the rest lose a tail or change wholly.
        match form.len() % 3 {
            0 => form.to_owned(),
            1 => form
                .chars()
                .take(form.chars().count().saturating_sub(2).max(1))
                .collect(),
            _ => format!("лемма{}", form.len()),
        }
    };
    let sentence = |words: &[&String]| {
        let lemmas: Vec<String> = words.iter().map(|w| lemma(w)).collect();
        serde_json::json!({ "tokens": words, "lemmas": lemmas }).to_string()
    };
    let (train, held): (Vec<&String>, Vec<&String>) = forms.iter().partition(|f| f.len() % 5 != 0);
    let train_text: Vec<String> = train.chunks(7).map(sentence).collect();
    let mut test_words: Vec<&String> = train.iter().step_by(3).copied().collect();
    test_words.extend(&held);
    let capital = "Собака".to_owned();
    test_words.push(&capital);
    let test_text: Vec<String> = test_words.chunks(5).map(sentence).collect();
    std::fs::write(dir.join("inputs/xxx/train.jsonl"), train_text.join("\n")).unwrap();
    std::fs::write(dir.join("inputs/xxx/test.jsonl"), test_text.join("\n")).unwrap();
    let tsv: String = forms
        .iter()
        .step_by(2)
        .map(|f| format!("{}\t{f}\tX\n", lemma(f)))
        .collect();
    std::fs::write(dir.join("repo/dict.tsv"), tsv + "\nbroken line\n\t\n").unwrap();
    let inventory = serde_json::json!({ "languages": { "xxx": { "sources": [
        { "path": "dict.tsv", "lemma_first": true, "priority": 1 }
    ] } } });
    std::fs::write(dir.join("inputs/inventory.json"), inventory.to_string()).unwrap();

    let args =
        |list: &[&str]| parse_opts(&list.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>());
    let work = dir.join("work");
    let (inputs, repo) = (dir.join("inputs"), dir.join("repo"));
    let paths = [
        inputs.to_str().unwrap(),
        repo.to_str().unwrap(),
        work.to_str().unwrap(),
    ];
    let small = ["--uniform-queries", "3000", "--mixed-queries", "1000"];
    let io = [
        "--inputs", paths[0], "--repo", paths[1], "--work", paths[2], "--lang", "xxx",
    ];
    prepare(&args(&[&io[..], &small[..]].concat())).unwrap();

    let report: Value =
        serde_json::from_str(&std::fs::read_to_string(work.join("xxx/prepare.json")).unwrap())
            .unwrap();
    for keyset in ["exact", "external"] {
        let cands = report["keysets"][keyset]["candidates"].as_object().unwrap();
        assert_eq!(cands.len(), default_names().len());
        for (name, entry) in cands {
            assert_eq!(
                entry["verified"], true,
                "{keyset}/{name}: {}",
                entry["error"]
            );
        }
        for name in [
            "core+ids",
            "split8:ptrhash+rule",
            "bf8>marisa+ids",
            "phast+inline",
        ] {
            let opts = [
                "--work", paths[2], "--lang", "xxx", "--keyset", keyset, "--cand", name,
            ];
            run_bench(
                &args(&[&opts[..], &["--sample-ms", "1", "--samples", "1", "--cold"]].concat()),
                false,
            )
            .unwrap();
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn bundle_tables_read_back() {
    let rows = [
        ("cats", "cat"),
        ("geese", "goose"),
        ("ran", "run"),
        ("кошки", "кошка"),
    ];
    let mut sorted: Vec<(&str, &str)> = rows.to_vec();
    sorted.sort();
    let strings: Vec<&str> = {
        let mut lemmas: Vec<&str> = sorted.iter().map(|r| r.1).collect();
        lemmas.sort();
        lemmas.dedup();
        lemmas
    };
    let id = |lemma: &str| strings.iter().position(|s| *s == lemma).unwrap() as u32;
    let mut inline = Vec::new();
    for (form, lemma) in &sorted {
        let shared = form
            .bytes()
            .zip(lemma.bytes())
            .take_while(|(a, b)| a == b)
            .count();
        let mut key = form.as_bytes().to_vec();
        key.push(0);
        key.push((form.len() - shared) as u8);
        key.extend_from_slice(&lemma.as_bytes()[shared..]);
        inline.push(key);
    }
    inline.sort();
    let mut map = fst::MapBuilder::memory();
    sorted
        .iter()
        .for_each(|(f, l)| map.insert(f, u64::from(id(l))).unwrap());
    let mut set = fst::SetBuilder::memory();
    inline.iter().for_each(|k| set.insert(k).unwrap());
    let keyset: Vec<(&[u8], u32)> = sorted.iter().map(|(f, l)| (f.as_bytes(), id(l))).collect();
    let tables = [
        ("map", map.into_inner().unwrap()),
        ("inline", set.into_inner().unwrap()),
        (
            "da",
            yada::builder::DoubleArrayBuilder::build(&keyset).unwrap(),
        ),
    ];
    let root = std::env::temp_dir().join(format!("encoding-lab-bundle-{}", std::process::id()));
    for (codec, bytes) in tables {
        let lang = root.join(codec);
        let generation = lang.join(".generations/g1");
        std::fs::create_dir_all(&generation).unwrap();
        std::fs::write(lang.join("current"), "g1\n").unwrap();
        let meta = serde_json::json!({ "codecs": { "external": codec }, "entries": { "external": rows.len() } });
        std::fs::write(generation.join("meta.json"), meta.to_string()).unwrap();
        std::fs::write(generation.join("strings.txt"), strings.join("\n") + "\n").unwrap();
        std::fs::write(generation.join(format!("external.{codec}")), bytes).unwrap();
        let (set, _) = crate::data::bundle_keyset(&lang, "external")
            .unwrap()
            .unwrap();
        let got: Vec<(Vec<u8>, Vec<u8>)> = (0..set.keys.len())
            .map(|i| {
                (
                    set.keys.get(i).to_vec(),
                    set.lemmas.get(set.lemma[i] as usize).to_vec(),
                )
            })
            .collect();
        let want: Vec<(Vec<u8>, Vec<u8>)> = sorted
            .iter()
            .map(|(f, l)| (f.as_bytes().to_vec(), l.as_bytes().to_vec()))
            .collect();
        assert_eq!(got, want, "{codec}");
        assert!(
            crate::data::bundle_keyset(&lang, "missing")
                .unwrap()
                .is_none()
        );
    }
    std::fs::remove_dir_all(&root).unwrap();
}
