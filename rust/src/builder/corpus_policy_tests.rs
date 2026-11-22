#![allow(clippy::expect_used, clippy::unwrap_used, clippy::indexing_slicing)]

//! Source annotation repairs must reach every training consumer, never gold.

use super::corpus::{pairs_from, read_jsonl_observed, read_training_data};
use super::*;
use crate::lexicon::Lexicon;
use crate::probe::Options;

fn spec<'a>(destination: &'a Path, name: &'a str) -> Spec<'a> {
    Spec {
        destination,
        train: None,
        dev: None,
        test: None,
        paradigms: &[],
        entities: &[],
        name,
        language: "Fixture",
        iso639_3: "eng",
        span_case: None,
        span_multiword: None,
        entity_lower: false,
        compound_marker: None,
        token_case: None,
        lemma_adapter: false,
        context_keys: true,
        phrases: &[],
        proper_noun_threshold: None,
    }
}

#[test]
fn source_placeholders_keep_literal_tokens_other_sources_and_raw_bytes() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("arbitrary-file-name.jsonl");
    let row = serde_json::json!({
        "tokens": ["Mitnahme", "lief", "-", "unknown", "f~", "X", "label"],
        "lemmas": ["unknown", "-", "-", "unknown", "X", "X", "Unknown"],
        "upos": ["NOUN", "VERB", "PUNCT", "X", "DET", "X", "NOUN"],
    });
    let bytes = format!("{row}\n");
    std::fs::write(&path, &bytes).unwrap();
    let (raw, original) = read_jsonl_observed(&path, None).unwrap();
    assert_eq!(pairs_from(&raw).len(), 7);

    for (config, expected) in [
        (
            "de_hdt",
            vec![
                None,
                None,
                Some("-"),
                Some("unknown"),
                Some("X"),
                Some("X"),
                Some("Unknown"),
            ],
        ),
        (
            "pcm_nsc",
            vec![
                Some("unknown"),
                Some("-"),
                Some("-"),
                Some("unknown"),
                None,
                Some("X"),
                Some("Unknown"),
            ],
        ),
    ] {
        let (sentences, observed) = read_jsonl_observed(&path, Some(config)).unwrap();
        let rows = graded_rows(&sentences);
        assert_eq!(
            rows.iter()
                .map(|row| row.lemma.as_deref())
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(sentences[0].tokens, raw[0].tokens);
        assert!(rows[0].sentence_initial);
        assert!(!rows[1].sentence_initial);
        assert_eq!(observed, original);
    }
    for config in ["en_ewt", "de_other", "pcm_other", "de_merged", "DE_HDT"] {
        let (sentences, observed) = read_jsonl_observed(&path, Some(config)).unwrap();
        assert_eq!(sentences[0].lemmas, raw[0].lemmas);
        assert_eq!(observed, original);
    }
    assert_eq!(std::fs::read_to_string(path).unwrap(), bytes);
}

#[test]
fn streaming_training_masks_common_entity_and_context_observations_together() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("train.jsonl");
    for (config, marker) in [
        ("de_hdt", "unknown"),
        ("de_hdt", "-"),
        ("de_hdt", "NULL"),
        ("pcm_nsc", "X"),
        ("cu_proiel", "?"),
        ("sk_snk", "?"),
        ("qaf_arabizi", "X"),
        ("say_autogramm", "X"),
        ("myv_jr", "-"),
        ("cop_scriptorium", "unknown"),
    ] {
        let ordinary =
            serde_json::json!({"tokens": ["Entry"], "lemmas": ["entry"], "upos": ["NOUN"]});
        let placeholder =
            serde_json::json!({"tokens": ["Entry"], "lemmas": [marker], "upos": ["PROPN"]});
        std::fs::write(
            &path,
            format!("{ordinary}\n{ordinary}\n{ordinary}\n{placeholder}\n{placeholder}\n"),
        )
        .unwrap();
        let (training, _) = read_training_data(
            &path,
            config,
            Some(Observations::new(crate::normalization::Casing::Root)),
        )
        .unwrap();
        assert_eq!(training.pairs.len(), 3);
        assert!(training.pairs.iter().all(|pair| pair.lemma == "entry"));
        assert_eq!(training.proper_nouns, Vec::<(String, String)>::new());
        // Without masking, the two PROPN observations make a minority context
        // key which returns the missing-annotation marker for a real word.
        assert_eq!(
            training.context.unwrap().keys(),
            Vec::<(String, String)>::new()
        );
    }
}

#[test]
fn build_does_not_reintroduce_placeholders_on_dev_refit_or_rewrite_raw_test() {
    let temporary = tempfile::tempdir().unwrap();
    for (case, config, marker) in [
        ("hdt-unknown", "de_hdt", "unknown"),
        ("hdt-dash", "de_hdt", "-"),
        ("hdt-null", "de_hdt", "NULL"),
        ("nsc-x", "pcm_nsc", "X"),
        ("proiel-question", "cu_proiel", "?"),
        ("snk-question", "sk_snk", "?"),
        ("arabizi-x", "qaf_arabizi", "X"),
        ("autogramm-x", "say_autogramm", "X"),
        ("jr-dash", "myv_jr", "-"),
        ("scriptorium-unknown", "cop_scriptorium", "unknown"),
    ] {
        let directory = temporary.path().join(case);
        std::fs::create_dir(&directory).unwrap();
        let train = directory.join("train.jsonl");
        let dev = directory.join("dev.jsonl");
        let destination = directory.join(config);
        let training = serde_json::json!({
            "tokens": ["cats", "trainbad", "Trainium", "cats"],
            "lemmas": ["cat", marker, marker, "cat"],
            "upos": ["NOUN", "NOUN", "PROPN", "NOUN"],
        });
        let development = serde_json::json!({
            "tokens": ["cats", "devbad", "Devium"],
            "lemmas": ["cat", marker, marker],
            "upos": ["NOUN", "NOUN", "PROPN"],
        });
        std::fs::write(&train, format!("{training}\n")).unwrap();
        // Enough scoring positions to clear the existing dev acceptance
        // margin; the policy does not change that margin.
        std::fs::write(&dev, format!("{development}\n").repeat(16)).unwrap();
        let spec = Spec {
            train: Some(&train),
            dev: Some(&dev),
            test: Some(&dev),
            ..spec(&destination, config)
        };
        let built = build_lexicon_required(&spec).unwrap();
        assert_eq!(built.training_tokens, 18);
        assert_eq!(built.selected_on, Some("dev"));
        assert_eq!(built.selection.as_ref().unwrap().sample().name(), "dev");
        assert_eq!(built.selection.as_ref().unwrap().tokens(), 16);
        assert_eq!(built.gold_tokens, Some(48));
        assert_eq!(built.accuracy, Some(1.0 / 3.0));
        let dev_integrity = &built
            .observed_inputs
            .iter()
            .find(|input| input.label == format!("{config}.dev"))
            .unwrap()
            .integrity;
        let test_integrity = &built
            .observed_inputs
            .iter()
            .find(|input| input.label == format!("{config}.test"))
            .unwrap()
            .integrity;
        assert_eq!(dev_integrity, test_integrity);

        let lexicon = Lexicon::open(&destination).unwrap();
        let lookup = Options {
            use_rules: false,
            ..Options::default()
        };
        assert_eq!(lexicon.lemma_with_info("cats", false, lookup).lemma, "cat");
        for form in ["trainbad", "Trainium", "devbad", "Devium"] {
            assert_eq!(lexicon.lemma_with_info(form, false, lookup).lemma, form);
            assert_eq!(
                lexicon
                    .lemma_with_info(form, false, lookup.entity_lookup())
                    .lemma,
                form
            );
        }
    }
}

#[test]
fn missing_corpus_annotations_allow_external_fallback_without_filtering_its_words() {
    let temporary = tempfile::tempdir().unwrap();
    let train = temporary.path().join("train.jsonl");
    let table = temporary.path().join("external.tsv");
    for (config, form, marker) in [
        ("de_hdt", "unknowns", "unknown"),
        ("de_hdt", "nulls", "NULL"),
        ("pcm_nsc", "exes", "X"),
    ] {
        let row = serde_json::json!({"tokens": ["unlabelled"], "lemmas": [marker]});
        std::fs::write(&train, format!("{row}\n")).unwrap();
        std::fs::write(&table, format!("{form}\t{marker}\n")).unwrap();
        let sources = [Paradigm {
            label: "external-fixture".to_owned(),
            path: table.clone(),
            lemma_first: false,
            normalized_unique: false,
            expected: None,
        }];
        let spec = Spec {
            train: Some(&train),
            paradigms: &sources,
            ..spec(temporary.path(), config)
        };
        let (signal, observed) = training_signal_observed(&spec).unwrap();
        let signal = signal.unwrap();
        assert_eq!(signal.source, "external-fixture/eng");
        assert_eq!(signal.pairs.len(), 1);
        assert_eq!(signal.pairs[0].form, form);
        assert_eq!(signal.pairs[0].lemma, marker);
        assert_eq!(observed.len(), 2);
    }
}

#[test]
fn hdt_null_marker_does_not_remove_literal_null_case_variants_or_real_lemmas() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("train.jsonl");
    let row = serde_json::json!({
        "tokens": ["minimale", "NULL", "Nullen", "null", "None", "kleinere", "keinlabel"],
        "lemmas": ["NULL", "NULL", "Null", "null", "None", "klein", null],
        "upos": ["ADJ", "X", "NOUN", "NUM", "X", "ADJ", "ADJ"],
    });
    std::fs::write(&path, format!("{row}\n")).unwrap();
    let (masked, observed) = read_jsonl_observed(&path, Some("de_hdt")).unwrap();
    assert_eq!(
        masked[0]
            .lemmas
            .as_ref()
            .unwrap()
            .iter()
            .map(Option::as_deref)
            .collect::<Vec<_>>(),
        vec![
            None,
            Some("NULL"),
            Some("Null"),
            Some("null"),
            Some("None"),
            Some("klein"),
            None
        ]
    );
    for config in [None, Some("en_ewt"), Some("de_gsd"), Some("de_merged")] {
        let (literal, integrity) = read_jsonl_observed(&path, config).unwrap();
        assert_eq!(
            literal[0].lemmas.as_ref().unwrap()[0].as_deref(),
            Some("NULL")
        );
        assert_eq!(integrity, observed);
    }
}

#[test]
fn confirmed_source_markers_keep_literal_punctuation_and_other_spellings() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("train.jsonl");
    for (config, marker) in [
        ("cu_proiel", "?"),
        ("sk_snk", "?"),
        ("qaf_arabizi", "X"),
        ("say_autogramm", "X"),
        ("myv_jr", "-"),
        ("cop_scriptorium", "unknown"),
    ] {
        let row = serde_json::json!({
            "tokens": ["unlabelled", marker, "?", "-", "X", "unknown", "Nullen", "lowercase", "titled"],
            "lemmas": [marker, marker, "?", "-", "X", "unknown", "Null", "x", "Unknown"],
            "upos": ["NOUN", "X", "PUNCT", "PUNCT", "X", "X", "NOUN", "NOUN", "NOUN"],
        });
        std::fs::write(&path, format!("{row}\n")).unwrap();
        let (raw, integrity) = read_jsonl_observed(&path, None).unwrap();
        let (masked, observed) = read_jsonl_observed(&path, Some(config)).unwrap();
        let expected = [
            None,
            Some(marker),
            Some("?"),
            Some("-"),
            Some("X"),
            Some("unknown"),
            Some("Null"),
            Some("x"),
            Some("Unknown"),
        ];
        assert_eq!(
            masked[0]
                .lemmas
                .as_ref()
                .unwrap()
                .iter()
                .map(Option::as_deref)
                .collect::<Vec<_>>(),
            expected
        );
        // A non-punctuation token with lemma `?` is missing annotation only
        // for the declared source; the literal question mark remains scored.
        assert_eq!(graded_rows(&masked)[0].lemma, None);
        assert_eq!(graded_rows(&masked)[2].lemma.as_deref(), Some("?"));
        assert_eq!(observed, integrity);
        let (other_source, _) = read_jsonl_observed(&path, Some("en_ewt")).unwrap();
        assert_eq!(other_source[0].lemmas, raw[0].lemmas);
    }
}
