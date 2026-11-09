#![allow(clippy::expect_used, clippy::unwrap_used, clippy::indexing_slicing)]

use std::collections::HashSet;

use super::build_lexicon_required as build_lexicon;
use super::corpus::{Line, read_jsonl, scorable};
use super::input::{MAX_SOURCE_LINE_BYTES, for_each_line};
use super::sources::{entity_pairs, external_pairs, read_paradigm, training_signal};
use super::*;

fn write_jsonl(path: &std::path::Path, row: &serde_json::Value) {
    std::fs::write(path, format!("{row}\n")).unwrap();
}

/// The underscore convention: a marker against an ordinary form, a real
/// lemma when the form is an underscore itself.
#[test]
fn an_underscore_lemma_is_only_a_marker_for_other_forms() {
    assert!(!scorable("кот", "_"));
    assert!(scorable("_", "_"));
    assert!(scorable("кот", "кот"));
}

/// Python's text mode translated `\r\n` and lone `\r`; the reader must
/// see the same rows either way.
#[test]
fn carriage_returns_are_line_breaks() {
    let directory = std::env::temp_dir().join(format!("udlex-crlf-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("zxx.tsv");
    std::fs::write(&path, b"cats\tcat\r\ndogs\tdog\rbirds\tbird\n").unwrap();
    let pairs = read_paradigm(&path, false).unwrap();
    std::fs::remove_dir_all(&directory).ok();
    assert_eq!(
        pairs,
        vec![
            ("cats".to_owned(), "cat".to_owned()),
            ("dogs".to_owned(), "dog".to_owned()),
            ("birds".to_owned(), "bird".to_owned()),
        ]
    );
}

/// `UniMorph` ships `lemma <TAB> form`; the flag swaps the columns.
#[test]
fn column_order_follows_the_source() {
    let directory = std::env::temp_dir().join(format!("udlex-cols-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("zxx.tsv");
    std::fs::write(&path, "кот\tкотами\tINS;PL\n").unwrap();
    let swapped = read_paradigm(&path, true).unwrap();
    std::fs::remove_dir_all(&directory).ok();
    assert_eq!(swapped, vec![("котами".to_owned(), "кот".to_owned())]);
}

#[test]
fn malformed_paradigm_rows_report_their_source_line() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("bad.tsv");
    std::fs::write(&path, "cats\tcat\nmissing-column\n").unwrap();
    assert!(matches!(
        read_paradigm(&path, false),
        Err(Error::Paradigm { line: 2, .. })
    ));
    std::fs::write(&path, "cats\tcat\nname\tbad\0lemma\n").unwrap();
    assert!(matches!(
        read_paradigm(&path, false),
        Err(Error::Paradigm { line: 2, .. })
    ));
}

#[test]
fn legacy_paradigms_ignore_only_declared_compatibility_rows() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("legacy.tsv");
    std::fs::write(&path, "\n# upstream metadata\ncats\tcat\nmissing\t\n\r\n").unwrap();
    assert_eq!(
        read_paradigm(&path, false).unwrap(),
        vec![("cats".to_owned(), "cat".to_owned())]
    );

    std::fs::write(&path, " \tlemma\ncats\tcat\n").unwrap();
    assert!(matches!(
        read_paradigm(&path, false),
        Err(Error::Paradigm { line: 1, .. })
    ));

    for payload in ["#not-a-comment\ncats\tcat\n", " # comment\ncats\tcat\n"] {
        std::fs::write(&path, payload).unwrap();
        assert!(matches!(
            read_paradigm(&path, false),
            Err(Error::Paradigm { line: 1, .. })
        ));
    }
}

#[test]
fn paradigm_rows_are_normalized_to_nfc() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("nfd.tsv");
    std::fs::write(&path, "cafe\u{301}s\tcafe\u{301}\n").unwrap();
    assert_eq!(
        read_paradigm(&path, false).unwrap(),
        vec![("cafés".to_owned(), "café".to_owned())]
    );
}

#[test]
fn overlong_source_lines_are_rejected_before_unbounded_allocation() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("long.jsonl");
    std::fs::write(&path, vec![b'x'; MAX_SOURCE_LINE_BYTES + 1]).unwrap();
    assert!(matches!(
        read_jsonl(&path),
        Err(Error::LineTooLong { line: 1, .. })
    ));
}

#[cfg(unix)]
#[test]
fn builder_inputs_never_follow_symlinks() {
    use std::os::unix::fs::symlink;

    let directory = tempfile::tempdir().unwrap();
    let outside = directory.path().join("outside.tsv");
    let linked = directory.path().join("linked.tsv");
    std::fs::write(&outside, "cats\tcat\n").unwrap();
    symlink(&outside, &linked).unwrap();
    assert!(matches!(
        read_paradigm(&linked, false),
        Err(Error::Io { .. })
    ));

    let paradigms = [Paradigm {
        label: "linked".to_owned(),
        path: linked,
        lemma_first: false,
        normalized_unique: false,
        expected: None,
    }];
    let spec = Spec {
        destination: directory.path(),
        train: None,
        dev: None,
        test: None,
        paradigms: &paradigms,
        entities: &[],
        name: "xx_linked",
        language: "Test",
        iso639_3: "zxx",
        span_case: None,
        span_multiword: None,
        entity_lower: false,
        compound_marker: None,
        token_case: None,
        lemma_adapter: false,
        context_keys: false,
        phrases: &[],
        proper_noun_threshold: None,
    };
    assert!(matches!(training_signal(&spec), Err(Error::Io { .. })));
}

#[test]
fn builder_rejects_an_authenticated_digest_that_does_not_match_parsed_bytes() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("authenticated.tsv");
    std::fs::write(&path, "cats\tcat\n").unwrap();
    let paradigms = [Paradigm {
        label: "authenticated".to_owned(),
        path,
        lemma_first: false,
        normalized_unique: true,
        expected: Some(crate::lexicon::FileIntegrity {
            size: 9,
            sha256: "0".repeat(64),
        }),
    }];
    let spec = Spec {
        destination: directory.path(),
        train: None,
        dev: None,
        test: None,
        paradigms: &paradigms,
        entities: &[],
        name: "xx_authenticated",
        language: "Test",
        iso639_3: "zxx",
        span_case: None,
        span_multiword: None,
        entity_lower: false,
        compound_marker: None,
        token_case: None,
        lemma_adapter: false,
        context_keys: false,
        phrases: &[],
        proper_noun_threshold: None,
    };

    assert!(matches!(
        training_signal(&spec),
        Err(Error::InputIntegrity { .. })
    ));
}

#[cfg(unix)]
#[test]
fn builder_detects_a_regular_path_swap_after_descriptor_open() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("source.tsv");
    let opened = directory.path().join("opened.tsv");
    std::fs::write(&path, "cats\tcat\n").unwrap();

    let result = for_each_line(&path, |_line, _text| {
        std::fs::rename(&path, &opened).unwrap();
        std::fs::write(&path, "dogs\tdog\n").unwrap();
        Ok(())
    });

    assert!(matches!(result, Err(Error::Io { .. })));
}

#[test]
fn training_source_exclusion_uses_identity_not_label_substrings() {
    let directory = tempfile::tempdir().unwrap();
    let foobar = directory.path().join("foobar.tsv");
    let foo = directory.path().join("foo.tsv");
    std::fs::write(&foobar, "cats\tcat\n").unwrap();
    std::fs::write(&foo, "dogs\tdog\n").unwrap();
    let paradigms = vec![
        Paradigm {
            label: "foobar".to_owned(),
            path: foobar,
            lemma_first: false,
            normalized_unique: false,
            expected: None,
        },
        Paradigm {
            label: "foo".to_owned(),
            path: foo,
            lemma_first: false,
            normalized_unique: false,
            expected: None,
        },
    ];
    let spec = Spec {
        destination: directory.path(),
        train: None,
        dev: None,
        test: None,
        paradigms: &paradigms,
        entities: &[],
        name: "xx_test",
        language: "Test",
        iso639_3: "zxx",
        span_case: None,
        span_multiword: None,
        entity_lower: false,
        compound_marker: None,
        token_case: None,
        lemma_adapter: false,
        context_keys: false,
        phrases: &[],
        proper_noun_threshold: None,
    };
    let signal = training_signal(&spec).unwrap().unwrap();
    assert_eq!(signal.paradigm_index, Some(0));
    assert_eq!(
        external_pairs(&spec, signal.paradigm_index)
            .unwrap()
            .into_iter()
            .collect::<Vec<_>>(),
        vec![("dogs".to_owned(), "dog".to_owned())]
    );
}

#[test]
fn earlier_external_source_owns_conflicting_forms() {
    let directory = tempfile::tempdir().unwrap();
    let preferred = directory.path().join("preferred.tsv");
    let fallback = directory.path().join("fallback.tsv");
    std::fs::write(&preferred, "axes\taxis\naxes\taxe\ncats\tcat\n").unwrap();
    std::fs::write(&fallback, "axes\tax\naxes\tax\naxes\tax\ndogs\tdog\n").unwrap();
    let paradigms = vec![
        Paradigm {
            label: "preferred".to_owned(),
            path: preferred,
            lemma_first: false,
            normalized_unique: false,
            expected: None,
        },
        Paradigm {
            label: "fallback".to_owned(),
            path: fallback,
            lemma_first: false,
            normalized_unique: false,
            expected: None,
        },
    ];
    let spec = Spec {
        destination: directory.path(),
        train: None,
        dev: None,
        test: None,
        paradigms: &paradigms,
        entities: &[],
        name: "xx_test",
        language: "Test",
        iso639_3: "zxx",
        span_case: None,
        span_multiword: None,
        entity_lower: false,
        compound_marker: None,
        token_case: None,
        lemma_adapter: false,
        context_keys: false,
        phrases: &[],
        proper_noun_threshold: None,
    };

    assert_eq!(
        external_pairs(&spec, None)
            .unwrap()
            .into_iter()
            .collect::<Vec<_>>(),
        vec![
            ("axes".to_owned(), "axe".to_owned()),
            ("cats".to_owned(), "cat".to_owned()),
            ("dogs".to_owned(), "dog".to_owned()),
        ]
    );
}

#[test]
fn later_source_frequency_cannot_flip_an_earlier_ambiguous_form() {
    let directory = tempfile::tempdir().unwrap();
    let preferred = directory.path().join("preferred.tsv");
    let fallback = directory.path().join("fallback.tsv");
    std::fs::write(&preferred, "axes\taxis\naxes\taxe\n").unwrap();
    std::fs::write(&fallback, "other-a\taxis\nother-b\taxis\nother-c\taxis\n").unwrap();
    let paradigms = vec![
        Paradigm {
            label: "preferred".to_owned(),
            path: preferred,
            lemma_first: false,
            normalized_unique: false,
            expected: None,
        },
        Paradigm {
            label: "fallback".to_owned(),
            path: fallback,
            lemma_first: false,
            normalized_unique: false,
            expected: None,
        },
    ];
    let spec = Spec {
        destination: directory.path(),
        train: None,
        dev: None,
        test: None,
        paradigms: &paradigms,
        entities: &[],
        name: "xx_test",
        language: "Test",
        iso639_3: "zxx",
        span_case: None,
        span_multiword: None,
        entity_lower: false,
        compound_marker: None,
        token_case: None,
        lemma_adapter: false,
        context_keys: false,
        phrases: &[],
        proper_noun_threshold: None,
    };

    let merged = external_pairs(&spec, None).unwrap();
    assert_eq!(merged.get("axes").map(String::as_str), Some("axe"));
}

#[test]
fn external_rewrites_preserve_strong_unicode_scripts() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("arabic.tsv");
    std::fs::write(
        &path,
        concat!(
            "għoxorَيْ\tعشر\n",
            "الكتب\tكِتَاب\n",
            "كَتَبُوا!\tكَتَب\n",
            "123\t456\n",
            "Latinعربي\tLatinعربي\n",
        ),
    )
    .unwrap();
    let paradigms = [Paradigm {
        label: "arabic".to_owned(),
        path,
        lemma_first: false,
        normalized_unique: false,
        expected: None,
    }];
    let spec = Spec {
        destination: directory.path(),
        train: None,
        dev: None,
        test: None,
        paradigms: &paradigms,
        entities: &[],
        name: "ar_test",
        language: "Arabic",
        iso639_3: "ara",
        span_case: None,
        span_multiword: None,
        entity_lower: false,
        compound_marker: None,
        token_case: None,
        lemma_adapter: false,
        context_keys: false,
        phrases: &[],
        proper_noun_threshold: None,
    };

    let merged = external_pairs(&spec, None).unwrap();
    assert_eq!(merged.get("الكتب").map(String::as_str), Some("كِتَاب"));
    assert_eq!(merged.get("كَتَبُوا!").map(String::as_str), Some("كَتَب"));
    assert_eq!(
        merged.get("Latinعربي").map(String::as_str),
        Some("Latinعربي")
    );
    assert!(!merged.contains_key("għoxorَيْ"));
    assert!(!merged.contains_key("123"));
}

#[test]
fn japanese_han_and_kana_are_one_compatible_writing_system() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("japanese.tsv");
    std::fs::write(
        &path,
        "食べた\t食べる\nカタカナ\tかたかな\n東京へ\t東京\nTokyo\t東京\n",
    )
    .unwrap();
    let paradigms = [Paradigm {
        label: "japanese".to_owned(),
        path,
        lemma_first: false,
        normalized_unique: false,
        expected: None,
    }];
    let spec = Spec {
        destination: directory.path(),
        train: None,
        dev: None,
        test: None,
        paradigms: &paradigms,
        entities: &[],
        name: "ja_test",
        language: "Japanese",
        iso639_3: "jpn",
        span_case: None,
        span_multiword: None,
        entity_lower: false,
        compound_marker: None,
        token_case: None,
        lemma_adapter: false,
        context_keys: false,
        phrases: &[],
        proper_noun_threshold: None,
    };

    let merged = external_pairs(&spec, None).unwrap();
    assert_eq!(merged.get("食べた").map(String::as_str), Some("食べる"));
    assert_eq!(merged.get("カタカナ").map(String::as_str), Some("かたかな"));
    assert_eq!(merged.get("東京へ").map(String::as_str), Some("東京"));
    assert!(!merged.contains_key("Tokyo"));
}

#[test]
fn authenticated_import_skips_a_cross_script_semantic_mismatch() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("strict.tsv");
    std::fs::write(&path, "għoxorَيْ\tعشر\n").unwrap();
    let paradigms = [Paradigm {
        label: "strict".to_owned(),
        path,
        lemma_first: false,
        normalized_unique: true,
        expected: None,
    }];
    let spec = Spec {
        destination: directory.path(),
        train: None,
        dev: None,
        test: None,
        paradigms: &paradigms,
        entities: &[],
        name: "ar_test",
        language: "Arabic",
        iso639_3: "ara",
        span_case: None,
        span_multiword: None,
        entity_lower: false,
        compound_marker: None,
        token_case: None,
        lemma_adapter: false,
        context_keys: false,
        phrases: &[],
        proper_noun_threshold: None,
    };

    assert!(external_pairs(&spec, None).unwrap().is_empty());
}

#[test]
fn normalized_unique_fast_path_validates_its_explicit_contract() {
    let directory = tempfile::tempdir().unwrap();
    for (name, payload, expected) in [
        ("blank", "\na\tA\n", "expected at least two"),
        (
            "empty-field",
            "a\t\n",
            "first two columns must be non-empty",
        ),
        ("unsorted", "b\tB\na\tA\n", "strictly sorted and unique"),
        ("not-nfc", "e\u{301}\tlemma\n", "must already be NFC"),
    ] {
        let path = directory.path().join(format!("{name}.tsv"));
        std::fs::write(&path, payload).unwrap();
        let paradigms = [Paradigm {
            label: name.to_owned(),
            path,
            lemma_first: false,
            normalized_unique: true,
            expected: None,
        }];
        let spec = Spec {
            destination: directory.path(),
            train: None,
            dev: None,
            test: None,
            paradigms: &paradigms,
            entities: &[],
            name: "xx_test",
            language: "Test",
            iso639_3: "zxx",
            span_case: None,
            span_multiword: None,
            entity_lower: false,
            compound_marker: None,
            token_case: None,
            lemma_adapter: false,
            context_keys: false,
            phrases: &[],
            proper_noun_threshold: None,
        };

        assert!(matches!(
            external_pairs(&spec, None),
            Err(Error::Paradigm { message, .. }) if message.contains(expected)
        ));
    }

    let path = directory.path().join("valid-combining.tsv");
    std::fs::write(&path, "Na\u{331}mg\u{331}is\tname\n").unwrap();
    let paradigms = [Paradigm {
        label: "valid-combining".to_owned(),
        path,
        lemma_first: false,
        normalized_unique: true,
        expected: None,
    }];
    let spec = Spec {
        destination: directory.path(),
        train: None,
        dev: None,
        test: None,
        paradigms: &paradigms,
        entities: &[],
        name: "xx_test",
        language: "Test",
        iso639_3: "zxx",
        span_case: None,
        span_multiword: None,
        entity_lower: false,
        compound_marker: None,
        token_case: None,
        lemma_adapter: false,
        context_keys: false,
        phrases: &[],
        proper_noun_threshold: None,
    };
    assert_eq!(
        external_pairs(&spec, None)
            .unwrap()
            .get("Na\u{331}mg\u{331}is")
            .map(String::as_str),
        Some("name")
    );
}

#[test]
fn entity_sources_fail_closed_on_conflicting_inflections() {
    let directory = tempfile::tempdir().unwrap();
    let preferred = directory.path().join("preferred-entities.tsv");
    let fallback = directory.path().join("fallback-entities.tsv");
    std::fs::write(
        &preferred,
        "New York\tNew York\tinflection\nWarsaw Pacts\tWarsaw Pact\tinflection\n",
    )
    .unwrap();
    std::fs::write(
        &fallback,
        "Ile de Paris\tIle de Paris\tidentity\nNew York\tNew York City\tinflection\nWarsaw Pacts\tWarsaw Pacts\tidentity\n",
    )
    .unwrap();
    let entities = vec![
        Paradigm {
            label: "preferred".to_owned(),
            path: preferred,
            lemma_first: false,
            normalized_unique: false,
            expected: None,
        },
        Paradigm {
            label: "fallback".to_owned(),
            path: fallback,
            lemma_first: false,
            normalized_unique: false,
            expected: None,
        },
    ];
    let spec = Spec {
        destination: directory.path(),
        train: None,
        dev: None,
        test: None,
        paradigms: &[],
        entities: &entities,
        name: "xx_test",
        language: "Test",
        iso639_3: "zxx",
        span_case: None,
        span_multiword: None,
        entity_lower: false,
        compound_marker: None,
        token_case: None,
        lemma_adapter: false,
        context_keys: false,
        phrases: &[],
        proper_noun_threshold: None,
    };

    assert_eq!(
        entity_pairs(&spec, &[]).unwrap(),
        vec![
            ("New York".to_owned(), "New York".to_owned()),
            ("Warsaw Pacts".to_owned(), "Warsaw Pacts".to_owned()),
            ("Ile de Paris".to_owned(), "Ile de Paris".to_owned()),
        ]
    );
}

#[test]
fn entity_inflections_fall_back_to_identity_across_scripts() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("entities.tsv");
    std::fs::write(
        &path,
        "għoxorَيْ عشرين\tعشر عشرين\tinflection\nكَتَبُوا! فِيهَا\tكَتَب فِيهَا\tinflection\n",
    )
    .unwrap();
    let entities = [Paradigm {
        label: "entities".to_owned(),
        path,
        lemma_first: false,
        normalized_unique: false,
        expected: None,
    }];
    let spec = Spec {
        destination: directory.path(),
        train: None,
        dev: None,
        test: None,
        paradigms: &[],
        entities: &entities,
        name: "ar_test",
        language: "Arabic",
        iso639_3: "ara",
        span_case: None,
        span_multiword: None,
        entity_lower: false,
        compound_marker: None,
        token_case: None,
        lemma_adapter: false,
        context_keys: false,
        phrases: &[],
        proper_noun_threshold: None,
    };

    assert_eq!(
        entity_pairs(&spec, &[]).unwrap(),
        vec![
            ("għoxorَيْ عشرين".to_owned(), "għoxorَيْ عشرين".to_owned()),
            ("كَتَبُوا! فِيهَا".to_owned(), "كَتَب فِيهَا".to_owned()),
        ]
    );
}

#[test]
fn entity_identity_protection_dominates_inflections_in_either_source_order() {
    let directory = tempfile::tempdir().unwrap();
    let identity = directory.path().join("identity.tsv");
    let inflection = directory.path().join("inflection.tsv");
    std::fs::write(&identity, "Shared Name\tShared Name\tidentity\n").unwrap();
    std::fs::write(&inflection, "Shared Name\tShare Name\tinflection\n").unwrap();

    for (first, second) in [(&identity, &inflection), (&inflection, &identity)] {
        let entities = [
            Paradigm {
                label: "first".to_owned(),
                path: first.clone(),
                lemma_first: false,
                normalized_unique: false,
                expected: None,
            },
            Paradigm {
                label: "second".to_owned(),
                path: second.clone(),
                lemma_first: false,
                normalized_unique: false,
                expected: None,
            },
        ];
        let spec = Spec {
            destination: directory.path(),
            train: None,
            dev: None,
            test: None,
            paradigms: &[],
            entities: &entities,
            name: "xx_test",
            language: "Test",
            iso639_3: "zxx",
            span_case: None,
            span_multiword: None,
            entity_lower: false,
            compound_marker: None,
            token_case: None,
            lemma_adapter: false,
            context_keys: false,
            phrases: &[],
            proper_noun_threshold: None,
        };

        assert_eq!(
            entity_pairs(&spec, &[]).unwrap(),
            vec![("Shared Name".to_owned(), "Shared Name".to_owned())]
        );
    }
}

#[test]
fn single_word_source_rows_yield_to_the_corpus_that_claims_them() {
    // Measured policy, not a heuristic: a source row that contradicts a corpus
    // `PROPN` is reduced to identity together with it, so the annotated rewrite
    // is destroyed. Where no corpus `PROPN` claims the surface there is nothing
    // to destroy, and those rows are the gate's gain.
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("entities.tsv");
    std::fs::write(
        &path,
        "Parises\tParis\tinflection\nOpenAI\tOpenAI\tidentity\nNew Yorkses\tNew York\tinflection\n",
    )
    .unwrap();
    let entities = [Paradigm {
        label: "entities".to_owned(),
        path,
        lemma_first: false,
        normalized_unique: false,
        expected: None,
    }];
    let spec = Spec {
        destination: directory.path(),
        train: None,
        dev: None,
        test: None,
        paradigms: &[],
        entities: &entities,
        name: "xx_test",
        language: "Test",
        iso639_3: "zxx",
        span_case: None,
        span_multiword: None,
        entity_lower: false,
        compound_marker: None,
        token_case: None,
        lemma_adapter: false,
        context_keys: false,
        phrases: &[],
        proper_noun_threshold: None,
    };

    let claimed = [("Parises".to_owned(), "Paris".to_owned())];
    assert_eq!(
        entity_pairs(&spec, &claimed).unwrap(),
        vec![
            ("OpenAI".to_owned(), "OpenAI".to_owned()),
            ("New Yorkses".to_owned(), "New York".to_owned()),
        ]
    );
    assert_eq!(
        entity_pairs(&spec, &[]).unwrap(),
        vec![
            ("Parises".to_owned(), "Paris".to_owned()),
            ("OpenAI".to_owned(), "OpenAI".to_owned()),
            ("New Yorkses".to_owned(), "New York".to_owned()),
        ]
    );
}

#[test]
fn corpus_claims_are_matched_on_the_normalization_the_reducer_uses() {
    // The corpus keeps whatever spelling the treebank shipped, while entity
    // tables pair the two sides only after NFC. Comparing the raw corpus form
    // would admit this row and let it destroy the corpus rewrite downstream.
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("entities.tsv");
    std::fs::write(&path, "\u{c5}ngstr\u{f6}m\tAngstrom\tinflection\n").unwrap();
    let entities = [Paradigm {
        label: "entities".to_owned(),
        path,
        lemma_first: false,
        normalized_unique: false,
        expected: None,
    }];
    let spec = Spec {
        destination: directory.path(),
        train: None,
        dev: None,
        test: None,
        paradigms: &[],
        entities: &entities,
        name: "xx_test",
        language: "Test",
        iso639_3: "zxx",
        span_case: None,
        span_multiword: None,
        entity_lower: false,
        compound_marker: None,
        token_case: None,
        lemma_adapter: false,
        context_keys: false,
        phrases: &[],
        proper_noun_threshold: None,
    };

    let decomposed = [(
        "A\u{30a}ngstro\u{308}m".to_owned(),
        "A\u{30a}ngstro\u{308}m".to_owned(),
    )];
    assert_eq!(
        entity_pairs(&spec, &decomposed).unwrap(),
        [] as [(std::string::String, std::string::String); 0]
    );
}

#[test]
fn legacy_or_dishonest_entity_rows_are_rejected() {
    let directory = tempfile::tempdir().unwrap();
    let legacy = directory.path().join("legacy.tsv");
    std::fs::write(&legacy, "London\tOther\n").unwrap();
    let sources = [Paradigm {
        label: "legacy".to_owned(),
        path: legacy,
        lemma_first: false,
        normalized_unique: false,
        expected: None,
    }];
    let spec = Spec {
        destination: directory.path(),
        train: None,
        dev: None,
        test: None,
        paradigms: &[],
        entities: &sources,
        name: "xx_test",
        language: "Test",
        iso639_3: "zxx",
        span_case: None,
        span_multiword: None,
        entity_lower: false,
        compound_marker: None,
        token_case: None,
        lemma_adapter: false,
        context_keys: false,
        phrases: &[],
        proper_noun_threshold: None,
    };
    assert!(matches!(
        entity_pairs(&spec, &[]),
        Err(Error::Paradigm { message, .. }) if message.contains("semantics")
    ));

    std::fs::write(&sources[0].path, "London\tOther\tidentity\n").unwrap();
    assert!(matches!(
        entity_pairs(&spec, &[]),
        Err(Error::Paradigm { message, .. }) if message.contains("preserve")
    ));
}

#[test]
fn train_cv_is_deterministic_and_form_disjoint() {
    let training: Vec<Pair> = (0..1024)
        .flat_map(|index| {
            let form = format!("word{index:04}");
            [
                Pair {
                    form: form.clone(),
                    lemma: form.clone(),
                    sentence_initial: false,
                },
                Pair {
                    form: form.clone(),
                    lemma: form,
                    sentence_initial: true,
                },
            ]
        })
        .collect();
    let (retained, held_out) = train_cv_split(&training).unwrap();
    let retained_forms: HashSet<&str> = retained.iter().map(|pair| pair.form.as_str()).collect();
    let held_out_forms: HashSet<&str> = held_out.iter().map(|pair| pair.form.as_str()).collect();
    assert!(retained_forms.is_disjoint(&held_out_forms));
    assert_eq!(retained.len() + held_out.len(), training.len());

    let (again_retained, again_held_out) = train_cv_split(&training).unwrap();
    assert_eq!(
        retained.iter().map(|pair| &pair.form).collect::<Vec<_>>(),
        again_retained
            .iter()
            .map(|pair| &pair.form)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        held_out.iter().map(|pair| &pair.form).collect::<Vec<_>>(),
        again_held_out
            .iter()
            .map(|pair| &pair.form)
            .collect::<Vec<_>>()
    );
}

#[test]
fn missing_dev_selects_on_train_cv_without_test_data() {
    let temporary = tempfile::tempdir().unwrap();
    let train = temporary.path().join("train.jsonl");
    let forms: Vec<String> = (0..512).map(|index| format!("word{index:04}")).collect();
    let row = serde_json::json!({"tokens": forms, "lemmas": forms});
    std::fs::write(&train, format!("{row}\n")).unwrap();
    let destination = temporary.path().join("xx_train_cv");
    let spec = Spec {
        destination: &destination,
        train: Some(&train),
        dev: None,
        test: None,
        paradigms: &[],
        entities: &[],
        name: "xx_train_cv",
        language: "Test",
        iso639_3: "zxx",
        span_case: None,
        span_multiword: None,
        entity_lower: false,
        compound_marker: None,
        token_case: None,
        lemma_adapter: false,
        context_keys: false,
        phrases: &[],
        proper_noun_threshold: None,
    };

    let built = build_lexicon(&spec).unwrap();
    assert_eq!(built.selected_on, Some("train-cv"));
    assert_eq!(built.variant, Variant::Identity.name());
    assert!(!built.tuned_on_dev);
}

/// The sentence reader keeps unannotated positions as rows without a
/// lemma and drops nothing, exactly like the reference's `graded_rows`.
#[test]
fn graded_rows_keep_unannotated_positions() {
    let sentences = vec![Line {
        tokens: vec!["Коты".to_owned(), "спят".to_owned(), "_".to_owned()],
        lemmas: Some(vec![Some("кот".to_owned()), None, Some("_".to_owned())]),
        upos: Some(vec![Some("NOUN".to_owned()), Some("VERB".to_owned()), None]),
    }];
    let rows = graded_rows(&sentences);
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0].lemma.as_deref(), Some("кот"));
    assert!(rows[0].sentence_initial);
    assert_eq!(rows[1].lemma, None);
    // `_` against the form `_` is a real lemma, not a marker.
    assert_eq!(rows[2].lemma.as_deref(), Some("_"));

    let pairs = pairs_from(&sentences);
    assert_eq!(pairs.len(), 2);
    assert_eq!(pairs[0].form, "Коты");
}

#[test]
fn misaligned_annotation_arrays_fail_at_the_source_line() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("bad.jsonl");
    std::fs::write(
        &path,
        r#"{"tokens":["cats","dogs"],"lemmas":["cat"],"upos":["NOUN","NOUN"]}
"#,
    )
    .unwrap();
    assert!(matches!(
        read_jsonl(&path),
        Err(Error::Alignment {
            line: 1,
            field: "lemmas",
            tokens: 2,
            found: 1,
            ..
        })
    ));
}

#[test]
fn corpus_jsonl_accepts_position_level_null_upos() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("nullable-upos.jsonl");
    std::fs::write(
        &path,
        "{\"tokens\":[\"word\"],\"lemmas\":[\"word\"],\"upos\":[null]}\n",
    )
    .unwrap();

    let sentences = read_jsonl(&path).unwrap();

    assert_eq!(sentences[0].upos, Some(vec![None]));
}

#[test]
fn invalid_utf8_is_not_collapsed_to_a_replacement_character() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("bad.tsv");
    std::fs::write(&path, b"cat\tcat\n\xff\tbad\n").unwrap();
    assert!(matches!(
        read_paradigm(&path, false),
        Err(Error::Encoding { .. })
    ));
}

#[test]
fn losing_the_last_source_publishes_identity_for_new_readers() {
    let temporary = tempfile::tempdir().unwrap();
    let destination = temporary.path().join("xx_stale");
    let train = temporary.path().join("train.jsonl");
    std::fs::write(
        &train,
        "{\"tokens\":[\"Cats\"],\"lemmas\":[\"cat\"],\"upos\":[\"NOUN\"]}\n",
    )
    .unwrap();
    let spec = Spec {
        destination: &destination,
        train: Some(&train),
        dev: None,
        test: None,
        paradigms: &[],
        entities: &[],
        name: "xx_stale",
        language: "Test",
        iso639_3: "zxx",
        span_case: None,
        span_multiword: None,
        entity_lower: false,
        compound_marker: None,
        token_case: None,
        lemma_adapter: false,
        context_keys: false,
        phrases: &[],
        proper_noun_threshold: None,
    };
    build_lexicon(&spec).unwrap();
    let old = crate::lexicon::Lexicon::open(&destination).unwrap();
    assert_eq!(old.lemma("Cats", false), "cat");
    let no_op = crate::lexicon::compile(&destination).unwrap();
    assert!(!no_op.from_sources);

    std::fs::write(&train, "").unwrap();
    let identity = build_lexicon(&spec).unwrap();
    assert_eq!(identity.source, "identity/no-signal");
    assert_eq!(identity.training_tokens, 0);
    assert_eq!(identity.variant, Variant::Identity.name());
    assert_eq!(identity.selected_on, None);
    assert_eq!(identity.gold_tokens, None);
    assert!(destination.join("current").is_file());
    let replacement = crate::lexicon::Lexicon::open(&destination).unwrap();
    assert_eq!(replacement.lemma("Cats", false), "Cats");
    assert_eq!(old.lemma("Cats", false), "cat");
}

#[test]
fn no_common_signal_keeps_entities_and_never_opens_dev_or_test() {
    let temporary = tempfile::tempdir().unwrap();
    let destination = temporary.path().join("xx_entity_identity");
    let dev = temporary.path().join("invalid-dev.jsonl");
    let test = temporary.path().join("invalid-test.jsonl");
    let entities = temporary.path().join("entities.tsv");
    std::fs::write(&dev, "this is deliberately not JSON\n").unwrap();
    std::fs::write(&test, "this is deliberately not JSON either\n").unwrap();
    std::fs::write(
        &entities,
        "Open AI\tOpen AI\tidentity\nNew Yorkses\tNew York\tinflection\n",
    )
    .unwrap();
    let entity_sources = [Paradigm {
        label: "entities/test".to_owned(),
        path: entities,
        lemma_first: false,
        normalized_unique: false,
        expected: None,
    }];
    let spec = Spec {
        destination: &destination,
        train: None,
        dev: Some(&dev),
        test: Some(&test),
        paradigms: &[],
        entities: &entity_sources,
        name: "xx_entity_identity",
        language: "Test",
        iso639_3: "eng",
        span_case: None,
        span_multiword: None,
        entity_lower: false,
        compound_marker: None,
        token_case: None,
        lemma_adapter: false,
        context_keys: false,
        phrases: &[],
        proper_noun_threshold: None,
    };

    let built = build_lexicon(&spec).unwrap();
    assert_eq!(built.source, "identity/no-signal");
    assert_eq!(built.training_tokens, 0);
    assert_eq!(built.thresholds, (0.0, 0.0));
    assert!(!built.tuned_on_dev);
    assert_eq!(built.variant, Variant::Identity.name());
    assert_eq!(built.selected_on, None);
    assert_eq!(built.gold_tokens, None);
    assert_eq!(built.accuracy, None);
    assert_eq!(built.baseline, None);
    assert_eq!(built.beats_baseline, None);
    assert_eq!(
        built
            .entries
            .iter()
            .filter(|(name, _)| name != "entities" && name != "strings")
            .map(|(_, count)| *count)
            .sum::<usize>(),
        0
    );
    assert!(
        built
            .entries
            .iter()
            .any(|(name, count)| name == "entities" && *count >= 2)
    );
    assert_eq!(
        built
            .observed_inputs
            .iter()
            .map(|input| input.role)
            .collect::<Vec<_>>(),
        vec!["entity"]
    );

    let lexicon = crate::lexicon::Lexicon::open(&destination).unwrap();
    let options = crate::probe::Options::default().entity_lookup();
    assert_eq!(
        lexicon.lemma_with_info("New Yorkses", false, options).lemma,
        "New York"
    );
    assert_eq!(lexicon.lemma("ordinary", false), "ordinary");
}

#[test]
fn span_conventions_round_trip_from_spec_through_metadata_to_the_probe() {
    let temporary = tempfile::tempdir().unwrap();
    let destination = temporary.path().join("xx_conventions");
    let entities = temporary.path().join("entities.tsv");
    // A multiword row: single-word source rows are not admitted by policy.
    std::fs::write(&entities, "CNN'ler kanalı\tcnn kanal\tinflection\n").unwrap();
    let entity_sources = [Paradigm {
        label: "entities/test".to_owned(),
        path: entities,
        lemma_first: false,
        normalized_unique: false,
        expected: None,
    }];
    let spec = Spec {
        destination: &destination,
        train: None,
        dev: None,
        test: None,
        paradigms: &[],
        entities: &entity_sources,
        name: "xx_conventions",
        language: "Test",
        iso639_3: "eng",
        span_case: Some("verbatim"),
        span_multiword: Some("compose"),
        entity_lower: false,
        compound_marker: None,
        token_case: Some("surface"),
        lemma_adapter: true,
        context_keys: true,
        phrases: &[],
        proper_noun_threshold: None,
    };
    let built = build_lexicon(&spec).unwrap();
    let meta_path = destination
        .join(".generations")
        .join(&built.generation)
        .join("meta.json");
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(meta_path).unwrap()).unwrap();
    assert_eq!(meta.get("lemma_adapter"), Some(&serde_json::json!(true)));
    assert_eq!(meta.get("context_keys"), Some(&serde_json::json!(true)));

    let lexicon = crate::lexicon::Lexicon::open(&destination).unwrap();
    let conventions = lexicon.span_conventions();
    assert!(!conventions.case_transfer);
    assert_eq!(conventions.multiword, crate::format::SpanMultiword::Compose);
    // The stamp turns the token case convention on; spans never consult it.
    assert!(lexicon.surface_case());

    // Behavioural check through the span path: the entity value keeps its
    // stored lowercase spelling instead of inheriting the surface casing.
    let options = crate::probe::Options::default().entity_lookup();
    let answer = lexicon.lemma_span_with_info(
        "CNN'ler kanalı",
        &["CNN'ler", "kanalı"],
        None,
        false,
        options,
    );
    assert_eq!(answer.lemma, "cnn kanal");
    assert_eq!(answer.source, crate::probe::Source::Entity);

    // A build that does not stamp the conventions opens with the defaults.
    let plain_destination = temporary.path().join("xx_plain");
    let plain = Spec {
        destination: &plain_destination,
        span_case: None,
        span_multiword: None,
        entity_lower: false,
        compound_marker: None,
        token_case: None,
        lemma_adapter: false,
        context_keys: false,
        phrases: &[],
        name: "xx_plain",
        ..spec
    };
    build_lexicon(&plain).unwrap();
    let plain_lexicon = crate::lexicon::Lexicon::open(&plain_destination).unwrap();
    assert_eq!(
        plain_lexicon.span_conventions(),
        crate::probe::SpanConventions::default()
    );
    assert!(!plain_lexicon.surface_case());
}

#[test]
fn test_data_never_selects_the_shipped_variant() {
    let temporary = tempfile::tempdir().unwrap();
    let train = temporary.path().join("train.jsonl");
    let test = temporary.path().join("test.jsonl");
    std::fs::write(
        &train,
        "{\"tokens\":[\"Cats\"],\"lemmas\":[\"cat\"],\"upos\":[\"NOUN\"]}\n",
    )
    .unwrap();
    // Copying wins on this deliberately hostile test set.  It may change
    // the reported score, but it must not make the builder select an
    // identity/table subset on the data it later reports.
    std::fs::write(
        &test,
        "{\"tokens\":[\"Cats\"],\"lemmas\":[\"Cats\"],\"upos\":[\"NOUN\"]}\n",
    )
    .unwrap();
    let destination = temporary.path().join("xx_no_test_selection");
    let spec = Spec {
        destination: &destination,
        train: Some(&train),
        dev: None,
        test: Some(&test),
        paradigms: &[],
        entities: &[],
        name: "xx_no_test_selection",
        language: "Test",
        iso639_3: "zxx",
        span_case: None,
        span_multiword: None,
        entity_lower: false,
        compound_marker: None,
        token_case: None,
        lemma_adapter: false,
        context_keys: false,
        phrases: &[],
        proper_noun_threshold: None,
    };

    let built = build_lexicon(&spec).unwrap();
    assert_eq!(built.variant, Variant::Full.name());
    assert_eq!(built.selected_on, None);
    assert_eq!(built.beats_baseline, Some(false));
    assert_eq!(
        crate::lexicon::Lexicon::open(&destination)
            .unwrap()
            .lemma("Cats", false),
        "cat"
    );
}

#[test]
#[allow(clippy::too_many_lines)] // два поля Spec добавили строки поверх лимита
fn dev_proper_nouns_arrive_only_after_selection_and_test_never_trains() {
    let temporary = tempfile::tempdir().unwrap();
    let train = temporary.path().join("train.jsonl");
    let dev_proper = temporary.path().join("dev-proper.jsonl");
    let dev_common = temporary.path().join("dev-common.jsonl");
    let test = temporary.path().join("test.jsonl");
    let entities = temporary.path().join("entities.tsv");
    write_jsonl(
        &train,
        &serde_json::json!({
            "tokens": ["cats", "Berlins", "Ambigs", "Ambigs"],
            "lemmas": ["cat", "Berlin", "Ambig", "Other"],
            "upos": ["NOUN", "PROPN", "PROPN", "PROPN"]
        }),
    );
    let dev_tokens = ["Parises", "Anchors"];
    let dev_lemmas = ["Paris", "Anchor"];
    write_jsonl(
        &dev_proper,
        &serde_json::json!({
            "tokens": dev_tokens,
            "lemmas": dev_lemmas,
            "upos": ["PROPN", "PROPN"]
        }),
    );
    write_jsonl(
        &dev_common,
        &serde_json::json!({
            "tokens": dev_tokens,
            "lemmas": dev_lemmas,
            "upos": ["NOUN", "NOUN"]
        }),
    );
    write_jsonl(
        &test,
        &serde_json::json!({
            "tokens": ["Londons"],
            "lemmas": ["London"],
            "upos": ["PROPN"]
        }),
    );
    std::fs::write(
        &entities,
        "Anchors\tAnchors\tidentity\nBerlins\tBerlins\tidentity\n",
    )
    .unwrap();
    let entity_sources = [Paradigm {
        label: "entities/test".to_owned(),
        path: entities,
        lemma_first: false,
        normalized_unique: false,
        expected: None,
    }];

    let proper_destination = temporary.path().join("xx_dev_proper");
    let common_destination = temporary.path().join("xx_dev_common");
    let proper_spec = Spec {
        destination: &proper_destination,
        train: Some(&train),
        dev: Some(&dev_proper),
        test: Some(&test),
        paradigms: &[],
        entities: &entity_sources,
        name: "xx_dev_proper",
        language: "Test",
        iso639_3: "eng",
        span_case: None,
        span_multiword: None,
        entity_lower: false,
        compound_marker: None,
        token_case: None,
        lemma_adapter: false,
        context_keys: false,
        phrases: &[],
        proper_noun_threshold: None,
    };
    let common_spec = Spec {
        destination: &common_destination,
        dev: Some(&dev_common),
        name: "xx_dev_common",
        ..proper_spec
    };
    let proper = build_lexicon(&proper_spec).unwrap();
    let common = build_lexicon(&common_spec).unwrap();

    // UPOS changes only the entity evidence here: with fewer than 50 PROPN
    // rows it cannot select another proper-noun threshold. Common dev forms,
    // lemmas and every selection outcome remain exactly equal.
    assert_eq!(proper.variant, common.variant);
    assert_eq!(proper.thresholds, common.thresholds);
    assert_eq!(proper.selected_on, common.selected_on);
    assert_eq!(proper.traffic, common.traffic);

    let options = crate::probe::Options::default().entity_lookup();
    let proper_lexicon = crate::lexicon::Lexicon::open(&proper_destination).unwrap();
    let common_lexicon = crate::lexicon::Lexicon::open(&common_destination).unwrap();
    for (surface, lemma) in [("Berlins", "Berlin"), ("Parises", "Paris")] {
        let answer = proper_lexicon.lemma_with_info(surface, false, options);
        assert_eq!(answer.lemma, lemma);
        assert_eq!(answer.source, crate::probe::Source::Entity);
    }
    assert_eq!(
        common_lexicon
            .lemma_with_info("Parises", false, options)
            .source,
        crate::probe::Source::Identity
    );
    // Ambiguous train PROPN evidence still protects itself. `Berlins` above
    // shows the guard doing its work end to end: the source carries a
    // single-word identity row for it, and because the train split claims that
    // surface the row never enters, so the annotated rewrite survives.
    let ambiguous = proper_lexicon.lemma_with_info("Ambigs", false, options);
    assert_eq!(ambiguous.lemma, "Ambigs");
    assert_eq!(ambiguous.source, crate::probe::Source::Entity);
    // The guard is the training signal, which is all that exists when sources
    // are read; development proper nouns only arrive after selection. A source
    // row about a surface only the dev split knows is therefore still admitted
    // and still dominates. The measured gate buys back the train collisions,
    // not these, and comes out ahead anyway.
    let anchors = proper_lexicon.lemma_with_info("Anchors", false, options);
    assert_eq!(anchors.lemma, "Anchors");
    assert_eq!(anchors.source, crate::probe::Source::Entity);
    for lexicon in [&proper_lexicon, &common_lexicon] {
        assert_eq!(
            lexicon.lemma_with_info("Londons", false, options).source,
            crate::probe::Source::Identity
        );
    }
}

#[test]
fn corpus_proper_nouns_use_the_common_script_guard() {
    assert_eq!(
        guarded_proper_nouns(
            vec![
                ("كَتَبُوا!".to_owned(), "كَتَب".to_owned()),
                ("għoxorَيْ".to_owned(), "عشر".to_owned()),
                ("Latinعربي".to_owned(), "Latinعربي".to_owned()),
            ],
            "ara",
        ),
        vec![
            ("كَتَبُوا!".to_owned(), "كَتَب".to_owned()),
            ("għoxorَيْ".to_owned(), "għoxorَيْ".to_owned()),
            ("Latinعربي".to_owned(), "Latinعربي".to_owned()),
        ]
    );
}

#[test]
fn compiled_entities_keep_exact_and_lowercase_namespaces_separate() {
    let temporary = tempfile::tempdir().unwrap();
    let train = temporary.path().join("train.jsonl");
    let entities = temporary.path().join("entities.tsv");
    std::fs::write(
        &train,
        "{\"tokens\":[\"cats\"],\"lemmas\":[\"cat\"],\"upos\":[\"NOUN\"]}\n",
    )
    .unwrap();
    std::fs::write(
        &entities,
        "2 2\t2 2\tidentity\nOpen AI\tOpen AI\tidentity\n",
    )
    .unwrap();
    let destination = temporary.path().join("xx_entity_case");
    let entity_sources = [Paradigm {
        label: "entities/test".to_owned(),
        path: entities,
        lemma_first: false,
        normalized_unique: false,
        expected: None,
    }];
    let spec = Spec {
        destination: &destination,
        train: Some(&train),
        dev: None,
        test: None,
        paradigms: &[],
        entities: &entity_sources,
        name: "xx_entity_case",
        language: "Test",
        iso639_3: "zxx",
        span_case: None,
        span_multiword: None,
        entity_lower: false,
        compound_marker: None,
        token_case: None,
        lemma_adapter: false,
        context_keys: false,
        phrases: &[],
        proper_noun_threshold: None,
    };
    build_lexicon(&spec).unwrap();
    let lexicon = crate::lexicon::Lexicon::open(&destination).unwrap();

    let exact = lexicon.lemma_with_info(
        "Open AI",
        false,
        crate::probe::Options::default().entity_lookup(),
    );
    assert_eq!(exact.lemma, "Open AI");
    assert_eq!(exact.source, crate::probe::Source::Entity);

    let case_sensitive_miss = lexicon.lemma_with_info(
        "OPEN AI",
        false,
        crate::probe::Options::default().entity_lookup(),
    );
    assert_eq!(case_sensitive_miss.source, crate::probe::Source::Identity);

    let folded = lexicon.lemma_with_info(
        "OPEN AI",
        false,
        crate::probe::Options {
            use_entities_lower: true,
            ..crate::probe::Options::default().entity_lookup()
        },
    );
    assert_eq!(folded.lemma, "Open AI");
    assert_eq!(folded.source, crate::probe::Source::EntityLower);

    // Stored keys are `0<exact>` and `1<lower>`. A raw surface that happens
    // to equal either internal key must remain a miss; the removed legacy
    // unprefixed fallback used to turn both of these into false entity hits.
    for surface in ["02 2", "12 2", "0Open AI", "1open ai"] {
        let collision = lexicon.lemma_with_info(
            surface,
            false,
            crate::probe::Options::default().entity_lookup(),
        );
        assert_eq!(collision.lemma, surface);
        assert_eq!(
            collision.source,
            crate::probe::Source::Identity,
            "{surface}"
        );
    }

    let protected = lexicon.lemma_with_info(
        "Open AI",
        false,
        crate::probe::Options {
            protected: true,
            ..crate::probe::Options::default().entity_lookup()
        },
    );
    assert_eq!(protected.lemma, "Open AI");
    assert_eq!(protected.source, crate::probe::Source::Protected);
}

#[test]
fn identical_builds_reuse_one_content_addressed_generation() {
    let temporary = tempfile::tempdir().unwrap();
    let train = temporary.path().join("train.jsonl");
    std::fs::write(
        &train,
        "{\"tokens\":[\"Cats\"],\"lemmas\":[\"cat\"],\"upos\":[\"NOUN\"]}\n",
    )
    .unwrap();
    let destination = temporary.path().join("xx_reproducible");
    let spec = Spec {
        destination: &destination,
        train: Some(&train),
        dev: None,
        test: None,
        paradigms: &[],
        entities: &[],
        name: "xx_reproducible",
        language: "Test",
        iso639_3: "zxx",
        span_case: None,
        span_multiword: None,
        entity_lower: false,
        compound_marker: None,
        token_case: None,
        lemma_adapter: false,
        context_keys: false,
        phrases: &[],
        proper_noun_threshold: None,
    };

    build_lexicon(&spec).unwrap();
    let first = std::fs::read_to_string(destination.join("current")).unwrap();
    build_lexicon(&spec).unwrap();
    let second = std::fs::read_to_string(destination.join("current")).unwrap();
    assert_eq!(first, second);
    let leaves = std::fs::read_dir(destination.join(".generations"))
        .unwrap()
        .filter_map(std::result::Result::ok)
        .filter(|entry| !entry.file_name().to_string_lossy().starts_with('.'))
        .count();
    assert_eq!(leaves, 1);
}

#[test]
fn the_written_receipt_carries_both_profiles_and_every_pairwise_counter() {
    let temporary = tempfile::tempdir().unwrap();
    let train = temporary.path().join("train.jsonl");
    let dev = temporary.path().join("dev.jsonl");
    // A leading full stop keeps every interesting token out of the
    // sentence-initial bucket, and four agreeing `-ed` pairs give the suffix
    // rule the support it needs.
    write_jsonl(
        &train,
        &serde_json::json!({
            "tokens": [".", "walked", "talked", "jumped", "looked"],
            "lemmas": [".", "walk", "talk", "jump", "look"],
            "upos": ["PUNCT", "VERB", "VERB", "VERB", "VERB"]
        }),
    );
    // Sixty unseen forms only the rule answers: the lattice ships `rules`
    // with a lead of 60 on 60 discordant tokens, and with rules off every
    // candidate falls back to copying — the NER profile the receipt must say.
    let dev_tokens: Vec<&str> = std::iter::once(".").chain([("helped"); 60]).collect();
    let dev_lemmas: Vec<&str> = std::iter::once(".").chain([("help"); 60]).collect();
    write_jsonl(
        &dev,
        &serde_json::json!({
            "tokens": dev_tokens,
            "lemmas": dev_lemmas,
            "upos": vec!["X"; 61]
        }),
    );
    let destination = temporary.path().join("xx_receipt");
    let spec = Spec {
        destination: &destination,
        train: Some(&train),
        dev: Some(&dev),
        test: None,
        paradigms: &[],
        entities: &[],
        name: "xx_receipt",
        language: "Test",
        iso639_3: "zxx",
        span_case: None,
        span_multiword: None,
        entity_lower: false,
        compound_marker: None,
        token_case: None,
        lemma_adapter: false,
        context_keys: false,
        phrases: &[],
        proper_noun_threshold: None,
    };

    let built = build_lexicon(&spec).unwrap();
    assert_eq!(built.variant, Variant::Rules.name());
    let receipt = built.selection.expect("dev selection receipt");
    assert_eq!(receipt.tokens(), 61);
    assert_eq!(receipt.copied(), 1);
    let hits: Vec<u64> = receipt
        .candidates()
        .iter()
        .map(|candidate| candidate.hits)
        .collect();
    assert_eq!(hits, [1, 1, 61, 1, 61]);
    let profiles = receipt.profiles().expect("both profiles measured");
    assert_eq!(profiles.map(|profile| profile.ner_hits), [1, 1, 1, 1, 1]);
    assert_eq!(
        profiles.map(|profile| profile.rivals),
        [
            [0, 0, 60, 0, 60],
            [0, 0, 60, 0, 60],
            [60, 60, 0, 60, 0],
            [0, 0, 60, 0, 60],
            [60, 60, 0, 60, 0],
        ]
    );

    // The generation's own metadata carries the same rows, and the reader —
    // which re-derives the verdict from them — accepts the bundle.
    let generation = std::fs::read_to_string(destination.join("current")).unwrap();
    let meta_path = destination
        .join(".generations")
        .join(generation.trim_end())
        .join("meta.json");
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(meta_path).unwrap()).unwrap();
    let rules_row = meta
        .get("selection")
        .and_then(|selection| selection.get("candidates"))
        .and_then(|candidates| candidates.get("rules"))
        .unwrap();
    assert_eq!(rules_row.get("hits"), Some(&serde_json::json!(61)));
    assert_eq!(rules_row.get("ner_hits"), Some(&serde_json::json!(1)));
    assert_eq!(
        rules_row.get("rivals"),
        Some(&serde_json::json!({
            "identity": 60, "corpus": 60, "rules": 0, "lookup": 60, "full": 0
        }))
    );
    assert_eq!(
        crate::lexicon::Lexicon::open(&destination)
            .unwrap()
            .lemma("helped", false),
        "help"
    );
}

/// The refusals the imported-table contract owes its Python ancestor, worded
/// and numbered exactly as that loop worded and numbered them.
#[test]
fn a_normalized_import_is_refused_the_way_the_pipeline_always_refused_it() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("zxx.tsv");
    let name = path.display().to_string();

    let cases: [(&[u8], String); 8] = [
        (
            b"cats\tcat".as_slice(),
            format!("{name}:1: final row must end with a newline"),
        ),
        (
            b"ants\tant\ncats\tcat".as_slice(),
            format!("{name}:2: final row must end with a newline"),
        ),
        // A lone carriage return ends a row for Python's text reader, so the
        // row it opens is one that does not end with a newline.
        (
            b"ants\tant\rcats\tcat\n".as_slice(),
            format!("{name}:1: final row must end with a newline"),
        ),
        (
            b"cats\n".as_slice(),
            format!("{name}:1: expected non-empty form<TAB>lemma"),
        ),
        (
            b"cats\t\n".as_slice(),
            format!("{name}:1: expected non-empty form<TAB>lemma"),
        ),
        (
            b"cats\tcat\tNOUN\n".as_slice(),
            format!("{name}:1: expected non-empty form<TAB>lemma"),
        ),
        (
            "cafe\u{301}s\tcafe\u{301}\n".as_bytes(),
            format!("{name}:1: table text must be NFC-normalized"),
        ),
        (
            b"cats\tca\x00t\n".as_slice(),
            format!("{name}:1: table contains a control character"),
        ),
    ];
    for (bytes, expected) in cases {
        std::fs::write(&path, bytes).unwrap();
        assert_eq!(
            validate_normalized_rows(&path).unwrap_err().to_string(),
            expected
        );
    }

    for bytes in [
        b"cats\tcat\nants\tant\n".as_slice(),
        b"cats\tcat\ncats\tcat\n".as_slice(),
    ] {
        std::fs::write(&path, bytes).unwrap();
        assert_eq!(
            validate_normalized_rows(&path).unwrap_err().to_string(),
            format!("{name}:2: forms must be strictly sorted and unique")
        );
    }

    // Ordering is over the form column alone, whatever the lemma does, and a
    // CRLF row is as terminated as an LF one.
    std::fs::write(&path, b"ants\tzebra\r\ncats\tcat\n").unwrap();
    assert_eq!(validate_normalized_rows(&path).unwrap(), 2);

    // Zero-width joiners are not controls; the Persian imports need them.
    std::fs::write(&path, "\u{645}\u{6cc}\u{200c}\u{631}\u{648}\u{645}\tx\n").unwrap();
    assert_eq!(validate_normalized_rows(&path).unwrap(), 1);

    std::fs::write(&path, b"").unwrap();
    assert_eq!(validate_normalized_rows(&path).unwrap(), 0);

    std::fs::write(&path, b"cats\tca\xfft\n").unwrap();
    assert_eq!(
        validate_normalized_rows(&path).unwrap_err().to_string(),
        format!("{name}:1: table is not valid UTF-8")
    );
}

/// A row breaking two rules is refused for the one its ancestor named first.
#[test]
fn imported_row_refusals_keep_their_precedence() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("zxx.tsv");
    let name = path.display().to_string();

    // Unordered *and* not NFC: NFC came first in the Python loop.
    std::fs::write(&path, "zzz\tz\ncafe\u{301}\tcafe\u{301}\n").unwrap();
    assert_eq!(
        validate_normalized_rows(&path).unwrap_err().to_string(),
        format!("{name}:2: table text must be NFC-normalized")
    );

    // Unordered *and* carrying a control: the control came first.
    std::fs::write(&path, b"zzz\tz\nca\x01t\tcat\n").unwrap();
    assert_eq!(
        validate_normalized_rows(&path).unwrap_err().to_string(),
        format!("{name}:2: table contains a control character")
    );
}

// --- The source-order ranker's own table reader -----------------------------

use super::source_table::{Policy, next_batch};

/// Everything one whole table produced, whatever block size it was read at.
#[derive(Debug, Default, PartialEq, Eq)]
struct Streamed {
    rows: Vec<(String, String, usize)>,
    lines: usize,
    blank: usize,
    invalid: usize,
    comment: usize,
    mismatch: usize,
    previous: Option<String>,
}

/// The ranker's Python loop, so a test can read a table out of a byte slice.
///
/// Same shape as `stream_tsv`: refill a window, ask for batches until the
/// reader wants more, hand every returned pair to the caller. `block` is the
/// knob every test below varies and no test below expects to matter.
fn stream_table(
    table: &[u8],
    policy: &Policy<'_>,
    block: usize,
) -> std::result::Result<Streamed, String> {
    let path = "zxx.tsv";
    let mut streamed = Streamed::default();
    let mut window: Vec<u8> = Vec::new();
    let mut offset = 0_usize;
    let mut read = 0_usize;
    let mut at_eof = false;
    loop {
        loop {
            let first_line = streamed.lines.saturating_add(1);
            let parsed = next_batch(
                &window,
                offset,
                first_line,
                streamed.previous.as_deref(),
                policy,
                at_eof,
            )
            .map_err(|refusal| refusal.render(path))?;
            let Some(batch) = parsed else { break };
            if batch.lines == 0 {
                break;
            }
            assert_eq!(
                batch
                    .rows
                    .len()
                    .saturating_add(batch.blank)
                    .saturating_add(batch.invalid)
                    .saturating_add(batch.comment)
                    .saturating_add(batch.mismatch),
                batch.lines,
                "the reader lost or invented a row",
            );
            offset = offset.saturating_add(batch.consumed);
            streamed.lines = streamed.lines.saturating_add(batch.lines);
            streamed.blank = streamed.blank.saturating_add(batch.blank);
            streamed.invalid = streamed.invalid.saturating_add(batch.invalid);
            streamed.comment = streamed.comment.saturating_add(batch.comment);
            streamed.mismatch = streamed.mismatch.saturating_add(batch.mismatch);
            streamed.previous = batch.previous_form.map(std::borrow::Cow::into_owned);
            for (form, lemma, line) in batch.rows {
                streamed
                    .rows
                    .push((form.into_owned(), lemma.into_owned(), line));
            }
        }
        if at_eof {
            break;
        }
        window.drain(..offset);
        offset = 0;
        let next = read.saturating_add(block).min(table.len());
        if next == read {
            at_eof = true;
            continue;
        }
        window.extend_from_slice(&table[read..next]);
        read = next;
    }
    assert_eq!(offset, window.len(), "the reader left bytes unread");
    Ok(streamed)
}

/// Reads one table at five block sizes and insists all five agree: the output
/// is a function of the table alone.
fn stream_at_every_block(
    table: &[u8],
    policy: &Policy<'_>,
) -> std::result::Result<Streamed, String> {
    let reference = stream_table(table, policy, 1);
    for block in [3_usize, 7, 64, 1 << 12, 1 << 22] {
        assert_eq!(
            stream_table(table, policy, block),
            reference,
            "block size {block} changed the answer",
        );
    }
    reference
}

const fn legacy(iso639_3: &str, lemma_first: bool) -> Policy<'_> {
    Policy {
        iso639_3,
        lemma_first,
        normalized_unique: false,
        require_newline: false,
        max_line_bytes: 16 * 1024 * 1024,
        max_batch_rows: 4096,
        max_batch_bytes: 16 * 1024 * 1024,
    }
}

const fn imported(iso639_3: &str) -> Policy<'_> {
    Policy {
        iso639_3,
        lemma_first: false,
        normalized_unique: true,
        require_newline: true,
        max_line_bytes: 16 * 1024 * 1024,
        max_batch_rows: 4096,
        max_batch_bytes: 16 * 1024 * 1024,
    }
}

fn rows(pairs: &[(&str, &str, usize)]) -> Vec<(String, String, usize)> {
    pairs
        .iter()
        .map(|&(form, lemma, line)| (form.to_owned(), lemma.to_owned(), line))
        .collect()
}

/// A legacy table skips what it has always skipped, and counts each skip.
#[test]
fn legacy_source_rows_are_skipped_exactly_where_they_were() {
    let table = concat!(
        "\n", // blank
        "# upstream note\n",
        "cats\tcat\n",
        "\tlemma\n",         // empty first column
        "form\t\n",          // empty second column
        "birds\tbird\tPL\n", // third column ignored in silence
        "\r\n",              // a bare carriage return is still a blank row
    );
    let streamed = stream_at_every_block(table.as_bytes(), &legacy("eng", false)).unwrap();
    assert_eq!(
        streamed.rows,
        rows(&[("cats", "cat", 3), ("birds", "bird", 6)])
    );
    assert_eq!(
        (
            streamed.lines,
            streamed.blank,
            streamed.comment,
            streamed.invalid
        ),
        (7, 2, 1, 2)
    );
}

/// `UniMorph` ships `lemma <TAB> form`; the flag swaps them, nothing else.
#[test]
fn the_column_order_flag_only_swaps_the_columns() {
    let table = "кот\tкотами\tINS;PL\n";
    assert_eq!(
        stream_at_every_block(table.as_bytes(), &legacy("rus", true))
            .unwrap()
            .rows,
        rows(&[("котами", "кот", 1)])
    );
    assert_eq!(
        stream_at_every_block(table.as_bytes(), &legacy("rus", false))
            .unwrap()
            .rows,
        rows(&[("кот", "котами", 1)])
    );
}

/// Only `# ` opens a comment, and only after the row has been read properly.
#[test]
fn a_comment_is_that_prefix_and_is_still_checked_for_what_rows_are_checked_for() {
    let policy = legacy("eng", false);
    // `#` without the space is an ordinary — and here one-column — row.
    assert_eq!(
        stream_at_every_block(b"#note\n", &policy).unwrap_err(),
        "zxx.tsv:1: expected at least two tab-separated columns"
    );
    // A comment that is not UTF-8 refuses the table rather than being skipped.
    assert!(
        stream_at_every_block(b"# \xff\n", &policy)
            .unwrap_err()
            .starts_with("zxx.tsv:1: invalid UTF-8: ")
    );
    assert_eq!(
        stream_at_every_block(b"# note\x01\n", &policy).unwrap_err(),
        "zxx.tsv:1: TSV row contains a control character"
    );
    let streamed = stream_at_every_block(b"# note\n", &policy).unwrap();
    assert_eq!((streamed.lines, streamed.comment), (1, 1));
}

/// A row that breaks two rules is refused for the one its ancestor named first.
#[test]
fn source_row_refusals_keep_their_precedence() {
    let policy = legacy("eng", false);
    // Not UTF-8 *and* carrying a control *and* missing its second column.
    assert!(
        stream_at_every_block(b"ca\xfft\x01\n", &policy)
            .unwrap_err()
            .starts_with("zxx.tsv:1: invalid UTF-8: ")
    );
    // Carrying a control *and* missing its second column: the control first.
    assert_eq!(
        stream_at_every_block(b"ca\x01t\n", &policy).unwrap_err(),
        "zxx.tsv:1: TSV row contains a control character"
    );
    // Empty *and* whitespace-only cannot both hold, but empty is checked
    // first, and only there does the legacy tolerance apply.
    assert_eq!(stream_at_every_block(b"\t\n", &policy).unwrap().invalid, 1);
    assert_eq!(
        stream_at_every_block(b" \tcat\n", &policy).unwrap_err(),
        "zxx.tsv:1: the first two columns must not be whitespace-only"
    );
    // Non-breaking space is whitespace to `str.strip()` and to this reader.
    assert_eq!(
        stream_at_every_block("cat\t\u{a0}\n".as_bytes(), &policy).unwrap_err(),
        "zxx.tsv:1: the first two columns must not be whitespace-only"
    );
    // A zero-width non-joiner is not whitespace, and the Persian imports need it.
    assert_eq!(
        stream_at_every_block("\u{200c}\tcat\n".as_bytes(), &policy)
            .unwrap()
            .lines,
        1
    );
    // DELETE and the C1 controls are `Cc` too, and only the tab is exempt.
    for control in ["\u{7f}", "\u{85}", "\u{9f}"] {
        assert_eq!(
            stream_at_every_block(format!("ca{control}t\tcat\n").as_bytes(), &policy).unwrap_err(),
            "zxx.tsv:1: TSV row contains a control character"
        );
    }
}

/// A carriage return terminates a row for `for_each_paradigm` and refuses one
/// here. The two readers replace different Python streams and stay different.
#[test]
fn an_inner_carriage_return_is_a_control_character_not_a_terminator() {
    let policy = legacy("eng", false);
    assert_eq!(
        stream_at_every_block(b"cats\rdogs\tdog\n", &policy).unwrap_err(),
        "zxx.tsv:1: TSV row contains a control character"
    );
    // One at the very end of the row is the CRLF the format allows.
    assert_eq!(
        stream_at_every_block(b"cats\tcat\r\n", &policy)
            .unwrap()
            .rows,
        rows(&[("cats", "cat", 1)])
    );
    // Including on a final row that never got its newline.
    assert_eq!(
        stream_at_every_block(b"cats\tcat\r", &policy).unwrap().rows,
        rows(&[("cats", "cat", 1)])
    );
}

/// An imported table refuses everything a legacy table forgives.
#[test]
fn imported_tables_refuse_what_legacy_tables_skip() {
    let policy = imported("eng");
    assert_eq!(
        stream_at_every_block(b"\n", &policy).unwrap_err(),
        "zxx.tsv:1: expected at least two tab-separated columns"
    );
    assert_eq!(
        stream_at_every_block(b"# note\n", &policy).unwrap_err(),
        "zxx.tsv:1: expected at least two tab-separated columns"
    );
    assert_eq!(
        stream_at_every_block(b"\tcat\n", &policy).unwrap_err(),
        "zxx.tsv:1: the first two columns must be non-empty"
    );
    assert_eq!(
        stream_at_every_block(b"cats\tcat", &policy).unwrap_err(),
        "zxx.tsv:1: imported table row must end with a newline"
    );
    // The same unterminated row is an ordinary row in a legacy table.
    assert_eq!(
        stream_at_every_block(b"cats\tcat", &legacy("eng", false))
            .unwrap()
            .rows,
        rows(&[("cats", "cat", 1)])
    );
}

/// The row length bound is measured, refused and numbered as it always was.
#[test]
fn an_overlong_row_is_refused_at_its_own_line() {
    let mut policy = legacy("eng", false);
    policy.max_line_bytes = 8;
    // Eight bytes and a newline is the longest row that fits.
    assert_eq!(
        stream_at_every_block(b"cats\tcat\nlonger\tlemma\n", &policy).unwrap_err(),
        "zxx.tsv:2: row exceeds 8 bytes"
    );
    assert_eq!(
        stream_at_every_block(b"cats\tcat\n", &policy)
            .unwrap()
            .lines,
        1
    );
    // A final row with no newline is bounded by the same eight bytes.
    assert_eq!(
        stream_at_every_block(b"catss\tcat", &policy).unwrap_err(),
        "zxx.tsv:1: row exceeds 8 bytes"
    );
}

/// The sort and NFC contract holds across batches, not only inside one.
#[test]
fn imported_ordering_state_crosses_the_batch_boundary() {
    let mut policy = imported("eng");
    policy.max_batch_rows = 2;
    let streamed = stream_at_every_block(b"aa\ta\nbb\tb\ncc\tc\ndd\td\nee\te\n", &policy).unwrap();
    assert_eq!(streamed.rows.len(), 5);
    assert_eq!(streamed.previous.as_deref(), Some("ee"));
    // The duplicate sits in the batch after the one its predecessor ended.
    assert_eq!(
        stream_at_every_block(b"aa\ta\nbb\tb\ncc\tc\ncc\tc\n", &policy).unwrap_err(),
        "zxx.tsv:4: imported forms must be strictly sorted and unique"
    );
    assert_eq!(
        stream_at_every_block("aa\ta\ncafe\u{301}\tx\nzz\tz\n".as_bytes(), &policy).unwrap_err(),
        "zxx.tsv:2: imported table must be NFC-normalized"
    );
}

/// A refusal the flush is holding must wait for the row that triggers it:
/// a row that is itself wrong is refused first.
#[test]
fn a_flush_refusal_never_overtakes_the_row_that_triggers_the_flush() {
    let mut policy = imported("eng");
    policy.max_batch_rows = 2;
    // Rows 1 and 2 fill the batch and row 2 is out of order; row 3 triggers
    // the flush that would say so — but row 3 cannot be decoded, and its own
    // refusal is the one the table earns.
    assert_eq!(
        stream_at_every_block(b"bb\tb\naa\ta\nca\x01t\tcat\n", &policy).unwrap_err(),
        "zxx.tsv:3: TSV row contains a control character"
    );
    // With a row 3 that decodes, the held refusal is the one that arrives.
    assert_eq!(
        stream_at_every_block(b"bb\tb\naa\ta\ncc\tc\n", &policy).unwrap_err(),
        "zxx.tsv:2: imported forms must be strictly sorted and unique"
    );
    // And at end of file the flush happens with no row left to outrank it.
    assert_eq!(
        stream_at_every_block(b"bb\tb\naa\ta\n", &policy).unwrap_err(),
        "zxx.tsv:2: imported forms must be strictly sorted and unique"
    );
}

/// A batch is cut where the ancestor cut it: the trigger row starts the next.
#[test]
fn a_full_batch_ends_before_the_row_that_filled_it() {
    let mut policy = legacy("eng", false);
    policy.max_batch_rows = 2;
    let streamed = stream_at_every_block(b"aa\ta\nbb\tb\n\ncc\tc\n", &policy).unwrap();
    assert_eq!(
        streamed.rows,
        rows(&[("aa", "a", 1), ("bb", "b", 2), ("cc", "c", 4)])
    );
    assert_eq!((streamed.lines, streamed.blank), (4, 1));
    // The byte bound cuts in the same place as the row bound.
    policy.max_batch_rows = 4096;
    policy.max_batch_bytes = 6;
    let streamed = stream_at_every_block(b"aa\ta\nbb\tb\ncc\tc\n", &policy).unwrap();
    assert_eq!(streamed.rows.len(), 3);
}

/// The builder's script guard rejects a rewrite; it never refuses the table.
#[test]
fn script_incompatible_rows_are_counted_and_dropped() {
    let table = "\u{43a}\u{43e}\u{442}\u{430}\u{43c}\u{438}\tcat\n\u{43a}\u{43e}\u{442}\u{430}\u{43c}\u{438}\t\u{43a}\u{43e}\u{442}\n";
    let streamed = stream_at_every_block(table.as_bytes(), &legacy("rus", false)).unwrap();
    assert_eq!(streamed.mismatch, 1);
    assert_eq!(streamed.rows.len(), 1);
    assert_eq!(streamed.lines, 2);
}

/// An empty table is an empty answer, not a refusal and not a lost line.
#[test]
fn an_empty_table_reads_as_nothing() {
    for policy in [legacy("eng", false), imported("eng")] {
        let streamed = stream_at_every_block(b"", &policy).unwrap();
        assert_eq!(streamed, Streamed::default());
    }
}

#[test]
fn context_keys_learn_from_train_and_dev_and_answer_only_when_asked() {
    use crate::probe::{Options, Source};

    let temporary = tempfile::tempdir().unwrap();
    let train = temporary.path().join("train.jsonl");
    let dev = temporary.path().join("dev.jsonl");
    let verb = serde_json::json!({
        "tokens": ["Они", "стали", "ждать"],
        "lemmas": ["они", "стать", "ждать"],
        "upos": ["PRON", "VERB", "VERB"]
    });
    let noun = serde_json::json!({
        "tokens": ["Из", "стали"],
        "lemmas": ["из", "сталь"],
        "upos": ["ADP", "NOUN"]
    });
    let rows = [&verb, &verb, &verb, &noun].map(ToString::to_string);
    std::fs::write(&train, rows.join("\n") + "\n").unwrap();
    // The second noun arrives with dev: the keys exist only if dev joins.
    // The filler gives the lattice a lead to ship the corpus tables on.
    let filler = serde_json::json!({
        "tokens": vec!["Они"; 12],
        "lemmas": vec!["они"; 12],
        "upos": vec!["PRON"; 12]
    });
    std::fs::write(&dev, format!("{noun}\n{filler}\n")).unwrap();
    let build = |name: &str, context_keys: bool| {
        let destination = temporary.path().join(name);
        let spec = Spec {
            destination: &destination,
            train: Some(&train),
            dev: Some(&dev),
            test: None,
            paradigms: &[],
            entities: &[],
            name,
            language: "Russian",
            iso639_3: "rus",
            span_case: None,
            span_multiword: None,
            entity_lower: false,
            compound_marker: None,
            token_case: None,
            lemma_adapter: false,
            context_keys,
            phrases: &[],
            proper_noun_threshold: None,
        };
        build_lexicon(&spec).unwrap();
        crate::lexicon::Lexicon::open(&destination).unwrap()
    };

    let lexicon = build("ru_context", true);
    let options = Options::default();
    let ask = |upos, previous, next| {
        let answer = lexicon.lemma_in_context("стали", false, options, upos, previous, next);
        (answer.lemma.into_owned(), answer.source)
    };
    let context = ("сталь".to_owned(), Source::Context);
    let exact = ("стать".to_owned(), Source::Exact);
    assert_eq!(lexicon.lemma("стали", false), "стать");
    assert_eq!(ask(None, None, None), exact);
    assert_eq!(ask(Some("NOUN"), None, None), context);
    assert_eq!(ask(Some("VERB"), Some("ИЗ"), None), context);
    assert_eq!(ask(Some("VERB"), Some("они"), Some("ждать")), exact);
    let sentence = lexicon.lemmatize_in_context(&["Из", "стали"], None, true);
    assert_eq!(
        sentence.get(1).map(|answer| answer.source),
        Some(Source::Context)
    );

    let plain = build("ru_plain", false);
    let answer = plain.lemma_in_context("стали", false, options, Some("NOUN"), None, None);
    assert_eq!(
        (answer.lemma.as_ref(), answer.source),
        ("стать", Source::Exact)
    );
}
