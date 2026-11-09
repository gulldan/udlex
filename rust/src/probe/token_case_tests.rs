use std::borrow::Cow;

use super::{Options, Source, Tables, lemma_with_info, recase_surface, span_lemma_with_info};
use crate::rule::{CaseAction, Rule};

/// One stored row per table and a title-casing rule that drops a final `n`.
#[derive(Default)]
struct Stub {
    exact: Option<(&'static str, &'static str)>,
    external: Option<(&'static str, &'static str)>,
}

fn row(stored: Option<(&'static str, &'static str)>, key: &str) -> Option<Cow<'static, str>> {
    stored.and_then(|(form, lemma)| (form == key).then_some(Cow::Borrowed(lemma)))
}

impl Tables for Stub {
    fn entities(&self, _key: &str) -> Option<Cow<'_, str>> {
        None
    }

    fn entities_lower(&self, _key: &str) -> Option<Cow<'_, str>> {
        None
    }

    fn exact(&self, key: &str) -> Option<Cow<'_, str>> {
        row(self.exact, key)
    }

    fn lower(&self, _key: &str) -> Option<Cow<'_, str>> {
        None
    }

    fn external(&self, key: &str) -> Option<Cow<'_, str>> {
        row(self.external, key)
    }

    fn rule(&self, _bucket: u8, suffix: &str) -> Option<(Rule, Cow<'_, str>)> {
        let rule = Rule {
            cut: 1,
            add: 0,
            case: CaseAction::Title,
        };
        (suffix == "n").then_some((rule, Cow::Borrowed("")))
    }

    fn is_empty(&self) -> bool {
        false
    }

    fn common_is_empty(&self) -> bool {
        false
    }

    fn max_suffix(&self) -> usize {
        1
    }
}

/// The probe's answer, then — when `surface` — the convention a stamped
/// lexicon's token wrappers apply after it.
fn answer(tables: &Stub, token: &str, sentence_initial: bool, surface: bool) -> (String, Source) {
    let mut answer = lemma_with_info(tables, token, sentence_initial, Options::default());
    if surface {
        recase_surface(&mut answer, token, sentence_initial, tables.casing());
    }
    (answer.lemma.into_owned(), answer.source)
}

#[test]
fn surface_case_restores_a_mid_sentence_rule_answer_and_keep_does_not() {
    let tables = Stub::default();
    let token = "Europol-Mitarbeitern";
    assert_eq!(
        answer(&tables, token, false, false),
        ("Europol-mitarbeiter".to_owned(), Source::Rule)
    );
    assert_eq!(
        answer(&tables, token, false, true),
        ("Europol-Mitarbeiter".to_owned(), Source::Rule)
    );
    // Sentence-initial capitals are not names by position, so they stay.
    assert_eq!(answer(&tables, token, true, true).0, "Europol-mitarbeiter");
}

#[test]
fn surface_case_copies_only_the_shared_prefix_of_a_lowered_answer() {
    let tables = Stub {
        external: Some(("рівненщини", "рівненщина")),
        ..Stub::default()
    };
    assert_eq!(
        answer(&tables, "Рівненщини", false, true),
        ("Рівненщина".to_owned(), Source::ExternalLower)
    );
    assert_eq!(answer(&tables, "Рівненщини", false, false).0, "рівненщина");
}

#[test]
fn surface_case_leaves_exact_hits_and_length_changing_casing_alone() {
    let exact = Stub {
        exact: Some(("QNX", "qnx")),
        ..Stub::default()
    };
    assert_eq!(
        answer(&exact, "QNX", false, true),
        ("qnx".to_owned(), Source::Exact)
    );
    // `İ` lowercases to `i` and a combining dot: the characters no longer
    // line up, so the answer is returned as the table spelled it.
    let dotted = Stub {
        external: Some(("i\u{307}stanbulda", "istanbul")),
        ..Stub::default()
    };
    assert_eq!(
        answer(&dotted, "İstanbulda", false, true),
        ("istanbul".to_owned(), Source::ExternalLower)
    );
}

#[test]
fn a_rule_answer_on_a_span_keeps_the_surface_casing() {
    let tables = Stub::default();
    let span = span_lemma_with_info(
        &tables,
        "Europol-Mitarbeitern",
        &[],
        None,
        false,
        Options::default(),
    );
    assert_eq!(span.lemma, "Europol-Mitarbeiter");
    assert_eq!(span.source, Source::Rule);
}

#[test]
fn surface_case_updates_owned_answers_and_keeps_noop_storage() {
    let mut answer = super::Answer {
        lemma: Cow::Owned("qnx".to_owned()),
        source: Source::Exact,
    };
    let original = answer.lemma.as_ptr();
    recase_surface(
        &mut answer,
        "QNX",
        false,
        crate::normalization::Casing::Root,
    );
    assert_eq!(answer.lemma, "qnx");
    assert_eq!(answer.lemma.as_ptr(), original);
    assert_eq!(answer.source, Source::Exact);
    assert!(matches!(answer.lemma, Cow::Owned(_)));

    let mut answer = super::Answer {
        lemma: Cow::Owned("рівненщина".to_owned()),
        source: Source::ExternalLower,
    };
    recase_surface(
        &mut answer,
        "Рівненщини",
        false,
        crate::normalization::Casing::Root,
    );
    assert_eq!(answer.lemma, "Рівненщина");
    assert_eq!(answer.source, Source::ExternalLower);
    assert!(matches!(answer.lemma, Cow::Owned(_)));
}
