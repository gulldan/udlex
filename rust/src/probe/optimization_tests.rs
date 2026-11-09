use std::borrow::Cow;
use std::cell::Cell;

use super::{
    Casing, NoUser, Options, Rule, Source, SpanConventions, Tables, UserTables,
    inherit_surface_case, lemma_with_info, lemma_with_layer, token_span_answer,
};

struct CountingTables {
    exact_calls: Cell<usize>,
    entity_calls: Cell<usize>,
    entity_lower_calls: Cell<usize>,
    exact_hit: Option<(&'static str, &'static str)>,
    entity_hit: Option<(&'static str, &'static str)>,
    entity_lower_hit: Option<(&'static str, &'static str)>,
    empty: bool,
    common_empty: bool,
}

impl CountingTables {
    const fn populated() -> Self {
        Self {
            exact_calls: Cell::new(0),
            entity_calls: Cell::new(0),
            entity_lower_calls: Cell::new(0),
            exact_hit: None,
            entity_hit: None,
            entity_lower_hit: None,
            empty: false,
            common_empty: false,
        }
    }

    fn lookup(hit: Option<(&'static str, &'static str)>, key: &str) -> Option<Cow<'static, str>> {
        hit.and_then(|(stored, lemma)| (stored == key).then_some(Cow::Borrowed(lemma)))
    }
}

impl Tables for CountingTables {
    fn entities(&self, key: &str) -> Option<Cow<'_, str>> {
        self.entity_calls
            .set(self.entity_calls.get().saturating_add(1));
        Self::lookup(self.entity_hit, key)
    }

    fn entities_lower(&self, key: &str) -> Option<Cow<'_, str>> {
        self.entity_lower_calls
            .set(self.entity_lower_calls.get().saturating_add(1));
        Self::lookup(self.entity_lower_hit, key)
    }

    fn exact(&self, key: &str) -> Option<Cow<'_, str>> {
        self.exact_calls
            .set(self.exact_calls.get().saturating_add(1));
        Self::lookup(self.exact_hit, key)
    }

    fn lower(&self, _key: &str) -> Option<Cow<'_, str>> {
        None
    }

    fn external(&self, _key: &str) -> Option<Cow<'_, str>> {
        None
    }

    fn rule(&self, _bucket: u8, _suffix: &str) -> Option<(Rule, Cow<'_, str>)> {
        None
    }

    fn is_empty(&self) -> bool {
        self.empty
    }

    fn common_is_empty(&self) -> bool {
        self.common_empty
    }

    fn max_suffix(&self) -> usize {
        0
    }
}

#[test]
fn borrowed_nfc_does_not_repeat_a_failed_raw_exact_probe() {
    let tables = CountingTables::populated();
    let answer = lemma_with_info(&tables, "unknown", false, Options::default());
    assert_eq!(answer.source, Source::Identity);
    assert_eq!(tables.exact_calls.get(), 1);
}

#[test]
fn changed_nfc_still_probes_the_normalised_exact_key() {
    let mut tables = CountingTables::populated();
    tables.exact_hit = Some(("café", "coffee"));
    let answer = lemma_with_info(&tables, "cafe\u{301}", false, Options::default());
    assert_eq!(answer.lemma, "coffee");
    assert_eq!(answer.source, Source::Exact);
    assert_eq!(tables.exact_calls.get(), 2);
}

#[test]
fn token_span_reuses_its_normalised_key_in_the_common_chain() {
    let mut tables = CountingTables::populated();
    tables.exact_hit = Some(("café", "coffee"));
    let answer = token_span_answer(
        &tables,
        &NoUser,
        "cafe\u{301}",
        false,
        Options::default(),
        SpanConventions::default(),
        Casing::Root,
    );
    assert_eq!(answer.source, Source::Exact);
    assert_eq!(tables.entity_calls.get(), 1);
    assert_eq!(tables.exact_calls.get(), 1);
}

#[test]
fn common_empty_shortcut_preserves_entity_lookup() {
    let mut tables = CountingTables::populated();
    tables.common_empty = true;
    tables.entity_hit = Some(("Name", "Canonical"));
    let answer = lemma_with_info(&tables, "Name", false, Options::default().entity_lookup());
    assert_eq!(answer.lemma, "Canonical");
    assert_eq!(answer.source, Source::Entity);
    assert_eq!(tables.entity_calls.get(), 1);
    assert_eq!(tables.exact_calls.get(), 0);
}

#[test]
fn source_classification_distinguishes_unchanged_lemmas_from_fallbacks() {
    for source in [
        Source::Entity,
        Source::EntityLower,
        Source::Exact,
        Source::Lower,
        Source::ExactLower,
        Source::External,
        Source::ExternalLower,
        Source::User,
        Source::UserLower,
        Source::Context,
    ] {
        assert!(source.is_known(), "{source:?}");
        assert!(source.is_lemmatized(), "{source:?}");
    }
    for source in [Source::Rule, Source::Composed, Source::Phrase] {
        assert!(!source.is_known(), "{source:?}");
        assert!(source.is_lemmatized(), "{source:?}");
    }
    for source in [Source::Identity, Source::Protected] {
        assert!(!source.is_known(), "{source:?}");
        assert!(!source.is_lemmatized(), "{source:?}");
    }
}

#[test]
fn source_abi_codes_and_names_are_exhaustive() {
    let names = [
        "entity",
        "entity_lower",
        "exact",
        "lower",
        "exact_lower",
        "external",
        "external_lower",
        "rule",
        "identity",
        "protected",
        "composed",
        "phrase",
        "user",
        "user_lower",
        "context",
    ];

    assert_eq!(Source::ALL.len(), names.len());
    for (code, (source, name)) in Source::ALL.into_iter().zip(names).enumerate() {
        assert_eq!(usize::from(source.code()), code);
        assert_eq!(Source::from_code(source.code()), Some(source));
        assert_eq!(source.name(), name);
    }
    assert_eq!(Source::from_code(15), None);
    assert_eq!(Source::from_code(u8::MAX), None);
}

#[test]
fn common_empty_shortcut_preserves_lowercased_entity_lookup() {
    let mut tables = CountingTables::populated();
    tables.common_empty = true;
    tables.entity_lower_hit = Some(("name", "Canonical"));
    let options = Options {
        use_entities_lower: true,
        ..Options::default()
    };
    let answer = lemma_with_info(&tables, "NAME", false, options);
    assert_eq!(answer.lemma, "Canonical");
    assert_eq!(answer.source, Source::EntityLower);
    assert_eq!(tables.entity_lower_calls.get(), 1);
    assert_eq!(tables.exact_calls.get(), 0);
}

#[test]
fn empty_tables_return_before_any_probe() {
    let mut tables = CountingTables::populated();
    tables.empty = true;
    tables.common_empty = true;
    let token = "cafe\u{301}";
    for (options, source) in [
        (Options::default(), Source::Identity),
        (Options::default().entity_lookup(), Source::Identity),
        (
            Options {
                protected: true,
                ..Options::default()
            },
            Source::Protected,
        ),
    ] {
        let answer = lemma_with_info(&tables, token, false, options);
        assert_eq!(answer.source, source);
        assert!(matches!(answer.lemma, Cow::Borrowed(found) if std::ptr::eq(found, token)));
    }
    assert_eq!(tables.entity_calls.get(), 0);
    assert_eq!(tables.entity_lower_calls.get(), 0);
    assert_eq!(tables.exact_calls.get(), 0);
}

#[test]
fn empty_base_keeps_a_populated_user_layer() {
    struct User {
        empty_calls: Cell<usize>,
    }

    impl UserTables for User {
        const ENABLED: bool = true;

        fn exact(&self, key: &str) -> Option<Cow<'_, str>> {
            (key == "word").then_some(Cow::Borrowed("override"))
        }

        fn lower(&self, _key: &str) -> Option<Cow<'_, str>> {
            None
        }

        fn is_empty(&self) -> bool {
            self.empty_calls
                .set(self.empty_calls.get().saturating_add(1));
            false
        }
    }

    let mut tables = CountingTables::populated();
    tables.empty = true;
    tables.common_empty = true;
    let user = User {
        empty_calls: Cell::new(0),
    };
    let answer = lemma_with_layer(&tables, &user, "word", false, Options::default());
    assert_eq!(answer.lemma, "override");
    assert_eq!(answer.source, Source::User);
    assert_eq!(user.empty_calls.get(), 1);
    assert_eq!(tables.exact_calls.get(), 0);
}

#[test]
fn surface_case_transfer_preserves_lemma_spacing_without_a_pattern_vector() {
    assert_eq!(
        inherit_surface_case("OLD Town", Cow::Borrowed("old  town"), Casing::Root),
        "OLD  Town"
    );
    assert_eq!(
        inherit_surface_case("New York", Cow::Borrowed("old"), Casing::Root),
        "Old"
    );
}
