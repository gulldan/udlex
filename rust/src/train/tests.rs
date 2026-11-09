use crate::format::{SelectionSample, SpanMultiword};
use crate::normalization::Casing;
use crate::probe::{self, Options};
use crate::rule::CaseAction;

use super::building::{Shape, case_of};
use super::evaluation::round3;
use super::writing::pack;
use super::*;

fn pair(form: &str, lemma: &str) -> Pair {
    Pair {
        form: form.to_owned(),
        lemma: lemma.to_owned(),
        sentence_initial: false,
    }
}

/// Every one of these was verified against the Python implementation on
/// the published catalogue before that implementation was removed. They
/// are here because the comparison cannot be run again.
#[test]
fn ties_go_to_the_first_seen() {
    // Two lemmas for one form, one observation each: whichever was seen
    // first wins, which is `Counter.most_common(1)` and not "the smaller".
    let tables = build(&[pair("бор", "бета"), pair("бор", "альфа")], vec![]);
    assert_eq!(tables.exact.get("бор").map(String::as_str), Some("бета"));

    let swapped = build(&[pair("бор", "альфа"), pair("бор", "бета")], vec![]);
    assert_eq!(swapped.exact.get("бор").map(String::as_str), Some("альфа"));
}

#[test]
fn identity_variant_still_matches_lowercased_entity_keys() {
    let tables = build_with_entities(
        &[],
        vec![],
        vec![("OpenAI".to_owned(), "OpenAI".to_owned())],
    );
    let view = View::new(&tables, Thresholds::default(), Variant::Identity);
    let answer = probe::lemma_with_info(
        &view,
        "OPENAI",
        false,
        Options {
            use_entities_lower: true,
            ..Options::default().entity_lookup()
        },
    );
    assert_eq!(answer.lemma, "OpenAI");
    assert_eq!(answer.source, probe::Source::EntityLower);
}

#[test]
fn ambiguous_entity_surface_preserves_itself() {
    let tables = build_with_entities(
        &[],
        vec![],
        vec![
            ("London".to_owned(), "London".to_owned()),
            ("London".to_owned(), "Other entity".to_owned()),
            ("London".to_owned(), "London".to_owned()),
            ("london".to_owned(), "London".to_owned()),
        ],
    );
    assert_eq!(
        tables.entities.get("London").map(String::as_str),
        Some("London")
    );
    assert!(!tables.entities_lower.contains_key("london"));

    let view = View::new(&tables, Thresholds::default(), Variant::Identity);
    let answer = probe::lemma_with_info(
        &view,
        "LONDON",
        false,
        Options {
            use_entities_lower: true,
            ..Options::default().entity_lookup()
        },
    );
    assert_eq!(answer.lemma, "LONDON");
    assert_eq!(answer.source, probe::Source::Identity);
}

#[test]
fn conflicting_casefolded_entities_have_no_lower_fallback() {
    let tables = build_with_entities(
        &[],
        vec![],
        vec![
            ("London".to_owned(), "London".to_owned()),
            ("LONDON".to_owned(), "London postal district".to_owned()),
            ("london".to_owned(), "London".to_owned()),
        ],
    );
    assert!(!tables.entities_lower.contains_key("london"));
    let view = View::new(&tables, Thresholds::default(), Variant::Identity);
    let answer = probe::lemma_with_info(
        &view,
        "lOnDoN",
        false,
        Options {
            use_entities_lower: true,
            ..Options::default().entity_lookup()
        },
    );
    assert_eq!(answer.lemma, "lOnDoN");
    assert_eq!(answer.source, probe::Source::Identity);
}

#[test]
fn differently_cased_identity_entities_have_no_arbitrary_lower_canonical() {
    let tables = build_with_entities(
        &[],
        vec![],
        vec![
            ("OpenAI".to_owned(), "OpenAI".to_owned()),
            ("OPENAI".to_owned(), "OPENAI".to_owned()),
        ],
    );

    assert!(!tables.entities_lower.contains_key("openai"));
    let view = View::new(&tables, Thresholds::default(), Variant::Identity);
    let answer = probe::lemma_with_info(
        &view,
        "openai",
        false,
        Options {
            use_entities_lower: true,
            ..Options::default().entity_lookup()
        },
    );
    assert_eq!(answer.lemma, "openai");
    assert_eq!(answer.source, probe::Source::Identity);
}

#[test]
fn a_rule_needs_four_observations() {
    let three: Vec<Pair> = ["кота", "рота", "лота"]
        .iter()
        .map(|form| pair(form, &form[..form.len() - 2]))
        .collect();
    assert!(build(&three, vec![]).rules.is_empty());

    let mut four = three;
    four.push(pair("мота", "мот"));
    assert!(!build(&four, vec![]).rules.is_empty());
}

#[test]
fn confidence_rounds_half_to_even() {
    // What Python's `round(x, 3)` does, and the reason a rule's confidence
    // compares equal to a threshold on both sides of the port. The values
    // are copied from the reference, not reasoned out: `0.0615` rounds
    // *down* because the nearest double to it is below the exact decimal,
    // while `0.0625` is representable and ties to even.
    assert!((round3(0.062_5) - 0.062).abs() < f64::EPSILON);
    assert!((round3(0.061_5) - 0.061).abs() < f64::EPSILON);
    assert!((round3(0.124_5) - 0.124).abs() < f64::EPSILON);
    assert!((round3(1.0 / 3.0) - 0.333).abs() < f64::EPSILON);
}

#[test]
fn the_cut_counts_characters_not_bytes() {
    // A byte-wise cut would slice a Cyrillic word in half and panic.
    let shape = Shape::between("котами", "кот", Casing::Root);
    assert_eq!(shape.cut, 3);
    assert_eq!(shape.add, "");
}

#[test]
fn case_is_inferred_from_the_lemma() {
    assert_eq!(case_of("кот", Casing::Root), CaseAction::Lower);
    assert_eq!(case_of("Москва", Casing::Root), CaseAction::Title);
    assert_eq!(case_of("ООН", Casing::Root), CaseAction::Upper);
    // One uppercase character is a title, not an acronym — `len > 1`.
    assert_eq!(case_of("Я", Casing::Root), CaseAction::Title);
    // `º` (U+00BA) is a cased lowercase character with no uppercase
    // mapping, so the lemma is not uppercase even though it survives an
    // uppercasing round-trip unchanged. The uncased degree sign in
    // `20°API` takes no side, and there the acronym wins. Both verified
    // against the reference on pt_petrogold, the one catalogue language
    // that exercises the difference.
    assert_eq!(case_of("20ºC", Casing::Root), CaseAction::Keep);
    assert_eq!(case_of("ºAPI", Casing::Root), CaseAction::Keep);
    assert_eq!(case_of("20°API", Casing::Root), CaseAction::Upper);
}

#[test]
fn a_stop_marker_is_written_as_255() {
    // The gap that cost a bundle: `CaseAction::Stop as u8` is 255 only
    // because the enum says so. Left implicit it is 4, which every reader
    // still decodes as Stop — so the bundle works and the bytes are wrong.
    let packed = pack(0, CaseAction::Stop, 0).unwrap();
    assert_eq!((packed >> 48) & 0xFF, 255);
    assert_eq!((pack(0, CaseAction::Upper, 0).unwrap() >> 48) & 0xFF, 3);
}

#[test]
fn identity_answers_with_the_token() {
    let tables = build(&[pair("котами", "кот")], vec![]);
    let identity = View::new(&tables, Thresholds::default(), Variant::Identity);
    assert_eq!(identity.lemma("котами", false), "котами");
    let full = View::new(&tables, Thresholds::default(), Variant::Full);
    assert_eq!(full.lemma("котами", false), "кот");
}

#[test]
fn a_variant_hides_the_tables_it_drops() {
    let tables = build(
        &[pair("кот", "кот")],
        vec![("собаками".to_owned(), "собака".to_owned())],
    );
    let thresholds = Thresholds::default();
    assert_eq!(
        View::new(&tables, thresholds, Variant::Full).lemma("собаками", false),
        "собака"
    );
    assert_eq!(
        View::new(&tables, thresholds, Variant::Rules).lemma("собаками", false),
        "собаками"
    );
}

#[test]
fn a_rule_below_its_threshold_stops_rather_than_softens() {
    // Three of four observations agree, so confidence is 0.75. Above the
    // threshold it fires; below it the token comes back untouched instead
    // of falling through to a shorter, vaguer suffix.
    let training = vec![
        pair("ата", "ат"),
        pair("бта", "бт"),
        pair("вта", "вт"),
        pair("гта", "гтая"),
    ];
    let tables = build(&training, vec![]);
    let permissive = Thresholds {
        general: 0.7,
        proper_noun: 0.7,
    };
    let strict = Thresholds {
        general: 0.8,
        proper_noun: 0.8,
    };
    assert_eq!(
        View::new(&tables, permissive, Variant::Full).lemma("дта", false),
        "дт"
    );
    assert_eq!(
        View::new(&tables, strict, Variant::Full).lemma("дта", false),
        "дта"
    );
}

#[test]
fn external_never_overwrites_the_corpus() {
    let tables = build(
        &[pair("кот", "кот")],
        vec![("кот".to_owned(), "нечто".to_owned())],
    );
    assert!(!tables.external.contains_key("кот"));
}

#[test]
fn one_ambiguous_external_source_keeps_the_documented_frequency_vote() {
    let tables = build(
        &[],
        vec![
            ("axes".to_owned(), "axis".to_owned()),
            ("axes".to_owned(), "axe".to_owned()),
            ("axial".to_owned(), "axis".to_owned()),
        ],
    );
    assert_eq!(
        tables.external.get("axes").map(String::as_str),
        Some("axis")
    );
}

#[test]
fn a_tie_keeps_the_simpler_variant() {
    // Nothing the tables know appears in the gold data, so all five
    // configurations score the same — and the first, simplest one wins.
    let tables = build(&[pair("кот", "кот")], vec![]);
    let gold = [Graded {
        form: "щука".to_owned(),
        lemma: Some("щука".to_owned()),
        sentence_initial: false,
    }];
    let (variant, _) = choose(&tables, Thresholds::default(), &gold, SelectionSample::Dev)
        .expect("one gold token is enough to decide on");
    assert_eq!(variant, Variant::Identity);
}

#[test]
fn nothing_scorable_is_not_a_verdict() {
    let tables = build(&[pair("кот", "кот")], vec![]);
    let unannotated = [Graded {
        form: "щука".to_owned(),
        lemma: None,
        sentence_initial: false,
    }];
    assert!(
        choose(
            &tables,
            Thresholds::default(),
            &unannotated,
            SelectionSample::Dev
        )
        .is_none()
    );
}

#[test]
fn repeated_tokens_do_not_fake_rule_support() {
    let repeated = vec![pair("went", "go"); 100];
    assert!(build(&repeated, vec![]).rules.is_empty());
}

#[test]
fn arbitrary_mixed_case_is_exact_only() {
    let training = [
        pair("iPhones", "iPhone"),
        pair("xPhones", "xPhone"),
        pair("yPhones", "yPhone"),
        pair("zPhones", "zPhone"),
    ];
    let tables = build(&training, vec![]);
    let view = View::new(&tables, Thresholds::default(), Variant::Full);
    assert_eq!(view.lemma("iPhones", false), "iPhone");
    assert_eq!(view.lemma("jPhones", false), "jPhones");
}

#[test]
fn entity_lookup_never_falls_into_common_vocabulary_or_rules() {
    let tables = build_with_entities(
        &[pair("cats", "cat")],
        vec![("Stone".to_owned(), "ston".to_owned())],
        vec![("Stone".to_owned(), "Stone".to_owned())],
    );
    let view = View::new(&tables, Thresholds::default(), Variant::Full);
    let normal = probe::lemma_with_info(&view, "Stone", false, Options::default());
    assert_eq!(normal.lemma, "ston");
    assert_eq!(normal.source, probe::Source::External);

    let entity = probe::lemma_with_info(&view, "Stone", false, Options::default().entity_lookup());
    assert_eq!(entity.lemma, "Stone");
    assert_eq!(entity.source, probe::Source::Entity);
    let unknown =
        probe::lemma_with_info(&view, "Stones", false, Options::default().entity_lookup());
    assert_eq!(unknown.lemma, "Stones");
    assert_eq!(unknown.source, probe::Source::Identity);
}

#[test]
fn protected_tokens_are_unconditionally_identical() {
    let tables = build(&[pair("cats", "cat")], vec![]);
    let view = View::new(&tables, Thresholds::default(), Variant::Full);
    let answer = probe::lemma_with_info(
        &view,
        "cats",
        false,
        Options {
            protected: true,
            ..Options::default()
        },
    );
    assert_eq!(answer.lemma, "cats");
    assert_eq!(answer.source, probe::Source::Protected);
}

#[test]
fn span_probe_answers_with_the_whole_span_before_anything_else() {
    let tables = build_with_entities(
        &[pair("Yorks", "York")],
        vec![],
        vec![("New Yorku".to_owned(), "New York".to_owned())],
    );
    let view = View::new(&tables, Thresholds::default(), Variant::Full);

    let hit = probe::span_lemma_with_info(
        &view,
        "New Yorku",
        &["New", "Yorku"],
        None,
        false,
        Options::default(),
    );
    assert_eq!(hit.lemma, "New York");
    assert_eq!(hit.source, probe::Source::Entity);

    let lower = probe::span_lemma_with_info(
        &view,
        "NEW YORKU",
        &["NEW", "YORKU"],
        None,
        false,
        Options {
            use_entities_lower: true,
            ..Options::default()
        },
    );
    assert_eq!(lower.lemma, "New York");
    assert_eq!(lower.source, probe::Source::EntityLower);
}

#[test]
fn span_answers_inherit_the_surface_casing_when_spelled_lowercase() {
    // bg_btb lowercases every lemma by corpus convention; scored against
    // case-preserving gold that convention alone destroys the answer, so an
    // all-lowercase answer inherits the surface casing at every stage.
    let tables = build_with_entities(
        &[pair("Пловдив", "пловдив")],
        vec![],
        vec![
            ("София".to_owned(), "софия".to_owned()),
            ("Стара Загора".to_owned(), "стара загора".to_owned()),
        ],
    );
    let view = View::new(&tables, Thresholds::default(), Variant::Full);

    let entity = probe::span_lemma_with_info(&view, "София", &[], None, false, Options::default());
    assert_eq!(entity.lemma, "София");
    assert_eq!(entity.source, probe::Source::Entity);

    let multiword = probe::span_lemma_with_info(
        &view,
        "Стара Загора",
        &["Стара", "Загора"],
        None,
        false,
        Options::default(),
    );
    assert_eq!(multiword.lemma, "Стара Загора");
    assert_eq!(multiword.source, probe::Source::Entity);

    let chain = probe::span_lemma_with_info(&view, "Пловдив", &[], None, false, Options::default());
    assert_eq!(chain.lemma, "Пловдив");
    assert_eq!(chain.source, probe::Source::Exact);

    // An ALL-CAPS surface lifts the whole answer through the lower probe, and
    // an answer that carries its own capital — or a lowercase surface —
    // passes untouched.
    let shouted = probe::span_lemma_with_info(
        &view,
        "СОФИЯ",
        &[],
        None,
        false,
        Options {
            use_entities_lower: true,
            ..Options::default()
        },
    );
    assert_eq!(shouted.lemma, "СОФИЯ");
    assert_eq!(shouted.source, probe::Source::EntityLower);
    let cased = build_with_entities(
        &[],
        vec![],
        vec![("Härnösand".to_owned(), "HärNösand".to_owned())],
    );
    let cased_view = View::new(&cased, Thresholds::default(), Variant::Identity);
    let kept = probe::span_lemma_with_info(
        &cased_view,
        "Härnösand",
        &[],
        None,
        false,
        Options::default(),
    );
    assert_eq!(kept.lemma, "HärNösand");
    let lowercase_surface =
        probe::span_lemma_with_info(&view, "софия", &[], None, false, Options::default());
    assert_eq!(lowercase_surface.lemma, "софия");
}

#[test]
fn unknown_multiword_span_survives_byte_for_byte() {
    let tables = build_with_entities(
        &[pair("Rady", "Rada"), pair("Europy", "Europa")],
        vec![],
        vec![("Praha".to_owned(), "Praha".to_owned())],
    );
    let view = View::new(&tables, Thresholds::default(), Variant::Full);

    // The common tables know every word of the span; the span probe still
    // refuses to lemmatise them independently — the measured optimum for an
    // uncovered multiword mention is to leave it alone, punctuation included.
    let answer = probe::span_lemma_with_info(
        &view,
        "«Rady — Europy»",
        &["«Rady", "—", "Europy»"],
        None,
        false,
        Options::default(),
    );
    assert_eq!(answer.lemma, "«Rady — Europy»");
    assert_eq!(answer.source, probe::Source::Identity);
}

#[test]
fn single_word_span_falls_through_to_the_common_chain() {
    let tables = build_with_entities(
        &[pair("cats", "cat")],
        vec![("Stone".to_owned(), "ston".to_owned())],
        vec![("Praha".to_owned(), "Praha".to_owned())],
    );
    let view = View::new(&tables, Thresholds::default(), Variant::Full);

    let common = probe::span_lemma_with_info(&view, "cats", &[], None, false, Options::default());
    assert_eq!(common.lemma, "cat");
    assert_eq!(common.source, probe::Source::Exact);

    // The external table spelled the answer lowercase; the surface is a
    // capitalised name, so the answer inherits its case.
    let external =
        probe::span_lemma_with_info(&view, "Stone", &[], None, false, Options::default());
    assert_eq!(external.lemma, "Ston");
    assert_eq!(external.source, probe::Source::External);

    // The entity table still answers first: the span profile is the entity
    // path, and the fall-through exists for the miss, not instead of it.
    let entity = probe::span_lemma_with_info(&view, "Praha", &[], None, false, Options::default());
    assert_eq!(entity.lemma, "Praha");
    assert_eq!(entity.source, probe::Source::Entity);
}

#[test]
fn span_probe_honours_protection_before_any_lookup() {
    let tables = build_with_entities(
        &[],
        vec![],
        vec![("New Yorku".to_owned(), "New York".to_owned())],
    );
    let view = View::new(&tables, Thresholds::default(), Variant::Identity);
    let answer = probe::span_lemma_with_info(
        &view,
        "New Yorku",
        &["New", "Yorku"],
        None,
        false,
        Options {
            protected: true,
            ..Options::default()
        },
    );
    assert_eq!(answer.lemma, "New Yorku");
    assert_eq!(answer.source, probe::Source::Protected);
}

/// A view with non-default span conventions stamped on, as a compiled
/// lexicon carries them in its metadata after acceptance.
struct Stamped<'a> {
    inner: &'a View<'a>,
    conventions: probe::SpanConventions,
}

impl probe::Tables for Stamped<'_> {
    fn entities(&self, key: &str) -> Option<std::borrow::Cow<'_, str>> {
        self.inner.entities(key)
    }
    fn entities_lower(&self, key: &str) -> Option<std::borrow::Cow<'_, str>> {
        self.inner.entities_lower(key)
    }
    fn exact(&self, key: &str) -> Option<std::borrow::Cow<'_, str>> {
        self.inner.exact(key)
    }
    fn lower(&self, key: &str) -> Option<std::borrow::Cow<'_, str>> {
        self.inner.lower(key)
    }
    fn external(&self, key: &str) -> Option<std::borrow::Cow<'_, str>> {
        self.inner.external(key)
    }
    fn rule(
        &self,
        bucket: u8,
        suffix: &str,
    ) -> Option<(crate::rule::Rule, std::borrow::Cow<'_, str>)> {
        self.inner.rule(bucket, suffix)
    }
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    fn common_is_empty(&self) -> bool {
        self.inner.common_is_empty()
    }
    fn max_suffix(&self) -> usize {
        self.inner.max_suffix()
    }
    fn casing(&self) -> Casing {
        self.inner.casing()
    }
    fn span_conventions(&self) -> probe::SpanConventions {
        self.conventions
    }
}

#[test]
fn verbatim_case_convention_returns_the_table_value_untouched() {
    // tr_penn stores the corpus's own casing decision in the table —
    // `CNN'de → cnn` next to `DPC → DPC` — and the surface-case transfer
    // alone destroys the lowercased answers. The verbatim convention hands
    // the stored value back; the default keeps transferring.
    let tables = build_with_entities(&[], vec![], vec![("CNN'de".to_owned(), "cnn".to_owned())]);
    let view = View::new(&tables, Thresholds::default(), Variant::Identity);

    let transferred =
        probe::span_lemma_with_info(&view, "CNN'de", &[], None, false, Options::default());
    assert_eq!(transferred.lemma, "Cnn");

    let stamped = Stamped {
        inner: &view,
        conventions: probe::SpanConventions {
            case_transfer: false,
            multiword: SpanMultiword::Identity,
        },
    };
    let verbatim =
        probe::span_lemma_with_info(&stamped, "CNN'de", &[], None, false, Options::default());
    assert_eq!(verbatim.lemma, "cnn");
    assert_eq!(verbatim.source, probe::Source::Entity);
}

#[test]
fn compose_convention_assembles_a_missed_multiword_span_per_token() {
    // ko_kaist keys its clitic rewrites token by token — the whole-span key
    // does not exist. Composition sends each token through its own span
    // treatment, entity probes included, and joins with the original gaps.
    let tables = build_with_entities(
        &[pair("Rady", "Rada")],
        vec![],
        vec![("이집트의".to_owned(), "이집트+의".to_owned())],
    );
    let view = View::new(&tables, Thresholds::default(), Variant::Full);
    let stamped = Stamped {
        inner: &view,
        conventions: probe::SpanConventions {
            case_transfer: true,
            multiword: SpanMultiword::Compose,
        },
    };

    // First token answers from the entity table, second from the common
    // chain, third from nowhere; the unknown word survives byte-for-byte.
    let composed = probe::span_lemma_with_info(
        &stamped,
        "이집트의 Rady 나일강",
        &["이집트의", "Rady", "나일강"],
        None,
        false,
        Options::default(),
    );
    assert_eq!(composed.lemma, "이집트+의 Rada 나일강");
    assert_eq!(composed.source, probe::Source::Composed);
    assert!(!composed.source.is_known());

    // The caller's separators survive composition exactly as they survive
    // identity.
    let hyphenated = probe::span_lemma_with_info(
        &stamped,
        "이집트의-나일강",
        &["이집트의", "나일강"],
        Some(&["-"]),
        false,
        Options::default(),
    );
    assert_eq!(hyphenated.lemma, "이집트+의-나일강");

    // A whole-span entity key still answers first: composition exists for
    // the miss, not instead of the hit.
    let whole = build_with_entities(
        &[],
        vec![],
        vec![("New Yorku".to_owned(), "New York".to_owned())],
    );
    let whole_view = View::new(&whole, Thresholds::default(), Variant::Identity);
    let whole_stamped = Stamped {
        inner: &whole_view,
        conventions: probe::SpanConventions {
            case_transfer: true,
            multiword: SpanMultiword::Compose,
        },
    };
    let hit = probe::span_lemma_with_info(
        &whole_stamped,
        "New Yorku",
        &["New", "Yorku"],
        None,
        false,
        Options::default(),
    );
    assert_eq!(hit.lemma, "New York");
    assert_eq!(hit.source, probe::Source::Entity);
}

#[test]
fn head_convention_answers_for_the_first_token_and_leaves_the_tail_alone() {
    // A declined Slavic mention whose modifier tail never moves: the head is
    // the only word with a lemma worth asking for, and composing the tail
    // rewrites a word the corpus happens to know in another sense.
    let tables = build_with_entities(
        &[
            pair("Ленинградской", "ленинградский"),
            pair("области", "область"),
        ],
        vec![],
        vec![],
    );
    let view = View::new(&tables, Thresholds::default(), Variant::Full);
    let composed = Stamped {
        inner: &view,
        conventions: probe::SpanConventions {
            case_transfer: true,
            multiword: SpanMultiword::Compose,
        },
    };
    let head = Stamped {
        inner: &view,
        conventions: probe::SpanConventions {
            case_transfer: true,
            multiword: SpanMultiword::Head,
        },
    };
    let tokens = ["Ленинградской", "области"];
    let whole = probe::span_lemma_with_info(
        &composed,
        "Ленинградской области",
        &tokens,
        None,
        false,
        Options::default(),
    );
    assert_eq!(whole.lemma, "Ленинградский область");
    let only_head = probe::span_lemma_with_info(
        &head,
        "Ленинградской области",
        &tokens,
        None,
        false,
        Options::default(),
    );
    assert_eq!(only_head.lemma, "Ленинградский области");
    assert_eq!(only_head.source, probe::Source::Composed);
    assert!(!only_head.source.is_known());

    // The caller's separators are the span's own text under every convention.
    let hyphenated = probe::span_lemma_with_info(
        &head,
        "Ленинградской-области",
        &tokens,
        Some(&["-"]),
        false,
        Options::default(),
    );
    assert_eq!(hyphenated.lemma, "Ленинградский-области");

    // A one-token span is the token chain's business, not the convention's.
    let single = probe::span_lemma_with_info(
        &head,
        "области",
        &["области"],
        None,
        false,
        Options::default(),
    );
    assert_eq!(single.lemma, "область");
}

#[test]
fn a_mention_given_as_one_string_reaches_the_same_probe_as_its_tokens() {
    // What a NER that emits ready mentions hands over: «Ленинградские
    // области» as one string. Splitting it is the whole difference — the
    // probe, the conventions and the fall-through must be the same objects,
    // not a second implementation of the order.
    let mut pairs: Vec<Pair> = ["кота", "рота", "лота", "мота"]
        .iter()
        .map(|form| pair(form, &form[..form.len() - 2]))
        .collect();
    pairs.push(pair("Ленинградские", "ленинградский"));
    let tables = build_with_entities(
        &pairs,
        vec![],
        vec![(
            "Ленинградские области".to_owned(),
            "Ленинградская область".to_owned(),
        )],
    );
    let view = View::new(&tables, Thresholds::default(), Variant::Full);
    let answer = |surface: &'static str| {
        let (tokens, gaps) = crate::span::split_surface(surface).unwrap();
        let separators = (!gaps.is_empty()).then_some(gaps.as_slice());
        probe::span_lemma_with_info(
            &view,
            surface,
            &tokens,
            separators,
            false,
            Options::default(),
        )
    };

    let whole = answer("Ленинградские области");
    assert_eq!(whole.lemma, "Ленинградская область");
    assert_eq!(whole.source, probe::Source::Entity);
    let by_hand = probe::span_lemma_with_info(
        &view,
        "Ленинградские области",
        &["Ленинградские", "области"],
        Some(&[" "]),
        false,
        Options::default(),
    );
    assert_eq!((whole.lemma, whole.source), (by_hand.lemma, by_hand.source));

    // The gap is part of the mention. Collapsing the second space would probe
    // a key the caller never sent and score the answer against a surface the
    // text never held.
    let widened = answer("Ленинградские  области");
    assert_eq!(widened.lemma, "Ленинградские  области");
    assert_eq!(widened.source, probe::Source::Identity);

    // One word is the ordinary token chain, not a copy of the surface: the
    // table answers here and the suffix rule answers below, both through the
    // span case convention.
    let known = answer("Ленинградские");
    assert_eq!(known.lemma, "Ленинградский");
    assert_eq!(known.source, probe::Source::Exact);
    let guessed = answer("жбота");
    assert_eq!(guessed.lemma, "жбот");
    assert_eq!(guessed.source, probe::Source::Rule);
}

#[test]
fn canonical_equivalents_and_turkic_case_share_lookup_keys() {
    let nfc_tables = build(&[pair("cafe\u{301}s", "cafe\u{301}")], vec![]);
    let nfc_view = View::new(&nfc_tables, Thresholds::default(), Variant::Full);
    assert_eq!(nfc_view.lemma("cafés", false), "café");

    let turkish = build_for_language(&[pair("istanbul", "istanbul")], vec![], "tur");
    let turkish_view = View::new(&turkish, Thresholds::default(), Variant::Full);
    assert_eq!(turkish_view.lemma("İSTANBUL", false), "istanbul");
}

#[test]
fn unrepresentable_rules_and_nonfinite_thresholds_are_rejected() {
    assert!(pack(256, CaseAction::Lower, 0).is_err());
    assert!(pack(0, CaseAction::Lower, 1_u64 << 48).is_err());
    assert!(
        !Thresholds {
            general: f64::NAN,
            proper_noun: 0.0,
        }
        .is_valid()
    );
    let directory = tempfile::tempdir().unwrap();
    let tables = build(&[pair("cats", "cat")], vec![]);
    assert!(
        write_sources(
            &tables,
            Variant::Full,
            Thresholds {
                general: f64::NAN,
                proper_noun: 0.0,
            },
            directory.path(),
        )
        .is_err()
    );
}

#[test]
fn source_delimiters_are_rejected_before_writing() {
    let directory = tempfile::tempdir().unwrap();
    let tables = build(&[pair("bad\nkey", "lemma")], vec![]);
    assert!(
        write_sources(
            &tables,
            Variant::Full,
            Thresholds::default(),
            directory.path(),
        )
        .is_err()
    );
}

#[test]
fn empty_entity_keys_are_rejected_before_compilation() {
    let directory = tempfile::tempdir().unwrap();
    let mut tables = Tables::default();
    tables.entities.insert(String::new(), "lemma".to_owned());
    assert!(
        write_sources(
            &tables,
            Variant::Identity,
            Thresholds::default(),
            directory.path(),
        )
        .is_err()
    );
}

#[test]
fn refit_from_resolved_auxiliary_tables_equals_a_clean_full_rebuild() {
    let calibration_training = vec![pair("cats", "cat")];
    let mut full_training = calibration_training.clone();
    full_training.extend([
        pair("axes", "axis"),
        pair("İstanbul", "İstanbul"),
        pair("talked", "talk"),
        pair("walked", "walk"),
        pair("looked", "look"),
        pair("cooked", "cook"),
    ]);
    let external = vec![
        ("axes".to_owned(), "axis".to_owned()),
        ("axes".to_owned(), "axe".to_owned()),
        ("İSTANBUL".to_owned(), "external-city".to_owned()),
        ("dogs".to_owned(), "dog".to_owned()),
        ("doggies".to_owned(), "dog".to_owned()),
    ];
    let entities = vec![
        ("OpenAI".to_owned(), "OpenAI".to_owned()),
        ("OPENAI".to_owned(), "OpenAI".to_owned()),
        ("York".to_owned(), "York".to_owned()),
    ];

    let calibration = build_for_language_with_entities(
        &calibration_training,
        external.clone(),
        entities.clone(),
        "tur",
    );
    let reused = refit_for_language(&full_training, calibration);
    let clean = build_for_language_with_entities(&full_training, external, entities, "tur");

    assert_eq!(reused.casing, clean.casing);
    assert_eq!(reused.exact, clean.exact);
    assert_eq!(reused.lower, clean.lower);
    assert_eq!(reused.external, clean.external);
    assert_eq!(reused.entities, clean.entities);
    assert_eq!(reused.entities_lower, clean.entities_lower);
    assert_eq!(reused.rules, clean.rules);
    assert!(!reused.external.contains_key("axes"));
    assert!(!reused.external.contains_key("İSTANBUL"));

    let temporary = tempfile::tempdir().unwrap();
    let reused_path = temporary.path().join("reused");
    let clean_path = temporary.path().join("clean");
    assert_eq!(
        write_sources(&reused, Variant::Full, Thresholds::default(), &reused_path,).unwrap(),
        write_sources(&clean, Variant::Full, Thresholds::default(), &clean_path,).unwrap(),
    );
    for name in [
        "exact.tsv",
        "lower.tsv",
        "external.tsv",
        "entities.tsv",
        "rules.tsv",
        "strings.txt",
    ] {
        assert_eq!(
            std::fs::read(reused_path.join(name)).unwrap(),
            std::fs::read(clean_path.join(name)).unwrap(),
            "{name} differs",
        );
    }
}

#[test]
fn entity_refit_preserves_lower_conflicts_and_fails_closed_on_new_evidence() {
    let calibration = build_for_language_with_entities(
        &[pair("cats", "cat")],
        vec![],
        vec![
            ("London".to_owned(), "London".to_owned()),
            ("LONDON".to_owned(), "London postal district".to_owned()),
            ("Anchors".to_owned(), "Anchors".to_owned()),
        ],
        "eng",
    );
    assert!(!calibration.entities_lower.contains_key("london"));

    let refitted = refit_for_language_with_entities(
        &[pair("cats", "cat"), pair("dogs", "dog")],
        calibration,
        vec![
            ("Parises".to_owned(), "Paris".to_owned()),
            ("Anchors".to_owned(), "Anchor".to_owned()),
        ],
    );

    assert!(!refitted.entities_lower.contains_key("london"));
    assert_eq!(
        refitted.entities.get("Parises").map(String::as_str),
        Some("Paris")
    );
    assert_eq!(
        refitted.entities.get("Anchors").map(String::as_str),
        Some("Anchors")
    );
}

#[test]
fn suffix_rules_never_rewrite_a_token_without_letters() {
    let pairs: Vec<Pair> = ["кот", "пёс", "дом", "лес"]
        .iter()
        .map(|stem| pair(&format!("{stem}."), stem))
        .collect();
    let tables = build(&pairs, vec![]);
    let view = View::new(&tables, Thresholds::default(), Variant::Full);
    // The learned `.` rule still serves words…
    assert_eq!(view.lemma("сад.", false), "сад");
    // …but punctuation, digits and symbols keep their own spelling.
    for token in [".", "1987.", "§."] {
        assert_eq!(view.lemma(token, false), token);
    }
}
