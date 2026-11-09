use std::borrow::Cow;
use std::cmp::Reverse;

use indexmap::{IndexMap, IndexSet};

use crate::normalization::{Casing, into_nfc, nfc};
use crate::rule::{CaseAction, CaseKey};

use super::{MAX_SUFFIX_LENGTH, Pair, Tables, Trained};

/// Observations a suffix bucket needs before its majority rule is trusted.
const MIN_RULE_SUPPORT: u64 = 4;

/// The identity of a rule, without its confidence — what the vote is over.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct Shape {
    pub(super) cut: usize,
    pub(super) add: String,
    pub(super) case: CaseAction,
}

impl Shape {
    /// Derives the rule that turns `form` into `lemma`.
    ///
    /// Expressed over the lowercased strings so that it generalises across
    /// capitalisation, and over characters rather than bytes so that the cut
    /// means the same thing here as where it is applied.
    pub(super) fn between(form: &str, lemma: &str, casing: Casing) -> Self {
        let lowered_form = casing.lowercase(form);
        let lowered_lemma = casing.lowercase(lemma);
        let shared = lowered_form
            .chars()
            .zip(lowered_lemma.chars())
            .take_while(|(a, b)| a == b)
            .count();
        Self {
            cut: lowered_form.chars().count().saturating_sub(shared),
            add: lowered_lemma.chars().skip(shared).collect(),
            case: case_of(lemma, casing),
        }
    }
}

/// Whether every cased character is uppercase and at least one exists.
///
/// This mirrors Python's `str.isupper`, which the reference trainer used, and
/// differs from an `s == s.to_uppercase()` round-trip on characters that are
/// cased but have no uppercase mapping: `º` (U+00BA) counts as lowercase, so
/// `20ºC` is not uppercase here, yet it survives the round-trip unchanged.
/// Titlecase letters such as `ǅ` are cased without being uppercase too; they
/// are the ones the lowering probe in the last arm catches.
fn is_upper(text: &str) -> bool {
    let mut has_cased = false;
    for character in text.chars() {
        if character.is_uppercase() {
            has_cased = true;
        } else if character.is_lowercase()
            || character.to_lowercase().ne(std::iter::once(character))
        {
            return false;
        }
    }
    has_cased
}

/// Infers the action that reproduces `lemma` from its lowercase form.
pub(super) fn case_of(lemma: &str, casing: Casing) -> CaseAction {
    if lemma == casing.lowercase(lemma) {
        return CaseAction::Lower;
    }
    if lemma.chars().count() > 1 && is_upper(lemma) {
        return CaseAction::Upper;
    }
    let mut chars = lemma.chars();
    let first_upper = chars.next().is_some_and(char::is_uppercase);
    let rest: String = chars.collect();
    if first_upper && rest == casing.lowercase(&rest) {
        return CaseAction::Title;
    }
    CaseAction::Keep
}

/// Returns the entry with the highest count, ties going to the first inserted.
///
/// This is `Counter.most_common(1)[0]`, and the tie rule is the whole reason
/// the map has to preserve insertion order.
pub fn into_most_common<K>(counts: IndexMap<K, u64>) -> Option<(K, u64)>
where
    K: std::hash::Hash + Eq,
{
    // `min_by_key` keeps the first tied entry, unlike `max_by_key`.
    counts.into_iter().min_by_key(|(_, count)| Reverse(*count))
}

/// Adds format-bounded suffix observations, shortest first.
fn observe_suffixes(
    buckets: &mut IndexMap<String, IndexMap<Shape, u64>>,
    lowered: &str,
    bucket: char,
    shape: &Shape,
) {
    for (start, _) in lowered.char_indices().rev().take(MAX_SUFFIX_LENGTH) {
        let Some(suffix) = lowered.get(start..) else {
            continue;
        };
        let mut key = String::with_capacity(suffix.len().saturating_add(1));
        key.push(bucket);
        key.push_str(suffix);
        let bucket_count = buckets
            .entry(key)
            .or_default()
            .entry(shape.clone())
            .or_insert(0);
        // Rule support counts distinct observed `(form, lemma, position)`
        // pairs. Repeating one corpus token strengthens its exact/lower vote,
        // but must not make a one-example suffix pattern look productive.
        *bucket_count = bucket_count.saturating_add(1);
    }
}

/// A source-local ambiguity reducer; resolving each source before the merge
/// keeps a later source from flipping an earlier source's answer.
#[derive(Default)]
pub struct ExternalResolver {
    counts: IndexMap<String, IndexMap<usize, u64>>,
    lemmas: IndexSet<String>,
    lemma_frequency: Vec<u64>,
}

impl ExternalResolver {
    /// Add one analysis while preserving first-seen tie order.
    pub fn add(&mut self, form: String, lemma: String) {
        let lemma_id = if let Some(found) = self.lemmas.get_index_of(&lemma) {
            found
        } else {
            let id = self.lemmas.len();
            self.lemmas.insert(lemma);
            self.lemma_frequency.push(0);
            id
        };
        let analyses = self
            .counts
            .entry(form)
            .or_default()
            .entry(lemma_id)
            .or_insert(0);
        *analyses = analyses.saturating_add(1);
        if let Some(frequency) = self.lemma_frequency.get_mut(lemma_id) {
            *frequency = frequency.saturating_add(1);
        }
    }

    /// Resolve every form independently of all other external sources.
    pub fn finish(self) -> IndexMap<String, String> {
        let mut table = IndexMap::new();
        for (form, candidates) in self.counts {
            // `max(candidates, key=(count, lemma_frequency, -len(lemma)))`, first
            // maximal element winning — hence `>` and not `>=`.
            let mut best: Option<(usize, (u64, u64, isize))> = None;
            for (&lemma_id, &count) in &candidates {
                let Some(lemma) = self.lemmas.get_index(lemma_id) else {
                    continue;
                };
                let rank = (
                    count,
                    self.lemma_frequency.get(lemma_id).copied().unwrap_or(0),
                    // Negated so the shorter lemma wins a frequency tie.
                    isize::try_from(lemma.chars().count())
                        .unwrap_or(isize::MAX)
                        .saturating_neg(),
                );
                match best {
                    Some((_, high)) if rank <= high => {}
                    _ => best = Some((lemma_id, rank)),
                }
            }
            if let Some((lemma_id, _)) = best
                && let Some(lemma) = self.lemmas.get_index(lemma_id)
            {
                table.insert(form, lemma.clone());
            }
        }
        table
    }
}

/// Reduces external paradigms to a form-to-lemma table.
///
/// External sources carry no corpus frequencies, so an ambiguous form is
/// resolved in favour of the lemma with the larger paradigm, which correlates
/// with how common the word is. Forms the corpus already answers are dropped.
fn external_table(
    pairs: impl Iterator<Item = (String, String)>,
    exact: &IndexMap<String, String>,
    lower: &IndexMap<String, String>,
    casing: Casing,
) -> IndexMap<String, String> {
    let mut resolver = ExternalResolver::default();
    for (form, lemma) in pairs {
        resolver.add(form, lemma);
    }
    resolver
        .finish()
        .into_iter()
        .filter(|(form, _)| !corpus_answers(form, exact, lower, casing))
        .collect()
}

/// Whether either corpus lookup namespace already owns an external form.
fn corpus_answers(
    form: &str,
    exact: &IndexMap<String, String>,
    lower: &IndexMap<String, String>,
    casing: Casing,
) -> bool {
    if exact.contains_key(form) || lower.contains_key(form) {
        return true;
    }
    let lowered = casing.lowercase(form);
    exact.contains_key(lowered.as_ref()) || lower.contains_key(lowered.as_ref())
}

#[derive(Clone, PartialEq, Eq)]
enum EntityChoice {
    Lemma(String),
    Conflict,
}

impl EntityChoice {
    /// Conflicting evidence stays a conflict even if a later row agrees again.
    fn merge(&mut self, other: &Self) {
        if self != other {
            *self = Self::Conflict;
        }
    }
}

/// Builds disjoint exact and locale-lowercased indexes over entity forms.
///
/// Entity data must fail closed: frequency is not a disambiguator for two
/// different named entities. An exact surface with incompatible lemmas keeps
/// itself, while a case-folded key with incompatible lemmas is omitted because
/// one fixed spelling cannot preserve every caller's original surface.
fn entity_tables(
    entities: impl IntoIterator<Item = (String, String)>,
    casing: Casing,
) -> (IndexMap<String, String>, IndexMap<String, String>) {
    let mut exact_choices = IndexMap::new();
    for (form, lemma) in entities {
        let choice = EntityChoice::Lemma(lemma);
        exact_choices
            .entry(form)
            .and_modify(|existing: &mut EntityChoice| existing.merge(&choice))
            .or_insert(choice);
    }
    let mut lower_choices = IndexMap::new();
    for (form, choice) in &exact_choices {
        let key = casing.lowercase(form).into_owned();
        // Keys are case-insensitive; canonical lemmas are not.
        lower_choices
            .entry(key)
            .and_modify(|existing: &mut EntityChoice| existing.merge(choice))
            .or_insert_with(|| choice.clone());
    }
    let lower = lower_choices
        .into_iter()
        .filter_map(|(form, choice)| match choice {
            EntityChoice::Lemma(lemma) => Some((form, lemma)),
            EntityChoice::Conflict => None,
        })
        .collect();
    let exact = exact_choices
        .into_iter()
        .map(|(form, choice)| match choice {
            EntityChoice::Lemma(lemma) => (form, lemma),
            EntityChoice::Conflict => (form.clone(), form),
        })
        .collect();
    (exact, lower)
}

/// Learns the four lookup tables.
///
/// Aggregating over unique pairs weighted by frequency gives exactly the same
/// statistics as a token-level pass for orders of magnitude less work.
#[must_use]
pub fn build(training: &[Pair], external: Vec<(String, String)>) -> Tables {
    build_for_language_with_entities(training, external, Vec::new(), "")
}

/// Learns common and isolated named-entity lookup tables.
#[must_use]
pub fn build_with_entities(
    training: &[Pair],
    external: Vec<(String, String)>,
    entities: Vec<(String, String)>,
) -> Tables {
    build_for_language_with_entities(training, external, entities, "")
}

/// Learns the four tables with the language's Unicode case mapping.
#[must_use]
pub fn build_for_language(
    training: &[Pair],
    external: Vec<(String, String)>,
    iso639_3: &str,
) -> Tables {
    build_for_language_with_entities(training, external, Vec::new(), iso639_3)
}

/// Learns all tables with language-aware casing and isolated entity data.
#[must_use]
pub fn build_for_language_with_entities(
    training: &[Pair],
    external: Vec<(String, String)>,
    entities: Vec<(String, String)>,
    iso639_3: &str,
) -> Tables {
    let casing = Casing::for_language(iso639_3);
    let mut tables = corpus_tables(training, casing);
    let external = external
        .into_iter()
        .map(|(form, lemma)| (into_nfc(form), into_nfc(lemma)));
    let entities = entities
        .into_iter()
        .map(|(form, lemma)| (into_nfc(form), into_nfc(lemma)));
    tables.external = external_table(external, &tables.exact, &tables.lower, casing);
    (tables.entities, tables.entities_lower) = entity_tables(entities, casing);
    tables
}

/// Learns tables from external pairs already reduced to one lemma per form.
#[must_use]
pub fn build_for_language_with_resolved_external(
    training: &[Pair],
    mut external: IndexMap<String, String>,
    entities: Vec<(String, String)>,
    iso639_3: &str,
) -> Tables {
    let casing = Casing::for_language(iso639_3);
    let mut tables = corpus_tables(training, casing);
    let entities = entities
        .into_iter()
        .map(|(form, lemma)| (into_nfc(form), into_nfc(lemma)));
    external.retain(|form, _| !corpus_answers(form, &tables.exact, &tables.lower, casing));
    tables.external = external;
    (tables.entities, tables.entities_lower) = entity_tables(entities, casing);
    tables
}

/// Relearn corpus-owned tables while moving resolved auxiliary tables forward;
/// external forms the larger corpus now covers are dropped.
#[must_use]
pub fn refit_for_language(training: &[Pair], calibration: Tables) -> Tables {
    let Tables {
        casing,
        exact,
        lower,
        mut external,
        entities,
        entities_lower,
        rules,
    } = calibration;
    // Release calibration corpus maps before allocating their replacements.
    drop((exact, lower, rules));
    let mut tables = corpus_tables(training, casing);
    external.retain(|form, _| !corpus_answers(form, &tables.exact, &tables.lower, casing));
    tables.external = external;
    tables.entities = entities;
    tables.entities_lower = entities_lower;
    tables
}

/// Relearns corpus tables and extends calibrated entities after selection,
/// with development-set `PROPN` rows.
#[must_use]
pub fn refit_for_language_with_entities(
    training: &[Pair],
    calibration: Tables,
    additional_entities: Vec<(String, String)>,
) -> Tables {
    let mut tables = refit_for_language(training, calibration);
    let entities = std::mem::take(&mut tables.entities);
    tables.entities_lower = IndexMap::new();
    let entities = entities.into_iter().chain(
        additional_entities
            .into_iter()
            .map(|(form, lemma)| (into_nfc(form), into_nfc(lemma))),
    );
    (tables.entities, tables.entities_lower) = entity_tables(entities, tables.casing);
    tables
}

/// Learns only the tables whose contents depend on corpus observations.
fn corpus_tables(training: &[Pair], casing: Casing) -> Tables {
    // Builder inputs are already NFC; only copy for a decomposed caller.
    let normalized;
    let training = if training.iter().any(|pair| {
        matches!(nfc(&pair.form), Cow::Owned(_)) || matches!(nfc(&pair.lemma), Cow::Owned(_))
    }) {
        normalized = training
            .iter()
            .map(|pair| Pair {
                form: nfc(&pair.form).into_owned(),
                lemma: nfc(&pair.lemma).into_owned(),
                sentence_initial: pair.sentence_initial,
            })
            .collect::<Vec<_>>();
        normalized.as_slice()
    } else {
        training
    };
    let mut observed: IndexMap<&Pair, u64> = IndexMap::new();
    for pair in training {
        let seen = observed.entry(pair).or_insert(0);
        *seen = seen.saturating_add(1);
    }

    let mut exact: IndexMap<&str, IndexMap<&str, u64>> = IndexMap::new();
    let mut lower: IndexMap<String, IndexMap<&str, u64>> = IndexMap::new();
    let mut buckets: IndexMap<String, IndexMap<Shape, u64>> = IndexMap::new();

    for (pair, count) in &observed {
        let exact_count = exact
            .entry(pair.form.as_str())
            .or_default()
            .entry(pair.lemma.as_str())
            .or_insert(0);
        *exact_count = exact_count.saturating_add(*count);
        let lowered = casing.lowercase(&pair.form).into_owned();
        let bucket = CaseKey::of(&pair.form, pair.sentence_initial).byte() as char;
        let shape = Shape::between(&pair.form, &pair.lemma, casing);
        // Mixed casing (`Keep`) cannot generalise into a rule.
        if shape.case != CaseAction::Keep {
            observe_suffixes(&mut buckets, &lowered, bucket, &shape);
        }

        let lower_count = lower
            .entry(lowered)
            .or_default()
            .entry(pair.lemma.as_str())
            .or_insert(0);
        *lower_count = lower_count.saturating_add(*count);
    }

    // The frequency map is no longer needed once it has fed the three voting
    // structures. Dropping it before materialising their winners keeps the
    // largest training-only allocation out of the finalisation peak.
    drop(observed);

    let mut exact_table = IndexMap::with_capacity(exact.len());
    for (form, candidates) in exact {
        if let Some((lemma, _)) = into_most_common(candidates) {
            exact_table.insert(form.to_owned(), lemma.to_owned());
        }
    }

    let mut lower_table = IndexMap::new();
    for (form, candidates) in lower {
        if let Some((lemma, _)) = into_most_common(candidates) {
            // Keep only what the exact table would not already answer.
            if exact_table.get(&form).map(String::as_str) != Some(lemma) {
                lower_table.insert(form, lemma.to_owned());
            }
        }
    }

    let mut rule_table = IndexMap::new();
    for (key, candidates) in buckets {
        let total: u64 = candidates.values().sum();
        if total < MIN_RULE_SUPPORT {
            continue;
        }
        if let Some((shape, hits)) = into_most_common(candidates) {
            rule_table.insert(
                key,
                Trained {
                    cut: shape.cut,
                    add: shape.add,
                    case: shape.case,
                    confidence: super::evaluation::round3(hits as f64 / total as f64),
                },
            );
        }
    }

    Tables {
        casing,
        exact: exact_table,
        lower: lower_table,
        external: IndexMap::new(),
        entities: IndexMap::new(),
        entities_lower: IndexMap::new(),
        rules: rule_table,
    }
}
