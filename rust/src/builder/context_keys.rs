//! Context keys: exact-table rows that settle an ambiguous form by what the
//! caller knows of its token — the UPOS tag, the previous or the next word.
//!
//! A form the training split saw with several lemmas always answers with the
//! majority one. Where one feature of its occurrences reliably picks another
//! lemma, the pair `form + feature` is filed in the exact table under a key
//! no token can spell (see [`crate::format::context_key`]), and only a caller
//! who passes context ever asks for it.

use std::collections::HashMap;

use indexmap::{IndexMap, IndexSet};

use super::corpus::{Line, lemma_of};
use crate::format::{CONTEXT_NEXT, CONTEXT_PREVIOUS, CONTEXT_UPOS, context_key, neighbours};
use crate::normalization::{Casing, nfc};
use crate::probe::context_word;
use crate::train::into_most_common;

/// Feature prefixes in the order the reader asks them.
const FEATURES: [&str; 3] = [CONTEXT_UPOS, CONTEXT_PREVIOUS, CONTEXT_NEXT];
/// Observations a key needs: one occurrence is as likely chance as signal.
const MIN_SUPPORT: u64 = 2;

/// One scorable token: form, lemma and each feature, as interned ids.
struct Row {
    form: u32,
    lemma: u32,
    features: [Option<u32>; 3],
}

/// Every token the pairs learn from, with its features.
///
/// Strings are interned: the corpus repeats them, and the rows of a
/// multi-million-token treebank stay at 32 bytes each.
pub(super) struct Observations {
    casing: Casing,
    strings: IndexSet<String>,
    rows: Vec<Row>,
}

impl Observations {
    pub(super) fn new(casing: Casing) -> Self {
        Self {
            casing,
            strings: IndexSet::new(),
            rows: Vec::new(),
        }
    }

    fn intern(&mut self, text: &str) -> Option<u32> {
        let index = match self.strings.get_index_of(text) {
            Some(index) => index,
            None => self.strings.insert_full(text.to_owned()).0,
        };
        u32::try_from(index).ok()
    }

    /// Records the tokens of one sentence that [`super::corpus`] pairs.
    pub(super) fn add(&mut self, sentence: &Line) {
        let casing = self.casing;
        for (index, form) in sentence.tokens.iter().enumerate() {
            let Some(lemma) = lemma_of(sentence, index, form) else {
                continue;
            };
            let (previous, next) = neighbours(&sentence.tokens, index, |at| at == 0);
            let upos = sentence
                .upos
                .as_ref()
                .and_then(|tags| tags.get(index))
                .and_then(Option::as_deref);
            let (Some(form), Some(lemma)) = (self.intern(&nfc(form)), self.intern(&nfc(lemma)))
            else {
                continue;
            };
            let features = [
                upos.and_then(|tag| self.intern(tag)),
                self.intern(&context_word(previous, casing)),
                self.intern(&context_word(next, casing)),
            ];
            self.rows.push(Row {
                form,
                lemma,
                features,
            });
        }
    }

    /// The learned `key -> lemma` rows.
    ///
    /// A key is kept when its form is ambiguous, its own majority lemma is
    /// not the form's, it has at least two observations, and that majority
    /// holds at least three quarters of them. Ties go to the first lemma
    /// seen, as in the exact table.
    pub(super) fn keys(&self) -> Vec<(String, String)> {
        let mut forms: IndexMap<u32, IndexMap<u32, u64>> = IndexMap::new();
        for row in &self.rows {
            bump(forms.entry(row.form).or_default().entry(row.lemma));
        }
        let majority: HashMap<u32, u32> = forms
            .into_iter()
            .filter(|(_, lemmas)| lemmas.len() > 1)
            .filter_map(|(form, lemmas)| Some((form, into_most_common(lemmas)?.0)))
            .collect();
        let mut votes: IndexMap<(u32, usize, u32), IndexMap<u32, u64>> = IndexMap::new();
        for row in self
            .rows
            .iter()
            .filter(|row| majority.contains_key(&row.form))
        {
            for (feature, value) in row.features.iter().enumerate() {
                if let Some(value) = value {
                    bump(
                        votes
                            .entry((row.form, feature, *value))
                            .or_default()
                            .entry(row.lemma),
                    );
                }
            }
        }
        let mut keys = Vec::new();
        let mut key = String::new();
        for ((form, feature, value), lemmas) in votes {
            let total: u64 = lemmas.values().sum();
            let Some((lemma, hits)) = into_most_common(lemmas) else {
                continue;
            };
            if total < MIN_SUPPORT
                || majority.get(&form) == Some(&lemma)
                || hits.saturating_mul(4) < total.saturating_mul(3)
            {
                continue;
            }
            let (Some(form), Some(feature), Some(value), Some(lemma)) = (
                self.string(form),
                FEATURES.get(feature),
                self.string(value),
                self.string(lemma),
            ) else {
                continue;
            };
            context_key(&mut key, form, feature, value);
            keys.push((key.clone(), lemma.to_owned()));
        }
        keys
    }

    fn string(&self, id: u32) -> Option<&str> {
        self.strings
            .get_index(usize::try_from(id).ok()?)
            .map(String::as_str)
    }
}

fn bump(entry: indexmap::map::Entry<'_, u32, u64>) {
    let count = entry.or_insert(0);
    *count = count.saturating_add(1);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sentence(tokens: &[&str], lemmas: &[&str], upos: &[&str]) -> Line {
        let owned = |items: &[&str]| items.iter().map(|item| Some((*item).to_owned())).collect();
        Line {
            tokens: tokens.iter().map(|token| (*token).to_owned()).collect(),
            lemmas: Some(owned(lemmas)),
            upos: Some(owned(upos)),
        }
    }

    fn learned(sentences: &[Line]) -> Vec<(String, String)> {
        let mut observations = Observations::new(Casing::Root);
        for line in sentences {
            observations.add(line);
        }
        observations.keys()
    }

    #[test]
    fn a_feature_that_picks_the_minority_lemma_becomes_a_key() {
        // `стали` is mostly the verb; twice a noun after `Из`.
        let verb = sentence(
            &["Они", "стали", "ждать"],
            &["они", "стать", "ждать"],
            &["PRON", "VERB", "VERB"],
        );
        let noun = sentence(&["Из", "стали"], &["из", "сталь"], &["ADP", "NOUN"]);
        let keys = learned(&[verb.clone(), verb.clone(), verb, noun.clone(), noun]);
        assert_eq!(
            keys,
            [
                ("стали\u{1f}u:NOUN".to_owned(), "сталь".to_owned()),
                ("стали\u{1f}p:из".to_owned(), "сталь".to_owned()),
                ("стали\u{1f}n:</s>".to_owned(), "сталь".to_owned()),
            ]
        );
    }

    #[test]
    fn weak_or_majority_features_and_unambiguous_forms_learn_nothing() {
        let verb = sentence(&["стали"], &["стать"], &["VERB"]);
        let noun = sentence(&["стали"], &["сталь"], &["NOUN"]);
        let none: Vec<(String, String)> = Vec::new();
        // One observation is not support; the majority lemma needs no key.
        assert_eq!(learned(&[verb.clone(), verb.clone(), noun.clone()]), none);
        // Two of three is under the three-quarter share.
        let mixed = sentence(&["стали"], &["стать"], &["NOUN"]);
        assert_eq!(
            learned(&[verb.clone(), verb.clone(), verb, noun.clone(), noun, mixed]),
            none
        );
        // A form with one lemma has nothing to disambiguate.
        let only = sentence(&["кот"], &["кот"], &["NOUN"]);
        assert_eq!(learned(&[only.clone(), only]), none);
    }
}
