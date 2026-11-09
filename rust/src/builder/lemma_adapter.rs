//! External lemmas respelled in the treebank's own convention.
//!
//! Dictionaries lemmatise by their own rules: `превърна` where `bg_btb` writes
//! `превърна-(се)`, the participle `згорілий` where `uk_iu` wants the verb
//! `згоріти`. Where an external row and the training split answer the same
//! form, the pair shows how one convention maps onto the other; the regular
//! part of that mapping is applied to every external row the training split
//! does not answer. Whether a treebank takes this at all is the acceptance
//! gate's call, made on dev.

use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;

use crate::train::Pair;

/// A lemma mapping seen once is as likely chance as convention.
const MIN_LEMMA_SUPPORT: usize = 2;
/// Longest external-lemma suffix an edit is keyed on, in characters.
const MAX_SUFFIX_CHARS: usize = 6;
/// Pairs a suffix needs before its majority edit is trusted.
const MIN_SUFFIX_SUPPORT: usize = 4;

/// Characters cut from the end of the lowercased lemma, then what is appended.
type Edit = (usize, String);

/// Rewrites external lemmas the way the training split spells them.
///
/// Returns how many external rows changed. Rows whose form the training
/// split answers are the evidence and stay as they are.
pub fn adapt_external(external: &mut IndexMap<String, String>, training: &[Pair]) -> usize {
    let mut votes: HashMap<String, IndexMap<&str, usize>> = HashMap::new();
    for pair in training {
        bump(
            votes
                .entry(pair.form.to_lowercase())
                .or_default()
                .entry(pair.lemma.as_str()),
        );
    }
    let gold: HashMap<String, &str> = votes
        .into_iter()
        .filter_map(|(form, lemmas)| Some((form, *majority(&lemmas)?.0)))
        .collect();

    let mut known = HashSet::new();
    let mut by_lemma: HashMap<&str, IndexMap<&str, usize>> = HashMap::new();
    let mut by_suffix: HashMap<String, IndexMap<Edit, usize>> = HashMap::new();
    for (index, (form, lemma)) in external.iter().enumerate() {
        let Some(&answer) = gold.get(&form.to_lowercase()) else {
            continue;
        };
        known.insert(index);
        bump(by_lemma.entry(lemma).or_default().entry(answer));
        let lower: Vec<char> = lemma.to_lowercase().chars().collect();
        let edit = edit_between(&lower, &answer.to_lowercase());
        for length in 1..=lower.len().min(MAX_SUFFIX_CHARS) {
            let suffix = lower.get(lower.len().saturating_sub(length)..);
            let suffix: String = suffix.unwrap_or_default().iter().collect();
            bump(by_suffix.entry(suffix).or_default().entry(edit.clone()));
        }
    }
    let lemmas: HashMap<String, String> = by_lemma
        .into_iter()
        .filter_map(|(lemma, answers)| {
            let (answer, count) = majority(&answers)?;
            (*answer != lemma
                && count >= MIN_LEMMA_SUPPORT
                && count.saturating_mul(3) >= total(&answers).saturating_mul(2))
            .then(|| (lemma.to_owned(), (*answer).to_owned()))
        })
        .collect();
    let suffixes: HashMap<String, Edit> = by_suffix
        .into_iter()
        .filter_map(|(suffix, edits)| {
            let support = total(&edits);
            let (edit, count) = majority(&edits)?;
            (support >= MIN_SUFFIX_SUPPORT && count.saturating_mul(10) >= support.saturating_mul(9))
                .then(|| (suffix, edit.clone()))
        })
        .collect();
    if lemmas.is_empty() && suffixes.is_empty() {
        return 0;
    }

    let mut memo: HashMap<String, Option<String>> = HashMap::new();
    let mut changed = 0_usize;
    for (index, lemma) in external.values_mut().enumerate() {
        if known.contains(&index) {
            continue;
        }
        let replacement = memo.entry(lemma.clone()).or_insert_with(|| {
            lemmas
                .get(lemma.as_str())
                .cloned()
                .or_else(|| respell(lemma, &suffixes))
        });
        if let Some(replacement) = replacement {
            lemma.clone_from(replacement);
            changed = changed.saturating_add(1);
        }
    }
    changed
}

fn bump<K>(entry: indexmap::map::Entry<'_, K, usize>) {
    let count = entry.or_insert(0);
    *count = count.saturating_add(1);
}

/// The most frequent key, ties going to the first inserted.
fn majority<K>(counts: &IndexMap<K, usize>) -> Option<(&K, usize)> {
    counts.iter().fold(None, |best, (key, &count)| match best {
        Some((_, high)) if count <= high => best,
        _ => Some((key, count)),
    })
}

fn total<K>(counts: &IndexMap<K, usize>) -> usize {
    counts
        .values()
        .fold(0, |sum, count| sum.saturating_add(*count))
}

/// The edit over the shared prefix, as `Shape::between` derives a rule.
fn edit_between(lemma: &[char], answer: &str) -> Edit {
    let shared = lemma
        .iter()
        .zip(answer.chars())
        .take_while(|(a, b)| **a == *b)
        .count();
    (
        lemma.len().saturating_sub(shared),
        answer.chars().skip(shared).collect(),
    )
}

/// Applies the longest suffix's edit; an identity edit there keeps the lemma.
fn respell(lemma: &str, suffixes: &HashMap<String, Edit>) -> Option<String> {
    let original: Vec<char> = lemma.chars().collect();
    let lower: Vec<char> = lemma.to_lowercase().chars().collect();
    // A case mapping that changes the length would misplace the cut.
    if lower.len() != original.len() {
        return None;
    }
    let mut suffix = String::new();
    for length in (1..=lower.len().min(MAX_SUFFIX_CHARS)).rev() {
        suffix.clear();
        suffix.extend(lower.get(lower.len().saturating_sub(length)..)?);
        let Some((cut, add)) = suffixes.get(&suffix) else {
            continue;
        };
        if *cut == 0 && add.is_empty() {
            return None;
        }
        let kept = original.get(..original.len().checked_sub(*cut)?)?;
        return Some(kept.iter().copied().chain(add.chars()).collect());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn training(rows: &[(&str, &str)]) -> Vec<Pair> {
        rows.iter()
            .map(|(form, lemma)| Pair {
                form: (*form).to_owned(),
                lemma: (*lemma).to_owned(),
                sentence_initial: false,
            })
            .collect()
    }

    fn table(rows: &[(&str, &str)]) -> IndexMap<String, String> {
        rows.iter()
            .map(|(form, lemma)| ((*form).to_owned(), (*lemma).to_owned()))
            .collect()
    }

    /// Participles the corpus lemmatises to their verbs (`va` -> `a`), each
    /// as `(training form, external lemma, gold)`, plus two unseen rows.
    fn participles(extra: &[(&str, &str, &str)]) -> (Vec<Pair>, IndexMap<String, String>) {
        let rows = [
            ("edustavaksi", "edustava", "edustaa"),
            ("kuuluvaksi", "kuuluva", "kuulua"),
            ("Puhuvaa", "puhuva", "puhua"),
            ("lukevaksi", "lukeva", "lukea"),
        ];
        let rows: Vec<_> = rows.iter().chain(extra).collect();
        let pairs: Vec<_> = rows.iter().map(|(form, _, gold)| (*form, *gold)).collect();
        let external = rows
            .iter()
            .map(|(form, lemma, _)| (form.to_lowercase(), (*lemma).to_owned()))
            .chain(
                [("laulavaksi", "laulava"), ("talossa", "talo")]
                    .map(|(form, lemma)| (form.to_owned(), lemma.to_owned())),
            )
            .collect();
        (training(&pairs), external)
    }

    #[test]
    fn a_repeated_lemma_mapping_carries_to_unseen_forms() {
        let training = training(&[
            ("превърнем", "превърна-(се)"),
            ("Превърне", "превърна-(се)"),
            ("превърнах", "превърна"),
        ]);
        let mut external = table(&[
            ("превърнем", "превърна"),
            ("превърне", "превърна"),
            ("превърнах", "превърна"),
            ("превърнете", "превърна"),
        ]);
        // Two of three pairs: exactly the 2/3 floor.
        assert_eq!(adapt_external(&mut external, &training), 1);
        let lemmas: Vec<&str> = external.values().map(String::as_str).collect();
        // Evidence rows keep the dictionary's answer.
        assert_eq!(
            lemmas,
            ["превърна", "превърна", "превърна", "превърна-(се)"]
        );
    }

    #[test]
    fn a_consistent_suffix_edit_respells_other_lemmas() {
        let (training, mut external) = participles(&[]);
        assert_eq!(adapt_external(&mut external, &training), 1);
        assert_eq!(
            external.get("laulavaksi").map(String::as_str),
            Some("laulaa")
        );
        assert_eq!(external.get("talossa").map(String::as_str), Some("talo"));
    }

    #[test]
    fn a_minority_mapping_is_not_applied() {
        let training = training(&[
            ("згоріла", "згоріти"),
            ("згоріле", "згоріти"),
            ("згорілою", "згорілий"),
            ("згорілому", "згорілий"),
        ]);
        let mut external = table(&[
            ("згоріла", "згорілий"),
            ("згоріле", "згорілий"),
            ("згорілою", "згорілий"),
            ("згорілому", "згорілий"),
            ("згорілих", "згорілий"),
        ]);
        // Half the pairs keep the lemma: below 2/3 for the map and 0.9 for
        // the suffix edit.
        assert_eq!(adapt_external(&mut external, &training), 0);
        assert_eq!(
            external.get("згорілих").map(String::as_str),
            Some("згорілий")
        );
    }

    #[test]
    fn identity_votes_block_a_suffix_edit() {
        // One adjective the corpus keeps as is: `va` now backs the edit 4:1.
        let (training, mut external) = participles(&[("vahvaa", "vahva", "vahva")]);
        assert_eq!(adapt_external(&mut external, &training), 0);
        assert_eq!(
            external.get("laulavaksi").map(String::as_str),
            Some("laulava")
        );
    }
}
