//! Compound boundaries for external lemmas, learned from the training split.
//!
//! Some treebanks spell a compound's lemma with a boundary marker
//! (`kirja#kauppa` in `fi_tdt`, `eba_võrdsus` in `et_edt`) that no external
//! dictionary carries, so every compound the corpus never saw comes back
//! unmarked and therefore wrong. The marker is restored offline: an external
//! lemma that splits into components the training lemmas themselves marked —
//! in exactly one fewest-part way — gets the marker at those cuts. Whether a
//! treebank takes this at all is the acceptance gate's call, made on dev.

use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;

use crate::train::Pair;

/// Shorter pieces would split ordinary words on chance substrings.
const MIN_COMPONENT_CHARS: usize = 3;
/// A component seen once is as likely a typo as a convention.
const MIN_COMPONENT_COUNT: usize = 2;

/// Marks compound boundaries in every external lemma that lacks them.
///
/// Returns how many external rows changed. Lemmas the training split knows
/// whole and unmarked are left alone: the corpus says they are not compounds.
pub fn mark_compounds(
    external: &mut IndexMap<String, String>,
    training: &[Pair],
    marker: char,
) -> usize {
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut whole: HashSet<String> = HashSet::new();
    for pair in training {
        let lemma = pair.lemma.to_lowercase();
        if lemma.trim_matches(marker).contains(marker) {
            for part in lemma.split(marker) {
                if part.chars().count() >= MIN_COMPONENT_CHARS {
                    let count = counts.entry(part.to_owned()).or_default();
                    *count = count.saturating_add(1);
                }
            }
        } else {
            whole.insert(lemma);
        }
    }
    let components: HashSet<String> = counts
        .into_iter()
        .filter(|(_, count)| *count >= MIN_COMPONENT_COUNT)
        .map(|(component, _)| component)
        .collect();
    if components.is_empty() {
        return 0;
    }
    let mut memo: HashMap<String, Option<String>> = HashMap::new();
    let mut changed = 0_usize;
    for lemma in external.values_mut() {
        let replacement = memo
            .entry(lemma.clone())
            .or_insert_with(|| split(lemma, marker, &components, &whole));
        if let Some(replacement) = replacement {
            lemma.clone_from(replacement);
            changed = changed.saturating_add(1);
        }
    }
    changed
}

/// The fewest-part segmentation into known components, if there is exactly one.
fn split(
    lemma: &str,
    marker: char,
    components: &HashSet<String>,
    whole: &HashSet<String>,
) -> Option<String> {
    if lemma.contains(marker) {
        return None;
    }
    let original: Vec<char> = lemma.chars().collect();
    let lower: Vec<char> = lemma.to_lowercase().chars().collect();
    // A case mapping that changes the length would misplace every cut.
    if lower.len() != original.len() || whole.contains(&lower.iter().collect::<String>()) {
        return None;
    }
    // best[i]: fewest parts covering the first i chars, how many ways (capped
    // at two) reach that count, and where the last part starts.
    let mut best: Vec<Option<(usize, u8, usize)>> = vec![None; lower.len().saturating_add(1)];
    *best.first_mut()? = Some((0, 1, 0));
    let mut piece = String::new();
    for end in 1..=lower.len() {
        for start in 0..end {
            let Some(Some((parts, ways, _))) = best.get(start).copied() else {
                continue;
            };
            piece.clear();
            piece.extend(lower.get(start..end)?);
            if !components.contains(piece.as_str()) {
                continue;
            }
            let parts = parts.saturating_add(1);
            let slot = best.get_mut(end)?;
            *slot = match *slot {
                Some((fewest, reach, first)) if parts == fewest => {
                    Some((fewest, reach.saturating_add(ways).min(2), first))
                }
                Some((fewest, ..)) if parts > fewest => *slot,
                _ => Some((parts, ways, start)),
            };
        }
    }
    let (parts, ways, _) = (*best.last()?)?;
    if parts < 2 || ways != 1 {
        return None;
    }
    let mut cuts = Vec::new();
    let mut at = lower.len();
    while at > 0 {
        let (_, _, start) = (*best.get(at)?)?;
        if start > 0 {
            cuts.push(start);
        }
        at = start;
    }
    let mut rewritten = String::with_capacity(lemma.len().saturating_add(cuts.len()));
    for (index, character) in original.iter().enumerate() {
        if cuts.contains(&index) {
            rewritten.push(marker);
        }
        rewritten.push(*character);
    }
    Some(rewritten)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(form: &str, lemma: &str) -> Pair {
        Pair {
            form: form.to_owned(),
            lemma: lemma.to_owned(),
            sentence_initial: false,
        }
    }

    #[test]
    fn external_compounds_take_the_training_marker() {
        let training = [
            pair("kirjakaupassa", "kirja#kauppa"),
            pair("kirjakauppaan", "kirja#kauppa"),
            pair("kauppatorilla", "kauppa#tori"),
            pair("kauppatori", "kauppa#tori"),
            pair("kirjasto", "kirjasto"),
        ];
        let mut external: IndexMap<String, String> = [
            ("kirjakaupoissa", "kirjakauppa"),
            ("torikauppiaalla", "torikauppa"),
            ("kirjastossa", "kirjasto"),
            ("talossa", "talo"),
        ]
        .into_iter()
        .map(|(form, lemma)| (form.to_owned(), lemma.to_owned()))
        .collect();
        assert_eq!(mark_compounds(&mut external, &training, '#'), 2);
        let lemmas: Vec<&str> = external.values().map(String::as_str).collect();
        // `kirjasto` is a whole training lemma, `talo` no known components.
        assert_eq!(lemmas, ["kirja#kauppa", "tori#kauppa", "kirjasto", "talo"]);
    }

    #[test]
    fn an_ambiguous_segmentation_is_left_alone() {
        let training = [
            pair("a", "xxx#yyyzzz"),
            pair("a", "xxx#yyyzzz"),
            pair("b", "xxxyyy#zzz"),
            pair("b", "xxxyyy#zzz"),
        ];
        let mut external: IndexMap<String, String> = [("f", "xxxyyyzzz"), ("g", "zzzxxx")]
            .into_iter()
            .map(|(form, lemma)| (form.to_owned(), lemma.to_owned()))
            .collect();
        // xxx|yyyzzz and xxxyyy|zzz tie on two parts: no guess.
        assert_eq!(mark_compounds(&mut external, &training, '#'), 1);
        assert_eq!(external.get("f").map(String::as_str), Some("xxxyyyzzz"));
        assert_eq!(external.get("g").map(String::as_str), Some("zzz#xxx"));
    }
}
