//! Paradigm decoding, training-source selection, and ordered source merging.

use std::borrow::Cow;
use std::collections::HashSet;
use std::path::Path;

use indexmap::IndexMap;

use super::context_keys::Observations;
use super::corpus::{TrainingData, read_training_data};
use super::input::{exists_nonempty, for_each_line, verify_expected};
use super::{Error, ObservedInput, Result, Spec};
use crate::lexicon::FileIntegrity;
use crate::normalization::scripts_compatible;
use crate::train::{ExternalResolver, Pair};

type LemmaPairs = Vec<(String, String)>;
type ObservedLemmaPairs = (LemmaPairs, Vec<ObservedInput>);

/// Streams an external `form -> lemma` table without hiding corrupt rows.
fn for_each_paradigm(
    path: &Path,
    lemma_first: bool,
    legacy_tolerances: bool,
    iso639_3: &str,
    mut visit: impl FnMut(usize, String, String, bool, bool) -> Result<()>,
) -> Result<FileIntegrity> {
    for_each_line(path, |line_number, line| {
        if legacy_tolerances && line.is_empty() {
            return Ok(());
        }
        if line
            .chars()
            .any(|character| character != '\t' && character.is_control())
        {
            return Err(Error::Paradigm {
                path: path.to_owned(),
                line: line_number,
                message: "rows may not contain control characters other than tab".to_owned(),
            });
        }
        if legacy_tolerances && line.starts_with("# ") {
            return Ok(());
        }
        let mut columns = line.split('\t');
        let (Some(first), Some(second)) = (columns.next(), columns.next()) else {
            return Err(Error::Paradigm {
                path: path.to_owned(),
                line: line_number,
                message: "expected at least two tab-separated columns".to_owned(),
            });
        };
        if first.is_empty() || second.is_empty() {
            if legacy_tolerances {
                return Ok(());
            }
            return Err(Error::Paradigm {
                path: path.to_owned(),
                line: line_number,
                message: "the first two columns must be non-empty".to_owned(),
            });
        }
        if first.trim().is_empty() || second.trim().is_empty() {
            return Err(Error::Paradigm {
                path: path.to_owned(),
                line: line_number,
                message: "the first two columns must not be whitespace-only".to_owned(),
            });
        }
        let (lemma, form) = if lemma_first {
            (first, second)
        } else {
            (second, first)
        };
        let already_nfc =
            unicode_normalization::is_nfc(form) && unicode_normalization::is_nfc(lemma);
        let normalized_form = crate::normalization::nfc(form);
        let normalized_lemma = crate::normalization::nfc(lemma);
        let compatible_scripts = scripts_compatible(&normalized_form, &normalized_lemma, iso639_3);
        visit(
            line_number,
            normalized_form.into_owned(),
            normalized_lemma.into_owned(),
            already_nfc,
            compatible_scripts,
        )
    })
}

/// Collects a paradigm only when it is itself the training signal.
#[cfg(test)]
pub(super) fn read_paradigm(path: &Path, lemma_first: bool) -> Result<Vec<(String, String)>> {
    read_paradigm_with_policy(path, lemma_first, true, "zxx").map(|(pairs, _integrity)| pairs)
}

fn read_paradigm_with_policy(
    path: &Path,
    lemma_first: bool,
    legacy_tolerances: bool,
    iso639_3: &str,
) -> Result<(Vec<(String, String)>, FileIntegrity)> {
    let mut pairs = Vec::new();
    let integrity = for_each_paradigm(
        path,
        lemma_first,
        legacy_tolerances,
        iso639_3,
        |_line, form, lemma, _already_nfc, compatible_scripts| {
            if compatible_scripts {
                pairs.push((form, lemma));
            }
            Ok(())
        },
    )?;
    Ok((pairs, integrity))
}

/// The training signal: the treebank's own split when it yields anything,
/// else the first external source that does.
pub(super) struct TrainingSignal {
    pub(super) pairs: Vec<Pair>,
    pub(super) proper_nouns: Vec<(String, String)>,
    /// Context features of the treebank's tokens; never of a paradigm.
    pub(super) context: Option<Observations>,
    pub(super) source: String,
    pub(super) paradigm_index: Option<usize>,
}

pub(super) fn training_signal_observed(
    spec: &Spec<'_>,
) -> Result<(Option<TrainingSignal>, Vec<ObservedInput>)> {
    let mut observed = Vec::new();
    if let Some(train) = spec.train
        && exists_nonempty(train)?
    {
        let (
            TrainingData {
                pairs,
                proper_nouns,
                context,
            },
            integrity,
        ) = read_training_data(
            train,
            spec.name,
            spec.context_keys.then(|| {
                Observations::new(crate::normalization::Casing::for_language(spec.iso639_3))
            }),
        )?;
        observed.push(ObservedInput {
            role: "corpus",
            label: format!("{}.train", spec.name),
            path: train.to_owned(),
            integrity,
        });
        if !pairs.is_empty() {
            return Ok((
                Some(TrainingSignal {
                    pairs,
                    proper_nouns,
                    context,
                    source: format!("{}.train", spec.name),
                    paradigm_index: None,
                }),
                observed,
            ));
        }
    }
    for (index, paradigm) in spec.paradigms.iter().enumerate() {
        if !exists_nonempty(&paradigm.path)? {
            continue;
        }
        let (raw_pairs, integrity) = read_paradigm_with_policy(
            &paradigm.path,
            paradigm.lemma_first,
            !paradigm.normalized_unique,
            spec.iso639_3,
        )?;
        verify_expected(&paradigm.path, &integrity, paradigm.expected.as_ref())?;
        observed.push(ObservedInput {
            role: "paradigm",
            label: paradigm.label.clone(),
            path: paradigm.path.clone(),
            integrity,
        });
        let pairs: Vec<Pair> = raw_pairs
            .into_iter()
            .map(|(form, lemma)| Pair {
                form,
                lemma,
                sentence_initial: false,
            })
            .collect();
        if !pairs.is_empty() {
            return Ok((
                Some(TrainingSignal {
                    pairs,
                    proper_nouns: Vec::new(),
                    context: None,
                    source: format!("{}/{}", paradigm.label, spec.iso639_3),
                    paradigm_index: Some(index),
                }),
                observed,
            ));
        }
    }
    Ok((None, observed))
}

#[cfg(test)]
pub(super) fn training_signal(spec: &Spec<'_>) -> Result<Option<TrainingSignal>> {
    training_signal_observed(spec).map(|(signal, _observed)| signal)
}

/// External vocabulary from every source that is not already the training
/// signal, in preference order.
pub(super) fn external_pairs_observed(
    spec: &Spec<'_>,
    training_paradigm: Option<usize>,
) -> Result<(IndexMap<String, String>, Vec<ObservedInput>)> {
    let mut pairs = IndexMap::new();
    let mut observed = Vec::new();
    for (index, paradigm) in spec.paradigms.iter().enumerate() {
        if training_paradigm == Some(index) || !exists_nonempty(&paradigm.path)? {
            continue;
        }
        let integrity = if paradigm.normalized_unique {
            stream_normalized_unique(
                &mut pairs,
                &paradigm.path,
                paradigm.lemma_first,
                spec.iso639_3,
            )?
        } else {
            stream_resolved_preferred(
                &mut pairs,
                &paradigm.path,
                paradigm.lemma_first,
                spec.iso639_3,
            )?
        };
        verify_expected(&paradigm.path, &integrity, paradigm.expected.as_ref())?;
        observed.push(ObservedInput {
            role: "paradigm",
            label: paradigm.label.clone(),
            path: paradigm.path.clone(),
            integrity,
        });
    }
    Ok((pairs, observed))
}

#[cfg(test)]
pub(super) fn external_pairs(
    spec: &Spec<'_>,
    training_paradigm: Option<usize>,
) -> Result<IndexMap<String, String>> {
    external_pairs_observed(spec, training_paradigm).map(|(pairs, _observed)| pairs)
}

/// Curated named-entity forms from every configured source.
///
/// A single-word row whose NFC surface the corpus already claims as `PROPN`
/// is skipped: the entity reducer would turn the disagreement into identity
/// and destroy the annotated rewrite (measured on the dev `PROPN` gate).
pub(super) fn entity_pairs_observed(
    spec: &Spec<'_>,
    corpus_proper_nouns: &[(String, String)],
) -> Result<ObservedLemmaPairs> {
    let claimed: HashSet<Cow<'_, str>> = corpus_proper_nouns
        .iter()
        .map(|(form, _lemma)| crate::normalization::nfc(form))
        .collect();
    let mut decisions = IndexMap::new();
    let mut observed = Vec::new();
    for paradigm in spec.entities {
        if exists_nonempty(&paradigm.path)? {
            let integrity = for_each_entity(
                &paradigm.path,
                paradigm.lemma_first,
                |form, lemma, semantics| {
                    if !form.chars().any(char::is_whitespace) && claimed.contains(form.as_str()) {
                        return;
                    }
                    match semantics {
                        // An alias may name another entity; protection always wins.
                        EntitySemantics::Identity => {
                            decisions.insert(form, EntityDecision::Protection);
                        }
                        EntitySemantics::Inflection
                            if !scripts_compatible(&form, &lemma, spec.iso639_3) =>
                        {
                            decisions.insert(form, EntityDecision::Protection);
                        }
                        EntitySemantics::Inflection => match decisions.entry(form) {
                            indexmap::map::Entry::Vacant(entry) => {
                                entry.insert(EntityDecision::Rewrite(lemma));
                            }
                            indexmap::map::Entry::Occupied(mut entry) => match entry.get() {
                                EntityDecision::Protection | EntityDecision::Conflict => {}
                                EntityDecision::Rewrite(existing) if *existing == lemma => {}
                                EntityDecision::Rewrite(_) => {
                                    entry.insert(EntityDecision::Conflict);
                                }
                            },
                        },
                    }
                },
            )?;
            verify_expected(&paradigm.path, &integrity, paradigm.expected.as_ref())?;
            observed.push(ObservedInput {
                role: "entity",
                label: paradigm.label.clone(),
                path: paradigm.path.clone(),
                integrity,
            });
        }
    }
    Ok((
        decisions
            .into_iter()
            .map(|(form, decision)| match decision {
                EntityDecision::Rewrite(lemma) => (form, lemma),
                EntityDecision::Protection | EntityDecision::Conflict => (form.clone(), form),
            })
            .collect(),
        observed,
    ))
}

#[cfg(test)]
pub(super) fn entity_pairs(
    spec: &Spec<'_>,
    corpus_proper_nouns: &[(String, String)],
) -> Result<Vec<(String, String)>> {
    entity_pairs_observed(spec, corpus_proper_nouns).map(|(pairs, _observed)| pairs)
}

/// Converts corpus `PROPN` observations into entity evidence; a cross-script
/// annotation becomes identity, still protecting the surface.
pub(super) fn guarded_proper_nouns(
    pairs: Vec<(String, String)>,
    iso639_3: &str,
) -> Vec<(String, String)> {
    pairs
        .into_iter()
        .map(|(form, lemma)| {
            if scripts_compatible(&form, &lemma, iso639_3) {
                (form, lemma)
            } else {
                (form.clone(), form)
            }
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EntitySemantics {
    Identity,
    Inflection,
}

enum EntityDecision {
    /// Alias/anchor evidence protects the spelling and dominates every rewrite.
    Protection,
    /// One or more morphological sources agree on this lemma.
    Rewrite(String),
    /// Morphological sources disagree, so the surface must survive unchanged.
    Conflict,
}

/// Read the entity table format; the semantics column is mandatory, it tells
/// a morphological form from an alias.
fn for_each_entity(
    path: &Path,
    lemma_first: bool,
    mut visit: impl FnMut(String, String, EntitySemantics),
) -> Result<FileIntegrity> {
    for_each_line(path, |line_number, line| {
        if line
            .chars()
            .any(|character| character != '\t' && character.is_control())
        {
            return Err(Error::Paradigm {
                path: path.to_owned(),
                line: line_number,
                message: "rows may not contain control characters other than tab".to_owned(),
            });
        }
        let mut columns = line.split('\t');
        let (Some(first), Some(second), Some(raw_semantics)) =
            (columns.next(), columns.next(), columns.next())
        else {
            return Err(Error::Paradigm {
                path: path.to_owned(),
                line: line_number,
                message: "expected surface, lemma, and entity semantics columns".to_owned(),
            });
        };
        if columns.next().is_some() {
            return Err(Error::Paradigm {
                path: path.to_owned(),
                line: line_number,
                message: "entity rows must contain exactly three columns".to_owned(),
            });
        }
        if first.is_empty() || second.is_empty() {
            return Err(Error::Paradigm {
                path: path.to_owned(),
                line: line_number,
                message: "the first two columns must be non-empty".to_owned(),
            });
        }
        let (lemma, form) = if lemma_first {
            (first, second)
        } else {
            (second, first)
        };
        let semantics = match raw_semantics {
            "identity" if form == lemma => EntitySemantics::Identity,
            "identity" => {
                return Err(Error::Paradigm {
                    path: path.to_owned(),
                    line: line_number,
                    message: "identity entity rows must preserve the surface".to_owned(),
                });
            }
            "inflection" => EntitySemantics::Inflection,
            other => {
                return Err(Error::Paradigm {
                    path: path.to_owned(),
                    line: line_number,
                    message: format!("unknown entity semantics {other:?}"),
                });
            }
        };
        visit(
            crate::normalization::nfc(form).into_owned(),
            crate::normalization::nfc(lemma).into_owned(),
            semantics,
        );
        Ok(())
    })
}

/// Stream an authenticated form-unique import without a redundant reducer.
fn stream_normalized_unique(
    destination: &mut IndexMap<String, String>,
    path: &Path,
    lemma_first: bool,
    iso639_3: &str,
) -> Result<FileIntegrity> {
    let mut previous: Option<String> = None;
    for_each_paradigm(
        path,
        lemma_first,
        false,
        iso639_3,
        |line, form, lemma, already_nfc, compatible_scripts| {
            if !already_nfc {
                return Err(Error::Paradigm {
                    path: path.to_owned(),
                    line,
                    message: "normalized imports must already be NFC".to_owned(),
                });
            }
            if previous.as_ref().is_some_and(|prior| form <= *prior) {
                return Err(Error::Paradigm {
                    path: path.to_owned(),
                    line,
                    message: "normalized import forms must be strictly sorted and unique"
                        .to_owned(),
                });
            }
            previous = Some(form.clone());
            // A cross-script row is skipped; structural violations above are fatal.
            if !compatible_scripts {
                return Ok(());
            }
            if let indexmap::map::Entry::Vacant(entry) = destination.entry(form) {
                entry.insert(lemma);
            }
            Ok(())
        },
    )
}

/// Resolve one legacy source on its own, then let its answers claim new forms,
/// so a later source cannot change an earlier one's answer.
fn stream_resolved_preferred(
    destination: &mut IndexMap<String, String>,
    path: &Path,
    lemma_first: bool,
    iso639_3: &str,
) -> Result<FileIntegrity> {
    let mut resolver = ExternalResolver::default();
    let integrity = for_each_paradigm(
        path,
        lemma_first,
        true,
        iso639_3,
        |_line, form, lemma, _already_nfc, compatible_scripts| {
            if compatible_scripts {
                resolver.add(form, lemma);
            }
            Ok(())
        },
    )?;
    for (form, lemma) in resolver.finish() {
        if let indexmap::map::Entry::Vacant(entry) = destination.entry(form) {
            entry.insert(lemma);
        }
    }
    Ok(integrity)
}
