//! Strict corpus decoding, flattening, and train-only calibration splits.

use std::collections::HashSet;
use std::path::Path;

use serde::Deserialize;
use sha2::{Digest as _, Sha256};

use super::context_keys::Observations;
use super::input::for_each_line;
use super::{Error, Result};
use crate::lexicon::FileIntegrity;
use crate::train::{Graded, Pair};

/// Small sources do not contain enough independent forms for a stable
/// train-only selection signal.
const MIN_CV_FORMS: usize = 256;
/// Avoid cloning multi-million-row dictionaries just to calibrate a build.
const MAX_CV_PAIRS: usize = 2_000_000;
/// One of eight stable hash buckets becomes the train-only holdout.
const CV_BUCKETS: u8 = 8;
/// The same count as a width, for the arithmetic below.
const CV_BUCKET_COUNT: usize = CV_BUCKETS as usize;
/// Fewest holdout forms worth splitting for: an eighth of the floor, halved.
const MIN_HELD_OUT_FORMS: usize = MIN_CV_FORMS / CV_BUCKET_COUNT / 2;

/// One corpus sentence, as a line of the project's JSONL spells it.
///
/// Unknown keys (`ner`, whatever a future exporter adds) are ignored, which
/// is what `payload.get(...)` on a parsed dictionary did.
#[derive(Debug, Clone, Deserialize)]
pub(super) struct Line {
    /// The surface tokens.
    pub(super) tokens: Vec<String>,
    /// Gold lemmas aligned to `tokens`; absent or `null` when unannotated.
    #[serde(default)]
    pub(super) lemmas: Option<Vec<Option<String>>>,
    /// Universal part-of-speech tags aligned to `tokens`.
    #[serde(default)]
    pub(super) upos: Option<Vec<Option<String>>>,
}

/// The Universal Dependencies tag for proper nouns.
const PROPER_NOUN: &str = "PROPN";

/// The marker Universal Dependencies uses for a deliberately absent lemma.
const UNANNOTATED: &str = "_";

/// Whether a lemma may be learned from or scored against.
///
/// Universal Dependencies marks a deliberately unannotated lemma with an
/// underscore; such positions must count neither for nor against anyone. The
/// underscore is only a marker when the form itself is not an underscore —
/// the token `_` legitimately has the lemma `_`.
pub(super) fn scorable(form: &str, lemma: &str) -> bool {
    !(lemma == UNANNOTATED && form != UNANNOTATED)
}

/// Source-specific missing annotations, not words to ban from dictionaries.
///
/// Apply the listed missing-lemma conventions only to their exact treebanks,
/// as confirmed against pinned train/dev sources. The values can be real
/// lemmas elsewhere; keep literal identity observations, including punctuation.
/// Match the exact treebank identifier and spelling, without a UPOS condition:
/// Naija-NSC's `X`, for example, also occurs under DET/INTJ/NUM. Merged corpora
/// without per-sentence provenance cannot inherit a component's convention.
/// `None` retains generic absent/underscore handling without source conventions.
pub fn training_annotation_allowed(config: Option<&str>, form: &str, lemma: Option<&str>) -> bool {
    let Some(lemma) = lemma else {
        return false;
    };
    if form.is_empty() || !scorable(form, lemma) {
        return false;
    }
    let markers: &[&str] = match config.unwrap_or_default() {
        "de_hdt" => &["unknown", "-", "NULL"],
        "pcm_nsc" | "qaf_arabizi" | "say_autogramm" => &["X"],
        "cu_proiel" | "sk_snk" => &["?"],
        "myv_jr" => &["-"],
        "cop_scriptorium" => &["unknown"],
        _ => &[],
    };
    lemma == form || !markers.contains(&lemma)
}

/// Apply the same annotation admission predicate used by source ranking.
fn mask_training_placeholders(sentence: &mut Line, config: &str) {
    let Some(lemmas) = sentence.lemmas.as_mut() else {
        return;
    };
    for (form, lemma) in sentence.tokens.iter().zip(lemmas) {
        if !training_annotation_allowed(Some(config), form, lemma.as_deref()) {
            *lemma = None;
        }
    }
}

/// Streams validated JSONL sentences to one consumer, skipping blank lines.
/// `training_config` applies source annotation conventions only to train/dev;
/// a raw scoring read leaves it absent. The input descriptor remains unchanged.
fn for_each_sentence(
    path: &Path,
    training_config: Option<&str>,
    mut visit: impl FnMut(Line),
) -> Result<FileIntegrity> {
    for_each_line(path, |line_number, line| {
        if line.trim().is_empty() {
            return Ok(());
        }
        let mut parsed: Line = serde_json::from_str(line).map_err(|source| Error::Json {
            path: path.to_owned(),
            line: line_number,
            source,
        })?;
        for (field, found) in [
            ("lemmas", parsed.lemmas.as_ref().map(Vec::len)),
            ("upos", parsed.upos.as_ref().map(Vec::len)),
        ] {
            if let Some(found) = found {
                let tokens = parsed.tokens.len();
                if found != tokens {
                    return Err(Error::Alignment {
                        path: path.to_owned(),
                        line: line_number,
                        field,
                        tokens,
                        found,
                    });
                }
            }
        }
        if let Some(config) = training_config {
            mask_training_placeholders(&mut parsed, config);
        }
        visit(parsed);
        Ok(())
    })
}

/// Reads a held-out split whose sentences must remain available together.
#[cfg(test)]
pub(super) fn read_jsonl(path: &Path) -> Result<Vec<Line>> {
    read_jsonl_observed(path, None).map(|(sentences, _integrity)| sentences)
}

/// Reads a held-out split and returns the integrity of the parsed descriptor.
/// Pass the exact source config for dev calibration/refit, `None` for raw test.
pub(super) fn read_jsonl_observed(
    path: &Path,
    training_config: Option<&str>,
) -> Result<(Vec<Line>, FileIntegrity)> {
    let mut sentences = Vec::new();
    let integrity = for_each_sentence(path, training_config, |sentence| {
        sentences.push(sentence);
    })?;
    Ok((sentences, integrity))
}

/// Corpus-owned training signals kept in disjoint runtime namespaces.
pub(super) struct TrainingData {
    /// Every scorable observation used by the ordinary exact/lower/rule model.
    pub(super) pairs: Vec<Pair>,
    /// Only UD `PROPN` observations, reserved for entity-marked lookup.
    pub(super) proper_nouns: Vec<(String, String)>,
    /// Context features of the same tokens as `pairs`, when context keys
    /// were asked for.
    pub(super) context: Option<Observations>,
}

/// Extracts training observations one sentence at a time, avoiding a second
/// file read while keeping proper nouns isolated from common vocabulary.
pub(super) fn read_training_data(
    path: &Path,
    config: &str,
    mut context: Option<Observations>,
) -> Result<(TrainingData, FileIntegrity)> {
    let mut pairs = Vec::new();
    let mut proper_nouns = Vec::new();
    let integrity = for_each_sentence(path, Some(config), |sentence| {
        extend_pairs(&sentence, &mut pairs);
        extend_proper_noun_pairs(&sentence, &mut proper_nouns);
        if let Some(observations) = context.as_mut() {
            observations.add(&sentence);
        }
    })?;
    Ok((
        TrainingData {
            pairs,
            proper_nouns,
            context,
        },
        integrity,
    ))
}

/// The lemma of one token, if the sentence carries one worth using.
pub(super) fn lemma_of<'a>(sentence: &'a Line, index: usize, form: &str) -> Option<&'a str> {
    let lemma = sentence.lemmas.as_ref()?.get(index)?.as_deref()?;
    (!form.is_empty() && scorable(form, lemma)).then_some(lemma)
}

/// Appends the training observations from one sentence.
fn extend_pairs(sentence: &Line, pairs: &mut Vec<Pair>) {
    for (index, form) in sentence.tokens.iter().enumerate() {
        if let Some(lemma) = lemma_of(sentence, index, form) {
            pairs.push(Pair {
                form: form.clone(),
                lemma: lemma.to_owned(),
                sentence_initial: index == 0,
            });
        }
    }
}

/// Appends scorable `PROPN` observations without teaching common lookup.
fn extend_proper_noun_pairs(sentence: &Line, pairs: &mut Vec<(String, String)>) {
    let Some(upos) = sentence.upos.as_ref() else {
        return;
    };
    for (index, form) in sentence.tokens.iter().enumerate() {
        if upos.get(index).and_then(Option::as_deref) != Some(PROPER_NOUN) {
            continue;
        }
        if let Some(lemma) = lemma_of(sentence, index, form) {
            pairs.push((form.clone(), lemma.to_owned()));
        }
    }
}

/// Extracts the training pairs a lexicon may be learned from.
pub(super) fn pairs_from(sentences: &[Line]) -> Vec<Pair> {
    let mut pairs = Vec::new();
    for sentence in sentences {
        extend_pairs(sentence, &mut pairs);
    }
    pairs
}

/// Extracts entity-only proper-noun observations from an in-memory split.
pub(super) fn proper_noun_pairs(sentences: &[Line]) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    for sentence in sentences {
        extend_proper_noun_pairs(sentence, &mut pairs);
    }
    pairs
}

/// Flattens gold sentences into the rows scoring and tuning take.
///
/// A position the corpus left unannotated keeps its row with no lemma rather
/// than being dropped: the scorer skips it, and passing every token means
/// nobody keeps two parallel views of one split.
pub(super) fn graded_rows(sentences: &[Line]) -> Vec<Graded> {
    let mut rows = Vec::new();
    for sentence in sentences {
        for (index, form) in sentence.tokens.iter().enumerate() {
            rows.push(Graded {
                form: form.clone(),
                lemma: lemma_of(sentence, index, form).map(str::to_owned),
                sentence_initial: index == 0,
            });
        }
    }
    rows
}

/// Borrows every non-empty surface form for the probe-traffic measurement.
///
/// The held-out split already owns these strings. Returning an iterator keeps
/// traffic calibration at constant auxiliary memory instead of cloning the
/// entire split into a second `Vec<String>`.
pub(super) fn forms_of(sentences: &[Line]) -> impl Iterator<Item = &str> {
    sentences
        .iter()
        .flat_map(|sentence| sentence.tokens.iter())
        .filter(|form| !form.is_empty())
        .map(String::as_str)
}

/// Borrows surface forms from an already flattened calibration split.
pub(super) fn forms_of_graded(rows: &[Graded]) -> impl Iterator<Item = &str> {
    rows.iter().map(|row| row.form.as_str())
}

/// A deterministic form-disjoint holdout made solely from the training signal.
///
/// All observations for the same surface form stay on one side, so repeated
/// corpus tokens cannot leak an exact lookup into their own evaluation. The
/// final build still refits on the original full training vector.
pub(super) fn train_cv_split(training: &[Pair]) -> Option<(Vec<Pair>, Vec<Graded>)> {
    if training.len() < MIN_CV_FORMS || training.len() > MAX_CV_PAIRS {
        return None;
    }
    let unique: HashSet<&str> = training.iter().map(|pair| pair.form.as_str()).collect();
    if unique.len() < MIN_CV_FORMS {
        return None;
    }
    let held_out_forms: HashSet<&str> = unique
        .iter()
        .copied()
        .filter(|form| {
            Sha256::digest(form.as_bytes())
                .first()
                .copied()
                .unwrap_or_default()
                % CV_BUCKETS
                == 0
        })
        .collect();
    let retained_forms = unique.len().saturating_sub(held_out_forms.len());
    if held_out_forms.len() < MIN_HELD_OUT_FORMS || retained_forms < MIN_CV_FORMS / 2 {
        return None;
    }

    let mut retained =
        Vec::with_capacity(training.len().saturating_mul(CV_BUCKET_COUNT - 1) / CV_BUCKET_COUNT);
    let mut held_out = Vec::new();
    for pair in training {
        if held_out_forms.contains(pair.form.as_str()) {
            held_out.push(Graded {
                form: pair.form.clone(),
                lemma: Some(pair.lemma.clone()),
                sentence_initial: pair.sentence_initial,
            });
        } else {
            retained.push(pair.clone());
        }
    }
    Some((retained, held_out))
}
