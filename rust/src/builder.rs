//! One call per language, taking paths: read the corpora, learn, accept, write.
//!
//! Steps: pick the training signal (the treebank's train split, else the first
//! external source covering the language), learn the tables, fit thresholds
//! and choose a variant on dev (or a train-CV holdout), measure probe traffic,
//! refit on train+dev, score on test, write sources and metadata. Ties break
//! as in [`crate::train`]; nothing here may reorder an insertion.
//!
//! Inputs are streamed as strict UTF-8 with bounded lines; bad dictionary rows
//! are refused, never silently dropped.

use std::path::{Path, PathBuf};

use indexmap::IndexMap;

use crate::format::{SelectionEvidence, SelectionSample};
use crate::train::{self, Graded, Pair, Thresholds, Variant};

mod compounds;
mod context_keys;
mod corpus;
mod input;
mod lemma_adapter;
mod metadata;
mod normalized;
#[cfg(any(feature = "python", test))]
mod source_table;
mod sources;

use context_keys::Observations;
use corpus::{
    forms_of, forms_of_graded, graded_rows, pairs_from, proper_noun_pairs, read_jsonl_observed,
    train_cv_split,
};
use input::exists_nonempty;
use metadata::write_meta;
use sources::{
    TrainingSignal, entity_pairs_observed, external_pairs_observed, guarded_proper_nouns,
    training_signal_observed,
};

#[cfg(feature = "python")]
pub use corpus::training_annotation_allowed;
pub use normalized::validate_normalized_rows;

#[cfg(feature = "python")]
pub use source_table::{Policy as SourceTablePolicy, next_batch as next_source_table_batch};

/// Anything that can go wrong building one lexicon.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A source could not be read or a destination written.
    #[error("{path}: {source}")]
    Io {
        /// The file involved.
        path: PathBuf,
        /// The underlying cause.
        #[source]
        source: std::io::Error,
    },
    /// A corpus line is not the JSON the format promises.
    #[error("{path}: line {line}: {source}")]
    Json {
        /// The file involved.
        path: PathBuf,
        /// One-based line number, as an editor counts them.
        line: usize,
        /// The underlying cause.
        #[source]
        source: serde_json::Error,
    },
    /// A JSONL sentence carries annotation arrays of different lengths.
    #[error("{path}: line {line}: {field} has {found} items, but tokens has {tokens}")]
    Alignment {
        /// Corpus file involved.
        path: PathBuf,
        /// One-based source line.
        line: usize,
        /// Misaligned field.
        field: &'static str,
        /// Number of tokens.
        tokens: usize,
        /// Number of annotations.
        found: usize,
    },
    /// A textual source is not valid UTF-8.
    #[error("{path}: source is not valid UTF-8: {source}")]
    Encoding {
        /// Source file involved.
        path: PathBuf,
        /// UTF-8 decoding failure.
        #[source]
        source: std::string::FromUtf8Error,
    },
    /// One source line exceeded the defensive allocation bound.
    #[error("{path}: line {line} exceeds the {limit}-byte limit")]
    LineTooLong {
        /// Source file involved.
        path: PathBuf,
        /// One-based source line.
        line: usize,
        /// Maximum accepted bytes before the newline.
        limit: usize,
    },
    /// A normalized imported table row breaks the contract its manifest claims.
    #[error("{path}:{line}: {message}")]
    ImportedRow {
        /// Source file involved.
        path: PathBuf,
        /// One-based source line, as an editor counts them.
        line: usize,
        /// The exact invariant the row broke.
        message: &'static str,
    },
    /// An external paradigm row violates its tabular contract.
    #[error("{path}: line {line}: invalid paradigm row: {message}")]
    Paradigm {
        /// Source file involved.
        path: PathBuf,
        /// One-based source line.
        line: usize,
        /// Specific violated invariant.
        message: String,
    },
    /// An authenticated input declaration did not match the bytes consumed.
    #[error("{path}: observed input {observed:?}, expected {expected:?}")]
    InputIntegrity {
        /// Input path.
        path: PathBuf,
        /// Size and SHA-256 computed while parsing the opened descriptor.
        observed: crate::lexicon::FileIntegrity,
        /// Manifest-backed size and SHA-256 supplied by the caller.
        expected: crate::lexicon::FileIntegrity,
    },
    /// Publishing an immutable generation failed.
    #[error(transparent)]
    Publish(#[from] crate::lexicon::Error),
}

type Result<T> = std::result::Result<T, Error>;

/// One external `form -> lemma` source, best sources first.
#[derive(Debug, Clone)]
pub struct Paradigm {
    /// What the metadata records when a lexicon is learned from here.
    pub label: String,
    /// The `<iso639_3>.tsv` file itself, which may not exist.
    pub path: PathBuf,
    /// Whether the lemma is the first column, as in `UniMorph`.
    pub lemma_first: bool,
    /// Whether a verified manifest guarantees NFC-sorted unique forms.
    pub normalized_unique: bool,
    /// Authenticated size and digest, when this is a manifested artifact.
    pub expected: Option<crate::lexicon::FileIntegrity>,
}

/// Size and digest of one exact input descriptor consumed by the builder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservedInput {
    /// Input namespace: `corpus`, `paradigm`, or `entity`.
    pub role: &'static str,
    /// Split name or configured source label inside the namespace.
    pub label: String,
    /// Concrete path opened for this input; never persisted as provenance.
    pub path: PathBuf,
    /// SHA-256 and byte count computed during parsing.
    pub integrity: crate::lexicon::FileIntegrity,
}

/// Everything one language's build needs, all of it names rather than data.
#[derive(Debug, Clone)]
pub struct Spec<'a> {
    /// Directory to create and fill.
    pub destination: &'a Path,
    /// The treebank's training split, when the catalogue says it has one.
    pub train: Option<&'a Path>,
    /// The development split; may not exist.
    pub dev: Option<&'a Path>,
    /// The held-out test split; may not exist.
    pub test: Option<&'a Path>,
    /// External sources in preference order, already minus the excluded ones.
    pub paradigms: &'a [Paradigm],
    /// Curated entity sources in `surface<TAB>lemma<TAB>semantics` format,
    /// kept separate from common paradigms.
    pub entities: &'a [Paradigm],
    /// The lexicon identifier, normally a Universal Dependencies treebank.
    pub name: &'a str,
    /// Human-readable language name.
    pub language: &'a str,
    /// ISO 639-3 code of the language.
    pub iso639_3: &'a str,
    /// Span case convention from the applied acceptance policy: `verbatim`
    /// to stamp into the metadata, `None` for the fail-closed default.
    pub span_case: Option<&'a str>,
    /// Span multiword convention from the applied acceptance policy:
    /// `compose` to stamp, `None` for the fail-closed default.
    pub span_multiword: Option<&'a str>,
    /// Whether the applied entity policy opened the lowercase entity retry;
    /// stamped only when true, the fail-closed default being absence.
    pub entity_lower: bool,
    /// Compound-boundary marker the applied policy accepted for this
    /// treebank's lemmas (`#` for `fi_tdt`): external lemmas get it at the
    /// boundaries the training lemmas mark. `None` leaves them as they are.
    pub compound_marker: Option<char>,
    /// Token case convention from the applied acceptance policy: `surface`
    /// to stamp, `None` for the fail-closed default. The build itself does
    /// not read it; the runtime token path does.
    pub token_case: Option<&'a str>,
    /// Whether the applied policy accepted respelling external lemmas the
    /// way the training split spells them (`превърна` -> `превърна-(се)` for
    /// `bg_btb`). `false` leaves them as the dictionaries give them.
    pub lemma_adapter: bool,
    /// Whether the applied policy accepted context keys: rows of the exact
    /// table that let a caller's UPOS or neighbouring words pick another
    /// lemma of an ambiguous form. Learned from the same sentences as the
    /// shipped exact table; `false` writes none.
    pub context_keys: bool,
    /// Compiled phrase patterns from the applied acceptance policy, as
    /// `span key -> component-wise suffix rewrite`; empty means no table.
    pub phrases: &'a [(String, String)],
    /// Capital-bucket rule threshold from the applied acceptance policy,
    /// replacing the tuner's `Thresholds::proper_noun` before variant
    /// selection; `None` keeps the tuner's value.
    pub proper_noun_threshold: Option<f64>,
}

/// What building one lexicon produced, mirroring the pipeline's outcome row.
#[derive(Debug, Clone)]
pub struct Built {
    /// Exact immutable generation produced by this build.
    pub generation: String,
    /// Where the training signal came from.
    pub source: String,
    /// How many observations were learned from, after the refit.
    pub training_tokens: u64,
    /// The fitted general and proper-noun confidence gates.
    pub thresholds: (f64, f64),
    /// Whether a development split was there to fit them on.
    pub tuned_on_dev: bool,
    /// Which parts of the learned lexicon were shipped.
    pub variant: String,
    /// Which split chose that variant, or `None` when nothing could.
    pub selected_on: Option<&'static str>,
    /// Row counts of the written tables, in writing order.
    pub entries: Vec<(String, usize)>,
    /// Measured probe shares for `lower` and `external`, when a development
    /// split existed to measure them on.
    pub traffic: Vec<(&'static str, f64)>,
    /// How many gold tokens the lexicon was scored on.
    pub gold_tokens: Option<u64>,
    /// Exact-match accuracy on those tokens.
    pub accuracy: Option<f64>,
    /// Accuracy of copying the surface form on the same tokens.
    pub baseline: Option<f64>,
    /// Whether the lexicon is at least as good as copying.
    pub beats_baseline: Option<bool>,
    /// What the selection sample measured for every lattice candidate, when
    /// there was one. This is the receipt the shipped variant is derived from.
    pub selection: Option<SelectionEvidence>,
    /// Exact descriptor bytes consumed, for provenance binding by callers.
    pub observed_inputs: Vec<ObservedInput>,
}

struct Fit {
    tables: train::Tables,
    thresholds: Thresholds,
    variant: Variant,
    selection: Option<SelectionEvidence>,
    traffic: Vec<(&'static str, f64)>,
    tuned_on_dev: bool,
    observed_inputs: Vec<ObservedInput>,
}

struct Selection {
    thresholds: Thresholds,
    variant: Variant,
    evidence: Option<SelectionEvidence>,
    traffic: Vec<(&'static str, f64)>,
}

/// Tunes and selects one already-built calibration view, for both the dev
/// and the train-CV path.
fn select_calibration<'a>(
    calibration: &train::Tables,
    graded: &[Graded],
    forms: impl IntoIterator<Item = &'a str>,
    sample: SelectionSample,
    proper_noun_override: Option<f64>,
) -> Selection {
    let mut thresholds = train::tune(calibration, graded);
    if let Some(value) = proper_noun_override {
        thresholds.proper_noun = value;
    }
    let evidence = train::measure(calibration, thresholds, graded, sample);
    let variant = evidence
        .as_ref()
        .map_or(Variant::Full, |measured| measured.verdict().0);
    let traffic = train::traffic_iter(calibration, forms)
        .map_or_else(Vec::new, |(lower, external)| {
            vec![("lower", lower), ("external", external)]
        });
    Selection {
        thresholds,
        variant,
        evidence,
        traffic,
    }
}

/// Adds descriptor observations while rejecting a source that changed between reads.
fn extend_observed(
    destination: &mut Vec<ObservedInput>,
    additions: impl IntoIterator<Item = ObservedInput>,
) -> Result<()> {
    for addition in additions {
        if let Some(existing) = destination
            .iter()
            .find(|existing| existing.role == addition.role && existing.label == addition.label)
        {
            if existing.path != addition.path {
                return Err(Error::Io {
                    path: addition.path,
                    source: std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "one builder input label resolved to multiple paths",
                    ),
                });
            }
            if existing.integrity != addition.integrity {
                return Err(Error::InputIntegrity {
                    path: addition.path,
                    observed: addition.integrity,
                    expected: existing.integrity.clone(),
                });
            }
            continue;
        }
        destination.push(addition);
    }
    Ok(())
}

fn fit_tables(
    spec: &Spec<'_>,
    training: &mut Vec<Pair>,
    context: &mut Option<Observations>,
    external: IndexMap<String, String>,
    entities: Vec<(String, String)>,
) -> Result<Fit> {
    let mut selection = Selection {
        thresholds: Thresholds::default(),
        variant: Variant::Full,
        evidence: None,
        traffic: Vec::new(),
    };
    let mut observed_inputs = Vec::new();
    let dev = match spec.dev {
        Some(path) if exists_nonempty(path)? => Some(path),
        _ => None,
    };
    let tables = if let Some(dev) = dev {
        let calibration = train::build_for_language_with_resolved_external(
            training,
            external,
            entities,
            spec.iso639_3,
        );
        let (held_out, integrity) = read_jsonl_observed(dev, Some(spec.name))?;
        observed_inputs.push(ObservedInput {
            role: "corpus",
            label: format!("{}.dev", spec.name),
            path: dev.to_owned(),
            integrity,
        });
        let graded = graded_rows(&held_out);
        selection = select_calibration(
            &calibration,
            &graded,
            forms_of(&held_out),
            SelectionSample::Dev,
            spec.proper_noun_threshold,
        );
        // Selection is complete; development vocabulary may now improve the
        // shipped tables, while test remains score-only below.
        let additional_entities = guarded_proper_nouns(proper_noun_pairs(&held_out), spec.iso639_3);
        drop(graded);
        training.extend(pairs_from(&held_out));
        if let Some(observations) = context.as_mut() {
            for sentence in &held_out {
                observations.add(sentence);
            }
        }
        drop(held_out);
        train::refit_for_language_with_entities(training, calibration, additional_entities)
    } else if let Some((calibration_training, held_out)) = train_cv_split(training) {
        let calibration = train::build_for_language_with_resolved_external(
            &calibration_training,
            external,
            entities,
            spec.iso639_3,
        );
        selection = select_calibration(
            &calibration,
            &held_out,
            forms_of_graded(&held_out),
            SelectionSample::TrainCv,
            spec.proper_noun_threshold,
        );
        drop(calibration_training);
        drop(held_out);
        train::refit_for_language(training, calibration)
    } else {
        train::build_for_language_with_resolved_external(
            training,
            external,
            entities,
            spec.iso639_3,
        )
    };
    Ok(Fit {
        tables,
        thresholds: selection.thresholds,
        variant: selection.variant,
        selection: selection.evidence,
        traffic: selection.traffic,
        tuned_on_dev: dev.is_some(),
        observed_inputs,
    })
}

/// Builds the only valid view when no ordinary-word training signal exists.
fn fit_identity(spec: &Spec<'_>, training: &[Pair], entities: Vec<(String, String)>) -> Fit {
    Fit {
        tables: train::build_for_language_with_resolved_external(
            training,
            IndexMap::new(),
            entities,
            spec.iso639_3,
        ),
        thresholds: Thresholds::default(),
        variant: Variant::Identity,
        selection: None,
        traffic: Vec::new(),
        tuned_on_dev: false,
        observed_inputs: Vec::new(),
    }
}

/// Reads held-out scoring data only for a genuinely trained lexicon.
fn held_out_gold(
    spec: &Spec<'_>,
    identity_only: bool,
    observed_inputs: &mut Vec<ObservedInput>,
) -> Result<Option<Vec<Graded>>> {
    if identity_only {
        return Ok(None);
    }
    let Some(test) = spec.test else {
        return Ok(None);
    };
    if !exists_nonempty(test)? {
        return Ok(None);
    }
    let (sentences, integrity) = read_jsonl_observed(test, None)?;
    extend_observed(
        observed_inputs,
        [ObservedInput {
            role: "corpus",
            label: format!("{}.test", spec.name),
            path: test.to_owned(),
            integrity,
        }],
    )?;
    Ok(Some(graded_rows(&sentences)))
}

/// Learns, checks and writes the lexicon for one language.
///
/// Touches only its inputs and destination, so languages build concurrently.
/// A language with no common-word signal still gets an identity generation,
/// possibly with entity tables.
///
/// # Errors
///
/// Returns an error when a split or table cannot be read, a corpus line is
/// not valid JSON, or the destination cannot be written.
pub fn build_lexicon_required(spec: &Spec<'_>) -> Result<Built> {
    build_lexicon_with_publication(spec, true)
}

/// Builds and seals one lexicon without changing the active generation, so
/// the Python pipeline can publish it together with its manifest.
///
/// # Errors
///
/// Returns the same parsing, integrity, and publication errors as
/// [`build_lexicon_required`].
#[cfg(feature = "python")]
pub fn build_lexicon_unpublished(spec: &Spec<'_>) -> Result<Built> {
    build_lexicon_with_publication(spec, false)
}

#[allow(clippy::too_many_lines)]
fn build_lexicon_with_publication(spec: &Spec<'_>, publish: bool) -> Result<Built> {
    let (signal, mut observed_inputs) = training_signal_observed(spec)?;
    let (mut training, proper_nouns, mut context, source, paradigm_index, identity_only) =
        match signal {
            Some(TrainingSignal {
                pairs,
                proper_nouns,
                context,
                source,
                paradigm_index,
            }) => (pairs, proper_nouns, context, source, paradigm_index, false),
            None => (
                Vec::new(),
                Vec::new(),
                None,
                "identity/no-signal".to_owned(),
                None,
                true,
            ),
        };
    // Corpus proper nouns decide which single-word entity rows are admitted.
    let (mut entities, entity_inputs) = entity_pairs_observed(spec, &proper_nouns)?;
    extend_observed(&mut observed_inputs, entity_inputs)?;
    entities.extend(guarded_proper_nouns(proper_nouns, spec.iso639_3));
    let fit = if identity_only {
        // No paradigm yielded a pair; dev/test must not become a fallback.
        fit_identity(spec, &training, entities)
    } else {
        let (mut external, external_inputs) = external_pairs_observed(spec, paradigm_index)?;
        extend_observed(&mut observed_inputs, external_inputs)?;
        // Before calibration, so selection scores exactly what ships; and
        // from the training split alone, since dev joins only at the refit.
        if let Some(marker) = spec.compound_marker {
            compounds::mark_compounds(&mut external, &training, marker);
        }
        // After the marker, which the training lemmas it learns from carry;
        // and before calibration drops the train forms it learns on.
        if spec.lemma_adapter {
            lemma_adapter::adapt_external(&mut external, &training);
        }
        fit_tables(spec, &mut training, &mut context, external, entities)?
    };
    let Fit {
        mut tables,
        thresholds,
        variant,
        selection,
        traffic,
        tuned_on_dev,
        observed_inputs: dev_inputs,
    } = fit;
    let selected_on = selection.as_ref().map(|evidence| evidence.sample().name());
    extend_observed(&mut observed_inputs, dev_inputs)?;

    let gold = held_out_gold(spec, identity_only, &mut observed_inputs)?;

    // Test only scores; it never tunes or selects. An empty split measures nothing.
    let measured = gold
        .as_deref()
        .filter(|rows| !rows.is_empty())
        .and_then(|rows| train::score(&train::View::new(&tables, thresholds, variant), rows));
    let (gold_tokens, accuracy, baseline, beats_baseline) =
        measured.map_or((None, None, None, None), |score| {
            (
                Some(score.tokens),
                Some(score.accuracy),
                Some(score.baseline),
                Some(score.accuracy >= score.baseline),
            )
        });
    let training_tokens = training.len() as u64;
    drop(gold);
    drop(training);
    // Context keys go into the shipped exact table only, after scoring.
    if let Some(observations) = context {
        tables.exact.extend(observations.keys());
    }

    let writer = crate::lexicon::GenerationWriter::begin(spec.destination)?;
    let mut entries = train::write_sources(&tables, variant, thresholds, writer.directory())
        .map_err(|source| Error::Io {
            path: writer.directory().to_owned(),
            source,
        })?;
    entries.push((
        "phrases".to_owned(),
        train::write_phrase_source(spec.phrases, writer.directory()).map_err(|source| {
            Error::Io {
                path: writer.directory().join("phrases.tsv"),
                source,
            }
        })?,
    ));
    drop(tables);

    let mut built = Built {
        generation: String::new(),
        source,
        training_tokens,
        thresholds: (thresholds.general, thresholds.proper_noun),
        tuned_on_dev,
        variant: variant.name().to_owned(),
        selected_on,
        entries,
        traffic,
        gold_tokens,
        accuracy,
        baseline,
        beats_baseline,
        selection,
        observed_inputs,
    };
    write_meta(spec, &built, writer.directory(), writer.id())?;
    let (_compiled, generation) = if publish {
        writer.compile_and_publish_identified()?
    } else {
        writer.compile_and_seal()?
    };
    built.generation = generation;
    Ok(built)
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod corpus_policy_tests;
