use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crate::format::{BundleVariant, SelectionSample, SpanCase, SpanMultiword, TokenCase};
use crate::{builder, train};

use super::support::build_error;

/// The tables a build learned, kept opaque on the Rust side so they never
/// cross the boundary per scored configuration.
#[pyclass(module = "udlex", frozen, name = "Tables")]
#[derive(Debug)]
pub struct Tables {
    inner: train::Tables,
}

#[pymethods]
impl Tables {
    /// Surface form to lemma, case sensitive.
    fn exact(&self, py: Python<'_>) -> Vec<(String, String)> {
        py.detach(|| pairs(&self.inner.exact))
    }

    /// Lowercased surface form to lemma.
    fn lower(&self, py: Python<'_>) -> Vec<(String, String)> {
        py.detach(|| pairs(&self.inner.lower))
    }

    /// Form to lemma harvested from a dictionary rather than the corpus.
    fn external(&self, py: Python<'_>) -> Vec<(String, String)> {
        py.detach(|| pairs(&self.inner.external))
    }

    /// Curated named-entity forms, isolated from common vocabulary.
    fn entities(&self, py: Python<'_>) -> Vec<(String, String)> {
        py.detach(|| pairs(&self.inner.entities))
    }

    /// Case bucket and suffix, to `(cut, add, case, confidence)`.
    fn rules(&self, py: Python<'_>) -> Vec<(String, (usize, String, u8, f64))> {
        py.detach(|| {
            self.inner
                .rules
                .iter()
                .map(|(key, rule)| {
                    (
                        key.clone(),
                        (rule.cut, rule.add.clone(), rule.case as u8, rule.confidence),
                    )
                })
                .collect()
        })
    }

    /// Writes the four tables and the string table into `destination`.
    ///
    /// Args:
    ///     `destination`: Directory to create and fill.
    ///     `variant`: Which parts of the lexicon to ship.
    ///     `thresholds`: Confidence gates, baked into the rule table.
    ///
    /// Returns:
    ///     How many rows each table holds, in writing order.
    fn write_sources(
        &self,
        py: Python<'_>,
        destination: &str,
        variant: &str,
        thresholds: (f64, f64),
    ) -> PyResult<Vec<(String, usize)>> {
        let variant = variant_by_name(variant)?;
        py.detach(|| {
            train::write_sources(
                &self.inner,
                variant,
                train::Thresholds {
                    general: thresholds.0,
                    proper_noun: thresholds.1,
                },
                std::path::Path::new(destination),
            )
        })
        .map_err(|error| PyOSError::new_err(error.to_string()))
    }

    /// Share of tokens reaching the `lower` and `external` probes; measure on
    /// forms these tables never saw.
    fn traffic(&self, forms: &Bound<'_, PyList>) -> PyResult<Option<(f64, f64)>> {
        let py = forms.py();
        let forms: Vec<String> = forms.extract()?;
        Ok(py.detach(|| train::traffic(&self.inner, &forms)))
    }

    /// How many entries the four tables hold together.
    fn __len__(&self) -> usize {
        [
            self.inner.exact.len(),
            self.inner.lower.len(),
            self.inner.external.len(),
            self.inner.entities.len(),
            self.inner.entities_lower.len(),
            self.inner.rules.len(),
        ]
        .into_iter()
        .fold(0_usize, usize::saturating_add)
    }
}

/// Copies an insertion-ordered map out as a list of pairs.
fn pairs(map: &indexmap::IndexMap<String, String>) -> Vec<(String, String)> {
    map.iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

/// Resolves a variant name, refusing anything the lattice does not define.
fn variant_by_name(name: &str) -> PyResult<train::Variant> {
    train::Variant::parse(name).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("нет конфигурации {name:?}"))
    })
}

/// Turns what Python hands over into gold tokens.
fn graded(rows: Vec<(String, Option<String>, bool)>) -> Vec<train::Graded> {
    rows.into_iter()
        .map(|(form, lemma, sentence_initial)| train::Graded {
            form,
            lemma,
            sentence_initial,
        })
        .collect()
}

/// What building one lexicon produced, read-only.
#[pyclass(module = "udlex", frozen)]
#[derive(Debug)]
pub struct Outcome {
    inner: builder::Built,
}

#[pymethods]
impl Outcome {
    /// Exact immutable generation produced by this build.
    #[getter]
    fn generation(&self) -> &str {
        &self.inner.generation
    }

    /// Where the training signal came from: ``<config>.train`` for a treebank
    /// split, ``<label>/<iso639_3>`` for an external source.
    #[getter]
    fn source(&self) -> &str {
        &self.inner.source
    }

    /// How many observations were learned from, after the refit on train+dev.
    #[getter]
    const fn training_tokens(&self) -> u64 {
        self.inner.training_tokens
    }

    /// The fitted general and proper-noun confidence gates.
    #[getter]
    const fn thresholds(&self) -> (f64, f64) {
        self.inner.thresholds
    }

    /// Whether a development split was there to fit them on.
    #[getter]
    const fn tuned_on_dev(&self) -> bool {
        self.inner.tuned_on_dev
    }

    /// Which parts of the learned lexicon were shipped.
    #[getter]
    fn variant(&self) -> &str {
        &self.inner.variant
    }

    /// Which independent split chose that variant: ``"dev"``,
    /// form-disjoint ``"train-cv"``, or ``None``.
    #[getter]
    const fn selected_on(&self) -> Option<&'static str> {
        self.inner.selected_on
    }

    /// Row counts of the written tables, in writing order.
    #[getter]
    fn entries(&self) -> Vec<(String, usize)> {
        self.inner.entries.clone()
    }

    /// Measured probe shares for ``lower`` and ``external``, empty when no
    /// development split existed to measure them on.
    #[getter]
    fn traffic(&self) -> Vec<(&'static str, f64)> {
        self.inner.traffic.clone()
    }

    /// How many gold tokens the lexicon was scored on.
    #[getter]
    const fn gold_tokens(&self) -> Option<u64> {
        self.inner.gold_tokens
    }

    /// Exact-match accuracy on those tokens.
    #[getter]
    const fn accuracy(&self) -> Option<f64> {
        self.inner.accuracy
    }

    /// Accuracy of copying the surface form on the same tokens.
    #[getter]
    const fn baseline(&self) -> Option<f64> {
        self.inner.baseline
    }

    /// Whether the lexicon is at least as good as copying the surface form.
    #[getter]
    const fn beats_baseline(&self) -> Option<bool> {
        self.inner.beats_baseline
    }

    /// The lattice receipt behind ``variant``, mirroring the ``selection``
    /// block of ``meta.json``: sample name, token counts, and per-candidate
    /// rows with both profiles and the pairwise ``rivals`` counters, or
    /// ``None`` when no independent selection was possible.
    ///
    /// # Errors
    ///
    /// Returns an error only when a Python collection cannot be built.
    #[getter]
    fn selection<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(evidence) = self.inner.selection.as_ref() else {
            return Ok(None);
        };
        let receipt = PyDict::new(py);
        receipt.set_item("on", evidence.sample().name())?;
        receipt.set_item("tokens", evidence.tokens())?;
        receipt.set_item("copied", evidence.copied())?;
        let candidates = PyDict::new(py);
        for (index, candidate) in evidence.candidates().iter().enumerate() {
            let row = PyDict::new(py);
            row.set_item("hits", candidate.hits)?;
            row.set_item("discordant", candidate.discordant)?;
            if let Some(profile) = evidence.profiles().and_then(|all| all.get(index)) {
                row.set_item("ner_hits", profile.ner_hits)?;
                let rivals = PyDict::new(py);
                for (rival, count) in BundleVariant::CANDIDATES.iter().zip(profile.rivals) {
                    rivals.set_item(rival.name(), count)?;
                }
                row.set_item("rivals", rivals)?;
            }
            candidates.set_item(candidate.variant.name(), row)?;
        }
        receipt.set_item("candidates", candidates)?;
        Ok(Some(receipt))
    }

    /// Exact bytes parsed by the native builder, keyed by provenance role.
    #[getter]
    fn observed_inputs(&self) -> Vec<(&'static str, String, u64, String)> {
        self.inner
            .observed_inputs
            .iter()
            .map(|input| {
                (
                    input.role,
                    input.label.clone(),
                    input.integrity.size,
                    input.integrity.sha256.clone(),
                )
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Outcome(source='{}', variant='{}', training_tokens={})",
            self.inner.source, self.inner.variant, self.inner.training_tokens
        )
    }
}

fn expected_input_map(
    rows: Vec<(String, u64, String)>,
) -> PyResult<BTreeMap<PathBuf, crate::lexicon::FileIntegrity>> {
    let mut expected = BTreeMap::new();
    for (raw_path, size, sha256) in rows {
        if sha256.len() != 64
            || sha256
                .bytes()
                .any(|byte| !matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
        {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected input {raw_path:?} has a malformed SHA-256 digest"
            )));
        }
        let path = PathBuf::from(raw_path);
        if expected
            .insert(path.clone(), crate::lexicon::FileIntegrity { size, sha256 })
            .is_some()
        {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "duplicate expected input path {}",
                path.display()
            )));
        }
    }
    Ok(expected)
}
/// Learns the four lookup tables from annotated tokens.
///
/// Args:
///     `training`: ``(form, lemma, sentence_initial)`` per scorable token.
///     `external`: ``(form, lemma)`` from a dictionary, used only for forms
///         the corpus does not cover.
#[pyfunction]
#[pyo3(signature = (training, external, entities = Vec::new()))]
pub(super) fn build(
    py: Python<'_>,
    training: Vec<(String, String, bool)>,
    external: Vec<(String, String)>,
    entities: Vec<(String, String)>,
) -> Tables {
    py.detach(|| {
        let pairs: Vec<crate::train::Pair> = training
            .into_iter()
            .map(|(form, lemma, sentence_initial)| crate::train::Pair {
                form,
                lemma,
                sentence_initial,
            })
            .collect();
        Tables {
            inner: crate::train::build_with_entities(&pairs, external, entities),
        }
    })
}

/// Fits both confidence thresholds on a development split.
///
/// Args:
///     `tables`: What `build` learned from the train split.
///     `development`: ``(form, lemma, sentence_initial)``.
///
/// Returns:
///     The general and proper-noun thresholds.
#[pyfunction]
pub(super) fn tune(
    py: Python<'_>,
    tables: &Tables,
    development: Vec<(String, Option<String>, bool)>,
) -> (f64, f64) {
    py.detach(|| {
        let fitted = crate::train::tune(&tables.inner, &graded(development));
        (fitted.general, fitted.proper_noun)
    })
}

/// Scores the five candidate configurations and returns the one that ships.
///
/// Args:
///     `selected_on`: Which sample `gold` is — ``"dev"`` or ``"train-cv"``.
///         The acceptance rule asks a train-only holdout for more evidence
///         than a development split.
///
/// Returns:
///     ``(variant, tokens, accuracy, baseline)``, or `None` when the gold
///     data holds nothing scorable and there is nothing to decide on.
#[pyfunction]
pub(super) fn choose(
    py: Python<'_>,
    tables: &Tables,
    thresholds: (f64, f64),
    gold: Vec<(String, Option<String>, bool)>,
    selected_on: &str,
) -> PyResult<Option<(String, u64, f64, f64)>> {
    let sample = SelectionSample::parse(selected_on).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "selected_on must be \"dev\" or \"train-cv\", got {selected_on:?}"
        ))
    })?;
    Ok(py.detach(|| {
        let (variant, measured) = crate::train::choose(
            &tables.inner,
            crate::train::Thresholds {
                general: thresholds.0,
                proper_noun: thresholds.1,
            },
            &graded(gold),
            sample,
        )?;
        Some((
            variant.name().to_owned(),
            measured.tokens,
            measured.accuracy,
            measured.baseline,
        ))
    }))
}

/// Measures one configuration against gold tokens.
///
/// Returns:
///     ``(tokens, accuracy, baseline)``, or `None` when nothing is scorable.
#[pyfunction]
pub(super) fn score(
    py: Python<'_>,
    tables: &Tables,
    thresholds: (f64, f64),
    variant: &str,
    gold: Vec<(String, Option<String>, bool)>,
) -> PyResult<Option<(u64, f64, f64)>> {
    let variant = variant_by_name(variant)?;
    Ok(py.detach(|| {
        let view = crate::train::View::new(
            &tables.inner,
            crate::train::Thresholds {
                general: thresholds.0,
                proper_noun: thresholds.1,
            },
            variant,
        );
        crate::train::score(&view, &graded(gold))
            .map(|measured| (measured.tokens, measured.accuracy, measured.baseline))
    }))
}

/// Builds one language's lexicon end to end, from paths rather than data,
/// with the interpreter detached.
///
/// Args:
///     `destination`: Lexicon directory to create and fill.
///     `train`: Path to the training split, or `None` when the catalogue
///         says the treebank has none.
///     `dev`: Path to the development split; may not exist.
///     `test`: Path to the held-out test split; may not exist.
///     `paradigms`: External sources as
///         ``(label, path, lemma_first, normalized_unique)``, best first,
///         already minus the excluded ones. The final flag is true only
///         after manifest-backed table validation.
///     `entities`: Isolated entity sources with mandatory
///         ``surface<TAB>lemma<TAB>semantics`` rows.
///     `name`: The lexicon identifier, normally a treebank code.
///     `language`: Human-readable language name.
///     `iso639_3`: ISO 639-3 code.
///     `publish`: Whether to switch ``current`` immediately. The bundle
///         pipeline seals first and commits it with the Python manifest.
///
/// Returns:
///     What was produced. A language without usable common-word training data
///     receives a sealed identity generation, optionally retaining its entity
///     tables.
#[pyfunction]
#[pyo3(signature = (
    destination, *, train, dev, test, paradigms, name, language, iso639_3,
    entities = Vec::new(), expected_inputs = Vec::new(), publish = true,
    span_case = None, span_multiword = None, phrases = Vec::new(),
    proper_noun_threshold = None, entity_lower = false, compound_marker = None,
    token_case = None,
    lemma_adapter = false, context_keys = false
))]
#[allow(
    clippy::too_many_arguments,
    clippy::fn_params_excessive_bools,
    clippy::needless_pass_by_value
)]
pub(super) fn build_lexicon(
    py: Python<'_>,
    destination: &str,
    train: Option<&str>,
    dev: Option<&str>,
    test: Option<&str>,
    paradigms: Vec<(String, String, bool, bool)>,
    name: &str,
    language: &str,
    iso639_3: &str,
    entities: Vec<(String, String, bool)>,
    expected_inputs: Vec<(String, u64, String)>,
    publish: bool,
    span_case: Option<&str>,
    span_multiword: Option<&str>,
    phrases: Vec<(String, String)>,
    proper_noun_threshold: Option<f64>,
    entity_lower: bool,
    compound_marker: Option<&str>,
    token_case: Option<&str>,
    lemma_adapter: bool,
    context_keys: bool,
) -> PyResult<Outcome> {
    let compound_marker = match compound_marker.map(|marker| {
        let mut characters = marker.chars();
        (characters.next(), characters.next())
    }) {
        None => None,
        Some((Some(marker), None)) => Some(marker),
        Some(_) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "compound_marker must be one character, got {compound_marker:?}"
            )));
        }
    };
    if span_case.is_some_and(|value| SpanCase::parse(value).is_none()) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "span_case must be \"transfer\" or \"verbatim\", got {span_case:?}"
        )));
    }
    if token_case.is_some_and(|value| TokenCase::parse(value).is_none()) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "token_case must be \"keep\" or \"surface\", got {token_case:?}"
        )));
    }
    if proper_noun_threshold.is_some_and(|value| !(0.0..=2.0).contains(&value)) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "proper_noun_threshold must lie in 0.0..=2.0, got \
             {proper_noun_threshold:?}"
        )));
    }
    if span_multiword.is_some_and(|value| SpanMultiword::parse(value).is_none()) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "span_multiword must be \"identity\" or \"compose\", got {span_multiword:?}"
        )));
    }
    let expected = expected_input_map(expected_inputs)?;
    let configured: BTreeSet<PathBuf> = paradigms
        .iter()
        .map(|row| PathBuf::from(&row.1))
        .chain(entities.iter().map(|row| PathBuf::from(&row.1)))
        .collect();
    if let Some(path) = expected.keys().find(|path| !configured.contains(*path)) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "expected input path {} is not a configured paradigm or entity source",
            path.display()
        )));
    }
    py.detach(|| {
        let paradigms: Vec<crate::builder::Paradigm> = paradigms
            .into_iter()
            .map(
                |(label, path, lemma_first, normalized_unique)| crate::builder::Paradigm {
                    label,
                    expected: expected.get(Path::new(&path)).cloned(),
                    path: PathBuf::from(path),
                    lemma_first,
                    normalized_unique,
                },
            )
            .collect();
        let entities: Vec<crate::builder::Paradigm> = entities
            .into_iter()
            .map(|(label, path, lemma_first)| crate::builder::Paradigm {
                label,
                expected: expected.get(Path::new(&path)).cloned(),
                path: PathBuf::from(path),
                lemma_first,
                normalized_unique: false,
            })
            .collect();
        let spec = crate::builder::Spec {
            destination: std::path::Path::new(destination),
            train: train.map(std::path::Path::new),
            dev: dev.map(std::path::Path::new),
            test: test.map(std::path::Path::new),
            paradigms: &paradigms,
            entities: &entities,
            name,
            language,
            iso639_3,
            span_case,
            span_multiword,
            entity_lower,
            compound_marker,
            token_case,
            lemma_adapter,
            context_keys,
            phrases: &phrases,
            proper_noun_threshold,
        };
        let result = if publish {
            crate::builder::build_lexicon_required(&spec)
        } else {
            crate::builder::build_lexicon_unpublished(&spec)
        };
        result
            .map(|inner| Outcome { inner })
            .map_err(|error| build_error(&error))
    })
}
