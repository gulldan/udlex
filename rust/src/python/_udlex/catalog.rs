use pyo3::prelude::*;

use crate::lexicon;

use super::support::to_py;

/// One language a bundle can lemmatise: a read-only view of its metadata.
#[pyclass(module = "udlex", frozen)]
#[derive(Debug)]
pub struct Language {
    inner: lexicon::Language,
}

#[pymethods]
impl Language {
    /// ISO 639-3 code, the name to ask for this language by.
    #[getter]
    fn iso639_3(&self) -> &str {
        &self.inner.iso639_3
    }

    /// Human-readable language name.
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// The lexicon backing it, normally a Universal Dependencies treebank.
    #[getter]
    fn lexicon(&self) -> &str {
        &self.inner.lexicon
    }

    /// Which parts of the learned lexicon the bundle ships.
    #[getter]
    fn variant(&self) -> &str {
        &self.inner.variant
    }

    /// How many annotated tokens it was learned from.
    #[getter]
    const fn training_tokens(&self) -> u64 {
        self.inner.training_tokens
    }

    /// How many form-to-lemma rows its lookup tables hold.
    #[getter]
    const fn entries(&self) -> u64 {
        self.inner.entries
    }

    /// Exact-match accuracy on the held-out test split, when one existed.
    #[getter]
    const fn accuracy(&self) -> Option<f64> {
        self.inner.accuracy
    }

    /// What copying the surface form scores on the same tokens.
    #[getter]
    const fn baseline(&self) -> Option<f64> {
        self.inner.baseline
    }

    /// Whether held-out scoring matched or exceeded copying, if measured.
    #[getter]
    const fn is_useful(&self) -> Option<bool> {
        self.inner.is_useful
    }

    /// The sample that chose the shipped variant: `dev` or `train-cv`.
    #[getter]
    fn selected_on(&self) -> Option<&str> {
        self.inner.selected_on.as_deref()
    }

    /// Accuracy of the shipped variant on its selection sample, before refit.
    #[getter]
    const fn selection_accuracy(&self) -> Option<f64> {
        self.inner.selection_accuracy
    }

    /// What copying scored on the selection sample.
    #[getter]
    const fn selection_baseline(&self) -> Option<f64> {
        self.inner.selection_baseline
    }

    /// How much accuracy the lexicon adds over copying the surface form;
    /// `None` when either side was never measured.
    #[getter]
    fn gain(&self) -> Option<f64> {
        self.inner.gain()
    }

    fn __repr__(&self) -> String {
        format!(
            "Language(iso639_3='{}', name='{}', lexicon='{}')",
            self.inner.iso639_3, self.inner.name, self.inner.lexicon
        )
    }
}

/// What compiling one lexicon directory did.
#[pyclass(module = "udlex", frozen)]
#[derive(Debug)]
pub struct Compiled {
    inner: lexicon::Compiled,
}

#[pymethods]
impl Compiled {
    /// How many lookup tables were written.
    #[getter]
    const fn tables(&self) -> usize {
        self.inner.tables
    }

    /// Whether the builder's sources were there to compile from.
    #[getter(from_sources)]
    const fn sources_used(&self) -> bool {
        self.inner.from_sources
    }

    fn __repr__(&self) -> String {
        format!(
            "Compiled(tables={}, from_sources={})",
            self.inner.tables,
            if self.inner.from_sources {
                "True"
            } else {
                "False"
            }
        )
    }
}
/// Compiles the sorted TSV tables in ``path`` into lookup tables.
///
/// The codec choice is recorded in the metadata; a directory without sources
/// is already compiled and left alone.
///
/// Args:
///     `path`: A lexicon directory with retained source tables.
///
/// Returns:
///     What compiling did.
#[pyfunction]
pub(super) fn compile_bundle(py: Python<'_>, path: &str) -> PyResult<Compiled> {
    Ok(Compiled {
        inner: py
            .detach(|| crate::lexicon::compile(path))
            .map_err(|error| to_py(&error))?,
    })
}

/// Atomically republishes a compiled bundle without builder sources.
#[pyfunction]
pub(super) fn prune_bundle(py: Python<'_>, path: &str) -> PyResult<bool> {
    py.detach(|| crate::lexicon::prune(path))
        .map_err(|error| to_py(&error))
}

/// Lists the languages a bundle can lemmatise, by ISO 639-3 code.
///
/// Args:
///     `bundle`: A bundle directory.
#[pyfunction]
pub(super) fn languages(py: Python<'_>, bundle: &str) -> PyResult<Vec<Language>> {
    Ok(py
        .detach(|| crate::lexicon::languages(bundle))
        .map_err(|error| to_py(&error))?
        .into_iter()
        .map(|inner| Language { inner })
        .collect())
}
