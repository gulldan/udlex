use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use super::support::to_py;

fn parse_match_mode(value: &str) -> PyResult<crate::UserMatch> {
    crate::UserMatch::parse(value).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "match must be \"exact\" or \"lower\", got {value:?}"
        ))
    })
}

fn update_tuple(
    update: crate::UserDictionaryUpdate,
) -> (bool, Option<String>, u64) {
    (update.changed, update.generation, update.entries)
}

#[pyfunction]
#[pyo3(signature = (root, iso639_3, actor, reason))]
pub(super) fn _user_dictionary_init(
    py: Python<'_>,
    root: &str,
    iso639_3: &str,
    actor: &str,
    reason: &str,
) -> PyResult<(bool, Option<String>, u64)> {
    py.detach(|| {
        let editor = crate::UserDictionaryEditor::new(root, iso639_3)?;
        editor.init(actor, reason)
    })
    .map(update_tuple)
    .map_err(|error| to_py(&error))
}

#[pyfunction]
#[pyo3(signature = (root, iso639_3, form, lemma, match_mode, replace, actor, reason))]
#[allow(clippy::too_many_arguments)]
pub(super) fn _user_dictionary_set(
    py: Python<'_>,
    root: &str,
    iso639_3: &str,
    form: &str,
    lemma: &str,
    match_mode: &str,
    replace: bool,
    actor: &str,
    reason: &str,
) -> PyResult<(bool, Option<String>, u64)> {
    let mode = parse_match_mode(match_mode)?;
    py.detach(|| {
        let editor = crate::UserDictionaryEditor::new(root, iso639_3)?;
        editor.set(form, lemma, mode, replace, actor, reason)
    })
    .map(update_tuple)
    .map_err(|error| to_py(&error))
}

#[pyfunction]
#[pyo3(signature = (root, iso639_3, form, match_mode, actor, reason))]
pub(super) fn _user_dictionary_remove(
    py: Python<'_>,
    root: &str,
    iso639_3: &str,
    form: &str,
    match_mode: &str,
    actor: &str,
    reason: &str,
) -> PyResult<(bool, Option<String>, u64)> {
    let mode = parse_match_mode(match_mode)?;
    py.detach(|| {
        let editor = crate::UserDictionaryEditor::new(root, iso639_3)?;
        editor.remove(form, mode, actor, reason)
    })
    .map(update_tuple)
    .map_err(|error| to_py(&error))
}

#[pyfunction]
#[pyo3(signature = (root, iso639_3, path, replace, actor, reason))]
pub(super) fn _user_dictionary_import(
    py: Python<'_>,
    root: &str,
    iso639_3: &str,
    path: &str,
    replace: bool,
    actor: &str,
    reason: &str,
) -> PyResult<(bool, Option<String>, u64)> {
    py.detach(|| {
        let editor = crate::UserDictionaryEditor::new(root, iso639_3)?;
        editor.import(path, replace, actor, reason)
    })
    .map(update_tuple)
    .map_err(|error| to_py(&error))
}

#[pyfunction]
#[pyo3(signature = (root, iso639_3, generation, actor, reason))]
pub(super) fn _user_dictionary_rollback(
    py: Python<'_>,
    root: &str,
    iso639_3: &str,
    generation: &str,
    actor: &str,
    reason: &str,
) -> PyResult<(bool, Option<String>, u64)> {
    py.detach(|| {
        let editor = crate::UserDictionaryEditor::new(root, iso639_3)?;
        editor.rollback(generation, actor, reason)
    })
    .map(update_tuple)
    .map_err(|error| to_py(&error))
}

#[pyfunction]
pub(super) fn _user_dictionary_entries(
    py: Python<'_>,
    root: &str,
    iso639_3: &str,
) -> PyResult<Vec<(String, String, String)>> {
    py.detach(|| crate::UserDictionary::open_from_bundle(root, iso639_3))
        .map_err(|error| to_py(&error))?
        .map_or_else(
            || Ok(Vec::new()),
            |dictionary| {
                dictionary.entries().map(|entries| {
                    entries
                        .into_iter()
                        .map(|entry| {
                            (
                                entry.form,
                                entry.lemma,
                                entry.match_mode.name().to_owned(),
                            )
                        })
                        .collect()
                })
            },
        )
        .map_err(|error| to_py(&error))
}

#[pyfunction]
pub(super) fn _user_dictionary_history<'py>(
    py: Python<'py>,
    root: &str,
    iso639_3: &str,
) -> PyResult<Bound<'py, PyList>> {
    let receipts = py
        .detach(|| {
            crate::UserDictionaryEditor::new(root, iso639_3)?.history()
        })
        .map_err(|error| to_py(&error))?;
    let mut rows = Vec::with_capacity(receipts.len());
    for receipt in receipts {
        let row = PyDict::new(py);
        row.set_item("schema", receipt.schema)?;
        row.set_item("generation", receipt.generation)?;
        row.set_item("iso639_3", receipt.iso639_3)?;
        row.set_item("operation", receipt.operation)?;
        row.set_item("previous_generation", receipt.previous_generation)?;
        row.set_item("rollback_from", receipt.rollback_from)?;
        row.set_item("actor", receipt.actor)?;
        row.set_item("reason", receipt.reason)?;
        row.set_item("timestamp_unix_ms", receipt.timestamp_unix_ms)?;
        let mut changes = Vec::with_capacity(receipt.changes.len());
        for change in receipt.changes {
            let item = PyDict::new(py);
            item.set_item("key", change.key)?;
            item.set_item("match", change.match_mode)?;
            item.set_item("old", change.old)?;
            item.set_item("new", change.new)?;
            changes.push(item.unbind());
        }
        row.set_item("changes", PyList::new(py, changes)?)?;
        rows.push(row.unbind());
    }
    PyList::new(py, rows)
}

#[pyfunction]
pub(super) fn _user_dictionary_verify(
    py: Python<'_>,
    root: &str,
) -> PyResult<(u64, u64, u64)> {
    py.detach(|| crate::verify_user_bundle(root))
        .map(|summary| {
            (
                summary.languages,
                summary.generations,
                summary.active_entries,
            )
        })
        .map_err(|error| to_py(&error))
}
