use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;
use pyo3::pybacked::{PyBackedBytes, PyBackedStr};
use pyo3::types::{PyBytes, PyList, PyTuple};

use crate::builder;
use crate::lexicon;
use crate::normalization::Casing;

const MAX_SUPPORT_BATCH_ROWS: usize = 4096;
const MAX_SUPPORT_BATCH_BYTES: usize = 16 * 1024 * 1024;

/// Ceilings on what a caller may declare for the source-table reader; the
/// real limits live in the ranker.
const MAX_SOURCE_TABLE_LINE_BYTES: usize = 64 * 1024 * 1024;
const MAX_SOURCE_TABLE_BATCH_ROWS: usize = 1 << 20;
const MAX_SOURCE_TABLE_BATCH_BYTES: usize = 256 * 1024 * 1024;

/// One settled flush batch on its way back to Python: consumed bytes,
/// consumed physical lines, the accepted `form, lemma, line` rows, the blank,
/// invalid, comment and script-mismatch counts of those lines, and the sort
/// state to hand back on the next call.
type SourceTableBatch<'py> = (
    usize,
    usize,
    Bound<'py, PyList>,
    usize,
    usize,
    usize,
    usize,
    Option<String>,
);

fn validate_iso639_3(iso639_3: &str) -> PyResult<()> {
    if iso639_3.len() == 3 && iso639_3.bytes().all(|byte| byte.is_ascii_lowercase()) {
        return Ok(());
    }
    Err(pyo3::exceptions::PyValueError::new_err(
        "iso639_3 must be exactly three lowercase ASCII letters",
    ))
}

fn bounded_strings(values: &Bound<'_, PyList>, operation: &str) -> PyResult<Vec<PyBackedStr>> {
    let value_count = values.len();
    if value_count > MAX_SUPPORT_BATCH_ROWS {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{operation} batch has {value_count} rows; maximum is {MAX_SUPPORT_BATCH_ROWS}"
        )));
    }
    let mut extracted = Vec::with_capacity(value_count);
    let mut total_bytes = 0_usize;
    for value in values.iter() {
        let value = value.extract::<PyBackedStr>()?;
        total_bytes = total_bytes.checked_add(value.len()).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "{operation} batch exceeds {MAX_SUPPORT_BATCH_BYTES} UTF-8 bytes"
            ))
        })?;
        if total_bytes > MAX_SUPPORT_BATCH_BYTES {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{operation} batch exceeds {MAX_SUPPORT_BATCH_BYTES} UTF-8 bytes"
            )));
        }
        extracted.push(value);
    }
    Ok(extracted)
}

/// Admit original train/dev annotations using the builder's exact source policy.
#[pyfunction]
#[pyo3(signature = (config, rows))]
pub(super) fn _training_annotation_mask<'py>(
    py: Python<'py>,
    config: Option<&str>,
    rows: &Bound<'py, PyList>,
) -> PyResult<Bound<'py, PyBytes>> {
    if rows.len() > MAX_SUPPORT_BATCH_ROWS {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "training annotation batch exceeds {MAX_SUPPORT_BATCH_ROWS} rows"
        )));
    }
    let mut extracted = Vec::with_capacity(rows.len());
    let mut total_bytes = 0_usize;
    for row in rows.iter() {
        let tuple = row.cast::<PyTuple>()?;
        if tuple.len() != 2 {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "training annotation rows must be two-item tuples",
            ));
        }
        let form = tuple.get_item(0)?.extract::<PyBackedStr>()?;
        let lemma = tuple.get_item(1)?.extract::<Option<PyBackedStr>>()?;
        total_bytes = total_bytes
            .checked_add(form.len())
            .and_then(|value| value.checked_add(lemma.as_ref().map_or(0, |text| text.len())))
            .filter(|&value| value <= MAX_SUPPORT_BATCH_BYTES)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "training annotation batch exceeds {MAX_SUPPORT_BATCH_BYTES} UTF-8 bytes"
                ))
            })?;
        extracted.push((form, lemma));
    }
    let flags: Vec<u8> = py.detach(|| {
        extracted
            .iter()
            .map(|(form, lemma)| {
                u8::from(builder::training_annotation_allowed(
                    config,
                    form,
                    lemma.as_ref().map(|text| &**text),
                ))
            })
            .collect()
    });
    Ok(PyBytes::new(py, &flags))
}

/// Evaluate the builder's exact Unicode-script guard for one bounded batch.
///
/// One byte per row avoids allocating thousands of Python booleans.
#[pyfunction]
pub(super) fn _script_compatibility<'py>(
    py: Python<'py>,
    iso639_3: &str,
    rows: &Bound<'py, PyList>,
) -> PyResult<Bound<'py, PyBytes>> {
    validate_iso639_3(iso639_3)?;
    let row_count = rows.len();
    if row_count > MAX_SUPPORT_BATCH_ROWS {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "script compatibility batch has {row_count} rows; maximum is {MAX_SUPPORT_BATCH_ROWS}"
        )));
    }

    let mut extracted = Vec::with_capacity(row_count);
    let mut total_bytes = 0_usize;
    for row in rows.iter() {
        let tuple = row.cast::<PyTuple>()?;
        if tuple.len() != 2 {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "script compatibility rows must be two-item tuples",
            ));
        }
        let form = tuple.get_item(0)?.extract::<PyBackedStr>()?;
        let lemma = tuple.get_item(1)?.extract::<PyBackedStr>()?;
        total_bytes = total_bytes
            .checked_add(form.len())
            .and_then(|value| value.checked_add(lemma.len()))
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "script compatibility batch exceeds {MAX_SUPPORT_BATCH_BYTES} UTF-8 bytes"
                ))
            })?;
        if total_bytes > MAX_SUPPORT_BATCH_BYTES {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "script compatibility batch exceeds {MAX_SUPPORT_BATCH_BYTES} UTF-8 bytes"
            )));
        }
        extracted.push((form, lemma));
    }
    let flags: Vec<u8> = py.detach(|| {
        extracted
            .into_iter()
            .map(|(form, lemma)| {
                u8::from(crate::normalization::scripts_compatible(
                    &form, &lemma, iso639_3,
                ))
            })
            .collect()
    });
    Ok(PyBytes::new(py, &flags))
}

/// Locale-lowercase one bounded batch with the runtime's exact ICU mapping.
#[pyfunction]
pub(super) fn _locale_lower(
    py: Python<'_>,
    iso639_3: &str,
    values: &Bound<'_, PyList>,
) -> PyResult<Vec<String>> {
    validate_iso639_3(iso639_3)?;
    let extracted = bounded_strings(values, "locale lower")?;
    let casing = Casing::for_language(iso639_3);
    Ok(py.detach(|| {
        extracted
            .into_iter()
            .map(|value| {
                let normalized = crate::normalization::nfc(&value);
                casing.lowercase(&normalized).into_owned()
            })
            .collect()
    }))
}

/// The phrase key one span is looked up under, from the runtime's own code.
///
/// The offline compiler must file patterns under exactly this key, so it never
/// re-implements the locale lowercasing. `None` when the span has no key.
#[pyfunction]
pub(super) fn _phrase_span_key(
    iso639_3: &str,
    tokens: &Bound<'_, PyList>,
) -> PyResult<Option<String>> {
    validate_iso639_3(iso639_3)?;
    let extracted = bounded_strings(tokens, "phrase span key")?;
    let borrowed: Vec<&str> = extracted.iter().map(|value| &**value).collect();
    Ok(crate::probe::phrase_key(
        &borrowed,
        Casing::for_language(iso639_3),
    ))
}

/// NFC-normalize one bounded batch with the builder's pinned Unicode tables.
#[pyfunction]
pub(super) fn _normalize_nfc(py: Python<'_>, values: &Bound<'_, PyList>) -> PyResult<Vec<String>> {
    let extracted = bounded_strings(values, "NFC normalization")?;
    Ok(py.detach(|| {
        extracted
            .into_iter()
            .map(|value| crate::normalization::nfc(&value).into_owned())
            .collect()
    }))
}

/// Validate one normalized imported table, and say how many rows it holds.
///
/// The declared count stays in Python, where an oversized manifest integer is
/// still refused as a count mismatch.
///
/// Args:
///     `path`: The table to read.
///
/// Returns:
///     How many rows it holds.
#[pyfunction]
pub(super) fn _validate_normalized_rows(py: Python<'_>, path: &str) -> PyResult<usize> {
    py.detach(|| builder::validate_normalized_rows(std::path::Path::new(path)))
        .map_err(|error| build_error(&error))
}

/// Reads one flush batch of a source table out of a window of its bytes.
///
/// A pure function of the window: opening the file safely stays in Python.
/// The batch and its sort state are explicit because refusal order depends on
/// where a flush happens.
///
/// Args:
///     `window`: Table bytes, as an immutable `bytes` object so the borrow is
///         free and cannot be resized while the reader holds a slice of it.
///     `start`: Where in `window` this batch begins.
///     `path`: Table path, used for nothing but the refusal message.
///     `first_line`: One-based number of the physical line found at `start`.
///     `iso639_3`: Language whose script guard admits or rejects a rewrite.
///     `previous_form`: Sort state of an imported table, carried across calls.
///     `lemma_first`: Whether the lemma is the first column.
///     `normalized_unique`: Whether a manifest promises NFC, sorted, unique
///         forms — which also means no row may be blank, commented or skipped.
///     `require_newline`: Whether every row must end with a newline.
///     `at_eof`: Whether more bytes may still follow the window.
///     `max_line_bytes`: Longest row body accepted before the table refuses.
///     `max_batch_rows`: Rows one flush batch may hold.
///     `max_batch_bytes`: Raw form and lemma bytes one flush batch may hold.
///
/// Returns:
///     `None` when the window ended before a batch could be settled and more
///     bytes may still arrive — ask again from the same `start` with a wider
///     window. Otherwise consumed bytes, consumed physical lines, the accepted
///     `form, lemma, line` rows, the blank, invalid, comment and
///     script-mismatch counts of those lines, and the sort state to hand back.
#[pyfunction]
#[pyo3(signature = (
    window,
    start,
    path,
    first_line,
    iso639_3,
    previous_form,
    *,
    lemma_first,
    normalized_unique,
    require_newline,
    at_eof,
    max_line_bytes,
    max_batch_rows,
    max_batch_bytes,
))]
// `PyBacked*` are owned borrow handles; `FromPyObject` extracts only by value.
#[allow(
    clippy::too_many_arguments,
    clippy::fn_params_excessive_bools,
    clippy::needless_pass_by_value
)]
pub(super) fn _source_table_batch(
    py: Python<'_>,
    window: PyBackedBytes,
    start: usize,
    path: PyBackedStr,
    first_line: usize,
    iso639_3: PyBackedStr,
    previous_form: Option<PyBackedStr>,
    lemma_first: bool,
    normalized_unique: bool,
    require_newline: bool,
    at_eof: bool,
    max_line_bytes: usize,
    max_batch_rows: usize,
    max_batch_bytes: usize,
) -> PyResult<Option<SourceTableBatch<'_>>> {
    validate_iso639_3(&iso639_3)?;
    if first_line < 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "first_line must be a one-based line number",
        ));
    }
    if start > window.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "start must lie inside the source table window",
        ));
    }
    if max_line_bytes == 0 || max_line_bytes > MAX_SOURCE_TABLE_LINE_BYTES {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "max_line_bytes must be between 1 and {MAX_SOURCE_TABLE_LINE_BYTES}"
        )));
    }
    if max_batch_rows == 0 || max_batch_rows > MAX_SOURCE_TABLE_BATCH_ROWS {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "max_batch_rows must be between 1 and {MAX_SOURCE_TABLE_BATCH_ROWS}"
        )));
    }
    if max_batch_bytes == 0 || max_batch_bytes > MAX_SOURCE_TABLE_BATCH_BYTES {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "max_batch_bytes must be between 1 and {MAX_SOURCE_TABLE_BATCH_BYTES}"
        )));
    }
    let policy = builder::SourceTablePolicy {
        iso639_3: &iso639_3,
        lemma_first,
        normalized_unique,
        require_newline,
        max_line_bytes,
        max_batch_rows,
        max_batch_bytes,
    };
    let bytes: &[u8] = &window;
    let previous = previous_form.as_deref();
    let parsed = py
        .detach(|| {
            builder::next_source_table_batch(bytes, start, first_line, previous, &policy, at_eof)
        })
        .map_err(|refusal| pyo3::exceptions::PyValueError::new_err(refusal.render(&path)))?;
    let Some(batch) = parsed else {
        return Ok(None);
    };
    let rows = PyList::new(
        py,
        batch
            .rows
            .iter()
            .map(|(form, lemma, line)| (&**form, &**lemma, *line)),
    )?;
    Ok(Some((
        batch.consumed,
        batch.lines,
        rows,
        batch.blank,
        batch.invalid,
        batch.comment,
        batch.mismatch,
        batch.previous_form.map(std::borrow::Cow::into_owned),
    )))
}

/// Turns a crate error into a Python `OSError`.
pub(super) fn to_py(error: &lexicon::Error) -> PyErr {
    PyOSError::new_err(error.to_string())
}

/// Turns a build error into a Python exception: bad input is `ValueError`,
/// everything else is I/O.
pub(super) fn build_error(error: &builder::Error) -> PyErr {
    match error {
        builder::Error::Io { .. } | builder::Error::Publish(_) => {
            PyOSError::new_err(error.to_string())
        }
        builder::Error::Json { .. }
        | builder::Error::Alignment { .. }
        | builder::Error::Encoding { .. }
        | builder::Error::ImportedRow { .. }
        | builder::Error::LineTooLong { .. }
        | builder::Error::Paradigm { .. }
        | builder::Error::InputIntegrity { .. } => {
            pyo3::exceptions::PyValueError::new_err(error.to_string())
        }
    }
}
