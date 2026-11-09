//! The contract an imported analyzer table must meet before it is ingested:
//! strictly sorted, form-unique, NFC — what lets [`super::sources`] stream it.
//!
//! Refusal order, line numbers and wording are a compatibility contract; a
//! lone carriage return ends a row, as in a universal-newline reader.

use std::io::{BufRead as _, Read as _};
use std::path::Path;

use unicode_normalization::is_nfc;

use super::input::{MAX_SOURCE_LINE_BYTES, open_regular, verify_stable};

use super::{Error, Result};

/// One line plus the terminator and one byte of overrun, so a too-long line is
/// seen as too long rather than truncated into a valid one.
const REFILL_LIMIT_BYTES: u64 = MAX_SOURCE_LINE_BYTES as u64 + 2;

const READ_BUFFER_BYTES: usize = 1 << 20;

/// One refused row.
fn refuse(path: &Path, line: usize, message: &'static str) -> Error {
    Error::ImportedRow {
        path: path.to_owned(),
        line,
        message,
    }
}

/// Checks one row's contract and remembers its form for the ordering test;
/// `previous` starts empty, below every non-empty form.
fn check_row(path: &Path, line: usize, body: &str, previous: &mut String) -> Result<()> {
    if body.len() > MAX_SOURCE_LINE_BYTES {
        return Err(Error::LineTooLong {
            path: path.to_owned(),
            line,
            limit: MAX_SOURCE_LINE_BYTES,
        });
    }
    let mut columns = body.split('\t');
    let (Some(form), Some(lemma), None) = (columns.next(), columns.next(), columns.next()) else {
        return Err(refuse(path, line, "expected non-empty form<TAB>lemma"));
    };
    if form.is_empty() || lemma.is_empty() {
        return Err(refuse(path, line, "expected non-empty form<TAB>lemma"));
    }
    if !is_nfc(form) || !is_nfc(lemma) {
        return Err(refuse(path, line, "table text must be NFC-normalized"));
    }
    if form.chars().any(char::is_control) || lemma.chars().any(char::is_control) {
        return Err(refuse(path, line, "table contains a control character"));
    }
    if form <= previous.as_str() {
        return Err(refuse(
            path,
            line,
            "forms must be strictly sorted and unique",
        ));
    }
    previous.clear();
    previous.push_str(form);
    Ok(())
}

/// Checks one refill (up to one line feed): UTF-8 first, then a stray CR,
/// which would end a row without a newline, then the row itself.
fn check_refill(path: &Path, line: usize, refill: &[u8], previous: &mut String) -> Result<()> {
    // An unterminated refill past the cap is an overlong row, not a missing newline.
    if refill.len() > MAX_SOURCE_LINE_BYTES + 1 && !refill.ends_with(b"\n") {
        return Err(Error::LineTooLong {
            path: path.to_owned(),
            line,
            limit: MAX_SOURCE_LINE_BYTES,
        });
    }
    let Ok(text) = std::str::from_utf8(refill) else {
        return Err(refuse(path, line, "table is not valid UTF-8"));
    };
    let Some(body) = text.strip_suffix('\n') else {
        return Err(refuse(path, line, "final row must end with a newline"));
    };
    let body = body.strip_suffix('\r').unwrap_or(body);
    if body.contains('\r') {
        return Err(refuse(path, line, "final row must end with a newline"));
    }
    check_row(path, line, body, previous)
}

/// Validates one normalized imported table and returns how many rows it holds:
/// two non-empty newline-terminated NFC columns without control characters,
/// forms strictly ascending.
///
/// # Errors
///
/// [`Error::Io`] when the table cannot be opened or read, or moved while it
/// was being read; [`Error::LineTooLong`] past the shared allocation bound;
/// [`Error::ImportedRow`] naming the first row that breaks the contract.
pub fn validate_normalized_rows(path: &Path) -> Result<usize> {
    let (file, descriptor_before) = open_regular(path)?;
    let mut reader = std::io::BufReader::with_capacity(READ_BUFFER_BYTES, file);
    let mut refill = Vec::new();
    let mut previous = String::new();
    let mut line = 0_usize;
    let mut size = 0_u64;
    loop {
        refill.clear();
        let read = (&mut reader)
            .take(REFILL_LIMIT_BYTES)
            .read_until(b'\n', &mut refill)
            .map_err(|source| Error::Io {
                path: path.to_owned(),
                source,
            })?;
        if read == 0 {
            break;
        }
        size = size
            .checked_add(u64::try_from(read).unwrap_or(u64::MAX))
            .ok_or_else(|| Error::Io {
                path: path.to_owned(),
                source: std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "imported table length overflowed u64",
                ),
            })?;
        line = line.saturating_add(1);
        check_refill(path, line, &refill, &mut previous)?;
    }
    verify_stable(path, &descriptor_before, reader.get_ref(), size)?;
    Ok(line)
}
