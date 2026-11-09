//! The source-order ranker's TSV reader: a pure function of a byte window,
//! one flush batch per call; Python opens the file.
//!
//! Refusal order per row: invalid UTF-8, a control character (not tab), a
//! missing second column, an empty column (skipped and counted in a legacy
//! table, refused in an imported one), a whitespace-only column. `# ` comments
//! count only in legacy tables, after decoding. Extra columns are ignored. A
//! CR inside a row is a control character here, unlike in [`super::sources`].
//!
//! The batch boundary is observable (a flush refusal follows a decode refusal
//! of the row that triggered it), so batch and sort state are explicit.

use std::borrow::Cow;

use crate::normalization::{nfc, scripts_compatible};

/// One refused row; the caller adds the path with [`Refusal::render`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal {
    /// One-based physical line number, as an editor counts them.
    pub line: usize,
    /// The message text that follows `path:line: `.
    pub message: String,
}

impl Refusal {
    fn new(line: usize, message: impl Into<String>) -> Self {
        Self {
            line,
            message: message.into(),
        }
    }

    /// Renders the refusal the way the ranker's `ValueError` reads.
    #[must_use]
    pub fn render(&self, path: &str) -> String {
        format!("{path}:{}: {}", self.line, self.message)
    }
}

/// The ranker's constants and per-source flags, passed in from Python, the
/// one place they are written down.
#[derive(Debug, Clone, Copy)]
pub struct Policy<'a> {
    /// Language whose script guard decides which rewrites are admissible.
    pub iso639_3: &'a str,
    /// Whether the lemma is the first column, as in `UniMorph`.
    pub lemma_first: bool,
    /// Whether a verified manifest promises NFC, sorted and unique forms —
    /// which also means no row may be blank, commented or skipped.
    pub normalized_unique: bool,
    /// Whether every row must end with a newline, as an authenticated table's
    /// must.
    pub require_newline: bool,
    /// Longest row body the reader accepts before refusing the table.
    pub max_line_bytes: usize,
    /// Rows one flush batch may hold.
    pub max_batch_rows: usize,
    /// Raw form and lemma bytes one flush batch may hold.
    pub max_batch_bytes: usize,
}

/// Everything one flush batch of the reader produced; rows borrow the window
/// when already NFC.
#[derive(Debug, Default)]
pub struct Batch<'w> {
    /// Window bytes this batch consumed, counted from the caller's `start`.
    pub consumed: usize,
    /// Physical lines those bytes held.
    pub lines: usize,
    /// Accepted `form, lemma, line` triples, in table order and NFC.
    pub rows: Vec<(Cow<'w, str>, Cow<'w, str>, usize)>,
    /// Consumed lines that were empty.
    pub blank: usize,
    /// Consumed lines whose first two columns were not both filled.
    pub invalid: usize,
    /// Consumed lines that were `# ` comments.
    pub comment: usize,
    /// Consumed lines the builder's script guard rejected.
    pub mismatch: usize,
    /// Ordering state to hand back on the next call.
    pub previous_form: Option<Cow<'w, str>>,
}

/// One physical line of the window, or why there is not one.
#[derive(Debug)]
enum Physical<'w> {
    /// A row body, with its terminator and one trailing carriage return
    /// already removed, whether it was newline-terminated, and where it ends.
    Body {
        body: &'w [u8],
        terminated: bool,
        end: usize,
    },
    /// No complete line yet, and the file may still supply one.
    Short,
    /// Nothing left to read.
    Spent,
}

/// One pending row after normalization, beside the text it came from.
struct Normalized<'w> {
    form: Cow<'w, str>,
    lemma: Cow<'w, str>,
    raw_form: &'w str,
    raw_lemma: &'w str,
    line: usize,
}

/// Where the batch may still be cut, and what had been counted by then: a
/// batch ends at its last admitted row, and later lines belong to the next call.
#[derive(Debug, Clone, Copy)]
struct Checkpoint {
    end: usize,
    lines: usize,
    blank: usize,
    invalid: usize,
    comment: usize,
}

/// Whether one valid-UTF-8 body holds a `Cc` character other than the tab.
///
/// On bytes: in valid UTF-8, C0 controls are bytes below 0x20, DEL is 0x7F
/// and C1 controls are `0xC2` followed by `0x80..=0x9F`.
fn has_control(body: &[u8]) -> bool {
    let mut previous = 0_u8;
    for &byte in body {
        if (byte < 0x20 && byte != b'\t') || byte == 0x7f || (previous == 0xc2 && byte <= 0x9f) {
            return true;
        }
        previous = byte;
    }
    false
}

/// Whether one non-empty column is what `str.strip()` would empty; Python's
/// extra whitespace (`U+001C..=U+001F`) is `Cc` and already refused.
fn blank_column(column: &str) -> bool {
    column.chars().all(char::is_whitespace)
}

/// Decodes one row body and refuses any non-tab control character.
fn decode_source_text(body: &[u8], line: usize) -> Result<&str, Refusal> {
    let text = std::str::from_utf8(body)
        .map_err(|error| Refusal::new(line, format!("invalid UTF-8: {error}")))?;
    if has_control(body) {
        return Err(Refusal::new(line, "TSV row contains a control character"));
    }
    Ok(text)
}

/// Splits the two leading columns, or says why the row cannot supply them;
/// `Ok(None)` is the legacy tolerance for an empty column.
fn decode_columns(
    text: &str,
    line: usize,
    lemma_first: bool,
    skip_empty_fields: bool,
) -> Result<Option<(&str, &str)>, Refusal> {
    let mut columns = text.split('\t');
    let (Some(first), Some(second)) = (columns.next(), columns.next()) else {
        return Err(Refusal::new(
            line,
            "expected at least two tab-separated columns",
        ));
    };
    if first.is_empty() || second.is_empty() {
        if skip_empty_fields {
            return Ok(None);
        }
        return Err(Refusal::new(
            line,
            "the first two columns must be non-empty",
        ));
    }
    if blank_column(first) || blank_column(second) {
        return Err(Refusal::new(
            line,
            "the first two columns must not be whitespace-only",
        ));
    }
    let (lemma, form) = if lemma_first {
        (first, second)
    } else {
        (second, first)
    };
    Ok(Some((form, lemma)))
}

/// Cuts one physical line out of the window, scanning `max_line_bytes + 2`
/// bytes so an overlong row is refused rather than cut into a valid one.
fn next_physical(
    window: &[u8],
    cursor: usize,
    line: usize,
    max_line_bytes: usize,
    at_eof: bool,
) -> Result<Physical<'_>, Refusal> {
    let Some(rest) = window.get(cursor..) else {
        return Ok(Physical::Spent);
    };
    if rest.is_empty() {
        return Ok(if at_eof {
            Physical::Spent
        } else {
            Physical::Short
        });
    }
    let too_long = || Refusal::new(line, format!("row exceeds {max_line_bytes} bytes"));
    let limit = max_line_bytes.saturating_add(2);
    let scan = rest.get(..limit).unwrap_or(rest);
    let raw = match memchr::memchr(b'\n', scan) {
        Some(index) => {
            let length = index.saturating_add(1);
            if length > max_line_bytes.saturating_add(1) {
                return Err(too_long());
            }
            scan.get(..length).unwrap_or(scan)
        }
        None if scan.len() >= limit => return Err(too_long()),
        None if !at_eof => return Ok(Physical::Short),
        None if scan.len() > max_line_bytes => return Err(too_long()),
        None => scan,
    };
    let end = cursor.saturating_add(raw.len());
    let (body, terminated) = raw
        .strip_suffix(b"\n")
        .map_or((raw, false), |body| (body, true));
    let body = body.strip_suffix(b"\r").unwrap_or(body);
    Ok(Physical::Body {
        body,
        terminated,
        end,
    })
}

/// The reader's state for one call: one flush batch and the counters it fills.
struct Reader<'w> {
    first_line: usize,
    cursor: usize,
    line: usize,
    blank: usize,
    invalid: usize,
    comment: usize,
    mismatch: usize,
    pending: Vec<(&'w str, &'w str, usize)>,
    pending_bytes: usize,
    rows: Vec<(Cow<'w, str>, Cow<'w, str>, usize)>,
    previous: Option<Cow<'w, str>>,
    checkpoint: Checkpoint,
}

impl<'w> Reader<'w> {
    fn new(start: usize, first_line: usize, previous_form: Option<&str>) -> Self {
        Self {
            first_line,
            cursor: start,
            line: first_line,
            blank: 0,
            invalid: 0,
            comment: 0,
            mismatch: 0,
            pending: Vec::new(),
            pending_bytes: 0,
            rows: Vec::new(),
            previous: previous_form.map(|value| Cow::Owned(value.to_owned())),
            checkpoint: Checkpoint {
                end: start,
                lines: 0,
                blank: 0,
                invalid: 0,
                comment: 0,
            },
        }
    }

    /// Classifies one row body: a pair to rank, or a counted skip.
    fn classify(
        &mut self,
        body: &'w [u8],
        policy: &Policy<'_>,
    ) -> Result<Option<(&'w str, &'w str)>, Refusal> {
        let line = self.line;
        if body.is_empty() && !policy.normalized_unique {
            self.blank = self.blank.saturating_add(1);
            return Ok(None);
        }
        if !policy.normalized_unique && body.starts_with(b"# ") {
            // A corrupt comment still refuses the table.
            decode_source_text(body, line)?;
            self.comment = self.comment.saturating_add(1);
            return Ok(None);
        }
        let text = decode_source_text(body, line)?;
        let Some(pair) = decode_columns(text, line, policy.lemma_first, !policy.normalized_unique)?
        else {
            self.invalid = self.invalid.saturating_add(1);
            return Ok(None);
        };
        Ok(Some(pair))
    }

    const fn advance(&mut self, end: usize) {
        self.cursor = end;
        self.line = self.line.saturating_add(1);
    }

    /// Admits one decoded row to the pending batch and moves the cut with it.
    fn push(&mut self, form: &'w str, lemma: &'w str, end: usize) {
        self.pending.push((form, lemma, self.line));
        self.pending_bytes = self
            .pending_bytes
            .saturating_add(form.len().saturating_add(lemma.len()));
        self.advance(end);
        self.checkpoint = Checkpoint {
            end,
            lines: self.line.saturating_sub(self.first_line),
            blank: self.blank,
            invalid: self.invalid,
            comment: self.comment,
        };
    }

    /// Whether admitting one more row would break the ancestor's batch bounds.
    const fn batch_is_full(&self, row_bytes: usize, policy: &Policy<'_>) -> bool {
        !self.pending.is_empty()
            && (self.pending.len() >= policy.max_batch_rows
                || self.pending_bytes.saturating_add(row_bytes) > policy.max_batch_bytes)
    }

    /// Normalizes, checks and script-guards the pending rows.
    fn flush(&mut self, policy: &Policy<'_>) -> Result<(), Refusal> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let mut normalized = Vec::with_capacity(self.pending.len());
        for &(raw_form, raw_lemma, line) in &self.pending {
            normalized.push(Normalized {
                form: nfc(raw_form),
                lemma: nfc(raw_lemma),
                raw_form,
                raw_lemma,
                line,
            });
        }
        if policy.normalized_unique {
            for row in &normalized {
                if &*row.form != row.raw_form || &*row.lemma != row.raw_lemma {
                    return Err(Refusal::new(
                        row.line,
                        "imported table must be NFC-normalized",
                    ));
                }
                if self
                    .previous
                    .as_deref()
                    .is_some_and(|previous| &*row.form <= previous)
                {
                    return Err(Refusal::new(
                        row.line,
                        "imported forms must be strictly sorted and unique",
                    ));
                }
                self.previous = Some(row.form.clone());
            }
        }
        for row in normalized {
            if scripts_compatible(&row.form, &row.lemma, policy.iso639_3) {
                self.rows.push((row.form, row.lemma, row.line));
            } else {
                self.mismatch = self.mismatch.saturating_add(1);
            }
        }
        self.pending.clear();
        self.pending_bytes = 0;
        Ok(())
    }

    /// Everything read so far, up to the last complete line.
    fn finish(&mut self, start: usize) -> Batch<'w> {
        Batch {
            consumed: self.cursor.saturating_sub(start),
            lines: self.line.saturating_sub(self.first_line),
            rows: std::mem::take(&mut self.rows),
            blank: self.blank,
            invalid: self.invalid,
            comment: self.comment,
            mismatch: self.mismatch,
            previous_form: self.previous.take(),
        }
    }

    /// Everything up to the last row of the batch that has just been flushed.
    fn finish_at_checkpoint(&mut self, start: usize) -> Batch<'w> {
        Batch {
            consumed: self.checkpoint.end.saturating_sub(start),
            lines: self.checkpoint.lines,
            rows: std::mem::take(&mut self.rows),
            blank: self.checkpoint.blank,
            invalid: self.checkpoint.invalid,
            comment: self.checkpoint.comment,
            mismatch: self.mismatch,
            previous_form: self.previous.take(),
        }
    }

    /// Answers a window that ran out before the batch could be settled.
    ///
    /// Pending rows are never cut here (the answer would depend on block size),
    /// so the caller widens the window; with nothing pending the lines read so
    /// far are consumed, so a comment-only table does not grow the window.
    fn stall(&mut self, start: usize, policy: &Policy<'_>) -> Result<Option<Batch<'w>>, Refusal> {
        if self.line == self.first_line || !self.pending.is_empty() {
            return Ok(None);
        }
        self.flush(policy)?;
        Ok(Some(self.finish(start)))
    }
}

/// Reads one flush batch of a source table out of a window of its bytes.
///
/// `start` is where in `window` this batch begins and `first_line` is the
/// one-based number of the physical line found there; `previous_form` carries
/// the sort state of an imported table across the call. `at_eof` says that no
/// further bytes exist, which is what turns an unterminated final line into a
/// row rather than into a request for more.
///
/// Returns `Ok(None)` when the window ends before a batch could be settled and
/// more bytes may arrive; the caller widens it and retries from `start`.
/// Otherwise the caller resumes past the consumed bytes and lines.
///
/// # Errors
///
/// [`Refusal`] for any row that breaks the format.
pub fn next_batch<'w>(
    window: &'w [u8],
    start: usize,
    first_line: usize,
    previous_form: Option<&str>,
    policy: &Policy<'_>,
    at_eof: bool,
) -> Result<Option<Batch<'w>>, Refusal> {
    let mut reader = Reader::new(start, first_line, previous_form);
    loop {
        match next_physical(
            window,
            reader.cursor,
            reader.line,
            policy.max_line_bytes,
            at_eof,
        )? {
            Physical::Spent => {
                reader.flush(policy)?;
                return Ok(Some(reader.finish(start)));
            }
            Physical::Short => return reader.stall(start, policy),
            Physical::Body {
                body,
                terminated,
                end,
            } => {
                if policy.require_newline && !terminated {
                    return Err(Refusal::new(
                        reader.line,
                        "imported table row must end with a newline",
                    ));
                }
                let Some((form, lemma)) = reader.classify(body, policy)? else {
                    reader.advance(end);
                    continue;
                };
                // Decode refusals of this row outrank the pending batch's.
                let row_bytes = form.len().saturating_add(lemma.len());
                if reader.batch_is_full(row_bytes, policy) {
                    reader.flush(policy)?;
                    return Ok(Some(reader.finish_at_checkpoint(start)));
                }
                reader.push(form, lemma, end);
            }
        }
    }
}
