//! Compact source representations and source-only rewrites.

use std::fs::{File, Metadata};
use std::io::{BufRead as _, BufReader, BufWriter, Read as _, Write as _};
use std::path::Path;

use super::super::{Error, Result, line_offsets};
use super::writer::write_file;
use crate::format::LookupTableKind;
use crate::secure_fs::{OpenRegularError, StabilityError};

/// The low 48 bits of a packed rule are its string identifier.
const RULE_STRING_MASK: u64 = (1_u64 << 48) - 1;

/// A string table with one allocation for text and one `u32` per boundary,
/// the layout the runtime maps from `strings.txt` plus `strings.idx`.
pub(super) struct StringTable {
    blob: String,
    offsets: Vec<u32>,
}

impl StringTable {
    /// Canonicalises a legacy table while preserving every numeric ID.
    ///
    /// A non-NFC table is re-read after the raw blob is dropped, so the two
    /// whole-file strings are never live together.
    pub(super) fn normalise(source: &Path, destination: &Path) -> Result<Self> {
        let raw = read_stable_string(source)?;
        validate_string_blob(source, &raw)?;
        let canonical = (raw.is_empty() || raw.ends_with('\n'))
            && raw.split_terminator('\n').all(|value| {
                matches!(
                    crate::normalization::nfc(value),
                    std::borrow::Cow::Borrowed(_)
                )
            });
        if canonical {
            let offsets = offsets(source, raw.as_bytes())?;
            if source != destination {
                write_file(destination, raw.as_bytes())?;
            }
            return Ok(Self { blob: raw, offsets });
        }

        drop(raw);
        let mut blob = String::new();
        for_each_line(source, |_, line| {
            let value = line.strip_suffix('\n').unwrap_or(line);
            validate_string_blob(source, value)?;
            blob.push_str(&crate::normalization::nfc(value));
            blob.push('\n');
            Ok(())
        })?;
        let offsets = offsets(source, blob.as_bytes())?;
        write_file(destination, blob.as_bytes())?;
        Ok(Self { blob, offsets })
    }

    pub(super) const fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub(super) fn get(&self, index: usize) -> Option<&str> {
        let start = usize::try_from(*self.offsets.get(index)?).ok()?;
        let end = usize::try_from(*self.offsets.get(index.checked_add(1)?)?).ok()?;
        self.blob
            .get(start..end)
            .map(|value| value.strip_suffix('\n').unwrap_or(value))
    }

    /// Writes the runtime index directly from the offsets already in memory.
    pub(super) fn write_index(&self, directory: &Path) -> Result<()> {
        let path = directory.join("strings.idx");
        let temporary = tempfile::NamedTempFile::new_in(directory).map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
        let mut output = BufWriter::new(temporary);
        for offset in &self.offsets {
            output
                .write_all(&offset.to_le_bytes())
                .map_err(|source| Error::Io {
                    path: path.clone(),
                    source,
                })?;
        }
        output.flush().map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
        let temporary = output.into_inner().map_err(|error| Error::Io {
            path: path.clone(),
            source: error.into_error(),
        })?;
        temporary.persist(&path).map_err(|error| Error::Io {
            path,
            source: error.error,
        })?;
        Ok(())
    }

    /// Keeps the selected IDs in original order and returns old-to-new IDs.
    fn retain(&mut self, live: &[bool], path: &Path) -> Result<Vec<u32>> {
        if live.len() != self.len() {
            return Err(Error::Invalid {
                path: path.to_owned(),
                message: "string liveness map does not match strings.txt".to_owned(),
            });
        }
        let mut remap = vec![u32::MAX; live.len()];
        let mut blob = String::new();
        let mut offsets =
            Vec::with_capacity(live.iter().filter(|&&keep| keep).count().saturating_add(1));
        offsets.push(0);
        for (old, &keep) in live.iter().enumerate() {
            if !keep {
                continue;
            }
            let new = u32::try_from(offsets.len().saturating_sub(1)).map_err(|_| {
                Error::StringsTooLarge {
                    path: path.to_owned(),
                }
            })?;
            if let Some(slot) = remap.get_mut(old) {
                *slot = new;
            }
            let value = self.get(old).ok_or_else(|| Error::Invalid {
                path: path.to_owned(),
                message: format!("string identifier {old} has no text"),
            })?;
            blob.push_str(value);
            blob.push('\n');
            offsets.push(
                u32::try_from(blob.len()).map_err(|_| Error::StringsTooLarge {
                    path: path.to_owned(),
                })?,
            );
        }
        self.blob = blob;
        self.offsets = offsets;
        Ok(remap)
    }
}

fn offsets(path: &Path, blob: &[u8]) -> Result<Vec<u32>> {
    line_offsets(blob).ok_or_else(|| Error::StringsTooLarge {
        path: path.to_owned(),
    })
}

fn validate_string_blob(path: &Path, raw: &str) -> Result<()> {
    if raw.contains(['\0', '\r']) {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: "string table contains NUL or CR".to_owned(),
        });
    }
    Ok(())
}

fn read_stable_string(path: &Path) -> Result<String> {
    let (mut input, before) = open_reader(path)?;
    let mut value = String::with_capacity(usize::try_from(before.len()).unwrap_or(0));
    input
        .read_to_string(&mut value)
        .map_err(|source| Error::Io {
            path: path.to_owned(),
            source,
        })?;
    let observed = u64::try_from(value.len()).unwrap_or(u64::MAX);
    verify_reader(path, &before, input.get_ref(), observed)?;
    Ok(value)
}

/// Visits a canonical sorted numeric TSV without retaining its rows.
///
/// `None` means a key is unsorted or not NFC, so the caller must materialise
/// and sort; `Some(count)` means every row was visited.
pub(super) fn visit_sorted_rows(
    path: &Path,
    string_count: usize,
    entities: bool,
    mut visit: impl FnMut(&str, u64) -> Result<()>,
) -> Result<Option<usize>> {
    let mut previous = String::new();
    let mut count = 0_usize;
    let mut canonical = true;
    for_each_line(path, |line_number, line| {
        let (key, value) = parse_numeric_row(path, line_number, line)?;
        validate_string_id(path, value, string_count)?;
        if entities {
            validate_entity_key(path, key)?;
        }
        if canonical {
            if !matches!(
                crate::normalization::nfc(key),
                std::borrow::Cow::Borrowed(_)
            ) {
                canonical = false;
            } else if !previous.is_empty() && key <= previous.as_str() {
                if key == previous {
                    return Err(Error::Source {
                        path: path.to_owned(),
                        line: line_number,
                        message: format!("duplicate key {key:?} after NFC normalization"),
                    });
                }
                canonical = false;
            }
            if canonical {
                visit(key, value)?;
                previous.clear();
                previous.push_str(key);
            }
        }
        count = line_number;
        Ok(())
    })?;
    Ok(canonical.then_some(count))
}

/// Reads and canonicalises a numeric source for the legacy fallback.
pub(super) fn read_rows(path: &Path) -> Result<Vec<(String, u64)>> {
    let mut rows = Vec::new();
    for_each_numeric_row(path, |_line, key, value| {
        rows.push((crate::normalization::nfc(key).into_owned(), value));
        Ok(())
    })?;
    rows.sort_by(|left, right| left.0.cmp(&right.0));
    reject_duplicate_rows(path, &rows)?;
    Ok(rows)
}

/// Reads and canonicalises the phrase source, whose value is text.
pub(super) fn read_text_rows(path: &Path) -> Result<Vec<(String, String)>> {
    let mut rows = Vec::new();
    for_each_line(path, |line_number, line| {
        let line = strip_line_ending(line);
        let mut columns = line.split('\t');
        let (Some(key), Some(value), None) = (columns.next(), columns.next(), columns.next())
        else {
            return Err(Error::Source {
                path: path.to_owned(),
                line: line_number,
                message: "expected exactly key<TAB>pattern".to_owned(),
            });
        };
        if key.is_empty() || value.is_empty() || line.contains(['\0', '\r']) {
            return Err(Error::Source {
                path: path.to_owned(),
                line: line_number,
                message: "key and pattern must be non-empty and contain no NUL/CR".to_owned(),
            });
        }
        rows.push((
            crate::normalization::nfc(key).into_owned(),
            crate::normalization::nfc(value).into_owned(),
        ));
        Ok(())
    })?;
    rows.sort_by(|left, right| left.0.cmp(&right.0));
    reject_duplicate_rows(path, &rows)?;
    Ok(rows)
}

fn reject_duplicate_rows<T>(path: &Path, rows: &[(String, T)]) -> Result<()> {
    if let Some((index, duplicate)) = rows.windows(2).enumerate().find_map(|(index, pair)| {
        let [left, right] = pair else { return None };
        (left.0 == right.0).then_some((index, left.0.as_str()))
    }) {
        return Err(Error::Source {
            path: path.to_owned(),
            line: index.saturating_add(2),
            message: format!("duplicate key {duplicate:?} after NFC normalization"),
        });
    }
    Ok(())
}

/// Removes strings used only by an external inline table and remaps every
/// source whose compiled representation still stores an ID.
pub(super) fn compact_inline_external(directory: &Path, strings: &mut StringTable) -> Result<bool> {
    let old_count = strings.len();
    let mut live = vec![false; old_count];
    for table in LookupTableKind::ALL
        .into_iter()
        .filter(|table| table.retains_string_id())
    {
        let name = table.name();
        mark_ids(
            &directory.join(format!("{name}.tsv")),
            old_count,
            false,
            &mut live,
        )?;
    }
    mark_ids(&directory.join("rules.tsv"), old_count, true, &mut live)?;
    if live.iter().all(|&keep| keep) {
        return Ok(false);
    }

    let strings_path = directory.join("strings.txt");
    let remap = strings.retain(&live, &strings_path)?;
    for table in LookupTableKind::ALL
        .into_iter()
        .filter(|table| table.retains_string_id())
    {
        let name = table.name();
        rewrite_ids(&directory.join(format!("{name}.tsv")), &remap, false)?;
    }
    rewrite_ids(&directory.join("rules.tsv"), &remap, true)?;
    write_file(&strings_path, strings.blob.as_bytes())?;
    Ok(true)
}

fn mark_ids(path: &Path, string_count: usize, packed_rule: bool, live: &mut [bool]) -> Result<()> {
    for_each_numeric_row(path, |line, _key, value| {
        let id = if packed_rule {
            value & RULE_STRING_MASK
        } else {
            value
        };
        let index = usize::try_from(id)
            .ok()
            .filter(|&id| id < string_count)
            .ok_or_else(|| Error::Source {
                path: path.to_owned(),
                line,
                message: format!("string identifier {id} is out of range 0..{string_count}"),
            })?;
        if let Some(slot) = live.get_mut(index) {
            *slot = true;
        }
        Ok(())
    })
}

fn rewrite_ids(path: &Path, remap: &[u32], packed_rule: bool) -> Result<()> {
    let parent = path.parent().ok_or_else(|| Error::Invalid {
        path: path.to_owned(),
        message: "source path has no parent".to_owned(),
    })?;
    let temporary = tempfile::NamedTempFile::new_in(parent).map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })?;
    let mut output = BufWriter::new(temporary);
    for_each_numeric_row(path, |line, key, value| {
        let old = if packed_rule {
            value & RULE_STRING_MASK
        } else {
            value
        };
        let new = usize::try_from(old)
            .ok()
            .and_then(|id| remap.get(id))
            .copied()
            .filter(|&id| id != u32::MAX)
            .ok_or_else(|| Error::Source {
                path: path.to_owned(),
                line,
                message: format!("string identifier {old} is not retained"),
            })?;
        let value = if packed_rule {
            (value & !RULE_STRING_MASK) | u64::from(new)
        } else {
            u64::from(new)
        };
        writeln!(output, "{key}\t{value}").map_err(|source| Error::Io {
            path: path.to_owned(),
            source,
        })
    })?;
    output.flush().map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })?;
    let temporary = output.into_inner().map_err(|error| Error::Io {
        path: path.to_owned(),
        source: error.into_error(),
    })?;
    temporary.persist(path).map_err(|error| Error::Io {
        path: path.to_owned(),
        source: error.error,
    })?;
    Ok(())
}

/// Reads `path` through a stable descriptor, one raw line (terminator
/// included) and its one-based number at a time.
fn for_each_line(path: &Path, mut visit: impl FnMut(usize, &str) -> Result<()>) -> Result<()> {
    let (mut input, before) = open_reader(path)?;
    let mut line = String::new();
    let mut observed = 0_u64;
    let mut line_number = 0_usize;
    loop {
        line.clear();
        let read = input.read_line(&mut line).map_err(|source| Error::Io {
            path: path.to_owned(),
            source,
        })?;
        if read == 0 {
            break;
        }
        observed = observed.saturating_add(u64::try_from(read).unwrap_or(u64::MAX));
        line_number = line_number.saturating_add(1);
        visit(line_number, &line)?;
    }
    verify_reader(path, &before, input.get_ref(), observed)
}

fn for_each_numeric_row(
    path: &Path,
    mut visit: impl FnMut(usize, &str, u64) -> Result<()>,
) -> Result<()> {
    for_each_line(path, |line_number, line| {
        let (key, value) = parse_numeric_row(path, line_number, line)?;
        visit(line_number, key, value)
    })
}

fn parse_numeric_row<'a>(path: &Path, line_number: usize, line: &'a str) -> Result<(&'a str, u64)> {
    let line = strip_line_ending(line);
    let mut columns = line.split('\t');
    let (Some(key), Some(value), None) = (columns.next(), columns.next(), columns.next()) else {
        return Err(Error::Source {
            path: path.to_owned(),
            line: line_number,
            message: "expected exactly key<TAB>unsigned-integer".to_owned(),
        });
    };
    if key.is_empty() || key.contains(['\0', '\r']) {
        return Err(Error::Source {
            path: path.to_owned(),
            line: line_number,
            message: "key must be non-empty and contain no NUL/CR".to_owned(),
        });
    }
    let value = value.parse().map_err(|_| Error::Source {
        path: path.to_owned(),
        line: line_number,
        message: format!("invalid unsigned integer {value:?}"),
    })?;
    Ok((key, value))
}

/// Matches `str::lines`: CR is a terminator only as the first half of CRLF.
fn strip_line_ending(line: &str) -> &str {
    line.strip_suffix('\n')
        .map_or(line, |line| line.strip_suffix('\r').unwrap_or(line))
}

pub(super) fn validate_string_id(path: &Path, id: u64, string_count: usize) -> Result<()> {
    if usize::try_from(id).map_or(true, |id| id >= string_count) {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: format!("string identifier {id} is out of range 0..{string_count}"),
        });
    }
    Ok(())
}

pub(super) fn validate_entity_key(path: &Path, key: &str) -> Result<()> {
    let Some((namespace, surface)) = key.as_bytes().split_first() else {
        return Err(invalid_entity(path, key));
    };
    if !matches!(namespace, b'0' | b'1') || surface.is_empty() {
        return Err(invalid_entity(path, key));
    }
    Ok(())
}

fn invalid_entity(path: &Path, key: &str) -> Error {
    Error::Invalid {
        path: path.to_owned(),
        message: format!("entity key {key:?} is not encoded as 0<exact> or 1<locale-lower>"),
    }
}

fn open_reader(path: &Path) -> Result<(BufReader<File>, Metadata)> {
    crate::secure_fs::open_regular(path)
        .map(|(file, metadata)| (BufReader::new(file), metadata))
        .map_err(|error| match error {
            OpenRegularError::Io(source) => Error::Io {
                path: path.to_owned(),
                source,
            },
            OpenRegularError::NotRegular => Error::Invalid {
                path: path.to_owned(),
                message: "compiler inputs must be regular files, not symlinks".to_owned(),
            },
            OpenRegularError::Changed => Error::Invalid {
                path: path.to_owned(),
                message: "compiler input path changed while it was opened".to_owned(),
            },
        })
}

fn verify_reader(path: &Path, before: &Metadata, file: &File, observed: u64) -> Result<()> {
    crate::secure_fs::verify_stable(path, before, file, observed).map_err(|error| match error {
        StabilityError::Io(source) => Error::Io {
            path: path.to_owned(),
            source,
        },
        StabilityError::Changed => Error::Invalid {
            path: path.to_owned(),
            message: "compiler input descriptor or path changed while it was read".to_owned(),
        },
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::lexicon::runtime::read_to_string;

    #[test]
    fn sparse_ids_are_retained_in_order_without_per_string_storage() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("strings.txt");
        write_file(&path, b"zero\none\ntwo\nthree\n").unwrap();
        let mut strings = StringTable::normalise(&path, &path).unwrap();
        let remap = strings.retain(&[false, true, false, true], &path).unwrap();
        assert_eq!(remap, [u32::MAX, 0, u32::MAX, 1]);
        assert_eq!(strings.blob, "one\nthree\n");
        assert_eq!(strings.get(0), Some("one"));
        assert_eq!(strings.get(1), Some("three"));
    }

    #[test]
    fn legacy_normalisation_keeps_ids_and_writes_canonical_newlines() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("strings.txt");
        write_file(&path, "cafe\u{301}\nplain".as_bytes()).unwrap();
        let strings = StringTable::normalise(&path, &path).unwrap();
        assert_eq!(strings.len(), 2);
        assert_eq!(strings.get(0), Some("caf\u{e9}"));
        assert_eq!(read_to_string(&path).unwrap(), "caf\u{e9}\nplain\n");
    }

    #[test]
    fn sorted_visitor_reports_fallback_only_after_discardable_partial_work() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("exact.tsv");
        write_file(&path, b"beta\t0\nalpha\t1\n").unwrap();
        let mut visited = Vec::new();
        let result = visit_sorted_rows(&path, 2, false, |key, value| {
            visited.push((key.to_owned(), value));
            Ok(())
        })
        .unwrap();
        assert_eq!(result, None);
        assert_eq!(visited, [("beta".to_owned(), 0)]);
        assert_eq!(
            read_rows(&path).unwrap().first().map(|row| row.0.as_str()),
            Some("alpha")
        );
    }

    #[test]
    fn numeric_and_text_sources_accept_crlf_but_not_a_bare_carriage_return() {
        let temporary = tempfile::tempdir().unwrap();
        let numeric = temporary.path().join("exact.tsv");
        write_file(&numeric, b"cats\t0\r\n").unwrap();
        assert_eq!(
            visit_sorted_rows(&numeric, 1, false, |_key, _value| Ok(())).unwrap(),
            Some(1)
        );
        assert_eq!(read_rows(&numeric).unwrap(), [("cats".to_owned(), 0)]);

        let text = temporary.path().join("phrases.tsv");
        write_file(&text, b"key\tpattern\r\n").unwrap();
        assert_eq!(
            read_text_rows(&text).unwrap(),
            [("key".to_owned(), "pattern".to_owned())]
        );

        write_file(&numeric, b"cats\t0\r").unwrap();
        assert!(matches!(read_rows(&numeric), Err(Error::Source { .. })));
    }

    #[test]
    fn compact_external_inline_remaps_every_stored_id_but_not_dead_source_ids() {
        let temporary = tempfile::tempdir().unwrap();
        let directory = temporary.path();
        let strings_path = directory.join("strings.txt");
        write_file(&strings_path, b"dead-a\nexact\ndead-b\nentity\nrule\n").unwrap();
        write_file(&directory.join("exact.tsv"), b"cats\t1\n").unwrap();
        write_file(&directory.join("lower.tsv"), b"").unwrap();
        write_file(&directory.join("external.tsv"), b"outside\t2\n").unwrap();
        write_file(&directory.join("entities.tsv"), b"0Name\t3\n").unwrap();
        let packed = (2_u64 << 56) | (1_u64 << 48) | 4;
        write_file(
            &directory.join("rules.tsv"),
            format!("0s\t{packed}\n").as_bytes(),
        )
        .unwrap();

        let mut strings = StringTable::normalise(&strings_path, &strings_path).unwrap();
        assert!(compact_inline_external(directory, &mut strings).unwrap());
        assert_eq!(
            read_to_string(&strings_path).unwrap(),
            "exact\nentity\nrule\n"
        );
        assert_eq!(
            read_to_string(&directory.join("exact.tsv")).unwrap(),
            "cats\t0\n"
        );
        assert_eq!(
            read_to_string(&directory.join("entities.tsv")).unwrap(),
            "0Name\t1\n"
        );
        assert_eq!(
            read_to_string(&directory.join("rules.tsv")).unwrap(),
            format!("0s\t{}\n", (2_u64 << 56) | (1_u64 << 48) | 2)
        );
        // External IDs are deliberately inert once its codec embeds lemmas.
        assert_eq!(
            read_to_string(&directory.join("external.tsv")).unwrap(),
            "outside\t2\n"
        );
    }
}
