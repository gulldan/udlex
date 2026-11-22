//! Memory-bounded lookup-table compilation.

use std::fs::File;
use std::io::{Seek as _, SeekFrom, Write as _};
use std::path::Path;

use yada::builder::DoubleArrayBuilder;

use super::super::table::build_hybrid;
use super::super::{
    AUTOMATON_BYTES_PER_PROBE_SHARE, Codec, Error, Result, encode_key_into, fst_error,
};
use super::source::{
    StringTable, read_rows, validate_entity_key, validate_string_id, visit_sorted_rows,
};
use super::writer::remove;
use crate::format::LookupTableKind;

/// Largest value accepted by yada's terminal unit.
const MAX_DA_VALUE: u64 = (1_u64 << 31) - 1;
/// A serialized yada unit occupies four bytes and its builder allocates blocks.
const DA_UNIT_BYTES: usize = size_of::<u32>();
const DA_BLOCK_UNITS: usize = 256;

/// One compiled table held outside the generation directory until its content
/// identity has been assigned.
pub(super) struct PreparedTable {
    entries: usize,
    selected: Option<(Codec, Artifact)>,
}

impl PreparedTable {
    pub(super) const fn entries(&self) -> usize {
        self.entries
    }

    /// Installs exactly one encoding and removes every loser.
    pub(super) fn install(self, directory: &Path, table: LookupTableKind) -> Result<Option<Codec>> {
        let name = table.name();
        let selected = self
            .selected
            .map(|(codec, artifact)| {
                artifact.write_to(&codec.file(directory, name))?;
                Ok::<_, Error>(codec)
            })
            .transpose()?;
        for codec in Codec::ALL {
            if Some(codec) != selected {
                remove(&codec.file(directory, name))?;
            }
        }
        Ok(selected)
    }
}

/// Builds a table: `exact`/`lower` as `hybrid`, the cold tables by weighing a
/// double array against the smaller automaton.
///
/// Canonical input streams into anonymous temporary files; a double array is
/// skipped when a lower bound on its size already proves it unaffordable.
pub(super) fn prepare_table(
    directory: &Path,
    table: LookupTableKind,
    strings: &StringTable,
    traffic: f64,
) -> Result<PreparedTable> {
    if !table.is_cold() {
        return prepare_hybrid(directory, table, strings);
    }
    let name = table.name();
    let source_path = directory.join(format!("{name}.tsv"));
    let mut automata = AutomataBuilder::new(directory, name)?;
    let mut da_compatible = true;
    let streamed = visit_sorted_rows(
        &source_path,
        strings.len(),
        table.is_entity(),
        |form, id| {
            da_compatible &= id <= MAX_DA_VALUE;
            automata.insert(directory, name, form, id, strings)
        },
    )?;

    let Some(entries) = streamed else {
        // A decomposed or unsorted legacy source has to be normalised and
        // sorted before any candidate can be compared.
        drop(automata);
        let rows = read_rows(&source_path)?;
        return prepare_materialised(directory, table, strings, traffic, &rows);
    };
    if entries == 0 {
        return Ok(PreparedTable {
            entries,
            selected: None,
        });
    }

    let automaton = automata.finish(directory, name)?;
    if da_compatible && da_cannot_win(entries, automaton.1.len, traffic) {
        return Ok(PreparedTable {
            entries,
            selected: Some(automaton),
        });
    }

    // yada builds from a slice, so a double array still in play needs the rows.
    let rows = read_rows(&source_path)?;
    finish_with_da(directory, name, strings, traffic, &rows, Some(automaton))
}

/// The hot tables take the hybrid codec outright; no cost model weighs it.
fn prepare_hybrid(
    directory: &Path,
    table: LookupTableKind,
    strings: &StringTable,
) -> Result<PreparedTable> {
    let name = table.name();
    let source_path = directory.join(format!("{name}.tsv"));
    let rows = read_rows(&source_path)?;
    for (_, id) in &rows {
        validate_string_id(&source_path, *id, strings.len())?;
    }
    if rows.is_empty() {
        return Ok(PreparedTable {
            entries: 0,
            selected: None,
        });
    }
    let path = Codec::Hybrid.file(directory, name);
    let bytes = build_hybrid(&rows, |id| {
        usize::try_from(id)
            .ok()
            .and_then(|index| strings.get(index))
            .map(str::to_owned)
    })
    .map_err(|message| Error::Invalid {
        path: path.clone(),
        message,
    })?;
    Ok(PreparedTable {
        entries: rows.len(),
        selected: Some((Codec::Hybrid, Artifact::from_bytes(&path, &bytes)?)),
    })
}

fn prepare_materialised(
    directory: &Path,
    table: LookupTableKind,
    strings: &StringTable,
    traffic: f64,
    rows: &[(String, u64)],
) -> Result<PreparedTable> {
    let name = table.name();
    let source_path = directory.join(format!("{name}.tsv"));
    for (key, id) in rows {
        validate_string_id(&source_path, *id, strings.len())?;
        if table.is_entity() {
            validate_entity_key(&source_path, key)?;
        }
    }
    if rows.is_empty() {
        return Ok(PreparedTable {
            entries: 0,
            selected: None,
        });
    }
    finish_with_da(directory, table.name(), strings, traffic, rows, None)
}

fn finish_with_da(
    directory: &Path,
    name: &str,
    strings: &StringTable,
    traffic: f64,
    rows: &[(String, u64)],
    automaton: Option<(Codec, Artifact)>,
) -> Result<PreparedTable> {
    let entries = rows.len();
    let automaton = match automaton {
        Some(candidate) => candidate,
        None => build_automata(directory, name, strings, rows)?,
    };
    let da_path = Codec::Da.file(directory, name);
    let da_bytes = build_da(rows, &da_path)?;
    let da = Artifact::from_bytes(&da_path, &da_bytes)?;
    let affordable =
        da.len.saturating_sub(automaton.1.len) as f64 <= AUTOMATON_BYTES_PER_PROBE_SHARE * traffic;
    let selected = if affordable {
        (Codec::Da, da)
    } else {
        automaton
    };
    Ok(PreparedTable {
        entries,
        selected: Some(selected),
    })
}

/// A lower bound derived from yada's representation: the root and each key's
/// terminal leaf are distinct units, and serialization rounds to 256 units.
const fn minimum_da_bytes(entries: usize) -> usize {
    entries
        .saturating_add(1)
        .div_ceil(DA_BLOCK_UNITS)
        .saturating_mul(DA_BLOCK_UNITS)
        .saturating_mul(DA_UNIT_BYTES)
}

fn da_cannot_win(entries: usize, automaton: usize, traffic: f64) -> bool {
    minimum_da_bytes(entries).saturating_sub(automaton) as f64
        > AUTOMATON_BYTES_PER_PROBE_SHARE * traffic
}

fn build_da(rows: &[(String, u64)], path: &Path) -> Result<Vec<u8>> {
    let keyset: Vec<(&[u8], u32)> = rows
        .iter()
        .map(|(key, value)| {
            u32::try_from(*value)
                .map(|value| (key.as_bytes(), value))
                .map_err(|_| Error::Invalid {
                    path: path.to_owned(),
                    message: format!("value {value} exceeds the double-array u32 limit"),
                })
        })
        .collect::<Result<_>>()?;
    DoubleArrayBuilder::build(&keyset).map_err(|source| Error::Trie {
        path: path.to_owned(),
        source,
    })
}

fn build_automata(
    directory: &Path,
    name: &str,
    strings: &StringTable,
    rows: &[(String, u64)],
) -> Result<(Codec, Artifact)> {
    let mut builder = AutomataBuilder::new(directory, name)?;
    for (form, id) in rows {
        builder.insert(directory, name, form, *id, strings)?;
    }
    builder.finish(directory, name)
}

struct AutomataBuilder {
    map: fst::MapBuilder<File>,
    inline: fst::raw::Builder<File>,
    inline_key: Vec<u8>,
}

impl AutomataBuilder {
    fn new(directory: &Path, name: &str) -> Result<Self> {
        let map_path = Codec::Map.file(directory, name);
        let inline_path = Codec::Inline.file(directory, name);
        let map_file = anonymous_file(&map_path)?;
        let inline_file = anonymous_file(&inline_path)?;
        Ok(Self {
            map: fst::MapBuilder::new(map_file).map_err(|source| fst_error(&map_path, source))?,
            inline: fst::raw::Builder::new(inline_file)
                .map_err(|source| fst_error(&inline_path, source))?,
            inline_key: Vec::new(),
        })
    }

    fn insert(
        &mut self,
        directory: &Path,
        name: &str,
        form: &str,
        id: u64,
        strings: &StringTable,
    ) -> Result<()> {
        self.map
            .insert(form.as_bytes(), id)
            .map_err(|source| fst_error(&Codec::Map.file(directory, name), source))?;
        let index = usize::try_from(id).ok();
        let lemma = index
            .and_then(|index| strings.get(index))
            .ok_or_else(|| Error::Invalid {
                path: directory.join(format!("{name}.tsv")),
                message: format!(
                    "string identifier {id} is out of range 0..{}",
                    strings.len()
                ),
            })?;
        encode_key_into(form.as_bytes(), lemma.as_bytes(), &mut self.inline_key);
        self.inline
            .insert(&self.inline_key, 0)
            .map_err(|source| fst_error(&Codec::Inline.file(directory, name), source))
    }

    fn finish(self, directory: &Path, name: &str) -> Result<(Codec, Artifact)> {
        let map_path = Codec::Map.file(directory, name);
        let inline_path = Codec::Inline.file(directory, name);
        let map = Artifact::from_file(
            &map_path,
            self.map
                .into_inner()
                .map_err(|source| fst_error(&map_path, source))?,
        )?;
        let inline = Artifact::from_file(
            &inline_path,
            self.inline
                .into_inner()
                .map_err(|source| fst_error(&inline_path, source))?,
        )?;
        if inline.len < map.len {
            Ok((Codec::Inline, inline))
        } else {
            Ok((Codec::Map, map))
        }
    }
}

struct Artifact {
    file: File,
    len: usize,
}

impl Artifact {
    fn from_file(path: &Path, file: File) -> Result<Self> {
        let len = file
            .metadata()
            .map_err(|source| Error::Io {
                path: path.to_owned(),
                source,
            })?
            .len();
        let len = usize::try_from(len).map_err(|_| Error::Invalid {
            path: path.to_owned(),
            message: "compiled table is too large for this platform".to_owned(),
        })?;
        Ok(Self { file, len })
    }

    fn from_bytes(path: &Path, bytes: &[u8]) -> Result<Self> {
        let mut file = anonymous_file(path)?;
        file.write_all(bytes).map_err(|source| Error::Io {
            path: path.to_owned(),
            source,
        })?;
        Ok(Self {
            file,
            len: bytes.len(),
        })
    }

    fn write_to(mut self, path: &Path) -> Result<()> {
        self.file
            .seek(SeekFrom::Start(0))
            .map_err(|source| Error::Io {
                path: path.to_owned(),
                source,
            })?;
        let mut output = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(path)
            .map_err(|source| Error::Io {
                path: path.to_owned(),
                source,
            })?;
        let copied = std::io::copy(&mut self.file, &mut output).map_err(|source| Error::Io {
            path: path.to_owned(),
            source,
        })?;
        if usize::try_from(copied).ok() != Some(self.len) {
            return Err(Error::Invalid {
                path: path.to_owned(),
                message: format!("copied {copied} bytes from a {}-byte candidate", self.len),
            });
        }
        Ok(())
    }
}

fn anonymous_file(path: &Path) -> Result<File> {
    tempfile::tempfile().map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::lexicon::generation::source::StringTable;
    use crate::lexicon::generation::writer::write_file;

    #[test]
    fn lower_bound_never_skips_a_candidate_that_could_be_affordable() {
        assert!(!da_cannot_win(10, 100, 1.0));
        assert!(da_cannot_win(10_000_000, 1_000_000, 0.0));
        assert!(minimum_da_bytes(1) >= 2 * DA_UNIT_BYTES);
    }

    #[test]
    fn streaming_and_legacy_fallback_emit_identical_tables() {
        let canonical = tempfile::tempdir().unwrap();
        let legacy = tempfile::tempdir().unwrap();
        for directory in [canonical.path(), legacy.path()] {
            write_file(
                &directory.join("strings.txt"),
                "caf\u{e9}\ndog\n".as_bytes(),
            )
            .unwrap();
        }
        write_file(
            &canonical.path().join("exact.tsv"),
            "caf\u{e9}s\t0\ndogs\t1\n".as_bytes(),
        )
        .unwrap();
        // Both row order and the first key require the legacy fallback.
        write_file(
            &legacy.path().join("exact.tsv"),
            "dogs\t1\ncafe\u{301}s\t0\n".as_bytes(),
        )
        .unwrap();

        let compile = |directory: &Path| {
            let path = directory.join("strings.txt");
            let strings = StringTable::normalise(&path, &path).unwrap();
            let table = LookupTableKind::Exact;
            let prepared = prepare_table(directory, table, &strings, 1.0).unwrap();
            prepared.install(directory, table).unwrap().unwrap()
        };
        let canonical_codec = compile(canonical.path());
        let legacy_codec = compile(legacy.path());
        assert_eq!(canonical_codec, legacy_codec);
        assert_eq!(
            std::fs::read(canonical_codec.file(canonical.path(), "exact")).unwrap(),
            std::fs::read(legacy_codec.file(legacy.path(), "exact")).unwrap()
        );
    }
}
