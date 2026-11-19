//! Compilation, format migration, and source pruning.

use std::collections::BTreeMap;
use std::path::Path;

use yada::builder::DoubleArrayBuilder;

use super::super::{
    Codec, DEFAULT_TRAFFIC, DEFAULT_TRAFFIC_PARADIGM, ENTITY_ENCODING, Error, FORMAT_VERSION,
    Lexicon, Meta, NORMALIZATION, PHRASE_CODEC, Result, SOURCES, VALUE_WIDTH, build_phrase_table,
    file_integrity, read_metadata, resolve_generation, runtime_files, validate_meta,
};
use super::identity::{content_generation_id, render_meta};
use super::lookup::{PreparedTable, prepare_table};
use super::source::{StringTable, compact_inline_external, read_rows, read_text_rows};
use super::writer::{GenerationWriter, remove, write_file};
use crate::format::{FormatRange, LookupTableKind};
use crate::secure_fs::{OpenRegularError, StabilityError};

/// The share of tokens that reaches one lookup table: 1 for `exact`, else the
/// measured `traffic`, else a default by training source.
fn probe_share(meta: &serde_json::Value, table: LookupTableKind, path: &Path) -> Result<f64> {
    if table == LookupTableKind::Exact {
        return Ok(1.0);
    }
    let name = table.name();
    if let Some(measured) = meta
        .get("traffic")
        .and_then(|traffic| traffic.get(name))
        .and_then(serde_json::Value::as_f64)
    {
        if measured.is_finite() && (0.0..=1.0).contains(&measured) {
            return Ok(measured);
        }
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: format!("traffic.{name} must be finite and in 0..=1, got {measured}"),
        });
    }
    // A corpus-trained lexicon's source is "<treebank>.train"; anything else
    // ("unimorph/tgl", "wiktionary/sah") means external paradigms.
    let corpus_trained = meta
        .get("source")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|source| source.rsplit('.').next() == Some("train"));
    if corpus_trained {
        Ok(DEFAULT_TRAFFIC)
    } else {
        Ok(DEFAULT_TRAFFIC_PARADIGM)
    }
}

/// Builds one double array and writes it, returning whether anything was
/// written; an empty table removes any stale file instead.
fn write_trie(destination: &Path, keyset: &[(&[u8], u32)]) -> Result<bool> {
    if keyset.is_empty() {
        remove(destination)?;
        return Ok(false);
    }
    let bytes = DoubleArrayBuilder::build(keyset).map_err(|source| Error::Trie {
        path: destination.to_owned(),
        source,
    })?;
    write_file(destination, &bytes)?;
    Ok(true)
}

/// What [`compile`] did to one lexicon directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Compiled {
    /// How many lookup tables were written.
    pub tables: usize,
    /// Whether the builder's sources were there to compile from; `false`
    /// means the existing generation was only validated.
    pub from_sources: bool,
}

/// Whether a lexicon directory still holds the builder's sources.
///
/// # Errors
///
/// Returns [`Error::Sources`] when some are there and some are not.
fn has_sources(directory: &Path) -> Result<bool> {
    let missing: Vec<&str> = SOURCES
        .into_iter()
        .filter(|name| !directory.join(name).exists())
        .collect();
    match missing.len() {
        0 => Ok(true),
        // Format-3 sources predate the phrase table; migration synthesises it.
        1 if missing == ["phrases.tsv"] => Ok(true),
        count if count == SOURCES.len() => Ok(false),
        _ => Err(Error::Sources {
            path: directory.to_owned(),
            missing: missing.join(", "),
        }),
    }
}

impl GenerationWriter {
    fn compile_and_seal_inner(mut self) -> Result<(Self, Compiled, String)> {
        let meta_path = self.directory().join("meta.json");
        let mut meta: serde_json::Value = serde_json::from_str(&read_metadata(&meta_path)?)
            .map_err(|source| Error::Meta {
                path: meta_path.clone(),
                source,
            })?;
        let strings_path = self.directory().join("strings.txt");
        let mut strings = StringTable::normalise(&strings_path, &strings_path)?;
        let external =
            probe_share(&meta, LookupTableKind::External, &meta_path).and_then(|traffic| {
                prepare_table(
                    self.directory(),
                    LookupTableKind::External,
                    &strings,
                    traffic,
                )
            });
        let mut prepared_external = match external {
            Ok(prepared) => prepared,
            Err(external_error) => {
                // External is prepared first for compaction, but an invalid
                // exact/lower source must still win the diagnostic.
                for table in [LookupTableKind::Exact, LookupTableKind::Lower] {
                    let traffic = probe_share(&meta, table, &meta_path)?;
                    drop(prepare_table(self.directory(), table, &strings, traffic)?);
                }
                return Err(external_error);
            }
        };
        let compacted = if prepared_external.codec() == Some(Codec::Inline) {
            match compact_inline_external(self.directory(), &mut strings) {
                Ok(compacted) => compacted,
                Err(compaction_error) => {
                    validate_before_compaction_error(self.directory(), &meta, &strings)?;
                    return Err(compaction_error);
                }
            }
        } else {
            false
        };
        let external_identity = compacted
            .then(|| prepared_external.integrity(self.directory(), LookupTableKind::External))
            .transpose()?
            .flatten();
        let derived = external_identity
            .as_ref()
            .map(|integrity| vec![("external.inline", integrity)])
            .unwrap_or_default();
        let generation =
            content_generation_id(self.directory(), &meta, "compiled-source", &derived)?;
        self.assign_generation(generation);
        let generation = self.id().to_owned();
        let outcome = compile_generation(
            self.directory(),
            &mut meta,
            &generation,
            &strings,
            prepared_external,
        )?;
        let final_path = self.seal()?.to_owned();
        Lexicon::open(&final_path)?;
        Ok((self, outcome, generation))
    }

    /// Compiles and validates an immutable generation without changing readers.
    pub(crate) fn compile_and_seal(self) -> Result<(Compiled, String)> {
        let (_writer, outcome, generation) = self.compile_and_seal_inner()?;
        Ok((outcome, generation))
    }

    /// Compiles this private source generation, validates the complete leaf,
    /// and only then changes the root pointer.
    pub(crate) fn compile_and_publish_identified(self) -> Result<(Compiled, String)> {
        let (writer, outcome, generation) = self.compile_and_seal_inner()?;
        writer.publish()?;
        Ok((outcome, generation))
    }

    pub(crate) fn compile_and_publish(self) -> Result<Compiled> {
        self.compile_and_publish_identified()
            .map(|(outcome, _generation)| outcome)
    }

    /// Rebuilds the hot tables of a copied older generation and publishes it
    /// under the current format; see [`compile`].
    fn migrate_and_publish(mut self) -> Result<Compiled> {
        let meta_path = self.directory().join("meta.json");
        let mut meta: serde_json::Value = serde_json::from_str(&read_metadata(&meta_path)?)
            .map_err(|source| Error::Meta {
                path: meta_path.clone(),
                source,
            })?;
        let strings_path = self.directory().join("strings.txt");
        let strings = StringTable::normalise(&strings_path, &strings_path)?;
        // An inline external table outlived its compacted source, so it is
        // bound into the identity exactly as a fresh compile binds it.
        let inline_external =
            Codec::Inline.file(self.directory(), LookupTableKind::External.name());
        let external_identity = inline_external
            .exists()
            .then(|| file_integrity(&inline_external))
            .transpose()?;
        let derived = external_identity
            .as_ref()
            .map(|integrity| vec![("external.inline", integrity)])
            .unwrap_or_default();
        let generation =
            content_generation_id(self.directory(), &meta, "compiled-source", &derived)?;
        self.assign_generation(generation);
        let generation = self.id().to_owned();
        let outcome = migrate_generation(self.directory(), &mut meta, &generation, &strings)?;
        let final_path = self.seal()?.to_owned();
        Lexicon::open(&final_path)?;
        self.publish()?;
        Ok(outcome)
    }
}

/// Recompiles `exact` and `lower` under the current codec and carries every
/// other table, count and codec of the copied generation forward.
fn migrate_generation(
    directory: &Path,
    meta: &mut serde_json::Value,
    generation: &str,
    strings: &StringTable,
) -> Result<Compiled> {
    let retained = |key: &str| {
        meta.get(key)
            .and_then(serde_json::Value::as_object)
            .cloned()
            .unwrap_or_default()
    };
    let mut codecs = retained("codecs");
    let mut entries = retained("entries");
    initialise_generation_meta(meta, directory, generation)?;
    let mut compiled = 0_usize;
    for table in [LookupTableKind::Exact, LookupTableKind::Lower] {
        let name = table.name();
        let traffic = probe_share(meta, table, &directory.join("meta.json"))?;
        let prepared = prepare_table(directory, table, strings, traffic)?;
        entries.insert(
            name.to_owned(),
            u64::try_from(prepared.entries()).unwrap_or(u64::MAX).into(),
        );
        codecs.remove(name);
        if let Some(codec) = prepared.install(directory, table)? {
            codecs.insert(name.to_owned(), codec.name().into());
            compiled = compiled.saturating_add(1);
        }
    }
    install_table_manifest(meta, directory, codecs, entries)?;
    manifest_generation_files(meta, directory, generation)?;
    Ok(Compiled {
        tables: compiled,
        from_sources: true,
    })
}

/// After compaction failed, reports an earlier malformed source first.
fn validate_before_compaction_error(
    directory: &Path,
    meta: &serde_json::Value,
    strings: &StringTable,
) -> Result<()> {
    for table in LookupTableKind::ALL
        .into_iter()
        .filter(|table| table.retains_string_id())
    {
        let traffic = probe_share(meta, table, &directory.join("meta.json"))?;
        drop(prepare_table(directory, table, strings, traffic)?);
    }
    let mut entries = serde_json::Map::new();
    compile_phrase_table(directory, &mut entries)?;
    compile_rule_table(directory, strings.len(), &mut entries)?;
    Ok(())
}

/// Compiles the sorted TSV files a build produced into lookup tables,
/// recording each table's codec under `codecs` in `meta.json`.
///
/// A current compiled generation is only validated; an older readable one is
/// migrated by rebuilding `exact` and `lower`. When `external.inline` wins,
/// its now-unreferenced lemmas are dropped from `strings.txt`.
///
/// # Errors
///
/// Returns an error when a source is unreadable, a destination cannot be
/// written, only part of the sources is there, or the bundle declares a format
/// outside [`FormatRange::COMPILE`]. What it writes is always
/// [`FORMAT_VERSION`], so compiling an accepted older source migrates it.
pub fn compile(directory: impl AsRef<Path>) -> Result<Compiled> {
    let root = directory.as_ref();
    // Resolve only after taking the writer lock, so `current` cannot move.
    let writer = GenerationWriter::begin(root)?;
    let resolved = resolve_generation(root)?;
    let source_directory = resolved.directory;
    let source_meta_path = source_directory.join("meta.json");
    let meta: serde_json::Value = serde_json::from_str(&read_metadata(&source_meta_path)?)
        .map_err(|source| Error::Meta {
            path: source_meta_path.clone(),
            source,
        })?;
    let declared = u32::try_from(
        meta.get("format_version")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0),
    )
    .unwrap_or(u32::MAX);
    if !FormatRange::COMPILE.accepts(declared) {
        return Err(Error::Format {
            path: source_directory,
            found: declared,
            supported: FormatRange::COMPILE,
        });
    }
    if declared == 3 && source_directory.join("phrases.tsv").exists() {
        return Err(Error::Invalid {
            path: source_directory.join("phrases.tsv"),
            message: "format-3 did not define phrase patterns; remove the non-standard source or rebuild it with a format-4 writer"
                .to_owned(),
        });
    }
    if FormatRange::READ.accepts(declared)
        && meta
            .get("entity_encoding")
            .and_then(serde_json::Value::as_str)
            != Some(ENTITY_ENCODING)
    {
        return Err(Error::Invalid {
            path: source_meta_path,
            message: format!(
                "entity_encoding must be {ENTITY_ENCODING:?}; legacy format-3 entity tables are ambiguous and must be rebuilt"
            ),
        });
    }
    let compiled_before = meta
        .get("files")
        .and_then(serde_json::Value::as_object)
        .is_some_and(|files| !files.is_empty());
    // Already a current generation rather than a source tree to migrate.
    if declared == FORMAT_VERSION && compiled_before {
        Lexicon::open(root)?;
        return Ok(Compiled {
            tables: 0,
            from_sources: false,
        });
    }
    if !has_sources(&source_directory)? {
        // A no-source directory is a no-op only if the generation opens.
        Lexicon::open(root)?;
        return Ok(Compiled {
            tables: 0,
            from_sources: false,
        });
    }

    let staging = writer.directory().to_owned();
    for name in SOURCES {
        let source = source_directory.join(name);
        if name == "phrases.tsv" && !source.exists() {
            write_file(&staging.join(name), b"")?;
        } else {
            copy_regular(&source, &staging.join(name))?;
        }
    }
    copy_regular(
        &source_directory.join("strings.txt"),
        &staging.join("strings.txt"),
    )?;
    write_file(&staging.join("meta.json"), &render_meta(&meta))?;
    if compiled_before && FormatRange::READ.accepts(declared) {
        // Keep every compiled table but the hot ones; `external.tsv` cannot be
        // recompiled after inline compaction renumbered the other sources.
        let old: Meta = serde_json::from_value(meta).map_err(|source| Error::Meta {
            path: source_directory.join("meta.json"),
            source,
        })?;
        for name in runtime_files(&old, &source_directory)? {
            if name.starts_with("exact.") || name.starts_with("lower.") || name == "strings.txt" {
                continue;
            }
            copy_regular(&source_directory.join(&name), &staging.join(&name))?;
        }
        return writer.migrate_and_publish();
    }
    writer.compile_and_publish()
}

/// Publishes a runtime-only copy of the active generation; the active one is
/// never edited in place.
///
/// # Errors
///
/// Returns an error when the active generation is invalid, its source set is
/// partial, a runtime file is not regular, or publication fails.
pub fn prune(directory: impl AsRef<Path>) -> Result<bool> {
    let root = directory.as_ref();
    let mut writer = GenerationWriter::begin(root)?;
    let resolved = resolve_generation(root)?;
    // Verify hashes and table cross-references before copying anything.
    Lexicon::open(root)?;
    if !has_sources(&resolved.directory)? {
        return Ok(false);
    }
    let meta_path = resolved.directory.join("meta.json");
    let mut value: serde_json::Value =
        serde_json::from_str(&read_metadata(&meta_path)?).map_err(|source| Error::Meta {
            path: meta_path.clone(),
            source,
        })?;
    let meta: Meta = serde_json::from_value(value.clone()).map_err(|source| Error::Meta {
        path: meta_path,
        source,
    })?;
    for name in runtime_files(&meta, &resolved.directory)? {
        copy_regular(
            &resolved.directory.join(&name),
            &writer.directory().join(name),
        )?;
    }
    let generation = content_generation_id(writer.directory(), &value, "runtime-only", &[])?;
    writer.assign_generation(generation);
    value
        .as_object_mut()
        .ok_or_else(|| Error::Invalid {
            path: resolved.directory.join("meta.json"),
            message: "metadata root must be a JSON object".to_owned(),
        })?
        .insert("generation".to_owned(), writer.id().into());
    write_file(&writer.directory().join("meta.json"), &render_meta(&value))?;
    let leaf = writer.seal()?.to_owned();
    Lexicon::open(&leaf)?;
    writer.publish()?;
    Ok(true)
}

/// Copies one compiler input without following a symlink.
fn copy_regular(source: &Path, destination: &Path) -> Result<()> {
    copy_regular_with(source, destination, || {})
}

/// Descriptor-copy implementation with a deterministic post-open test hook.
fn copy_regular_with(source: &Path, destination: &Path, after_open: impl FnOnce()) -> Result<()> {
    let (mut input, descriptor_before) =
        crate::secure_fs::open_regular(source).map_err(|error| match error {
            OpenRegularError::Io(source_error) => Error::Io {
                path: source.to_owned(),
                source: source_error,
            },
            OpenRegularError::NotRegular => Error::Invalid {
                path: source.to_owned(),
                message: "compiler inputs must be regular files, not symlinks".to_owned(),
            },
            OpenRegularError::Changed => Error::Invalid {
                path: source.to_owned(),
                message: "compiler input path changed while it was opened".to_owned(),
            },
        })?;

    after_open();

    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)
        .map_err(|source_error| Error::Io {
            path: destination.to_owned(),
            source: source_error,
        })?;
    let copied = std::io::copy(&mut input, &mut output).map_err(|source_error| Error::Io {
        path: destination.to_owned(),
        source: source_error,
    })?;
    output.sync_all().map_err(|source_error| Error::Io {
        path: destination.to_owned(),
        source: source_error,
    })?;

    crate::secure_fs::verify_stable(source, &descriptor_before, &input, copied).map_err(|error| {
        match error {
            StabilityError::Io(source_error) => Error::Io {
                path: source.to_owned(),
                source: source_error,
            },
            StabilityError::Changed => Error::Invalid {
                path: source.to_owned(),
                message: "compiler input descriptor or path changed while it was copied".to_owned(),
            },
        }
    })
}

#[cfg(test)]
pub(in crate::lexicon) fn copy_regular_after_open(
    source: &Path,
    destination: &Path,
    after_open: impl FnOnce(),
) -> Result<()> {
    copy_regular_with(source, destination, after_open)
}

/// Builds and manifests every runtime file inside a fresh generation.
fn compile_generation(
    directory: &Path,
    meta: &mut serde_json::Value,
    generation: &str,
    strings: &StringTable,
    prepared_external: PreparedTable,
) -> Result<Compiled> {
    strings.write_index(directory)?;
    initialise_generation_meta(meta, directory, generation)?;
    let mut codecs = serde_json::Map::new();
    let mut entries = serde_json::Map::new();
    let mut compiled = compile_lookup_tables(
        directory,
        meta,
        strings,
        prepared_external,
        &mut codecs,
        &mut entries,
    )?;
    compiled = compiled
        .saturating_add(compile_phrase_table(directory, &mut entries)?)
        .saturating_add(compile_rule_table(directory, strings.len(), &mut entries)?);
    install_table_manifest(meta, directory, codecs, entries)?;
    manifest_generation_files(meta, directory, generation)?;

    Ok(Compiled {
        tables: compiled,
        from_sources: true,
    })
}

fn initialise_generation_meta(
    meta: &mut serde_json::Value,
    directory: &Path,
    generation: &str,
) -> Result<()> {
    let object = meta.as_object_mut().ok_or_else(|| Error::Invalid {
        path: directory.join("meta.json"),
        message: "metadata root must be a JSON object".to_owned(),
    })?;
    object.insert("format_version".to_owned(), FORMAT_VERSION.into());
    object.insert("entity_encoding".to_owned(), ENTITY_ENCODING.into());
    object.insert("generation".to_owned(), generation.into());
    object.insert("normalization".to_owned(), NORMALIZATION.into());
    object.insert(
        "files".to_owned(),
        serde_json::Value::Object(serde_json::Map::new()),
    );
    Ok(())
}

fn compile_lookup_tables(
    directory: &Path,
    meta: &serde_json::Value,
    strings: &StringTable,
    prepared_external: PreparedTable,
    codecs: &mut serde_json::Map<String, serde_json::Value>,
    entries: &mut serde_json::Map<String, serde_json::Value>,
) -> Result<usize> {
    let mut compiled = 0_usize;
    let mut prepared_external = Some(prepared_external);
    for table in LookupTableKind::ALL {
        let name = table.name();
        let prepared = if table == LookupTableKind::External {
            prepared_external.take().ok_or_else(|| Error::Invalid {
                path: directory.join("external.tsv"),
                message: "external table was prepared more than once".to_owned(),
            })?
        } else {
            let traffic = probe_share(meta, table, &directory.join("meta.json"))?;
            prepare_table(directory, table, strings, traffic)?
        };
        entries.insert(
            name.to_owned(),
            u64::try_from(prepared.entries()).unwrap_or(u64::MAX).into(),
        );
        if let Some(codec) = prepared.install(directory, table)? {
            codecs.insert(name.to_owned(), codec.name().into());
            compiled = compiled.saturating_add(1);
        }
    }
    Ok(compiled)
}

/// Compiles the phrase-pattern source into its single encoding; an empty
/// source means no file.
fn compile_phrase_table(
    directory: &Path,
    entries: &mut serde_json::Map<String, serde_json::Value>,
) -> Result<usize> {
    let source_path = directory.join("phrases.tsv");
    let mut rows = read_text_rows(&source_path)?;
    entries.insert(
        "phrases".to_owned(),
        u64::try_from(rows.len()).unwrap_or(u64::MAX).into(),
    );
    let destination = directory.join(format!("phrases.{PHRASE_CODEC}"));
    if rows.is_empty() {
        remove(&destination)?;
        return Ok(0);
    }
    let bytes = build_phrase_table(&mut rows, &source_path)?;
    write_file(&destination, &bytes)?;
    Ok(1)
}

fn compile_rule_table(
    directory: &Path,
    string_count: usize,
    entries: &mut serde_json::Map<String, serde_json::Value>,
) -> Result<usize> {
    let rows = read_rows(&directory.join("rules.tsv"))?;
    let keyset: Vec<(&[u8], u32)> = rows
        .iter()
        .enumerate()
        .map(|(index, (key, _))| {
            u32::try_from(index)
                .map(|index| (key.as_bytes(), index))
                .map_err(|_| Error::Invalid {
                    path: directory.join("rules.tsv"),
                    message: "rule count exceeds the double-array u32 limit".to_owned(),
                })
        })
        .collect::<Result<_>>()?;
    entries.insert(
        "rules".to_owned(),
        u64::try_from(rows.len()).unwrap_or(u64::MAX).into(),
    );
    entries.insert(
        "strings".to_owned(),
        u64::try_from(string_count).unwrap_or(u64::MAX).into(),
    );
    if write_trie(&directory.join("rules.da"), &keyset)? {
        let mut packed = Vec::with_capacity(rows.len().saturating_mul(VALUE_WIDTH));
        for (_, value) in &rows {
            packed.extend_from_slice(&value.to_le_bytes());
        }
        write_file(&directory.join("rules.values"), &packed)?;
        return Ok(1);
    }
    Ok(0)
}

fn install_table_manifest(
    meta: &mut serde_json::Value,
    directory: &Path,
    codecs: serde_json::Map<String, serde_json::Value>,
    entries: serde_json::Map<String, serde_json::Value>,
) -> Result<()> {
    let object = meta.as_object_mut().ok_or_else(|| Error::Invalid {
        path: directory.join("meta.json"),
        message: "metadata root must be a JSON object".to_owned(),
    })?;
    object.insert("codecs".to_owned(), serde_json::Value::Object(codecs));
    object.insert("entries".to_owned(), serde_json::Value::Object(entries));
    Ok(())
}

fn manifest_generation_files(
    meta: &mut serde_json::Value,
    directory: &Path,
    generation: &str,
) -> Result<()> {
    let preliminary: Meta = serde_json::from_value(meta.clone()).map_err(|source| Error::Meta {
        path: directory.join("meta.json"),
        source,
    })?;
    validate_meta(&preliminary, &directory.join("meta.json"), Some(generation))?;
    let mut files = BTreeMap::new();
    for name in runtime_files(&preliminary, directory)? {
        files.insert(name.clone(), file_integrity(&directory.join(name))?);
    }
    meta.as_object_mut()
        .ok_or_else(|| Error::Invalid {
            path: directory.join("meta.json"),
            message: "metadata root must be a JSON object".to_owned(),
        })?
        .insert(
            "files".to_owned(),
            serde_json::to_value(files).map_err(|source| Error::Meta {
                path: directory.join("meta.json"),
                source,
            })?,
        );
    write_file(&directory.join("meta.json"), &render_meta(meta))?;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::fmt::Write as _;

    use super::*;

    fn stage_sparse_external(root: &Path, lemma_suffix: &str) -> GenerationWriter {
        const EXTERNAL: usize = 4_096;

        std::fs::create_dir_all(root.parent().unwrap()).unwrap();
        let writer = GenerationWriter::begin(root).unwrap();
        let directory = writer.directory();

        let mut lemmas = vec![String::new(); EXTERNAL];
        let mut external = String::new();
        for index in 0..EXTERNAL {
            let form = format!("lemma{index:04}s");
            let lemma = format!("lemma{index:04}{lemma_suffix}");
            let id = index.saturating_mul(4_051) % EXTERNAL;
            *lemmas.get_mut(id).unwrap() = lemma;
            writeln!(external, "{form}\t{id}").unwrap();
        }
        let exact_id = lemmas.len();
        lemmas.push("known-lemma".to_owned());
        let strings = lemmas.join("\n") + "\n";
        write_file(&directory.join("strings.txt"), strings.as_bytes()).unwrap();
        write_file(
            &directory.join("exact.tsv"),
            format!("known\t{exact_id}\n").as_bytes(),
        )
        .unwrap();
        write_file(&directory.join("lower.tsv"), b"").unwrap();
        write_file(&directory.join("external.tsv"), external.as_bytes()).unwrap();
        write_file(&directory.join("entities.tsv"), b"").unwrap();
        write_file(&directory.join("phrases.tsv"), b"").unwrap();
        write_file(&directory.join("rules.tsv"), b"").unwrap();
        let name = root.file_name().unwrap().to_str().unwrap();
        write_meta(&writer, name, &serde_json::json!({"external": 0.0}));
        writer
    }

    /// Writes a minimal `lookup` metadata file into the staging directory.
    fn write_meta(writer: &GenerationWriter, name: &str, traffic: &serde_json::Value) {
        let meta = serde_json::json!({
            "name": name, "language": "Test", "iso639_3": "zxx",
            "source": "test.train", "training_tokens": 1, "variant": "lookup",
            "selected_on": null, "max_suffix": crate::format::MAX_SUFFIX_LENGTH,
            "format_version": FORMAT_VERSION, "entity_encoding": ENTITY_ENCODING,
            "generation": writer.id(), "normalization": NORMALIZATION,
            "files": {}, "entries": {}, "traffic": traffic,
            "gold_tokens": null, "accuracy": null, "baseline": null,
            "beats_baseline": null, "codecs": {}
        });
        write_file(&writer.directory().join("meta.json"), &render_meta(&meta)).unwrap();
    }

    fn publish_sparse_external(root: &Path, lemma_suffix: &str) -> String {
        stage_sparse_external(root, lemma_suffix)
            .compile_and_publish_identified()
            .unwrap()
            .1
    }

    #[test]
    fn external_inline_keeps_answers_while_reclaiming_sparse_strings() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("xx_compact");
        publish_sparse_external(&root, "");

        let resolved = resolve_generation(&root).unwrap();
        let compact =
            crate::lexicon::runtime::read_to_string(&resolved.directory.join("strings.txt"))
                .unwrap();
        assert_eq!(compact, "known-lemma\n");
        let lexicon = Lexicon::open(&root).unwrap();
        assert_eq!(
            lexicon.meta().codecs.get("external").map(String::as_str),
            Some("inline")
        );
        assert_eq!(lexicon.meta().entries.get("strings"), Some(&1));
        assert_eq!(lexicon.lemma("known", false), "known-lemma");
        assert_eq!(lexicon.lemma("lemma0042s", false), "lemma0042");
    }

    #[test]
    fn compacted_external_artifact_participates_in_generation_identity() {
        let temporary = tempfile::tempdir().unwrap();
        let left = temporary.path().join("left/xx_identity");
        let right = temporary.path().join("right/xx_identity");
        let left_generation = publish_sparse_external(&left, "");
        let right_generation = publish_sparse_external(&right, "x");

        // Compaction makes the sources identical; only the inline artifact
        // differs, so it must bind the content ID.
        let left_resolved = resolve_generation(&left).unwrap();
        let right_resolved = resolve_generation(&right).unwrap();
        for source in SOURCES.into_iter().chain(["strings.txt"]) {
            assert_eq!(
                std::fs::read(left_resolved.directory.join(source)).unwrap(),
                std::fs::read(right_resolved.directory.join(source)).unwrap()
            );
        }
        assert_ne!(left_generation, right_generation);
        assert_eq!(
            Lexicon::open(&left).unwrap().lemma("lemma0042s", false),
            "lemma0042"
        );
        assert_eq!(
            Lexicon::open(&right).unwrap().lemma("lemma0042s", false),
            "lemma0042x"
        );
    }

    #[test]
    fn early_external_failure_preserves_lookup_source_error_order() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("xx_error_order");
        let writer = GenerationWriter::begin(&root).unwrap();
        let directory = writer.directory();
        write_file(&directory.join("strings.txt"), b"lemma\n").unwrap();
        write_file(&directory.join("exact.tsv"), b"missing-tab\n").unwrap();
        for name in [
            "lower.tsv",
            "external.tsv",
            "entities.tsv",
            "phrases.tsv",
            "rules.tsv",
        ] {
            write_file(&directory.join(name), b"").unwrap();
        }
        write_meta(
            &writer,
            "xx_error_order",
            &serde_json::json!({"external": 2.0}),
        );

        let error = writer.compile_and_publish_identified().unwrap_err();
        assert!(
            matches!(&error, Error::Source { path, .. } if path.ends_with("exact.tsv")),
            "expected exact source error, got {error}"
        );
    }

    #[test]
    fn failed_compaction_preserves_phrase_before_rule_error_order() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("xx_compaction_error_order");
        let writer = stage_sparse_external(&root, "");
        let directory = writer.directory();
        write_file(&directory.join("phrases.tsv"), b"missing-tab\n").unwrap();
        write_file(&directory.join("rules.tsv"), b"missing-tab\n").unwrap();

        let error = writer.compile_and_publish_identified().unwrap_err();
        assert!(
            matches!(&error, Error::Source { path, .. } if path.ends_with("phrases.tsv")),
            "expected phrase source error, got {error}"
        );
    }

    #[test]
    fn legacy_migration_is_deterministic_and_recompiling_is_a_noop() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("xx_legacy_reproducible");
        let writer = GenerationWriter::begin(&root).unwrap();
        let directory = writer.directory();
        write_file(
            &directory.join("strings.txt"),
            "caf\u{e9}\ndog\n".as_bytes(),
        )
        .unwrap();
        // A decomposed, unsorted source forces the legacy fallback.
        write_file(
            &directory.join("exact.tsv"),
            "dogs\t1\ncafe\u{301}s\t0\n".as_bytes(),
        )
        .unwrap();
        for name in [
            "lower.tsv",
            "external.tsv",
            "entities.tsv",
            "phrases.tsv",
            "rules.tsv",
        ] {
            write_file(&directory.join(name), b"").unwrap();
        }
        write_meta(&writer, "xx_legacy_reproducible", &serde_json::json!({}));
        writer.publish().unwrap();

        let first = compile(&root).unwrap();
        assert!(first.from_sources);
        let generation = std::fs::read_to_string(root.join("current")).unwrap();
        let snapshot = || -> std::collections::BTreeMap<_, _> {
            std::fs::read_dir(resolve_generation(&root).unwrap().directory)
                .unwrap()
                .filter_map(std::result::Result::ok)
                .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_file()))
                .map(|entry| (entry.file_name(), std::fs::read(entry.path()).unwrap()))
                .collect()
        };
        let files_before = snapshot();

        let second = compile(&root).unwrap();
        assert_eq!(second.tables, 0);
        assert!(!second.from_sources);
        assert_eq!(
            std::fs::read_to_string(root.join("current")).unwrap(),
            generation
        );
        assert_eq!(files_before, snapshot());

        let lexicon = Lexicon::open(&root).unwrap();
        assert_eq!(lexicon.lemma("dogs", false), "dog");
        assert_eq!(lexicon.lemma("cafe\u{301}s", false), "caf\u{e9}");
    }
}
