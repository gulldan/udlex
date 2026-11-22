use std::borrow::Cow;
use std::collections::BTreeSet;
use std::fs::File;
use std::io::Read as _;
use std::mem::MaybeUninit;
use std::path::{Path, PathBuf};

use yada::DoubleArray;

use super::codec::{Codec, table_codec};
use super::metadata::{FileIntegrity, Meta, validate_meta};
use super::table::build_phrase_table;
use super::table::{
    Bytes, CachedMap, Entities, Found, HybridHit, HybridTable, Strings, Table, Tables,
    assemble_key, da_search, validate_tables,
};
use super::{CURRENT, Error, GENERATIONS, IDENTITY, PHRASE_CODEC, Result};
use crate::format::{LookupTableKind, TokenCase};
use crate::probe::{self, Options};
use crate::rule::Rule;
use crate::secure_fs::{MapVerifiedError, OpenRegularError, VerifiedMapping, valid_generation_id};

/// A loaded lexicon, ready to answer lemma queries.
pub struct Lexicon {
    tables: Tables,
    strings: Strings,
    meta: Meta,
    /// Resolved at open from the metadata; overridable only for measurement.
    conventions: probe::SpanConventions,
    /// Answers to the questions the probe asks of the lexicon itself.
    constants: Constants,
}

/// What the probe asks about the lexicon rather than about the token,
/// computed once at open instead of per token.
#[derive(Clone, Copy)]
#[allow(clippy::struct_excessive_bools)] // independent facts, not one state
struct Constants {
    casing: crate::normalization::Casing,
    max_suffix: usize,
    empty: bool,
    common_empty: bool,
    has_phrases: bool,
    surface_case: bool,
}

impl Constants {
    /// Reads every one of them off the tables that were just opened.
    fn of(tables: &Tables, meta: &Meta) -> Self {
        Self {
            casing: crate::normalization::Casing::for_language(&meta.iso639_3),
            // Metadata keeps the builder's bound even when no rules shipped.
            max_suffix: if tables.rules.is_some() {
                meta.max_suffix
            } else {
                0
            },
            empty: tables.is_empty(),
            common_empty: tables.common_is_empty(),
            has_phrases: tables.phrases.is_some(),
            surface_case: meta.token_case.as_deref().and_then(TokenCase::parse)
                == Some(TokenCase::Surface),
        }
    }
}

/// Metadata is a small control file, never a vocabulary payload.
pub(super) const MAX_METADATA_BYTES: u64 = 4 * 1024 * 1024;

/// Stack room for a rule key: one bucket byte and the suffix after it; a
/// longer key spills to the heap.
const RULE_KEY_BYTES: usize = 64;

/// Opens a regular file without following a final symlink.
fn open_regular(path: &Path) -> Result<File> {
    crate::secure_fs::open_regular(path)
        .map(|(file, _metadata)| file)
        .map_err(|error| match error {
            OpenRegularError::Io(source) => Error::Io {
                path: path.to_owned(),
                source,
            },
            OpenRegularError::NotRegular => Error::Invalid {
                path: path.to_owned(),
                message: "expected a regular file, not a symlink or special file".to_owned(),
            },
            OpenRegularError::Changed => Error::Invalid {
                path: path.to_owned(),
                message: "file path changed while it was opened".to_owned(),
            },
        })
}

/// Reads a whole regular file, tagging any error with the path.
pub fn read_to_string(path: &Path) -> Result<String> {
    let mut file = open_regular(path)?;
    let mut value = String::new();
    file.read_to_string(&mut value)
        .map_err(|source| Error::Io {
            path: path.to_owned(),
            source,
        })?;
    Ok(value)
}

/// Reads one bounded regular control file without trusting its declared size.
pub fn read_bounded_regular(path: &Path, maximum: u64, description: &str) -> Result<String> {
    let file = open_regular(path)?;
    let size = file
        .metadata()
        .map_err(|source| Error::Io {
            path: path.to_owned(),
            source,
        })?
        .len();
    if size > maximum {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: format!("{description} exceeds {maximum} bytes"),
        });
    }
    let mut value = String::with_capacity(usize::try_from(size).unwrap_or(0));
    file.take(maximum.saturating_add(1))
        .read_to_string(&mut value)
        .map_err(|source| Error::Io {
            path: path.to_owned(),
            source,
        })?;
    if value.len() as u64 > maximum {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: format!("{description} exceeds {maximum} bytes"),
        });
    }
    Ok(value)
}

/// Reads bounded metadata without allocating from an attacker-controlled size.
pub(super) fn read_metadata(path: &Path) -> Result<String> {
    read_bounded_regular(path, MAX_METADATA_BYTES, "metadata")
}

/// Authenticates and memory-maps the same regular-file descriptor.
pub fn map_verified_file(path: &Path, expected: &FileIntegrity) -> Result<Bytes> {
    let expected_integrity = crate::secure_fs::Integrity {
        size: expected.size,
        sha256: expected.sha256.clone(),
    };
    crate::secure_fs::map_verified(path, &expected_integrity)
        .map(|mapping| match mapping {
            VerifiedMapping::Empty => Bytes::Static(&[]),
            VerifiedMapping::Mapped(mapping) => Bytes::Mapped(mapping),
        })
        .map_err(|error| map_verified_error(path, expected, error))
}

fn map_verified_error(path: &Path, expected: &FileIntegrity, error: MapVerifiedError) -> Error {
    match error {
        MapVerifiedError::Open(OpenRegularError::Io(source))
        | MapVerifiedError::Read(source)
        | MapVerifiedError::Map(source) => Error::Io {
            path: path.to_owned(),
            source,
        },
        MapVerifiedError::Open(OpenRegularError::NotRegular) => Error::Invalid {
            path: path.to_owned(),
            message: "expected a regular file, not a symlink or special file".to_owned(),
        },
        MapVerifiedError::Open(OpenRegularError::Changed) | MapVerifiedError::Changed => {
            Error::Invalid {
                path: path.to_owned(),
                message: "file path or contents changed while it was read".to_owned(),
            }
        }
        MapVerifiedError::LengthOverflow => Error::Integrity {
            path: path.to_owned(),
            message: "file length overflowed u64 while hashing".to_owned(),
        },
        MapVerifiedError::Mismatch(observed) => Error::Integrity {
            path: path.to_owned(),
            message: format!("observed {observed:?}, expected {expected:?}"),
        },
    }
}

/// A root resolved to one immutable generation exactly once.
pub(super) struct ResolvedGeneration {
    pub(super) root: PathBuf,
    pub(super) directory: PathBuf,
    pub(super) generation: Option<String>,
}

/// A resolved generation and metadata validated for that exact directory.
pub(super) struct LoadedMetadata {
    pub(super) resolved: ResolvedGeneration,
    pub(super) meta: Meta,
}

/// Resolves a root's atomic `current` pointer without following symlinks.
pub(super) fn resolve_generation(root: &Path) -> Result<ResolvedGeneration> {
    let metadata = std::fs::symlink_metadata(root).map_err(|source| Error::Io {
        path: root.to_owned(),
        source,
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(Error::Invalid {
            path: root.to_owned(),
            message: "lexicon root must be a real directory, not a symlink".to_owned(),
        });
    }
    let current = root.join(CURRENT);
    let current_meta = match std::fs::symlink_metadata(&current) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(ResolvedGeneration {
                root: root.to_owned(),
                directory: root.to_owned(),
                generation: None,
            });
        }
        Err(source) => {
            return Err(Error::Io {
                path: current,
                source,
            });
        }
    };
    if current_meta.file_type().is_symlink() || !current_meta.is_file() || current_meta.len() > 129
    {
        return Err(Error::Invalid {
            path: current,
            message: "current must be a regular text file no larger than 129 bytes".to_owned(),
        });
    }
    let raw = read_to_string(&current)?;
    let generation = raw.trim_end_matches(['\r', '\n']);
    if !valid_generation_id(generation) || raw.trim() != generation {
        return Err(Error::Invalid {
            path: current,
            message: format!("invalid generation identifier {raw:?}"),
        });
    }
    let generations = root.join(GENERATIONS);
    let directory = generations.join(generation);
    for path in [&generations, &directory] {
        let metadata = std::fs::symlink_metadata(path).map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(Error::Invalid {
                path: path.clone(),
                message: "generation path must be a real directory".to_owned(),
            });
        }
    }
    Ok(ResolvedGeneration {
        root: root.to_owned(),
        directory,
        generation: Some(generation.to_owned()),
    })
}

/// Resolves one root and parses its bounded metadata exactly once.
pub(super) fn load_metadata(root: &Path) -> Result<LoadedMetadata> {
    let resolved = resolve_generation(root)?;
    let meta_path = resolved.directory.join("meta.json");
    let meta: Meta =
        serde_json::from_str(&read_metadata(&meta_path)?).map_err(|source| Error::Meta {
            path: meta_path.clone(),
            source,
        })?;
    validate_meta(&meta, &meta_path, resolved.generation.as_deref())?;
    if resolved.generation.is_some()
        && resolved.root.file_name().and_then(|name| name.to_str()) != Some(&meta.name)
    {
        return Err(Error::Invalid {
            path: resolved.root,
            message: format!(
                "directory name does not match metadata name {:?}",
                meta.name
            ),
        });
    }
    Ok(LoadedMetadata { resolved, meta })
}

/// Streams one file through SHA-256 without materialising it.
#[cfg(feature = "train")]
pub(super) fn file_integrity(path: &Path) -> Result<FileIntegrity> {
    crate::secure_fs::file_integrity(path)
        .map(|observed| FileIntegrity {
            size: observed.size,
            sha256: observed.sha256,
        })
        .map_err(|error| {
            map_verified_error(
                path,
                &FileIntegrity {
                    size: 0,
                    sha256: String::new(),
                },
                error,
            )
        })
}

/// Runtime files implied by metadata, excluding metadata itself.
pub(super) fn runtime_files(meta: &Meta, directory: &Path) -> Result<BTreeSet<String>> {
    let mut files = BTreeSet::from(["strings.idx".to_owned(), "strings.txt".to_owned()]);
    for table in LookupTableKind::ALL {
        let name = table.name();
        if meta.entries.get(name).copied().unwrap_or(0) > 0 {
            let codec = table_codec(&meta.codecs, directory, name)?;
            files.insert(format!("{name}.{}", codec.name()));
        }
    }
    if meta.entries.get("phrases").copied().unwrap_or(0) > 0 {
        files.insert(format!("phrases.{PHRASE_CODEC}"));
    }
    if meta.entries.get("rules").copied().unwrap_or(0) > 0 {
        files.insert("rules.da".to_owned());
        files.insert("rules.values".to_owned());
    }
    Ok(files)
}

/// Verifies that metadata declares exactly the files the runtime will open.
fn verify_file_manifest(directory: &Path, meta: &Meta) -> Result<()> {
    let expected = runtime_files(meta, directory)?;
    let declared: BTreeSet<String> = meta.files.keys().cloned().collect();
    if declared != expected {
        return Err(Error::Integrity {
            path: directory.join("meta.json"),
            message: format!("declared files {declared:?}, expected {expected:?}"),
        });
    }
    Ok(())
}

/// Returns one runtime file's authenticated manifest entry.
pub(super) fn expected_file<'a>(
    directory: &Path,
    meta: &'a Meta,
    name: &str,
) -> Result<&'a FileIntegrity> {
    meta.files.get(name).ok_or_else(|| Error::Integrity {
        path: directory.join(name),
        message: "file is absent from metadata".to_owned(),
    })
}

/// Wraps bytes as a double-array trie, tagging any error with a name.
fn open_trie(name: &Path, bytes: Bytes) -> Result<DoubleArray<Bytes>> {
    DoubleArray::new(bytes).map_err(|source| Error::Trie {
        path: name.to_owned(),
        source,
    })
}

/// Tags an `fst` error with the file it concerns.
pub fn fst_error(path: &Path, source: fst::Error) -> Error {
    Error::Fst {
        path: path.to_owned(),
        source,
    }
}

/// Wraps bytes as a table under the given codec.
pub fn open_encoded(path: &Path, codec: Codec, bytes: Bytes) -> Result<Table> {
    match codec {
        Codec::Da => open_trie(path, bytes).map(Table::Da),
        Codec::Map => {
            let map = fst::Map::new(bytes).map_err(|source| fst_error(path, source))?;
            map.as_fst()
                .verify()
                .map_err(|source| fst_error(path, source))?;
            Ok(Table::Map(CachedMap::new(map)))
        }
        Codec::Inline => {
            let fsa = fst::raw::Fst::new(bytes).map_err(|source| fst_error(path, source))?;
            fsa.verify().map_err(|source| fst_error(path, source))?;
            Ok(Table::Inline(fsa))
        }
        Codec::Hybrid => HybridTable::open(bytes)
            .map(Table::Hybrid)
            .map_err(|message| Error::Invalid {
                path: path.to_owned(),
                message,
            }),
    }
}

/// Loads one lookup table from the bundle directory, under the codec the
/// metadata records for it.
fn open_table(directory: &Path, table: LookupTableKind, meta: &Meta) -> Result<Option<Table>> {
    let name = table.name();
    if meta.entries.get(name).copied().unwrap_or(0) == 0 {
        return Ok(None);
    }
    let codec = table_codec(&meta.codecs, directory, name)?;
    let path = codec.file(directory, name);
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| Error::Integrity {
            path: path.clone(),
            message: "runtime file name is not valid UTF-8".to_owned(),
        })?;
    let bytes = map_verified_file(&path, expected_file(directory, meta, file_name)?)?;
    open_encoded(&path, codec, bytes).map(Some)
}

/// Names the entity table's file without reading a byte of it.
///
/// Only presence and length are checked here; the digest and mapping wait
/// for the first entity or span query.
fn plan_entities(directory: &Path, meta: &Meta) -> Result<Entities> {
    let name = LookupTableKind::Entities.name();
    let declared = meta.entries.get(name).copied().unwrap_or(0);
    if declared == 0 {
        return Ok(Entities::default());
    }
    let codec = table_codec(&meta.codecs, directory, name)?;
    let path = codec.file(directory, name);
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| Error::Integrity {
            path: path.clone(),
            message: "runtime file name is not valid UTF-8".to_owned(),
        })?;
    let expected = expected_file(directory, meta, file_name)?.clone();
    let observed = std::fs::symlink_metadata(&path).map_err(|source| Error::Io {
        path: path.clone(),
        source,
    })?;
    if observed.file_type().is_symlink() || !observed.is_file() || observed.len() != expected.size {
        return Err(Error::Integrity {
            path: path.clone(),
            message: format!(
                "expected a regular file of {} bytes, found {}",
                expected.size,
                observed.len()
            ),
        });
    }
    Ok(Entities::deferred(path, codec, expected, declared))
}

/// Loads the phrase-pattern table, which is always the inline acceptor.
///
/// Read eagerly: it is tens of kilobytes, and the span path then reads a
/// plain `Option` instead of a lazy cell.
fn open_phrases(directory: &Path, meta: &Meta) -> Result<Option<Table>> {
    if meta.entries.get("phrases").copied().unwrap_or(0) == 0 {
        return Ok(None);
    }
    let name = format!("phrases.{PHRASE_CODEC}");
    let path = directory.join(&name);
    let bytes = map_verified_file(&path, expected_file(directory, meta, &name)?)?;
    open_encoded(&path, Codec::Inline, bytes).map(Some)
}

/// Loads the rule table, which is always a double array.
fn open_rules(directory: &Path, meta: &Meta) -> Result<Option<DoubleArray<Bytes>>> {
    if meta.entries.get("rules").copied().unwrap_or(0) == 0 {
        return Ok(None);
    }
    let path = directory.join("rules.da");
    let bytes = map_verified_file(&path, expected_file(directory, meta, "rules.da")?)?;
    open_trie(&path, bytes).map(Some)
}

/// Wraps embedded bytes as a table.
fn embedded_table(
    table: LookupTableKind,
    bytes: Option<&'static [u8]>,
    meta: &Meta,
) -> Result<Option<Table>> {
    let name = table.name();
    let directory = Path::new("<embedded>");
    let codec = table_codec(&meta.codecs, directory, name)?;
    bytes
        .map(|bytes| open_encoded(&codec.file(directory, name), codec, Bytes::Static(bytes)))
        .transpose()
}

/// A lexicon compiled into the binary rather than read from disk.
///
/// Build one with `include_bytes!` and `include_str!` over a single bundle
/// directory, and the result needs no files at run time at all. Tables a
/// language does not have are `None`. Each lookup table's file name carries
/// its codec — `.da`, `.map`, `.inline` or `.hybrid` — as recorded under `codecs` in the
/// bundle's `meta.json`; include whichever file the compiled bundle holds:
///
/// ```ignore
/// use udlex_rs::{Embedded, Lexicon};
///
/// const RUSSIAN: Embedded = Embedded {
///     exact: Some(include_bytes!("../bundle/ru_taiga/exact.da")),
///     lower: Some(include_bytes!("../bundle/ru_taiga/lower.da")),
///     external: Some(include_bytes!("../bundle/ru_taiga/external.map")),
///     entities: None,
///     rules: Some(include_bytes!("../bundle/ru_taiga/rules.da")),
///     rule_values: Some(include_bytes!("../bundle/ru_taiga/rules.values")),
///     strings: include_str!("../bundle/ru_taiga/strings.txt"),
///     meta: include_str!("../bundle/ru_taiga/meta.json"),
/// };
///
/// let lexicon = Lexicon::embedded(RUSSIAN)?;
/// # Ok::<(), udlex_rs::Error>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Embedded {
    /// Contents of the compiled `exact` table.
    pub exact: Option<&'static [u8]>,
    /// Contents of the compiled `lower` table.
    pub lower: Option<&'static [u8]>,
    /// Contents of the compiled `external` table, when the language has one.
    pub external: Option<&'static [u8]>,
    /// Contents of the curated named-entity table, when present.
    pub entities: Option<&'static [u8]>,
    /// Contents of the compiled phrase-pattern table, when present.
    pub phrases: Option<&'static [u8]>,
    /// Contents of `rules.da`.
    pub rules: Option<&'static [u8]>,
    /// Contents of `rules.values`.
    pub rule_values: Option<&'static [u8]>,
    /// Contents of `strings.txt`.
    pub strings: &'static str,
    /// Contents of `meta.json`.
    pub meta: &'static str,
}

/// Prints provenance and which tables are loaded, never their contents.
impl std::fmt::Debug for Lexicon {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Lexicon")
            .field("name", &self.meta.name)
            .field("iso639_3", &self.meta.iso639_3)
            .field("variant", &self.meta.variant)
            .field("tables", &self.tables)
            .finish_non_exhaustive()
    }
}

impl Lexicon {
    /// Opens a compiled lexicon directory.
    ///
    /// # Errors
    ///
    /// Returns an error when a table is unreadable or written by an
    /// incompatible builder.
    pub fn open(directory: impl AsRef<Path>) -> Result<Self> {
        Self::open_loaded(load_metadata(directory.as_ref())?)
    }

    /// Opens a generation whose metadata was already resolved and validated.
    pub(super) fn open_loaded(loaded: LoadedMetadata) -> Result<Self> {
        let LoadedMetadata { resolved, meta } = loaded;
        let directory = resolved.directory.as_path();
        if meta.files.is_empty() {
            return Err(Error::NotCompiled {
                path: directory.to_owned(),
            });
        }
        verify_file_manifest(directory, &meta)?;

        let tables = Tables {
            exact: open_table(directory, LookupTableKind::Exact, &meta)?,
            lower: open_table(directory, LookupTableKind::Lower, &meta)?,
            external: open_table(directory, LookupTableKind::External, &meta)?,
            entities: plan_entities(directory, &meta)?,
            phrases: open_phrases(directory, &meta)?,
            rules: open_rules(directory, &meta)?,
            rule_values: if meta.entries.get("rules").copied().unwrap_or(0) == 0 {
                None
            } else {
                let path = directory.join("rules.values");
                Some(map_verified_file(
                    &path,
                    expected_file(directory, &meta, "rules.values")?,
                )?)
            },
        };
        // Only `identity` may be empty on purpose; anything else was never compiled.
        if meta.variant != IDENTITY && tables.is_empty() {
            return Err(Error::NotCompiled {
                path: directory.to_owned(),
            });
        }

        let strings = Strings::open(directory, &meta)?;
        validate_tables(directory, &tables, &strings, &meta)?;
        Ok(Self {
            constants: Constants::of(&tables, &meta),
            tables,
            strings,
            conventions: meta.span_conventions(),
            meta,
        })
    }

    /// Builds a lexicon from tables compiled into the binary.
    ///
    /// # Errors
    ///
    /// Returns an error when the embedded metadata is malformed or names an
    /// incompatible format version, or when a table's bytes do not parse
    /// under the codec the metadata records for it.
    pub fn embedded(source: Embedded) -> Result<Self> {
        let meta: Meta = serde_json::from_str(source.meta).map_err(|error| Error::Meta {
            path: PathBuf::from("<embedded>/meta.json"),
            source: error,
        })?;
        validate_meta(&meta, Path::new("<embedded>/meta.json"), None)?;
        let tables = Tables {
            exact: embedded_table(LookupTableKind::Exact, source.exact, &meta)?,
            lower: embedded_table(LookupTableKind::Lower, source.lower, &meta)?,
            external: embedded_table(LookupTableKind::External, source.external, &meta)?,
            // Embedded bytes need no deferred authentication.
            entities: embedded_table(LookupTableKind::Entities, source.entities, &meta)?
                .map_or_else(Entities::default, Entities::ready),
            phrases: source
                .phrases
                .map(|bytes| {
                    open_encoded(
                        Path::new("<embedded>/phrases.inline"),
                        Codec::Inline,
                        Bytes::Static(bytes),
                    )
                })
                .transpose()?,
            rules: source
                .rules
                .map(|bytes| open_trie(Path::new("<embedded>/rules.da"), Bytes::Static(bytes)))
                .transpose()?,
            rule_values: source.rule_values.map(Bytes::Static),
        };
        let strings = Strings::from_static(source.strings)?;
        validate_tables(Path::new("<embedded>"), &tables, &strings, &meta)?;
        Ok(Self {
            constants: Constants::of(&tables, &meta),
            tables,
            strings,
            conventions: meta.span_conventions(),
            meta,
        })
    }

    /// The span conventions currently in force for this lexicon.
    #[must_use]
    pub const fn span_conventions(&self) -> probe::SpanConventions {
        self.conventions
    }

    /// Authenticates the isolated entity table now.
    ///
    /// [`Lexicon::open`] defers it. Call this before any entity or span query
    /// whose failure must be visible: the probes cannot report errors and read
    /// a broken table as empty. Failures are not memoised.
    ///
    /// # Errors
    ///
    /// Returns the integrity, codec or count error the entity table carries.
    pub fn ensure_entities(&self) -> Result<()> {
        self.tables.entities.get().map(|_| ())
    }

    /// Whether an entity or span query can proceed without opening anything;
    /// cheap enough to guard [`Lexicon::ensure_entities`] per call.
    #[must_use]
    pub fn entities_ready(&self) -> bool {
        self.tables.entities.is_ready()
    }

    /// Opens the entity table if these options can reach it, ignoring failure.
    ///
    /// Kept outside the probe, whose inlining it would break (see [`Entities`]);
    /// [`Lexicon::ensure_entities`] reports a damaged table.
    #[inline]
    pub(crate) fn prime_entities(&self, options: Options) {
        if (options.use_entities || options.use_entities_lower) && !self.tables.entities.is_ready()
        {
            self.tables.entities.prime();
        }
    }

    /// Overrides the metadata's span conventions, for measuring a candidate.
    pub const fn set_span_conventions(&mut self, conventions: probe::SpanConventions) {
        self.conventions = conventions;
    }

    /// Replaces the compiled phrase-pattern table with candidate rows, for
    /// measuring a candidate; an empty slice removes the table.
    ///
    /// # Errors
    ///
    /// Returns an error when a row is empty, duplicated, carries a NUL, or the
    /// acceptor refuses the key set.
    #[deprecated(
        note = "candidate phrase compilation is a training concern; use a compiled bundle"
    )]
    pub fn set_phrase_patterns(&mut self, rows: &mut [(String, String)]) -> Result<()> {
        if rows.is_empty() {
            self.tables.phrases = None;
            self.constants.has_phrases = false;
            return Ok(());
        }
        let path = PathBuf::from("<override>/phrases.inline");
        let bytes = build_phrase_table(rows, &path)?;
        self.tables.phrases = Some(open_encoded(&path, Codec::Inline, Bytes::Owned(bytes))?);
        self.constants.has_phrases = true;
        Ok(())
    }

    /// Provenance of this lexicon.
    #[must_use]
    pub const fn meta(&self) -> &Meta {
        &self.meta
    }

    /// Whether independent scoring matched or exceeded copying the surface form.
    ///
    /// `None` means the installed generation carries no independent judgement
    /// at all — neither a held-out measurement nor a selection receipt.
    /// Unknown quality is never promoted to a positive verdict.
    #[must_use]
    pub fn is_useful(&self) -> Option<bool> {
        self.meta.useful()
    }

    /// Resolves a string identifier stored in a table.
    #[expect(clippy::inline_always, reason = "part of the inlined hybrid read")]
    #[inline(always)]
    fn string(&self, id: u64) -> Option<&str> {
        self.strings.entry(id)
    }

    /// Turns what a table produced into a lemma.
    fn resolve<'a>(&'a self, found: Found<'a>) -> Option<Cow<'a, str>> {
        match found {
            Found::Id(id) => Some(Cow::Borrowed(self.string(id)?)),
            Found::Lemma(lemma) => Some(Cow::Owned(lemma)),
            Found::Text(text) => Some(Cow::Borrowed(text)),
        }
    }

    /// One `exact`/`lower` probe. The compiler writes both tables as
    /// `hybrid`, so that codec is read inside the probe; [`Table::get`]
    /// stays out of line for every other codec.
    #[expect(
        clippy::inline_always,
        reason = "behind a call, the hybrid read costs ~30 more instructions"
    )]
    #[inline(always)]
    fn read_hot<'a>(&'a self, table: &'a Table, key: &str) -> Option<Cow<'a, str>> {
        if let Table::Hybrid(hybrid) = table {
            return Some(Cow::Borrowed(match hybrid.get(key.as_bytes())? {
                HybridHit::Id(id) => self.string(u64::from(id))?,
                HybridHit::Text(text) => text,
            }));
        }
        self.resolve(table.get(key)?)
    }

    /// The profile the span path of this generation was accepted with:
    /// [`Options::default`] plus the measured lowercase entity retry. Tokens
    /// keep [`Options::default`].
    #[must_use]
    pub fn span_options(&self) -> Options {
        Options {
            use_entities_lower: self.meta.entity_lower,
            ..Options::default()
        }
    }

    /// Whether the token path follows the measured `token_case = surface`
    /// convention; the span path never does, it was measured on tokens only.
    #[must_use]
    pub const fn surface_case(&self) -> bool {
        self.constants.surface_case
    }

    /// Overrides the generation's token case convention, for the acceptance
    /// machinery that scores a candidate on the lexicon that would receive it.
    pub const fn set_surface_case(&mut self, surface: bool) {
        self.constants.surface_case = surface;
    }

    /// Recases one token answer under `token_case = surface`.
    ///
    /// Cold and out of line so the token probe keeps its single call site.
    #[cold]
    #[inline(never)]
    pub(crate) fn recased<'a>(
        &self,
        mut answer: probe::Answer<'a>,
        token: &str,
        sentence_initial: bool,
    ) -> probe::Answer<'a> {
        probe::recase_surface(&mut answer, token, sentence_initial, self.constants.casing);
        answer
    }

    #[inline]
    pub(crate) fn lemma_recased<'a>(
        &'a self,
        token: &'a str,
        sentence_initial: bool,
        options: Options,
    ) -> Cow<'a, str> {
        self.prime_entities(options);
        let mut answer = probe::lemma_with_info_stamped(self, token, sentence_initial, options);
        probe::recase_surface(&mut answer, token, sentence_initial, self.constants.casing);
        answer.lemma
    }

    /// Returns the lemma of one token.
    ///
    /// The returned value borrows the token when nothing is known about it, so
    /// the common "leave it alone" path costs no allocation.
    #[must_use]
    pub fn lemma<'a>(&'a self, token: &'a str, sentence_initial: bool) -> Cow<'a, str> {
        self.lemma_with(token, sentence_initial, Options::default())
    }

    /// Returns the lemma of one token, choosing which tables to consult.
    #[must_use]
    pub fn lemma_with<'a>(
        &'a self,
        token: &'a str,
        sentence_initial: bool,
        options: Options,
    ) -> Cow<'a, str> {
        if self.constants.surface_case {
            return self.lemma_recased(token, sentence_initial, options);
        }
        self.lemma_with_plain(token, sentence_initial, options)
    }

    /// [`Lexicon::lemma_with`] without the stamp check, for batch loops that
    /// check it once.
    #[inline]
    pub(crate) fn lemma_with_plain<'a>(
        &'a self,
        token: &'a str,
        sentence_initial: bool,
        options: Options,
    ) -> Cow<'a, str> {
        self.prime_entities(options);
        probe::lemma(self, token, sentence_initial, options)
    }

    /// Returns a lemma together with the exact probe stage that produced it.
    #[must_use]
    pub fn lemma_with_info<'a>(
        &'a self,
        token: &'a str,
        sentence_initial: bool,
        options: Options,
    ) -> probe::Answer<'a> {
        self.prime_entities(options);
        let answer = probe::lemma_with_info(self, token, sentence_initial, options);
        if self.constants.surface_case {
            return self.recased(answer, token, sentence_initial);
        }
        answer
    }

    /// Returns the canonical text of one entity span, with provenance.
    ///
    /// `span` is the exact surface, rebuilt by the caller from `tokens` and
    /// `separators`; an unanswered span comes back byte for byte.
    #[must_use]
    pub fn lemma_span_with_info<'a>(
        &'a self,
        span: &'a str,
        tokens: &[&str],
        separators: Option<&[&str]>,
        sentence_initial: bool,
        options: Options,
    ) -> probe::Answer<'a> {
        // The span probe always asks the entity table.
        if !options.protected && !self.tables.entities.is_ready() {
            self.tables.entities.prime();
        }
        probe::span_lemma_with_info(self, span, tokens, separators, sentence_initial, options)
    }

    /// The same span query for a mention given as one string, split by
    /// [`crate::span::split_surface`] with its original gaps.
    ///
    /// # Errors
    ///
    /// Returns [`crate::span::SurfaceError`] when the surface is empty,
    /// whitespace only, or carries leading or trailing whitespace.
    pub fn lemma_span_text_with_info<'a>(
        &'a self,
        text: &'a str,
        sentence_initial: bool,
        options: Options,
    ) -> std::result::Result<probe::Answer<'a>, crate::span::SurfaceError> {
        let (tokens, gaps) = crate::span::split_surface(text)?;
        let separators = (!gaps.is_empty()).then_some(gaps.as_slice());
        Ok(self.lemma_span_with_info(text, &tokens, separators, sentence_initial, options))
    }

    /// Whether a lookup table answers this form, suffix rules aside.
    #[must_use]
    pub fn knows(&self, token: &str) -> bool {
        self.knows_with(token, false, Options::default())
    }

    /// Whether one of the enabled stored-vocabulary probes answers this form.
    #[must_use]
    pub fn knows_with(&self, token: &str, sentence_initial: bool, mut options: Options) -> bool {
        options.use_rules = false;
        self.prime_entities(options);
        probe::answered(self, token, sentence_initial, options)
    }

    /// Lemmatises a whole sentence, tracking which token opens it.
    #[must_use]
    pub fn lemmatize<'a>(&'a self, tokens: &[&'a str]) -> Vec<Cow<'a, str>> {
        // Resolve the token case convention once for the whole sentence.
        if self.constants.surface_case {
            return self.lemmatize_recased(tokens);
        }
        let options = Options::default();
        self.prime_entities(options);
        probe::lemmas(self, tokens)
    }

    #[inline(never)]
    fn lemmatize_recased<'a>(&'a self, tokens: &[&'a str]) -> Vec<Cow<'a, str>> {
        tokens
            .iter()
            .enumerate()
            .map(|(index, token)| self.lemma_recased(token, index == 0, Options::default()))
            .collect()
    }

    /// Lemmatises a whole sentence and retains each answer's provenance; the
    /// first token is sentence-initial.
    #[must_use]
    pub fn lemmatize_with_info<'a>(&'a self, tokens: &[&'a str]) -> Vec<probe::Answer<'a>> {
        tokens
            .iter()
            .enumerate()
            .map(|(index, token)| self.lemma_with_info(token, index == 0, Options::default()))
            .collect()
    }

    /// [`Lexicon::lemma_with_info`] with what the caller knows of the
    /// sentence: an exact answer gives way to a learned context key.
    ///
    /// Runs on the probe's cold copy, so the plain token paths keep theirs.
    #[must_use]
    pub fn lemma_in_context<'a>(
        &'a self,
        token: &'a str,
        sentence_initial: bool,
        options: Options,
        upos: Option<&str>,
        previous: Option<&str>,
        next: Option<&str>,
    ) -> probe::Answer<'a> {
        self.prime_entities(options);
        let mut answer = probe::lemma_with_info_contextual(self, token, sentence_initial, options);
        if self.constants.surface_case {
            answer = self.recased(answer, token, sentence_initial);
        }
        probe::in_context(self, answer, token, upos, previous, next)
    }

    /// [`Lexicon::lemmatize_with_info`] with context: one `upos` tag per
    /// token (a missing one skips its key) and, with `neighbours`, the keys
    /// of the previous and next word.
    #[must_use]
    pub fn lemmatize_in_context<'a>(
        &'a self,
        tokens: &[&'a str],
        upos: Option<&[&str]>,
        neighbours: bool,
    ) -> Vec<probe::Answer<'a>> {
        tokens
            .iter()
            .enumerate()
            .map(|(index, token)| {
                let (previous, next) = crate::format::neighbours(tokens, index, |at| at == 0);
                self.lemma_in_context(
                    token,
                    index == 0,
                    Options::default(),
                    upos.and_then(|tags| tags.get(index).copied()),
                    neighbours.then_some(previous),
                    neighbours.then_some(next),
                )
            })
            .collect()
    }
}

/// The compiled tables; a hit borrows from the string table, never allocates.
impl probe::Tables for Lexicon {
    #[expect(clippy::inline_always, reason = "see `Lexicon::read_hot`")]
    #[inline(always)]
    fn exact(&self, key: &str) -> Option<Cow<'_, str>> {
        self.read_hot(self.tables.exact.as_ref()?, key)
    }

    #[expect(clippy::inline_always, reason = "see `Lexicon::read_hot`")]
    #[inline(always)]
    fn lower(&self, key: &str) -> Option<Cow<'_, str>> {
        self.read_hot(self.tables.lower.as_ref()?, key)
    }

    fn external(&self, key: &str) -> Option<Cow<'_, str>> {
        self.resolve(self.tables.external.as_ref()?.get(key)?)
    }

    /// Reads the entity table, which by then is open or is not there.
    ///
    /// `#[inline(never)]` is measured: inlining the lazy cell into the generic
    /// probe cost `ru_taiga` 8 % of token throughput.
    #[inline(never)]
    fn entities(&self, key: &str) -> Option<Cow<'_, str>> {
        let table = self.tables.entities.for_probe()?;
        self.resolve(table.get_prefixed(b'0', key)?)
    }

    /// Not inlined, as `entities`.
    #[inline(never)]
    fn entities_lower(&self, key: &str) -> Option<Cow<'_, str>> {
        let table = self.tables.entities.for_probe()?;
        self.resolve(table.get_prefixed(b'1', key)?)
    }

    fn has_phrases(&self) -> bool {
        self.constants.has_phrases
    }

    /// Reads the phrase table; not inlined, as `entities`.
    #[inline(never)]
    fn phrase(&self, key: &str) -> Option<Cow<'_, str>> {
        self.resolve(self.tables.phrases.as_ref()?.get(key)?)
    }

    /// The key is `[bucket, suffix...]` in an unzeroed stack buffer:
    /// `assemble_key` writes every byte the walk reads.
    fn rule(&self, bucket: u8, suffix: &str) -> Option<(Rule, Cow<'_, str>)> {
        let rules = self.tables.rules.as_ref()?;
        let mut stack = [MaybeUninit::<u8>::uninit(); RULE_KEY_BYTES];
        let mut spill: Vec<u8>;
        // `map_or_else` cannot borrow a local that one arm creates.
        #[allow(clippy::option_if_let_else)]
        let key: &[u8] = if let Some(room) = assemble_key(bucket, suffix.as_bytes(), &mut stack) {
            room
        } else {
            spill = Vec::with_capacity(suffix.len().saturating_add(1));
            spill.push(bucket);
            spill.extend_from_slice(suffix.as_bytes());
            &spill
        };

        let index = da_search(rules, key)?;
        let rule = self.tables.rule(index)?;
        let add = Cow::Borrowed(self.string(rule.add)?);
        Some((rule, add))
    }

    fn is_empty(&self) -> bool {
        self.constants.empty
    }

    fn common_is_empty(&self) -> bool {
        self.constants.common_empty
    }

    /// Zero when the lexicon ships no rule table; see [`Constants::of`].
    fn max_suffix(&self) -> usize {
        self.constants.max_suffix
    }

    fn casing(&self) -> crate::normalization::Casing {
        self.constants.casing
    }

    fn span_conventions(&self) -> probe::SpanConventions {
        self.conventions
    }
}
