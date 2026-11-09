//! Independent, immutable manual overrides layered over a trained lexicon.
//!
//! Not a second [`crate::Lexicon`]: a miss continues into the base, and the
//! one authenticated table holds two namespaces, exact and locale-lowercase.

use std::borrow::Cow;
use std::collections::BTreeSet;
use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use unicode_normalization::is_nfc;

use crate::format::{
    MAX_USER_DICTIONARY_ENTRIES, MAX_USER_DICTIONARY_FIELD_BYTES, NORMALIZATION,
    USER_DICTIONARY_CASING, USER_DICTIONARY_SCHEMA, UserMatch,
};
use crate::lexicon::codec::Codec;
use crate::lexicon::runtime::{
    map_verified_file, open_encoded, read_bounded_regular, read_to_string,
};
use crate::lexicon::table::{Bytes, Found, Strings, Table};
use crate::lexicon::{CURRENT, Error, FileIntegrity, GENERATIONS, Result};
use crate::normalization::{Casing, scripts_compatible};
use crate::secure_fs::valid_generation_id;

#[cfg(feature = "user-dictionary-write")]
mod writer;

#[cfg(feature = "user-dictionary-write")]
pub use writer::{UserDictionaryEditor, UserDictionaryUpdate};

const ROOT_MANIFEST: &str = "user-bundle.json";
const META_FILE: &str = "meta.json";
const ENTRIES_FILE: &str = "entries.tsv";
const RECEIPT_FILE: &str = "receipt.json";
const STRINGS_FILE: &str = "strings.txt";
const STRINGS_INDEX_FILE: &str = "strings.idx";
const TABLE_NAME: &str = "user";
const MAX_CONTROL_BYTES: u64 = 4 * 1024 * 1024;

/// Root declaration of an independent user bundle.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RootManifest {
    schema: String,
}

/// Counts recorded by one user generation.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct UserEntryCounts {
    /// Case-sensitive rows.
    pub exact: u64,
    /// Locale-lowercase rows.
    pub lower: u64,
    /// Sum of both namespaces.
    pub total: u64,
    /// Distinct lemmas in `strings.txt`.
    pub strings: u64,
}

/// Metadata of one immutable user-dictionary generation.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct UserDictionaryMeta {
    /// [`USER_DICTIONARY_SCHEMA`].
    pub schema: String,
    /// ISO 639-3 language of every key.
    pub iso639_3: String,
    /// Identifier named by the language root's `current` file.
    pub generation: String,
    /// NFC and lookup-key normalization contract.
    pub normalization: String,
    /// Locale-aware casing implementation contract.
    pub casing: String,
    /// Compiled table encoding.
    pub codec: String,
    /// Canonical row and string counts.
    pub entries: UserEntryCounts,
    /// Every payload/control file except `meta.json`, authenticated by size and
    /// SHA-256.
    pub files: std::collections::BTreeMap<String, FileIntegrity>,
}

/// One auditable mutation inside a generation receipt.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct UserDictionaryChange {
    /// Canonical exact or lowercase key.
    pub key: String,
    /// Match namespace.
    #[serde(rename = "match")]
    pub match_mode: String,
    /// Value before the transaction, if any.
    pub old: Option<String>,
    /// Value after the transaction, if any.
    pub new: Option<String>,
}

/// Audit receipt stored with an immutable generation.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct UserDictionaryReceipt {
    /// [`USER_DICTIONARY_SCHEMA`].
    pub schema: String,
    /// Generation this receipt belongs to.
    pub generation: String,
    /// Language changed by the transaction.
    pub iso639_3: String,
    /// `init`, `set`, `remove`, `import`, or `rollback`.
    pub operation: String,
    /// Previously active generation, if there was one.
    pub previous_generation: Option<String>,
    /// Historical generation copied by a rollback.
    pub rollback_from: Option<String>,
    /// Caller identity supplied for audit.
    pub actor: String,
    /// Human-readable reason supplied for audit.
    pub reason: String,
    /// Milliseconds since the Unix epoch.
    pub timestamp_unix_ms: u128,
    /// Deterministic changes made by the transaction.
    pub changes: Vec<UserDictionaryChange>,
}

/// One canonical manual override.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UserEntry {
    /// Stored NFC surface; lowercase rows store their actual lowercase key.
    pub form: String,
    /// Stored NFC lemma, including an intentional identity answer.
    pub lemma: String,
    /// Exact or locale-lowercase matching.
    pub match_mode: UserMatch,
}

/// Counts returned by a complete user-bundle verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UserBundleVerification {
    /// ISO language directories verified.
    pub languages: u64,
    /// Immutable generations verified, active and historical.
    pub generations: u64,
    /// Rows in the active snapshots.
    pub active_entries: u64,
}

impl UserEntry {
    /// Construct and canonicalize an interactive row.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid ISO, empty/control-containing field,
    /// oversized field, or a changing pair with incompatible Unicode scripts.
    pub fn normalized(
        iso639_3: &str,
        form: &str,
        lemma: &str,
        match_mode: UserMatch,
    ) -> Result<Self> {
        validate_iso(iso639_3, Path::new("<user-entry>"))?;
        let form = crate::normalization::nfc(form).into_owned();
        let lemma = crate::normalization::nfc(lemma).into_owned();
        let form = match match_mode {
            UserMatch::Exact => form,
            UserMatch::Lower => Casing::for_language(iso639_3).lowercase(&form).into_owned(),
        };
        validate_fields(&form, &lemma, iso639_3, Path::new("<user-entry>"), 1)?;
        Ok(Self {
            form,
            lemma,
            match_mode,
        })
    }
}

/// A fully authenticated immutable snapshot, safe to share across threads.
pub struct UserDictionary {
    table: Table,
    strings: Strings,
    entries_bytes: Bytes,
    meta: UserDictionaryMeta,
    receipt: UserDictionaryReceipt,
    directory: PathBuf,
}

impl fmt::Debug for UserDictionary {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("UserDictionary")
            .field("iso639_3", &self.meta.iso639_3)
            .field("generation", &self.meta.generation)
            .field("entries", &self.meta.entries.total)
            .field("directory", &self.directory)
            .finish_non_exhaustive()
    }
}

impl UserDictionary {
    /// Open one language directory such as `user-bundle/rus`.
    ///
    /// # Errors
    ///
    /// Returns an error if the parent is not a valid user-bundle root or if
    /// any generation pointer, path, metadata field, digest, count, table,
    /// string index, receipt, or canonical snapshot invariant is invalid.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let parent = path.parent().ok_or_else(|| Error::Invalid {
            path: path.to_owned(),
            message: "user dictionary language directory has no bundle root".to_owned(),
        })?;
        validate_bundle_root(parent)?;
        Self::open_language(path)
    }

    /// Open one optional ISO layer from a user-bundle root.
    ///
    /// A valid root with no directory for `iso639_3` returns `None`. Every
    /// other missing or corrupt explicitly requested component is an error.
    ///
    /// # Errors
    ///
    /// Returns the root or generation validation error, or rejects an invalid
    /// ISO code.
    pub fn open_from_bundle(root: impl AsRef<Path>, iso639_3: &str) -> Result<Option<Self>> {
        let root = root.as_ref();
        validate_bundle_root(root)?;
        validate_iso(iso639_3, root)?;
        let path = root.join(iso639_3);
        match std::fs::symlink_metadata(&path) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() || !metadata.is_dir() {
                    return Err(Error::Invalid {
                        path,
                        message: "user language path must be a real directory".to_owned(),
                    });
                }
                Self::open_language(&path).map(Some)
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(source) => Err(Error::Io { path, source }),
        }
    }

    fn open_language(root: &Path) -> Result<Self> {
        real_directory(root, "user language root")?;
        let iso = root
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| Error::Invalid {
                path: root.to_owned(),
                message: "user language directory name is not UTF-8".to_owned(),
            })?;
        validate_iso(iso, root)?;
        let current = root.join(CURRENT);
        let current_meta = std::fs::symlink_metadata(&current).map_err(|source| Error::Io {
            path: current.clone(),
            source,
        })?;
        if current_meta.file_type().is_symlink()
            || !current_meta.is_file()
            || current_meta.len() > 129
        {
            return Err(Error::Invalid {
                path: current,
                message: "current must be a regular text file no larger than 129 bytes".to_owned(),
            });
        }
        let raw = read_to_string(&current)?;
        let generation = raw
            .strip_suffix("\r\n")
            .or_else(|| raw.strip_suffix('\n'))
            .unwrap_or(&raw);
        if !valid_generation_id(generation) {
            return Err(Error::Invalid {
                path: current,
                message: format!("invalid user generation identifier {raw:?}"),
            });
        }
        let generations = root.join(GENERATIONS);
        real_directory(&generations, "user generations path")?;
        let directory = generations.join(generation);
        real_directory(&directory, "user generation path")?;
        Self::open_directory(&directory, iso, generation)
    }

    fn open_directory(
        directory: &Path,
        expected_iso: &str,
        expected_generation: &str,
    ) -> Result<Self> {
        real_directory(directory, "user generation path")?;
        let meta_path = directory.join(META_FILE);
        let meta: UserDictionaryMeta = serde_json::from_str(&read_bounded_regular(
            &meta_path,
            MAX_CONTROL_BYTES,
            "user metadata",
        )?)
        .map_err(|source| Error::Meta {
            path: meta_path.clone(),
            source,
        })?;
        validate_meta(&meta, &meta_path, expected_iso, expected_generation)?;
        verify_generation_inventory(directory, &meta)?;

        let strings = Strings::open_verified(
            directory,
            expected_file(&meta, STRINGS_FILE, directory)?,
            expected_file(&meta, STRINGS_INDEX_FILE, directory)?,
        )?;
        if u64::try_from(strings.len()).ok() != Some(meta.entries.strings) {
            return Err(Error::Invalid {
                path: directory.join(STRINGS_FILE),
                message: format!(
                    "contains {} strings, metadata declares {}",
                    strings.len(),
                    meta.entries.strings
                ),
            });
        }

        let table = open_table(directory, &meta)?;

        let entries_path = directory.join(ENTRIES_FILE);
        let entries_bytes = map_verified_file(
            &entries_path,
            expected_file(&meta, ENTRIES_FILE, directory)?,
        )?;
        let mut exact = 0_u64;
        let mut lower = 0_u64;
        visit_entries(&entries_bytes, expected_iso, |entry| {
            match entry.match_mode {
                UserMatch::Exact => exact = exact.saturating_add(1),
                UserMatch::Lower => lower = lower.saturating_add(1),
            }
            let found = table
                .get_prefixed(entry.match_mode.prefix(), entry.form)
                .and_then(|value| resolve_found(&strings, value));
            if found.as_deref() != Some(entry.lemma) {
                return Err(Error::Integrity {
                    path: entries_path.clone(),
                    message: format!(
                        "compiled lookup for {} {:?} does not match snapshot lemma {:?}",
                        entry.match_mode.name(),
                        entry.form,
                        entry.lemma
                    ),
                });
            }
            Ok(())
        })?;
        if (exact, lower) != (meta.entries.exact, meta.entries.lower) {
            return Err(Error::Invalid {
                path: entries_path,
                message: format!(
                    "snapshot counts exact={exact}, lower={lower}; metadata declares exact={}, lower={}",
                    meta.entries.exact, meta.entries.lower
                ),
            });
        }

        let receipt = open_receipt(directory, &meta)?;

        Ok(Self {
            table,
            strings,
            entries_bytes,
            meta,
            receipt,
            directory: directory.to_owned(),
        })
    }

    /// ISO 639-3 language of this snapshot.
    #[must_use]
    pub fn iso639_3(&self) -> &str {
        &self.meta.iso639_3
    }

    /// Immutable generation identifier resolved at open.
    #[must_use]
    pub fn generation(&self) -> &str {
        &self.meta.generation
    }

    /// Number of exact plus lowercase rows.
    #[must_use]
    pub const fn len(&self) -> u64 {
        self.meta.entries.total
    }

    /// Whether this snapshot carries no override.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.meta.entries.total == 0
    }

    /// Authenticated generation metadata.
    #[must_use]
    pub const fn meta(&self) -> &UserDictionaryMeta {
        &self.meta
    }

    /// Authenticated audit receipt of the active generation.
    #[must_use]
    pub const fn receipt(&self) -> &UserDictionaryReceipt {
        &self.receipt
    }

    /// Materialize the canonical snapshot for listing or a writer transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if the authenticated bytes break the canonical row
    /// contract, which a healthy immutable generation cannot.
    pub fn entries(&self) -> Result<Vec<UserEntry>> {
        let capacity = usize::try_from(self.meta.entries.total).unwrap_or(0);
        let mut entries = Vec::with_capacity(capacity);
        visit_entries(&self.entries_bytes, &self.meta.iso639_3, |entry| {
            entries.push(UserEntry {
                form: entry.form.to_owned(),
                lemma: entry.lemma.to_owned(),
                match_mode: entry.match_mode,
            });
            Ok(())
        })?;
        Ok(entries)
    }

    pub(crate) fn exact(&self, key: &str) -> Option<Cow<'_, str>> {
        resolve_found(
            &self.strings,
            self.table.get_prefixed(UserMatch::Exact.prefix(), key)?,
        )
    }

    pub(crate) fn lower(&self, key: &str) -> Option<Cow<'_, str>> {
        resolve_found(
            &self.strings,
            self.table.get_prefixed(UserMatch::Lower.prefix(), key)?,
        )
    }
}

/// Authenticate every language and every immutable generation in one root.
///
/// # Errors
///
/// Returns an error for an invalid root schema, unexpected root entry,
/// symlink/special path, invalid ISO/generation name, or any corrupt current or
/// historical generation.
pub fn verify_user_bundle(root: impl AsRef<Path>) -> Result<UserBundleVerification> {
    let root = root.as_ref();
    validate_bundle_root(root)?;
    let mut summary = UserBundleVerification {
        languages: 0,
        generations: 0,
        active_entries: 0,
    };
    for item in std::fs::read_dir(root).map_err(|source| Error::Io {
        path: root.to_owned(),
        source,
    })? {
        let item = item.map_err(|source| Error::Io {
            path: root.to_owned(),
            source,
        })?;
        let name = item.file_name().into_string().map_err(|_| Error::Invalid {
            path: item.path(),
            message: "user bundle entry name is not valid UTF-8".to_owned(),
        })?;
        if name == ROOT_MANIFEST {
            continue;
        }
        validate_iso(&name, root)?;
        real_directory(&item.path(), "user language root")?;
        let active = UserDictionary::open_language(&item.path())?;
        summary.languages = summary.languages.saturating_add(1);
        summary.active_entries = summary.active_entries.saturating_add(active.len());
        let generations = item.path().join(GENERATIONS);
        for generation in std::fs::read_dir(&generations).map_err(|source| Error::Io {
            path: generations.clone(),
            source,
        })? {
            let generation = generation.map_err(|source| Error::Io {
                path: generations.clone(),
                source,
            })?;
            let generation_name =
                generation
                    .file_name()
                    .into_string()
                    .map_err(|_| Error::Invalid {
                        path: generation.path(),
                        message: "generation name is not valid UTF-8".to_owned(),
                    })?;
            if generation_name.starts_with(".building-") {
                real_directory(&generation.path(), "staged user generation path")?;
                continue;
            }
            if !valid_generation_id(&generation_name) {
                return Err(Error::Invalid {
                    path: generation.path(),
                    message: format!(
                        "invalid historical generation identifier {generation_name:?}"
                    ),
                });
            }
            let _ = UserDictionary::open_directory(&generation.path(), &name, &generation_name)?;
            summary.generations = summary.generations.saturating_add(1);
        }
    }
    Ok(summary)
}

fn resolve_found<'a>(strings: &'a Strings, found: Found<'a>) -> Option<Cow<'a, str>> {
    match found {
        Found::Id(id) => Some(Cow::Borrowed(strings.entry(id)?)),
        Found::Lemma(lemma) => Some(Cow::Owned(lemma)),
        Found::Text(text) => Some(Cow::Borrowed(text)),
    }
}

fn open_table(directory: &Path, meta: &UserDictionaryMeta) -> Result<Table> {
    let codec = Codec::parse(&meta.codec).ok_or_else(|| Error::Codec {
        path: directory.join(TABLE_NAME),
        codec: meta.codec.clone(),
    })?;
    let name = format!("{TABLE_NAME}.{}", codec.name());
    let path = directory.join(&name);
    let table = open_encoded(
        &path,
        codec,
        map_verified_file(&path, expected_file(meta, &name, directory)?)?,
    )?;
    if u64::try_from(table.len()).ok() != Some(meta.entries.total) {
        return Err(Error::Invalid {
            path,
            message: format!(
                "contains {} entries, metadata declares {}",
                table.len(),
                meta.entries.total
            ),
        });
    }
    Ok(table)
}

fn open_receipt(directory: &Path, meta: &UserDictionaryMeta) -> Result<UserDictionaryReceipt> {
    let path = directory.join(RECEIPT_FILE);
    let bytes = map_verified_file(&path, expected_file(meta, RECEIPT_FILE, directory)?)?;
    let text = std::str::from_utf8(&bytes).map_err(|_| Error::Invalid {
        path: path.clone(),
        message: "receipt is not valid UTF-8".to_owned(),
    })?;
    let receipt = serde_json::from_str(text).map_err(|source| Error::Meta {
        path: path.clone(),
        source,
    })?;
    validate_receipt(&receipt, &path, meta)?;
    Ok(receipt)
}

fn validate_bundle_root(root: &Path) -> Result<()> {
    real_directory(root, "user bundle root")?;
    let path = root.join(ROOT_MANIFEST);
    let manifest: RootManifest = serde_json::from_str(&read_bounded_regular(
        &path,
        MAX_CONTROL_BYTES,
        "user bundle manifest",
    )?)
    .map_err(|source| Error::Meta {
        path: path.clone(),
        source,
    })?;
    if manifest.schema != USER_DICTIONARY_SCHEMA {
        return Err(Error::Invalid {
            path,
            message: format!(
                "user bundle schema is {:?}, expected {USER_DICTIONARY_SCHEMA:?}",
                manifest.schema
            ),
        });
    }
    Ok(())
}

fn real_directory(path: &Path, description: &str) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path).map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: format!("{description} must be a real directory"),
        });
    }
    Ok(())
}

fn validate_iso(iso639_3: &str, path: &Path) -> Result<()> {
    if iso639_3.len() == 3 && iso639_3.bytes().all(|byte| byte.is_ascii_lowercase()) {
        return Ok(());
    }
    Err(Error::Invalid {
        path: path.to_owned(),
        message: "iso639_3 must be exactly three lowercase ASCII letters".to_owned(),
    })
}

fn validate_meta(
    meta: &UserDictionaryMeta,
    path: &Path,
    expected_iso: &str,
    expected_generation: &str,
) -> Result<()> {
    if meta.schema != USER_DICTIONARY_SCHEMA
        || meta.normalization != NORMALIZATION
        || meta.casing != USER_DICTIONARY_CASING
    {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: format!(
                "expected schema={USER_DICTIONARY_SCHEMA:?}, normalization={NORMALIZATION:?}, casing={USER_DICTIONARY_CASING:?}"
            ),
        });
    }
    validate_iso(&meta.iso639_3, path)?;
    if meta.iso639_3 != expected_iso {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: format!(
                "metadata ISO {:?} does not match language directory {expected_iso:?}",
                meta.iso639_3
            ),
        });
    }
    if meta.generation != expected_generation || !valid_generation_id(&meta.generation) {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: format!(
                "metadata generation {:?} does not match current {expected_generation:?}",
                meta.generation
            ),
        });
    }
    let total = meta
        .entries
        .exact
        .checked_add(meta.entries.lower)
        .ok_or_else(|| Error::Invalid {
            path: path.to_owned(),
            message: "user entry count overflow".to_owned(),
        })?;
    if total != meta.entries.total || total > MAX_USER_DICTIONARY_ENTRIES {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: format!(
                "user entry counts are inconsistent or exceed {MAX_USER_DICTIONARY_ENTRIES}"
            ),
        });
    }
    let codec = Codec::parse(&meta.codec).ok_or_else(|| Error::Codec {
        path: path.to_owned(),
        codec: meta.codec.clone(),
    })?;
    let expected = BTreeSet::from([
        ENTRIES_FILE.to_owned(),
        RECEIPT_FILE.to_owned(),
        STRINGS_FILE.to_owned(),
        STRINGS_INDEX_FILE.to_owned(),
        format!("{TABLE_NAME}.{}", codec.name()),
    ]);
    let declared: BTreeSet<String> = meta.files.keys().cloned().collect();
    if declared != expected {
        return Err(Error::Integrity {
            path: path.to_owned(),
            message: format!("declared files {declared:?}, expected {expected:?}"),
        });
    }
    for (name, integrity) in &meta.files {
        if integrity.sha256.len() != 64
            || !integrity
                .sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
        {
            return Err(Error::Invalid {
                path: path.to_owned(),
                message: format!("invalid SHA-256 for {name:?}"),
            });
        }
    }
    Ok(())
}

fn expected_file<'a>(
    meta: &'a UserDictionaryMeta,
    name: &str,
    directory: &Path,
) -> Result<&'a FileIntegrity> {
    meta.files.get(name).ok_or_else(|| Error::Integrity {
        path: directory.join(name),
        message: "file is absent from user metadata".to_owned(),
    })
}

fn verify_generation_inventory(directory: &Path, meta: &UserDictionaryMeta) -> Result<()> {
    let expected: BTreeSet<String> = meta
        .files
        .keys()
        .cloned()
        .chain(std::iter::once(META_FILE.to_owned()))
        .collect();
    let mut observed = BTreeSet::new();
    for item in std::fs::read_dir(directory).map_err(|source| Error::Io {
        path: directory.to_owned(),
        source,
    })? {
        let item = item.map_err(|source| Error::Io {
            path: directory.to_owned(),
            source,
        })?;
        let path = item.path();
        let name = item.file_name().into_string().map_err(|_| Error::Invalid {
            path: path.clone(),
            message: "generation file name is not valid UTF-8".to_owned(),
        })?;
        let metadata = std::fs::symlink_metadata(&path).map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(Error::Invalid {
                path,
                message: "user generation may contain regular files only".to_owned(),
            });
        }
        observed.insert(name);
    }
    if observed != expected {
        return Err(Error::Integrity {
            path: directory.to_owned(),
            message: format!("generation files {observed:?}, expected {expected:?}"),
        });
    }
    Ok(())
}

struct UserEntryRef<'a> {
    form: &'a str,
    lemma: &'a str,
    match_mode: UserMatch,
}

fn visit_entries<'a>(
    bytes: &'a [u8],
    iso639_3: &str,
    mut visit: impl FnMut(UserEntryRef<'a>) -> Result<()>,
) -> Result<()> {
    let path = Path::new(ENTRIES_FILE);
    let text = std::str::from_utf8(bytes).map_err(|_| Error::Invalid {
        path: path.to_owned(),
        message: "entries.tsv is not valid UTF-8".to_owned(),
    })?;
    if !text.is_empty() && !text.ends_with('\n') {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: "canonical entries.tsv must end with a newline".to_owned(),
        });
    }
    let mut previous: Option<(u8, &'a str)> = None;
    let mut count = 0_u64;
    for (index, line) in text.split_terminator('\n').enumerate() {
        let line_number = index.saturating_add(1);
        let mut fields = line.split('\t');
        let (Some(form), Some(lemma), Some(mode)) = (fields.next(), fields.next(), fields.next())
        else {
            return source_error(
                path,
                line_number,
                "expected exactly three tab-separated columns",
            );
        };
        if fields.next().is_some() {
            return source_error(
                path,
                line_number,
                "expected exactly three tab-separated columns",
            );
        }
        let match_mode = UserMatch::parse(mode).ok_or_else(|| Error::Source {
            path: path.to_owned(),
            line: line_number,
            message: format!("unknown match mode {mode:?}"),
        })?;
        validate_fields(form, lemma, iso639_3, path, line_number)?;
        if !is_nfc(form) || !is_nfc(lemma) {
            return source_error(
                path,
                line_number,
                "user dictionary text must be NFC-normalized",
            );
        }
        if match_mode == UserMatch::Lower
            && Casing::for_language(iso639_3).lowercase(form).as_ref() != form
        {
            return source_error(
                path,
                line_number,
                "lower rows must store their locale-lowercase key",
            );
        }
        let key = (match_mode.prefix(), form);
        if previous.is_some_and(|prior| prior >= key) {
            return source_error(
                path,
                line_number,
                "rows must be strictly sorted and unique by (match, key)",
            );
        }
        previous = Some(key);
        count = count.saturating_add(1);
        if count > MAX_USER_DICTIONARY_ENTRIES {
            return source_error(
                path,
                line_number,
                "user dictionary exceeds the one-million-entry limit",
            );
        }
        visit(UserEntryRef {
            form,
            lemma,
            match_mode,
        })?;
    }
    Ok(())
}

fn validate_fields(
    form: &str,
    lemma: &str,
    iso639_3: &str,
    path: &Path,
    line: usize,
) -> Result<()> {
    if form.is_empty() || lemma.is_empty() {
        return source_error(path, line, "form and lemma must be non-empty");
    }
    if form.len() > MAX_USER_DICTIONARY_FIELD_BYTES || lemma.len() > MAX_USER_DICTIONARY_FIELD_BYTES
    {
        return source_error(path, line, "form or lemma exceeds the field-size limit");
    }
    if form.chars().any(char::is_control) || lemma.chars().any(char::is_control) {
        return source_error(
            path,
            line,
            "form and lemma may not contain control characters",
        );
    }
    if !scripts_compatible(form, lemma, iso639_3) {
        return source_error(
            path,
            line,
            "form and lemma have incompatible Unicode Script profiles",
        );
    }
    Ok(())
}

fn source_error<T>(path: &Path, line: usize, message: &str) -> Result<T> {
    Err(Error::Source {
        path: path.to_owned(),
        line,
        message: message.to_owned(),
    })
}

fn validate_receipt(
    receipt: &UserDictionaryReceipt,
    path: &Path,
    meta: &UserDictionaryMeta,
) -> Result<()> {
    if receipt.schema != USER_DICTIONARY_SCHEMA
        || receipt.generation != meta.generation
        || receipt.iso639_3 != meta.iso639_3
        || receipt.actor.trim().is_empty()
        || receipt.reason.trim().is_empty()
        || receipt.actor.chars().any(char::is_control)
        || receipt.reason.chars().any(char::is_control)
        || !matches!(
            receipt.operation.as_str(),
            "init" | "set" | "remove" | "import" | "rollback"
        )
        || receipt
            .previous_generation
            .as_deref()
            .is_some_and(|value| !valid_generation_id(value))
        || receipt
            .rollback_from
            .as_deref()
            .is_some_and(|value| !valid_generation_id(value))
    {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: "receipt contradicts its generation or has invalid audit fields".to_owned(),
        });
    }
    let operation_consistent = match receipt.operation.as_str() {
        "init" => {
            receipt.previous_generation.is_none()
                && receipt.rollback_from.is_none()
                && receipt.changes.is_empty()
                && meta.entries.total == 0
        }
        "set" => {
            receipt.rollback_from.is_none()
                && receipt.changes.len() == 1
                && receipt
                    .changes
                    .first()
                    .is_some_and(|change| change.new.is_some())
        }
        "remove" => {
            receipt.rollback_from.is_none()
                && receipt.changes.len() == 1
                && receipt
                    .changes
                    .first()
                    .is_some_and(|change| change.old.is_some() && change.new.is_none())
        }
        "import" => {
            receipt.rollback_from.is_none()
                && !receipt.changes.is_empty()
                && receipt.changes.iter().all(|change| change.new.is_some())
        }
        "rollback" => receipt.rollback_from.is_some() && !receipt.changes.is_empty(),
        _ => false,
    };
    if !operation_consistent {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: "receipt operation has inconsistent history fields".to_owned(),
        });
    }
    let mut previous: Option<(u8, &str)> = None;
    for change in &receipt.changes {
        let Some(match_mode) = UserMatch::parse(&change.match_mode) else {
            return Err(Error::Invalid {
                path: path.to_owned(),
                message: "receipt contains an unknown match mode".to_owned(),
            });
        };
        let canonical_key = match_mode == UserMatch::Exact
            || Casing::for_language(&meta.iso639_3)
                .lowercase(&change.key)
                .as_ref()
                == change.key;
        let values_are_valid = change.old.iter().chain(change.new.iter()).all(|value| {
            is_nfc(value) && validate_fields(&change.key, value, &meta.iso639_3, path, 1).is_ok()
        });
        let order = (match_mode.prefix(), change.key.as_str());
        if change.old == change.new
            || !is_nfc(&change.key)
            || !canonical_key
            || !values_are_valid
            || previous.is_some_and(|prior| prior >= order)
        {
            return Err(Error::Invalid {
                path: path.to_owned(),
                message: "receipt contains an invalid, unsorted, duplicate, or no-op change"
                    .to_owned(),
            });
        }
        previous = Some(order);
    }
    Ok(())
}

#[cfg(feature = "user-dictionary-write")]
fn root_manifest_bytes() -> Result<Vec<u8>> {
    serde_json::to_vec_pretty(&RootManifest {
        schema: USER_DICTIONARY_SCHEMA.to_owned(),
    })
    .map(|mut bytes| {
        bytes.push(b'\n');
        bytes
    })
    .map_err(|source| Error::Meta {
        path: PathBuf::from(ROOT_MANIFEST),
        source,
    })
}

#[cfg(all(test, feature = "user-dictionary-write"))]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::sync::{Arc, Barrier};

    use super::*;

    #[cfg(feature = "train")]
    fn pair(form: &str, lemma: &str) -> crate::train::Pair {
        crate::train::Pair {
            form: form.to_owned(),
            lemma: lemma.to_owned(),
            sentence_initial: false,
        }
    }

    #[cfg(feature = "train")]
    fn publish_base(root: &Path) -> crate::Lexicon {
        let writer = crate::lexicon::GenerationWriter::begin(root).unwrap();
        let pairs = [
            pair("DASHBOARDS", "base-exact"),
            pair("cats", "cat"),
            pair("guard", "base-guard"),
        ];
        let tables = crate::train::build_for_language_with_entities(
            &pairs,
            vec![("outside".to_owned(), "external".to_owned())],
            vec![("DASHBOARDS".to_owned(), "base-entity".to_owned())],
            "eng",
        );
        let entries = crate::train::write_sources(
            &tables,
            crate::train::Variant::Full,
            crate::train::Thresholds::default(),
            writer.directory(),
        )
        .unwrap();
        let entries: serde_json::Map<String, serde_json::Value> = entries
            .into_iter()
            .chain([(
                "phrases".to_owned(),
                crate::train::write_phrase_source(&[], writer.directory()).unwrap(),
            )])
            .map(|(name, count)| (name, count.into()))
            .collect();
        let meta = serde_json::json!({
            "name": "xx_layered",
            "language": "Test",
            "iso639_3": "eng",
            "source": "test.train",
            "training_tokens": pairs.len(),
            "variant": "full",
            "selected_on": null,
            "max_suffix": crate::format::MAX_SUFFIX_LENGTH,
            "format_version": crate::FORMAT_VERSION,
            "entity_encoding": crate::ENTITY_ENCODING,
            "generation": writer.id(),
            "normalization": crate::NORMALIZATION,
            "span_case": "verbatim",
            "span_multiword": "compose",
            "files": {},
            "entries": entries,
            "traffic": {},
            "gold_tokens": 3,
            "accuracy": 0.5,
            "baseline": 0.25,
            "beats_baseline": true,
            "codecs": {}
        });
        std::fs::write(
            writer.directory().join("meta.json"),
            crate::lexicon::render_meta(&meta),
        )
        .unwrap();
        writer.compile_and_publish().unwrap();
        crate::Lexicon::open(root).unwrap()
    }

    fn editor(root: &Path) -> UserDictionaryEditor {
        UserDictionaryEditor::new(root, "rus").unwrap()
    }

    fn rewrite_receipt(
        root: &Path,
        generation: &str,
        mutate: impl FnOnce(&mut UserDictionaryReceipt),
    ) {
        let directory = root.join("rus").join(GENERATIONS).join(generation);
        let receipt_path = directory.join(RECEIPT_FILE);
        let mut receipt: UserDictionaryReceipt =
            serde_json::from_str(&std::fs::read_to_string(&receipt_path).unwrap()).unwrap();
        mutate(&mut receipt);
        let mut receipt_bytes = serde_json::to_vec_pretty(&receipt).unwrap();
        receipt_bytes.push(b'\n');
        std::fs::write(&receipt_path, receipt_bytes).unwrap();

        let observed = crate::secure_fs::file_integrity(&receipt_path).unwrap();
        let meta_path = directory.join(META_FILE);
        let mut meta: UserDictionaryMeta =
            serde_json::from_str(&std::fs::read_to_string(&meta_path).unwrap()).unwrap();
        let declared = meta.files.get_mut(RECEIPT_FILE).unwrap();
        declared.size = observed.size;
        declared.sha256 = observed.sha256;
        let mut meta_bytes = serde_json::to_vec_pretty(&meta).unwrap();
        meta_bytes.push(b'\n');
        std::fs::write(meta_path, meta_bytes).unwrap();
    }

    #[test]
    fn init_publishes_one_empty_generation_and_is_idempotent() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("user");
        let editor = editor(&root);
        let created = editor.init("analyst", "create empty overlay").unwrap();
        assert!(created.changed);
        assert_eq!(created.entries, 0);
        let dictionary = UserDictionary::open_from_bundle(&root, "rus")
            .unwrap()
            .unwrap();
        assert!(dictionary.is_empty());
        assert_eq!(dictionary.receipt().operation, "init");
        assert_eq!(
            dictionary.receipt().changes,
            [] as [UserDictionaryChange; 0]
        );

        let repeated = editor.init("analyst", "repeat init").unwrap();
        assert!(!repeated.changed);
        assert_eq!(repeated.generation, created.generation);
        assert_eq!(editor.history().unwrap().len(), 1);
    }

    #[test]
    fn set_replace_remove_and_loaded_snapshot_immutability_are_end_to_end() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("user");
        let editor = editor(&root);
        let first = editor
            .set(
                "Дашборды",
                "дашборд",
                UserMatch::Lower,
                false,
                "analyst",
                "terminology",
            )
            .unwrap();
        assert!(first.changed);
        let loaded = UserDictionary::open_from_bundle(&root, "rus")
            .unwrap()
            .unwrap();
        assert_eq!(loaded.lower("дашборды").as_deref(), Some("дашборд"));
        assert_eq!(loaded.len(), 1);

        let noop = editor
            .set(
                "ДАШБОРДЫ",
                "дашборд",
                UserMatch::Lower,
                false,
                "analyst",
                "same value",
            )
            .unwrap();
        assert!(!noop.changed);
        assert_eq!(noop.generation, first.generation);

        let conflict = editor.set(
            "дашборды",
            "панель",
            UserMatch::Lower,
            false,
            "analyst",
            "conflict",
        );
        assert!(conflict.is_err());
        let replaced = editor
            .set(
                "дашборды",
                "панель",
                UserMatch::Lower,
                true,
                "analyst",
                "approved replacement",
            )
            .unwrap();
        assert!(replaced.changed);
        // The first object keeps its resolved immutable generation.
        assert_eq!(loaded.lower("дашборды").as_deref(), Some("дашборд"));
        let fresh = UserDictionary::open_from_bundle(&root, "rus")
            .unwrap()
            .unwrap();
        assert_eq!(fresh.lower("дашборды").as_deref(), Some("панель"));

        let removed = editor
            .remove("ДАШБОРДЫ", UserMatch::Lower, "analyst", "base fixed")
            .unwrap();
        assert!(removed.changed);
        let empty = UserDictionary::open_from_bundle(&root, "rus")
            .unwrap()
            .unwrap();
        assert!(empty.is_empty());
        assert!(empty.lower("дашборды").is_none());
    }

    #[test]
    fn exact_identity_is_a_real_hit_and_lower_casing_is_locale_aware() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("user");
        let russian = editor(&root);
        russian
            .set(
                "ACME",
                "ACME",
                UserMatch::Exact,
                false,
                "analyst",
                "identity guard",
            )
            .unwrap();
        let dictionary = UserDictionary::open_from_bundle(&root, "rus")
            .unwrap()
            .unwrap();
        assert_eq!(dictionary.exact("ACME").as_deref(), Some("ACME"));
        assert!(dictionary.exact("acme").is_none());

        let turkish = UserDictionaryEditor::new(&root, "tur").unwrap();
        turkish
            .set(
                "I",
                "ı",
                UserMatch::Lower,
                false,
                "analyst",
                "Turkic casing",
            )
            .unwrap();
        let dictionary = UserDictionary::open_from_bundle(&root, "tur")
            .unwrap()
            .unwrap();
        assert_eq!(dictionary.lower("ı").as_deref(), Some("ı"));
        assert!(dictionary.lower("i").is_none());
    }

    #[test]
    fn bulk_import_rejects_negative_rows_without_moving_current() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("user");
        let editor = editor(&root);
        editor
            .set("ACME", "ACME", UserMatch::Exact, false, "analyst", "seed")
            .unwrap();
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/user_dictionary");
        let accepted = editor
            .import(
                fixtures.join("valid.tsv"),
                false,
                "analyst",
                "shared valid fixture",
            )
            .unwrap();
        let active_generation = accepted.generation.unwrap();
        let control_character = temp.path().join("control.tsv");
        std::fs::write(&control_character, b"bad\0form\tlemma\texact\n").unwrap();
        for path in [
            fixtures.join("non_nfc.tsv"),
            fixtures.join("unknown_mode.tsv"),
            fixtures.join("extra_column.tsv"),
            fixtures.join("incompatible_script.tsv"),
            fixtures.join("duplicate_after_lower.tsv"),
            fixtures.join("empty_field.tsv"),
            control_character,
        ] {
            assert!(
                editor
                    .import(&path, false, "analyst", "negative fixture")
                    .is_err(),
                "{path:?}"
            );
            let current = UserDictionary::open_from_bundle(&root, "rus")
                .unwrap()
                .unwrap();
            assert_eq!(current.generation(), active_generation, "{path:?}");
            assert_eq!(current.exact("ACME").as_deref(), Some("ACME"));
        }
    }

    #[test]
    fn rollback_copies_history_into_a_new_generation() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("user");
        let editor = editor(&root);
        let first = editor
            .set("форма", "лемма", UserMatch::Exact, false, "a", "first")
            .unwrap();
        let first_generation = first.generation.unwrap();
        editor
            .set("другая", "другой", UserMatch::Exact, false, "a", "second")
            .unwrap();
        let rolled = editor
            .rollback(&first_generation, "a", "undo second")
            .unwrap();
        assert_ne!(
            rolled.generation.as_deref(),
            Some(first_generation.as_str())
        );
        let current = UserDictionary::open_from_bundle(&root, "rus")
            .unwrap()
            .unwrap();
        assert!(current.exact("другая").is_none());
        assert_eq!(current.exact("форма").as_deref(), Some("лемма"));
        let history = editor.history().unwrap();
        assert_eq!(history.len(), 3);
        assert_eq!(history.last().unwrap().operation, "rollback");
        assert_eq!(
            history.last().unwrap().rollback_from.as_deref(),
            Some(first_generation.as_str())
        );
    }

    #[test]
    fn two_concurrent_writers_serialize_without_losing_an_update() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("user");
        // Initialize the root so this test isolates the per-language lock.
        editor(&root)
            .set("seed", "seed", UserMatch::Exact, false, "a", "seed")
            .unwrap();
        let barrier = Arc::new(Barrier::new(3));
        let mut handles = Vec::new();
        for (form, lemma) in [("форма", "лемма"), ("слова", "слово")] {
            let editor = editor(&root);
            let barrier = Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                editor
                    .set(form, lemma, UserMatch::Exact, false, "a", "concurrent")
                    .unwrap();
            }));
        }
        barrier.wait();
        for handle in handles {
            handle.join().unwrap();
        }
        let dictionary = UserDictionary::open_from_bundle(&root, "rus")
            .unwrap()
            .unwrap();
        assert_eq!(dictionary.len(), 3);
        assert_eq!(dictionary.exact("форма").as_deref(), Some("лемма"));
        assert_eq!(dictionary.exact("слова").as_deref(), Some("слово"));
    }

    #[test]
    fn every_prepublication_fault_keeps_current_and_answers_unchanged() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("user");
        let editor = editor(&root);
        let seed = editor
            .set("seed", "stable", UserMatch::Exact, false, "a", "seed")
            .unwrap();
        let seed_generation = seed.generation.unwrap();
        let points = [
            "before_staging",
            "after_staging",
            "after_entries",
            "after_strings",
            "after_strings_index",
            "after_table",
            "after_receipt",
            "after_meta",
            "after_self_open",
            "after_lookup_verify",
            "after_staging_sync",
            "after_generation_rename",
            "after_generation_dir_sync",
            "after_current_temp_sync",
        ];
        for point in points {
            super::writer::set_test_fail_point(Some(point));
            let result = editor.set(
                &format!("candidate-{point}"),
                "never-active",
                UserMatch::Exact,
                false,
                "a",
                "fault injection",
            );
            assert!(result.is_err(), "fault {point} did not fire");
            super::writer::set_test_fail_point(None);
            let current = UserDictionary::open_from_bundle(&root, "rus")
                .unwrap()
                .unwrap();
            assert_eq!(current.generation(), seed_generation, "fault {point}");
            assert_eq!(current.exact("seed").as_deref(), Some("stable"));
            assert!(current.exact(&format!("candidate-{point}")).is_none());
        }
        verify_user_bundle(&root).unwrap();
    }

    #[test]
    fn tampering_and_symlinked_payloads_fail_closed() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("user");
        let editor = editor(&root);
        let update = editor
            .set("форма", "лемма", UserMatch::Exact, false, "a", "seed")
            .unwrap();
        let generation = update.generation.unwrap();
        let generation_path = root.join("rus").join(GENERATIONS).join(generation);
        let entries = generation_path.join(ENTRIES_FILE);
        std::fs::write(&entries, "форма\tдругое\texact\n").unwrap();
        assert!(UserDictionary::open_from_bundle(&root, "rus").is_err());

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;

            let target = temp.path().join("outside");
            std::fs::write(&target, "x").unwrap();
            std::fs::remove_file(&entries).unwrap();
            symlink(&target, &entries).unwrap();
            assert!(UserDictionary::open_from_bundle(&root, "rus").is_err());
        }
    }

    #[test]
    fn noncanonical_pointers_receipts_and_staging_paths_fail_closed() {
        let temp = tempfile::tempdir().unwrap();

        let receipt_root = temp.path().join("receipt-user");
        let receipt_update = editor(&receipt_root)
            .set("форма", "лемма", UserMatch::Exact, false, "a", "seed")
            .unwrap();
        let receipt_generation = receipt_update.generation.unwrap();
        rewrite_receipt(&receipt_root, &receipt_generation, |receipt| {
            receipt.changes.first_mut().unwrap().key = "bad\nkey".to_owned();
        });
        assert!(UserDictionary::open_from_bundle(&receipt_root, "rus").is_err());

        let pointer_root = temp.path().join("pointer-user");
        let pointer_update = editor(&pointer_root)
            .set("форма", "лемма", UserMatch::Exact, false, "a", "seed")
            .unwrap();
        let pointer_generation = pointer_update.generation.unwrap();
        std::fs::write(
            pointer_root.join("rus").join(CURRENT),
            format!("{pointer_generation}\n\n"),
        )
        .unwrap();
        assert!(UserDictionary::open_from_bundle(&pointer_root, "rus").is_err());

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;

            let staging_root = temp.path().join("staging-user");
            editor(&staging_root)
                .set("форма", "лемма", UserMatch::Exact, false, "a", "seed")
                .unwrap();
            let outside = temp.path().join("outside-staging");
            std::fs::create_dir(&outside).unwrap();
            symlink(
                &outside,
                staging_root
                    .join("rus")
                    .join(GENERATIONS)
                    .join(".building-attacker"),
            )
            .unwrap();
            assert!(verify_user_bundle(&staging_root).is_err());
        }
    }

    #[test]
    fn valid_root_without_language_is_an_explicit_base_only_result() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("user");
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join(ROOT_MANIFEST), root_manifest_bytes().unwrap()).unwrap();
        assert!(
            UserDictionary::open_from_bundle(&root, "rus")
                .unwrap()
                .is_none()
        );
        assert!(UserDictionary::open_from_bundle(&root, "RU").is_err());
    }

    #[cfg(feature = "train")]
    fn layered_fixture() -> (tempfile::TempDir, crate::LayeredLexicon) {
        let temp = tempfile::tempdir().unwrap();
        let base_root = temp.path().join("xx_layered");
        let base = publish_base(&base_root);
        let user_root = temp.path().join("user");
        let editor = UserDictionaryEditor::new(&user_root, "eng").unwrap();
        editor
            .set(
                "DASHBOARDS",
                "manual-lower",
                UserMatch::Lower,
                false,
                "analyst",
                "lower must shadow base exact",
            )
            .unwrap();
        editor
            .set(
                "guard",
                "guard",
                UserMatch::Exact,
                false,
                "analyst",
                "identity guard",
            )
            .unwrap();
        let user = UserDictionary::open_from_bundle(&user_root, "eng")
            .unwrap()
            .unwrap();
        let layered = crate::LayeredLexicon::new(base, user).unwrap();
        (temp, layered)
    }

    #[cfg(feature = "train")]
    #[test]
    fn layered_probe_obeys_precedence_and_all_public_answer_shapes() {
        let (_temp, layered) = layered_fixture();

        let lower = layered.lemma_with_info("DASHBOARDS", false, crate::Options::default());
        assert_eq!(lower.lemma, "manual-lower");
        assert_eq!(lower.source, crate::Source::UserLower);
        assert!(layered.knows("DASHBOARDS"));

        let guard = layered.lemma_with_info("guard", false, crate::Options::default());
        assert_eq!(guard.lemma, "guard");
        assert_eq!(guard.source, crate::Source::User);
        assert!(guard.source.is_known());

        let miss = layered.lemma_with_info("cats", false, crate::Options::default());
        assert_eq!(
            (miss.lemma.as_ref(), miss.source),
            ("cat", crate::Source::Exact)
        );
        assert_eq!(layered.lemma("cats", false), "cat");

        let disabled = layered.lemma_with_info(
            "DASHBOARDS",
            false,
            crate::Options {
                use_user: false,
                ..crate::Options::default()
            },
        );
        assert_eq!(
            (disabled.lemma.as_ref(), disabled.source),
            ("base-exact", crate::Source::Exact)
        );

        let entity = layered.lemma_with_info(
            "DASHBOARDS",
            false,
            crate::Options {
                use_entities: true,
                ..crate::Options::default()
            },
        );
        assert_eq!(
            (entity.lemma.as_ref(), entity.source),
            ("base-entity", crate::Source::Entity)
        );
        let isolated =
            layered.lemma_with_info("guard", false, crate::Options::default().entity_lookup());
        assert_eq!(
            (isolated.lemma.as_ref(), isolated.source),
            ("guard", crate::Source::Identity)
        );

        let protected = layered.lemma_with_info(
            "DASHBOARDS",
            false,
            crate::Options {
                protected: true,
                ..crate::Options::default()
            },
        );
        assert_eq!(protected.source, crate::Source::Protected);

        let batch = layered.lemmatize_with_info(&["DASHBOARDS", "guard", "cats"]);
        assert_eq!(
            batch.iter().map(|answer| answer.source).collect::<Vec<_>>(),
            [
                crate::Source::UserLower,
                crate::Source::User,
                crate::Source::Exact,
            ]
        );
        let single = layered.lemma_span_with_info(
            "guard",
            &["guard"],
            None,
            false,
            crate::Options::default(),
        );
        assert_eq!(
            (single.lemma.as_ref(), single.source),
            ("guard", crate::Source::User)
        );
        let composed = layered.lemma_span_with_info(
            "guard cats",
            &["guard", "cats"],
            Some(&[" "]),
            false,
            crate::Options::default(),
        );
        assert_eq!(
            (composed.lemma.as_ref(), composed.source),
            ("guard cat", crate::Source::Composed)
        );

        assert_eq!(layered.is_useful(), None);
        assert_eq!(layered.base_is_useful(), Some(true));
    }
}
