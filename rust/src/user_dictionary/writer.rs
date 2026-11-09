//! Serialized CRUD transactions and atomic user-generation publication.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead as _, BufReader, Write as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use sha2::{Digest as _, Sha256};
use unicode_normalization::is_nfc;
use yada::builder::DoubleArrayBuilder;

use super::{
    ENTRIES_FILE, META_FILE, RECEIPT_FILE, ROOT_MANIFEST, STRINGS_FILE, STRINGS_INDEX_FILE,
    TABLE_NAME, UserDictionary, UserDictionaryChange, UserDictionaryMeta, UserDictionaryReceipt,
    UserEntry, UserEntryCounts, real_directory, root_manifest_bytes, source_error,
    validate_bundle_root, validate_fields, validate_iso,
};
use crate::format::{
    MAX_USER_DICTIONARY_ENTRIES, MAX_USER_DICTIONARY_FIELD_BYTES, NORMALIZATION,
    USER_DICTIONARY_CASING, USER_DICTIONARY_SCHEMA, UserMatch,
};
use crate::lexicon::codec::Codec;
use crate::lexicon::runtime::fst_error;
use crate::lexicon::table::encode_key_into;
use crate::lexicon::{CURRENT, Error, FileIntegrity, GENERATIONS, Result};
use crate::normalization::{Casing, nfc};
use crate::secure_fs::{OpenRegularError, StabilityError};

const LOCK_FILE: &str = ".write.lock";
const MAX_IMPORT_LINE_BYTES: usize = MAX_USER_DICTIONARY_FIELD_BYTES * 2 + 32;
const DA_SIZE_TOLERANCE: usize = 4;
const DA_MAX_PREMIUM_BYTES: usize = 1024 * 1024;
const DA_BLOCK_UNITS: usize = 256;

static GENERATION_NONCE: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
std::thread_local! {
    static TEST_FAIL_POINT: std::cell::Cell<Option<&'static str>> = const {
        std::cell::Cell::new(None)
    };
}

#[cfg(test)]
pub(super) fn set_test_fail_point(point: Option<&'static str>) {
    TEST_FAIL_POINT.set(point);
}

#[cfg(test)]
fn test_fault(point: &'static str, path: &Path) -> Result<()> {
    let selected = TEST_FAIL_POINT.get();
    if selected == Some(point) {
        TEST_FAIL_POINT.set(None);
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: format!("injected user-dictionary failure at {point}"),
        });
    }
    Ok(())
}

macro_rules! fault {
    ($point:literal, $path:expr) => {{
        #[cfg(test)]
        {
            test_fault($point, $path)
        }
        #[cfg(not(test))]
        {
            let _ = ($point, $path);
            Ok::<(), Error>(())
        }
    }};
}

/// Result of one idempotent user-dictionary transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserDictionaryUpdate {
    /// Whether a new immutable generation was published.
    pub changed: bool,
    /// Active generation after the operation, if one exists.
    pub generation: Option<String>,
    /// Active row count after the operation.
    pub entries: u64,
}

/// Transactional editor for one ISO language in one user-bundle root.
#[derive(Debug, Clone)]
pub struct UserDictionaryEditor {
    root: PathBuf,
    iso639_3: String,
}

impl UserDictionaryEditor {
    /// Select one root and language without mutating the filesystem.
    ///
    /// # Errors
    ///
    /// Returns an error when the ISO code is not exactly three lowercase ASCII
    /// letters.
    pub fn new(root: impl AsRef<Path>, iso639_3: &str) -> Result<Self> {
        let root = root.as_ref();
        validate_iso(iso639_3, root)?;
        Ok(Self {
            root: root.to_owned(),
            iso639_3: iso639_3.to_owned(),
        })
    }

    /// Create an authenticated empty language snapshot.
    ///
    /// Calling `init` for an already initialized language is an idempotent
    /// no-op and does not create another generation.
    ///
    /// # Errors
    ///
    /// Returns audit, integrity, locking, compilation, or publication errors.
    pub fn init(&self, actor: &str, reason: &str) -> Result<UserDictionaryUpdate> {
        validate_audit(actor, reason, &self.root)?;
        let transaction = Transaction::begin(self)?;
        if let Some(current) = transaction.current.as_ref() {
            return Ok(transaction.noop(current.len().try_into().unwrap_or(usize::MAX)));
        }
        transaction.commit(&BTreeMap::new(), "init", actor, reason, None, Vec::new())
    }

    /// Create or replace one override.
    ///
    /// An identical value is a no-op. A different existing value is rejected
    /// unless `replace` is true.
    ///
    /// # Errors
    ///
    /// Returns validation, conflict, integrity, locking, compilation, or
    /// publication errors. No error replaces `current`.
    #[allow(clippy::too_many_arguments)]
    pub fn set(
        &self,
        form: &str,
        lemma: &str,
        match_mode: UserMatch,
        replace: bool,
        actor: &str,
        reason: &str,
    ) -> Result<UserDictionaryUpdate> {
        validate_audit(actor, reason, &self.root)?;
        let entry = UserEntry::normalized(&self.iso639_3, form, lemma, match_mode)?;
        let transaction = Transaction::begin(self)?;
        let mut snapshot = transaction.snapshot()?;
        let key = (entry.match_mode, entry.form.clone());
        let old = snapshot.get(&key).cloned();
        if old.as_deref() == Some(entry.lemma.as_str()) {
            return Ok(transaction.noop(snapshot.len()));
        }
        if old.is_some() && !replace {
            return Err(Error::Invalid {
                path: self.root.join(&self.iso639_3),
                message: format!(
                    "{} key {:?} already maps to {:?}; pass replace=true to change it",
                    entry.match_mode.name(),
                    entry.form,
                    old
                ),
            });
        }
        snapshot.insert(key, entry.lemma.clone());
        transaction.commit(
            &snapshot,
            "set",
            actor,
            reason,
            None,
            vec![UserDictionaryChange {
                key: entry.form,
                match_mode: entry.match_mode.name().to_owned(),
                old,
                new: Some(entry.lemma),
            }],
        )
    }

    /// Remove one override, leaving the base lookup visible after reload.
    ///
    /// A missing key is a no-op.
    ///
    /// # Errors
    ///
    /// Returns validation, integrity, locking, or publication errors. No error
    /// replaces `current`.
    pub fn remove(
        &self,
        form: &str,
        match_mode: UserMatch,
        actor: &str,
        reason: &str,
    ) -> Result<UserDictionaryUpdate> {
        validate_audit(actor, reason, &self.root)?;
        let key = canonical_key(&self.iso639_3, form, match_mode)?;
        let transaction = Transaction::begin(self)?;
        let mut snapshot = transaction.snapshot()?;
        let old = snapshot.remove(&(match_mode, key.clone()));
        let Some(old) = old else {
            return Ok(transaction.noop(snapshot.len()));
        };
        transaction.commit(
            &snapshot,
            "remove",
            actor,
            reason,
            None,
            vec![UserDictionaryChange {
                key,
                match_mode: match_mode.name().to_owned(),
                old: Some(old),
                new: None,
            }],
        )
    }

    /// Merge a strict, headerless `form<TAB>lemma<TAB>mode` file.
    ///
    /// Imported text must already be NFC. Duplicate normalized keys are
    /// rejected. Existing conflicts require `replace`; identical values are
    /// ignored. The source is read through one stable regular-file descriptor
    /// without following a symlink, and each line is bounded before allocation.
    ///
    /// # Errors
    ///
    /// Returns source validation, conflict, integrity, locking, compilation,
    /// or publication errors. No error replaces `current`.
    pub fn import(
        &self,
        path: impl AsRef<Path>,
        replace: bool,
        actor: &str,
        reason: &str,
    ) -> Result<UserDictionaryUpdate> {
        validate_audit(actor, reason, &self.root)?;
        let imported = read_import(path.as_ref(), &self.iso639_3)?;
        let transaction = Transaction::begin(self)?;
        let mut snapshot = transaction.snapshot()?;
        let mut changes = Vec::new();
        for (key, lemma) in imported {
            let old = snapshot.get(&key).cloned();
            if old.as_deref() == Some(lemma.as_str()) {
                continue;
            }
            if old.is_some() && !replace {
                return Err(Error::Invalid {
                    path: path.as_ref().to_owned(),
                    message: format!(
                        "{} key {:?} conflicts with active value {:?}; pass replace=true",
                        key.0.name(),
                        key.1,
                        old
                    ),
                });
            }
            snapshot.insert(key.clone(), lemma.clone());
            changes.push(UserDictionaryChange {
                key: key.1,
                match_mode: key.0.name().to_owned(),
                old,
                new: Some(lemma),
            });
        }
        if changes.is_empty() {
            return Ok(transaction.noop(snapshot.len()));
        }
        transaction.commit(&snapshot, "import", actor, reason, None, changes)
    }

    /// Publish a new generation containing an authenticated historical
    /// snapshot.
    ///
    /// # Errors
    ///
    /// Returns an error if the target does not exist or is corrupt, or if the
    /// transaction cannot be locked, compiled, or atomically published.
    pub fn rollback(
        &self,
        generation: &str,
        actor: &str,
        reason: &str,
    ) -> Result<UserDictionaryUpdate> {
        validate_audit(actor, reason, &self.root)?;
        if !crate::secure_fs::valid_generation_id(generation) {
            return Err(Error::Invalid {
                path: self.root.join(&self.iso639_3).join(GENERATIONS),
                message: format!("invalid rollback generation {generation:?}"),
            });
        }
        let transaction = Transaction::begin(self)?;
        let target_path = transaction.language_root.join(GENERATIONS).join(generation);
        let target = UserDictionary::open_directory(&target_path, &self.iso639_3, generation)?;
        let target_snapshot = snapshot_map(target.entries()?)?;
        let current_snapshot = transaction.snapshot()?;
        if target_snapshot == current_snapshot {
            return Ok(transaction.noop(current_snapshot.len()));
        }
        let changes = diff_snapshots(&current_snapshot, &target_snapshot);
        transaction.commit(
            &target_snapshot,
            "rollback",
            actor,
            reason,
            Some(generation.to_owned()),
            changes,
        )
    }

    /// List every authenticated generation receipt, parents before descendants.
    ///
    /// Generation ancestry takes precedence over tied or reversed wall clocks.
    ///
    /// # Errors
    ///
    /// Returns an error if the root, language directory, any generation path,
    /// or any historical generation is corrupt.
    pub fn history(&self) -> Result<Vec<UserDictionaryReceipt>> {
        validate_bundle_root(&self.root)?;
        let language_root = self.root.join(&self.iso639_3);
        match std::fs::symlink_metadata(&language_root) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(source) => {
                return Err(Error::Io {
                    path: language_root,
                    source,
                });
            }
            Ok(_) => real_directory(&language_root, "user language root")?,
        }
        // Authenticate current as well; history must not hide a broken active
        // pointer behind otherwise healthy old generations.
        let _active = UserDictionary::open(&language_root)?;
        let generations = language_root.join(GENERATIONS);
        real_directory(&generations, "user generations path")?;
        let mut receipts = Vec::new();
        for item in std::fs::read_dir(&generations).map_err(|source| Error::Io {
            path: generations.clone(),
            source,
        })? {
            let item = item.map_err(|source| Error::Io {
                path: generations.clone(),
                source,
            })?;
            let name = item.file_name().into_string().map_err(|_| Error::Invalid {
                path: item.path(),
                message: "generation directory name is not valid UTF-8".to_owned(),
            })?;
            if name.starts_with(".building-") {
                real_directory(&item.path(), "staged user generation path")?;
                continue;
            }
            if !crate::secure_fs::valid_generation_id(&name) {
                return Err(Error::Invalid {
                    path: item.path(),
                    message: format!("invalid historical generation identifier {name:?}"),
                });
            }
            let dictionary = UserDictionary::open_directory(&item.path(), &self.iso639_3, &name)?;
            receipts.push(dictionary.receipt().clone());
        }
        receipts.sort_by(|left, right| {
            (left.timestamp_unix_ms, &left.generation)
                .cmp(&(right.timestamp_unix_ms, &right.generation))
        });
        let order: Vec<_> = receipts
            .iter()
            .map(|receipt| receipt.generation.clone())
            .collect();
        let mut remaining: BTreeMap<_, _> = receipts
            .into_iter()
            .map(|receipt| (receipt.generation.clone(), receipt))
            .collect();
        let mut history = Vec::with_capacity(order.len());
        for generation in order {
            let mut chain = Vec::new();
            let mut visited = BTreeSet::new();
            let mut next = Some(generation);
            while let Some(generation) = next.take() {
                if !visited.insert(generation.clone()) {
                    return Err(Error::Invalid {
                        path: generations.join(generation).join(RECEIPT_FILE),
                        message: "cyclic user dictionary history".to_owned(),
                    });
                }
                let Some(receipt) = remaining.remove(&generation) else {
                    break;
                };
                next.clone_from(&receipt.previous_generation);
                chain.push(receipt);
            }
            history.extend(chain.into_iter().rev());
        }
        Ok(history)
    }
}

struct Transaction {
    language_root: PathBuf,
    current: Option<UserDictionary>,
    _lock: File,
}

impl Transaction {
    fn begin(editor: &UserDictionaryEditor) -> Result<Self> {
        ensure_bundle_root(&editor.root)?;
        let language_root = editor.root.join(&editor.iso639_3);
        create_real_directory(&language_root, "user language root")?;
        let lock = open_writer_lock(&language_root.join(LOCK_FILE))?;
        let current_path = language_root.join(CURRENT);
        let current = match std::fs::symlink_metadata(&current_path) {
            Ok(_) => Some(UserDictionary::open(&language_root)?),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(source) => {
                return Err(Error::Io {
                    path: current_path,
                    source,
                });
            }
        };
        Ok(Self {
            language_root,
            current,
            _lock: lock,
        })
    }

    fn snapshot(&self) -> Result<BTreeMap<(UserMatch, String), String>> {
        self.current.as_ref().map_or_else(
            || Ok(BTreeMap::new()),
            |dictionary| snapshot_map(dictionary.entries()?),
        )
    }

    fn noop(&self, entries: usize) -> UserDictionaryUpdate {
        UserDictionaryUpdate {
            changed: false,
            generation: self
                .current
                .as_ref()
                .map(|dictionary| dictionary.generation().to_owned()),
            entries: u64::try_from(entries).unwrap_or(u64::MAX),
        }
    }

    fn commit(
        self,
        snapshot: &BTreeMap<(UserMatch, String), String>,
        operation: &str,
        actor: &str,
        reason: &str,
        rollback_from: Option<String>,
        mut changes: Vec<UserDictionaryChange>,
    ) -> Result<UserDictionaryUpdate> {
        let total = u64::try_from(snapshot.len()).map_err(|_| Error::Invalid {
            path: self.language_root.clone(),
            message: "user entry count does not fit u64".to_owned(),
        })?;
        if total > MAX_USER_DICTIONARY_ENTRIES {
            return Err(Error::Invalid {
                path: self.language_root,
                message: format!(
                    "user dictionary has {total} entries; maximum is {MAX_USER_DICTIONARY_ENTRIES}"
                ),
            });
        }
        changes.sort_by(|left, right| {
            (&left.match_mode, &left.key).cmp(&(&right.match_mode, &right.key))
        });
        let timestamp =
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|_| Error::Invalid {
                    path: self.language_root.clone(),
                    message: "system clock is before the Unix epoch".to_owned(),
                })?;
        let generation = generation_id(snapshot, operation, actor, reason, timestamp.as_nanos());
        let timestamp_unix_ms = timestamp.as_millis();
        let previous_generation = self
            .current
            .as_ref()
            .map(|dictionary| dictionary.generation().to_owned());
        let receipt = UserDictionaryReceipt {
            schema: USER_DICTIONARY_SCHEMA.to_owned(),
            generation: generation.clone(),
            iso639_3: self.current.as_ref().map_or_else(
                || {
                    self.language_root
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or_default()
                        .to_owned()
                },
                |dictionary| dictionary.iso639_3().to_owned(),
            ),
            operation: operation.to_owned(),
            previous_generation,
            rollback_from,
            actor: actor.to_owned(),
            reason: reason.to_owned(),
            timestamp_unix_ms,
            changes,
        };
        let generations = self.language_root.join(GENERATIONS);
        create_real_directory(&generations, "user generations path")?;
        fault!("before_staging", &generations)?;
        let staging = generations.join(format!(".building-{generation}"));
        std::fs::create_dir(&staging).map_err(|source| Error::Io {
            path: staging.clone(),
            source,
        })?;
        let mut guard = StagingGuard {
            path: staging.clone(),
            armed: true,
        };
        fault!("after_staging", &staging)?;
        write_generation(&staging, snapshot, &receipt)?;
        let opened = UserDictionary::open_directory(&staging, &receipt.iso639_3, &generation)?;
        fault!("after_self_open", &staging)?;
        for change in &receipt.changes {
            let mode = UserMatch::parse(&change.match_mode).ok_or_else(|| Error::Invalid {
                path: staging.clone(),
                message: "writer produced an unknown receipt match mode".to_owned(),
            })?;
            let answer = match mode {
                UserMatch::Exact => opened.exact(&change.key),
                UserMatch::Lower => opened.lower(&change.key),
            };
            if answer.as_deref() != change.new.as_deref() {
                return Err(Error::Integrity {
                    path: staging,
                    message: format!("self-open lookup mismatch for {:?}", change.key),
                });
            }
        }
        fault!("after_lookup_verify", &staging)?;
        sync_directory(&staging)?;
        fault!("after_staging_sync", &staging)?;
        let final_path = generations.join(&generation);
        std::fs::rename(&staging, &final_path).map_err(|source| Error::Io {
            path: final_path.clone(),
            source,
        })?;
        guard.armed = false;
        fault!("after_generation_rename", &final_path)?;
        sync_directory_entry(&generations)?;
        fault!("after_generation_dir_sync", &generations)?;
        publish_current(&self.language_root, &generation)?;
        Ok(UserDictionaryUpdate {
            changed: true,
            generation: Some(generation),
            entries: total,
        })
    }
}

struct StagingGuard {
    path: PathBuf,
    armed: bool,
}

impl Drop for StagingGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

fn snapshot_map(entries: Vec<UserEntry>) -> Result<BTreeMap<(UserMatch, String), String>> {
    let mut snapshot = BTreeMap::new();
    for entry in entries {
        if snapshot
            .insert((entry.match_mode, entry.form.clone()), entry.lemma)
            .is_some()
        {
            return Err(Error::Invalid {
                path: PathBuf::from(ENTRIES_FILE),
                message: format!("duplicate canonical entry {:?}", entry.form),
            });
        }
    }
    Ok(snapshot)
}

fn canonical_key(iso639_3: &str, form: &str, match_mode: UserMatch) -> Result<String> {
    let normalized = nfc(form).into_owned();
    validate_fields(
        &normalized,
        &normalized,
        iso639_3,
        Path::new("<user-entry>"),
        1,
    )?;
    Ok(match match_mode {
        UserMatch::Exact => normalized,
        UserMatch::Lower => Casing::for_language(iso639_3)
            .lowercase(&normalized)
            .into_owned(),
    })
}

fn read_import(path: &Path, iso639_3: &str) -> Result<BTreeMap<(UserMatch, String), String>> {
    let (file, before) =
        crate::secure_fs::open_regular(path).map_err(|error| map_open_error(path, error))?;
    let mut reader = BufReader::new(file);
    let mut rows = BTreeMap::new();
    let mut physical_line = 0_usize;
    let mut observed = 0_u64;
    loop {
        let (line, consumed, eof) = read_capped_line(&mut reader, path)?;
        observed = observed
            .checked_add(u64::try_from(consumed).unwrap_or(u64::MAX))
            .ok_or_else(|| Error::Invalid {
                path: path.to_owned(),
                message: "import size overflow".to_owned(),
            })?;
        if eof && consumed == 0 {
            break;
        }
        physical_line = physical_line.saturating_add(1);
        if eof && !line.is_empty() {
            return source_error(path, physical_line, "import must end with a newline");
        }
        let text = std::str::from_utf8(&line).map_err(|_| Error::Source {
            path: path.to_owned(),
            line: physical_line,
            message: "row is not valid UTF-8".to_owned(),
        })?;
        let mut fields = text.split('\t');
        let (Some(form), Some(lemma), Some(mode)) = (fields.next(), fields.next(), fields.next())
        else {
            return source_error(
                path,
                physical_line,
                "expected exactly three tab-separated columns",
            );
        };
        if fields.next().is_some() {
            return source_error(
                path,
                physical_line,
                "expected exactly three tab-separated columns",
            );
        }
        let match_mode = UserMatch::parse(mode).ok_or_else(|| Error::Source {
            path: path.to_owned(),
            line: physical_line,
            message: format!("unknown match mode {mode:?}"),
        })?;
        validate_fields(form, lemma, iso639_3, path, physical_line)?;
        if !is_nfc(form) || !is_nfc(lemma) {
            return source_error(
                path,
                physical_line,
                "bulk import must already be NFC-normalized",
            );
        }
        let canonical = match match_mode {
            UserMatch::Exact => form.to_owned(),
            UserMatch::Lower => Casing::for_language(iso639_3).lowercase(form).into_owned(),
        };
        let key = (match_mode, canonical);
        if rows.insert(key.clone(), lemma.to_owned()).is_some() {
            return Err(Error::Source {
                path: path.to_owned(),
                line: physical_line,
                message: format!("duplicate normalized {} key {:?}", match_mode.name(), key.1),
            });
        }
        if u64::try_from(rows.len()).unwrap_or(u64::MAX) > MAX_USER_DICTIONARY_ENTRIES {
            return source_error(
                path,
                physical_line,
                "user dictionary exceeds the one-million-entry limit",
            );
        }
        if eof {
            break;
        }
    }
    crate::secure_fs::verify_stable(path, &before, reader.get_ref(), observed).map_err(
        |error| match error {
            StabilityError::Io(source) => Error::Io {
                path: path.to_owned(),
                source,
            },
            StabilityError::Changed => Error::Invalid {
                path: path.to_owned(),
                message: "import path or contents changed while it was read".to_owned(),
            },
        },
    )?;
    Ok(rows)
}

fn read_capped_line(reader: &mut BufReader<File>, path: &Path) -> Result<(Vec<u8>, usize, bool)> {
    let mut line = Vec::new();
    let mut consumed = 0_usize;
    loop {
        let buffer = reader.fill_buf().map_err(|source| Error::Io {
            path: path.to_owned(),
            source,
        })?;
        if buffer.is_empty() {
            return Ok((line, consumed, true));
        }
        let newline = buffer.iter().position(|byte| *byte == b'\n');
        let take = newline.map_or(buffer.len(), |position| position.saturating_add(1));
        let body = if newline.is_some() {
            buffer.get(..take.saturating_sub(1)).unwrap_or_default()
        } else {
            buffer.get(..take).unwrap_or_default()
        };
        if line.len().saturating_add(body.len()) > MAX_IMPORT_LINE_BYTES {
            return Err(Error::Invalid {
                path: path.to_owned(),
                message: format!("import row exceeds {MAX_IMPORT_LINE_BYTES} bytes"),
            });
        }
        line.extend_from_slice(body);
        reader.consume(take);
        consumed = consumed.saturating_add(take);
        if newline.is_some() {
            return Ok((line, consumed, false));
        }
    }
}

fn write_generation(
    directory: &Path,
    snapshot: &BTreeMap<(UserMatch, String), String>,
    receipt: &UserDictionaryReceipt,
) -> Result<()> {
    let mut entries = Vec::new();
    let mut lemmas = BTreeSet::new();
    let mut exact = 0_u64;
    let mut lower = 0_u64;
    for ((match_mode, form), lemma) in snapshot {
        entries.extend_from_slice(form.as_bytes());
        entries.push(b'\t');
        entries.extend_from_slice(lemma.as_bytes());
        entries.push(b'\t');
        entries.extend_from_slice(match_mode.name().as_bytes());
        entries.push(b'\n');
        lemmas.insert(lemma.as_str());
        match match_mode {
            UserMatch::Exact => exact = exact.saturating_add(1),
            UserMatch::Lower => lower = lower.saturating_add(1),
        }
    }
    write_file(&directory.join(ENTRIES_FILE), &entries)?;
    drop(entries);
    fault!("after_entries", directory)?;

    let mut strings = Vec::new();
    let mut offsets = Vec::with_capacity(lemmas.len().saturating_add(1));
    offsets.push(0_u32);
    let mut ids = BTreeMap::new();
    for (index, &lemma) in lemmas.iter().enumerate() {
        let id = u64::try_from(index).map_err(|_| Error::Invalid {
            path: directory.join(STRINGS_FILE),
            message: "string identifier overflow".to_owned(),
        })?;
        ids.insert(lemma, id);
        strings.extend_from_slice(lemma.as_bytes());
        strings.push(b'\n');
        offsets.push(
            u32::try_from(strings.len()).map_err(|_| Error::StringsTooLarge {
                path: directory.join(STRINGS_FILE),
            })?,
        );
    }
    let string_count = u64::try_from(lemmas.len()).unwrap_or(u64::MAX);
    drop(lemmas);
    write_file(&directory.join(STRINGS_FILE), &strings)?;
    drop(strings);
    fault!("after_strings", directory)?;
    let mut index_bytes = Vec::with_capacity(offsets.len().saturating_mul(size_of::<u32>()));
    for offset in offsets {
        index_bytes.extend_from_slice(&offset.to_le_bytes());
    }
    write_file(&directory.join(STRINGS_INDEX_FILE), &index_bytes)?;
    drop(index_bytes);
    fault!("after_strings_index", directory)?;

    let (codec, table_bytes) = compile_table(directory, snapshot, &ids)?;
    drop(ids);
    let table_file = format!("{TABLE_NAME}.{}", codec.name());
    write_file(&directory.join(&table_file), &table_bytes)?;
    drop(table_bytes);
    fault!("after_table", directory)?;
    write_json(&directory.join(RECEIPT_FILE), receipt)?;
    fault!("after_receipt", directory)?;

    let file_names = [
        ENTRIES_FILE.to_owned(),
        RECEIPT_FILE.to_owned(),
        STRINGS_FILE.to_owned(),
        STRINGS_INDEX_FILE.to_owned(),
        table_file,
    ];
    let mut files = BTreeMap::new();
    for name in file_names {
        files.insert(name.clone(), integrity(&directory.join(&name))?);
    }
    let total = u64::try_from(snapshot.len()).unwrap_or(u64::MAX);
    let meta = UserDictionaryMeta {
        schema: USER_DICTIONARY_SCHEMA.to_owned(),
        iso639_3: receipt.iso639_3.clone(),
        generation: receipt.generation.clone(),
        normalization: NORMALIZATION.to_owned(),
        casing: USER_DICTIONARY_CASING.to_owned(),
        codec: codec.name().to_owned(),
        entries: UserEntryCounts {
            exact,
            lower,
            total,
            strings: string_count,
        },
        files,
    };
    write_json(&directory.join(META_FILE), &meta)?;
    fault!("after_meta", directory)
}

// Yada stores a distinct terminal node and value leaf per nonempty key, plus the root.
const fn minimum_da_bytes(entries: usize) -> usize {
    entries
        .saturating_mul(2)
        .saturating_add(1)
        .div_ceil(DA_BLOCK_UNITS)
        .saturating_mul(DA_BLOCK_UNITS)
        .saturating_mul(size_of::<u32>())
}

const fn da_fits_budget(bytes: usize, automaton: usize) -> bool {
    bytes <= automaton.saturating_mul(DA_SIZE_TOLERANCE)
        && bytes.saturating_sub(automaton) <= DA_MAX_PREMIUM_BYTES
}

fn compile_table(
    directory: &Path,
    snapshot: &BTreeMap<(UserMatch, String), String>,
    ids: &BTreeMap<&str, u64>,
) -> Result<(Codec, Vec<u8>)> {
    let map_path = directory.join(format!("{TABLE_NAME}.map"));
    let inline_path = directory.join(format!("{TABLE_NAME}.inline"));
    let da_path = directory.join(format!("{TABLE_NAME}.da"));
    let mut map = fst::MapBuilder::memory();
    let mut inline = fst::raw::Builder::memory();
    let mut inline_key = Vec::new();
    let mut keys = Vec::with_capacity(snapshot.len());
    for ((match_mode, form), lemma) in snapshot {
        let mut key = String::with_capacity(form.len().saturating_add(1));
        key.push(char::from(match_mode.prefix()));
        key.push_str(form);
        let id = ids
            .get(lemma.as_str())
            .copied()
            .ok_or_else(|| Error::Invalid {
                path: directory.join(STRINGS_FILE),
                message: format!("lemma {lemma:?} has no string identifier"),
            })?;
        map.insert(key.as_bytes(), id)
            .map_err(|source| fst_error(&map_path, source))?;
        encode_key_into(key.as_bytes(), lemma.as_bytes(), &mut inline_key);
        inline
            .insert(&inline_key, 0)
            .map_err(|source| fst_error(&inline_path, source))?;
        let value = u32::try_from(id).map_err(|_| Error::Invalid {
            path: da_path.clone(),
            message: "string identifier exceeds the double-array u32 limit".to_owned(),
        })?;
        keys.push((key, value));
    }
    let map_bytes = map
        .into_inner()
        .map_err(|source| fst_error(&map_path, source))?;
    let inline_bytes = inline
        .into_inner()
        .map_err(|source| fst_error(&inline_path, source))?;
    let (automaton_codec, automaton) = if inline_bytes.len() < map_bytes.len() {
        (Codec::Inline, inline_bytes)
    } else {
        (Codec::Map, map_bytes)
    };
    if keys.is_empty() || !da_fits_budget(minimum_da_bytes(keys.len()), automaton.len()) {
        return Ok((automaton_codec, automaton));
    }
    let key_refs: Vec<(&[u8], u32)> = keys
        .iter()
        .map(|(key, value)| (key.as_bytes(), *value))
        .collect();
    let da = DoubleArrayBuilder::build(&key_refs).map_err(|source| Error::Trie {
        path: da_path,
        source,
    })?;
    if da_fits_budget(da.len(), automaton.len()) {
        Ok((Codec::Da, da))
    } else {
        Ok((automaton_codec, automaton))
    }
}

fn diff_snapshots(
    old: &BTreeMap<(UserMatch, String), String>,
    new: &BTreeMap<(UserMatch, String), String>,
) -> Vec<UserDictionaryChange> {
    let keys: BTreeSet<_> = old.keys().chain(new.keys()).cloned().collect();
    keys.into_iter()
        .filter_map(|key| {
            let before = old.get(&key).cloned();
            let after = new.get(&key).cloned();
            (before != after).then(|| UserDictionaryChange {
                key: key.1,
                match_mode: key.0.name().to_owned(),
                old: before,
                new: after,
            })
        })
        .collect()
}

fn generation_id(
    snapshot: &BTreeMap<(UserMatch, String), String>,
    operation: &str,
    actor: &str,
    reason: &str,
    timestamp_nanos: u128,
) -> String {
    let nonce = GENERATION_NONCE.fetch_add(1, Ordering::Relaxed);
    let mut digest = Sha256::new();
    digest.update(operation.as_bytes());
    digest.update([0]);
    digest.update(actor.as_bytes());
    digest.update([0]);
    digest.update(reason.as_bytes());
    digest.update([0]);
    digest.update(timestamp_nanos.to_le_bytes());
    digest.update(std::process::id().to_le_bytes());
    digest.update(nonce.to_le_bytes());
    for ((mode, form), lemma) in snapshot {
        digest.update([mode.prefix()]);
        digest.update(form.as_bytes());
        digest.update([0]);
        digest.update(lemma.as_bytes());
        digest.update([0]);
    }
    let hex = crate::secure_fs::hex_lower(&digest.finalize());
    format!("u{}", hex.get(..32).unwrap_or(&hex))
}

fn ensure_bundle_root(root: &Path) -> Result<()> {
    create_real_directory(root, "user bundle root")?;
    let manifest_path = root.join(ROOT_MANIFEST);
    match std::fs::symlink_metadata(&manifest_path) {
        Ok(_) => validate_bundle_root(root),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let bytes = root_manifest_bytes()?;
            let mut temporary =
                tempfile::NamedTempFile::new_in(root).map_err(|source| Error::Io {
                    path: root.to_owned(),
                    source,
                })?;
            temporary.write_all(&bytes).map_err(|source| Error::Io {
                path: manifest_path.clone(),
                source,
            })?;
            temporary
                .as_file_mut()
                .sync_all()
                .map_err(|source| Error::Io {
                    path: manifest_path.clone(),
                    source,
                })?;
            match temporary.persist_noclobber(&manifest_path) {
                Ok(_) => sync_directory_entry(root),
                Err(error) if error.error.kind() == std::io::ErrorKind::AlreadyExists => {
                    validate_bundle_root(root)
                }
                Err(error) => Err(Error::Io {
                    path: manifest_path,
                    source: error.error,
                }),
            }
        }
        Err(source) => Err(Error::Io {
            path: manifest_path,
            source,
        }),
    }
}

fn create_real_directory(path: &Path, description: &str) -> Result<()> {
    match std::fs::create_dir(path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(source) => {
            return Err(Error::Io {
                path: path.to_owned(),
                source,
            });
        }
    }
    real_directory(path, description)
}

fn open_writer_lock(path: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.create(true).read(true).write(true);
    crate::secure_fs::no_follow(&mut options);
    let mut file = options.open(path).map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })?;
    let path_meta = std::fs::symlink_metadata(path).map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })?;
    let file_meta = file.metadata().map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })?;
    if path_meta.file_type().is_symlink()
        || !path_meta.is_file()
        || !file_meta.is_file()
        || !crate::secure_fs::same_file(&path_meta, &file_meta)
    {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: "writer lock must be one stable regular file".to_owned(),
        });
    }
    file.lock().map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })?;
    file.set_len(0).map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })?;
    writeln!(file, "{}", std::process::id()).map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })?;
    file.sync_all().map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })?;
    Ok(file)
}

fn publish_current(root: &Path, generation: &str) -> Result<()> {
    let current = root.join(CURRENT);
    if let Ok(metadata) = std::fs::symlink_metadata(&current)
        && (metadata.file_type().is_symlink() || !metadata.is_file())
    {
        return Err(Error::Invalid {
            path: current,
            message: "refusing to replace a symlink or special current pointer".to_owned(),
        });
    }
    let mut temporary = tempfile::NamedTempFile::new_in(root).map_err(|source| Error::Io {
        path: root.to_owned(),
        source,
    })?;
    writeln!(temporary, "{generation}").map_err(|source| Error::Io {
        path: current.clone(),
        source,
    })?;
    temporary
        .as_file_mut()
        .sync_all()
        .map_err(|source| Error::Io {
            path: current.clone(),
            source,
        })?;
    fault!("after_current_temp_sync", root)?;
    temporary.persist(&current).map_err(|error| Error::Io {
        path: current,
        source: error.error,
    })?;
    sync_directory_entry(root)
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut options = OpenOptions::new();
    options.create_new(true).write(true);
    crate::secure_fs::no_follow(&mut options);
    let mut file = options.open(path).map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })?;
    file.write_all(bytes).map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })?;
    file.sync_all().map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value).map_err(|source| Error::Meta {
        path: path.to_owned(),
        source,
    })?;
    bytes.push(b'\n');
    write_file(path, &bytes)
}

fn integrity(path: &Path) -> Result<FileIntegrity> {
    crate::secure_fs::file_integrity(path)
        .map(|value| FileIntegrity {
            size: value.size,
            sha256: value.sha256,
        })
        .map_err(|error| match error {
            crate::secure_fs::MapVerifiedError::Open(open) => map_open_error(path, open),
            crate::secure_fs::MapVerifiedError::Read(source)
            | crate::secure_fs::MapVerifiedError::Map(source) => Error::Io {
                path: path.to_owned(),
                source,
            },
            crate::secure_fs::MapVerifiedError::LengthOverflow => Error::Invalid {
                path: path.to_owned(),
                message: "file size overflow while hashing".to_owned(),
            },
            crate::secure_fs::MapVerifiedError::Changed => Error::Invalid {
                path: path.to_owned(),
                message: "file changed while hashing".to_owned(),
            },
            crate::secure_fs::MapVerifiedError::Mismatch(_) => Error::Invalid {
                path: path.to_owned(),
                message: "unexpected integrity mismatch while writing".to_owned(),
            },
        })
}

fn map_open_error(path: &Path, error: OpenRegularError) -> Error {
    match error {
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
    }
}

fn sync_directory(directory: &Path) -> Result<()> {
    for item in std::fs::read_dir(directory).map_err(|source| Error::Io {
        path: directory.to_owned(),
        source,
    })? {
        let path = item
            .map_err(|source| Error::Io {
                path: directory.to_owned(),
                source,
            })?
            .path();
        let metadata = std::fs::symlink_metadata(&path).map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(Error::Invalid {
                path,
                message: "staged generation may contain regular files only".to_owned(),
            });
        }
        File::open(&path)
            .and_then(|file| file.sync_all())
            .map_err(|source| Error::Io {
                path: path.clone(),
                source,
            })?;
    }
    sync_directory_entry(directory)
}

fn sync_directory_entry(directory: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        File::open(directory)
            .and_then(|file| file.sync_all())
            .map_err(|source| Error::Io {
                path: directory.to_owned(),
                source,
            })
    }
    #[cfg(not(unix))]
    {
        let _ = directory;
        Ok(())
    }
}

fn validate_audit(actor: &str, reason: &str, path: &Path) -> Result<()> {
    if actor.trim().is_empty() || reason.trim().is_empty() {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: "actor and reason must be non-empty".to_owned(),
        });
    }
    if actor.chars().any(char::is_control) || reason.chars().any(char::is_control) {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: "actor and reason may not contain control characters".to_owned(),
        });
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn da_lower_bound_respects_shared_prefixes_and_equal_values() {
        let mut keysets = vec![
            vec!["a".to_owned(), "aa".to_owned(), "aaa".to_owned()],
            vec!["а".to_owned(), "аб".to_owned(), "猫".to_owned()],
        ];
        for count in [1, 127, 128, 255, 256, 1024] {
            keysets.push((0..count).map(|index| format!("{index:04x}")).collect());
        }
        for keys in keysets {
            let rows: Vec<_> = keys.iter().map(|key| (key.as_bytes(), 0)).collect();
            let encoded = DoubleArrayBuilder::build(&rows).unwrap();
            assert!(minimum_da_bytes(rows.len()) <= encoded.len());
        }
        assert_eq!(minimum_da_bytes(127), 1024);
        assert_eq!(minimum_da_bytes(128), 2048);
        assert_eq!(minimum_da_bytes(1_000_000), 8_000_512);
        assert_eq!(minimum_da_bytes(usize::MAX), usize::MAX);
    }

    #[test]
    fn da_budget_preserves_ratio_and_premium_ties() {
        assert!(da_fits_budget(400, 100));
        assert!(!da_fits_budget(401, 100));
        assert!(da_fits_budget(
            2 * DA_MAX_PREMIUM_BYTES,
            DA_MAX_PREMIUM_BYTES
        ));
        assert!(!da_fits_budget(
            2 * DA_MAX_PREMIUM_BYTES + 1,
            DA_MAX_PREMIUM_BYTES
        ));
    }

    #[test]
    fn pruning_preserves_codec_and_encoded_bytes() {
        let mut selected_da = false;
        let mut pruned_da = false;
        for count in [0, 1, 64, 1024] {
            let lemmas: Vec<_> = (0..count).map(|index| format!("lemma{index:04}")).collect();
            let ids: BTreeMap<_, _> = lemmas
                .iter()
                .enumerate()
                .map(|(index, lemma)| (lemma.as_str(), u64::try_from(index).unwrap()))
                .collect();
            let snapshot: BTreeMap<_, _> = (0..count)
                .map(|index| {
                    (
                        (UserMatch::Exact, format!("form{index:04}")),
                        format!("lemma{:04}", index * 37 % count),
                    )
                })
                .collect();
            let mut map = fst::MapBuilder::memory();
            let mut inline = fst::raw::Builder::memory();
            let mut inline_key = Vec::new();
            let mut rows = Vec::new();
            for ((mode, form), lemma) in &snapshot {
                let key = format!("{}{form}", char::from(mode.prefix()));
                let id = *ids.get(lemma.as_str()).unwrap();
                map.insert(key.as_bytes(), id).unwrap();
                encode_key_into(key.as_bytes(), lemma.as_bytes(), &mut inline_key);
                inline.insert(&inline_key, 0).unwrap();
                rows.push((key, u32::try_from(id).unwrap()));
            }
            let map = map.into_inner().unwrap();
            let inline = inline.into_inner().unwrap();
            let mut expected = if inline.len() < map.len() {
                (Codec::Inline, inline)
            } else {
                (Codec::Map, map)
            };
            if !rows.is_empty() {
                pruned_da |= !da_fits_budget(minimum_da_bytes(rows.len()), expected.1.len());
                let da = DoubleArrayBuilder::build(&rows).unwrap();
                if da.len() <= expected.1.len() * DA_SIZE_TOLERANCE
                    && da.len().saturating_sub(expected.1.len()) <= DA_MAX_PREMIUM_BYTES
                {
                    expected = (Codec::Da, da);
                    selected_da = true;
                }
            }
            let actual = compile_table(Path::new("test-generation"), &snapshot, &ids).unwrap();
            assert_eq!(actual, expected, "{count} entries");
        }
        assert!(selected_da && pruned_da);
    }
}
