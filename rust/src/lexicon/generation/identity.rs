//! Stable identities for staging and content-addressed generations.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use sha2::{Digest, Sha256};

use super::super::{Error, FileIntegrity, Result, file_integrity};

/// Counter disambiguating multiple staging generations created in one tick.
static GENERATION_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Creates a filesystem-safe, process-unique staging identifier.
pub(super) fn new_generation_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    let counter = GENERATION_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("g{nanos:x}-{:x}-{counter:x}", std::process::id())
}

/// Serialises metadata in the on-disk style: one-space indent, raw UTF-8,
/// trailing newline.
pub fn render_meta(meta: &serde_json::Value) -> Vec<u8> {
    let formatter = serde_json::ser::PrettyFormatter::with_indent(b" ");
    let mut buffer = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut buffer, formatter);
    // Serialising a parsed value into a `Vec` cannot fail.
    #[allow(clippy::expect_used)]
    meta.serialize(&mut serializer)
        .expect("serialising JSON into memory cannot fail");
    buffer.push(b'\n');
    buffer
}

/// Recursively orders object keys so equivalent metadata hashes identically.
fn canonical_json(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(object) => {
            let ordered: BTreeMap<&str, &serde_json::Value> = object
                .iter()
                .map(|(key, value)| (key.as_str(), value))
                .collect();
            ordered
                .into_iter()
                .map(|(key, value)| (key.to_owned(), canonical_json(value)))
                .collect::<serde_json::Map<_, _>>()
                .into()
        }
        serde_json::Value::Array(items) => items.iter().map(canonical_json).collect(),
        scalar => scalar.clone(),
    }
}

/// Derives the public immutable-generation name from its logical content.
///
/// `meta.json` minus its own name and file hashes, every other staging file by
/// name, size and SHA-256, and `mode` (source-bearing or runtime-only).
pub(super) fn content_generation_id(
    directory: &Path,
    meta: &serde_json::Value,
    mode: &str,
    derived: &[(&str, &FileIntegrity)],
) -> Result<String> {
    let mut logical = meta.clone();
    let object = logical.as_object_mut().ok_or_else(|| Error::Invalid {
        path: directory.join("meta.json"),
        message: "metadata root must be a JSON object".to_owned(),
    })?;
    object.insert("generation".to_owned(), serde_json::Value::Null);
    object.insert(
        "files".to_owned(),
        serde_json::Value::Object(serde_json::Map::new()),
    );
    // The same sources under a newer format are a different generation.
    object.insert(
        "format_version".to_owned(),
        crate::format::FORMAT_VERSION.into(),
    );

    let mut names = Vec::new();
    for entry in std::fs::read_dir(directory).map_err(|source| Error::Io {
        path: directory.to_owned(),
        source,
    })? {
        let entry = entry.map_err(|source| Error::Io {
            path: directory.to_owned(),
            source,
        })?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| Error::Invalid {
                path: entry.path(),
                message: "generation file name must be UTF-8".to_owned(),
            })?;
        if name != "meta.json" {
            names.push(name);
        }
    }
    names.sort();

    let mut digest = Sha256::new();
    digest.update(b"udlex-content-generation-v1\0");
    digest.update(mode.as_bytes());
    digest.update(b"\0");
    digest.update(render_meta(&canonical_json(&logical)));
    for name in names {
        let path = directory.join(&name);
        let integrity = file_integrity(&path)?;
        digest.update(u64::try_from(name.len()).unwrap_or(u64::MAX).to_le_bytes());
        digest.update(name.as_bytes());
        digest.update(integrity.size.to_le_bytes());
        digest.update(integrity.sha256.as_bytes());
    }
    if !derived.is_empty() {
        // Derived artifacts whose source was compacted away must bind the ID.
        digest.update(b"\0udlex-derived-artifacts-v1\0");
        digest.update(
            u64::try_from(derived.len())
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        let mut ordered = derived.to_vec();
        ordered.sort_unstable_by_key(|(name, _)| *name);
        for (name, integrity) in ordered {
            digest.update(u64::try_from(name.len()).unwrap_or(u64::MAX).to_le_bytes());
            digest.update(name.as_bytes());
            digest.update(integrity.size.to_le_bytes());
            digest.update(integrity.sha256.as_bytes());
        }
    }
    Ok(format!(
        "g{}",
        crate::secure_fs::hex_lower(&digest.finalize())
    ))
}
