//! Bounded, strict text input shared by corpus and paradigm readers.

use std::fs::{File, Metadata};
use std::io::{BufRead as _, Read as _};
use std::path::Path;

use sha2::{Digest as _, Sha256};

use super::{Error, Result};
use crate::lexicon::FileIntegrity;
use crate::secure_fs::{OpenRegularError, StabilityError};

/// Bounds damage from a missing newline without constraining real sentences.
pub(super) const MAX_SOURCE_LINE_BYTES: usize = 16 * 1024 * 1024;

fn invalid_input(path: &Path, message: &'static str) -> Error {
    Error::Io {
        path: path.to_owned(),
        source: std::io::Error::new(std::io::ErrorKind::InvalidInput, message),
    }
}

/// Opens one regular input without following the final path component.
pub(super) fn open_regular(path: &Path) -> Result<(File, Metadata)> {
    crate::secure_fs::open_regular(path).map_err(|error| match error {
        OpenRegularError::Io(source) => Error::Io {
            path: path.to_owned(),
            source,
        },
        OpenRegularError::NotRegular => invalid_input(
            path,
            "builder inputs must be regular files, not symlinks or special files",
        ),
        OpenRegularError::Changed => {
            invalid_input(path, "builder input path changed while it was opened")
        }
    })
}

/// Verifies that the descriptor and its path still name the bytes just read.
pub(super) fn verify_stable(
    path: &Path,
    before: &Metadata,
    file: &File,
    observed_size: u64,
) -> Result<()> {
    crate::secure_fs::verify_stable(path, before, file, observed_size).map_err(
        |error| match error {
            StabilityError::Io(source) => Error::Io {
                path: path.to_owned(),
                source,
            },
            StabilityError::Changed => invalid_input(
                path,
                "builder input descriptor or path changed while it was read",
            ),
        },
    )
}

/// Checks an authenticated declaration against the exact descriptor bytes.
pub(super) fn verify_expected(
    path: &Path,
    observed: &FileIntegrity,
    expected: Option<&FileIntegrity>,
) -> Result<()> {
    if let Some(expected) = expected
        && expected != observed
    {
        return Err(Error::InputIntegrity {
            path: path.to_owned(),
            observed: observed.clone(),
            expected: expected.clone(),
        });
    }
    Ok(())
}

/// Whether a split or table is a non-empty regular file.
pub(super) fn exists_nonempty(path: &Path) -> Result<bool> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(source) => {
            return Err(Error::Io {
                path: path.to_owned(),
                source,
            });
        }
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(invalid_input(
            path,
            "builder inputs must be regular files, not symlinks or special files",
        ));
    }
    Ok(metadata.len() > 0)
}

/// Visits strict UTF-8 lines while keeping allocation bounded.
pub(super) fn for_each_line(
    path: &Path,
    mut visit: impl FnMut(usize, &str) -> Result<()>,
) -> Result<FileIntegrity> {
    let (file, descriptor_before) = open_regular(path)?;
    let mut reader = std::io::BufReader::new(file);
    let mut bytes = Vec::new();
    let mut line = 0_usize;
    let mut size = 0_u64;
    let mut digest = Sha256::new();
    loop {
        bytes.clear();
        let mut bounded = (&mut reader).take((MAX_SOURCE_LINE_BYTES + 2) as u64);
        let read = bounded
            .read_until(b'\n', &mut bytes)
            .map_err(|source| Error::Io {
                path: path.to_owned(),
                source,
            })?;
        if read == 0 {
            break;
        }
        size = size
            .checked_add(u64::try_from(read).unwrap_or(u64::MAX))
            .ok_or_else(|| invalid_input(path, "builder input length overflowed u64"))?;
        digest.update(&bytes);
        if bytes.last() == Some(&b'\n') {
            bytes.pop();
            if bytes.last() == Some(&b'\r') {
                bytes.pop();
            }
        }
        if bytes.len() > MAX_SOURCE_LINE_BYTES {
            return Err(Error::LineTooLong {
                path: path.to_owned(),
                line: line.saturating_add(1),
                limit: MAX_SOURCE_LINE_BYTES,
            });
        }
        let text =
            String::from_utf8(std::mem::take(&mut bytes)).map_err(|source| Error::Encoding {
                path: path.to_owned(),
                source,
            })?;
        if text.is_empty() {
            line = line.saturating_add(1);
            visit(line, "")?;
        } else {
            for logical in text.split_terminator('\r') {
                line = line.saturating_add(1);
                visit(line, logical)?;
            }
        }
        bytes = text.into_bytes();
    }
    verify_stable(path, &descriptor_before, reader.get_ref(), size)?;
    Ok(FileIntegrity {
        size,
        sha256: crate::secure_fs::hex_lower(&digest.finalize()),
    })
}
