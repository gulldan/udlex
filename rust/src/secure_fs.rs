//! Shared primitives for reading path-addressed files without following the
//! final component or confusing a replaced directory entry with the opened
//! descriptor.

use std::fs::{File, Metadata, OpenOptions};
use std::io::{Read as _, Seek as _, SeekFrom};
use std::path::Path;

use memmap2::Mmap;
use sha2::{Digest as _, Sha256};

/// Why a path could not be opened as the regular file it named at inspection.
#[derive(Debug)]
pub enum OpenRegularError {
    /// A filesystem operation failed.
    Io(std::io::Error),
    /// The path or descriptor is a symlink, directory, or special file.
    NotRegular,
    /// The path was replaced between inspection and descriptor acquisition.
    Changed,
}

/// Why a descriptor no longer represents the stable path that was opened.
#[derive(Debug)]
pub enum StabilityError {
    /// A filesystem operation failed.
    Io(std::io::Error),
    /// Identity, timestamps, or byte length changed.
    Changed,
}

/// Size and SHA-256 observed while streaming an opened descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Integrity {
    pub size: u64,
    pub sha256: String,
}

/// Why a regular file could not be authenticated and mapped.
#[derive(Debug)]
pub enum MapVerifiedError {
    Open(OpenRegularError),
    Read(std::io::Error),
    LengthOverflow,
    Changed,
    Mismatch(Integrity),
    Map(std::io::Error),
}

/// A verified mapping, with an explicit representation for a zero-byte file.
pub enum VerifiedMapping {
    Empty,
    Mapped(Mmap),
}

/// Whether an immutable generation identifier is one safe path component.
pub fn valid_generation_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && !matches!(value, "." | "..")
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

/// Adds the platform flag that opens the final path component itself.
pub fn no_follow(options: &mut OpenOptions) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NOFOLLOW);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt as _;
        const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
        options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    }
    #[cfg(not(any(unix, windows)))]
    let _ = options;
}

/// Opens a regular file and proves the path still names that descriptor.
pub fn open_regular(path: &Path) -> Result<(File, Metadata), OpenRegularError> {
    let path_metadata = std::fs::symlink_metadata(path).map_err(OpenRegularError::Io)?;
    if path_metadata.file_type().is_symlink() || !path_metadata.is_file() {
        return Err(OpenRegularError::NotRegular);
    }
    let mut options = OpenOptions::new();
    options.read(true);
    no_follow(&mut options);
    let file = options.open(path).map_err(OpenRegularError::Io)?;
    let descriptor_metadata = file.metadata().map_err(OpenRegularError::Io)?;
    if !descriptor_metadata.is_file() {
        return Err(OpenRegularError::NotRegular);
    }
    if !same_file(&path_metadata, &descriptor_metadata) {
        return Err(OpenRegularError::Changed);
    }
    Ok((file, descriptor_metadata))
}

/// Checks that a descriptor and its path still name the bytes just consumed.
pub fn verify_stable(
    path: &Path,
    before: &Metadata,
    file: &File,
    observed_size: u64,
) -> Result<(), StabilityError> {
    let descriptor_after = file.metadata().map_err(StabilityError::Io)?;
    let path_after = std::fs::symlink_metadata(path).map_err(StabilityError::Io)?;
    if path_after.file_type().is_symlink()
        || !path_after.is_file()
        || !unchanged(before, &descriptor_after)
        || !same_file(&descriptor_after, &path_after)
        || descriptor_after.len() != observed_size
    {
        return Err(StabilityError::Changed);
    }
    Ok(())
}

/// Streams one stable regular file through SHA-256.
#[cfg(any(feature = "train", feature = "user-dictionary-write"))]
pub fn file_integrity(path: &Path) -> Result<Integrity, MapVerifiedError> {
    let (mut file, before) = open_regular(path).map_err(MapVerifiedError::Open)?;
    let observed = opened_integrity(&mut file)?;
    verify_stable(path, &before, &file, observed.size).map_err(|error| match error {
        StabilityError::Io(source) => MapVerifiedError::Read(source),
        StabilityError::Changed => MapVerifiedError::Changed,
    })?;
    Ok(observed)
}

/// Authenticates and maps the same stable regular-file descriptor.
pub fn map_verified(
    path: &Path,
    expected: &Integrity,
) -> Result<VerifiedMapping, MapVerifiedError> {
    let (mut file, before) = open_regular(path).map_err(MapVerifiedError::Open)?;
    let observed = opened_integrity(&mut file)?;
    if &observed != expected {
        return Err(MapVerifiedError::Mismatch(observed));
    }
    verify_stable(path, &before, &file, observed.size).map_err(|error| match error {
        StabilityError::Io(source) => MapVerifiedError::Read(source),
        StabilityError::Changed => MapVerifiedError::Changed,
    })?;
    if observed.size == 0 {
        return Ok(VerifiedMapping::Empty);
    }
    file.seek(SeekFrom::Start(0))
        .map_err(MapVerifiedError::Read)?;
    // SAFETY: this is the descriptor whose full contents and stable path were
    // verified above. Immutable published generations must not mutate the
    // inode while a reader holds the mapping.
    let mapping = unsafe { Mmap::map(&file) }.map_err(MapVerifiedError::Map)?;
    Ok(VerifiedMapping::Mapped(mapping))
}

pub fn opened_integrity(file: &mut File) -> Result<Integrity, MapVerifiedError> {
    let mut digest = Sha256::new();
    let mut size = 0_u64;
    let mut buffer = vec![0_u8; 128 * 1024].into_boxed_slice();
    loop {
        let read = file.read(&mut buffer).map_err(MapVerifiedError::Read)?;
        if read == 0 {
            break;
        }
        size = size
            .checked_add(u64::try_from(read).unwrap_or(u64::MAX))
            .ok_or(MapVerifiedError::LengthOverflow)?;
        digest.update(buffer.get(..read).unwrap_or_default());
    }
    Ok(Integrity {
        size,
        sha256: hex_lower(&digest.finalize()),
    })
}

#[cfg(unix)]
pub fn same_file(left: &Metadata, right: &Metadata) -> bool {
    use std::os::unix::fs::MetadataExt as _;

    (left.dev(), left.ino()) == (right.dev(), right.ino())
}

#[cfg(windows)]
pub fn same_file(left: &Metadata, right: &Metadata) -> bool {
    use std::os::windows::fs::MetadataExt as _;

    left.file_attributes() == right.file_attributes()
        && left.creation_time() == right.creation_time()
        && left.file_size() == right.file_size()
}

#[cfg(not(any(unix, windows)))]
pub fn same_file(left: &Metadata, right: &Metadata) -> bool {
    left.is_file() && right.is_file() && left.len() == right.len()
}

#[cfg(unix)]
fn unchanged(left: &Metadata, right: &Metadata) -> bool {
    use std::os::unix::fs::MetadataExt as _;

    same_file(left, right)
        && left.len() == right.len()
        && (left.mtime(), left.mtime_nsec()) == (right.mtime(), right.mtime_nsec())
        && (left.ctime(), left.ctime_nsec()) == (right.ctime(), right.ctime_nsec())
}

#[cfg(windows)]
fn unchanged(left: &Metadata, right: &Metadata) -> bool {
    use std::os::windows::fs::MetadataExt as _;

    same_file(left, right)
        && left.file_size() == right.file_size()
        && left.last_write_time() == right.last_write_time()
}

#[cfg(not(any(unix, windows)))]
fn unchanged(left: &Metadata, right: &Metadata) -> bool {
    same_file(left, right) && left.modified().ok() == right.modified().ok()
}

/// Lower-case hex of a digest; `sha2` 0.11 output has no `LowerHex`.
pub fn hex_lower(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(
        String::with_capacity(bytes.len().saturating_mul(2)),
        |mut out, byte| {
            let _ = write!(out, "{byte:02x}");
            out
        },
    )
}

#[cfg(test)]
mod tests {
    use super::valid_generation_id;

    #[test]
    fn generation_ids_are_single_bounded_components() {
        assert!(valid_generation_id("sha256_0123-ABCD"));
        for invalid in ["", ".", "..", "with/slash", "with space", "é"] {
            assert!(!valid_generation_id(invalid), "{invalid:?}");
        }
        assert!(!valid_generation_id(&"a".repeat(129)));
    }
}
