//! Serialized construction and atomic publication of immutable generations.

use std::fs::{File, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};

use super::super::{CURRENT, Error, GENERATIONS, Lexicon, Result, read_metadata};
use super::identity::new_generation_id;

/// Deletes a file that must not survive this build, if it is there.
pub(super) fn remove(path: &Path) -> Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(Error::Io {
            path: path.to_owned(),
            source,
        }),
    }
}

/// Writes a file, tagging any error with the path.
pub(in crate::lexicon) fn write_file(path: &Path, bytes: &[u8]) -> Result<()> {
    std::fs::write(path, bytes).map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })
}

/// One serialized writer constructing an immutable generation.
pub struct GenerationWriter {
    root: PathBuf,
    generation: String,
    staging: PathBuf,
    final_path: PathBuf,
    /// Held for this writer's lifetime; the OS releases it after a crash.
    _lock: File,
    sealed: bool,
}

impl GenerationWriter {
    /// Acquires the root writer lock and creates a private staging directory.
    pub(crate) fn begin(root: &Path) -> Result<Self> {
        match std::fs::create_dir(root) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(source) => {
                return Err(Error::Io {
                    path: root.to_owned(),
                    source,
                });
            }
        }
        let root_meta = std::fs::symlink_metadata(root).map_err(|source| Error::Io {
            path: root.to_owned(),
            source,
        })?;
        if root_meta.file_type().is_symlink() || !root_meta.is_dir() {
            return Err(Error::Invalid {
                path: root.to_owned(),
                message: "generation root must be a real directory".to_owned(),
            });
        }
        let lock_path = root.join(".write.lock");
        let lock = open_writer_lock(&lock_path)?;

        let generations = root.join(GENERATIONS);
        if let Err(source) = std::fs::create_dir(&generations)
            && source.kind() != std::io::ErrorKind::AlreadyExists
        {
            return Err(Error::Io {
                path: generations,
                source,
            });
        }
        let generations_meta =
            std::fs::symlink_metadata(&generations).map_err(|source| Error::Io {
                path: generations.clone(),
                source,
            })?;
        if generations_meta.file_type().is_symlink() || !generations_meta.is_dir() {
            return Err(Error::Invalid {
                path: generations,
                message: "generations path must be a real directory".to_owned(),
            });
        }
        let generation = new_generation_id();
        let staging = generations.join(format!(".building-{generation}"));
        let final_path = generations.join(&generation);
        std::fs::create_dir(&staging).map_err(|source| Error::Io {
            path: staging.clone(),
            source,
        })?;
        Ok(Self {
            root: root.to_owned(),
            generation,
            staging,
            final_path,
            _lock: lock,
            sealed: false,
        })
    }

    /// Identifier to record in the generation metadata.
    pub(crate) fn id(&self) -> &str {
        &self.generation
    }

    /// Private directory into which all files must be written.
    pub(crate) fn directory(&self) -> &Path {
        &self.staging
    }

    /// Replaces the process-unique staging identity with a content address.
    pub(super) fn assign_generation(&mut self, generation: String) {
        self.final_path = self.root.join(GENERATIONS).join(&generation);
        self.generation = generation;
    }

    /// Makes the completed directory immutable by name but does not publish it.
    pub(crate) fn seal(&mut self) -> Result<&Path> {
        if !self.sealed {
            sync_directory(&self.staging)?;
            if let Ok(existing) = std::fs::symlink_metadata(&self.final_path) {
                if existing.file_type().is_symlink() || !existing.is_dir() {
                    return Err(Error::Invalid {
                        path: self.final_path.clone(),
                        message: "content-addressed generation is not a real directory".to_owned(),
                    });
                }
                let staged_meta = read_metadata(&self.staging.join("meta.json"))?;
                let existing_meta = read_metadata(&self.final_path.join("meta.json"))?;
                if staged_meta != existing_meta {
                    return Err(Error::Invalid {
                        path: self.final_path.clone(),
                        message: "content-address collision or inconsistent existing generation"
                            .to_owned(),
                    });
                }
                Lexicon::open(&self.final_path)?;
                std::fs::remove_dir_all(&self.staging).map_err(|source| Error::Io {
                    path: self.staging.clone(),
                    source,
                })?;
                self.sealed = true;
                return Ok(&self.final_path);
            }
            std::fs::rename(&self.staging, &self.final_path).map_err(|source| Error::Io {
                path: self.final_path.clone(),
                source,
            })?;
            sync_directory(self.final_path.parent().ok_or_else(|| Error::Invalid {
                path: self.final_path.clone(),
                message: "generation has no parent".to_owned(),
            })?)?;
            self.sealed = true;
        }
        Ok(&self.final_path)
    }

    /// Atomically points new readers at this complete generation.
    pub(crate) fn publish(mut self) -> Result<PathBuf> {
        self.seal()?;
        let current = self.root.join(CURRENT);
        if std::fs::symlink_metadata(&current).is_ok_and(|meta| meta.file_type().is_symlink()) {
            return Err(Error::Invalid {
                path: current,
                message: "refusing to replace a symlinked current pointer".to_owned(),
            });
        }
        let mut temporary =
            tempfile::NamedTempFile::new_in(&self.root).map_err(|source| Error::Io {
                path: self.root.clone(),
                source,
            })?;
        writeln!(temporary, "{}", self.generation).map_err(|source| Error::Io {
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
        temporary.persist(&current).map_err(|error| Error::Io {
            path: current,
            source: error.error,
        })?;
        sync_directory(&self.root)?;
        Ok(self.final_path.clone())
    }
}

impl Drop for GenerationWriter {
    fn drop(&mut self) {
        if !self.sealed {
            let _ = std::fs::remove_dir_all(&self.staging);
        }
    }
}

/// Opens the persistent writer lock without following a symlink and takes a
/// non-blocking advisory lock. Never unlinked: a second writer could then
/// lock a fresh inode while the first still holds the old one.
fn open_writer_lock(path: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.create(true).read(true).write(true);
    crate::secure_fs::no_follow(&mut options);
    let mut lock = options.open(path).map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })?;
    let path_metadata = std::fs::symlink_metadata(path).map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })?;
    let file_metadata = lock.metadata().map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })?;
    if path_metadata.file_type().is_symlink()
        || !path_metadata.is_file()
        || !file_metadata.is_file()
    {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: "writer lock must be a regular file, not a symlink or special file".to_owned(),
        });
    }
    if !crate::secure_fs::same_file(&path_metadata, &file_metadata) {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: "writer lock path changed while it was opened".to_owned(),
        });
    }
    lock.try_lock().map_err(|error| Error::Io {
        path: path.to_owned(),
        source: error.into(),
    })?;
    lock.set_len(0).map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })?;
    writeln!(lock, "{}", std::process::id()).map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })?;
    lock.sync_all().map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })?;
    Ok(lock)
}

/// Opens a regular file to flush it. Windows refuses `FlushFileBuffers` on a
/// read-only handle, so there the file is opened for writing (nothing is written).
fn open_for_sync(path: &Path) -> std::io::Result<File> {
    if cfg!(windows) {
        OpenOptions::new().write(true).open(path)
    } else {
        File::open(path)
    }
}

/// Flushes all regular child files and then the directory entry itself.
fn sync_directory(directory: &Path) -> Result<()> {
    for entry in std::fs::read_dir(directory).map_err(|source| Error::Io {
        path: directory.to_owned(),
        source,
    })? {
        let path = entry
            .map_err(|source| Error::Io {
                path: directory.to_owned(),
                source,
            })?
            .path();
        let metadata = std::fs::symlink_metadata(&path).map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
        if metadata.file_type().is_symlink() {
            return Err(Error::Invalid {
                path,
                message: "generation contains a symlink".to_owned(),
            });
        }
        if metadata.is_file() {
            open_for_sync(&path)
                .and_then(|file| file.sync_all())
                .map_err(|source| Error::Io {
                    path: path.clone(),
                    source,
                })?;
        }
    }
    sync_directory_entry(directory)
}

/// Flushes the directory entry; best effort off Unix, where std cannot open
/// a directory as a `File`.
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
