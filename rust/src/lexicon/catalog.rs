use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use super::metadata::Meta;
use super::runtime::{Lexicon, LoadedMetadata, load_metadata, read_bounded_regular};
use super::{Error, Result};
use crate::format::{FormatRange, LookupTableKind};
use crate::secure_fs::valid_generation_id;

const MAX_BUNDLE_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
pub(super) const WHEEL_MANIFESTS: &str = ".manifests";
pub(super) const WHEEL_MANIFEST_PREFIX: &str = "udlex_dict_";

/// One language a bundle can lemmatise.
///
/// Read [`accuracy`](Self::accuracy) against [`baseline`](Self::baseline):
/// the baseline runs from 0.13 for Arabic to 1.0 for Wu Chinese.
#[derive(Debug, Clone, PartialEq)]
pub struct Language {
    /// ISO 639-3 code, the name to ask for this language by.
    pub iso639_3: String,
    /// Human-readable language name.
    pub name: String,
    /// The lexicon backing it, normally a Universal Dependencies treebank.
    pub lexicon: String,
    /// Which parts of the learned lexicon the bundle ships: `full`, `lookup`,
    /// `rules`, `corpus`, or `identity`.
    pub variant: String,
    /// How many annotated tokens it was learned from.
    pub training_tokens: u64,
    /// How many form-to-lemma rows its lookup tables hold.
    pub entries: u64,
    /// Exact-match accuracy on the held-out test split, when one existed.
    pub accuracy: Option<f64>,
    /// What copying the surface form scores on the same tokens.
    pub baseline: Option<f64>,
    /// Whether held-out scoring matched or exceeded copying the surface form.
    /// `None` means no independent measurement is attached to this bundle.
    pub is_useful: Option<bool>,
    /// The sample that chose the shipped variant: `dev` or `train-cv`.
    pub selected_on: Option<String>,
    /// Exact-match accuracy of the shipped variant on that sample; read it
    /// against [`Language::selection_baseline`].
    pub selection_accuracy: Option<f64>,
    /// What copying the surface form scored on the same sample.
    pub selection_baseline: Option<f64>,
}

impl Language {
    /// How much accuracy the lexicon adds over copying the surface form;
    /// `None` when either side was never measured.
    #[must_use]
    pub fn gain(&self) -> Option<f64> {
        Some(self.accuracy? - self.baseline?)
    }

    /// The catalogue entry for one lexicon's metadata.
    fn of(meta: Meta, lexicon: String) -> Self {
        let is_useful = meta.useful();
        let (selection_accuracy, selection_baseline) = meta
            .selection
            .as_ref()
            .filter(|record| record.tokens > 0)
            .and_then(|record| {
                let hits = record.candidates.get(&meta.variant)?.hits;
                let tokens = record.tokens as f64;
                Some((hits as f64 / tokens, record.copied as f64 / tokens))
            })
            .unzip();
        Self {
            iso639_3: meta.iso639_3,
            name: meta.language,
            lexicon,
            variant: meta.variant,
            training_tokens: meta.training_tokens,
            entries: u64::try_from(
                LookupTableKind::ALL
                    .into_iter()
                    .filter_map(|table| meta.entries.get(table.name()).copied())
                    .map(u128::from)
                    .sum::<u128>(),
            )
            .unwrap_or(u64::MAX),
            accuracy: meta.accuracy,
            baseline: meta.baseline,
            is_useful,
            selected_on: meta.selected_on,
            selection_accuracy,
            selection_baseline,
        }
    }
}

#[derive(serde::Deserialize)]
struct BundleManifest {
    format_version: u32,
    lexicons: Vec<DeclaredLexicon>,
}

#[derive(serde::Deserialize)]
struct DeclaredLexicon {
    iso639_3: String,
    config: String,
    generation: Option<String>,
}

fn valid_iso639_3(value: &str) -> bool {
    value.len() == 3 && value.bytes().all(|byte| byte.is_ascii_lowercase())
}

fn valid_config(value: &str) -> bool {
    !value.is_empty()
        && value.split('_').all(|part| {
            !part.is_empty()
                && part
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
        })
}

fn read_catalogue_loaded(
    path: &Path,
    expected_iso: Option<&str>,
    expected_generation: Option<&str>,
) -> Result<LoadedMetadata> {
    let loaded = load_metadata(path)?;
    if path.file_name().and_then(|name| name.to_str()) != Some(&loaded.meta.name) {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: format!(
                "directory name does not match metadata name {:?}",
                loaded.meta.name
            ),
        });
    }
    if let Some(expected) = expected_iso
        && expected != loaded.meta.iso639_3
    {
        return Err(Error::Invalid {
            path: loaded.resolved.directory.join("meta.json"),
            message: format!(
                "manifest ISO 639-3 code {expected:?} does not match metadata code {:?}",
                loaded.meta.iso639_3,
            ),
        });
    }
    if let Some(expected) = expected_generation
        && loaded.resolved.generation.as_deref() != Some(expected)
    {
        return Err(Error::Invalid {
            path: path.join("current"),
            message: format!(
                "manifest generation {expected:?} does not match active generation {:?}",
                loaded.resolved.generation,
            ),
        });
    }
    Ok(loaded)
}

fn read_catalogue_meta(
    path: &Path,
    expected_iso: Option<&str>,
    expected_generation: Option<&str>,
) -> Result<(Meta, PathBuf)> {
    let loaded = read_catalogue_loaded(path, expected_iso, expected_generation)?;
    Ok((loaded.meta, path.to_owned()))
}

fn read_manifest(manifest_path: &Path) -> Result<Vec<DeclaredLexicon>> {
    let raw = read_bounded_regular(manifest_path, MAX_BUNDLE_MANIFEST_BYTES, "bundle manifest")?;
    let manifest: BundleManifest = serde_json::from_str(&raw).map_err(|source| Error::Invalid {
        path: manifest_path.to_owned(),
        message: format!("invalid bundle manifest JSON: {source}"),
    })?;
    if !FormatRange::READ.accepts(manifest.format_version) {
        return Err(Error::Format {
            path: manifest_path.to_owned(),
            found: manifest.format_version,
            supported: FormatRange::READ,
        });
    }

    let mut seen_iso = BTreeSet::new();
    let mut seen_config = BTreeSet::new();
    for declaration in &manifest.lexicons {
        if !valid_iso639_3(&declaration.iso639_3)
            || !valid_config(&declaration.config)
            || declaration
                .generation
                .as_deref()
                .is_some_and(|generation| !valid_generation_id(generation))
        {
            return Err(Error::Invalid {
                path: manifest_path.to_owned(),
                message: format!(
                    "manifest row needs canonical iso639_3/config, got {:?}/{:?}",
                    declaration.iso639_3, declaration.config,
                ),
            });
        }
        if !seen_iso.insert(declaration.iso639_3.clone())
            || !seen_config.insert(declaration.config.clone())
        {
            return Err(Error::Invalid {
                path: manifest_path.to_owned(),
                message: format!(
                    "duplicate manifest lexicon identity {:?}/{:?}",
                    declaration.iso639_3, declaration.config,
                ),
            });
        }
    }
    Ok(manifest.lexicons)
}

/// Reads the one authoritative declaration set for a bundle directory.
///
/// Either one root `manifest.json` or one `.manifests/udlex_dict_<iso>.json`
/// per dictionary wheel, never both.
fn read_declarations(directory: &Path) -> Result<Option<Vec<DeclaredLexicon>>> {
    let manifest_path = directory.join("manifest.json");
    let wheel_directory = directory.join(WHEEL_MANIFESTS);
    let has_manifest = manifest_exists(&manifest_path)?;
    let has_wheel_manifests = manifest_exists(&wheel_directory)?;
    if has_manifest && has_wheel_manifests {
        return Err(Error::Invalid {
            path: directory.to_owned(),
            message: "root manifest and wheel manifests cannot coexist".to_owned(),
        });
    }
    if has_manifest {
        return read_manifest(&manifest_path).map(Some);
    }
    if !has_wheel_manifests {
        return Ok(None);
    }
    let paths = wheel_manifest_paths(&wheel_directory)?;
    read_wheel_declarations(&wheel_directory, paths).map(Some)
}

/// Lists the manifest files a wheel container declares, in filename order;
/// any unexpected entry is refused.
fn wheel_manifest_paths(wheel_directory: &Path) -> Result<Vec<(String, PathBuf)>> {
    let metadata = std::fs::symlink_metadata(wheel_directory).map_err(|source| Error::Io {
        path: wheel_directory.to_owned(),
        source,
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(Error::Invalid {
            path: wheel_directory.to_owned(),
            message: "wheel manifest container must be a real directory".to_owned(),
        });
    }
    let entries = std::fs::read_dir(wheel_directory).map_err(|source| Error::Io {
        path: wheel_directory.to_owned(),
        source,
    })?;
    let mut paths = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|source| Error::Io {
            path: wheel_directory.to_owned(),
            source,
        })?;
        let path = entry.path();
        let file_type = entry.file_type().map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            return Err(Error::Invalid {
                path,
                message: "wheel manifest filename must be UTF-8".to_owned(),
            });
        };
        if !file_type.is_file()
            || !name.starts_with(WHEEL_MANIFEST_PREFIX)
            || path.extension().is_none_or(|extension| extension != "json")
        {
            return Err(Error::Invalid {
                path,
                message: "wheel manifest container has an unexpected entry".to_owned(),
            });
        }
        paths.push((name, path));
    }
    paths.sort_by(|left, right| left.0.cmp(&right.0));
    if paths.is_empty() {
        return Err(Error::Invalid {
            path: wheel_directory.to_owned(),
            message: "wheel manifest container is empty".to_owned(),
        });
    }
    Ok(paths)
}

/// Authenticates one lexicon per wheel manifest and rejects any overlap.
fn read_wheel_declarations(
    wheel_directory: &Path,
    paths: Vec<(String, PathBuf)>,
) -> Result<Vec<DeclaredLexicon>> {
    let mut declarations = Vec::with_capacity(paths.len());
    let mut seen_iso = BTreeSet::new();
    let mut seen_config = BTreeSet::new();
    for (name, path) in paths {
        let mut rows = read_manifest(&path)?;
        let (Some(declaration), true) = (rows.pop(), rows.is_empty()) else {
            return Err(Error::Invalid {
                path,
                message: "each dictionary wheel manifest must declare exactly one lexicon"
                    .to_owned(),
            });
        };
        if declaration.generation.is_none() {
            return Err(Error::Invalid {
                path,
                message: "dictionary wheel manifest must bind an immutable generation".to_owned(),
            });
        }
        let expected_name = format!("{WHEEL_MANIFEST_PREFIX}{}.json", declaration.iso639_3);
        if name != expected_name {
            return Err(Error::Invalid {
                path,
                message: format!(
                    "wheel manifest filename {name:?} does not match declared ISO code {:?}",
                    declaration.iso639_3,
                ),
            });
        }
        if !seen_iso.insert(declaration.iso639_3.clone())
            || !seen_config.insert(declaration.config.clone())
        {
            return Err(Error::Invalid {
                path: wheel_directory.to_owned(),
                message: format!(
                    "duplicate wheel manifest lexicon identity {:?}/{:?}",
                    declaration.iso639_3, declaration.config,
                ),
            });
        }
        declarations.push(declaration);
    }
    Ok(declarations)
}

/// Reads exactly the lexicons declared by a V1 bundle manifest; any bad row
/// fails the whole bundle.
fn scan_declarations(
    directory: &Path,
    declarations: Vec<DeclaredLexicon>,
) -> Result<Vec<(Meta, PathBuf)>> {
    let mut found = Vec::with_capacity(declarations.len());
    for declaration in declarations {
        found.push(read_catalogue_meta(
            &directory.join(&declaration.config),
            Some(&declaration.iso639_3),
            declaration.generation.as_deref(),
        )?);
    }
    found.sort_by(|(left, left_path), (right, right_path)| {
        (&left.iso639_3, left_path).cmp(&(&right.iso639_3, right_path))
    });
    Ok(found)
}

fn manifest_exists(path: &Path) -> Result<bool> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(source) => Err(Error::Io {
            path: path.to_owned(),
            source,
        }),
    }
}

/// Reads the metadata of every lexicon in a bundle, ordered by language code.
///
/// A manifest-backed bundle fails closed; an unmanifested directory is
/// scanned best-effort.
fn scan(directory: &Path) -> Result<Vec<(Meta, PathBuf)>> {
    if let Some(declarations) = read_declarations(directory)? {
        return scan_declarations(directory, declarations);
    }
    let entries = std::fs::read_dir(directory).map_err(|source| Error::Io {
        path: directory.to_owned(),
        source,
    })?;
    let mut found: Vec<(Meta, PathBuf)> = entries
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .filter_map(|path| read_catalogue_meta(&path, None, None).ok())
        .collect();
    found.sort_by(|(left, left_path), (right, right_path)| {
        (&left.iso639_3, left_path).cmp(&(&right.iso639_3, right_path))
    });
    if let Some((iso639_3, first, second)) = found.windows(2).find_map(|pair| match pair {
        [left, right] if left.0.iso639_3 == right.0.iso639_3 => {
            Some((&left.0.iso639_3, &left.1, &right.1))
        }
        _ => None,
    }) {
        return Err(Error::Invalid {
            path: directory.to_owned(),
            message: format!(
                "duplicate ISO 639-3 code {:?} in {} and {}",
                iso639_3,
                first.file_name().unwrap_or_default().display(),
                second.file_name().unwrap_or_default().display(),
            ),
        });
    }
    Ok(found)
}

/// Lists the languages a bundle can lemmatise, ordered by ISO 639-3 code,
/// without opening any lookup table.
///
/// # Errors
///
/// Returns an error when the bundle directory cannot be listed, or when a
/// manifest-declared lexicon is missing, malformed, or inconsistent.
pub fn languages(directory: impl AsRef<Path>) -> Result<Vec<Language>> {
    Ok(scan(directory.as_ref())?
        .into_iter()
        .map(|(meta, path)| {
            let lexicon = path
                .file_name()
                .map_or_else(String::new, |name| name.to_string_lossy().into_owned());
            Language::of(meta, lexicon)
        })
        .collect())
}

/// Lists the lexicons in a bundle as `(ISO 639-3 code, directory)` pairs.
///
/// # Errors
///
/// Returns an error when the directory cannot be listed, or when a
/// manifest-declared lexicon is missing, malformed, or inconsistent.
pub fn catalog(directory: impl AsRef<Path>) -> Result<Vec<(String, PathBuf)>> {
    Ok(scan(directory.as_ref())?
        .into_iter()
        .map(|(meta, path)| (meta.iso639_3, path))
        .collect())
}

/// Opens the lexicon a bundle holds for one language.
///
/// `language` may be an ISO 639-3 code (`rus`) or a treebank name (`ru_taiga`).
/// In a manifest-backed V1 bundle both forms must name a declared row; stale
/// directories beside the manifest are never treated as installed languages.
///
/// # Errors
///
/// Returns an error when the bundle cannot be listed, holds no such language,
/// or the lexicon itself cannot be opened.
pub fn load(bundle: impl AsRef<Path>, language: &str) -> Result<Lexicon> {
    let bundle = bundle.as_ref();
    let mut components = Path::new(language).components();
    if language.is_empty()
        || !matches!(components.next(), Some(std::path::Component::Normal(_)))
        || components.next().is_some()
    {
        return Err(Error::Invalid {
            path: bundle.to_owned(),
            message: format!("language {language:?} must be one normal path component"),
        });
    }
    if let Some(declarations) = read_declarations(bundle)? {
        return declarations
            .into_iter()
            .find(|declaration| declaration.iso639_3 == language || declaration.config == language)
            .map_or_else(
                || {
                    Err(Error::Io {
                        path: bundle.join(language),
                        source: std::io::Error::new(
                            std::io::ErrorKind::NotFound,
                            format!("no declared lexicon for {language:?} in this bundle"),
                        ),
                    })
                },
                |declaration| {
                    let path = bundle.join(declaration.config);
                    let loaded = read_catalogue_loaded(
                        &path,
                        Some(&declaration.iso639_3),
                        declaration.generation.as_deref(),
                    )?;
                    Lexicon::open_loaded(loaded)
                },
            );
    }
    let direct = bundle.join(language);
    if direct.is_dir() {
        return Lexicon::open(direct);
    }
    catalog(bundle)?
        .into_iter()
        .find(|(iso, _)| iso == language)
        .map_or_else(
            || {
                Err(Error::Io {
                    path: bundle.join(language),
                    source: std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("no lexicon for {language:?} in this bundle"),
                    ),
                })
            },
            |(_, path)| Lexicon::open(path),
        )
}
