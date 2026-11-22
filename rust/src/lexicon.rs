//! Loading a bundled lexicon and answering lemma queries from it.
//!
//! A lexicon is a handful of memory-mapped lookup tables plus a string table,
//! shareable across threads.
//!
//! Each table is compiled from a sorted TSV into one codec: a **double array**
//! (`.da`, fastest, sparse), a `fst::Map` (`.map`, roughly ten times smaller
//! at 4–7x the probe cost), an **inline** acceptor (`.inline`, keys
//! `form 0x00 varint(cut) suffix`, no string table) or `hybrid` (see
//! [`table`]). [`compile`] weighs the bytes an automaton saves against the
//! table's measured probe traffic and records the choice under `codecs`.
//!
//! A packed suffix rule is 64 bits wide, so `rules.da` stores an index into
//! `rules.values`, a flat array of little-endian `u64`.

use std::ops::Deref;
use std::path::{Path, PathBuf};

use yada::DoubleArray;

use crate::rule::Rule;

/// Anything that can go wrong loading or building a lexicon.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A file could not be read or written.
    #[error("{path}: {source}")]
    Io {
        /// The file involved.
        path: PathBuf,
        /// The underlying cause.
        #[source]
        source: std::io::Error,
    },
    /// A double-array trie could not be built or opened.
    #[error("{path}: {source}")]
    Trie {
        /// The file involved.
        path: PathBuf,
        /// The underlying cause.
        #[source]
        source: yada::errors::YadaError,
    },
    /// A finite-state table could not be built or opened.
    #[error("{path}: {source}")]
    Fst {
        /// The file involved.
        path: PathBuf,
        /// The underlying cause.
        #[source]
        source: fst::Error,
    },
    /// `meta.json` was missing or malformed.
    #[error("{path}: invalid metadata: {source}")]
    Meta {
        /// The file involved.
        path: PathBuf,
        /// The underlying cause.
        #[source]
        source: serde_json::Error,
    },
    /// `meta.json` names a table codec this crate does not implement.
    #[error("{path}: unknown table codec {codec:?}")]
    Codec {
        /// The table involved.
        path: PathBuf,
        /// What the metadata declared.
        codec: String,
    },
    /// The string table outgrew what a `u32` offset can address.
    #[error("{path}: string table larger than 4 GiB")]
    StringsTooLarge {
        /// The file involved.
        path: PathBuf,
    },
    /// The string table is not the UTF-8 the builder writes.
    #[error("{path}: string table is not valid UTF-8")]
    StringsEncoding {
        /// The file involved.
        path: PathBuf,
    },
    /// The tables were never compiled from their sources.
    #[error("{path}: not compiled; run `udlex compile` to build the tables")]
    NotCompiled {
        /// The bundle directory.
        path: PathBuf,
    },
    /// A lexicon directory holds some of the builder's sources but not all.
    #[error(
        "{path}: incomplete sources, missing {missing}; \
         restore all source tables or rebuild the bundle with udlex.build_lexicon"
    )]
    Sources {
        /// The lexicon directory.
        path: PathBuf,
        /// The source files that are not there.
        missing: String,
    },
    /// The bundle declares a format outside the range this operation accepts.
    ///
    /// Both numbers are named because either side can be the stale one, and
    /// the remedy differs: a bundle ahead of the core needs a newer core, a
    /// bundle behind it needs rebuilding.
    #[error(
        "{path}: unsupported bundle format {found}, this udlex reads format {supported}; {}",
        supported.remedy(*found)
    )]
    Format {
        /// The bundle directory.
        path: PathBuf,
        /// What the bundle declares.
        found: u32,
        /// What the failing operation accepts.
        supported: crate::format::FormatRange,
    },
    /// A bundle violates a semantic or cross-file format invariant.
    #[error("{path}: invalid bundle: {message}")]
    Invalid {
        /// File or directory that failed validation.
        path: PathBuf,
        /// Human-readable invariant failure.
        message: String,
    },
    /// A file does not match the generation manifest.
    #[error("{path}: integrity check failed: {message}")]
    Integrity {
        /// Compiled file involved.
        path: PathBuf,
        /// Expected versus observed detail.
        message: String,
    },
    /// A compiler source row is malformed or violates the text format.
    #[error("{path}: line {line}: {message}")]
    Source {
        /// TSV source involved.
        path: PathBuf,
        /// One-based line number.
        line: usize,
        /// Failed invariant.
        message: String,
    },
}

pub type Result<T> = std::result::Result<T, Error>;

pub use crate::format::{
    ENTITY_ENCODING, FORMAT_VERSION, FormatRange, MIN_FORMAT_VERSION, NORMALIZATION, PHRASE_CODEC,
    SOURCES,
};

/// Name of the atomically replaced generation pointer.
pub const CURRENT: &str = "current";

/// Directory holding immutable generations.
pub const GENERATIONS: &str = ".generations";

/// Bytes per entry in `rules.values`.
const VALUE_WIDTH: usize = size_of::<u64>();

/// Bytes per entry in `strings.idx`.
const OFFSET_WIDTH: usize = size_of::<u32>();

/// The variant whose tables are empty by design.
const IDENTITY: &str = "identity";

/// The byte separating the form from the payload in an inline-codec key;
/// [`build_inline`] refuses the codec if a form contains NUL.
const SEP: u8 = 0;

/// How many bytes of double array one unit of probe traffic is worth: a table
/// probed by every token may cost up to 10 MB extra as a double array.
#[cfg(feature = "train")]
const AUTOMATON_BYTES_PER_PROBE_SHARE: f64 = 10e6;

/// The probe share assumed for `lower` and `external` when the bundle
/// records none; a paradigm-seeded lexicon misses more of running text.
#[cfg(feature = "train")]
const DEFAULT_TRAFFIC: f64 = 0.2;

/// See [`DEFAULT_TRAFFIC`].
#[cfg(feature = "train")]
const DEFAULT_TRAFFIC_PARADIGM: f64 = 0.4;

mod catalog;
pub mod codec;
mod layered;
mod metadata;
pub mod runtime;
pub mod table;

#[cfg(feature = "train")]
mod generation;

pub use catalog::{Language, catalog, languages, load};
pub use layered::{LayeredLexicon, RuntimeLexicon};
pub use metadata::{CandidateRecord, FileIntegrity, Meta, SelectionRecord};
pub use runtime::{Embedded, Lexicon};

#[cfg(feature = "train")]
pub use generation::{Compiled, GenerationWriter, compile, prune, render_meta};

#[cfg(feature = "train")]
use codec::Codec;
#[cfg(feature = "train")]
use metadata::validate_meta;
#[cfg(feature = "train")]
use runtime::fst_error;
use runtime::{expected_file, map_verified_file, open_encoded};
#[cfg(feature = "train")]
use runtime::{read_metadata, resolve_generation};
#[cfg(feature = "train")]
use table::build_phrase_table;

#[cfg(feature = "train")]
use runtime::{file_integrity, runtime_files};
#[cfg(feature = "train")]
use table::{encode_key_into, line_offsets};

// The tests build bundles through the trainer and compiler.
#[cfg(all(test, feature = "train"))]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::arithmetic_side_effects
)]
mod tests;
