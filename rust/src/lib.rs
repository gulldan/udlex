//! udlex — a high-throughput lookup-and-suffix-rule lemmatizer.
//!
//! A token is answered by the first stage that knows it:
//!
//! 1. optional isolated entity exact/lower tables;
//! 2. corpus exact/lower tables;
//! 3. an offline table merged from dictionaries and analysers;
//! 4. the longest matching suffix rule — a guess for words never seen;
//! 5. the token itself.
//!
//! Every stage is a lookup in a memory-mapped table.
//!
//! # Example
//!
//! ```no_run
//! use udlex_rs::Lexicon;
//!
//! let lexicon = Lexicon::open("bundle/ru_taiga")?;
//! assert_eq!(lexicon.lemma("котами", false), "кот");
//! # Ok::<(), udlex_rs::Error>(())
//! ```

//! # Features
//!
//! * `infer` (default) — opening a bundle and answering from it.
//! * `train` — learning a lexicon and compiling a bundle; enables `infer`.
//!
//! The on-disk format constants live in [`format`](mod@crate::format), which
//! both halves share.

pub mod format;

#[cfg(feature = "infer")]
mod lexicon;
#[cfg(feature = "infer")]
mod normalization;
#[cfg(feature = "infer")]
mod probe;
#[cfg(feature = "infer")]
mod rule;
#[cfg(feature = "infer")]
mod secure_fs;
#[cfg(feature = "infer")]
mod span;
#[cfg(feature = "infer")]
mod user_dictionary;

#[cfg(feature = "train")]
mod builder;
#[cfg(feature = "train")]
mod train;

pub use format::{
    MAX_USER_DICTIONARY_ENTRIES, MAX_USER_DICTIONARY_FIELD_BYTES, USER_DICTIONARY_CASING,
    USER_DICTIONARY_SCHEMA, UserMatch,
};
#[cfg(feature = "infer")]
pub use lexicon::{
    CandidateRecord, ENTITY_ENCODING, Embedded, Error, FORMAT_VERSION, FileIntegrity, FormatRange,
    Language, LayeredLexicon, Lexicon, MIN_FORMAT_VERSION, Meta, NORMALIZATION, PHRASE_CODEC,
    RuntimeLexicon, SOURCES, SelectionRecord, catalog, languages, load,
};
#[cfg(feature = "infer")]
pub use probe::{Answer, Options, PHRASE_SEPARATOR, PHRASE_TAIL, Source, SpanConventions};
#[cfg(feature = "infer")]
pub use rule::{CaseAction, CaseKey, Rule};
#[cfg(feature = "infer")]
pub use span::{SurfaceError, split_surface};
#[cfg(feature = "infer")]
pub use user_dictionary::{
    UserBundleVerification, UserDictionary, UserDictionaryChange, UserDictionaryMeta,
    UserDictionaryReceipt, UserEntry, UserEntryCounts, verify_user_bundle,
};
#[cfg(feature = "user-dictionary-write")]
pub use user_dictionary::{UserDictionaryEditor, UserDictionaryUpdate};

#[cfg(feature = "train")]
pub use builder::{
    Built, Error as BuildError, Paradigm, Spec, build_lexicon_required, validate_normalized_rows,
};
#[cfg(feature = "train")]
pub use lexicon::{Compiled, compile, prune};
#[cfg(feature = "train")]
pub use train::{
    Graded, Pair, Score, Tables as TrainingTables, Thresholds, Trained, Variant, View, build,
    build_for_language, build_for_language_with_entities, build_with_entities, choose, score,
    traffic, tune, write_phrase_source, write_sources,
};

#[cfg(feature = "python")]
mod python;
