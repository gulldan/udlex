//! Python bindings, re-exported by the `udlex` package.
//!
//! Every call that computes or touches the disk detaches from the interpreter
//! ([`Python::detach`]); single-token queries stay attached, since parking the
//! thread state costs more than the probe itself.

use pyo3::prelude::*;

/// The compiled half of `udlex`, imported as `udlex._udlex`.
///
/// `gil_used = false` is stated as this module's free-threading contract.
// A leading underscore is the Python-private name, which `#[pyfunction]` reuses.
#[allow(clippy::used_underscore_items)]
#[pymodule(gil_used = false)]
mod _udlex {
    use pyo3::prelude::*;

    // A proc-macro input cannot declare out-of-line `mod file;`, hence `include!`.
    /// Python views over bundle metadata and compilation operations.
    mod catalog {
        include!("python/_udlex/catalog.rs");
    }
    /// Loaded-lexicon queries and their Python-facing provenance records.
    mod runtime {
        include!("python/_udlex/runtime.rs");
    }
    /// Shared exception conversion and lookup-option construction.
    mod support {
        include!("python/_udlex/support.rs");
    }
    /// Transactional user-dictionary CRUD and audit helpers.
    mod user_dictionary {
        include!("python/_udlex/user_dictionary.rs");
    }
    /// Python training tables, outcomes, and build/scoring entry points.
    mod training {
        include!("python/_udlex/training.rs");
    }

    #[pymodule_export]
    use self::catalog::{Compiled, Language, compile_bundle, languages, prune_bundle};
    #[pymodule_export]
    use self::runtime::{LemmaInfo, Lexicon, load};
    #[pymodule_export]
    use self::support::{
        _locale_lower, _normalize_nfc, _phrase_span_key, _script_compatibility,
        _source_table_batch, _training_annotation_mask, _validate_normalized_rows,
    };
    #[pymodule_export]
    use self::training::{Outcome, Tables, build, build_lexicon, choose, score, tune};
    #[pymodule_export]
    use self::user_dictionary::{
        _user_dictionary_entries, _user_dictionary_history, _user_dictionary_import,
        _user_dictionary_init, _user_dictionary_remove, _user_dictionary_rollback,
        _user_dictionary_set, _user_dictionary_verify,
    };

    /// Whether this build reads a bundle declaring `format_version`.
    ///
    /// Python asks this rather than re-implementing the range check.
    #[pyfunction]
    const fn format_supported(format_version: u32) -> bool {
        crate::format::FormatRange::READ.accepts(format_version)
    }

    /// What a caller must do about a format this build does not read.
    #[pyfunction]
    const fn format_remedy(format_version: u32) -> &'static str {
        crate::format::FormatRange::READ.remedy(format_version)
    }

    #[pymodule_init]
    fn init(module: &Bound<'_, PyModule>) -> PyResult<()> {
        // Bounds are exported only to name them in messages; `format_supported` decides.
        module.add("FORMAT_VERSION", crate::format::FORMAT_VERSION)?;
        module.add("MIN_FORMAT_VERSION", crate::format::MIN_FORMAT_VERSION)?;
        module.add("FORMAT_RANGE", crate::format::FormatRange::READ.to_string())?;
        module.add("MIN_CORE_RELEASE", crate::format::MIN_CORE_RELEASE)?;
        module.add("ENTITY_ENCODING", crate::format::ENTITY_ENCODING)?;
        module.add("NORMALIZATION", crate::format::NORMALIZATION)?;
        module.add(
            "USER_DICTIONARY_SCHEMA",
            crate::format::USER_DICTIONARY_SCHEMA,
        )?;
        module.add("PHRASE_CODEC", crate::format::PHRASE_CODEC)?;
        // The offline compiler must build the same phrase keys as the runtime.
        module.add(
            "PHRASE_SEPARATOR",
            crate::format::PHRASE_SEPARATOR.to_string(),
        )?;
        module.add("PHRASE_TAIL", crate::format::PHRASE_TAIL)?;
        module.add(
            "SOURCES",
            pyo3::types::PyTuple::new(module.py(), crate::format::SOURCES)?,
        )?;
        module.add("__version__", env!("CARGO_PKG_VERSION"))?;
        Ok(())
    }
}
