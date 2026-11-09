use std::borrow::Cow;
use std::collections::HashMap;
use std::ffi::CStr;
use std::hash::BuildHasherDefault;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

use pyo3::exceptions::{PyDeprecationWarning, PyTypeError};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::sync::{MutexExt, PyOnceLock};
use pyo3::types::{PyList, PySequence, PyString};

use crate::lexicon::table::AddressHasher;
use crate::lexicon::{self, RuntimeLexicon};
use crate::probe::Options;

#[path = "runtime/config.rs"]
mod config;

use super::support::to_py;
use config::RuntimeConfig;

/// One lemma together with the stage that produced it.
#[pyclass(module = "udlex", frozen)]
#[derive(Debug)]
pub struct LemmaInfo {
    lemma: String,
    source: crate::Source,
}

impl LemmaInfo {
    fn from_answer(answer: crate::probe::Answer<'_>) -> Self {
        Self {
            source: answer.source,
            lemma: answer.lemma.into_owned(),
        }
    }
}

#[pymethods]
impl LemmaInfo {
    /// Produced lemma text.
    #[getter]
    fn lemma(&self) -> &str {
        &self.lemma
    }

    /// The typed final stage, cached once per interpreter.
    #[getter]
    fn source(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        static MEMBERS: PyOnceLock<Vec<Py<PyAny>>> = PyOnceLock::new();
        let members = MEMBERS.get_or_try_init(py, || {
            let class = py.import("udlex._provenance")?.getattr("Source")?;
            crate::Source::ALL
                .iter()
                .map(|source| class.call1((source.name(),)).map(Bound::unbind))
                .collect::<PyResult<Vec<_>>>()
        })?;
        members
            .get(usize::from(self.source.code()))
            .map(|member| member.clone_ref(py))
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("unknown source code"))
    }

    fn __repr__(&self) -> String {
        format!(
            "LemmaInfo(lemma={:?}, source=Source.{})",
            self.lemma,
            self.source.member_name()
        )
    }
}

#[cfg(test)]
#[path = "runtime/tests.rs"]
mod tests;

/// Authenticates the deferred entity table before a probe that can reach it.
///
/// Hashing runs detached; failures propagate instead of becoming identity answers.
fn ensure_entities(py: Python<'_>, inner: &RuntimeLexicon) -> PyResult<()> {
    if inner.entities_ready() {
        return Ok(());
    }
    py.detach(|| inner.ensure_entities())
        .map_err(|error| to_py(&error))
}

/// Validates an optional per-token mask.
fn validate_mask<T>(name: &str, mask: Option<&[T]>, tokens: usize) -> PyResult<()> {
    if let Some(mask) = mask
        && mask.len() != tokens
    {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} has {} items, expected {tokens}",
            mask.len()
        )));
    }
    Ok(())
}

/// Whether any effective per-token profile can reach entity vocabulary.
fn batch_needs_entities(
    configured: Options,
    tokens: usize,
    protected: Option<&[bool]>,
    entities: Option<&[bool]>,
) -> bool {
    if tokens == 0 || configured.protected {
        return false;
    }
    if configured.use_entities {
        return protected.is_none_or(|mask| mask.iter().any(|protected| !protected));
    }
    let Some(entities) = entities else {
        // `use_entities_lower` belongs to the explicit entity profile and is
        // cleared by `Options::for_call` on every ordinary row.
        return false;
    };
    protected.map_or_else(
        || entities.contains(&true),
        |protected| {
            entities
                .iter()
                .zip(protected)
                .any(|(entity, protected)| *entity && !protected)
        },
    )
}

/// Validates the shared batch contract and opens deferred entity data once.
fn prepare_batch(
    py: Python<'_>,
    inner: &RuntimeLexicon,
    configured: Options,
    tokens: usize,
    sentence_initial: Option<&[bool]>,
    protected: Option<&[bool]>,
    entities: Option<&[bool]>,
) -> PyResult<()> {
    validate_mask("sentence_initial", sentence_initial, tokens)?;
    validate_mask("protected", protected, tokens)?;
    validate_mask("entities", entities, tokens)?;
    if batch_needs_entities(configured, tokens, protected, entities) {
        ensure_entities(py, inner)?;
    }
    Ok(())
}

/// Resolves one row's sentence boundary and effective lookup profile.
fn batch_call_options(
    configured: Options,
    index: usize,
    sentence_initial: Option<&[bool]>,
    protected: Option<&[bool]>,
    entities: Option<&[bool]>,
) -> (bool, Options) {
    let initial =
        sentence_initial.map_or(index == 0, |mask| mask.get(index).copied().unwrap_or(false));
    let protected = protected
        .and_then(|mask| mask.get(index))
        .copied()
        .unwrap_or(false);
    let entity = entities
        .and_then(|mask| mask.get(index))
        .copied()
        .unwrap_or(false);
    (initial, configured.for_call(protected, entity))
}

/// Per-row masks of one batch: sentence boundaries, protection, entity profile.
type Masks<'a> = (Option<&'a [bool]>, Option<&'a [bool]>, Option<&'a [bool]>);

/// A half-open token interval and its optional inter-token text.
type TokenSpan<'a> = (usize, Option<usize>, Option<&'a [PyBackedStr]>);

/// Warn once per deprecated name to keep warnings out of per-token loops.
fn deprecated(py: Python<'_>, warned: &AtomicBool, message: &CStr) -> PyResult<()> {
    if warned.load(Ordering::Relaxed) || warned.swap(true, Ordering::Relaxed) {
        return Ok(());
    }
    PyErr::warn(py, &py.get_type::<PyDeprecationWarning>(), message, 1)
}

/// Extracts a token list, refusing the one mistake extraction would garble.
fn tokens_of(tokens: &Bound<'_, PyAny>, method: &str) -> PyResult<Vec<PyBackedStr>> {
    if tokens.is_instance_of::<PyString>() {
        return Err(PyTypeError::new_err(format!(
            "{method}() takes a list of tokens, not a str: split the text into \
             tokens first, or call lemma() for one word"
        )));
    }
    // A list is walked by index, into one allocation: collecting a `Result`
    // iterator drops the size hint and reallocates its way up.
    if let Ok(list) = tokens.cast::<PyList>() {
        let mut out = Vec::with_capacity(list.len());
        for item in list.iter() {
            out.push(item.extract()?);
        }
        return Ok(out);
    }
    tokens.extract()
}

/// Python strings of the lemmas a table answered, keyed by their address in
/// the lexicon's storage, which is immutable while the object lives. A lemma
/// that repeats costs one hash probe and an incref instead of a UTF-8 decode.
#[derive(Default)]
struct LemmaStrings(HashMap<usize, Py<PyString>, BuildHasherDefault<AddressHasher>>);

/// Past this many distinct lemmas new ones are decoded without being kept.
const LEMMA_STRINGS_LIMIT: usize = 1 << 16;

impl LemmaStrings {
    fn object<'py>(&mut self, py: Python<'py>, lemma: &str) -> Bound<'py, PyString> {
        let key = lemma.as_ptr() as usize;
        if let Some(object) = self.0.get(&key) {
            return object.bind(py).clone();
        }
        let object = PyString::new(py, lemma);
        if self.0.len() < LEMMA_STRINGS_LIMIT {
            self.0.insert(key, object.clone().unbind());
        }
        object
    }
}

impl std::fmt::Debug for LemmaStrings {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "LemmaStrings({})", self.0.len())
    }
}

/// The cache, locked on first need so unchanged answers never take it.
type Strings<'a> = Option<MutexGuard<'a, LemmaStrings>>;

/// Whether answers share one cached string per lemma. Under a GIL the cache
/// is free; free-threaded, its lock and the shared refcounts serialise batches.
fn share_lemma_strings(py: Python<'_>) -> bool {
    static SHARE: PyOnceLock<bool> = PyOnceLock::new();
    *SHARE.get_or_init(py, || {
        // `sys._is_gil_enabled` is missing before 3.13, which has no t-build.
        py.import("sys")
            .and_then(|sys| sys.getattr("_is_gil_enabled")?.call0()?.extract())
            .unwrap_or(true)
    })
}

/// Reconstructs a validated token interval, preserving explicit separators.
fn joined_span(parts: &[PyBackedStr], separators: Option<&[PyBackedStr]>) -> PyResult<String> {
    let gaps = parts.len().saturating_sub(1);
    validate_mask("separators", separators, gaps)?;
    let width: usize = parts
        .iter()
        .map(|part| part.len())
        .sum::<usize>()
        .saturating_add(separators.map_or(gaps, |separators| {
            separators.iter().map(|separator| separator.len()).sum()
        }));
    let mut surface = String::with_capacity(width);
    let mut gap_texts = separators.map(<[PyBackedStr]>::iter);
    for (index, part) in parts.iter().enumerate() {
        if index > 0 {
            match gap_texts.as_mut().and_then(Iterator::next) {
                Some(separator) => surface.push_str(separator),
                None => surface.push(' '),
            }
        }
        surface.push_str(part);
    }
    if surface.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "span surface is empty",
        ));
    }
    Ok(surface)
}

/// A compiled lexicon, loaded once and queried many times.
#[pyclass(module = "udlex", frozen)]
#[derive(Debug)]
pub struct Lexicon {
    inner: RuntimeLexicon,
    options: Options,
    /// Lowercase entity retry of the span path, which alone may follow the
    /// generation's measured policy.
    span_lower: bool,
    lemma_strings: Mutex<LemmaStrings>,
}

impl Lexicon {
    /// The Python string for one answer: the token's own object when the
    /// lemma equals it, a shared string for a table entry, a fresh one else.
    #[expect(clippy::ptr_arg, reason = "borrowed or owned decides whether the address is a key")]
    fn lemma_object<'a, 'py>(
        &'a self,
        py: Python<'py>,
        strings: &mut Strings<'a>,
        token: &Bound<'py, PyString>,
        surface: &str,
        lemma: &Cow<'_, str>,
    ) -> Bound<'py, PyString> {
        if lemma.as_bytes() == surface.as_bytes() {
            return token.clone();
        }
        // A borrowed answer is a table entry; a slice of the token is not
        // produced today and would be the one unstable address, so it is
        // decoded rather than keyed.
        let Cow::Borrowed(text) = lemma else {
            return PyString::new(py, lemma);
        };
        if !share_lemma_strings(py)
            || surface.as_bytes().as_ptr_range().contains(&text.as_ptr())
        {
            return PyString::new(py, text);
        }
        strings
            .get_or_insert_with(|| {
                self.lemma_strings
                    .lock_py_attached(py)
                    .unwrap_or_else(PoisonError::into_inner)
            })
            .object(py, text)
    }

    fn answer_object<'py>(
        &self,
        py: Python<'py>,
        object: &Bound<'py, PyString>,
        surface: &str,
        answer: crate::probe::Answer<'_>,
        info: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        if info {
            return Ok(Bound::new(py, LemmaInfo::from_answer(answer))?.into_any());
        }
        Ok(self
            .lemma_object(py, &mut None, object, surface, &answer.lemma)
            .into_any())
    }

    fn token_options(&self, py: Python<'_>, protected: bool, entity: bool) -> PyResult<Options> {
        let options = self.options.for_call(protected, entity);
        if !options.protected && (options.use_entities || options.use_entities_lower) {
            ensure_entities(py, &self.inner)?;
        }
        Ok(options)
    }

    fn span_options(&self, py: Python<'_>, protected: bool) -> PyResult<Options> {
        let mut options = self.options;
        options.use_entities_lower = self.span_lower;
        options.protected |= protected;
        if !options.protected {
            ensure_entities(py, &self.inner)?;
        }
        Ok(options)
    }

    fn probe_text_span<'a>(
        &'a self,
        py: Python<'_>,
        text: &'a str,
        sentence_initial: bool,
        protected: bool,
    ) -> PyResult<crate::probe::Answer<'a>> {
        // Validate caller input before authenticating deferred entity data.
        let (tokens, gaps) = crate::split_surface(text)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        let options = self.span_options(py, protected)?;
        let separators = (!gaps.is_empty()).then_some(gaps.as_slice());
        Ok(self
            .inner
            .lemma_span_with_info(text, &tokens, separators, sentence_initial, options))
    }

    /// Extract only the interval so work scales with the mention's length.
    fn token_span<'py>(
        &self,
        py: Python<'py>,
        tokens: &Bound<'py, PyAny>,
        (start, end, separators): TokenSpan<'_>,
        sentence_initial: bool,
        protected: bool,
        info: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        if tokens.is_instance_of::<PyString>() {
            return Err(PyTypeError::new_err(
                "lemma_span_with_info() takes a list of tokens; use lemma_span(text)",
            ));
        }
        let sequence = tokens.cast::<PySequence>()?;
        let count = sequence.len()?;
        let end = end.unwrap_or(count);
        if start >= end || end > count {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "span [{start}, {end}) is empty or out of bounds for {count} tokens"
            )));
        }
        let tokens = (start..end)
            .map(|index| sequence.get_item(index)?.extract::<PyBackedStr>())
            .collect::<PyResult<Vec<_>>>()?;
        let surface = joined_span(&tokens, separators)?;
        let options = self.span_options(py, protected)?;
        let answer = if tokens.len() == 1 {
            self.inner
                .lemma_span_with_info(&surface, &[], None, sentence_initial, options)
        } else {
            let parts: Vec<&str> = tokens.iter().map(|token| &**token).collect();
            let gaps: Option<Vec<&str>> =
                separators.map(|gaps| gaps.iter().map(|gap| &**gap).collect());
            self.inner.lemma_span_with_info(
                &surface,
                &parts,
                gaps.as_deref(),
                sentence_initial,
                options,
            )
        };
        Ok(if info {
            Bound::new(py, LemmaInfo::from_answer(answer))?.into_any()
        } else {
            PyString::new(py, &answer.lemma).into_any()
        })
    }

    /// Probes detached, borrowing table hits until Python strings are needed.
    fn lemmatize_plain<'py>(
        &self,
        py: Python<'py>,
        tokens: &[PyBackedStr],
        (sentence_initial, protected, entities): Masks<'_>,
    ) -> PyResult<Bound<'py, PyList>> {
        prepare_batch(
            py,
            &self.inner,
            self.options,
            tokens.len(),
            sentence_initial,
            protected,
            entities,
        )?;
        // Resolve the casing convention once per batch.
        let stamped = self.inner.base().surface_case();
        let lemmas: Vec<Cow<'_, str>> = py.detach(|| {
            tokens
                .iter()
                .enumerate()
                .map(|(index, token)| {
                    let (initial, options) =
                        batch_call_options(self.options, index, sentence_initial, protected, entities);
                    if stamped {
                        self.inner.lemma_with_recased(token, initial, options)
                    } else {
                        self.inner.lemma_with_plain(token, initial, options)
                    }
                })
                .collect()
        });
        let mut strings = None;
        PyList::new(
            py,
            lemmas.iter().zip(tokens).map(|(lemma, token)| {
                self.lemma_object(py, &mut strings, token.as_py_str().bind(py), token, lemma)
            }),
        )
    }

    /// The same batch keeping one `LemmaInfo` per row; each token is probed once.
    fn lemmatize_info_rows(
        &self,
        py: Python<'_>,
        tokens: &[PyBackedStr],
        (sentence_initial, protected, entities): Masks<'_>,
    ) -> PyResult<Vec<LemmaInfo>> {
        prepare_batch(
            py,
            &self.inner,
            self.options,
            tokens.len(),
            sentence_initial,
            protected,
            entities,
        )?;
        let infos: Vec<LemmaInfo> = py.detach(|| {
            tokens
                .iter()
                .enumerate()
                .map(|(index, token)| {
                    let (initial, options) =
                        batch_call_options(self.options, index, sentence_initial, protected, entities);
                    LemmaInfo::from_answer(self.inner.lemma_with_info(token, initial, options))
                })
                .collect()
        });
        Ok(infos)
    }

    fn lemmatize_info<'py>(
        &self,
        py: Python<'py>,
        tokens: &[PyBackedStr],
        masks: Masks<'_>,
    ) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.lemmatize_info_rows(py, tokens, masks)?)
    }

    /// Context keys stop at the sentence boundaries supplied by the mask.
    #[cold]
    #[inline(never)]
    fn lemmatize_context<'py>(
        &self,
        py: Python<'py>,
        tokens: &[PyBackedStr],
        (sentence_initial, protected, entities): Masks<'_>,
        upos: Option<&[PyBackedStr]>,
        neighbours: bool,
        info: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        validate_mask("upos", upos, tokens.len())?;
        prepare_batch(
            py,
            &self.inner,
            self.options,
            tokens.len(),
            sentence_initial,
            protected,
            entities,
        )?;
        let answers: Vec<crate::probe::Answer<'_>> = py.detach(|| {
            let opens = |index| batch_call_options(self.options, index, sentence_initial, None, None).0;
            tokens
                .iter()
                .enumerate()
                .map(|(index, token)| {
                    let (initial, options) =
                        batch_call_options(self.options, index, sentence_initial, protected, entities);
                    let (previous, next) = crate::format::neighbours(tokens, index, opens);
                    let tag = upos.and_then(|tags| tags.get(index)).map(|tag| &**tag);
                    self.inner.lemma_in_context(
                        token,
                        initial,
                        options,
                        tag,
                        neighbours.then_some(previous),
                        neighbours.then_some(next),
                    )
                })
                .collect()
        });
        if info {
            return PyList::new(py, answers.into_iter().map(LemmaInfo::from_answer));
        }
        let mut strings = None;
        PyList::new(
            py,
            answers.iter().zip(tokens).map(|(answer, token)| {
                self.lemma_object(
                    py,
                    &mut strings,
                    token.as_py_str().bind(py),
                    token,
                    &answer.lemma,
                )
            }),
        )
    }
}

#[pymethods]
impl Lexicon {
    /// Opens a compiled lexicon directory.
    ///
    /// Args:
    ///     `path`: Directory holding the compiled tables.
    ///     `use_external`: Consult the merged offline dictionary/analyser table.
    ///     `use_rules`: Guess from suffixes for unknown words. For marked
    ///         entities prefer the isolated `entity=True` call profile.
    #[new]
    #[pyo3(signature = (
        path, *,
        use_external = true,
        use_rules = true,
        use_exact = true,
        use_lower = true,
        use_exact_lower = true,
        use_external_lower = true,
        use_entities = false,
        use_entities_lower = None,
        user_bundle = None,
        use_user = true,
        protected = false,
        span_case = None,
        span_multiword = None,
        phrase_patterns = None,
        token_case = None
    ))]
    #[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
    fn new(
        py: Python<'_>,
        path: &str,
        use_external: bool,
        use_rules: bool,
        use_exact: bool,
        use_lower: bool,
        use_exact_lower: bool,
        use_external_lower: bool,
        use_entities: bool,
        use_entities_lower: Option<bool>,
        user_bundle: Option<&str>,
        use_user: bool,
        protected: bool,
        span_case: Option<&str>,
        span_multiword: Option<&str>,
        phrase_patterns: Option<Vec<(String, String)>>,
        token_case: Option<&str>,
    ) -> PyResult<Self> {
        let config = RuntimeConfig::new(
            use_external,
            use_rules,
            use_exact,
            use_lower,
            use_exact_lower,
            use_external_lower,
            use_entities,
            use_entities_lower,
            use_user,
            protected,
            span_case,
            span_multiword,
            phrase_patterns,
            token_case,
        );
        // Authenticate and validate the vocabulary with the interpreter released.
        let base = py
            .detach(|| lexicon::Lexicon::open(path))
            .map_err(|error| to_py(&error))?;
        config.configure(py, base, user_bundle)
    }

    /// Returns the lemma of one token, or its `LemmaInfo` with `info=True`.
    ///
    /// An unchanged token retains its Python object; other answers are copied.
    /// Warm scalar lookup keeps the interpreter. Deferred entity
    /// authentication and batch calls release it.
    #[pyo3(signature = (
        token, *, sentence_initial = false, protected = false, entity = false, info = false
    ))]
    #[allow(clippy::fn_params_excessive_bools)]
    fn lemma<'py>(
        &self,
        py: Python<'py>,
        token: &Bound<'py, PyString>,
        sentence_initial: bool,
        protected: bool,
        entity: bool,
        info: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let text = token.to_str()?;
        let options = self.token_options(py, protected, entity)?;
        let answer = self.inner.lemma_with_info(text, sentence_initial, options);
        self.answer_object(py, token, text, answer, info)
    }

    /// Deprecated spelling of `lemma(..., info=True)`.
    #[pyo3(signature = (token, *, sentence_initial = false, protected = false, entity = false))]
    fn lemma_with_info<'py>(
        &self,
        py: Python<'py>,
        token: &Bound<'py, PyString>,
        sentence_initial: bool,
        protected: bool,
        entity: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        static WARNED: AtomicBool = AtomicBool::new(false);
        deprecated(
            py,
            &WARNED,
            c"lemma_with_info() is deprecated; use lemma(..., info=True)",
        )?;
        self.lemma(py, token, sentence_initial, protected, entity, true)
    }

    /// Returns the canonical text of one entity mention.
    ///
    /// `mention` is a string or a token list narrowed by `[start, end)`.
    /// Strings retain their whitespace; token lists use `separators` when
    /// supplied, otherwise one space between tokens.
    ///
    /// Entity tables take precedence, followed by token lookup for a single
    /// word or the measured multiword convention. `info=True` returns `LemmaInfo`.
    #[pyo3(signature = (
        mention, start = 0, end = None, *,
        separators = None, sentence_initial = false, protected = false, info = false
    ))]
    #[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
    fn lemma_span<'py>(
        &self,
        py: Python<'py>,
        mention: &Bound<'py, PyAny>,
        start: usize,
        end: Option<usize>,
        separators: Option<Vec<PyBackedStr>>,
        sentence_initial: bool,
        protected: bool,
        info: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        if let Ok(text) = mention.cast::<PyString>() {
            if start != 0 || end.is_some() || separators.is_some() {
                return Err(PyTypeError::new_err(
                    "start, end and separators apply to a token list; a str mention \
                     is split on its own whitespace",
                ));
            }
            let surface = text.to_str()?;
            let answer = self.probe_text_span(py, surface, sentence_initial, protected)?;
            return self.answer_object(py, text, surface, answer, info);
        }
        let span = (start, end, separators.as_deref());
        self.token_span(py, mention, span, sentence_initial, protected, info)
    }

    /// Deprecated spelling of `lemma_span(tokens, ..., info=True)`.
    #[pyo3(signature = (
        tokens, start = 0, end = None, *,
        separators = None, sentence_initial = false, protected = false
    ))]
    #[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
    fn lemma_span_with_info<'py>(
        &self,
        py: Python<'py>,
        tokens: &Bound<'py, PyAny>,
        start: usize,
        end: Option<usize>,
        separators: Option<Vec<PyBackedStr>>,
        sentence_initial: bool,
        protected: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        static WARNED: AtomicBool = AtomicBool::new(false);
        deprecated(
            py,
            &WARNED,
            c"lemma_span_with_info() is deprecated; use lemma_span(..., info=True)",
        )?;
        let span = (start, end, separators.as_deref());
        self.token_span(py, tokens, span, sentence_initial, protected, true)
    }

    /// Deprecated spelling of `lemma_span(text)`.
    #[pyo3(signature = (text, *, sentence_initial = false, protected = false))]
    fn lemma_span_text<'py>(
        &self,
        py: Python<'py>,
        text: &Bound<'py, PyString>,
        sentence_initial: bool,
        protected: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        static WARNED: AtomicBool = AtomicBool::new(false);
        deprecated(
            py,
            &WARNED,
            c"lemma_span_text() is deprecated; use lemma_span(text)",
        )?;
        let surface = text.to_str()?;
        let answer = self.probe_text_span(py, surface, sentence_initial, protected)?;
        self.answer_object(py, text, surface, answer, false)
    }

    /// Deprecated spelling of `lemma_span(text, info=True)`.
    #[pyo3(signature = (text, *, sentence_initial = false, protected = false))]
    fn lemma_span_text_with_info<'py>(
        &self,
        py: Python<'py>,
        text: &Bound<'py, PyString>,
        sentence_initial: bool,
        protected: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        static WARNED: AtomicBool = AtomicBool::new(false);
        deprecated(
            py,
            &WARNED,
            c"lemma_span_text_with_info() is deprecated; use lemma_span(text, info=True)",
        )?;
        let surface = text.to_str()?;
        let answer = self.probe_text_span(py, surface, sentence_initial, protected)?;
        self.answer_object(py, text, surface, answer, true)
    }

    /// Whether a lookup table answers this form, suffix rules aside.
    #[pyo3(signature = (token, *, sentence_initial = false, protected = false, entity = false))]
    fn knows(
        &self,
        py: Python<'_>,
        token: &str,
        sentence_initial: bool,
        protected: bool,
        entity: bool,
    ) -> PyResult<bool> {
        let options = self.token_options(py, protected, entity)?;
        Ok(self.inner.knows_with(token, sentence_initial, options))
    }

    /// Lemmatises a sentence, treating the first token as sentence-initial.
    ///
    /// Lookup runs with the interpreter released. Tokens are borrowed, and
    /// unchanged lemmas reuse their Python strings. `info=True` returns one
    /// `LemmaInfo` per token.
    ///
    /// `upos` supplies one tag per token; `context=True` uses neighbouring
    /// words to resolve ambiguous exact answers.
    #[pyo3(signature = (
        tokens, *, sentence_initial = None, protected = None, entities = None, info = false,
        upos = None, context = false
    ))]
    #[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
    fn lemmatize<'py>(
        &self,
        py: Python<'py>,
        tokens: &Bound<'py, PyAny>,
        sentence_initial: Option<Vec<bool>>,
        protected: Option<Vec<bool>>,
        entities: Option<Vec<bool>>,
        info: bool,
        upos: Option<Vec<PyBackedStr>>,
        context: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let tokens = tokens_of(tokens, "lemmatize")?;
        let masks = (
            sentence_initial.as_deref(),
            protected.as_deref(),
            entities.as_deref(),
        );
        if upos.is_some() || context {
            return self.lemmatize_context(py, &tokens, masks, upos.as_deref(), context, info);
        }
        if info {
            self.lemmatize_info(py, &tokens, masks)
        } else {
            self.lemmatize_plain(py, &tokens, masks)
        }
    }

    /// Lemmatises many sentences in one call, each opening its own sentence.
    ///
    /// Releases the interpreter once for the batch. Per-token masks are
    /// available through [`Self::lemmatize`].
    #[pyo3(signature = (sentences, *, info = false))]
    fn lemmatize_many<'py>(
        &self,
        py: Python<'py>,
        sentences: &Bound<'py, PyAny>,
        info: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        if sentences.is_instance_of::<PyString>() {
            return Err(PyTypeError::new_err(
                "lemmatize_many() takes a list of token lists, not a str",
            ));
        }
        let mut lengths = Vec::new();
        let mut flat = Vec::new();
        for sentence in sentences.try_iter()? {
            let tokens = tokens_of(&sentence?, "lemmatize_many")?;
            lengths.push(tokens.len());
            flat.extend(tokens);
        }
        // Reuse the single-sentence loop with explicit sentence boundaries.
        let starts: Vec<bool> = lengths
            .iter()
            .flat_map(|&length| (0..length).map(|index| index == 0))
            .collect();
        let masks = (Some(starts.as_slice()), None, None);
        let rows = if info {
            let mut infos = self.lemmatize_info_rows(py, &flat, masks)?.into_iter();
            lengths
                .iter()
                .map(|&length| PyList::new(py, infos.by_ref().take(length)))
                .collect::<PyResult<Vec<_>>>()?
        } else {
            let lemmas = self.lemmatize_plain(py, &flat, masks)?;
            let mut offset = 0_usize;
            lengths
                .iter()
                .map(|&length| {
                    // Lengths sum to the batch length, which fits in usize.
                    let end = offset.saturating_add(length);
                    let row = lemmas.get_slice(offset, end);
                    offset = end;
                    row
                })
                .collect()
        };
        PyList::new(py, rows)
    }

    /// Deprecated spelling of `lemmatize(..., info=True)`.
    #[pyo3(signature = (tokens, *, sentence_initial = None, protected = None, entities = None))]
    #[allow(clippy::needless_pass_by_value)]
    fn lemmatize_with_info<'py>(
        &self,
        py: Python<'py>,
        tokens: &Bound<'py, PyAny>,
        sentence_initial: Option<Vec<bool>>,
        protected: Option<Vec<bool>>,
        entities: Option<Vec<bool>>,
    ) -> PyResult<Bound<'py, PyList>> {
        static WARNED: AtomicBool = AtomicBool::new(false);
        deprecated(
            py,
            &WARNED,
            c"lemmatize_with_info() is deprecated; use lemmatize(..., info=True)",
        )?;
        let tokens = tokens_of(tokens, "lemmatize_with_info")?;
        let masks = (
            sentence_initial.as_deref(),
            protected.as_deref(),
            entities.as_deref(),
        );
        self.lemmatize_info(py, &tokens, masks)
    }

    /// Which parts of the learned lexicon this bundle ships.
    ///
    /// ``full`` has lookup tables and suffix rules, ``lookup`` has tables only,
    /// and ``identity`` has no common-word stages: ordinary tokens come back
    /// unchanged because that won on gold data. It may retain an isolated
    /// entity table for calls made with ``entity=True``.
    #[getter]
    fn variant(&self) -> &str {
        &self.inner.meta().variant
    }

    /// Which independent split chose that variant: ``"dev"``,
    /// form-disjoint ``"train-cv"``, or ``None``.
    #[getter]
    fn selected_on(&self) -> Option<&str> {
        self.inner.meta().selected_on.as_deref()
    }

    /// The lexicon identifier, normally a Universal Dependencies treebank code.
    #[getter]
    fn name(&self) -> &str {
        &self.inner.meta().name
    }

    /// Human-readable language name.
    #[getter]
    fn language(&self) -> &str {
        &self.inner.meta().language
    }

    /// ISO 639-3 code of the language.
    #[getter]
    fn iso639_3(&self) -> &str {
        &self.inner.meta().iso639_3
    }

    /// What the lexicon was learned from.
    #[getter]
    fn source(&self) -> &str {
        &self.inner.meta().source
    }

    /// How many annotated tokens went into it.
    #[getter]
    const fn training_tokens(&self) -> u64 {
        self.inner.meta().training_tokens
    }

    /// Immutable base generation resolved at open, if this is a generation root.
    #[getter]
    fn generation(&self) -> Option<&str> {
        self.inner.meta().generation.as_deref()
    }

    /// On-disk layout version declared by the base generation.
    #[getter]
    const fn format_version(&self) -> u32 {
        self.inner.meta().format_version
    }

    /// Composite accuracy, unknown for a non-empty user layer.
    #[getter]
    const fn accuracy(&self) -> Option<f64> {
        self.inner.composite_accuracy()
    }

    /// Composite baseline, unknown for a non-empty user layer.
    #[getter]
    const fn baseline(&self) -> Option<f64> {
        self.inner.composite_baseline()
    }

    /// Composite usefulness, unknown for a non-empty user layer.
    #[getter]
    fn is_useful(&self) -> Option<bool> {
        self.inner.composite_is_useful()
    }

    /// Held-out accuracy of the unchanged base generation.
    #[getter]
    const fn base_accuracy(&self) -> Option<f64> {
        self.inner.meta().accuracy
    }

    /// Copying baseline measured for the unchanged base generation.
    #[getter]
    const fn base_baseline(&self) -> Option<f64> {
        self.inner.meta().baseline
    }

    /// Usefulness verdict of the unchanged base generation.
    #[getter]
    fn base_is_useful(&self) -> Option<bool> {
        self.inner.base().is_useful()
    }

    /// Whether the loaded ISO has at least one active manual override.
    #[getter]
    const fn has_user_dictionary(&self) -> bool {
        self.inner.has_user_dictionary()
    }

    /// Active manual override count (zero for a base-only object).
    #[getter]
    const fn user_entries(&self) -> u64 {
        self.inner.user_entries()
    }

    /// Resolved immutable user generation, if a language layer was present.
    #[getter]
    fn user_generation(&self) -> Option<&str> {
        self.inner.user_generation()
    }

    fn __repr__(&self) -> String {
        let meta = self.inner.meta();
        format!(
            "Lexicon(name='{}', language='{}', training_tokens={})",
            meta.name, meta.language, meta.training_tokens
        )
    }
}
/// Opens the lexicon a bundle holds for one language.
///
/// Args:
///     `bundle`: A bundle directory.
///     `language`: An ISO 639-3 code such as ``"rus"``, or a treebank name.
///     `use_external`: Consult the merged offline dictionary/analyser table.
///     `use_rules`: Guess from suffixes for unknown words. For marked entities
///         prefer the isolated per-call entity profile.
#[pyfunction]
#[pyo3(signature = (
    bundle, language, *,
    use_external = true,
    use_rules = true,
    use_exact = true,
    use_lower = true,
    use_exact_lower = true,
    use_external_lower = true,
    use_entities = false,
    use_entities_lower = None,
    user_bundle = None,
    use_user = true,
    protected = false,
    span_case = None,
    span_multiword = None,
    phrase_patterns = None,
    token_case = None
))]
#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
pub(super) fn load(
    py: Python<'_>,
    bundle: &str,
    language: &str,
    use_external: bool,
    use_rules: bool,
    use_exact: bool,
    use_lower: bool,
    use_exact_lower: bool,
    use_external_lower: bool,
    use_entities: bool,
    use_entities_lower: Option<bool>,
    user_bundle: Option<&str>,
    use_user: bool,
    protected: bool,
    span_case: Option<&str>,
    span_multiword: Option<&str>,
    phrase_patterns: Option<Vec<(String, String)>>,
    token_case: Option<&str>,
) -> PyResult<Lexicon> {
    let config = RuntimeConfig::new(
        use_external,
        use_rules,
        use_exact,
        use_lower,
        use_exact_lower,
        use_external_lower,
        use_entities,
        use_entities_lower,
        use_user,
        protected,
        span_case,
        span_multiword,
        phrase_patterns,
        token_case,
    );
    let base = py
        .detach(|| crate::lexicon::load(bundle, language))
        .map_err(|error| to_py(&error))?;
    config.configure(py, base, user_bundle)
}
