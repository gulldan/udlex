//! A base lexicon plus one independent manual override snapshot.

use std::borrow::Cow;

use super::{Error, Lexicon, Meta, Result};
use crate::probe::{self, Options};
use crate::{UserDictionary, UserDictionaryMeta};

/// Immutable composition of a trained base lexicon and a manual user layer.
pub struct LayeredLexicon {
    base: Lexicon,
    user: UserDictionary,
}

impl std::fmt::Debug for LayeredLexicon {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LayeredLexicon")
            .field("base", &self.base)
            .field("user", &self.user)
            .finish()
    }
}

impl LayeredLexicon {
    /// Combine two already authenticated snapshots of the same ISO language.
    ///
    /// # Errors
    ///
    /// Returns an error if the user layer's ISO differs from the base
    /// metadata. Neither object is modified.
    pub fn new(base: Lexicon, user: UserDictionary) -> Result<Self> {
        if base.meta().iso639_3 != user.iso639_3() {
            return Err(Error::Invalid {
                path: std::path::PathBuf::from("<layered-lexicon>"),
                message: format!(
                    "base ISO {:?} does not match user dictionary ISO {:?}",
                    base.meta().iso639_3,
                    user.iso639_3()
                ),
            });
        }
        Ok(Self { base, user })
    }

    /// Trained base snapshot.
    #[must_use]
    pub const fn base(&self) -> &Lexicon {
        &self.base
    }

    /// Manual override snapshot.
    #[must_use]
    pub const fn user_dictionary(&self) -> &UserDictionary {
        &self.user
    }

    /// Base metadata; `variant`, `source`, and training fields describe only
    /// the trained artifact.
    #[must_use]
    pub const fn base_meta(&self) -> &Meta {
        self.base.meta()
    }

    /// User-generation metadata.
    #[must_use]
    pub const fn user_meta(&self) -> &UserDictionaryMeta {
        self.user.meta()
    }

    /// Composite quality is unknown without a separate gold evaluation.
    /// An authenticated empty generation cannot change an answer and therefore
    /// retains the base verdict.
    #[must_use]
    pub fn is_useful(&self) -> Option<bool> {
        if self.user.is_empty() {
            self.base.is_useful()
        } else {
            None
        }
    }

    /// Base-only quality verdict retained under an explicit name.
    #[must_use]
    pub fn base_is_useful(&self) -> Option<bool> {
        self.base.is_useful()
    }

    /// Authenticate the base entity table now.
    ///
    /// # Errors
    ///
    /// Returns the base entity-table integrity or codec error.
    pub fn ensure_entities(&self) -> Result<()> {
        self.base.ensure_entities()
    }

    /// Whether an entity/span query needs no deferred base-table opening.
    #[must_use]
    pub fn entities_ready(&self) -> bool {
        self.base.entities_ready()
    }

    /// Effective span conventions inherited from the base generation.
    #[must_use]
    pub const fn span_conventions(&self) -> probe::SpanConventions {
        self.base.span_conventions()
    }

    /// Override span conventions on this detached layered object.
    ///
    /// The user dictionary has no phrase namespace of its own, so the
    /// conventions remain a property of the base lexicon.
    pub const fn set_span_conventions(&mut self, conventions: probe::SpanConventions) {
        self.base.set_span_conventions(conventions);
    }

    /// Lemmatize one token under the default layered profile.
    #[must_use]
    pub fn lemma<'a>(&'a self, token: &'a str, sentence_initial: bool) -> Cow<'a, str> {
        self.lemma_with(token, sentence_initial, Options::default())
    }

    /// Lemmatize one token with explicit probe options.
    #[must_use]
    pub fn lemma_with<'a>(
        &'a self,
        token: &'a str,
        sentence_initial: bool,
        options: Options,
    ) -> Cow<'a, str> {
        self.lemma_with_info(token, sentence_initial, options).lemma
    }

    /// Lemmatize one token and retain exact provenance.
    #[must_use]
    pub fn lemma_with_info<'a>(
        &'a self,
        token: &'a str,
        sentence_initial: bool,
        options: Options,
    ) -> probe::Answer<'a> {
        self.base.prime_entities(options);
        let answer =
            probe::lemma_with_user(&self.base, &self.user, token, sentence_initial, options);
        if self.base.surface_case() {
            return self.base.recased(answer, token, sentence_initial);
        }
        answer
    }

    /// See [`Lexicon::lemma_in_context`]; a user-layer answer is never overridden.
    #[must_use]
    pub fn lemma_in_context<'a>(
        &'a self,
        token: &'a str,
        sentence_initial: bool,
        options: Options,
        upos: Option<&str>,
        previous: Option<&str>,
        next: Option<&str>,
    ) -> probe::Answer<'a> {
        let answer = self.lemma_with_info(token, sentence_initial, options);
        probe::in_context(&self.base, answer, token, upos, previous, next)
    }

    /// Canonicalize an already-delimited span and retain provenance.
    #[must_use]
    pub fn lemma_span_with_info<'a>(
        &'a self,
        span: &'a str,
        tokens: &[&str],
        separators: Option<&[&str]>,
        sentence_initial: bool,
        options: Options,
    ) -> probe::Answer<'a> {
        if !options.protected {
            self.base.prime_entities(Options {
                use_entities: true,
                ..options
            });
        }
        probe::span_lemma_with_user(
            &self.base,
            &self.user,
            span,
            tokens,
            separators,
            sentence_initial,
            options,
        )
    }

    /// Canonicalize a whole-string span.
    ///
    /// # Errors
    ///
    /// Returns [`crate::SurfaceError`] for an empty, whitespace-only, or
    /// untrimmed surface.
    pub fn lemma_span_text_with_info<'a>(
        &'a self,
        text: &'a str,
        sentence_initial: bool,
        options: Options,
    ) -> std::result::Result<probe::Answer<'a>, crate::SurfaceError> {
        let (tokens, gaps) = crate::split_surface(text)?;
        let separators = (!gaps.is_empty()).then_some(gaps.as_slice());
        Ok(self.lemma_span_with_info(text, &tokens, separators, sentence_initial, options))
    }

    /// Whether an enabled stored table, including an identity override,
    /// answers this token.
    #[must_use]
    pub fn knows(&self, token: &str) -> bool {
        self.knows_with(token, false, Options::default())
    }

    /// Coverage query with explicit options and suffix rules disabled.
    #[must_use]
    pub fn knows_with(&self, token: &str, sentence_initial: bool, mut options: Options) -> bool {
        options.use_rules = false;
        self.base.prime_entities(options);
        probe::answered_with_user(&self.base, &self.user, token, sentence_initial, options)
    }

    /// Lemmatize one sentence.
    #[must_use]
    pub fn lemmatize<'a>(&'a self, tokens: &[&'a str]) -> Vec<Cow<'a, str>> {
        tokens
            .iter()
            .enumerate()
            .map(|(index, token)| self.lemma(token, index == 0))
            .collect()
    }

    /// Lemmatize one sentence with provenance and one probe per token.
    #[must_use]
    pub fn lemmatize_with_info<'a>(&'a self, tokens: &[&'a str]) -> Vec<probe::Answer<'a>> {
        tokens
            .iter()
            .enumerate()
            .map(|(index, token)| self.lemma_with_info(token, index == 0, Options::default()))
            .collect()
    }
}

/// A base lexicon, or one with a user layer attached: what a binding holds.
///
/// Matched once per scalar call and once outside each batch, so the base
/// [`Lexicon`] hot loop stays monomorphized and pays no user-layer miss.
#[derive(Debug)]
pub enum RuntimeLexicon {
    /// No user layer was supplied.
    Base(Box<Lexicon>),
    /// A user layer sits in front of the base.
    Layered(Box<LayeredLexicon>),
}

impl RuntimeLexicon {
    /// Open `base`, attaching `user` when one was supplied.
    ///
    /// # Errors
    ///
    /// Returns an error if the user layer belongs to another language.
    pub fn new(base: Lexicon, user: Option<UserDictionary>) -> Result<Self> {
        Ok(match user {
            Some(user) => Self::Layered(Box::new(LayeredLexicon::new(base, user)?)),
            None => Self::Base(Box::new(base)),
        })
    }

    /// Trained base snapshot.
    #[must_use]
    pub const fn base(&self) -> &Lexicon {
        match self {
            Self::Base(base) => base,
            Self::Layered(layered) => layered.base(),
        }
    }

    /// Base metadata.
    #[must_use]
    pub const fn meta(&self) -> &Meta {
        self.base().meta()
    }

    /// See [`Lexicon::span_options`].
    #[must_use]
    pub fn span_options(&self) -> Options {
        self.base().span_options()
    }

    /// Span conventions currently in force.
    #[must_use]
    pub const fn span_conventions(&self) -> probe::SpanConventions {
        match self {
            Self::Base(base) => base.span_conventions(),
            Self::Layered(layered) => layered.span_conventions(),
        }
    }

    /// Override the measured span conventions; for the acceptance machinery.
    pub const fn set_span_conventions(&mut self, conventions: probe::SpanConventions) {
        match self {
            Self::Base(base) => base.set_span_conventions(conventions),
            Self::Layered(layered) => layered.set_span_conventions(conventions),
        }
    }

    /// Authenticate the base entity table now.
    ///
    /// # Errors
    ///
    /// Returns the base entity-table integrity or codec error.
    pub fn ensure_entities(&self) -> Result<()> {
        self.base().ensure_entities()
    }

    /// Whether the entity table has already been authenticated.
    #[must_use]
    pub fn entities_ready(&self) -> bool {
        self.base().entities_ready()
    }

    /// See [`Lexicon::lemma_with`].
    #[must_use]
    pub fn lemma_with<'a>(
        &'a self,
        token: &'a str,
        sentence_initial: bool,
        options: Options,
    ) -> Cow<'a, str> {
        match self {
            Self::Base(base) => base.lemma_with(token, sentence_initial, options),
            Self::Layered(layered) => layered.lemma_with(token, sentence_initial, options),
        }
    }

    /// The token lemma for a batch loop that already checked an unstamped base.
    #[cfg(feature = "python")]
    pub(crate) fn lemma_with_plain<'a>(
        &'a self,
        token: &'a str,
        sentence_initial: bool,
        options: Options,
    ) -> Cow<'a, str> {
        match self {
            Self::Base(base) => base.lemma_with_plain(token, sentence_initial, options),
            Self::Layered(layered) => layered.lemma_with(token, sentence_initial, options),
        }
    }

    /// The token lemma for a batch loop over a stamped lexicon.
    #[cfg(feature = "python")]
    pub(crate) fn lemma_with_recased<'a>(
        &'a self,
        token: &'a str,
        sentence_initial: bool,
        options: Options,
    ) -> Cow<'a, str> {
        match self {
            Self::Base(base) => base.lemma_recased(token, sentence_initial, options),
            Self::Layered(layered) => layered.lemma_with(token, sentence_initial, options),
        }
    }

    /// See [`Lexicon::lemma_with_info`].
    #[must_use]
    pub fn lemma_with_info<'a>(
        &'a self,
        token: &'a str,
        sentence_initial: bool,
        options: Options,
    ) -> probe::Answer<'a> {
        match self {
            Self::Base(base) => base.lemma_with_info(token, sentence_initial, options),
            Self::Layered(layered) => layered.lemma_with_info(token, sentence_initial, options),
        }
    }

    /// See [`Lexicon::lemma_in_context`].
    #[must_use]
    pub fn lemma_in_context<'a>(
        &'a self,
        token: &'a str,
        sentence_initial: bool,
        options: Options,
        upos: Option<&str>,
        previous: Option<&str>,
        next: Option<&str>,
    ) -> probe::Answer<'a> {
        match self {
            Self::Base(base) => {
                base.lemma_in_context(token, sentence_initial, options, upos, previous, next)
            }
            Self::Layered(layered) => {
                layered.lemma_in_context(token, sentence_initial, options, upos, previous, next)
            }
        }
    }

    /// See [`Lexicon::lemma_span_with_info`].
    #[must_use]
    pub fn lemma_span_with_info<'a>(
        &'a self,
        span: &'a str,
        tokens: &[&str],
        separators: Option<&[&str]>,
        sentence_initial: bool,
        options: Options,
    ) -> probe::Answer<'a> {
        match self {
            Self::Base(base) => {
                base.lemma_span_with_info(span, tokens, separators, sentence_initial, options)
            }
            Self::Layered(layered) => {
                layered.lemma_span_with_info(span, tokens, separators, sentence_initial, options)
            }
        }
    }

    /// See [`Lexicon::knows_with`].
    #[must_use]
    pub fn knows_with(&self, token: &str, sentence_initial: bool, options: Options) -> bool {
        match self {
            Self::Base(base) => base.knows_with(token, sentence_initial, options),
            Self::Layered(layered) => layered.knows_with(token, sentence_initial, options),
        }
    }

    /// Whether a non-empty user layer is attached.
    #[must_use]
    pub const fn has_user_dictionary(&self) -> bool {
        matches!(self, Self::Layered(layered) if !layered.user_dictionary().is_empty())
    }

    /// Active manual override count.
    #[must_use]
    pub const fn user_entries(&self) -> u64 {
        match self {
            Self::Base(_) => 0,
            Self::Layered(layered) => layered.user_dictionary().len(),
        }
    }

    /// Resolved user generation, if a layer is attached.
    #[must_use]
    pub fn user_generation(&self) -> Option<&str> {
        match self {
            Self::Base(_) => None,
            Self::Layered(layered) => Some(layered.user_dictionary().generation()),
        }
    }

    /// Held-out accuracy, unknown once a non-empty user layer changes answers.
    #[must_use]
    pub const fn composite_accuracy(&self) -> Option<f64> {
        if self.has_user_dictionary() {
            None
        } else {
            self.meta().accuracy
        }
    }

    /// Copying baseline, unknown under a non-empty user layer.
    #[must_use]
    pub const fn composite_baseline(&self) -> Option<f64> {
        if self.has_user_dictionary() {
            None
        } else {
            self.meta().baseline
        }
    }

    /// Usefulness verdict, unknown under a non-empty user layer.
    #[must_use]
    pub fn composite_is_useful(&self) -> Option<bool> {
        match self {
            Self::Base(base) => base.is_useful(),
            Self::Layered(layered) => layered.is_useful(),
        }
    }
}
