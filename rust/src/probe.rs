//! The order in which tables are consulted, written once.
//!
//! Generic over [`Tables`], so the compiled lexicon and the builder's
//! in-memory tables are read under exactly the same rules.

use std::borrow::Cow;

use crate::format::SpanMultiword;
use crate::normalization::{Casing, nfc};
use crate::rule::{CaseKey, Rule};

mod prepared;

#[cfg(test)]
mod optimization_tests;
#[cfg(test)]
mod token_case_tests;

/// Which tables a lemmatiser is allowed to consult.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
// Each flag is an independently measurable probe stage exposed by the public
// compatibility API; an enum/state machine would forbid valid combinations.
#[allow(clippy::struct_excessive_bools)]
pub struct Options {
    /// Consult the isolated proper-name table with exact case.
    pub use_entities: bool,
    /// Retry the isolated proper-name table with a lowercased key.
    pub use_entities_lower: bool,
    /// Consult an explicitly loaded user dictionary before base common tables.
    pub use_user: bool,
    /// Consult the case-sensitive corpus table with the surface spelling.
    pub use_exact: bool,
    /// Consult the lowercase corpus table.
    pub use_lower: bool,
    /// Retry the case-sensitive corpus table with the lowercased spelling.
    pub use_exact_lower: bool,
    /// Consult the merged offline dictionary/analyser table for unknown forms.
    pub use_external: bool,
    /// Retry the external table with the lowercased spelling.
    pub use_external_lower: bool,
    /// Guess from suffixes when nothing else answers.
    ///
    /// Turn this off when guesses are unacceptable. For marked entities use
    /// [`Options::entity_lookup`], which also isolates common vocabulary.
    pub use_rules: bool,
    /// Return the original surface form without consulting any table.
    pub protected: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            use_entities: false,
            use_entities_lower: false,
            use_user: true,
            use_exact: true,
            use_lower: true,
            use_exact_lower: true,
            use_external: true,
            use_external_lower: true,
            use_rules: true,
            protected: false,
        }
    }
}

impl Options {
    /// Folds per-token protection and entity masks into this probe profile.
    ///
    /// An entity-marked token gets the entity-only profile; an ordinary token
    /// loses the lowercase entity fallback, which only applies inside entity
    /// calls.
    #[must_use]
    pub const fn for_call(mut self, protected: bool, entity: bool) -> Self {
        self.protected |= protected;
        if entity {
            self.entity_lookup()
        } else {
            self.use_entities_lower = false;
            self
        }
    }

    /// Entity-safe lookup: curated entity forms only, then exact identity.
    #[must_use]
    pub const fn entity_lookup(self) -> Self {
        Self {
            use_entities: true,
            use_entities_lower: self.use_entities_lower,
            use_user: false,
            use_exact: false,
            use_lower: false,
            use_exact_lower: false,
            use_external: false,
            use_external_lower: false,
            use_rules: false,
            protected: self.protected,
        }
    }
}

/// Measured answer conventions of the span path, properties of one lexicon
/// read from its metadata rather than caller options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpanConventions {
    /// Carry surface casing onto an all-lowercase span answer (`bg_btb`
    /// lowercases every lemma; `tr_penn` stores gold casing in its tables).
    pub case_transfer: bool,
    /// How a multiword span is answered when the whole-span entity keys miss.
    pub multiword: SpanMultiword,
}

impl Default for SpanConventions {
    fn default() -> Self {
        Self {
            case_transfer: true,
            multiword: SpanMultiword::Identity,
        }
    }
}

/// The stage that produced a lemma.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    /// Exact-case curated named-entity entry.
    Entity = 0,
    /// Curated named-entity entry retried with a lowercase key.
    EntityLower = 1,
    /// Exact surface-form corpus entry.
    Exact = 2,
    /// Lowercase corpus table.
    Lower = 3,
    /// Exact corpus table retried with a lowercase key.
    ExactLower = 4,
    /// Exact surface-form external entry.
    External = 5,
    /// External entry retried with a lowercase key.
    ExternalLower = 6,
    /// Productive suffix rule.
    Rule = 7,
    /// No stage answered; the original surface form was retained.
    Identity = 8,
    /// The caller explicitly protected the surface form.
    Protected = 9,
    /// A multiword span assembled from per-token answers after the whole-span
    /// entity keys missed; not a stored entry.
    Composed = 10,
    /// A multiword span rewritten by a compiled phrase pattern; a guess like
    /// [`Source::Rule`].
    Phrase = 11,
    /// Exact-case manual override from an explicitly loaded user dictionary.
    User = 12,
    /// Locale-lowercase manual override from an explicitly loaded user dictionary.
    UserLower = 13,
    /// Exact-table context key the caller's UPOS or neighbours selected over
    /// the form's majority lemma; reached only through the context calls.
    Context = 14,
}

impl Source {
    /// Every valid source in its stable C ABI order.
    pub const ALL: [Self; 15] = [
        Self::Entity,
        Self::EntityLower,
        Self::Exact,
        Self::Lower,
        Self::ExactLower,
        Self::External,
        Self::ExternalLower,
        Self::Rule,
        Self::Identity,
        Self::Protected,
        Self::Composed,
        Self::Phrase,
        Self::User,
        Self::UserLower,
        Self::Context,
    ];

    /// Stable numeric code used by the C ABI and language bindings.
    #[must_use]
    pub const fn code(self) -> u8 {
        self as u8
    }

    /// Decodes a stable ABI code, rejecting values unknown to this release.
    #[must_use]
    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Entity),
            1 => Some(Self::EntityLower),
            2 => Some(Self::Exact),
            3 => Some(Self::Lower),
            4 => Some(Self::ExactLower),
            5 => Some(Self::External),
            6 => Some(Self::ExternalLower),
            7 => Some(Self::Rule),
            8 => Some(Self::Identity),
            9 => Some(Self::Protected),
            10 => Some(Self::Composed),
            11 => Some(Self::Phrase),
            12 => Some(Self::User),
            13 => Some(Self::UserLower),
            14 => Some(Self::Context),
            _ => None,
        }
    }

    /// Stable machine-readable stage name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Entity => "entity",
            Self::EntityLower => "entity_lower",
            Self::Exact => "exact",
            Self::Lower => "lower",
            Self::ExactLower => "exact_lower",
            Self::External => "external",
            Self::ExternalLower => "external_lower",
            Self::Rule => "rule",
            Self::Identity => "identity",
            Self::Protected => "protected",
            Self::Composed => "composed",
            Self::Phrase => "phrase",
            Self::User => "user",
            Self::UserLower => "user_lower",
            Self::Context => "context",
        }
    }

    /// Uppercase member name used by the Python representation.
    #[cfg(feature = "python")]
    #[must_use]
    pub(crate) const fn member_name(self) -> &'static str {
        match self {
            Self::Entity => "ENTITY",
            Self::EntityLower => "ENTITY_LOWER",
            Self::Exact => "EXACT",
            Self::Lower => "LOWER",
            Self::ExactLower => "EXACT_LOWER",
            Self::External => "EXTERNAL",
            Self::ExternalLower => "EXTERNAL_LOWER",
            Self::Rule => "RULE",
            Self::Identity => "IDENTITY",
            Self::Protected => "PROTECTED",
            Self::Composed => "COMPOSED",
            Self::Phrase => "PHRASE",
            Self::User => "USER",
            Self::UserLower => "USER_LOWER",
            Self::Context => "CONTEXT",
        }
    }

    /// Whether the result came from a stored vocabulary entry rather than a guess.
    #[must_use]
    pub const fn is_known(self) -> bool {
        matches!(
            self,
            Self::Entity
                | Self::EntityLower
                | Self::Exact
                | Self::Lower
                | Self::ExactLower
                | Self::External
                | Self::ExternalLower
                | Self::User
                | Self::UserLower
                | Self::Context
        )
    }

    /// Whether a lemmatisation stage produced the final answer, even one
    /// spelled like the input; only `Identity` and `Protected` are not.
    #[must_use]
    pub const fn is_lemmatized(self) -> bool {
        !matches!(self, Self::Identity | Self::Protected)
    }
}

/// A lemma and the stage that produced it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Answer<'a> {
    /// Lemma text, borrowing either the token or a mapped string when possible.
    pub lemma: Cow<'a, str>,
    /// Probe stage that produced `lemma`.
    pub source: Source,
}

/// The six lookup tables, however they happen to be stored.
///
/// A rule arrives resolved with the string it appends. Low-confidence rules
/// come back as [`CaseAction::Stop`] on both paths.
///
/// [`CaseAction::Stop`]: crate::rule::CaseAction::Stop
pub trait Tables {
    /// Exact-case curated proper-name form to canonical lemma.
    fn entities(&self, key: &str) -> Option<Cow<'_, str>>;
    /// Locale-lowercased curated proper-name form to canonical lemma.
    fn entities_lower(&self, key: &str) -> Option<Cow<'_, str>>;
    /// Surface form to lemma, case sensitive.
    fn exact(&self, key: &str) -> Option<Cow<'_, str>>;
    /// Lowercased surface form to lemma.
    fn lower(&self, key: &str) -> Option<Cow<'_, str>>;
    /// Form to lemma harvested from a dictionary rather than the corpus.
    fn external(&self, key: &str) -> Option<Cow<'_, str>>;
    /// The rule filed under one case bucket and one suffix, and what it appends.
    fn rule(&self, bucket: u8, suffix: &str) -> Option<(Rule, Cow<'_, str>)>;
    /// Whether compiled phrase patterns are there to be asked at all, so a
    /// lexicon without them never builds a phrase key.
    fn has_phrases(&self) -> bool {
        false
    }
    /// The component-wise suffix rewrite filed under one phrase key; only a
    /// compiled lexicon carries the table.
    fn phrase(&self, key: &str) -> Option<Cow<'_, str>> {
        let _ = key;
        None
    }
    /// Whether no table holds anything.
    fn is_empty(&self) -> bool;
    /// Whether every non-entity lookup/rule stage is empty.
    fn common_is_empty(&self) -> bool;
    /// Longest suffix the rule table indexes, in characters.
    fn max_suffix(&self) -> usize;
    /// Language-aware case mapping for lookup keys and generated lemmas.
    fn casing(&self) -> Casing {
        Casing::Root
    }
    /// Measured span-path conventions of this lexicon.
    fn span_conventions(&self) -> SpanConventions {
        SpanConventions::default()
    }
}

/// Optional manual layer, generic so the base-only monomorphization contains
/// neither a user branch nor a second table miss.
pub trait UserTables {
    /// Compile-time false for [`NoUser`].
    const ENABLED: bool;
    fn exact(&self, key: &str) -> Option<Cow<'_, str>>;
    fn lower(&self, key: &str) -> Option<Cow<'_, str>>;
    fn is_empty(&self) -> bool;
}

/// Zero-sized implementation used by every unchanged base-only entry point.
struct NoUser;

impl UserTables for NoUser {
    const ENABLED: bool = false;

    fn exact(&self, _key: &str) -> Option<Cow<'_, str>> {
        None
    }

    fn lower(&self, _key: &str) -> Option<Cow<'_, str>> {
        None
    }

    fn is_empty(&self) -> bool {
        true
    }
}

/// One table set seen through a second type, so a second entry point gets
/// its own monomorphisation of the probe instead of sharing — and spoiling the
/// inlining of — the plain one. `KIND` only tells the instantiations apart.
#[repr(transparent)]
pub struct View<T: ?Sized, const KIND: u8>(T);

impl<T: Tables + ?Sized, const KIND: u8> View<T, KIND> {
    const fn of(tables: &T) -> &Self {
        // SAFETY: `repr(transparent)` gives the view the layout of `T`.
        unsafe { &*(std::ptr::from_ref(tables) as *const Self) }
    }
}

impl<T: Tables + ?Sized, const KIND: u8> Tables for View<T, KIND> {
    fn entities(&self, key: &str) -> Option<Cow<'_, str>> {
        self.0.entities(key)
    }
    fn entities_lower(&self, key: &str) -> Option<Cow<'_, str>> {
        self.0.entities_lower(key)
    }
    fn exact(&self, key: &str) -> Option<Cow<'_, str>> {
        self.0.exact(key)
    }
    fn lower(&self, key: &str) -> Option<Cow<'_, str>> {
        self.0.lower(key)
    }
    fn external(&self, key: &str) -> Option<Cow<'_, str>> {
        self.0.external(key)
    }
    fn rule(&self, bucket: u8, suffix: &str) -> Option<(Rule, Cow<'_, str>)> {
        self.0.rule(bucket, suffix)
    }
    fn has_phrases(&self) -> bool {
        self.0.has_phrases()
    }
    fn phrase(&self, key: &str) -> Option<Cow<'_, str>> {
        self.0.phrase(key)
    }
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    fn common_is_empty(&self) -> bool {
        self.0.common_is_empty()
    }
    fn max_suffix(&self) -> usize {
        self.0.max_suffix()
    }
    fn casing(&self) -> Casing {
        self.0.casing()
    }
    fn span_conventions(&self) -> SpanConventions {
        self.0.span_conventions()
    }
}

impl UserTables for crate::UserDictionary {
    const ENABLED: bool = true;

    fn exact(&self, key: &str) -> Option<Cow<'_, str>> {
        self.exact(key)
    }

    fn lower(&self, key: &str) -> Option<Cow<'_, str>> {
        self.lower(key)
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }
}

/// Returns the lemma of one token.
///
/// Borrows the token when nothing is known about it, so the common
/// "leave it alone" path costs no allocation.
pub fn lemma<'a, T: Tables + ?Sized>(
    tables: &'a T,
    token: &'a str,
    sentence_initial: bool,
    options: Options,
) -> Cow<'a, str> {
    lemma_with_info(tables, token, sentence_initial, options).lemma
}

/// Probes a base-only sentence with default options, checking empty tables once.
pub fn lemmas<'a, T: Tables + ?Sized>(tables: &'a T, tokens: &[&'a str]) -> Vec<Cow<'a, str>> {
    if tables.is_empty() {
        return tokens.iter().map(|token| Cow::Borrowed(*token)).collect();
    }
    let options = Options::default();
    tokens
        .iter()
        .enumerate()
        .map(|(index, token)| {
            lemma_with_tables(tables, &NoUser, token, index == 0, options, true).lemma
        })
        .collect()
}

/// Returns a lemma together with the exact stage that produced it.
///
/// Keep the known-answer checks small enough to inline into callers.
#[inline]
pub fn lemma_with_info<'a, T: Tables + ?Sized>(
    tables: &'a T,
    token: &'a str,
    sentence_initial: bool,
    options: Options,
) -> Answer<'a> {
    lemma_with_layer(tables, &NoUser, token, sentence_initial, options)
}

/// [`lemma_with_info`] for the `token_case = surface` wrappers: the same probe
/// on its own instantiation ([`View`]), so the plain path keeps its shape.
#[inline]
pub fn lemma_with_info_stamped<'a, T: Tables + ?Sized>(
    tables: &'a T,
    token: &'a str,
    sentence_initial: bool,
    options: Options,
) -> Answer<'a> {
    lemma_with_layer(
        View::<T, 1>::of(tables),
        &NoUser,
        token,
        sentence_initial,
        options,
    )
}

/// [`lemma_with_info`] for the context calls, on a third instantiation.
#[inline]
pub fn lemma_with_info_contextual<'a, T: Tables + ?Sized>(
    tables: &'a T,
    token: &'a str,
    sentence_initial: bool,
    options: Options,
) -> Answer<'a> {
    lemma_with_layer(
        View::<T, 2>::of(tables),
        &NoUser,
        token,
        sentence_initial,
        options,
    )
}

/// Layered token probe used by [`crate::LayeredLexicon`].
pub fn lemma_with_user<'a, T: Tables + ?Sized>(
    tables: &'a T,
    user: &'a crate::UserDictionary,
    token: &'a str,
    sentence_initial: bool,
    options: Options,
) -> Answer<'a> {
    lemma_with_layer(tables, user, token, sentence_initial, options)
}

#[expect(
    clippy::inline_always,
    reason = "keeps known answers ahead of the measured per-token probe call and stack frame"
)]
#[inline(always)]
fn lemma_with_layer<'a, T: Tables + ?Sized, U: UserTables + ?Sized>(
    tables: &'a T,
    user: &'a U,
    token: &'a str,
    sentence_initial: bool,
    options: Options,
) -> Answer<'a> {
    if options.protected {
        return Answer {
            lemma: Cow::Borrowed(token),
            source: Source::Protected,
        };
    }
    let user_empty = !U::ENABLED || !options.use_user || user.is_empty();
    if tables.is_empty() && user_empty {
        return Answer {
            lemma: Cow::Borrowed(token),
            source: Source::Identity,
        };
    }
    lemma_with_tables(tables, user, token, sentence_initial, options, user_empty)
}

// Separate known-answer checks from normalisation and table probing.
#[inline]
fn lemma_with_tables<'a, T: Tables + ?Sized, U: UserTables + ?Sized>(
    tables: &'a T,
    user: &'a U,
    token: &'a str,
    sentence_initial: bool,
    options: Options,
    user_empty: bool,
) -> Answer<'a> {
    // Stored keys are NFC, so a byte-exact hit needs no normalisation. Entity
    // lookups retain their precedence; only misses inspect the remaining tables.
    let no_entity_probe = !options.use_entities && !options.use_entities_lower;
    let raw_user_exact_missed = U::ENABLED && !user_empty && no_entity_probe;
    if raw_user_exact_missed && let Some(found) = user.exact(token) {
        return Answer {
            lemma: found,
            source: Source::User,
        };
    }
    // A populated user layer must be able to override even an exact base row,
    // so the base exact fast path runs only without one.
    let raw_exact_missed = options.use_exact && no_entity_probe && user_empty;
    if raw_exact_missed && let Some(found) = tables.exact(token) {
        return Answer {
            lemma: found,
            source: Source::Exact,
        };
    }
    // Stop before NFC when nothing else can answer; an entity-only bundle
    // still continues to its entity tables.
    let base_common_empty = tables.common_is_empty();
    let common_empty = base_common_empty && user_empty;
    if common_empty && no_entity_probe {
        return Answer {
            lemma: Cow::Borrowed(token),
            source: Source::Identity,
        };
    }
    if common_empty && tables.is_empty() {
        return Answer {
            lemma: Cow::Borrowed(token),
            source: Source::Identity,
        };
    }
    let normalised = nfc(token);
    if options.use_entities
        && let Some(found) = tables.entities(&normalised)
    {
        return Answer {
            lemma: found,
            source: Source::Entity,
        };
    }

    // A borrowed NFC is the same bytes the raw exact probe already missed.
    let exact_already_missed = raw_exact_missed && matches!(normalised, Cow::Borrowed(_));
    prepared::continue_with(
        tables,
        token,
        sentence_initial,
        options,
        &prepared::Prepared {
            normalised: &normalised,
            lowered: None,
            casing: tables.casing(),
            base_common_empty,
            user_empty,
            exact_already_missed,
            user_exact_already_missed: raw_user_exact_missed
                && matches!(normalised, Cow::Borrowed(_)),
        },
        user,
    )
}

/// A neighbour as context keys spell it: NFC, then lowercased.
pub fn context_word(word: &str, casing: Casing) -> Cow<'_, str> {
    match nfc(word) {
        Cow::Borrowed(word) => casing.lowercase(word),
        Cow::Owned(word) => Cow::Owned(casing.lowercase(&word).into_owned()),
    }
}

/// Lets the caller's context override an exact-table answer.
///
/// The first context key found — `upos`, then `previous`, then `next` —
/// answers as [`Source::Context`]; `None` or an empty tag skips a key.
/// Applied after the probe, never inside it, to keep the probe's shape.
#[inline(never)]
pub fn in_context<'a, T: Tables + ?Sized>(
    tables: &'a T,
    answer: Answer<'a>,
    token: &str,
    upos: Option<&str>,
    previous: Option<&str>,
    next: Option<&str>,
) -> Answer<'a> {
    use crate::format::{CONTEXT_NEXT, CONTEXT_PREVIOUS, CONTEXT_UPOS, context_key};

    if answer.source != Source::Exact {
        return answer;
    }
    let form = nfc(token);
    let casing = tables.casing();
    let features = [
        (
            CONTEXT_UPOS,
            upos.filter(|tag| !tag.is_empty()).map(Cow::Borrowed),
        ),
        (
            CONTEXT_PREVIOUS,
            previous.map(|word| context_word(word, casing)),
        ),
        (CONTEXT_NEXT, next.map(|word| context_word(word, casing))),
    ];
    let mut key = String::new();
    for (feature, value) in features {
        let Some(value) = value else {
            continue;
        };
        context_key(&mut key, &form, feature, &value);
        if let Some(lemma) = tables.exact(&key) {
            return Answer {
                lemma,
                source: Source::Context,
            };
        }
    }
    answer
}

/// Copies the surface casing of a mid-sentence capital onto the prefix its
/// answer shares with it, compared lowercased: `Europol-Mitarbeitern` ruled
/// to `Europol-mitarbeiter` becomes `Europol-Mitarbeiter`.
///
/// Only lowered, external and rule answers qualify, and not when casing
/// changes a string's length. Applied after the probe for `token_case =
/// surface`; the eligibility check stays inline, the body out of line.
#[inline]
pub fn recase_surface(answer: &mut Answer<'_>, form: &str, sentence_initial: bool, casing: Casing) {
    if !matches!(
        answer.source,
        Source::Lower
            | Source::ExactLower
            | Source::External
            | Source::ExternalLower
            | Source::Rule
    ) || CaseKey::of(form, sentence_initial) != CaseKey::MidSentenceCapital
    {
        return;
    }
    recase_eligible_surface(answer, form, casing);
}

#[inline(never)]
fn recase_eligible_surface(answer: &mut Answer<'_>, form: &str, casing: Casing) {
    let lower_form = casing.lowercase(form);
    let lower_lemma = casing.lowercase(&answer.lemma);
    if lower_form.chars().count() != form.chars().count()
        || lower_lemma.chars().count() != answer.lemma.chars().count()
    {
        return;
    }
    let mut result = String::with_capacity(answer.lemma.len());
    let mut shared = 0_usize;
    for ((surface, lowered), (original, answered)) in form
        .chars()
        .zip(lower_form.chars())
        .zip(answer.lemma.chars().zip(lower_lemma.chars()))
    {
        if lowered != answered {
            break;
        }
        result.push(surface);
        shared = shared.saturating_add(original.len_utf8());
    }
    let Some(rest) = answer.lemma.get(shared..) else {
        return;
    };
    result.push_str(rest);
    if result == answer.lemma {
        return;
    }
    answer.lemma = Cow::Owned(result);
}

/// Applies one surface word's case shape to one all-lowercase answer word.
fn recase_word(word: &str, pattern: &str, casing: Casing, result: &mut String) {
    if !pattern.chars().next().is_some_and(char::is_uppercase) {
        result.push_str(word);
        return;
    }
    let mut capitals = 0_usize;
    let mut lowers = 0_usize;
    for character in pattern.chars() {
        if character.is_uppercase() {
            capitals = capitals.saturating_add(1);
        } else if character.is_lowercase() {
            lowers = lowers.saturating_add(1);
        }
    }
    if capitals > 1 && lowers == 0 {
        result.push_str(&casing.uppercase(word));
    } else {
        result.push_str(&casing.title(word));
    }
}

/// Carries the surface casing onto an answer spelled entirely lowercase.
///
/// Word by word when the word counts agree, otherwise the whole answer by the
/// whole surface. An answer with any capital, or a span not opening with one,
/// passes untouched.
fn inherit_surface_case<'a>(span: &str, lemma: Cow<'a, str>, casing: Casing) -> Cow<'a, str> {
    if !span.chars().next().is_some_and(char::is_uppercase) || lemma.chars().any(char::is_uppercase)
    {
        return lemma;
    }
    let mut result = String::with_capacity(lemma.len().saturating_add(8));
    let mut patterns = span.split_whitespace();
    let mut remainder: &str = &lemma;
    let mut same_word_count = true;
    while let Some(position) = remainder.find(|c: char| !c.is_whitespace()) {
        let (gap, tail) = remainder.split_at(position);
        result.push_str(gap);
        let end = tail.find(char::is_whitespace).unwrap_or(tail.len());
        let (word, rest) = tail.split_at(end);
        let Some(pattern) = patterns.next() else {
            same_word_count = false;
            break;
        };
        recase_word(word, pattern, casing, &mut result);
        remainder = rest;
    }
    if same_word_count && patterns.next().is_none() {
        result.push_str(remainder);
    } else {
        result.clear();
        recase_word(&lemma, span, casing, &mut result);
    }
    Cow::Owned(result)
}

/// Applies the lexicon's case convention to one span answer.
fn convention_case<'a>(
    span: &str,
    lemma: Cow<'a, str>,
    casing: Casing,
    conventions: SpanConventions,
) -> Cow<'a, str> {
    if conventions.case_transfer {
        inherit_surface_case(span, lemma, casing)
    } else {
        lemma
    }
}

/// One token treated as a complete span: entity keys first, then the token
/// chain with the entity probes off (they were just asked).
///
/// Also the per-word step of composing a multiword span.
fn token_span_answer<'a, T: Tables + ?Sized, U: UserTables + ?Sized>(
    tables: &'a T,
    user: &'a U,
    token: &'a str,
    sentence_initial: bool,
    options: Options,
    conventions: SpanConventions,
    casing: Casing,
) -> Answer<'a> {
    let normalised = nfc(token);
    if let Some(found) = tables.entities(&normalised) {
        return Answer {
            lemma: convention_case(token, found, casing, conventions),
            source: Source::Entity,
        };
    }
    let prepared_lowered = if options.use_entities_lower {
        let lowered = casing.lowercase(&normalised);
        if lowered != normalised
            && let Some(found) = tables.entities_lower(&lowered)
        {
            return Answer {
                lemma: convention_case(token, found, casing, conventions),
                source: Source::EntityLower,
            };
        }
        Some(lowered)
    } else {
        None
    };
    let mut common = options;
    common.use_entities = false;
    common.use_entities_lower = false;
    let mut answer = prepared::continue_with(
        tables,
        token,
        sentence_initial,
        common,
        &prepared::Prepared {
            normalised: &normalised,
            lowered: prepared_lowered.as_deref(),
            casing,
            base_common_empty: tables.common_is_empty(),
            user_empty: !U::ENABLED || !common.use_user || user.is_empty(),
            exact_already_missed: false,
            user_exact_already_missed: false,
        },
        user,
    );
    // A rule answers from the lowercased form (`BioNTech` → `Biontech`).
    if answer.source == Source::Rule {
        recase_surface(&mut answer, token, false, casing);
    }
    Answer {
        lemma: convention_case(token, answer.lemma, casing, conventions),
        source: answer.source,
    }
}

pub use crate::format::{PHRASE_SEPARATOR, PHRASE_TAIL};

/// Builds the phrase key of a span: the locale-lowercased final
/// [`PHRASE_TAIL`] characters of every token, joined by [`PHRASE_SEPARATOR`].
/// `None` when a token is empty or carries the separator.
pub fn phrase_key(tokens: &[&str], casing: Casing) -> Option<String> {
    // Four bytes per character plus one separator.
    const PER_TOKEN_BYTES: usize = PHRASE_TAIL * 4 + 1;
    let mut key = String::with_capacity(tokens.len().saturating_mul(PER_TOKEN_BYTES));
    for (index, token) in tokens.iter().enumerate() {
        if index > 0 {
            key.push(PHRASE_SEPARATOR);
        }
        let normalised = nfc(token);
        let lowered = casing.lowercase(&normalised);
        let start = lowered
            .char_indices()
            .rev()
            .take(PHRASE_TAIL)
            .last()
            .map_or(lowered.len(), |(position, _)| position);
        let tail = lowered.get(start..)?;
        if tail.is_empty() || tail.contains(PHRASE_SEPARATOR) {
            return None;
        }
        key.push_str(tail);
    }
    Some(key)
}

/// Rewrites a span from the pattern its key selected, or gives up whole.
///
/// A pattern is `cut` then `add` per token, joined by [`PHRASE_SEPARATOR`];
/// each token must end with `cut` and keep a non-empty stem. The span's own
/// `separators` are restored.
fn apply_phrase_pattern(
    tokens: &[&str],
    separators: Option<&[&str]>,
    pattern: &str,
) -> Option<String> {
    let mut fields = pattern.split(PHRASE_SEPARATOR);
    let mut result = String::with_capacity(pattern.len().saturating_add(16));
    for (index, token) in tokens.iter().enumerate() {
        let cut = fields.next()?;
        let add = fields.next()?;
        if let Some(previous) = index.checked_sub(1) {
            result.push_str(
                separators
                    .and_then(|gaps| gaps.get(previous))
                    .copied()
                    .unwrap_or(" "),
            );
        }
        let normalised = nfc(token);
        let kept = normalised.strip_suffix(cut)?;
        if kept.is_empty() {
            return None;
        }
        result.push_str(kept);
        result.push_str(add);
    }
    if fields.next().is_some() {
        return None;
    }
    Some(result)
}

/// Returns the canonical text of one already-delimited entity span.
///
/// `span` is exactly `tokens` joined by `separators` (`None` means single
/// spaces); a single-token span may pass an empty `tokens`.
///
/// Order, as measured: the entity table with the whole span (exact, then
/// lowercase); a single word then falls through to the token chain; a
/// multiword span tries phrase patterns, then the [`SpanConventions`]
/// composition, and otherwise comes back unchanged.
pub fn span_lemma_with_info<'a, T: Tables + ?Sized>(
    tables: &'a T,
    span: &'a str,
    tokens: &[&str],
    separators: Option<&[&str]>,
    sentence_initial: bool,
    options: Options,
) -> Answer<'a> {
    span_lemma_with_layer(
        tables,
        &NoUser,
        span,
        tokens,
        separators,
        sentence_initial,
        options,
    )
}

pub fn span_lemma_with_user<'a, T: Tables + ?Sized>(
    tables: &'a T,
    user: &'a crate::UserDictionary,
    span: &'a str,
    tokens: &[&str],
    separators: Option<&[&str]>,
    sentence_initial: bool,
    options: Options,
) -> Answer<'a> {
    span_lemma_with_layer(
        tables,
        user,
        span,
        tokens,
        separators,
        sentence_initial,
        options,
    )
}

fn span_lemma_with_layer<'a, T: Tables + ?Sized, U: UserTables + ?Sized>(
    tables: &'a T,
    user: &'a U,
    span: &'a str,
    tokens: &[&str],
    separators: Option<&[&str]>,
    sentence_initial: bool,
    options: Options,
) -> Answer<'a> {
    if options.protected {
        return Answer {
            lemma: Cow::Borrowed(span),
            source: Source::Protected,
        };
    }
    let conventions = tables.span_conventions();
    let casing = tables.casing();
    if tokens.len() <= 1 {
        return token_span_answer(
            tables,
            user,
            span,
            sentence_initial,
            options,
            conventions,
            casing,
        );
    }
    let normalised = nfc(span);
    if let Some(found) = tables.entities(&normalised) {
        return Answer {
            lemma: convention_case(span, found, casing, conventions),
            source: Source::Entity,
        };
    }
    if options.use_entities_lower {
        let lowered = casing.lowercase(&normalised);
        if lowered != normalised
            && let Some(found) = tables.entities_lower(&lowered)
        {
            return Answer {
                lemma: convention_case(span, found, casing, conventions),
                source: Source::EntityLower,
            };
        }
    }
    if tables.has_phrases()
        && let Some(rewritten) = phrase_key(tokens, casing)
            .and_then(|key| tables.phrase(&key))
            .and_then(|pattern| apply_phrase_pattern(tokens, separators, &pattern))
    {
        return Answer {
            lemma: convention_case(span, Cow::Owned(rewritten), casing, conventions),
            source: Source::Phrase,
        };
    }
    let assembled = conventions.multiword.assembled_prefix();
    if assembled != Some(0) {
        let mut result = String::with_capacity(span.len().saturating_add(8));
        for (index, token) in tokens.iter().enumerate() {
            if let Some(previous) = index.checked_sub(1) {
                let gap = separators
                    .and_then(|gaps| gaps.get(previous))
                    .copied()
                    .unwrap_or(" ");
                result.push_str(gap);
            }
            if assembled.is_some_and(|prefix| index >= prefix) {
                result.push_str(token);
                continue;
            }
            let answer = token_span_answer(
                tables,
                user,
                token,
                sentence_initial && index == 0,
                options,
                conventions,
                casing,
            );
            result.push_str(&answer.lemma);
        }
        return Answer {
            lemma: Cow::Owned(result),
            source: Source::Composed,
        };
    }
    Answer {
        lemma: Cow::Borrowed(span),
        source: Source::Identity,
    }
}

/// Whether any table answered, running exactly the probes [`lemma`] runs; a
/// table that answers with the form itself counts.
pub fn answered<T: Tables + ?Sized>(
    tables: &T,
    token: &str,
    sentence_initial: bool,
    options: Options,
) -> bool {
    lemma_with_info(tables, token, sentence_initial, options)
        .source
        .is_known()
}

pub fn answered_with_user<T: Tables + ?Sized>(
    tables: &T,
    user: &crate::UserDictionary,
    token: &str,
    sentence_initial: bool,
    options: Options,
) -> bool {
    lemma_with_user(tables, user, token, sentence_initial, options)
        .source
        .is_known()
}

/// Applies the longest matching suffix rule; a stop marker there gives up
/// rather than falling back to a vaguer rule.
fn guess<T: Tables + ?Sized>(
    tables: &T,
    token: &str,
    lowered: &str,
    sentence_initial: bool,
) -> Option<String> {
    let max_suffix = tables.max_suffix();
    // Punctuation, digits and symbols carry no inflection (`.` → `ој` in mk_mtb).
    if max_suffix == 0 || !lowered.chars().any(char::is_alphabetic) {
        return None;
    }
    let bucket = CaseKey::of(token, sentence_initial).byte();
    let bytes = lowered.as_bytes();

    // Where the longest suffix starts, and how many characters it holds.
    let mut start = bytes.len();
    let mut length = 0_usize;
    for (position, _) in lowered.char_indices().rev() {
        start = position;
        length = length.saturating_add(1);
        if length == max_suffix {
            break;
        }
    }

    while length > 0 {
        if let Some((rule, add)) = tables.rule(bucket, &lowered[start..]) {
            return rule.apply_with(lowered, &add, tables.casing());
        }
        // Drop the suffix's leading character; its lead byte spells its length.
        let &lead = bytes.get(start)?;
        start = start.saturating_add(if lead < 0x80 {
            1
        } else {
            lead.leading_ones() as usize
        });
        length = length.saturating_sub(1);
    }
    None
}
