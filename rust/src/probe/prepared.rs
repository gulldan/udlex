//! Continuation shared by token and span probes after key preparation.

use std::borrow::Cow;

use super::{Answer, Options, Source, Tables, UserTables, guess};
use crate::normalization::Casing;

/// NFC/case state that a caller has already computed or ruled out.
// Independent skipped probes, not one state.
#[allow(clippy::struct_excessive_bools)]
pub(super) struct Prepared<'key> {
    pub(super) normalised: &'key str,
    pub(super) lowered: Option<&'key str>,
    pub(super) casing: Casing,
    pub(super) base_common_empty: bool,
    pub(super) user_empty: bool,
    pub(super) exact_already_missed: bool,
    pub(super) user_exact_already_missed: bool,
}

/// Continues after NFC and the exact-case entity probe.
#[inline]
#[allow(clippy::too_many_lines)]
pub(super) fn continue_with<'a, T: Tables + ?Sized, U: UserTables + ?Sized>(
    tables: &'a T,
    token: &'a str,
    sentence_initial: bool,
    options: Options,
    prepared: &Prepared<'_>,
    user: &'a U,
) -> Answer<'a> {
    if prepared.base_common_empty && prepared.user_empty && !options.use_entities_lower {
        return identity(token);
    }

    // Skip lowercasing when no enabled stage reads it (the entity profile).
    let wants_lower = options.use_entities_lower
        || (U::ENABLED && options.use_user && !prepared.user_empty)
        || options.use_lower
        || options.use_exact_lower
        || options.use_external_lower
        || options.use_rules;
    let owned_lowered;
    let (lowered, changed_case) = if wants_lower {
        let lowered = if let Some(lowered) = prepared.lowered {
            lowered
        } else {
            owned_lowered = prepared.casing.lowercase(prepared.normalised);
            &owned_lowered
        };
        (lowered, lowered != prepared.normalised)
    } else {
        // Every branch below that reads either one is disabled.
        (prepared.normalised, false)
    };
    if options.use_entities_lower
        && (!options.use_entities || changed_case)
        && let Some(found) = tables.entities_lower(lowered)
    {
        return Answer {
            lemma: found,
            source: Source::EntityLower,
        };
    }
    if U::ENABLED
        && options.use_user
        && !prepared.user_empty
        && !prepared.user_exact_already_missed
        && let Some(found) = user.exact(prepared.normalised)
    {
        return Answer {
            lemma: found,
            source: Source::User,
        };
    }
    if U::ENABLED
        && options.use_user
        && !prepared.user_empty
        && let Some(found) = user.lower(lowered)
    {
        return Answer {
            lemma: found,
            source: Source::UserLower,
        };
    }
    if options.use_exact
        && !prepared.base_common_empty
        && !prepared.exact_already_missed
        && let Some(found) = tables.exact(prepared.normalised)
    {
        return Answer {
            lemma: found,
            source: Source::Exact,
        };
    }
    if prepared.base_common_empty {
        return identity(token);
    }
    if options.use_lower
        && let Some(found) = tables.lower(lowered)
    {
        return Answer {
            lemma: found,
            source: Source::Lower,
        };
    }
    if options.use_exact_lower
        && (!options.use_exact || changed_case)
        && let Some(found) = tables.exact(lowered)
    {
        return Answer {
            lemma: found,
            source: Source::ExactLower,
        };
    }
    if options.use_external
        && let Some(found) = tables.external(prepared.normalised)
    {
        return Answer {
            lemma: found,
            source: Source::External,
        };
    }
    if options.use_external_lower
        && (!options.use_external || changed_case)
        && let Some(found) = tables.external(lowered)
    {
        return Answer {
            lemma: found,
            source: Source::ExternalLower,
        };
    }
    if options.use_rules
        && let Some(guessed) = guess(tables, prepared.normalised, lowered, sentence_initial)
    {
        return Answer {
            lemma: Cow::Owned(guessed),
            source: Source::Rule,
        };
    }
    identity(token)
}

const fn identity(token: &str) -> Answer<'_> {
    Answer {
        lemma: Cow::Borrowed(token),
        source: Source::Identity,
    }
}
