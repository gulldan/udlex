//! How a surface form becomes a lemma.
//!
//! Two value types. A [`CaseAction`] says what to do with capitalisation once
//! the letters are right; a [`Rule`] says which letters to change. Both are
//! decoded from a single `u64`, the packing the builder writes and
//! `rules.values` stores, so they are cheap to copy and carry no allocation.

use crate::normalization::Casing;

/// The bucket a suffix rule is indexed under; splitting capitals by position
/// lets each language learn its own convention for proper nouns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaseKey {
    /// The form starts with a lowercase character.
    Lowercase,
    /// Capitalised and first in its sentence.
    SentenceInitial,
    /// Capitalised anywhere else, so almost always a proper noun.
    MidSentenceCapital,
}

impl CaseKey {
    /// Classifies a token by its capitalisation and position.
    #[must_use]
    pub fn of(token: &str, sentence_initial: bool) -> Self {
        match token.chars().next() {
            Some(first) if first.to_lowercase().ne(std::iter::once(first)) && sentence_initial => {
                Self::SentenceInitial
            }
            Some(first) if first.to_lowercase().ne(std::iter::once(first)) => {
                Self::MidSentenceCapital
            }
            _ => Self::Lowercase,
        }
    }

    /// The single byte this bucket is spelled with inside a rule key.
    #[must_use]
    pub const fn byte(self) -> u8 {
        match self {
            Self::Lowercase => b'0',
            Self::SentenceInitial => b'1',
            Self::MidSentenceCapital => b'2',
        }
    }
}

/// What to do with the capitalisation of a generated lemma.
///
/// The discriminants are on-disk values and must not be renumbered.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CaseAction {
    /// Leave the string exactly as the rule produced it.
    Keep = 0,
    /// The lemma is lowercase, which is the common case.
    Lower = 1,
    /// Capitalise the first character only.
    Title = 2,
    /// Uppercase the whole lemma, for acronyms.
    Upper = 3,
    /// Not a case action but a marker: the rule did not clear the confidence
    /// threshold at build time, and the caller must fall back to the surface
    /// form rather than try a vaguer rule.
    Stop = 255,
}

impl CaseAction {
    /// Decodes one byte of the stable on-disk representation; `None` for an
    /// unknown value, unlike the explicit [`CaseAction::Stop`].
    #[must_use]
    pub const fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(Self::Keep),
            1 => Some(Self::Lower),
            2 => Some(Self::Title),
            3 => Some(Self::Upper),
            255 => Some(Self::Stop),
            _ => None,
        }
    }

    /// Encodes this action as its stable on-disk byte.
    #[must_use]
    pub const fn byte(self) -> u8 {
        match self {
            Self::Keep => 0,
            Self::Lower => 1,
            Self::Title => 2,
            Self::Upper => 3,
            Self::Stop => 255,
        }
    }

    /// Applies the action to a lemma candidate that is already lowercase.
    #[must_use]
    pub fn apply(self, text: &str) -> String {
        self.apply_with(text, Casing::Root)
    }

    /// Applies the action with language-aware case mapping.
    #[must_use]
    pub(crate) fn apply_with(self, text: &str, casing: Casing) -> String {
        match self {
            Self::Title => casing.title(text),
            Self::Upper => casing.uppercase(text),
            _ => text.to_owned(),
        }
    }
}

/// A recipe for turning an unseen surface form into a lemma.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rule {
    /// How many characters to remove from the end of the lowercased form.
    pub cut: usize,
    /// Identifier of the string to append, into the bundle's string table.
    pub add: u64,
    /// What to do with the capitalisation afterwards.
    pub case: CaseAction,
}

/// Bits 0..48 of a packed rule hold the identifier of the string to append.
const STRING_ID_MASK: u64 = (1 << 48) - 1;

impl Rule {
    /// Encodes one rule in the stable `u64` representation.
    ///
    /// Returns `None` when `cut` does not fit in its byte or `add` does not fit
    /// in the 48-bit string identifier field.
    #[must_use]
    pub fn pack(cut: usize, case: CaseAction, add: u64) -> Option<u64> {
        if cut > u8::MAX as usize || add > STRING_ID_MASK {
            return None;
        }
        let cut = cut as u64;
        let case = u64::from(case.byte());
        Some((cut << 56) | (case << 48) | add)
    }

    /// Decodes a packed rule only when its case-action byte is known; the
    /// validation entry point.
    #[must_use]
    pub const fn unpack_checked(packed: u64) -> Option<Self> {
        let Some(case) = CaseAction::from_byte(((packed >> 48) & 0xFF) as u8) else {
            return None;
        };
        Some(Self {
            cut: (packed >> 56) as usize,
            add: packed & STRING_ID_MASK,
            case,
        })
    }

    /// Decodes one packed `u64` from `rules.values`.
    ///
    /// # Layout
    ///
    /// | bits  | meaning                                    |
    /// |-------|--------------------------------------------|
    /// | 56–63 | characters to cut from the lowercased form  |
    /// | 48–55 | case action, or 255 for the stop marker     |
    /// | 0–47  | identifier of the string to append         |
    #[must_use]
    pub const fn unpack(packed: u64) -> Self {
        match Self::unpack_checked(packed) {
            Some(rule) => rule,
            None => Self {
                // Fail closed: applying this keeps the surface form.
                cut: (packed >> 56) as usize,
                add: packed & STRING_ID_MASK,
                case: CaseAction::Stop,
            },
        }
    }

    /// Applies the rule to a lowercased form.
    ///
    /// Returns `None` when the rule is a stop marker or would produce an empty
    /// string; in both cases the caller keeps the surface form.
    #[must_use]
    pub fn apply(&self, lowered: &str, suffix: &str) -> Option<String> {
        self.apply_with(lowered, suffix, Casing::Root)
    }

    /// Applies the rule under a language's case mapping.
    #[must_use]
    pub(crate) fn apply_with(&self, lowered: &str, suffix: &str, casing: Casing) -> Option<String> {
        if self.case == CaseAction::Stop {
            return None;
        }
        // `cut` counts characters off the end.
        let kept = if self.cut == 0 {
            lowered.len()
        } else {
            lowered
                .char_indices()
                .rev()
                .nth(self.cut.saturating_sub(1))
                .map(|(position, _)| position)?
        };
        let mut lemma = String::with_capacity(kept.saturating_add(suffix.len()));
        lemma.push_str(&lowered[..kept]);
        lemma.push_str(suffix);
        if lemma.is_empty() {
            return None;
        }
        Some(match self.case {
            CaseAction::Title | CaseAction::Upper => self.case.apply_with(&lemma, casing),
            _ => lemma,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_and_unpacks_round_trip() {
        let packed = Rule::pack(3, CaseAction::Title, 0x2A);
        assert!(packed.is_some());
        let packed = packed.unwrap_or_default();
        let rule = Rule::unpack(packed);
        assert_eq!(rule.cut, 3);
        assert_eq!(rule.add, 0x2A);
        assert_eq!(rule.case, CaseAction::Title);
    }

    #[test]
    fn rejects_out_of_range_packed_fields() {
        assert_eq!(Rule::pack(256, CaseAction::Lower, 0), None);
        assert_eq!(Rule::pack(0, CaseAction::Lower, STRING_ID_MASK + 1), None);
    }

    #[test]
    fn checked_unpack_distinguishes_unknown_actions_from_stop() {
        assert_eq!(
            Rule::unpack_checked(u64::from(CaseAction::Stop.byte()) << 48).map(|rule| rule.case),
            Some(CaseAction::Stop)
        );
        assert_eq!(Rule::unpack_checked(7_u64 << 48), None);
        assert_eq!(Rule::unpack(7_u64 << 48).case, CaseAction::Stop);
    }

    #[test]
    fn stop_marker_refuses_to_apply() {
        let rule = Rule::unpack(0xFF_u64 << 48);
        assert_eq!(rule.apply("дом", ""), None);
    }

    #[test]
    fn cuts_by_characters_not_bytes() {
        let rule = Rule::unpack((3_u64 << 56) | (1_u64 << 48));
        assert_eq!(rule.apply("котами", "").as_deref(), Some("кот"));
    }

    #[test]
    fn a_rule_cannot_cut_more_characters_than_the_token_contains() {
        let rule = Rule {
            cut: 5,
            add: 0,
            case: CaseAction::Lower,
        };
        assert_eq!(rule.apply("as", "replacement"), None);
    }

    #[test]
    fn classifies_case_buckets() {
        assert_eq!(CaseKey::of("дом", false), CaseKey::Lowercase);
        assert_eq!(CaseKey::of("Дом", true), CaseKey::SentenceInitial);
        assert_eq!(CaseKey::of("Дом", false), CaseKey::MidSentenceCapital);
    }
}
