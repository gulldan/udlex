//! Canonical Unicode and language-aware casing shared by training and lookup.
//!
//! On-disk keys are NFC; normalising at every ingress makes equivalent
//! spellings address the same entry, without allocating for normal text.

use std::borrow::Cow;

use icu_casemap::CaseMapper;
use icu_casemap::options::TitlecaseOptions;
use icu_locale_core::{LanguageIdentifier, langid};
use icu_properties::props::Script;
use icu_properties::script::ScriptWithExtensions;
use unicode_normalization::{IsNormalized, UnicodeNormalization, is_nfc, is_nfc_quick};

const SCRIPT_PROFILE_WORDS: usize = 4;
const SCRIPT_PROFILE_BITS: usize = u64::BITS as usize;

/// Strong Unicode scripts in one surface or lemma, without heap allocation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ScriptProfile {
    words: [u64; SCRIPT_PROFILE_WORDS],
    unsupported: bool,
}

impl ScriptProfile {
    fn insert(&mut self, script: Script) {
        // Deprecated in icu_properties 2.3 without a replacement (icu4x#6067);
        // the numeric UScriptCode is still what indexes the profile bits.
        #[allow(deprecated)]
        let value = usize::from(script.to_icu4c_value());
        let Some(word) = self.words.get_mut(value / SCRIPT_PROFILE_BITS) else {
            self.unsupported = true;
            return;
        };
        *word |= 1_u64 << (value % SCRIPT_PROFILE_BITS);
    }

    fn has_strong_script(self) -> bool {
        self.words.iter().any(|word| *word != 0)
    }
}

fn compatible_script(script: Script, iso639_3: &str) -> Script {
    if iso639_3 == "jpn" && matches!(script, Script::Han | Script::Hiragana | Script::Katakana) {
        Script::Han
    } else {
        script
    }
}

fn script_profile(value: &str, iso639_3: &str) -> ScriptProfile {
    let mut profile = ScriptProfile::default();
    if value.is_ascii() {
        if value.bytes().any(|byte| byte.is_ascii_alphabetic()) {
            profile.insert(Script::Latin);
        }
        return profile;
    }
    let scripts = ScriptWithExtensions::new();
    for character in value.chars() {
        let script = scripts.get_script_val(character);
        if matches!(script, Script::Common | Script::Inherited) {
            continue;
        }
        if script == Script::Unknown {
            profile.unsupported = true;
            continue;
        }
        profile.insert(compatible_script(script, iso639_3));
    }
    profile
}

/// Whether a changing form-to-lemma pair preserves its Unicode script set.
///
/// Identity rows are always safe. Punctuation and inherited marks do not
/// manufacture a script, while non-identity pairs made only of such code
/// points fail closed. The builder and the user-dictionary writer call this
/// one implementation against the same pinned ICU data.
#[must_use]
pub fn scripts_compatible(form: &str, lemma: &str, iso639_3: &str) -> bool {
    if form == lemma {
        return true;
    }
    let form_profile = script_profile(form, iso639_3);
    let lemma_profile = script_profile(lemma, iso639_3);
    !form_profile.unsupported
        && !lemma_profile.unsupported
        && form_profile.has_strong_script()
        && form_profile == lemma_profile
}

/// The case mapping a language uses for dictionary keys and generated lemmas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Casing {
    /// Unicode's default, locale-independent mapping.
    #[default]
    Root,
    /// Turkish casing.
    Turkish,
    /// Azerbaijani casing.
    Azerbaijani,
    /// Lithuanian dotted-I context.
    Lithuanian,
    /// Greek contextual case mapping.
    Greek,
    /// Armenian ligature tailoring.
    Armenian,
    /// Dutch IJ title casing.
    Dutch,
}

impl Casing {
    /// Selects the casing from an ISO 639-3 language code.
    #[must_use]
    pub fn for_language(iso639_3: &str) -> Self {
        match iso639_3 {
            "tur" | "ota" => Self::Turkish,
            "aze" | "azb" => Self::Azerbaijani,
            "lit" => Self::Lithuanian,
            "ell" | "grc" => Self::Greek,
            "hye" | "hyw" => Self::Armenian,
            "nld" => Self::Dutch,
            _ => Self::Root,
        }
    }

    /// ICU locale identifier for this mapping.
    const fn language(self) -> LanguageIdentifier {
        match self {
            Self::Root => langid!("und"),
            Self::Turkish => langid!("tr"),
            Self::Azerbaijani => langid!("az"),
            Self::Lithuanian => langid!("lt"),
            Self::Greek => langid!("el"),
            Self::Armenian => langid!("hy"),
            Self::Dutch => langid!("nl"),
        }
    }

    /// Lowercases a string under this mapping.
    #[must_use]
    pub fn lowercase(self, text: &str) -> Cow<'_, str> {
        // Only the Turkic `I` makes ASCII lowercasing locale-dependent.
        if text.is_ascii() && !matches!(self, Self::Turkish | Self::Azerbaijani) {
            return if text.bytes().any(|byte| byte.is_ascii_uppercase()) {
                Cow::Owned(text.to_ascii_lowercase())
            } else {
                Cow::Borrowed(text)
            };
        }
        if !has_lowercase_mapping(text) {
            return Cow::Borrowed(text);
        }
        match CaseMapper::new().lowercase_to_string(text, &self.language()) {
            Cow::Borrowed(value) => Cow::Borrowed(value),
            Cow::Owned(value) => Cow::Owned(nfc(&value).into_owned()),
        }
    }

    /// Applies title casing to a lowercase candidate.
    #[must_use]
    pub fn title(self, text: &str) -> String {
        let mapped = CaseMapper::new()
            .titlecase_segment_with_only_case_data_to_string(
                text,
                &self.language(),
                TitlecaseOptions::default(),
            )
            .into_owned();
        into_nfc(mapped)
    }

    /// Applies uppercase casing to a lowercase candidate.
    #[must_use]
    pub fn uppercase(self, text: &str) -> String {
        let mapped = CaseMapper::new()
            .uppercase_to_string(text, &self.language())
            .into_owned();
        into_nfc(mapped)
    }
}

/// Whether no byte sequence can start a character with the property in
/// `two` (two-byte characters, both bytes) / `wide` (lead byte only).
///
/// Conservative: flagged text falls through to the exact check.
// `index` stays within `bytes.len() + 3` and `byte >= 0xE0` before the
// subtraction; this is the crate's hottest loop, so no saturating ops.
#[allow(clippy::arithmetic_side_effects)]
fn none_flagged(bytes: &[u8], two: &[u64; 30], wide: u32, ascii: impl Fn(u8) -> bool) -> bool {
    let mut index = 0;
    while let Some(&byte) = bytes.get(index) {
        index += 1;
        if byte < 0x80 {
            if ascii(byte) {
                return false;
            }
            continue;
        }
        if byte < 0xE0 {
            let (Some(&next), Some(&row)) = (
                bytes.get(index),
                two.get(usize::from(byte.wrapping_sub(0xC2))),
            ) else {
                return false;
            };
            index += 1;
            if row & (1_u64 << (next & 0x3F)) != 0 {
                return false;
            }
            continue;
        }
        if (wide >> (byte - 0xE0)) & 1 != 0 {
            return false;
        }
        index += if byte < 0xF0 { 2 } else { 3 };
    }
    true
}

/// Second bytes that make a two-byte character composable, per lead byte
/// `0xC2..=0xDF`: bit `n` stands for the continuation byte `0x80 | n`.
/// `byte_tables_match_unicode` checks every table here against Unicode data.
const COMPOSING_TWO: [u64; 30] = [
    0x0000_0000_0000_0000, // 0xC2
    0x0000_0000_0000_0000, // 0xC3
    0x0000_0000_0000_0000, // 0xC4
    0x0000_0000_0000_0000, // 0xC5
    0x0000_0000_0000_0000, // 0xC6
    0x0000_0000_0000_0000, // 0xC7
    0x0000_0000_0000_0000, // 0xC8
    0x0000_0000_0000_0000, // 0xC9
    0x0000_0000_0000_0000, // 0xCA
    0x0000_0000_0000_0000, // 0xCB
    0xffff_ffff_ffff_ffff, // 0xCC
    0x4010_ffff_ffff_7fff, // 0xCD
    0x0000_0000_0000_0080, // 0xCE
    0x0000_0000_0000_0000, // 0xCF
    0x0000_0000_0000_0000, // 0xD0
    0x0000_0000_0000_0000, // 0xD1
    0x0000_0000_0000_00f8, // 0xD2
    0x0000_0000_0000_0000, // 0xD3
    0x0000_0000_0000_0000, // 0xD4
    0x0000_0000_0000_0000, // 0xD5
    0xbfff_ffff_fffe_0000, // 0xD6
    0x0000_0000_0000_00b6, // 0xD7
    0x0000_0000_07ff_0000, // 0xD8
    0x0001_0000_ffff_f800, // 0xD9
    0x0000_0000_0000_0000, // 0xDA
    0x0000_3d9f_9fc0_0000, // 0xDB
    0xffff_0000_0002_0000, // 0xDC
    0x0000_0000_0000_07ff, // 0xDD
    0x0000_0000_0000_0000, // 0xDE
    0x200f_f800_0000_0000, // 0xDF
];

/// Lead bytes `0xE0..=0xFF` that can start a composable three- or four-byte
/// character, one bit each.
const COMPOSING_WIDE: u32 = 0x0001_840f;

/// Second bytes of the two-byte characters lowercasing changes, laid out like
/// [`COMPOSING_TWO`].
const CASE_MAPPED_TWO: [u64; 30] = [
    0x0000_0000_0000_0000, // 0xC2
    0x0000_0000_7f7f_ffff, // 0xC3
    0xaa55_5555_5555_5555, // 0xC4
    0x2b55_5555_5555_54aa, // 0xC5
    0x11ae_d2d5_b1db_ced6, // 0xC6
    0x55d6_5555_4aaa_adb0, // 0xC7
    0x6c05_5555_5555_5555, // 0xC8
    0x0000_0000_0000_557a, // 0xC9
    0x0000_0000_0000_0000, // 0xCA
    0x0000_0000_0000_0000, // 0xCB
    0x0000_0000_0000_0000, // 0xCC
    0x8045_0000_0000_0000, // 0xCD
    0x0000_0ffb_fffe_d740, // 0xCE
    0xe690_5555_5500_8000, // 0xCF
    0x0000_ffff_ffff_ffff, // 0xD0
    0x5555_5555_0000_0000, // 0xD1
    0x5555_5555_5555_5401, // 0xD2
    0x5555_5555_5555_2aab, // 0xD3
    0xfffe_5555_5555_5555, // 0xD4
    0x0000_0000_007f_ffff, // 0xD5
    0x0000_0000_0000_0000, // 0xD6
    0x0000_0000_0000_0000, // 0xD7
    0x0000_0000_0000_0000, // 0xD8
    0x0000_0000_0000_0000, // 0xD9
    0x0000_0000_0000_0000, // 0xDA
    0x0000_0000_0000_0000, // 0xDB
    0x0000_0000_0000_0000, // 0xDC
    0x0000_0000_0000_0000, // 0xDD
    0x0000_0000_0000_0000, // 0xDE
    0x0000_0000_0000_0000, // 0xDF
];

/// Lead bytes of the wider characters lowercasing changes, laid out like
/// [`COMPOSING_WIDE`].
const CASE_MAPPED_WIDE: u32 = 0x0001_8406;

/// Returns NFC, borrowing an already-normalised string.
///
/// A string is NFC as soon as every character is a starter (combining class
/// zero) that `NFC_QC` marks `Yes` — the quick check's accepting path.
#[must_use]
pub fn nfc(text: &str) -> Cow<'_, str> {
    if none_flagged(text.as_bytes(), &COMPOSING_TWO, COMPOSING_WIDE, |_| false) {
        return Cow::Borrowed(text);
    }
    match is_nfc_quick(text.chars()) {
        IsNormalized::Yes => Cow::Borrowed(text),
        IsNormalized::Maybe if is_nfc(text) => Cow::Borrowed(text),
        IsNormalized::No | IsNormalized::Maybe => Cow::Owned(text.nfc().collect()),
    }
}

/// Normalises an owned string, retaining its allocation when it is already NFC.
pub fn into_nfc(value: String) -> String {
    match nfc(&value) {
        Cow::Borrowed(_) => value,
        Cow::Owned(normalized) => normalized,
    }
}

/// Whether lowercasing can change a string, including Unicode titlecase.
fn has_lowercase_mapping(text: &str) -> bool {
    !none_flagged(
        text.as_bytes(),
        &CASE_MAPPED_TWO,
        CASE_MAPPED_WIDE,
        |byte| byte.is_ascii_uppercase(),
    ) && text
        .chars()
        .any(|character| character.to_lowercase().ne(std::iter::once(character)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_script_profiles_match_icu() {
        let scripts = ScriptWithExtensions::new();
        for byte in 0..=0x7F_u8 {
            let character = char::from(byte);
            let script = scripts.get_script_val(character);
            let mut expected = ScriptProfile::default();
            if script == Script::Latin {
                expected.insert(script);
            } else {
                assert_eq!(script, Script::Common, "{character:?}");
            }
            for language in ["und", "eng", "jpn"] {
                for text in [character.to_string(), format!("0_{character}.9")] {
                    assert_eq!(script_profile(&text, language), expected, "{text:?}");
                }
            }
        }
        let mut latin = ScriptProfile::default();
        latin.insert(Script::Latin);
        assert_eq!(script_profile("012_AbZ.z9", "eng"), latin);
        assert_eq!(script_profile("", "eng"), ScriptProfile::default());
    }

    #[test]
    fn mixed_script_profiles_preserve_compatibility_checks() {
        for (form, lemma, language, expected) in [
            ("cafe\u{301}", "café", "fra", true),
            ("A\u{430}", "aa", "eng", false),
            ("A\u{430}", "a\u{430}", "eng", true),
            ("A漢かな", "a漢カナ", "jpn", true),
            ("A漢かな", "a漢カナ", "und", false),
            ("漢かな", "a漢カナ", "jpn", false),
            ("123", "45", "eng", false),
            ("123", "123", "eng", true),
            ("A\u{0378}", "a\u{0378}", "eng", false),
        ] {
            assert_eq!(
                scripts_compatible(form, lemma, language),
                expected,
                "{form:?} → {lemma:?} ({language})"
            );
        }
    }

    /// Rebuilds every byte table over the whole code space from Unicode data.
    #[test]
    fn byte_tables_match_unicode() {
        let mut tables = [([0_u64; 30], 0_u32, false), ([0_u64; 30], 0_u32, false)];
        let mut buffer = [0_u8; 4];
        for code in 0..=0x0010_FFFF_u32 {
            let Some(character) = char::from_u32(code) else {
                continue;
            };
            let composing = unicode_normalization::char::canonical_combining_class(character) != 0
                || !matches!(is_nfc_quick(std::iter::once(character)), IsNormalized::Yes);
            let case_mapped = character.to_lowercase().ne(std::iter::once(character));
            for (table, flagged) in tables.iter_mut().zip([composing, case_mapped]) {
                if !flagged {
                    continue;
                }
                let encoded = character.encode_utf8(&mut buffer).as_bytes();
                match (encoded.first(), encoded.get(1)) {
                    (Some(&lead), _) if lead < 0x80 => table.2 = true,
                    (Some(&lead), Some(&next)) if lead < 0xE0 => {
                        if let Some(row) = table.0.get_mut(usize::from(lead - 0xC2)) {
                            *row |= 1_u64 << (next & 0x3F);
                        }
                    }
                    (Some(&lead), _) => table.1 |= 1_u32 << (lead - 0xE0),
                    _ => unreachable!("every character encodes to at least one byte"),
                }
            }
        }
        assert_eq!(tables[0], (COMPOSING_TWO, COMPOSING_WIDE, false));
        // The ASCII half of the case table is `A..=Z`, spelled in the scan.
        assert_eq!(tables[1], (CASE_MAPPED_TWO, CASE_MAPPED_WIDE, true));
        for byte in 0..=0x7F_u8 {
            let character = char::from(byte);
            assert_eq!(
                byte.is_ascii_uppercase(),
                character.to_lowercase().ne(std::iter::once(character))
            );
        }
    }

    /// ICU is the oracle for the ASCII shortcut in [`Casing::lowercase`].
    #[test]
    fn the_ascii_shortcut_answers_exactly_what_icu_would() {
        let mapper = CaseMapper::new();
        for casing in [
            Casing::Root,
            Casing::Turkish,
            Casing::Azerbaijani,
            Casing::Lithuanian,
            Casing::Greek,
            Casing::Armenian,
            Casing::Dutch,
        ] {
            for first in 0..=0x7F_u8 {
                for second in 0..=0x7F_u8 {
                    let text: String = [char::from(first), char::from(second)].iter().collect();
                    assert_eq!(
                        casing.lowercase(&text),
                        mapper.lowercase_to_string(&text, &casing.language()),
                        "{casing:?} {text:?}"
                    );
                }
            }
            for text in ["", "Dazu", "E-Mail", "ISBN", "McDonald's", "x"] {
                assert_eq!(
                    casing.lowercase(text),
                    mapper.lowercase_to_string(text, &casing.language()),
                    "{casing:?} {text:?}"
                );
            }
        }
    }

    /// The cheap answer must never contradict the exact one.
    #[test]
    fn byte_tables_never_contradict_the_exact_check() {
        for text in [
            "",
            "plain",
            "PLAIN",
            "café",
            "cafe\u{301}",
            "Привет",
            "привет",
            "Ёлка",
            "е\u{308}лка",
            "Ελληνικά",
            "ΟΔΟΣ",
            "ανο\u{387}",
            "ırmak",
            "İstanbul",
            "한국어",
            "ハンカク",
            "中文",
            "\u{1E69}",
            "s\u{323}\u{307}",
            "ǅungla",
        ] {
            assert_eq!(nfc(text), text.chars().nfc().collect::<String>());
            assert_eq!(
                has_lowercase_mapping(text),
                text.chars()
                    .any(|character| character.to_lowercase().ne(std::iter::once(character))),
                "{text:?}"
            );
        }
    }

    #[test]
    fn nfc_borrows_normal_text_and_composes_equivalents() {
        assert!(matches!(nfc("café"), Cow::Borrowed(_)));
        assert!(matches!(nfc("a\u{331}"), Cow::Borrowed(_)));
        assert!(matches!(nfc("cafe\u{301}"), Cow::Owned(_)));
        assert_eq!(nfc("cafe\u{301}"), "café");
    }

    #[test]
    fn owned_nfc_keeps_normalised_storage_and_composes_decomposed_text() {
        for text in ["", "café", "a\u{331}", "ёлка"] {
            let owned = text.to_owned();
            let allocation = owned.as_ptr();
            let normalized = into_nfc(owned);
            assert_eq!(normalized, text);
            assert_eq!(normalized.as_ptr(), allocation);
        }
        for (text, expected) in [
            ("cafe\u{301}", "café"),
            ("е\u{308}лка", "ёлка"),
            ("A\u{30a}", "Å"),
        ] {
            assert_eq!(into_nfc(text.to_owned()), expected);
        }
    }

    #[test]
    fn owned_casing_matches_normalised_icu_across_locales() {
        let mapper = CaseMapper::new();
        for (casing, text) in [
            (Casing::Root, "CAFÉ"),
            (Casing::Root, "CAFE\u{301}"),
            (Casing::Root, "ЁЛКА"),
            (Casing::Turkish, "I\u{307}STANBUL"),
            (Casing::Azerbaijani, "İZMİR"),
            (Casing::Lithuanian, "I\u{301}"),
            (Casing::Greek, "ΟΔΟΣ"),
            (Casing::Armenian, "ԵՒ"),
            (Casing::Dutch, "IJKDIJK"),
        ] {
            let language = casing.language();
            assert_eq!(
                casing.lowercase(text),
                nfc(&mapper.lowercase_to_string(text, &language)),
                "lowercase: {casing:?} {text:?}"
            );
            assert_eq!(
                casing.title(text),
                nfc(&mapper.titlecase_segment_with_only_case_data_to_string(
                    text,
                    &language,
                    TitlecaseOptions::default(),
                )),
                "title: {casing:?} {text:?}"
            );
            assert_eq!(
                casing.uppercase(text),
                nfc(&mapper.uppercase_to_string(text, &language)),
                "uppercase: {casing:?} {text:?}"
            );
        }
    }

    #[test]
    fn turkic_i_mapping_is_language_aware() {
        assert_eq!(Casing::Turkish.lowercase("İSTANBUL"), "istanbul");
        assert_eq!(Casing::Turkish.lowercase("IRMAK"), "ırmak");
        assert_eq!(Casing::Turkish.title("istanbul"), "İstanbul");
        assert_eq!(Casing::Turkish.uppercase("izmir"), "İZMİR");
    }

    #[test]
    fn titlecase_letters_are_seen_as_case_mapped() {
        assert_eq!(Casing::Root.lowercase("ǅungla"), "ǆungla");
        assert_eq!(Casing::Root.title("ǆungla"), "ǅungla");
        assert_eq!(Casing::Root.title("ßeta"), "Sseta");
        assert_eq!(Casing::Dutch.title("ijkdijk"), "IJkdijk");
    }

    #[test]
    fn locale_mappings_remain_nfc() {
        let lowered = Casing::Lithuanian.lowercase("I\u{301}");
        assert!(unicode_normalization::is_nfc(&lowered));
    }
}
