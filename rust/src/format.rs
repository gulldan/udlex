//! Constants of the on-disk bundle format.
//!
//! Compiled under every feature combination so the reader and the builder
//! share them without depending on each other.

/// The on-disk format a build writes; independent of the crate version.
///
/// Format 5 adds the `hybrid` codec for `exact` and `lower`. What a reader
/// accepts is [`FormatRange::READ`].
pub const FORMAT_VERSION: u32 = 5;

/// The oldest compiled bundle this crate still opens.
///
/// Format 3 lacks the phrase table the manifest check requires; its sources
/// still compile, see [`FormatRange::COMPILE`].
pub const MIN_FORMAT_VERSION: u32 = 4;

/// The earliest udlex release that reads [`FORMAT_VERSION`].
///
/// Dictionary wheels pin the core by this floor, never by the packing crate's
/// version; there is no ceiling, a newer core refuses via [`FormatRange::remedy`].
pub const MIN_CORE_RELEASE: &str = "1.0.0";

/// A closed range of on-disk format versions that one operation accepts;
/// the single compatibility rule for the reader, the compiler and Python.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormatRange {
    /// Oldest version accepted.
    pub min: u32,
    /// Newest version accepted.
    pub max: u32,
}

impl FormatRange {
    /// What a reader opens: `meta.json`, bundle and wheel manifests, archives.
    pub const READ: Self = Self {
        min: MIN_FORMAT_VERSION,
        max: FORMAT_VERSION,
    };

    /// What `compile` accepts as *sources*: one version further back than a
    /// reader, since compiling produces the table format 4 added.
    pub const COMPILE: Self = Self {
        min: 3,
        max: FORMAT_VERSION,
    };

    /// Whether a bundle declaring `version` falls inside this range.
    #[must_use]
    pub const fn accepts(self, version: u32) -> bool {
        self.min <= version && version <= self.max
    }

    /// What a caller must do about a version this range does not accept.
    #[must_use]
    pub const fn remedy(self, version: u32) -> &'static str {
        if version > self.max {
            "the dictionary is ahead of this udlex, so upgrade udlex"
        } else {
            "the dictionary is behind this udlex, so rebuild the bundle with udlex.build_lexicon"
        }
    }
}

impl core::fmt::Display for FormatRange {
    /// Spells the range the way every refusal names it.
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.min == self.max {
            write!(formatter, "{}", self.min)
        } else {
            write!(formatter, "{}-{}", self.min, self.max)
        }
    }
}

/// Encoding of the isolated entity table in format-3 bundles.
///
/// Entity exact and lowercase keys share one compiled table.  A leading `0`
/// or `1` keeps the two namespaces disjoint; readers require this marker
/// rather than guessing whether an older unprefixed prototype was written.
pub const ENTITY_ENCODING: &str = "namespaced-v1";

/// Canonical Unicode and locale-aware case-mapping contract of format 3.
pub const NORMALIZATION: &str = "NFC+locale-v1";

/// Machine contract of the independent user-dictionary root and generations.
///
/// Not [`FORMAT_VERSION`]: an overlay survives rebuilds of its base.
pub const USER_DICTIONARY_SCHEMA: &str = "udlex-user-dictionary/v1";

/// Casing contract recorded by every user-dictionary generation.
pub const USER_DICTIONARY_CASING: &str = "icu-locale-v1";

/// Maximum number of manual overrides in one language.
pub const MAX_USER_DICTIONARY_ENTRIES: u64 = 1_000_000;

/// Maximum encoded size of one canonical user-dictionary field; bounds the
/// allocation one malformed TSV row can cause.
pub const MAX_USER_DICTIONARY_FIELD_BYTES: usize = 16 * 1024 * 1024;

/// Exact or locale-lowercased matching for a manual override.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UserMatch {
    /// Match only the NFC surface spelling supplied by the caller.
    Exact,
    /// Match the language-aware lowercase key of the supplied surface.
    Lower,
}

impl UserMatch {
    /// Every supported match mode in stable canonical order.
    pub const ALL: [Self; 2] = [Self::Exact, Self::Lower];

    /// Parse the machine-readable TSV/API spelling.
    #[must_use]
    pub const fn parse(value: &str) -> Option<Self> {
        match value.as_bytes() {
            b"exact" => Some(Self::Exact),
            b"lower" => Some(Self::Lower),
            _ => None,
        }
    }

    /// Stable TSV/API spelling.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Lower => "lower",
        }
    }

    /// Namespace byte used by the single compiled user table.
    #[must_use]
    pub(crate) const fn prefix(self) -> u8 {
        match self {
            Self::Exact => b'0',
            Self::Lower => b'1',
        }
    }
}

/// Encoding of the compiled phrase-pattern table: keys
/// `key 0x00 varint(cut) pattern`, with no values and no string table.
pub const PHRASE_CODEC: &str = "inline";

/// The sources a build writes and `compile` reads, and `--prune` deletes.
///
/// Always written, an empty table as an empty file, so a pruned directory is
/// recognisable. `strings.txt` is mapped at runtime and survives pruning.
pub const SOURCES: [&str; 6] = [
    "exact.tsv",
    "lower.tsv",
    "external.tsv",
    "entities.tsv",
    "phrases.tsv",
    "rules.tsv",
];

/// Separator between the fields of a phrase key and of a phrase pattern.
///
/// UNIT SEPARATOR occurs in no writing system, and both sides refuse a token
/// carrying it, so keys of different span lengths cannot collide.
pub const PHRASE_SEPARATOR: char = '\u{1f}';

/// How many final characters of each token the phrase key records; 2 beat
/// 3 and 4 on held-out mentions.
pub const PHRASE_TAIL: usize = 2;

/// Feature prefix of a context key: the caller's UPOS tag of the token.
pub const CONTEXT_UPOS: &str = "u:";
/// Feature prefix of a context key: the previous word, lowercased.
pub const CONTEXT_PREVIOUS: &str = "p:";
/// Feature prefix of a context key: the next word, lowercased.
pub const CONTEXT_NEXT: &str = "n:";
/// The previous word of a sentence's first token.
pub const SENTENCE_START: &str = "<s>";
/// The next word of a sentence's last token.
pub const SENTENCE_END: &str = "</s>";

/// Writes the exact-table key of one context feature of `form` into `key`:
/// `form`, [`PHRASE_SEPARATOR`], the feature prefix, its value.
pub fn context_key(key: &mut String, form: &str, feature: &str, value: &str) {
    key.clear();
    key.reserve(
        form.len()
            .saturating_add(PHRASE_SEPARATOR.len_utf8())
            .saturating_add(feature.len())
            .saturating_add(value.len()),
    );
    key.push_str(form);
    key.push(PHRASE_SEPARATOR);
    key.push_str(feature);
    key.push_str(value);
}

/// The previous and next word of `tokens[index]` as context keys see them.
///
/// A token that `opens` a sentence has [`SENTENCE_START`] before it; the last
/// token, and one followed by an opening token, has [`SENTENCE_END`] after it.
pub fn neighbours<T: AsRef<str>>(
    tokens: &[T],
    index: usize,
    opens: impl Fn(usize) -> bool,
) -> (&str, &str) {
    let word = |at: Option<usize>| at.and_then(|at| tokens.get(at)).map(AsRef::as_ref);
    let previous = if opens(index) {
        None
    } else {
        word(index.checked_sub(1))
    };
    let next = index.checked_add(1).filter(|&at| !opens(at));
    (
        previous.unwrap_or(SENTENCE_START),
        word(next).unwrap_or(SENTENCE_END),
    )
}

/// Longest suffix indexed by the rule table, in characters; the reader
/// refuses a bundle claiming a longer one.
pub const MAX_SUFFIX_LENGTH: usize = 6;

/// One of the four lookup tables whose codec is selected per bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LookupTableKind {
    /// Case-sensitive corpus lookup.
    Exact,
    /// Locale-lowercased corpus lookup.
    Lower,
    /// External vocabulary lookup.
    External,
    /// Isolated, namespaced entity lookup.
    Entities,
}

impl LookupTableKind {
    /// Lookup-table order used by validation, compilation, and loading.
    pub(crate) const ALL: [Self; 4] = [Self::Exact, Self::Lower, Self::External, Self::Entities];

    /// Stable metadata name and on-disk file stem.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Lower => "lower",
            Self::External => "external",
            Self::Entities => "entities",
        }
    }

    /// Whether keys must use the entity namespace encoding.
    #[cfg(any(feature = "train", test))]
    pub(crate) const fn is_entity(self) -> bool {
        matches!(self, Self::Entities)
    }

    /// Whether the table compiles for size (a double array or an automaton)
    /// rather than as a hybrid; only the rarely probed entity table does.
    #[cfg(any(feature = "train", test))]
    pub(crate) const fn is_cold(self) -> bool {
        matches!(self, Self::Entities)
    }
}

/// Which learned tables a bundle promises to contain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BundleVariant {
    /// Ship no common-word tables or rules.
    Identity,
    /// Ship only tables learned directly from the corpus.
    Corpus,
    /// Ship corpus tables and suffix rules.
    Rules,
    /// Ship corpus and external lookup tables.
    Lookup,
    /// Ship every learned table.
    Full,
}

impl BundleVariant {
    /// Candidate lattice in increasing order of capability.
    pub const CANDIDATES: [Self; 5] = [
        Self::Identity,
        Self::Corpus,
        Self::Rules,
        Self::Lookup,
        Self::Full,
    ];

    /// Parses the exact stable name used in metadata and public bindings.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        Self::CANDIDATES
            .into_iter()
            .find(|variant| variant.name() == name)
    }

    /// Stable name recorded in `meta.json`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::Corpus => "corpus",
            Self::Rules => "rules",
            Self::Lookup => "lookup",
            Self::Full => "full",
        }
    }

    /// Whether lookup-table counts obey this variant's storage promise.
    ///
    /// Entity and phrase tables are intentionally absent: they are orthogonal
    /// to the common-word acceptance lattice and may survive `identity`.
    #[must_use]
    pub const fn entries_compatible(
        self,
        exact: u64,
        lower: u64,
        external: u64,
        rules: u64,
    ) -> bool {
        match self {
            Self::Identity => exact == 0 && lower == 0 && external == 0 && rules == 0,
            Self::Corpus => external == 0 && rules == 0,
            Self::Rules => external == 0,
            Self::Lookup => rules == 0,
            Self::Full => true,
        }
    }

    /// Whether corpus-derived tables survive this variant.
    #[cfg(feature = "train")]
    pub(crate) const fn keeps_corpus(self) -> bool {
        !matches!(self, Self::Identity)
    }

    /// Whether external vocabulary survives this variant.
    #[cfg(feature = "train")]
    pub(crate) const fn keeps_external(self) -> bool {
        matches!(self, Self::Full | Self::Lookup)
    }

    /// Whether suffix rules survive this variant.
    #[cfg(feature = "train")]
    pub(crate) const fn keeps_rules(self) -> bool {
        matches!(self, Self::Full | Self::Rules)
    }
}

/// Which held-out sample decided the lattice, and how far it is trusted:
/// `dev` is running text, `train-cv` may be a slice of a dictionary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionSample {
    /// A held-out development split of running text.
    Dev,
    /// A deterministic form-disjoint holdout of the training signal.
    TrainCv,
}

impl SelectionSample {
    /// Parses the exact stable name used in metadata and public bindings.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "dev" => Some(Self::Dev),
            "train-cv" => Some(Self::TrainCv),
            _ => None,
        }
    }

    /// Stable name recorded in `meta.json`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Dev => "dev",
            Self::TrainCv => "train-cv",
        }
    }

    /// Standard errors of the paired lead a candidate must clear to ship.
    pub(crate) const fn strictness(self) -> f64 {
        match self {
            Self::Dev => DEV_STRICTNESS,
            Self::TrainCv => TRAIN_CV_STRICTNESS,
        }
    }
}

/// Standard errors of the paired lead required on a development split:
/// two-sided `McNemar` `p <= 0.01`.
const DEV_STRICTNESS: f64 = 2.576;

/// Standard errors of the paired lead required on a train-only holdout
/// (`p` near 0.003), fitted on the development splits of 57 languages.
const TRAIN_CV_STRICTNESS: f64 = 3.0;

/// One lattice candidate as the selection sample measured it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Candidate {
    /// Which configuration was measured.
    pub variant: BundleVariant,
    /// Tokens it answered exactly right.
    pub hits: u64,
    /// Tokens where it and plain copying disagree about being right: the
    /// discordant count of the paired comparison.
    pub discordant: u64,
}

/// What one candidate measures beyond its base row: the NER-mode profile and
/// the discordance against every rival. Older receipts carry only [`Candidate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandidateProfile {
    /// Tokens answered exactly right with suffix rules disabled — the same
    /// tables probed the way the NER mode probes them (`use_rules = false`).
    pub ner_hits: u64,
    /// Tokens on which this candidate and each lattice candidate disagree
    /// about being right, in lattice order; the identity column repeats
    /// [`Candidate::discordant`] and the diagonal is zero.
    pub rivals: [u64; BundleVariant::CANDIDATES.len()],
}

/// What the selection sample said about every candidate, and the rule that
/// reads it; a reader re-derives the shipped variant from this evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectionEvidence {
    sample: SelectionSample,
    tokens: u64,
    copied: u64,
    candidates: [Candidate; BundleVariant::CANDIDATES.len()],
    /// Per-candidate NER profiles and pairwise counters; `None` in older receipts.
    profiles: Option<[CandidateProfile; BundleVariant::CANDIDATES.len()]>,
}

impl SelectionEvidence {
    /// Accepts one internally consistent measurement, or nothing.
    ///
    /// Rejects anything a scoring pass cannot have produced: an empty sample,
    /// counts above it, candidates out of lattice order, or an identity row
    /// that disagrees with the copying count it is by definition equal to.
    #[must_use]
    pub fn new(
        sample: SelectionSample,
        tokens: u64,
        copied: u64,
        candidates: [Candidate; BundleVariant::CANDIDATES.len()],
    ) -> Option<Self> {
        if tokens == 0 || copied > tokens {
            return None;
        }
        for (candidate, expected) in candidates.iter().zip(BundleVariant::CANDIDATES) {
            if candidate.variant != expected
                || candidate.hits > tokens
                || candidate.discordant > tokens
            {
                return None;
            }
            if expected == BundleVariant::Identity
                && (candidate.hits != copied || candidate.discordant != 0)
            {
                return None;
            }
        }
        Some(Self {
            sample,
            tokens,
            copied,
            candidates,
            profiles: None,
        })
    }

    /// Accepts a measurement that also carries per-candidate profiles, or
    /// nothing; the profiles must be what one scoring pass can produce.
    #[must_use]
    pub fn with_profiles(
        sample: SelectionSample,
        tokens: u64,
        copied: u64,
        candidates: [Candidate; BundleVariant::CANDIDATES.len()],
        profiles: [CandidateProfile; BundleVariant::CANDIDATES.len()],
    ) -> Option<Self> {
        let mut evidence = Self::new(sample, tokens, copied, candidates)?;
        if !profiles_consistent(tokens, copied, &candidates, &profiles) {
            return None;
        }
        evidence.profiles = Some(profiles);
        Some(evidence)
    }

    /// Which sample this is.
    #[must_use]
    pub const fn sample(&self) -> SelectionSample {
        self.sample
    }

    /// How many scorable tokens it holds.
    #[must_use]
    pub const fn tokens(&self) -> u64 {
        self.tokens
    }

    /// How many of them plain copying already answered right.
    #[must_use]
    pub const fn copied(&self) -> u64 {
        self.copied
    }

    /// Every candidate, in lattice order.
    #[must_use]
    pub const fn candidates(&self) -> &[Candidate; BundleVariant::CANDIDATES.len()] {
        &self.candidates
    }

    /// Every candidate's NER profile and rivalry row, in lattice order, when
    /// the receipt is new enough to carry them.
    #[must_use]
    pub const fn profiles(&self) -> Option<&[CandidateProfile; BundleVariant::CANDIDATES.len()]> {
        self.profiles.as_ref()
    }

    /// The variant this evidence ships, its hits, and the lead it had to clear.
    ///
    /// A candidate must lead copying by `z * sqrt(b + c)` (paired `McNemar`).
    /// Without profiles the best such candidate wins, ties keeping the simpler;
    /// with profiles a challenger must also beat the incumbent by that margin
    /// on their pairwise discordance, or tie there and have more NER hits.
    #[must_use]
    pub fn verdict(&self) -> (BundleVariant, u64, f64) {
        let strictness = self.sample.strictness();
        let Some(profiles) = self.profiles.as_ref() else {
            let mut variant = BundleVariant::Identity;
            let mut hits = self.copied;
            let mut cleared = 0.0;
            for candidate in self.candidates {
                let required = strictness * (candidate.discordant as f64).sqrt();
                let lead = candidate.hits as f64 - self.copied as f64;
                if candidate.hits > hits && lead >= required {
                    variant = candidate.variant;
                    hits = candidate.hits;
                    cleared = required;
                }
            }
            return (variant, hits, cleared);
        };
        // The incumbent starts as identity: copying, with its own NER row.
        let mut winner = (BundleVariant::Identity, self.copied, 0.0);
        let mut winner_index = 0_usize;
        let mut winner_ner = self.copied;
        for (index, (candidate, profile)) in self.candidates.iter().zip(profiles).enumerate() {
            let required = strictness * (candidate.discordant as f64).sqrt();
            let lead = candidate.hits.saturating_sub(self.copied);
            if lead == 0 || (lead as f64) < required {
                continue;
            }
            let pair = profile.rivals.get(winner_index).copied().unwrap_or(0);
            let noise = strictness * (pair as f64).sqrt();
            let margin = candidate.hits.abs_diff(winner.1) as f64;
            let displaces = if margin > 0.0 && margin >= noise {
                candidate.hits > winner.1
            } else {
                profile.ner_hits > winner_ner
            };
            if displaces {
                winner = (candidate.variant, candidate.hits, required);
                winner_index = index;
                winner_ner = profile.ner_hits;
            }
        }
        winner
    }
}

/// Whether a set of candidate profiles can have come from one scoring pass.
fn profiles_consistent(
    tokens: u64,
    copied: u64,
    candidates: &[Candidate; BundleVariant::CANDIDATES.len()],
    profiles: &[CandidateProfile; BundleVariant::CANDIDATES.len()],
) -> bool {
    for (row, (candidate, profile)) in candidates.iter().zip(profiles).enumerate() {
        if profile.ner_hits > tokens {
            return false;
        }
        if candidate.variant == BundleVariant::Identity && profile.ner_hits != copied {
            return false;
        }
        if profile.rivals.first() != Some(&candidate.discordant) {
            return false;
        }
        for (column, &count) in profile.rivals.iter().enumerate() {
            if count > tokens || (column == row && count != 0) {
                return false;
            }
            let mirrored = profiles.get(column).and_then(|other| other.rivals.get(row));
            if mirrored != Some(&count) {
                return false;
            }
            let Some(rival) = candidates.get(column) else {
                return false;
            };
            if candidate.hits.abs_diff(rival.hits) > count {
                return false;
            }
        }
    }
    true
}

/// Span-path policy for transferring surface casing to a lemma.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SpanCase {
    /// Transfer casing, the legacy and metadata default.
    #[default]
    Transfer,
    /// Return the stored value without case transfer.
    Verbatim,
}

impl SpanCase {
    /// Parses the exact metadata/binding spelling.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "transfer" => Some(Self::Transfer),
            "verbatim" => Some(Self::Verbatim),
            _ => None,
        }
    }

    /// Stable metadata/binding spelling.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Transfer => "transfer",
            Self::Verbatim => "verbatim",
        }
    }

    /// Whether runtime case transfer is enabled.
    #[must_use]
    pub const fn transfers(self) -> bool {
        matches!(self, Self::Transfer)
    }
}

/// Token-path case convention for a mid-sentence capital: whether
/// `Europol-Mitarbeitern` answers `Europol-mitarbeiter` or `Europol-Mitarbeiter`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TokenCase {
    /// Return the answer as its stage spelled it, the metadata default.
    #[default]
    Keep,
    /// Copy surface casing onto the prefix the answer shares with the form.
    Surface,
}

impl TokenCase {
    /// Parses the exact metadata/binding spelling.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "keep" => Some(Self::Keep),
            "surface" => Some(Self::Surface),
            _ => None,
        }
    }

    /// Stable metadata/binding spelling.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Keep => "keep",
            Self::Surface => "surface",
        }
    }
}

/// Span-path fallback for uncovered multiword spans.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SpanMultiword {
    /// Return the whole original span, the legacy and metadata default.
    #[default]
    Identity,
    /// Assemble the result from per-token answers.
    Compose,
    /// Answer for the first token and keep the rest of the span verbatim,
    /// as declined multiword names in Slavic treebanks often need.
    Head,
}

impl SpanMultiword {
    /// Parses the exact metadata/binding spelling.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "identity" => Some(Self::Identity),
            "compose" => Some(Self::Compose),
            "head" => Some(Self::Head),
            _ => None,
        }
    }

    /// Stable metadata/binding spelling.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::Compose => "compose",
            Self::Head => "head",
        }
    }

    /// How many leading tokens of an uncovered span are looked up, `None`
    /// meaning every one of them.
    #[must_use]
    pub const fn assembled_prefix(self) -> Option<usize> {
        match self {
            Self::Identity => Some(0),
            Self::Compose => None,
            Self::Head => Some(1),
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::indexing_slicing)]
mod tests {
    use super::{
        BundleVariant, Candidate, CandidateProfile, FORMAT_VERSION, FormatRange, LookupTableKind,
        MIN_CORE_RELEASE, MIN_FORMAT_VERSION, SelectionEvidence, SelectionSample, SpanCase,
        SpanMultiword, TokenCase,
    };

    #[test]
    fn the_read_range_accepts_its_own_bounds_and_nothing_outside_them() {
        let range = FormatRange::READ;
        assert!(range.min <= range.max);
        assert!(range.accepts(range.min));
        assert!(range.accepts(range.max));
        assert!(!range.accepts(range.min.saturating_sub(1)));
        assert!(!range.accepts(range.max.saturating_add(1)));
        assert!(!range.accepts(0));
        assert!(!range.accepts(u32::MAX));
        // A writer writes the newest readable version, never anything else.
        assert_eq!(range.max, FORMAT_VERSION);
        assert_eq!(range.min, MIN_FORMAT_VERSION);
    }

    #[test]
    fn compiling_reaches_further_back_than_opening_does() {
        const {
            assert!(FormatRange::COMPILE.min < FormatRange::READ.min);
            assert!(FormatRange::COMPILE.max == FormatRange::READ.max);
            assert!(FormatRange::COMPILE.accepts(3));
            assert!(!FormatRange::READ.accepts(3));
            assert!(!FormatRange::COMPILE.accepts(2));
        }
    }

    #[test]
    fn a_refusal_names_the_action_that_can_actually_help() {
        let range = FormatRange::READ;
        let newer = range.remedy(range.max.saturating_add(1));
        let older = range.remedy(range.min.saturating_sub(1));
        assert!(newer.contains("upgrade udlex"), "{newer}");
        assert!(
            older.contains("rebuild the bundle with udlex.build_lexicon"),
            "{older}"
        );
        assert!(!newer.contains("udlex.build_lexicon"), "{newer}");
    }

    #[test]
    fn a_range_prints_a_single_version_as_one_number() {
        assert_eq!(FormatRange { min: 4, max: 4 }.to_string(), "4");
        assert_eq!(FormatRange { min: 4, max: 6 }.to_string(), "4-6");
    }

    #[test]
    fn the_wheel_pin_floor_is_a_release_number_and_not_the_crate_version() {
        let parts: Vec<&str> = MIN_CORE_RELEASE.split('.').collect();
        assert_eq!(parts.len(), 3, "{MIN_CORE_RELEASE}");
        for part in parts {
            assert!(!part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()));
        }
    }

    #[test]
    fn lookup_table_names_and_properties_cover_the_format_exactly_once() {
        use LookupTableKind::{Entities, Exact, External, Lower};

        assert_eq!(LookupTableKind::ALL.len(), 4);
        assert_eq!(Exact.name(), "exact");
        assert_eq!(Lower.name(), "lower");
        assert_eq!(External.name(), "external");
        assert_eq!(Entities.name(), "entities");
        assert_eq!(
            LookupTableKind::ALL.map(LookupTableKind::is_entity),
            [false, false, false, true]
        );
        assert_eq!(
            LookupTableKind::ALL.map(LookupTableKind::is_cold),
            [false, false, false, true]
        );
    }

    #[test]
    fn every_variant_name_round_trips_and_unknown_names_fail() {
        let names = ["identity", "corpus", "rules", "lookup", "full"];
        assert_eq!(BundleVariant::CANDIDATES.map(BundleVariant::name), names);
        for (variant, name) in BundleVariant::CANDIDATES.into_iter().zip(names) {
            assert_eq!(BundleVariant::parse(name), Some(variant));
        }
        for invalid in ["", "FULL", "rule", "unknown"] {
            assert_eq!(BundleVariant::parse(invalid), None);
        }
    }

    #[test]
    fn every_variant_enforces_its_entry_contract() {
        use BundleVariant::{Corpus, Full, Identity, Lookup, Rules};

        assert!(Identity.entries_compatible(0, 0, 0, 0));
        for counts in [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)] {
            assert!(!Identity.entries_compatible(counts.0, counts.1, counts.2, counts.3));
        }

        assert!(Corpus.entries_compatible(2, 3, 0, 0));
        assert!(!Corpus.entries_compatible(2, 3, 1, 0));
        assert!(!Corpus.entries_compatible(2, 3, 0, 1));

        assert!(Rules.entries_compatible(2, 3, 0, 5));
        assert!(!Rules.entries_compatible(2, 3, 1, 5));

        assert!(Lookup.entries_compatible(2, 3, 4, 0));
        assert!(!Lookup.entries_compatible(2, 3, 4, 1));

        assert!(Full.entries_compatible(u64::MAX, u64::MAX, u64::MAX, u64::MAX));
    }

    /// Evidence in which one candidate wins `b` tokens and loses `c` of them.
    fn evidence(
        sample: SelectionSample,
        tokens: u64,
        copied: u64,
        won: u64,
        lost: u64,
    ) -> SelectionEvidence {
        let candidates = BundleVariant::CANDIDATES.map(|variant| Candidate {
            variant,
            hits: if variant == BundleVariant::Rules {
                copied.saturating_add(won).saturating_sub(lost)
            } else {
                copied
            },
            discordant: if variant == BundleVariant::Rules {
                won.saturating_add(lost)
            } else {
                0
            },
        });
        SelectionEvidence::new(sample, tokens, copied, candidates).expect("consistent counts")
    }

    #[test]
    fn a_lead_inside_the_samples_own_error_does_not_ship() {
        // 61 tokens won, 33 lost: a lead of 28 on 94 discordant, which is
        // 2.9 standard errors. Enough for a development split, not enough for
        // a train-only holdout that may not even be running text.
        let counted = |sample| evidence(sample, 1389, 583, 61, 33).verdict().0;
        assert_eq!(counted(SelectionSample::Dev), BundleVariant::Rules);
        assert_eq!(counted(SelectionSample::TrainCv), BundleVariant::Identity);

        // The same lead measured on ten times the tokens is ten times the
        // evidence, and then even the train-only holdout accepts it.
        let ample = evidence(SelectionSample::TrainCv, 13890, 5830, 610, 330);
        assert_eq!(ample.verdict().0, BundleVariant::Rules);
    }

    #[test]
    fn the_shipped_configuration_never_loses_to_copying_on_its_own_sample() {
        for sample in [SelectionSample::Dev, SelectionSample::TrainCv] {
            for (won, lost) in [(0, 0), (0, 40), (3, 40), (40, 40), (400, 0)] {
                let measured = evidence(sample, 1000, 500, won, lost);
                let (variant, hits, _cleared) = measured.verdict();
                assert!(hits >= measured.copied(), "{variant:?} lost to copying");
            }
        }
    }

    #[test]
    fn selection_evidence_rejects_counts_no_scoring_pass_produces() {
        let ok = BundleVariant::CANDIDATES.map(|variant| Candidate {
            variant,
            hits: 5,
            discordant: 0,
        });
        assert!(SelectionEvidence::new(SelectionSample::Dev, 10, 5, ok).is_some());
        assert!(SelectionEvidence::new(SelectionSample::Dev, 0, 0, ok).is_none());
        assert!(SelectionEvidence::new(SelectionSample::Dev, 10, 11, ok).is_none());
        // Identity is copying, so its row may not disagree with the count.
        assert!(SelectionEvidence::new(SelectionSample::Dev, 10, 4, ok).is_none());

        let mut swapped = ok;
        swapped.swap(1, 2);
        assert!(SelectionEvidence::new(SelectionSample::Dev, 10, 5, swapped).is_none());

        let mut oversized = ok;
        oversized[4].hits = 11;
        assert!(SelectionEvidence::new(SelectionSample::Dev, 10, 5, oversized).is_none());
    }

    /// Consistent profiled evidence: hits and NER hits per candidate in
    /// lattice order, plus the full symmetric rivalry matrix whose identity
    /// row doubles as each candidate's against-copying discordance.
    fn profiled(
        sample: SelectionSample,
        tokens: u64,
        copied: u64,
        hits: [u64; 5],
        ner: [u64; 5],
        rivals: [[u64; 5]; 5],
    ) -> Option<SelectionEvidence> {
        let (candidates, profiles) = rows(hits, ner, rivals);
        SelectionEvidence::with_profiles(sample, tokens, copied, candidates, profiles)
    }

    fn rows(
        hits: [u64; 5],
        ner: [u64; 5],
        rivals: [[u64; 5]; 5],
    ) -> ([Candidate; 5], [CandidateProfile; 5]) {
        let mut candidates = Vec::new();
        let mut profiles = Vec::new();
        for (index, variant) in BundleVariant::CANDIDATES.into_iter().enumerate() {
            candidates.push(Candidate {
                variant,
                hits: hits[index],
                discordant: rivals[0][index],
            });
            profiles.push(CandidateProfile {
                ner_hits: ner[index],
                rivals: rivals[index],
            });
        }
        (
            candidates.try_into().expect("five candidates"),
            profiles.try_into().expect("five profiles"),
        )
    }

    #[test]
    fn a_statistical_tie_between_rivals_is_resolved_by_the_ner_profile() {
        // `rules` leads `lookup` by 2 hits on 40 discordant tokens — far
        // inside the pair's own error — and `full` repeats `lookup` exactly.
        // The old rule shipped the argmax, `rules`; the NER profile prefers
        // `lookup`, which keeps the external table the no-rules mode lives on,
        // and the exact `full` tie then falls back to the simpler `lookup`.
        let hits = [500, 600, 650, 648, 648];
        let ner = [500, 600, 600, 640, 640];
        let rivals = [
            [0, 120, 170, 160, 160],
            [120, 0, 60, 50, 50],
            [170, 60, 0, 40, 40],
            [160, 50, 40, 0, 10],
            [160, 50, 40, 10, 0],
        ];
        let evidence =
            profiled(SelectionSample::Dev, 1000, 500, hits, ner, rivals).expect("consistent");
        assert_eq!(evidence.verdict().0, BundleVariant::Lookup);

        // The same counts without profiles must keep the verdict the receipt
        // was written under: bare argmax, which ships `rules`.
        let (candidates, _profiles) = rows(hits, ner, rivals);
        let legacy = SelectionEvidence::new(SelectionSample::Dev, 1000, 500, candidates)
            .expect("consistent");
        assert_eq!(legacy.verdict().0, BundleVariant::Rules);
    }

    #[test]
    fn a_statistical_tie_with_equal_ner_falls_to_the_simpler_candidate() {
        // `rules` leads `corpus` by 3 hits on 50 discordant tokens and offers
        // the same NER profile — the pair the lattice cannot tell apart in
        // either mode ships the simpler configuration. `lookup` and `full`
        // repeat copying and are cut by the margin.
        let hits = [500, 600, 603, 500, 500];
        let ner = [500, 600, 600, 500, 500];
        let rivals = [
            [0, 120, 130, 0, 0],
            [120, 0, 50, 120, 120],
            [130, 50, 0, 130, 130],
            [0, 120, 130, 0, 0],
            [0, 120, 130, 0, 0],
        ];
        let evidence =
            profiled(SelectionSample::Dev, 1000, 500, hits, ner, rivals).expect("consistent");
        assert_eq!(evidence.verdict().0, BundleVariant::Corpus);
    }

    #[test]
    fn a_significant_pairwise_winner_is_never_displaced_by_the_ner_profile() {
        // `rules` beats `corpus` by 50 hits on 60 discordant tokens — about
        // 6.5 standard errors — so a better NER profile on the loser changes
        // nothing: the tie rule only ever touches the tie zone.
        let hits = [500, 600, 650, 500, 500];
        let ner = [500, 620, 600, 500, 500];
        let rivals = [
            [0, 120, 170, 0, 0],
            [120, 0, 60, 120, 120],
            [170, 60, 0, 170, 170],
            [0, 120, 170, 0, 0],
            [0, 120, 170, 0, 0],
        ];
        let profiled_verdict = profiled(SelectionSample::Dev, 1000, 500, hits, ner, rivals)
            .expect("consistent")
            .verdict();
        let (candidates, _profiles) = rows(hits, ner, rivals);
        let legacy_verdict = SelectionEvidence::new(SelectionSample::Dev, 1000, 500, candidates)
            .expect("consistent")
            .verdict();
        assert_eq!(profiled_verdict, legacy_verdict);
        assert_eq!(profiled_verdict.0, BundleVariant::Rules);
    }

    #[test]
    fn a_candidate_inside_the_copying_margin_ships_nothing_whatever_its_ner() {
        // A 5-token lead on 100 discordant tokens is noise against copying;
        // the margin cuts the candidate before any tie rule sees its NER row.
        let hits = [500, 505, 500, 500, 500];
        let ner = [500, 900, 500, 500, 500];
        let rivals = [
            [0, 100, 0, 0, 0],
            [100, 0, 100, 100, 100],
            [0, 100, 0, 0, 0],
            [0, 100, 0, 0, 0],
            [0, 100, 0, 0, 0],
        ];
        let evidence =
            profiled(SelectionSample::Dev, 1000, 500, hits, ner, rivals).expect("consistent");
        assert_eq!(evidence.verdict().0, BundleVariant::Identity);
    }

    #[test]
    fn candidate_profiles_reject_counts_no_scoring_pass_produces() {
        let hits = [500, 600, 603, 500, 500];
        let ner = [500, 600, 600, 500, 500];
        let rivals = [
            [0, 120, 130, 0, 0],
            [120, 0, 50, 120, 120],
            [130, 50, 0, 130, 130],
            [0, 120, 130, 0, 0],
            [0, 120, 130, 0, 0],
        ];
        assert!(profiled(SelectionSample::Dev, 1000, 500, hits, ner, rivals).is_some());

        // The matrix must be symmetric with a zero diagonal.
        let mut asymmetric = rivals;
        asymmetric[1][2] = 51;
        assert!(profiled(SelectionSample::Dev, 1000, 500, hits, ner, asymmetric).is_none());
        let mut diagonal = rivals;
        diagonal[2][2] = 1;
        assert!(profiled(SelectionSample::Dev, 1000, 500, hits, ner, diagonal).is_none());

        // A lead larger than the pair's discordance is impossible for two
        // configurations scored on the same tokens: `rules` leads `corpus`
        // by 3, so a pair count of 2 is a receipt nothing measured.
        let mut narrow = rivals;
        narrow[1][2] = 2;
        narrow[2][1] = 2;
        assert!(profiled(SelectionSample::Dev, 1000, 500, hits, ner, narrow).is_none());

        // NER hits are bounded by the sample, and identity's NER row is
        // copying by definition.
        let mut oversized = ner;
        oversized[3] = 1001;
        assert!(profiled(SelectionSample::Dev, 1000, 500, hits, oversized, rivals).is_none());
        let mut shifted = ner;
        shifted[0] = 501;
        assert!(profiled(SelectionSample::Dev, 1000, 500, hits, shifted, rivals).is_none());

        // The identity column of the matrix must repeat each candidate's
        // own against-copying counter.
        let (mut candidates, profiles) = rows(hits, ner, rivals);
        candidates[2].discordant = 131;
        assert!(
            SelectionEvidence::with_profiles(SelectionSample::Dev, 1000, 500, candidates, profiles)
                .is_none()
        );
    }

    #[test]
    fn every_selection_sample_name_round_trips_and_unknown_names_fail() {
        for sample in [SelectionSample::Dev, SelectionSample::TrainCv] {
            assert_eq!(SelectionSample::parse(sample.name()), Some(sample));
        }
        assert_eq!(SelectionSample::parse("test"), None);
        assert!(SelectionSample::TrainCv.strictness() > SelectionSample::Dev.strictness());
    }

    #[test]
    fn span_policy_names_round_trip_and_defaults_match_legacy_behaviour() {
        assert_eq!(SpanCase::default(), SpanCase::Transfer);
        assert!(SpanCase::default().transfers());
        for policy in [SpanCase::Transfer, SpanCase::Verbatim] {
            assert_eq!(SpanCase::parse(policy.name()), Some(policy));
        }
        assert_eq!(SpanCase::parse("copy"), None);

        assert_eq!(SpanMultiword::default(), SpanMultiword::Identity);
        assert_eq!(SpanMultiword::default().assembled_prefix(), Some(0));
        assert_eq!(SpanMultiword::Compose.assembled_prefix(), None);
        assert_eq!(SpanMultiword::Head.assembled_prefix(), Some(1));
        for policy in [
            SpanMultiword::Identity,
            SpanMultiword::Compose,
            SpanMultiword::Head,
        ] {
            assert_eq!(SpanMultiword::parse(policy.name()), Some(policy));
        }
        assert_eq!(SpanMultiword::parse("tokens"), None);

        assert_eq!(TokenCase::default(), TokenCase::Keep);
        for policy in [TokenCase::Keep, TokenCase::Surface] {
            assert_eq!(TokenCase::parse(policy.name()), Some(policy));
        }
        assert_eq!(TokenCase::parse("title"), None);
    }
}
