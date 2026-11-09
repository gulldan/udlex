use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::codec::Codec;
use super::{ENTITY_ENCODING, Error, NORMALIZATION, Result};
use crate::format::{
    BundleVariant, Candidate, CandidateProfile, FormatRange, LookupTableKind, SelectionEvidence,
    SelectionSample, SpanCase, SpanMultiword, TokenCase,
};
use crate::secure_fs::valid_generation_id;

/// What a bundle written before `variant` existed necessarily contained.
fn default_variant() -> String {
    BundleVariant::Full.name().to_owned()
}

/// Size and cryptographic digest of one file in an immutable generation.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct FileIntegrity {
    /// Exact byte length.
    pub size: u64,
    /// Lowercase hexadecimal SHA-256 digest.
    pub sha256: String,
}

/// One lattice candidate as `meta.json` records it.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct CandidateRecord {
    /// Tokens of the selection sample it answered exactly right.
    pub hits: u64,
    /// Tokens on which it and plain copying disagree about being right.
    pub discordant: u64,
    /// Tokens answered exactly right with suffix rules disabled — the NER
    /// profile; present together with `rivals` or not at all.
    #[serde(default)]
    pub ner_hits: Option<u64>,
    /// Discordance against every lattice candidate, keyed by variant name.
    #[serde(default)]
    pub rivals: Option<BTreeMap<String, u64>>,
}

/// The whole lattice receipt as `meta.json` records it.
#[derive(Debug, Clone, Deserialize)]
pub struct SelectionRecord {
    /// Which sample: `dev` or `train-cv`.
    pub on: String,
    /// How many scorable tokens it held.
    pub tokens: u64,
    /// How many of them plain copying already answered right.
    pub copied: u64,
    /// Every candidate, keyed by variant name.
    pub candidates: BTreeMap<String, CandidateRecord>,
}

/// Provenance recorded alongside a lexicon.
#[derive(Debug, Clone, Deserialize)]
pub struct Meta {
    /// The lexicon identifier, normally a Universal Dependencies treebank code.
    pub name: String,
    /// Human-readable language name.
    pub language: String,
    /// ISO 639-3 code of the language.
    #[serde(rename = "iso639_3")]
    pub iso639_3: String,
    /// What the lexicon was learned from.
    pub source: String,
    /// How many annotated tokens went into it.
    pub training_tokens: u64,
    /// Which parts of the learned lexicon this bundle ships: `full`, `lookup`
    /// (no suffix rules), `rules` (no external vocabulary), `corpus` (neither),
    /// or `identity` (nothing beat copying; an entity table may remain).
    #[serde(default = "default_variant")]
    pub variant: String,
    /// Which data chose that variant: `dev`, form-disjoint `train-cv`, or
    /// absent when no independent selection was possible; never `test`.
    #[serde(default)]
    pub selected_on: Option<String>,
    /// Longest suffix indexed by the rule table.
    pub max_suffix: usize,
    /// Version of the on-disk layout, as this bundle declares it.
    pub format_version: u32,
    /// Exact encoding contract for the shared exact/lower entity table;
    /// optional only so a format-3 bundle gets an actionable error.
    #[serde(default)]
    pub entity_encoding: Option<String>,
    /// Immutable generation identifier named by the root's `current` pointer.
    #[serde(default)]
    pub generation: Option<String>,
    /// Unicode normal form used by every stored key and string.
    #[serde(default)]
    pub normalization: String,
    /// Runtime files tied to this generation by size and SHA-256.
    #[serde(default)]
    pub files: BTreeMap<String, FileIntegrity>,
    /// How each compiled lookup table is encoded, keyed by table name; a
    /// missing entry means the double array.
    #[serde(default)]
    pub codecs: BTreeMap<String, String>,
    /// How many rows each table holds, keyed by table name.
    #[serde(default)]
    pub entries: BTreeMap<String, u64>,
    /// The share of running-text tokens that reaches each lookup table,
    /// measured on held-out text; the compiler weighs codecs against it.
    #[serde(default)]
    pub traffic: BTreeMap<String, f64>,
    /// How many gold tokens the lexicon was scored on, when a test split
    /// existed at build time.
    #[serde(default)]
    pub gold_tokens: Option<u64>,
    /// Exact-match accuracy on those tokens.
    #[serde(default)]
    pub accuracy: Option<f64>,
    /// Accuracy of copying the surface form on the same tokens.
    #[serde(default)]
    pub baseline: Option<f64>,
    /// Whether the lexicon beat copying the surface form on gold data;
    /// `None` without gold data.
    #[serde(default)]
    pub beats_baseline: Option<bool>,
    /// What the selection sample measured for every lattice candidate; the
    /// reader re-derives `variant` from it.
    #[serde(default)]
    pub selection: Option<SelectionRecord>,
    /// Span-path case convention: `transfer` (default when absent) carries
    /// surface casing onto all-lowercase answers, `verbatim` returns table
    /// values untouched.
    #[serde(default)]
    pub span_case: Option<String>,
    /// Span-path multiword convention: `identity` (default when absent),
    /// `compose` or `head`.
    #[serde(default)]
    pub span_multiword: Option<String>,
    /// Whether the measured entity policy opened the lowercase entity retry;
    /// the default of [`crate::Lexicon::span_options`].
    #[serde(default)]
    pub entity_lower: bool,
    /// Token-path case convention: `keep` (default when absent) or `surface`,
    /// which copies surface casing onto the shared prefix.
    #[serde(default)]
    pub token_case: Option<String>,
}

impl Meta {
    /// Whether the shipped configuration beat plain copying on the data that
    /// judged it: gold data if any, else the selection receipt, else unknown.
    #[must_use]
    pub fn useful(&self) -> Option<bool> {
        self.beats_baseline.or_else(|| {
            let record = self.selection.as_ref()?;
            let shipped = record.candidates.get(&self.variant)?;
            Some(shipped.hits >= record.copied)
        })
    }

    /// The span conventions this metadata declares, defaults filled in.
    #[must_use]
    pub fn span_conventions(&self) -> crate::probe::SpanConventions {
        let case = self
            .span_case
            .as_deref()
            .and_then(SpanCase::parse)
            .unwrap_or_default();
        let multiword = self
            .span_multiword
            .as_deref()
            .and_then(SpanMultiword::parse)
            .unwrap_or_default();
        crate::probe::SpanConventions {
            case_transfer: case.transfers(),
            multiword,
        }
    }
}

/// Validates metadata fields that do not require opening the table files.
#[allow(clippy::too_many_lines)]
pub(super) fn validate_meta(
    meta: &Meta,
    path: &Path,
    expected_generation: Option<&str>,
) -> Result<()> {
    if !FormatRange::READ.accepts(meta.format_version) {
        return Err(Error::Format {
            path: path.to_owned(),
            found: meta.format_version,
            supported: FormatRange::READ,
        });
    }
    if meta.entity_encoding.as_deref() != Some(ENTITY_ENCODING) {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: format!(
                "entity_encoding must be {ENTITY_ENCODING:?}; legacy format-3 entity tables are ambiguous and must be rebuilt"
            ),
        });
    }
    let valid_iso =
        meta.iso639_3.len() == 3 && meta.iso639_3.bytes().all(|byte| byte.is_ascii_lowercase());
    if meta.name.is_empty() || meta.language.is_empty() || meta.source.is_empty() || !valid_iso {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: "name/language/source must be non-empty and iso639_3 must be exactly three ASCII lowercase letters"
                .to_owned(),
        });
    }
    if meta.normalization != NORMALIZATION {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: format!(
                "normalization is {:?}, expected {NORMALIZATION:?}",
                meta.normalization
            ),
        });
    }
    let Some(variant) = BundleVariant::parse(&meta.variant) else {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: format!("unknown variant {:?}", meta.variant),
        });
    };
    if meta.max_suffix > crate::format::MAX_SUFFIX_LENGTH {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: format!(
                "max_suffix {} exceeds supported limit {}",
                meta.max_suffix,
                crate::format::MAX_SUFFIX_LENGTH
            ),
        });
    }
    if meta
        .selected_on
        .as_deref()
        .is_some_and(|value| !matches!(value, "dev" | "train-cv"))
    {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: format!("unknown selected_on value {:?}", meta.selected_on),
        });
    }
    if meta
        .span_case
        .as_deref()
        .is_some_and(|value| SpanCase::parse(value).is_none())
    {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: format!("unknown span_case value {:?}", meta.span_case),
        });
    }
    if meta
        .span_multiword
        .as_deref()
        .is_some_and(|value| SpanMultiword::parse(value).is_none())
    {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: format!("unknown span_multiword value {:?}", meta.span_multiword),
        });
    }
    if meta
        .token_case
        .as_deref()
        .is_some_and(|value| TokenCase::parse(value).is_none())
    {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: format!("unknown token_case value {:?}", meta.token_case),
        });
    }
    let Some(generation) = meta.generation.as_deref() else {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: "format-3 metadata must name its immutable generation".to_owned(),
        });
    };
    if !valid_generation_id(generation) {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: format!("invalid metadata generation identifier {generation:?}"),
        });
    }

    validate_measurements(meta, path)?;
    validate_selection(meta, path, variant)?;
    validate_table_manifest(meta, path, variant)?;
    if let Some(expected) = expected_generation
        && meta.generation.as_deref() != Some(expected)
    {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: format!(
                "metadata generation {:?} does not match current {expected:?}",
                meta.generation
            ),
        });
    }
    Ok(())
}

/// Validates held-out measurements independently from the storage layout.
fn validate_measurements(meta: &Meta, path: &Path) -> Result<()> {
    for (name, value) in &meta.traffic {
        let valid_table = [LookupTableKind::Lower, LookupTableKind::External]
            .into_iter()
            .any(|table| table.name() == name);
        if !valid_table || !value.is_finite() || !(0.0..=1.0).contains(value) {
            return Err(Error::Invalid {
                path: path.to_owned(),
                message: format!("invalid traffic entry {name:?}: {value}"),
            });
        }
    }
    for (name, value) in [("accuracy", meta.accuracy), ("baseline", meta.baseline)] {
        if value.is_some_and(|number| !number.is_finite() || !(0.0..=1.0).contains(&number)) {
            return Err(Error::Invalid {
                path: path.to_owned(),
                message: format!("invalid {name}: {value:?}"),
            });
        }
    }
    if meta.accuracy.is_some() != meta.baseline.is_some()
        || meta.accuracy.is_some() != meta.gold_tokens.is_some()
    {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: "gold_tokens, accuracy and baseline must be present together".to_owned(),
        });
    }
    if meta.gold_tokens == Some(0) {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: "gold_tokens must be positive when measurements are present".to_owned(),
        });
    }
    if meta.beats_baseline.is_some() != meta.accuracy.is_some() {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: "beats_baseline must be present exactly with held-out measurements".to_owned(),
        });
    }
    if let (Some(accuracy), Some(baseline), Some(beats_baseline)) =
        (meta.accuracy, meta.baseline, meta.beats_baseline)
        && beats_baseline != (accuracy >= baseline)
    {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: "beats_baseline contradicts accuracy and baseline".to_owned(),
        });
    }
    Ok(())
}

/// Re-derives the shipped variant from the lattice receipt and refuses a
/// bundle whose `variant` disagrees.
fn validate_selection(meta: &Meta, path: &Path, variant: BundleVariant) -> Result<()> {
    let Some(record) = meta.selection.as_ref() else {
        return Ok(());
    };
    let invalid = |message: String| Error::Invalid {
        path: path.to_owned(),
        message,
    };
    if meta.selected_on.as_deref() != Some(record.on.as_str()) {
        return Err(invalid(format!(
            "selection.on {:?} does not match selected_on {:?}",
            record.on, meta.selected_on
        )));
    }
    let Some(sample) = SelectionSample::parse(&record.on) else {
        return Err(invalid(format!(
            "unknown selection.on value {:?}",
            record.on
        )));
    };
    let mut candidates = Vec::with_capacity(BundleVariant::CANDIDATES.len());
    let mut profiles = Vec::with_capacity(BundleVariant::CANDIDATES.len());
    for expected in BundleVariant::CANDIDATES {
        let Some(counted) = record.candidates.get(expected.name()) else {
            return Err(invalid(format!(
                "selection.candidates is missing {:?}",
                expected.name()
            )));
        };
        candidates.push(Candidate {
            variant: expected,
            hits: counted.hits,
            discordant: counted.discordant,
        });
        profiles.push(candidate_profile(counted).map_err(invalid)?);
    }
    if record.candidates.len() != candidates.len() {
        return Err(invalid(
            "selection.candidates names something outside the lattice".to_owned(),
        ));
    }
    // One scoring pass profiles every candidate or none.
    let recorded = profiles.iter().filter(|profile| profile.is_some()).count();
    if recorded != 0 && recorded != profiles.len() {
        return Err(invalid(
            "selection profiles must cover every candidate or none".to_owned(),
        ));
    }
    let counted: [Candidate; BundleVariant::CANDIDATES.len()] = candidates
        .try_into()
        .map_err(|_| invalid("selection counts are not internally consistent".to_owned()))?;
    let evidence = if recorded == 0 {
        SelectionEvidence::new(sample, record.tokens, record.copied, counted)
    } else {
        let complete: Vec<CandidateProfile> = profiles.into_iter().flatten().collect();
        complete.try_into().ok().and_then(|complete| {
            SelectionEvidence::with_profiles(
                sample,
                record.tokens,
                record.copied,
                counted,
                complete,
            )
        })
    }
    .ok_or_else(|| invalid("selection counts are not internally consistent".to_owned()))?;
    let (chosen, _hits, _lead) = evidence.verdict();
    if chosen != variant {
        return Err(invalid(format!(
            "selection evidence chooses {:?}, metadata ships {:?}",
            chosen.name(),
            variant.name()
        )));
    }
    Ok(())
}

/// One candidate's recorded profile, `Ok(None)` when the receipt predates it.
fn candidate_profile(
    counted: &CandidateRecord,
) -> std::result::Result<Option<CandidateProfile>, String> {
    match (counted.ner_hits, counted.rivals.as_ref()) {
        (None, None) => Ok(None),
        (Some(ner_hits), Some(rivals)) => {
            let mut row = Vec::with_capacity(BundleVariant::CANDIDATES.len());
            for rival in BundleVariant::CANDIDATES {
                let Some(count) = rivals.get(rival.name()) else {
                    return Err(format!("selection rivals are missing {:?}", rival.name()));
                };
                row.push(*count);
            }
            if rivals.len() != row.len() {
                return Err("selection rivals name something outside the lattice".to_owned());
            }
            let rivals = row
                .try_into()
                .map_err(|_| "selection rivals must spell the lattice exactly".to_owned())?;
            Ok(Some(CandidateProfile { ner_hits, rivals }))
        }
        _ => Err("selection ner_hits and rivals must be present together".to_owned()),
    }
}

/// Validates that the declared variant, table counts, and codecs agree.
fn validate_table_manifest(meta: &Meta, path: &Path, variant: BundleVariant) -> Result<()> {
    let fixed_entry_keys = ["phrases", "rules", "strings"];
    if meta.entries.len()
        != LookupTableKind::ALL
            .len()
            .saturating_add(fixed_entry_keys.len())
        || !LookupTableKind::ALL
            .into_iter()
            .all(|table| meta.entries.contains_key(table.name()))
        || !fixed_entry_keys
            .into_iter()
            .all(|name| meta.entries.contains_key(name))
    {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message:
                "entries must contain exactly exact/lower/external/entities/phrases/rules/strings"
                    .to_owned(),
        });
    }
    let count = |name: &str| meta.entries.get(name).copied().unwrap_or(0);
    if LookupTableKind::ALL
        .into_iter()
        .map(|table| u128::from(count(table.name())))
        .sum::<u128>()
        > u128::from(u64::MAX)
    {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: "aggregate lookup entry count exceeds u64".to_owned(),
        });
    }
    if !variant.entries_compatible(
        count(LookupTableKind::Exact.name()),
        count(LookupTableKind::Lower.name()),
        count(LookupTableKind::External.name()),
        count("rules"),
    ) {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: format!("entries contradict variant {:?}", meta.variant),
        });
    }
    // `phrases` and `rules` have one fixed encoding, so no codec is recorded.
    for (name, codec) in &meta.codecs {
        if !LookupTableKind::ALL
            .into_iter()
            .any(|table| table.name() == name)
            || Codec::parse(codec).is_none()
        {
            return Err(Error::Invalid {
                path: path.to_owned(),
                message: format!("invalid codec {name:?}: {codec:?}"),
            });
        }
    }
    for table in LookupTableKind::ALL {
        let name = table.name();
        if !meta.files.is_empty() && (count(name) > 0) != meta.codecs.contains_key(name) {
            return Err(Error::Invalid {
                path: path.to_owned(),
                message: format!("codec presence disagrees with {name} entry count"),
            });
        }
    }
    Ok(())
}
