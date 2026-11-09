//! Stable `meta.json` rendering for one built lexicon.

use std::path::Path;

use super::{Built, Error, Result, Spec};
use crate::format::{MAX_SUFFIX_LENGTH, SelectionEvidence};
use crate::lexicon::{ENTITY_ENCODING, FORMAT_VERSION, render_meta};

/// A finite float as a JSON value; the shares and accuracies this module
/// writes are ratios in `[0, 1]`, so the `None` arm is unreachable.
fn float(value: f64) -> serde_json::Value {
    serde_json::Number::from_f64(value).map_or(serde_json::Value::Null, serde_json::Value::Number)
}

/// An optional value as JSON, `null` when absent — dataclass `asdict` shape.
fn optional<T>(value: Option<T>, encode: impl FnOnce(T) -> serde_json::Value) -> serde_json::Value {
    value.map_or(serde_json::Value::Null, encode)
}

/// The lattice receipt: what the selection sample measured, candidate by
/// candidate, so a reader can re-derive the shipped variant from the counts.
///
/// A candidate row carries its NER profile and its pairwise discordance
/// against every rival whenever the receipt measured them — which every
/// build's `measure` now does; the plain two-field row survives only for an
/// evidence value constructed without profiles.
fn selection(evidence: &SelectionEvidence) -> serde_json::Value {
    let profile_of = |index: usize| evidence.profiles().and_then(|profiles| profiles.get(index));
    let candidates: serde_json::Map<String, serde_json::Value> = evidence
        .candidates()
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            let mut row = serde_json::Map::new();
            row.insert("hits".into(), candidate.hits.into());
            row.insert("discordant".into(), candidate.discordant.into());
            if let Some(profile) = profile_of(index) {
                row.insert("ner_hits".into(), profile.ner_hits.into());
                let rivals: serde_json::Map<String, serde_json::Value> =
                    crate::format::BundleVariant::CANDIDATES
                        .iter()
                        .zip(profile.rivals)
                        .map(|(rival, count)| (rival.name().to_owned(), count.into()))
                        .collect();
                row.insert("rivals".into(), rivals.into());
            }
            (candidate.variant.name().to_owned(), row.into())
        })
        .collect();
    serde_json::json!({
        "on": evidence.sample().name(),
        "tokens": evidence.tokens(),
        "copied": evidence.copied(),
        "candidates": candidates,
    })
}

/// Writes `meta.json` exactly as the reference writer did: the dataclass's
/// field order, one-space indent, UTF-8 as is, `null` for absent values, and
/// a trailing newline.
pub(super) fn write_meta(
    spec: &Spec<'_>,
    built: &Built,
    destination: &Path,
    generation: &str,
) -> Result<()> {
    let mut meta = serde_json::Map::new();
    meta.insert("name".into(), spec.name.into());
    meta.insert("language".into(), spec.language.into());
    meta.insert("iso639_3".into(), spec.iso639_3.into());
    meta.insert("source".into(), built.source.as_str().into());
    meta.insert("training_tokens".into(), built.training_tokens.into());
    meta.insert("variant".into(), built.variant.as_str().into());
    meta.insert(
        "selected_on".into(),
        optional(built.selected_on, Into::into),
    );
    meta.insert("max_suffix".into(), MAX_SUFFIX_LENGTH.into());
    meta.insert("format_version".into(), FORMAT_VERSION.into());
    meta.insert("entity_encoding".into(), ENTITY_ENCODING.into());
    meta.insert("generation".into(), generation.into());
    meta.insert("normalization".into(), crate::lexicon::NORMALIZATION.into());
    meta.insert("files".into(), serde_json::Map::new().into());
    meta.insert("codecs".into(), serde_json::Map::new().into());
    meta.insert(
        "entries".into(),
        built
            .entries
            .iter()
            .map(|(name, count)| (name.clone(), serde_json::Value::from(*count)))
            .collect::<serde_json::Map<_, _>>()
            .into(),
    );
    meta.insert(
        "traffic".into(),
        built
            .traffic
            .iter()
            .map(|(name, share)| ((*name).to_owned(), float(*share)))
            .collect::<serde_json::Map<_, _>>()
            .into(),
    );
    meta.insert(
        "gold_tokens".into(),
        optional(built.gold_tokens, Into::into),
    );
    meta.insert("accuracy".into(), optional(built.accuracy, float));
    meta.insert("baseline".into(), optional(built.baseline, float));
    meta.insert(
        "beats_baseline".into(),
        optional(built.beats_baseline, Into::into),
    );
    meta.insert(
        "selection".into(),
        optional(built.selection.as_ref(), selection),
    );
    // Span conventions are written only when the applied acceptance policy
    // deviates from the fail-closed default, so an unmeasured lexicon's
    // metadata does not change shape.
    if let Some(value) = spec.span_case {
        meta.insert("span_case".into(), value.into());
    }
    if let Some(value) = spec.span_multiword {
        meta.insert("span_multiword".into(), value.into());
    }
    if spec.entity_lower {
        meta.insert("entity_lower".into(), true.into());
    }
    if let Some(marker) = spec.compound_marker {
        meta.insert("compound_marker".into(), marker.to_string().into());
    }
    if let Some(value) = spec.token_case {
        meta.insert("token_case".into(), value.into());
    }
    if spec.lemma_adapter {
        meta.insert("lemma_adapter".into(), true.into());
    }
    if spec.context_keys {
        meta.insert("context_keys".into(), true.into());
    }

    let path = destination.join("meta.json");
    std::fs::write(&path, render_meta(&serde_json::Value::Object(meta))).map_err(|source| {
        Error::Io {
            path: path.clone(),
            source,
        }
    })
}
