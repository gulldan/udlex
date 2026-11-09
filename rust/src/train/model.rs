use indexmap::IndexMap;

use crate::normalization::Casing;
use crate::rule::{CaseAction, CaseKey};

/// A rule as the builder holds it, with its confidence; compilation turns a
/// rule below threshold into [`CaseAction::Stop`].
#[derive(Debug, Clone, PartialEq)]
pub struct Trained {
    /// How many characters to remove from the end of the lowercased form.
    pub cut: usize,
    /// What to append after cutting.
    pub add: String,
    /// What to do with the capitalisation afterwards.
    pub case: CaseAction,
    /// Share of training pairs under the same key that this rule got right.
    pub confidence: f64,
}

/// Confidence a suffix rule must reach before it is allowed to fire.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Thresholds {
    /// Applied to lowercase and sentence-initial forms.
    pub general: f64,
    /// Applied to capitalised mid-sentence forms, which are almost all proper
    /// nouns a rule would destroy.
    pub proper_noun: f64,
}

impl Thresholds {
    /// The threshold governing one rule bucket.
    pub(super) const fn for_bucket(self, bucket: u8) -> f64 {
        if bucket == CaseKey::MidSentenceCapital.byte() {
            self.proper_noun
        } else {
            self.general
        }
    }

    /// Whether both confidence gates are finite, non-negative probabilities,
    /// or the documented `2.0` sentinel that disables a bucket.
    #[must_use]
    pub fn is_valid(self) -> bool {
        let valid = |value: f64| value.is_finite() && (0.0..=2.0).contains(&value);
        valid(self.general) && valid(self.proper_noun)
    }
}

/// One observation used to build a lexicon.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Pair {
    /// The surface form.
    pub form: String,
    /// Its gold lemma.
    pub lemma: String,
    /// Whether the form opened its sentence.
    pub sentence_initial: bool,
}

/// The lookup tables, as the builder holds them before shipping.
#[derive(Debug, Default)]
pub struct Tables {
    /// Language-aware case mapping shared by training and lookup.
    pub(super) casing: Casing,
    /// Surface form to lemma, case sensitive.
    pub exact: IndexMap<String, String>,
    /// Lowercased surface form to lemma.
    pub lower: IndexMap<String, String>,
    /// Form to lemma harvested from a dictionary rather than the corpus.
    pub external: IndexMap<String, String>,
    /// Curated named-entity forms, isolated from common vocabulary.
    pub entities: IndexMap<String, String>,
    /// Locale-lowercased entity aliases, isolated from exact entity lookups.
    pub entities_lower: IndexMap<String, String>,
    /// Case bucket and suffix, to the rule learned for it.
    pub rules: IndexMap<String, Trained>,
}

/// Which parts of a learned lexicon are kept.
pub use crate::format::BundleVariant as Variant;

/// One gold token, as scoring and tuning need it.
#[derive(Debug, Clone)]
pub struct Graded {
    /// The surface form.
    pub form: String,
    /// Its gold lemma, or `None` where the corpus left the position unannotated.
    pub lemma: Option<String>,
    /// Whether the form opened its sentence.
    pub sentence_initial: bool,
}

/// Exact-match accuracy of a lemmatiser and of the trivial baseline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Score {
    /// How many gold tokens carried a lemma worth scoring.
    pub tokens: u64,
    /// Share the lemmatiser got exactly right.
    pub accuracy: f64,
    /// Share that copying the surface form would have got right.
    pub baseline: f64,
}
