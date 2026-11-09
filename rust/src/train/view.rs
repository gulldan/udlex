use std::borrow::Cow;
use std::cell::Cell;

use crate::probe::{self, Options};
use crate::rule::{CaseAction, Rule};

use super::{MAX_SUFFIX_LENGTH, Tables, Thresholds, Variant};

/// One configuration of one set of tables, ready to lemmatise; a variant
/// refuses to answer from the tables it does not ship.
#[derive(Debug)]
pub struct View<'a> {
    tables: &'a Tables,
    thresholds: Thresholds,
    variant: Variant,
    rule_observer: Option<&'a Cell<Option<(u8, f64)>>>,
}

impl<'a> View<'a> {
    /// Restricts `tables` to what `variant` ships.
    #[must_use]
    pub const fn new(tables: &'a Tables, thresholds: Thresholds, variant: Variant) -> Self {
        Self {
            tables,
            thresholds,
            variant,
            rule_observer: None,
        }
    }

    /// Builds the same view while exposing the longest suffix rule the probe
    /// reached, so tuning scores every threshold from one probe.
    pub(super) const fn with_rule_observer(
        tables: &'a Tables,
        thresholds: Thresholds,
        variant: Variant,
        observer: &'a Cell<Option<(u8, f64)>>,
    ) -> Self {
        Self {
            tables,
            thresholds,
            variant,
            rule_observer: Some(observer),
        }
    }

    /// The lemma of one token.
    #[must_use]
    pub fn lemma<'t>(&'t self, token: &'t str, sentence_initial: bool) -> Cow<'t, str>
    where
        'a: 't,
    {
        probe::lemma(self, token, sentence_initial, Options::default())
    }
}

impl probe::Tables for View<'_> {
    fn exact(&self, key: &str) -> Option<Cow<'_, str>> {
        if !self.variant.keeps_corpus() {
            return None;
        }
        self.tables
            .exact
            .get(key)
            .map(|found| Cow::Borrowed(&**found))
    }

    fn lower(&self, key: &str) -> Option<Cow<'_, str>> {
        if !self.variant.keeps_corpus() {
            return None;
        }
        self.tables
            .lower
            .get(key)
            .map(|found| Cow::Borrowed(&**found))
    }

    fn external(&self, key: &str) -> Option<Cow<'_, str>> {
        if !self.variant.keeps_external() {
            return None;
        }
        self.tables
            .external
            .get(key)
            .map(|found| Cow::Borrowed(&**found))
    }

    fn entities(&self, key: &str) -> Option<Cow<'_, str>> {
        self.tables
            .entities
            .get(key)
            .map(|found| Cow::Borrowed(&**found))
    }

    fn entities_lower(&self, key: &str) -> Option<Cow<'_, str>> {
        self.tables
            .entities_lower
            .get(key)
            .map(|found| Cow::Borrowed(&**found))
    }

    /// A rule below its threshold comes back as a stop marker, as compiled.
    fn rule(&self, bucket: u8, suffix: &str) -> Option<(Rule, Cow<'_, str>)> {
        if !self.variant.keeps_rules() {
            return None;
        }
        let mut key = String::with_capacity(suffix.len().saturating_add(1));
        key.push(bucket as char);
        key.push_str(suffix);
        let trained = self.tables.rules.get(&key)?;
        if let Some(observer) = self.rule_observer {
            observer.set(Some((bucket, trained.confidence)));
        }
        let case = if trained.confidence < self.thresholds.for_bucket(bucket) {
            CaseAction::Stop
        } else {
            trained.case
        };
        let rule = Rule {
            cut: trained.cut,
            add: 0,
            case,
        };
        Some((rule, Cow::Borrowed(&trained.add)))
    }

    fn is_empty(&self) -> bool {
        self.variant == Variant::Identity
            && self.tables.entities.is_empty()
            && self.tables.entities_lower.is_empty()
    }

    fn common_is_empty(&self) -> bool {
        (!self.variant.keeps_corpus()
            || (self.tables.exact.is_empty() && self.tables.lower.is_empty()))
            && (!self.variant.keeps_external() || self.tables.external.is_empty())
            && (!self.variant.keeps_rules() || self.tables.rules.is_empty())
    }

    fn max_suffix(&self) -> usize {
        if self.variant.keeps_rules() {
            MAX_SUFFIX_LENGTH
        } else {
            0
        }
    }

    fn casing(&self) -> crate::normalization::Casing {
        self.tables.casing
    }
}
