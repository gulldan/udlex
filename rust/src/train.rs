//! Learning a lexicon from annotated text, and deciding what of it to ship.
//!
//! Four steps, in the order the builder runs them:
//!
//! 1. [`build`] turns `(form, lemma)` observations into the four tables.
//! 2. [`tune`] fits the two confidence thresholds on a development split.
//! 3. [`choose`] scores five fixed configurations and keeps the best.
//! 4. [`score`] reports what the winner is worth against held-out gold data.
//!
//! Steps 2–4 lemmatise through [`crate::probe`], so a dictionary is chosen
//! under exactly the rules it is later read under.
//!
//! # Ties
//!
//! Ties break as the original Python implementation broke them, or published
//! tables change silently:
//!
//! * `Counter.most_common(1)` returns the **first inserted** of the tied keys.
//! * `max(iterable, key=…)` returns the **first** maximal element.
//! * `round(x, 3)` rounds half to even, which `format!("{x:.3}")` also does —
//!   verified equal on every `hits/total` a rule can carry.
//!
//! Insertion order is therefore load-bearing, which is why the maps here are
//! [`IndexMap`](indexmap::IndexMap) and not [`HashMap`](std::collections::HashMap).

mod building;
mod evaluation;
mod model;
mod view;
mod writing;

pub use building::ExternalResolver;
pub use building::build_for_language_with_resolved_external;
pub use building::into_most_common;
pub use building::refit_for_language;
pub use building::refit_for_language_with_entities;
pub use building::{
    build, build_for_language, build_for_language_with_entities, build_with_entities,
};
pub use evaluation::traffic_iter;
pub use evaluation::{choose, measure, score, traffic, tune};
pub use model::{Graded, Pair, Score, Tables, Thresholds, Trained, Variant};
pub use view::View;
pub use writing::{write_phrase_source, write_sources};

pub use crate::format::MAX_SUFFIX_LENGTH;

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests;
