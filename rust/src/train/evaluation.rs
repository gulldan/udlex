use std::cell::Cell;

use crate::format::{Candidate, CandidateProfile, SelectionEvidence, SelectionSample};
use crate::normalization::nfc;
use crate::probe::{self, Source};
use crate::rule::{CaseKey, CaseKey::MidSentenceCapital};

use super::{Graded, Score, Tables, Thresholds, Variant, View};

/// Thresholds tried during tuning. `2.0` means "never fire a rule".
const CONFIDENCE_GRID: [f64; 8] = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 2.0];

/// Measures one configuration against annotated tokens.
///
/// Positions the corpus left unannotated are skipped, so they count neither for
/// nor against anyone. Returns `None` when nothing is scorable.
#[must_use]
pub fn score(view: &View<'_>, gold: &[Graded]) -> Option<Score> {
    let (mut tokens, mut hits, mut copied) = (0_u64, 0_u64, 0_u64);
    for token in gold {
        let Some(lemma) = token.lemma.as_deref() else {
            continue;
        };
        tokens = tokens.saturating_add(1);
        let expected = nfc(lemma);
        if nfc(&view.lemma(&token.form, token.sentence_initial)) == expected {
            hits = hits.saturating_add(1);
        }
        if nfc(&token.form) == expected {
            copied = copied.saturating_add(1);
        }
    }
    if tokens == 0 {
        return None;
    }
    Some(Score {
        tokens,
        accuracy: hits as f64 / tokens as f64,
        baseline: copied as f64 / tokens as f64,
    })
}

#[derive(Clone, Copy)]
struct AccuracyTrace {
    rule: Option<(u8, f64)>,
    active_correct: bool,
    stopped_correct: bool,
    /// Whether this token falls in the bucket the proper-noun gate governs.
    mid_sentence_capital: bool,
}

impl AccuracyTrace {
    fn correct_at(self, thresholds: Thresholds) -> bool {
        self.rule
            .map_or(self.active_correct, |(bucket, confidence)| {
                // The View predicate verbatim, NaN behaviour included.
                if confidence < thresholds.for_bucket(bucket) {
                    self.stopped_correct
                } else {
                    self.active_correct
                }
            })
    }
}

/// Records everything threshold tuning can change in one real probe.
///
/// A confidence gate affects only the first (longest) suffix rule the shared
/// probe finds. Below the gate that rule answers; above it the stop marker
/// returns the original token. Lookup answers never depend on the threshold.
fn accuracy_traces(tables: &Tables, development: &[Graded]) -> Vec<AccuracyTrace> {
    let observer = Cell::new(None);
    let view = View::with_rule_observer(tables, Thresholds::default(), Variant::Full, &observer);
    development
        .iter()
        .filter_map(|token| {
            let lemma = token.lemma.as_deref()?;
            observer.set(None);
            let answer = view.lemma(&token.form, token.sentence_initial);
            let gold = nfc(lemma);
            Some(AccuracyTrace {
                rule: observer.get(),
                active_correct: nfc(&answer) == gold,
                stopped_correct: nfc(&token.form) == gold,
                mid_sentence_capital: CaseKey::of(&token.form, token.sentence_initial)
                    == MidSentenceCapital,
            })
        })
        .collect()
}

/// Fits both confidence thresholds on a development split.
///
/// The general threshold maximises overall accuracy; the proper-noun one
/// maximises accuracy on the bucket it governs — every capitalised
/// mid-sentence token. Both fall back to zero when the split holds nothing
/// scorable.
///
/// The proper-noun gate is fitted on the whole [`MidSentenceCapital`] bucket,
/// the population it decides at lookup time, not on `PROPN` tags.
#[must_use]
pub fn tune(tables: &Tables, development: &[Graded]) -> Thresholds {
    let traces = accuracy_traces(tables, development);
    if traces.is_empty() {
        return Thresholds::default();
    }

    // `max(GRID, key=(accuracy, -value))`: among equal accuracies the smallest
    // threshold wins, and the grid is ascending, so a plain `>` gets there.
    let mut general = CONFIDENCE_GRID[0];
    let mut best = f64::NEG_INFINITY;
    for value in CONFIDENCE_GRID {
        let thresholds = Thresholds {
            general: value,
            proper_noun: value,
        };
        let hits = traces
            .iter()
            .filter(|trace| trace.correct_at(thresholds))
            .count();
        let measured = hits as f64 / traces.len() as f64;
        if measured > best {
            best = measured;
            general = value;
        }
    }

    let bucket_count = traces
        .iter()
        .filter(|trace| trace.mid_sentence_capital)
        .count();
    if bucket_count < 50 {
        return Thresholds {
            general,
            proper_noun: general,
        };
    }

    let mut proper_noun = CONFIDENCE_GRID[0];
    let mut best = f64::NEG_INFINITY;
    for value in CONFIDENCE_GRID {
        let thresholds = Thresholds {
            general,
            proper_noun: value,
        };
        let hits = traces
            .iter()
            .filter(|trace| trace.mid_sentence_capital && trace.correct_at(thresholds))
            .count();
        let measured = hits as f64 / bucket_count as f64;
        if measured > best {
            best = measured;
            proper_noun = value;
        }
    }
    Thresholds {
        general,
        proper_noun,
    }
}

/// Scores every candidate configuration against one selection sample.
///
/// Per candidate: hits, discordance against copying and against every rival
/// (for [`SelectionEvidence::verdict`]), and the NER profile with
/// `use_rules = false`, all read off one shared probe per token (a brute-force
/// test holds the shortcut to the definition). `None` when nothing is scorable.
#[must_use]
pub fn measure(
    tables: &Tables,
    thresholds: Thresholds,
    gold: &[Graded],
    sample: SelectionSample,
) -> Option<SelectionEvidence> {
    let full = View::new(tables, thresholds, Variant::Full);
    let corpus_rules = View::new(tables, thresholds, Variant::Rules);
    let mut hits = [0_u64; Variant::CANDIDATES.len()];
    let mut ner_hits = [0_u64; Variant::CANDIDATES.len()];
    let mut discordant = [0_u64; Variant::CANDIDATES.len()];
    let mut rivals = [[0_u64; Variant::CANDIDATES.len()]; Variant::CANDIDATES.len()];
    let mut copied = 0_u64;
    let mut tokens = 0_u64;
    for token in gold {
        let Some(lemma) = token.lemma.as_deref() else {
            continue;
        };
        tokens = tokens.saturating_add(1);
        let expected = nfc(lemma);
        let identity_correct = nfc(&token.form) == expected;
        copied = copied.saturating_add(u64::from(identity_correct));

        let answer = probe::lemma_with_info(
            &full,
            &token.form,
            token.sentence_initial,
            probe::Options::default(),
        );
        let full_correct = nfc(&answer.lemma) == expected;
        let (correct, ner_correct) =
            candidate_rows(answer.source, identity_correct, full_correct, || {
                let rules_answer = probe::lemma_with_info(
                    &corpus_rules,
                    &token.form,
                    token.sentence_initial,
                    probe::Options::default(),
                );
                nfc(&rules_answer.lemma) == expected
            });
        for (((total, ner), apart), (main, without_rules)) in hits
            .iter_mut()
            .zip(ner_hits.iter_mut())
            .zip(discordant.iter_mut())
            .zip(correct.into_iter().zip(ner_correct))
        {
            *total = total.saturating_add(u64::from(main));
            *ner = ner.saturating_add(u64::from(without_rules));
            *apart = apart.saturating_add(u64::from(main != identity_correct));
        }
        for (row, one) in rivals.iter_mut().zip(correct) {
            for (cell, other) in row.iter_mut().zip(correct) {
                *cell = cell.saturating_add(u64::from(one != other));
            }
        }
    }
    let counted = Variant::CANDIDATES
        .into_iter()
        .zip(hits)
        .zip(discordant)
        .map(|((variant, hits), discordant)| Candidate {
            variant,
            hits,
            discordant,
        });
    let profiled = ner_hits
        .into_iter()
        .zip(rivals)
        .map(|(ner_hits, rivals)| CandidateProfile { ner_hits, rivals });
    SelectionEvidence::with_profiles(sample, tokens, copied, array(counted)?, array(profiled)?)
}

/// Both correctness rows of one token — rules on, then rules off — read from
/// the stage that answered the shared full probe.
///
/// `rules_correct` is the one extra probe, needed only after an external hit,
/// for the `rules` candidate that lacks the external table.
fn candidate_rows(
    source: Source,
    identity_correct: bool,
    full_correct: bool,
    rules_correct: impl FnOnce() -> bool,
) -> (
    [bool; Variant::CANDIDATES.len()],
    [bool; Variant::CANDIDATES.len()],
) {
    match source {
        Source::Exact | Source::Lower | Source::ExactLower | Source::Context => {
            let row = [
                identity_correct,
                full_correct,
                full_correct,
                full_correct,
                full_correct,
            ];
            (row, row)
        }
        Source::External | Source::ExternalLower => (
            [
                identity_correct,
                identity_correct,
                rules_correct(),
                full_correct,
                full_correct,
            ],
            [
                identity_correct,
                identity_correct,
                identity_correct,
                full_correct,
                full_correct,
            ],
        ),
        Source::Rule => (
            [
                identity_correct,
                identity_correct,
                full_correct,
                identity_correct,
                full_correct,
            ],
            [identity_correct; Variant::CANDIDATES.len()],
        ),
        // Not reachable by token scoring; the same answer for every variant.
        Source::Entity
        | Source::EntityLower
        | Source::Composed
        | Source::Phrase
        | Source::User
        | Source::UserLower => (
            [full_correct; Variant::CANDIDATES.len()],
            [full_correct; Variant::CANDIDATES.len()],
        ),
        Source::Identity | Source::Protected => (
            [identity_correct; Variant::CANDIDATES.len()],
            [identity_correct; Variant::CANDIDATES.len()],
        ),
    }
}

/// Collects an exhausted per-candidate iterator into the lattice-sized array;
/// `None` is unreachable for iterators built from [`Variant::CANDIDATES`].
fn array<T>(items: impl Iterator<Item = T>) -> Option<[T; Variant::CANDIDATES.len()]> {
    let mut collected = Vec::with_capacity(Variant::CANDIDATES.len());
    collected.extend(items);
    collected.try_into().ok()
}

/// Scores every candidate configuration and returns the one that ships.
///
/// Candidates are ordered simplest first and a later one must be **strictly**
/// better to displace the one before it, so a tie always resolves towards fewer
/// bytes and fewer ways to be wrong. Being better than the incumbent is not
/// enough on its own: the acceptance rule also makes a candidate clear plain
/// copying by more than the sample's own binomial error, so the returned
/// configuration is never worse than copying on the data that chose it.
#[must_use]
pub fn choose(
    tables: &Tables,
    thresholds: Thresholds,
    gold: &[Graded],
    sample: SelectionSample,
) -> Option<(Variant, Score)> {
    let evidence = measure(tables, thresholds, gold, sample)?;
    let (variant, hits, _lead) = evidence.verdict();
    let tokens = evidence.tokens();
    Some((
        variant,
        Score {
            tokens,
            accuracy: hits as f64 / tokens as f64,
            baseline: evidence.copied() as f64 / tokens as f64,
        },
    ))
}

/// Measures the share of tokens that reaches the `lower` and `external`
/// probes, for the compiler's codec choice. `None` when there is nothing to
/// measure.
#[must_use]
pub fn traffic(tables: &Tables, forms: &[String]) -> Option<(f64, f64)> {
    traffic_iter(tables, forms.iter().map(String::as_str))
}

/// [`traffic`] over borrowed forms.
pub fn traffic_iter<'a>(
    tables: &Tables,
    forms: impl IntoIterator<Item = &'a str>,
) -> Option<(f64, f64)> {
    let (mut reached_lower, mut reached_external) = (0_u64, 0_u64);
    let mut tokens = 0_u64;
    for form in forms {
        tokens = tokens.saturating_add(1);
        let normalised = nfc(form);
        if tables.exact.contains_key(normalised.as_ref()) {
            continue;
        }
        reached_lower = reached_lower.saturating_add(1);
        let lowered = tables.casing.lowercase(&normalised);
        if tables.lower.contains_key(lowered.as_ref()) {
            continue;
        }
        if lowered != normalised && tables.exact.contains_key(lowered.as_ref()) {
            continue;
        }
        reached_external = reached_external.saturating_add(1);
    }
    if tokens == 0 {
        return None;
    }
    let share = |count: u64| round_to(count as f64 / tokens as f64, 4);
    Some((share(reached_lower), share(reached_external)))
}

/// Rounds to three decimals the way Python's `round` does — half to even.
pub(super) fn round3(value: f64) -> f64 {
    round_to(value, 3)
}

/// Rounds to `places` decimals the way Python's `round` does — half to even.
fn round_to(value: f64, places: usize) -> f64 {
    format!("{value:.places$}").parse().unwrap_or(value)
}

#[cfg(test)]
mod tests {
    use indexmap::IndexMap;

    use super::*;
    use crate::normalization::Casing;
    use crate::rule::CaseAction;
    use crate::train::Trained;

    fn brute_force_tune(tables: &Tables, development: &[Graded]) -> Thresholds {
        let graded: Vec<&Graded> = development
            .iter()
            .filter(|token| token.lemma.is_some())
            .collect();
        let accuracy = |thresholds, subset: &[&Graded]| {
            let view = View::new(tables, thresholds, Variant::Full);
            subset
                .iter()
                .filter(|token| {
                    token.lemma.as_deref().is_some_and(|lemma| {
                        nfc(lemma) == nfc(&view.lemma(&token.form, token.sentence_initial))
                    })
                })
                .count() as f64
                / subset.len() as f64
        };
        let mut general = CONFIDENCE_GRID[0];
        let mut best = f64::NEG_INFINITY;
        for value in CONFIDENCE_GRID {
            let measured = accuracy(
                Thresholds {
                    general: value,
                    proper_noun: value,
                },
                &graded,
            );
            if measured > best {
                best = measured;
                general = value;
            }
        }
        let bucket: Vec<&Graded> = graded
            .into_iter()
            .filter(|token| CaseKey::of(&token.form, token.sentence_initial) == MidSentenceCapital)
            .collect();
        if bucket.len() < 50 {
            return Thresholds {
                general,
                proper_noun: general,
            };
        }
        let mut proper_noun = CONFIDENCE_GRID[0];
        let mut best = f64::NEG_INFINITY;
        for value in CONFIDENCE_GRID {
            let measured = accuracy(
                Thresholds {
                    general,
                    proper_noun: value,
                },
                &bucket,
            );
            if measured > best {
                best = measured;
                proper_noun = value;
            }
        }
        Thresholds {
            general,
            proper_noun,
        }
    }

    /// The same receipt reached without the shortcut: every configuration is
    /// probed separately — the NER profile through a real `use_rules = false`
    /// probe, never derived — and every paired count is recomputed token by
    /// token. Nothing is shared with the implementation but the types.
    fn brute_force_measure(
        tables: &Tables,
        thresholds: Thresholds,
        gold: &[Graded],
        sample: SelectionSample,
    ) -> Option<SelectionEvidence> {
        let scorable: Vec<&Graded> = gold.iter().filter(|row| row.lemma.is_some()).collect();
        let tokens = u64::try_from(scorable.len()).ok()?;
        let with_rules = probe::Options::default();
        let without_rules = probe::Options {
            use_rules: false,
            ..probe::Options::default()
        };
        let right = |variant: Variant, options: probe::Options, row: &Graded| {
            let view = View::new(tables, thresholds, variant);
            nfc(&probe::lemma(
                &view,
                &row.form,
                row.sentence_initial,
                options,
            ))
            .as_ref()
                == nfc(row.lemma.as_deref().unwrap_or_default()).as_ref()
        };
        let count = |judge: &dyn Fn(&Graded) -> bool| {
            scorable.iter().filter(|row| judge(row)).count() as u64
        };
        let copied = count(&|row| right(Variant::Identity, with_rules, row));
        let candidates = Variant::CANDIDATES.map(|variant| Candidate {
            variant,
            hits: count(&|row| right(variant, with_rules, row)),
            discordant: count(&|row| {
                right(variant, with_rules, row) != right(Variant::Identity, with_rules, row)
            }),
        });
        let profiles = Variant::CANDIDATES.map(|variant| CandidateProfile {
            ner_hits: count(&|row| right(variant, without_rules, row)),
            rivals: Variant::CANDIDATES.map(|rival| {
                count(&|row| right(variant, with_rules, row) != right(rival, with_rules, row))
            }),
        });
        SelectionEvidence::with_profiles(sample, tokens, copied, candidates, profiles)
    }

    /// The same verdict reached from the brute-force receipt with the rule
    /// spelled out flat: a candidate must clear copying by the sample's
    /// margin; against the standing winner it needs either a pairwise lead
    /// the pair's own error admits, or a strictly better NER profile inside
    /// that tie zone; every remaining tie keeps the earlier, simpler winner.
    fn brute_force_choose(
        tables: &Tables,
        thresholds: Thresholds,
        gold: &[Graded],
        sample: SelectionSample,
    ) -> Option<(Variant, Score)> {
        let evidence = brute_force_measure(tables, thresholds, gold, sample)?;
        let tokens = evidence.tokens();
        let copied = evidence.copied();
        let z = sample.strictness();
        let rows: Vec<(Candidate, CandidateProfile)> = evidence
            .candidates()
            .iter()
            .copied()
            .zip(evidence.profiles()?)
            .map(|(candidate, profile)| (candidate, *profile))
            .collect();
        let mut winner = (Variant::Identity, copied, copied);
        for (candidate, profile) in &rows {
            let lead = candidate.hits.saturating_sub(copied);
            if lead == 0 || (lead as f64) < z * (candidate.discordant as f64).sqrt() {
                continue;
            }
            let standing = rows
                .iter()
                .position(|(rival, _)| rival.variant == winner.0)
                .unwrap_or(0);
            let pair = profile.rivals.get(standing).copied().unwrap_or(0);
            let margin = candidate.hits.abs_diff(winner.1);
            let wins = if margin > 0 && margin as f64 >= z * (pair as f64).sqrt() {
                candidate.hits > winner.1
            } else {
                profile.ner_hits > winner.2
            };
            if wins {
                winner = (candidate.variant, candidate.hits, profile.ner_hits);
            }
        }
        Some((
            winner.0,
            Score {
                tokens,
                accuracy: winner.1 as f64 / tokens as f64,
                baseline: copied as f64 / tokens as f64,
            },
        ))
    }

    #[test]
    fn one_probe_threshold_trace_matches_the_full_grid() {
        let mut rules = IndexMap::new();
        for (key, confidence) in [("0ts", 0.6), ("0gs", 0.9), ("2es", 0.5)] {
            rules.insert(
                key.to_owned(),
                Trained {
                    cut: 1,
                    add: String::new(),
                    case: CaseAction::Lower,
                    confidence,
                },
            );
        }
        let mut tables = Tables {
            casing: Casing::Root,
            rules,
            ..Tables::default()
        };
        tables
            .exact
            .insert("known".to_owned(), "known-lemma".to_owned());
        tables
            .external
            .insert("borrowed".to_owned(), "borrow".to_owned());
        // A corpus row can shadow the same suffix rule that Full reaches only
        // after an external miss. This guards the source-based shortcut in
        // `choose`: `Corpus` must keep its own lookup answer here.
        tables
            .exact
            .insert("cats".to_owned(), "corpus-cat".to_owned());
        let mut development = vec![
            Graded {
                form: "cats".to_owned(),
                lemma: Some("corpus-cat".to_owned()),
                sentence_initial: false,
            },
            Graded {
                form: "dogs".to_owned(),
                lemma: Some("dogs".to_owned()),
                sentence_initial: false,
            },
            Graded {
                form: "known".to_owned(),
                lemma: Some("known-lemma".to_owned()),
                sentence_initial: false,
            },
            Graded {
                form: "borrowed".to_owned(),
                lemma: Some("borrow".to_owned()),
                sentence_initial: false,
            },
        ];
        development.extend((0..60).map(|index| Graded {
            form: "Names".to_owned(),
            lemma: Some(if index % 3 == 0 { "name" } else { "Names" }.to_owned()),
            sentence_initial: false,
        }));

        let thresholds = tune(&tables, &development);
        assert_eq!(thresholds, brute_force_tune(&tables, &development));
        for sample in [SelectionSample::Dev, SelectionSample::TrainCv] {
            // The whole receipt — both profiles and every pairwise counter —
            // must match what independent probes of every candidate measure,
            // and the verdict must match the rule spelled out flat.
            assert_eq!(
                measure(&tables, thresholds, &development, sample),
                brute_force_measure(&tables, thresholds, &development, sample)
            );
            assert_eq!(
                choose(&tables, thresholds, &development, sample),
                brute_force_choose(&tables, thresholds, &development, sample)
            );
        }
    }

    #[test]
    fn the_proper_noun_gate_is_fitted_on_its_bucket_and_nothing_else() {
        // One rule, learned on `-s`, right on ordinary words and wrong on the
        // names it has never seen. Whether the gate fires is decided by the
        // mid-sentence capitals alone: the same evidence moved to the start of
        // a sentence, or lowercased, belongs to the general gate and must
        // leave the proper-noun one exactly where the general one landed.
        let mut rules = IndexMap::new();
        rules.insert(
            "2ts".to_owned(),
            Trained {
                cut: 1,
                add: String::new(),
                case: CaseAction::Title,
                confidence: 0.7,
            },
        );
        rules.insert(
            "1ts".to_owned(),
            Trained {
                cut: 1,
                add: String::new(),
                case: CaseAction::Title,
                confidence: 0.7,
            },
        );
        let tables = Tables {
            casing: Casing::Root,
            rules,
            ..Tables::default()
        };
        let names = |sentence_initial: bool| -> Vec<Graded> {
            (0..60)
                .map(|_| Graded {
                    form: "Fronts".to_owned(),
                    lemma: Some("Fronts".to_owned()),
                    sentence_initial,
                })
                .collect()
        };

        // Mid-sentence: 60 capitals the rule would destroy, so the bucket
        // gate climbs above the rule's confidence while the general gate,
        // fitted on the same 60 rows plus nothing else, does not.
        let mid = names(false);
        assert_eq!(
            tune(&tables, &mid),
            Thresholds {
                general: 0.8,
                proper_noun: 0.8
            }
        );
        assert_eq!(tune(&tables, &mid), brute_force_tune(&tables, &mid));

        // Sentence-initial: the very same forms and lemmas now fall in bucket
        // 1, the bucket sample is empty, and the proper-noun gate can only
        // follow the general one.
        let initial = names(true);
        let fitted = tune(&tables, &initial);
        assert_eq!(
            fitted,
            Thresholds {
                general: fitted.general,
                proper_noun: fitted.general
            }
        );
        assert_eq!(fitted, brute_force_tune(&tables, &initial));
    }

    #[test]
    fn trace_preserves_view_semantics_for_nan_confidence() {
        let mut rules = IndexMap::new();
        rules.insert(
            "0ts".to_owned(),
            Trained {
                cut: 1,
                add: String::new(),
                case: CaseAction::Lower,
                confidence: f64::NAN,
            },
        );
        let tables = Tables {
            casing: Casing::Root,
            rules,
            ..Tables::default()
        };
        let development = [Graded {
            form: "cats".to_owned(),
            lemma: Some("cat".to_owned()),
            sentence_initial: false,
        }];
        assert_eq!(
            tune(&tables, &development),
            brute_force_tune(&tables, &development)
        );
    }

    #[test]
    fn borrowed_traffic_stream_matches_the_public_slice_api() {
        let mut tables = Tables {
            casing: Casing::Root,
            ..Tables::default()
        };
        tables.exact.insert("known".to_owned(), "lemma".to_owned());
        tables.lower.insert("lower".to_owned(), "lemma".to_owned());
        let forms = vec!["known".to_owned(), "LOWER".to_owned(), "unknown".to_owned()];

        assert_eq!(
            traffic_iter(&tables, forms.iter().map(String::as_str)),
            traffic(&tables, &forms)
        );
        assert_eq!(traffic_iter(&tables, std::iter::empty()), None);
    }
}
