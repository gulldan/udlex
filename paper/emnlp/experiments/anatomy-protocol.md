# Where lemmatization accuracy comes from: protocol for the second study

Written on 19 November 2026 after exploratory measurements on the official UD 2.19 test splits of the 66 anchor treebanks. Those splits have now been inspected repeatedly, so every number computed on them is exploratory. Confirmation uses data that no decision below has touched: the screened transfer corpora of `transfer-protocol.md` and, where a model must be selected, out-of-fold predictions inside the training split.

## What the exploration showed

Scripts and outputs are kept in `exploratory/`. On 58 languages whose Stanza model neither saw the test split nor lacks a lemmatizer (1,808,517 words):

| Words | Share | Cascade | Stanza 1.15.1 | Either correct |
| --- | ---: | ---: | ---: | ---: |
| Form seen in training with one lemma | 73.83% | 99.59 | 99.47 | 99.59 |
| Form seen in training with several lemmas | 16.21% | 91.87 | 95.75 | 96.36 |
| Unseen, answered by lower-cased corpus form | 0.67% | 79.20 | 86.74 | 95.20 |
| Unseen, answered by a dictionary | 5.63% | 82.92 | 83.59 | 94.68 |
| Unseen, answered by a suffix rule or copied | 3.66% | 57.57 | 82.38 | 86.48 |
| All | 100% | 95.72 | 97.27 | 98.28 |

- The best form-to-lemma function fitted to the test itself scores 98.82% macro over 66 languages. Stanza exceeds this bound in 5 of 58 languages.
- On forms seen with several lemmas, the majority lemma of (form, gold UPOS) scores 97.18% and of the form alone 91.93%. A linear window tagger trained in seconds lifted this stratum from 93.79 to 95.53 on 14 languages where Stanza has 96.18.
- On rule-or-copy words, suffix decision lists reached 67% on 14 languages where Stanza has 84%. Lists built from dictionary paradigms did not help.
- On every catalog language with an official training and test split (100 languages, 2,200,177 words; `exploratory/census.py`) the cascade scores 91.23% macro: 83.73 on 13 languages without a dictionary, 88.38 on 21 with one, 93.62 on the 66 with two or more. On 68 further languages that have only a test treebank, halved into two folds, it scores 79.10. UD 2.19 holds 55 languages that the catalog lacks (`language-codes-ud219-extra.json`); 43 of them carry lemmas and were added without dictionaries (`exploratory/census-extra.json`), 12 carry none. Over all 211 languages with gold lemmas the cascade scores 83.78% macro: 90.62 on 103 with an official split and 77.26 on 108 scored by two folds. UDPipe 2 with its UD 2.17 models covers 83 of the 100 and scores 94.51 where the cascade scores 93.35; it is above the cascade in 67 languages and below in 14, and its advantage has the same anatomy as Stanza's (seen with several lemmas 96.14 against 91.26, rule-or-copy words 74.99 against 54.93, seen with one lemma 99.32 against 99.57).
- The source order fitted on all training tokens gains 2.33 points on its training objective and 205 words on test; the two gains correlate at r = 0.05 across languages. Fitted only on training words whose form is absent from the other four fifths of training, it gains 1,120 words, with 35 languages up and 5 down.

## Scope

The population is every language of the catalog that has gold lemmas, not the 66 languages on which a source order can be fitted. Results are reported for three groups: languages with an official training and test split, languages evaluated by two folds of a test-only treebank, and the subset on which a given comparator has a model. No headline counts languages; a claim holds for a group or it is not made.

The released lemmatizer contains no neural network and none is added. Neural pipelines appear only as comparators and as the reference for what context and character-level generalization buy; rungs 6 to 8 below use linear models and tables.

## Questions and hypotheses

**Q1. Anatomy.** How much of accuracy is memory of the training corpus, lexical resources, generalization to unseen forms, and context, and where does a neural pipeline win?
- H1a. The context-free bound is at least 98% macro, and the neural pipeline exceeds it in at most one language in ten.
- H1b. At least half of the neural advantage lies on unseen forms outside the lexicons and at least a quarter on forms seen with several lemmas.
- H1c. On unseen forms that a lexicon covers, lexicon and neural pipeline are within two points, and at least one of them is right eight points more often than the better one.

**Q2. Residual fitting.** A fallback stage acts only on words that earlier stages pass on. Does fitting it on such words, obtained out of fold, beat fitting on all training tokens, and is the cheap proxy of rare forms enough?
- H2a. For fits on all tokens the training gain does not predict the held-out gain (|r| < 0.2); for out-of-fold fits it does (r > 0.5).
- H2b. On held-out corpora, out-of-fold fitting is at least as good as the rare-form proxy, which is at least as good as all-token fitting, in at least 70% of languages, for the source order, the choice between a dictionary answer and a guess, the (form, tag) table and the guesser.

**Q3. Cost.** How close does a cascade with a linear tagger and a guesser for residual words come to a neural pipeline, and at what cost?
- H3a. Its macro accuracy is not lower than Stanza's by more than 0.5 points (one-sided, languages as the unit). Being higher is a stretch goal; being higher in every language is not a goal, since Stanza exceeds the context-free bound in some.
- H3b. At that accuracy it processes at least 100 times more words per second on one CPU core.
- H3c. From anchor test to transfer corpora its accuracy falls less than Stanza's.

**Q4. Validity.** How often do lexical resources and released models contain test material, and does coverage of unseen forms detect it?
- H4. The coverage statistic separates the documented positives (Stanza's five treebanks with swapped splits, the Ancient Greek lookup table) from all other source–corpus pairs.

**Q5. Encoding.** Which dictionary encoding should a lookup lemmatizer use, and does the answer depend on the language?
- H5a. On real text a structure that splits keys by length (double array for short keys, hash with full keys for long ones) is at least as fast as every single structure at no more than 1.2 times the space of the smallest one.
- H5b. The ranking of structures under a uniform draw of keys differs from the ranking under the token stream of a test set, which is Zipfian and contains misses.
- H5c. The best split length is predicted by the byte length distribution of the keys of a language.

The benchmark lives in `encoding/`; every candidate must answer byte for byte like a reference map before its timings count, and timings are taken on an otherwise idle machine.

## Systems

One harness, tables from the training split only, selection on development data or out of fold, one pass over held-out data.

| Rung | Adds | Needs at inference |
| --- | --- | --- |
| 0 | Copy | the word |
| 1 | Corpus tables | the word |
| 2 | Suffix rules, zero thresholds | the word |
| 3 | External dictionaries, declared order (the published baseline) | the word |
| 4 | Source order fitted out of fold | the word |
| 5 | Rule thresholds, compound marker, lemma adapter, each accepted on development data | the word |
| 6 | (form, tag) table with a linear tagger; tags for the table are jackknifed | the sentence |
| 7 | Guesser for words that reach the rule stage, trained out of fold | the word and its tag |
| 8 | Learned choice between dictionary answer and guess | the word and its tag |

A rung is reported whether or not it helps. The guesser of rung 7 is a linear edit-script classifier over suffix and tag features; a character-level neural guesser is not built, because Stanza and UDPipe 2 already measure what one achieves on the same words.

Comparators on the same words, each at its latest release on the day of the run: UDPipe 2 (models `ud-2.17-251125`, the newest published; the LINDAT service lists none for UD 2.18 or 2.19), UDPipe 1 (`ud-2.5-191206`), Stanza 1.15.1, Trankit, spaCy 3.8 (`sm`, `lg`, `trf`, lookup tables), Stanford CoreNLP for English, simplemma 2.0.0, LemmaGen3, and tools that are the standard for one language (`compare_local.py`). Models without a lemmatizer are excluded.

A comparator trained on an older UD release may have seen sentences that are test sentences in UD 2.19, and its lemma conventions may differ from the 2.19 annotation. Both are measured: the share of 2.19 test sentences found in the training and development data of the release a model was trained on, and, as a control, the cascade rebuilt from that release and scored with the comparator on that release's own test split. A model is excluded from a headline when more than 1% of the test lies in its training data or when its lemma dictionary holds at least half of the test words absent from the official training split.

## Measurements

- Strata as in the table above, per language and per system; bounds from the test itself, from (form, gold UPOS), and from training alone.
- For each rung: corrections, regressions and changed predictions against the previous rung, on anchor tests and on transfer corpora.
- Cost on one session and one machine: words per second on one core and on 16 cores, GPU separately, load time, peak memory, size on disk; systems interleaved in random order, windows of at least one second, medians with ranges; batch sizes of neural systems tuned in their favour.

## Statistics

Paired bootstrap over documents within language (2,000 replicates, seed 20261118). With languages as the unit: exact sign test, Wilcoxon signed-rank test and a percentile interval over 10,000 resamples of languages. Leave-one-language-out and leave-two-out ranges for every headline number. H3a is tested once, on transfer corpora.

## Not claimed

Residual fitting has precedents (rare-word suffix guessers, cascade training on survivors, jackknifed tags); the contribution is its measurement across stages and languages. Numbers from the official tests that were inspected before this protocol are labelled exploratory wherever they appear.
