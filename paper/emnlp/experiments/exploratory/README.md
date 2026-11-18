# Exploratory measurements behind `anatomy-protocol.md`

One-off scripts with the server paths they ran with, and their outputs. They read the official test splits, so nothing here is confirmatory.

| Script | Output | What it measures |
| --- | --- | --- |
| `crossfit.py` | `crossfit.json` | Source order fitted on training words unseen in the other four folds, scored through the full cascade |
| `ceiling.py` | `ceiling.json` | Best form-to-lemma function on each test, training memory, shares of seen and unseen words |
| `stages.py` | `stages.json` | Accuracy of the declared cascade by answering stage, on seen and unseen forms |
| `strata.py` | `strata.json` | Cascade and Stanza 1.15.1 on the same strata of words |
| `upos.py` | printed only | Majority lemma of (form, gold UPOS) on forms seen with several lemmas |
| `proto.py` | `proto.json`, `proto2.json` | Linear window tagger and suffix decision lists on 14 languages; the second run adds lists built from dictionary paradigms |
| `census.py` | `census.json` | Cascade on every catalog language with gold lemmas (100 official splits, 68 two-fold) and each comparator, by stratum of test words |
| `ladder.py` | `ladder.json` | What a (form, tag) table with a linear tagger and suffix decision lists add to the cascade on the 100 languages with official splits, against UDPipe 2 |
