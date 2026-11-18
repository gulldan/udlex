# Dictionary packages

This directory builds one `udlex-dict-<iso>` package per language of
Universal Dependencies 2.19, from the treebank and from form-to-lemma
dictionaries whose licenses allow redistribution. Every input is pinned by URL
and SHA-256 in [`sources.json`](sources.json); the license policy lives in
[`policy.json`](policy.json).

```sh
dictionaries/build.sh /path/to/workdir [/path/to/1.0.0-dictionaries]
```

The command needs Python 3.11 or newer with `udlex` 1.0.0 built from this
checkout and `pymorphy3==2.0.6`. It runs these stages, each resumable:

| Stage | Script | Output |
| --- | --- | --- |
| Download and verify every input | `fetch.py` | `cache/<sha256>` |
| Pick one treebank per language, write splits | `splits.py` | `work/roster.json`, `work/splits/<iso>/` |
| Turn every dictionary into `form, lemma, count` tables | `extract.py` | `work/pairs/<source>/<iso>.tsv.gz` |
| Choose order and options on dev, build | `choose.py` | `work/build/<iso>/decision.json` |
| Wheels, archives, licenses, catalog | `package.py` | `release/packages/`, `release/DICTIONARIES.json` |
| Score new, 1.0.0 and copying | `evaluate.py` | `release/evaluation.json` |
| Per-language table | `report.py` | `release/EVALUATION.md` |

## Data use

A language whose anchor treebank has train and test splits learns from train.
Dev, when there is one, chooses the dictionary order, the engine options and the
rule thresholds, and joins the final refit. Test is read only by `evaluate.py`.
Dev and test lose every sentence whose document, sentence id or text occurs in
an earlier split. When a treebank has no dev split, a tenth of the training
documents, chosen by hash, stands in for it and joins the final refit; the
engine's own fallback holds out unseen forms only and then often prefers
copying.

The dictionary order follows the study's protocol: the exact best permutation
and ordered subset of the dictionaries are fitted on training words that the
other four fifths of training do not contain, and replace the declared order
only when the full cascade trained on train is better on dev by an exact
McNemar test (p ≤ 0.05) and by at least max(5, dev tokens / 10000) words. The
lemma adapter and the compound marker pass the same gate. Combining marks that
the training split never writes (Wiktionary stress, Latin vowel length) are
removed from dictionary entries before lookup.

A language with gold lemmas but no training split learns from its whole
treebank. Its accuracy is a two-fold estimate: the treebank is cut into two
halves of whole documents, and each half is scored by a build that learned from
the other. Such packages keep the declared order; thresholds and the shipped
variant come from a dev carved out of their own training half. Languages without gold lemmas
get a package only when a dictionary covers them.

## Licenses

A package whose treebank, or one of whose dictionaries, is non-commercial is
released under CC BY-NC-SA 4.0 and named `udlex-dict-<iso>-nc`. Otherwise the
package takes the strictest license of its parts. A treebank whose README and
LICENSE disagree, and the treebanks listed in `noncommercial_review`, count as
non-commercial and carry `requires_license_review`. Sources with copyleft or
unclear licenses, and the sources named in `excluded_sources`, are left out;
both lists are configuration. Where a left-out copyleft dictionary covers a
language with a dev split, `decision.json` records what including it would
change on dev.

Each wheel and archive holds the treebank README and LICENSE, a NOTICE with the
attribution of every input, and `SOURCES.json` with digests, the selection
evidence and the recipe.

## Pins

Some upstream files are not versioned (kaikki.org extracts, BÍN, SALDO
morphology). They are pinned by SHA-256, so a rebuild fails
loudly once upstream changes; `fetch.py --pin` records new digests.
