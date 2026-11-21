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

The build code is MIT, like the rest of udlex. Each dictionary package has its
own license, which follows from the sources listed in its `NOTICE` and in the
`sources` of its metadata; packages named `-nc` are for non-commercial use only.

Every table keeps the license of its source. The corpus tables (`exact`,
`lower`, rules, `entities`) are learned from the treebank alone and carry its
license; every row of the `external` table comes from exactly one dictionary
and carries that dictionary's license. A package with a non-commercial part is
named `udlex-dict-<iso>-nc` and released as a whole under CC BY-NC-SA 4.0;
otherwise the package takes the strictest license of its parts.

Like spaCy's `meta.json`, `SOURCES.json` (also the package's entry in
`DICTIONARIES.json`) lists `sources` with name, URL, version, author and SPDX
license, and for each the generation files it fills; `tables` groups the files
by part, `part_licenses` lists the distinct licenses, and `license`, the one
declared in the wheel's `License-Expression`, is CC-BY-NC-SA-4.0 for `-nc`
packages and the strictest part license otherwise. The generation format
is unchanged, so the parts are separable by row, not by file: `external` merges
all dictionaries, each row taken from the first dictionary in the shipped
order that holds the form, minus the forms the corpus tables answer; rows whose
shipped lemma differs from the dictionary's (lemma adapter, compound marker,
script guard) are counted as `changed_lemmas`. `strings.txt`/`strings.idx` are
one string pool shared by all tables and `meta.json` holds counts and the dev
selection evidence; both are listed as the `shared` part under the package
license. The NOTICE has one section per part. A treebank whose README and
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
morphology). They stay pinned by URL and SHA-256 as the primary source, and
their sources also carry a `mirror`: `pairs-<id>.tar` release assets of the tag
named in `policy.json`, holding the extracted `<iso>.tsv.gz` tables and the
manifest that names the upstream digests (`mirror.py` packs them, splitting
any archive above GitHub's 2 GiB asset limit). When an upstream file is gone
or no longer matches its pin, `fetch.py` downloads the archive instead and
`extract.py` starts from the pairs, after checking every table against the
manifest; the build is the same byte for byte. `fetch.py --pin` records new
digests.

## Catalog of this build

[`catalog/`](catalog) holds the `DICTIONARIES.json`, `evaluation.json`,
`EVALUATION.md` and `SHA256SUMS` of the build the packages were made from; the
wheels and archives themselves are release assets, not part of the repository.
