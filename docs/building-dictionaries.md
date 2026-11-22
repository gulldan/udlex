# Building dictionaries

There are two ways to get a dictionary that the release does not have: build a
small one from your own form/lemma tables, or rerun the release pipeline. For
fixes on top of a shipped dictionary, a [user dictionary](user-dictionary.md)
is simpler.

## From a form/lemma table

`udlex.build_lexicon` builds one lexicon and publishes it atomically into a
directory. Without a treebank it is a pure lookup dictionary:

```py
from pathlib import Path

import udlex

Path("my-bundle").mkdir(exist_ok=True)  # the bundle directory must exist
udlex.build_lexicon(
    "my-bundle/eng",
    train=None,
    dev=None,
    test=None,
    paradigms=[
        ("terms", "terms.tsv", False, False),  # consulted first
        ("general", "general.tsv", True, False),
    ],
    name="eng",
    language="English",
    iso639_3="eng",
)
lexicon = udlex.load("my-bundle", "eng")
```

Each table is plain UTF-8 TSV, one pair per line; further columns are
ignored. `paradigms` lists them best first, as
`(label, path, lemma_first, normalized_unique)`:

- `lemma_first`: `False` for `form<TAB>lemma`, `True` for `lemma<TAB>form`.
- `normalized_unique`: `True` only for a table already NFC-normalised, sorted
  and with one row per form, as the release pipeline writes them. Pass `False`
  otherwise.

`udlex --bundle my-bundle verify` checks that every lexicon in the directory
opens; `udlex --bundle my-bundle languages` lists them.

With a training corpus (`train`, and optionally `dev`, as the JSONL splits the
release pipeline writes from CoNLL-U), the same call also learns the treebank
tables and suffix rules and tunes them on dev; the returned `Outcome` reports
what was built and its accuracy on `test` when given.

## The release pipeline

[`dictionaries/`](../dictionaries) holds the scripts that built the 224 v1.0.0
packages: download and verify every pinned input, split the UD 2.19
treebanks, extract the dictionaries, choose the dictionary order and options
on dev, package wheels and archives with their licenses, and score them. The
stages, the data-use rules and the license policy are documented in
[`dictionaries/README.md`](../dictionaries/README.md); the inputs are pinned by
URL and SHA-256 in [`dictionaries/sources.json`](../dictionaries/sources.json)
and the license policy is [`dictionaries/policy.json`](../dictionaries/policy.json).

To add a dictionary source, add it to `sources.json` with its license and
pins, rebuild, and compare the new `evaluation.json` with the committed one in
[`dictionaries/catalog`](../dictionaries/catalog). After the catalog changes,
regenerate the documentation pages:

```sh
uv run --frozen python dictionaries/pages.py
```
