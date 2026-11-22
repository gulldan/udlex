# udlex

**English** · [Русский](https://github.com/gulldan/udlex/blob/main/README.ru.md) · [简体中文](https://github.com/gulldan/udlex/blob/main/README.zh-CN.md)

udlex is a dictionary lemmatizer. It looks up each token in compiled tables
learned from Universal Dependencies treebanks and open dictionaries, and guesses
unknown words from suffix rules. The core is Rust; the package adds Python
bindings and a command line, and the same core builds as a C library. There is
no neural network: nothing to run on a GPU, and the answers are deterministic.

## Install

With [uv](https://docs.astral.sh/uv/), add the engine and a dictionary to your
project:

```sh
uv add udlex
uv add "udlex-dict-eng-nc @ https://github.com/gulldan/udlex/releases/download/v1.0.0/udlex_dict_eng_nc-1.0.0-py3-none-any.whl"
uv run udlex lemma eng mice were running
```

Python 3.11 or newer. Wheels are built for Linux (x86_64, aarch64), macOS
(arm64, x86_64) and Windows (x86_64); on other platforms uv builds from source
and needs a Rust toolchain.

The engine is on PyPI; the dictionaries are not. Each language is a separate
wheel attached to the [v1.0.0 release](https://github.com/gulldan/udlex/releases/tag/v1.0.0),
and `uv add` with the wheel URL records it in `uv.lock` like any dependency.
`uv run udlex install --dry-run fin rus` prints the URLs for any languages.
`udlex install <iso> ...` installs them into the current environment with
`uv pip` (pip when uv is missing); inside a uv project it prints the `uv add`
command to run, because `uv sync` removes packages the lock does not list.

Wheels are named `udlex_dict_<iso>-1.0.0-py3-none-any.whl`, with `_nc` after
the code for [non-commercial](#licenses) packages such as English. They install
into the `udlex_bundle` namespace, where `udlex.load` and the command line find
them. Every package is also published as `udlex-dict-<iso>-1.0.0-bundle.tar.gz`
(`-nc` likewise), which unpacks into `bundle/<iso>`. Unpack several into the same
place; udlex reads `./bundle` when no dictionary wheel is installed, or the
directory in `UDLEX_BUNDLE`, or one you pass as `udlex.load(directory, iso)`.

## Use

```py
import udlex

lexicon = udlex.load("eng")  # load once per process; safe to share across threads
print(lexicon.lemmatize(["The", "mice", "were", "running"]))
```

The input is a list of tokens: udlex has no tokenizer. Languages are named by
ISO 639-3 code. From the shell:

```sh
uv run udlex lemma eng mice were running
uv run udlex languages
```

### Try it without downloading anything

This builds a two-word dictionary from a `form<TAB>lemma` table and runs it.
Save it as `quickstart.py` and run `uv run quickstart.py`.

```python
from pathlib import Path

import udlex

bundle = Path("example-bundle")
bundle.mkdir()
words = bundle / "lemmas.tsv"
words.write_text("cats\tcat\nmice\tmouse\n", encoding="utf-8")

udlex.build_lexicon(
    str(bundle / "eng"),
    train=None,
    dev=None,
    test=None,
    paradigms=[("example", str(words), False, False)],
    name="eng",
    language="English",
    iso639_3="eng",
)
lexicon = udlex.load(bundle, "eng")
print(lexicon.lemmatize(["cats", "mice", "unknown"]))
# ['cat', 'mouse', 'unknown']
```

```sh
uv run udlex --bundle example-bundle lemma -q eng cats mice
```

## Documentation

- [Quick start](https://github.com/gulldan/udlex/blob/main/docs/quickstart.md)
- [API: Python, command line, C](https://github.com/gulldan/udlex/blob/main/docs/api.md)
- [Languages and accuracy](https://github.com/gulldan/udlex/blob/main/docs/languages.md)
- [User dictionary: your own overrides](https://github.com/gulldan/udlex/blob/main/docs/user-dictionary.md)
- [Building dictionaries](https://github.com/gulldan/udlex/blob/main/docs/building-dictionaries.md)

## Languages

The v1.0.0 release has dictionaries for 224 languages, one package per
language. The [language table](https://github.com/gulldan/udlex/blob/main/docs/languages.md)
lists each package with its license, how it was scored and its accuracy.

## Accuracy

Exact lemma match on the gold tokens of Universal Dependencies 2.19
([`evaluation.json`](https://github.com/gulldan/udlex/blob/main/dictionaries/catalog/evaluation.json)).
*Copy* returns every token unchanged.

| Languages | Scored on | udlex, macro % | Copy, % |
| --- | --- | ---: | ---: |
| 103 | official test split | 89.85 | 58.41 |
| 109 | two-fold estimate (no training split) | 73.87 | 57.54 |
| 12 | no gold lemmas: unmeasured | — | — |

Against other lemmatizers on identical test words (from
[`benchmarks/generated/numbers.tex`](https://github.com/gulldan/udlex/blob/main/benchmarks/generated/numbers.tex)).
Each row averages over the languages both systems cover, after dropping
languages whose model saw our test sentences, so rows are not comparable with
each other.

| System | Languages | System, % | udlex, % |
| --- | ---: | ---: | ---: |
| Stanza | 65 | 95.32 | 93.81 |
| UDPipe 2 | 77 | 94.48 | 93.47 |
| Trankit | 46 | 91.32 | 94.44 |
| UDPipe 1 | 44 | 91.36 | 95.16 |
| simplemma | 33 | 90.75 | 95.44 |
| spaCy sm | 17 | 89.55 | 95.77 |
| spaCy lg | 17 | 90.63 | 95.77 |
| spaCy trf | 7 | 93.98 | 97.14 |
| LemmaGen3 | 16 | 82.27 | 96.33 |
| CoreNLP | 1 | 96.37 | 97.87 |

Stanza and UDPipe 2 are more accurate than udlex by about 1 to 1.5 points:
they are neural models that see the whole sentence and its part-of-speech
tags. udlex is ahead of the other systems on average.

## Speed and memory

All systems were timed on one AMD Ryzen 9 7950X (16 cores) with an RTX 4090,
on the pre-tokenized UD 2.19 test sentences of every language each system
supports. Rates are medians over languages. *udlex faster* is the geometric
mean, over the languages a system was timed on, of how many times faster the
Rust core of udlex is on one core. *Parallel* is the better of 16 processes and
16 threads, or one GPU for neural systems. *Peak memory* is the resident set of
the timing process on one core, which also holds the test text: that system's
median, then udlex's on the same languages.

| System | Languages | Words/s, one core | udlex faster, × | Words/s, parallel | Peak memory, MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| **udlex, Rust** | 169 | 54.4M | — | 759M | 147 |
| **udlex, Python** | 169 | 18.0M | — | 142M | 1,172 |
| simplemma | 37 | 6.6M | 5.4 | 57.8M | 324 / 110 |
| LemmaGen3 | 17 | 3.5M | 9.5 | 32.5M | 1,445 / 118 |
| CoreNLP | 1 | 27k | 1,965 | 336k | 1,600 / 111 |
| spaCy sm | 20 | 27k | 1,422 | 218k | 773 / 114 |
| spaCy lg | 20 | 25k | 1,568 | 165k | 1,546 / 114 |
| UDPipe 1 | 50 | 15k | 2,177 | 151k | 153 / 99 |
| Stanza | 90 | 4.5k | 8,160 | 16k (GPU) | 1,240 / 99 |
| spaCy trf | 8 | 278 | 143,271 | 20k (GPU) | 2,840 / 111 |
| Trankit | 53 | 74 | 433,457 | 3.9k (GPU) | 4,665 / 94 |

On one core udlex ranges from 6.2 million to 858 million words per second
depending on the language. simplemma, the fastest other system, is faster only
on Georgian, by 1 %. The Python binding gains 1.46 times on 16 threads and 8.9
times on 16 processes, and its memory includes the Python token lists of the
test text. UDPipe 2 was not timed. Numbers depend on the CPU and on the token
mix of the test texts. The values come from
[`numbers.tex`](https://github.com/gulldan/udlex/blob/main/benchmarks/generated/numbers.tex);
the receipts are in
[`benchmarks/evidence/speed`](https://github.com/gulldan/udlex/tree/main/benchmarks/evidence/speed).

## Limitations

- **Tokens in, lemmas out.** There is no tokenizer, sentence splitter or
  tagger. Split the text first.
- **No context.** Each token gets one lemma, whatever the sentence. A form
  with several possible lemmas (English *saw*: *see* or *saw*) always gets the
  same one. This is the main gap to Stanza and UDPipe 2. The `upos=` and
  `context=` arguments of `lemmatize` only matter for dictionaries built with
  context keys, and the 1.0.0 packages are built without them.
- **Unknown words** are guessed from suffix rules learned from the treebank.
  On words the dictionary has never seen, this is weaker than a neural model.
  Pass `use_rules=False` (`--lookup-only` on the command line) to return the
  token unchanged instead of guessing.
- **Accuracy varies by language**, from near 100 % to under 50 %; check the
  [language table](https://github.com/gulldan/udlex/blob/main/docs/languages.md)
  before relying on a language. For the 109 languages without a training split
  the accuracy is a two-fold estimate on small treebanks, and 12 languages have
  no gold lemmas at all.
- **Partly exploratory scores.** For languages without an official dev split,
  the dev stand-in was introduced after the first test evaluation.
- **No machine-labelled lemmas.** The released dictionaries contain no lemmas
  labelled by a language model; each package lists its sources in
  `SOURCES.json`.
- **Non-commercial treebanks.** 39 packages, among them English, Czech,
  Arabic and Italian, learned from a treebank under a non-commercial license
  and may only be used non-commercially.

## Other interfaces

- **C library**: `cargo build --locked --release --manifest-path rust/Cargo.toml -p udlex-capi`
  builds `libudlex_capi` with the header
  [`rust/capi/include/udlex.h`](https://github.com/gulldan/udlex/blob/main/rust/capi/include/udlex.h);
  see the [C API](https://github.com/gulldan/udlex/blob/main/docs/api.md#c-api).
- **Go**: [gulldan/udlex-go](https://github.com/gulldan/udlex-go).

## Licenses

The code is [MIT](https://github.com/gulldan/udlex/blob/main/LICENSE).

Each dictionary package has its own license, which follows from its sources:
in v1.0.0, 184 packages are CC BY-SA (182 version 4.0, 2 version 3.0), 39 are
CC BY-NC-SA 4.0 and 1 is CC BY 4.0. Packages whose name ends in `-nc` are for
non-commercial use only. Inside a package every table keeps the license of its
source, and `SOURCES.json` lists each source with its URL, version, author and
license, as spaCy's `meta.json` does. If you ship a dictionary, keep its
`NOTICE` and `SOURCES.json` and follow its license. All sources and package
licenses are listed in
[DICTIONARY_LICENSES.md](https://github.com/gulldan/udlex/blob/main/DICTIONARY_LICENSES.md).

## Citation

See [CITATION.cff](https://github.com/gulldan/udlex/blob/main/CITATION.cff).
Contributions: [CONTRIBUTING.md](https://github.com/gulldan/udlex/blob/main/CONTRIBUTING.md).
