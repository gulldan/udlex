# Quick start

## 1. Install the engine

```sh
uv add udlex
```

Python 3.11 or newer. Wheels exist for Linux (x86_64, aarch64), macOS (arm64,
x86_64) and Windows (x86_64); elsewhere uv builds the package from source, which
needs a Rust toolchain.

## 2. Install a dictionary

The engine has no dictionaries. Each language is a separate package attached
to the [v1.0.0 release](https://github.com/gulldan/udlex/releases/tag/v1.0.0),
not on PyPI. Name languages by their ISO 639-3 code; the
[language table](languages.md) lists all of them. Add a dictionary to the
project by its wheel URL, so that `uv.lock` records it:

```sh
uv run udlex install --dry-run rus     # prints the wheel URL
uv add "udlex-dict-rus @ https://github.com/gulldan/udlex/releases/download/v1.0.0/udlex_dict_rus-1.0.0-py3-none-any.whl"
```

`udlex install rus` installs the wheel into the current environment with
`uv pip` (or pip when uv is missing). That is enough outside a project; inside a
uv project the next `uv sync` removes it, so `udlex install` prints the
`uv add` command to run instead.

Non-commercial packages carry `_nc` in the wheel name
(`udlex_dict_eng_nc-1.0.0-py3-none-any.whl`) and `-nc` in the distribution name
(`udlex-dict-eng-nc`); see [DICTIONARY_LICENSES.md](../DICTIONARY_LICENSES.md).

Without network access, download the wheels once and `uv add` the files. The
`udlex-dict-<iso>-1.0.0-bundle.tar.gz` archives are the same dictionaries as a
plain directory: each unpacks into `bundle/<iso>`, and several can share one
`bundle/`.

## 3. Lemmatize

```py
import udlex

lexicon = udlex.load("rus")
print(lexicon.lemmatize(["Мама", "мыла", "раму"]))
```

`load` is the expensive step: it opens memory-mapped tables. Do it once per
process and share the object; a loaded lexicon is immutable and safe to use
from several threads.

The input is a list of tokens; udlex does not tokenize. The first token of a
call counts as sentence-initial, which matters only for capitalised tokens: it
tells a capitalised ordinary word from a proper noun. To pass several
sentences in one call, mark where each begins:

```py
tokens = ["Мама", "мыла", "раму", ".", "Папа", "спал", "."]
starts = [True, False, False, False, True, False, False]
lemmas = lexicon.lemmatize(tokens, sentence_initial=starts)
```

or give a list of sentences to `lemmatize_many`, which releases the GIL once
for the whole batch.

To see where a lemma came from, ask for `info=True`:

```py
for info in lexicon.lemmatize(["мыла", "Ыыы"], info=True):
    print(info.lemma, info.source, info.source.is_known())
```

`source` is a `udlex.Source`: `exact`, `lower`, `external` and their variants
mean a table entry, `rule` a suffix guess, `identity` that nothing matched and
the token came back unchanged. To never guess, load with `use_rules=False`.

## 4. From the shell

```sh
udlex languages                  # installed dictionaries, with accuracy
udlex lemma rus Мама мыла раму   # one lemma per line
echo мыла | udlex lemma -q --json rus
```

`lemma` reads tokens from standard input, one per line, when none are given.
`--lookup-only` disables suffix guesses, `--with-info` and `--json` print the
source of each lemma.

## Where dictionaries are found

`udlex.load("<iso>")` and the command line look in this order:

1. the directory in `UDLEX_BUNDLE`, if set;
2. every installed dictionary wheel (the `udlex_bundle` namespace package);
3. `./bundle`.

`udlex.load(directory, "<iso>")` and `udlex --bundle directory` open one
directory and skip the search. `udlex.bundle_paths()` shows what will be
searched. When a language is missing, the error names the places it looked.

Load dictionaries only from sources you trust, and keep their directories
writable only by the accounts that manage them.

Next: the [API reference](api.md), [your own overrides](user-dictionary.md),
[building a dictionary](building-dictionaries.md).
