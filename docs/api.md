# API

The Python package is the full interface; its docstrings, in
[`src/udlex/_udlex.pyi`](../src/udlex/_udlex.pyi) and
[`src/udlex/__init__.py`](../src/udlex/__init__.py), are the reference for
every argument. This page is the map. The command line and the C library
expose a subset.

## Python

### Opening a dictionary

| Call | Does |
| --- | --- |
| `udlex.load("rus")` | Find an installed language by ISO 639-3 code (or treebank name such as `ru_taiga`) and open it. See [where dictionaries are found](quickstart.md#where-dictionaries-are-found). |
| `udlex.load(directory, "rus")` | Open a language from one bundle directory. |
| `udlex.Lexicon(path)` | Open one lexicon directory directly. |
| `udlex.supported_languages(directory=None)` | `Language` records (code, name, variant, accuracy, entries) without opening any table. |
| `udlex.bundle_paths()`, `udlex.bundle_path(iso)` | Where installed dictionaries are, and which directory holds a language. |

Options of `load`:

- `use_rules=True`: guess unknown words from suffix rules. `False` returns
  them unchanged.
- `use_external=True`: consult the merged dictionary table for forms the
  treebank tables do not cover.
- `user_bundle=None`: a [user dictionary](user-dictionary.md) root; defaults to
  `UDLEX_USER_BUNDLE`.

Other keyword arguments (`udlex.Probes`) switch single lookup stages off for
ablations; production code does not need them.

`load` raises `ValueError` when no installed dictionary has the language,
`FileNotFoundError` for a missing directory and `OSError` when a lexicon
cannot be opened or was written by an incompatible version.

### Lemmatizing

| Method | Input | Returns |
| --- | --- | --- |
| `lemmatize(tokens, *, sentence_initial=None, info=False)` | one or more sentences as a token list | a lemma per token |
| `lemmatize_many(sentences, *, info=False)` | a list of token lists | a list of lemma lists; GIL released once |
| `lemma(token, *, sentence_initial=False, info=False)` | one token | its lemma |
| `knows(token)` | one token | whether a table (not a rule) answers it |
| `lemma_span(mention, start=0, end=None, *, separators=None)` | an entity mention as text or tokens | its canonical form |

Shared arguments:

- `sentence_initial`: a bool per token for `lemmatize`, one bool for `lemma`.
  Without it, the first token of the call is sentence-initial. It only changes
  capitalised tokens.
- `info=True`: return `LemmaInfo(lemma, source)` instead of strings, with the
  same lookups.
- `protected` and `entities` (per-token masks for `lemmatize`; `protected`
  and `entity` flags for `lemma`): return a token unchanged, or look it up only
  in the entity table.
- `upos` and `context` on `lemmatize` pick between the lemmas of an ambiguous
  form by part of speech or neighbours. They only affect dictionaries built
  with context keys; the 1.0.0 packages are built without them.

A `str` passed as `tokens` is refused with `TypeError` instead of being read
as characters. A lemma spelled like its token is the same string object.

`lemma_span` is for named-entity mentions found by your own NER: it looks the
whole span up in the entity table first, then falls back to the token chain,
and returns the surface unchanged when nothing answers.

### Provenance

`LemmaInfo.source` is a `udlex.Source` member; its value is a stable lowercase
string for logs.

| Source | Meaning |
| --- | --- |
| `exact`, `lower`, `exact_lower` | treebank vocabulary, by exact or lowercased key |
| `external`, `external_lower` | the merged dictionary table |
| `entity`, `entity_lower` | the isolated entity table |
| `user`, `user_lower` | a user-dictionary override |
| `phrase`, `composed`, `context` | span patterns, multiword composition, a context key |
| `rule` | a suffix guess |
| `identity` | nothing matched; the token is returned |
| `protected` | the caller asked to keep the token |

`Source.is_known()` is true when a stored entry answered;
`Source.is_lemmatized()` is false only for `identity` and `protected`.

### Lexicon metadata

A loaded `Lexicon` reports `name`, `language`, `iso639_3`, `variant`
(which stages the package ships: `full`, `lookup` without suffix rules, `rules`
without the dictionary table, `corpus` without both, `identity` without any),
`accuracy` and `baseline` (test accuracy and copy baseline, `None` if
unmeasured), `generation` and `format_version`, and, with a user dictionary,
`has_user_dictionary` and `user_entries`.

### Building

`udlex.build_lexicon(destination, train=..., dev=..., test=..., paradigms=..., name=..., language=..., iso639_3=...)`
builds and publishes one lexicon; see [building dictionaries](building-dictionaries.md).
`udlex.compile_bundle` and `udlex.prune_bundle` recompile or strip retained
build sources of a local lexicon.

### Environment

| Variable | Effect |
| --- | --- |
| `UDLEX_BUNDLE` | Use this bundle directory instead of the installed wheels. |
| `UDLEX_USER_BUNDLE` | Default user-dictionary root for `load` and the command line. |
| `UDLEX_DICTIONARIES` | Base URL or local directory `udlex install` takes the wheels from (default: the v1.0.0 release). |

## Command line

`udlex [--bundle DIR] [--user-bundle DIR] [--no-user] COMMAND`. Run
`udlex COMMAND --help` for every option.

| Command | Does |
| --- | --- |
| `install ISO...` | Install dictionary wheels from the v1.0.0 release with `uv pip` bound to this interpreter, or its pip when uv is absent; in a uv project, print the `uv add` command that keeps them. `--dry-run` prints the wheel locations. |
| `languages [PATTERN]` | List dictionaries with accuracy, copy baseline, entries and variant. |
| `lemma ISO [TOKEN...]` | Print one lemma per token; reads standard input when no token is given. `-q` drops the header, `--lookup-only` disables guesses, `--with-info`/`--json` add the source, `--entity`/`--protected` as in Python. |
| `user-dict ...` | Manage [overrides](user-dictionary.md): `init`, `set`, `remove`, `import`, `list`, `history`, `rollback`, `status`, `verify`. |
| `verify` | Open every lexicon of a bundle and report failures. |
| `compile [ISO...]` | Compile retained source tables of a local bundle; `--prune` deletes them afterwards. |

Put options before the language or after the tokens, not between tokens.

## C API

The C library reads the same dictionaries without Python. Build it with Rust
and Cargo:

```sh
cargo build --locked --release --manifest-path rust/Cargo.toml -p udlex-capi
```

Link `libudlex_capi` from `rust/target/release` and include
[`rust/capi/include/udlex.h`](../rust/capi/include/udlex.h). All strings are
UTF-8 byte ranges with explicit lengths, without terminators.

| Function | Does |
| --- | --- |
| `udlex_native_load(bundle, len, language, len, use_rules, &out)` | Open a language in a bundle directory. |
| `udlex_native_open(path, len, &out)` | Open one lexicon directory. |
| `udlex_native_lemma(lexicon, token, len, sentence_initial, out, cap, &out_len)` | Write the lemma of one token into `out`. |
| `udlex_native_close(lexicon)` | Close a handle; `NULL` is allowed. |
| `udlex_native_version()` | Static version string. |

Return codes: `UDLEX_NATIVE_OK` (0); `UDLEX_NATIVE_MORE_SPACE` (1), with
`out_len` set to the capacity needed and `out` untouched; and negative errors
`INVALID_ARGUMENT`, `INVALID_UTF8`, `OPEN_FAILED`, `PANIC`. A handle may be read
from several threads at once, each with its own buffer; keep the dictionary
files unchanged until every handle is closed.

```c
#include <stdio.h>
#include <string.h>

#include "udlex.h"

int main(void) {
  udlex_native_lexicon *lexicon = NULL;
  const char *bundle = "bundle", *language = "eng", *token = "mice";
  if (udlex_native_load((const uint8_t *)bundle, strlen(bundle),
                        (const uint8_t *)language, strlen(language), 1,
                        &lexicon) != UDLEX_NATIVE_OK)
    return 1;
  uint8_t lemma[256];
  size_t length = 0;
  if (udlex_native_lemma(lexicon, (const uint8_t *)token, strlen(token), 1,
                         lemma, sizeof lemma, &length) == UDLEX_NATIVE_OK)
    printf("%.*s\n", (int)length, (const char *)lemma);
  udlex_native_close(lexicon);
  return 0;
}
```

For a lemma longer than the buffer, the call returns `UDLEX_NATIVE_MORE_SPACE`;
allocate `length` bytes and call again. The C library has no user-dictionary or
span functions.

## Other bindings

- Go: [gulldan/udlex-go](https://github.com/gulldan/udlex-go).
