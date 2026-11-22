# Changelog

The engine (`udlex` on PyPI) and the dictionary packages are versioned
together; a dictionary package declares the engine versions it works with.

## [1.0.0] - 2026-11-22

First public release.

### Engine

- Rust core that lemmatizes a token from memory-mapped compiled tables: the
  treebank vocabulary, a merged dictionary table, an isolated entity table and
  suffix rules for unknown words. Loaded lexicons are immutable and safe to
  share across threads.
- Python package (PyO3, stable ABI for CPython 3.11 and newer, plus
  free-threaded builds): `udlex.load`, `Lexicon.lemmatize`, `lemmatize_many`,
  `lemma`, `lemma_span` for entity mentions, and the provenance of every lemma
  (`info=True`, `udlex.Source`).
- Command line: `udlex install`, `languages`, `lemma`, `user-dict`, `verify`,
  `compile`.
- User dictionaries: transactional, audited overrides kept outside the shipped
  packages, with history, rollback and verification.
- C library (`udlex-capi`, header `rust/capi/include/udlex.h`) on the same
  dictionary format, without Python.
- Dictionary format 5.

### Dictionaries

- 224 language packages built from Universal Dependencies 2.19 and pinned open
  dictionaries, published as wheels and archives on the GitHub release, not on
  PyPI. Each package carries a `NOTICE`, the treebank's README and LICENSE,
  and `SOURCES.json` with the license of every source.
- 39 packages are non-commercial (`-nc`) because of their treebank's license.
- Accuracy of every package is in [docs/languages.md](docs/languages.md); the
  build pipeline is in [dictionaries/](dictionaries/README.md).

### Integrations

- Go bindings in the separate repository gulldan/udlex-go.

[1.0.0]: https://github.com/gulldan/udlex/releases/tag/v1.0.0
