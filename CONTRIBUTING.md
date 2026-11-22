# Contributing

Bug reports, fixes and new dictionary sources are welcome. Open an issue first
for anything larger than a fix, so we can agree on the approach. Report
security problems privately, as described in [SECURITY.md](SECURITY.md).

By contributing you agree that your code is released under the
[MIT license](LICENSE). Dictionary data keeps the license of its source; see
[DICTIONARY_LICENSES.md](DICTIONARY_LICENSES.md).

## Setup

You need [uv](https://docs.astral.sh/uv/) and the Rust toolchain pinned in
`rust-toolchain.toml` (rustup installs it on first use).

```sh
git clone https://github.com/gulldan/udlex
cd udlex
uv run --frozen pytest -q     # builds the extension with maturin, then tests
```

After changing Rust code, rebuild the extension with
`uv run --frozen maturin develop --release` before running Python tests.

## Checks

Run what CI runs before you open a pull request:

```sh
uv run --frozen ruff check
uv run --frozen ruff format --check
uv run --frozen ty check src/udlex tests
uv run --frozen pytest -q
cargo test --locked --manifest-path rust/Cargo.toml --workspace \
  --features udlex-rs/train,udlex-rs/user-dictionary-write
```

CI also executes the single ```` ```python ```` block in `README.md` and builds
the C smoke test against the dictionary it creates. Keep exactly one such block
there; other examples use ```` ```py ````.

## Documentation

- [`docs/languages.md`](docs/languages.md) and
  [`DICTIONARY_LICENSES.md`](DICTIONARY_LICENSES.md) are generated from the
  catalog. Edit [`dictionaries/pages.py`](dictionaries/pages.py), not the
  pages, and run `uv run --frozen python dictionaries/pages.py`;
  `--check` fails when a page is stale.
- Numbers about accuracy and speed come from committed receipts:
  [`dictionaries/catalog`](dictionaries/catalog) and
  [`benchmarks/generated/numbers.tex`](benchmarks/generated/numbers.tex).
  Do not type new numbers by hand.
- `README.ru.md` and `README.zh-CN.md` translate `README.md`; update them in
  the same change.

## Dictionaries

Dictionary packages are built by the pipeline in
[`dictionaries/`](dictionaries/README.md), never by hand. A new source needs a
license that allows redistribution (copyleft and unclear licenses are
excluded by [`policy.json`](dictionaries/policy.json)), a pinned URL and
SHA-256 in [`sources.json`](dictionaries/sources.json), and a rebuild whose
`evaluation.json` shows what it changes. Every input is used under the rules in
[Data use](dictionaries/README.md#data-use): test splits are read only by the
evaluation.

## Pull requests

Keep a change to one topic, with tests for new behaviour. Write commit
messages in the imperative ("Add ...", "Fix ..."), and add a line to
[CHANGELOG.md](CHANGELOG.md) for anything a user would notice.
