# udlex

**English** · [Русский](https://github.com/gulldan/udlex/blob/main/README.ru.md) · [简体中文](https://github.com/gulldan/udlex/blob/main/README.zh-CN.md)

Fast local lemmatization for Python. It looks up word forms in a dictionary and
can apply suffix rules to unknown words. Input is a list of tokens.

The Go bindings live in [udlex-go](https://github.com/gulldan/udlex-go).
Build the OpenSearch 3.10 plugin from source or install it with all 224 dictionaries in Docker Compose. [Build the ZIP](integrations/opensearch/plugin/README.md) · [Install in Compose](integrations/opensearch/docker/README.md) · [Index and search ten languages in one index](integrations/opensearch/docker/example.http)
The [standalone Docker benchmark](benchmarks/opensearch/README.md) reproduces the comparison on 224 languages. [Measured CPU throughput on AMD64 and ARM64](benchmarks/opensearch/reports/README.md)

## Install

After the first PyPI release, add `udlex` to your project with `uv`.

```sh
uv add udlex
```

For an existing Python environment, use `python -m pip install udlex`.
Python 3.11 or later is required.

## Use

Build a small dictionary from a UTF-8 table with one `form<TAB>lemma` per line.
Save this as `quickstart.py` and run it with `uv run python quickstart.py`.

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

Use the same bundle from the command line.

```sh
uv run udlex --bundle example-bundle lemma -q eng cats mice
uv run udlex --bundle example-bundle languages
```

For an existing trusted dictionary bundle, pass its directory to `udlex.load`
and use its ISO 639-3 code. Load a lexicon once and reuse it for each batch.
The package ships the engine. You supply the dictionary.

## Native library

Build the C library on Linux or macOS with Rust and Cargo. It uses the same
compiled dictionary format and runs without Python.

```sh
cargo build --locked --release --manifest-path rust/Cargo.toml -p udlex-capi
```

Link `libudlex_capi` from `rust/target/release` and include
[udlex.h](rust/capi/include/udlex.h) in your application.

Code is licensed under [MIT](LICENSE).

[Dictionary packages](https://github.com/gulldan/udlex/releases/tag/v1.0.0) include
their own licenses and attribution. If you ship a dictionary with your
application, preserve that attribution and follow the dictionary's license.
Packages ending in `-nc` are for noncommercial use only.
