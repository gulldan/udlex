# udlex

Fast local lemmatization for Python. It looks up word forms in a dictionary and
can apply suffix rules to unknown words. Input is a list of tokens.

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

Code is licensed under [MIT](LICENSE).
