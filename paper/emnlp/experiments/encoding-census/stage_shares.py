"""Share of UD 2.19 test tokens answered by each cascade stage of the 1.0.0.post1 catalog.

usage: exact_hits.py BUNDLES ORDER_INPUTS OUT.json
"""

import collections
import json
import sys
from pathlib import Path

import udlex

bundles, inputs, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
result = {}
for lang_dir in sorted(p for p in inputs.iterdir() if (p / "test.jsonl").exists()):
    iso = lang_dir.name
    if not (bundles / iso).is_dir():
        continue
    lex = udlex.Lexicon(str(bundles / iso))
    c = collections.Counter()
    with (lang_dir / "test.jsonl").open() as f:
        for line in f:
            s = json.loads(line)
            for i, (tok, gold) in enumerate(zip(s["tokens"], s["lemmas"], strict=True)):
                if gold is None:
                    continue
                c[lex.lemma(tok, sentence_initial=i == 0, info=True).source.value] += 1
    n = sum(c.values())
    if not n:
        print(iso, "no lemmatised test tokens", flush=True)
        continue
    result[iso] = {"tokens": n, **dict(sorted(c.items()))}
    print(iso, n, f"exact {100 * c['exact'] / n:.1f}%", flush=True)
out.write_text(json.dumps(result, indent=1, sort_keys=True))
