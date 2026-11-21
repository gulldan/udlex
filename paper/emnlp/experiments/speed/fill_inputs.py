"""Timing inputs: the census inputs, with UD 2.19 test text for three twofold languages.

The census test of bul, fro and tha is empty; timing needs running text, not a held-out
split.
    fill_inputs.py OUT   (server) OUT gets symlinks to the census dirs plus three real dirs
UDLEX_SPEED_COMPARE is the quality comparison, UDLEX_SPEED_UD the UD 2.19 treebanks.
"""

import json
import os
import sys
from pathlib import Path

for name in ("UDLEX_SPEED_COMPARE", "UDLEX_SPEED_UD"):
    if not os.environ.get(name):
        raise SystemExit(f"{name} is not set (see the docstring)")
CENSUS = Path(os.environ["UDLEX_SPEED_COMPARE"], "census/inputs/order")
UD = Path(os.environ["UDLEX_SPEED_UD"])
# The treebank each language was scored on (evaluation.json "treebank").
FILL = {
    "bul": "UD_Bulgarian-BTB/bg_btb-ud-test.conllu",
    "fro": "UD_Old_French-PROFITEROLE/fro_profiterole-ud-test.conllu",
    "tha": "UD_Thai-PUD/th_pud-ud-test.conllu",
}


def sentences(path):
    tokens, lemmas, n = [], [], 0
    for raw in path.open(encoding="utf-8"):
        line = raw.rstrip("\n")
        if not line:
            if tokens:
                n += 1
                yield {"tokens": tokens, "lemmas": lemmas, "block": f"s:{n}"}
            tokens, lemmas = [], []
            continue
        cols = line.split("\t")
        if line.startswith("#") or not cols[0].isdigit():  # MWT ranges and empty nodes
            continue
        tokens.append(cols[1])
        lemmas.append(None if cols[2] == "_" and cols[1] != "_" else cols[2])
    if tokens:
        yield {"tokens": tokens, "lemmas": lemmas, "block": f"s:{n + 1}"}


out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
for src in CENSUS.iterdir():
    if src.name not in FILL and not (out / src.name).exists():
        (out / src.name).symlink_to(src)
for iso, rel in FILL.items():
    assert (CENSUS / iso / "test.jsonl").stat().st_size == 0, f"{iso}: census test is not empty"
    rows = list(sentences(UD / rel))
    assert rows
    assert all(len(r["tokens"]) == len(r["lemmas"]) for r in rows)
    (out / iso).mkdir(exist_ok=True)
    with (out / iso / "test.jsonl").open("w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(iso, len(rows), "sentences", sum(len(r["tokens"]) for r in rows), "tokens", rel)
