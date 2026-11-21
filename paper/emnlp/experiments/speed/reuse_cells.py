"""Copy finished competitor cells of an earlier run into OUT/cells.jsonl, so `bench.py run`
skips them. udlex cells are never copied: they depend on the dictionaries being timed.
    reuse_cells.py FROM OUT
"""

import json
import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
langs = set((out / "languages.txt").read_text().split())
have = set()
if (out / "cells.jsonl").exists():
    have = {(c["lang"], c["system"], c["mode"], c["round"]) for c in map(json.loads, open(out / "cells.jsonl"))}
copied = 0
with open(out / "cells.jsonl", "a", encoding="utf-8") as fh:
    for c in map(json.loads, open(src / "cells.jsonl")):
        key = (c["lang"], c["system"], c["mode"], c["round"])
        if c["status"] != "ok" or c["system"].startswith("udlex") or c["lang"] not in langs or key in have:
            continue
        have.add(key)
        fh.write(json.dumps(dict(c, reused_from=src.name), ensure_ascii=False) + "\n")
        copied += 1
print(f"copied {copied} cells from {src}")
