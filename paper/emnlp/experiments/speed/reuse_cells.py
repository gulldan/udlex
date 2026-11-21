"""Copy finished competitor cells of an earlier run into OUT/cells.jsonl.

`bench.py run` then skips them. udlex cells are never copied: they depend on the
dictionaries being timed.
    reuse_cells.py FROM OUT
"""

import json
import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
langs = set((out / "languages.txt").read_text().split())
have = set()
if (out / "cells.jsonl").exists():
    with (out / "cells.jsonl").open() as old:
        have = {
            (c["lang"], c["system"], c["mode"], c["round"])
            for c in map(json.loads, old)
        }
copied = 0
with (
    (out / "cells.jsonl").open("a", encoding="utf-8") as fh,
    (src / "cells.jsonl").open() as cells,
):
    for c in map(json.loads, cells):
        key = (c["lang"], c["system"], c["mode"], c["round"])
        if (
            c["status"] != "ok"
            or c["system"].startswith("udlex")
            or c["lang"] not in langs
            or key in have
        ):
            continue
        have.add(key)
        fh.write(json.dumps(dict(c, reused_from=src.name), ensure_ascii=False) + "\n")
        copied += 1
print(f"copied {copied} cells from {src}")
