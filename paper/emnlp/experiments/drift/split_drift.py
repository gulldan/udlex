"""How many sentences of our UD 2.19 tests do older UD releases hold in train+dev or in test?

usage: python -I split_drift.py OUTPUT.json

Two keys per sentence: the exact sequence of word forms, and the fingerprint the experiment
uses to drop repeated sentences (forms joined without spaces, casefolded, NFC).
Counts are sentences of <iso>/test.jsonl found in a split of a treebank of the same
language code in the old release; "language" counts a sentence once over all its treebanks.
"""

import hashlib
import json
import os
import sys
import unicodedata
from pathlib import Path

COMPARE_DIR = os.environ["UDLEX_COMPARE"]
ORDER_RUN = os.environ["UDLEX_ORDER_RUN"]

C = Path(COMPARE_DIR)
OLD = C / "ud-old"
# Handles as listed on universaldependencies.org/download.html.
RELEASES = {
    "2.5": ("http://hdl.handle.net/11234/1-3105", OLD / "ud-treebanks-v2.5"),
    "2.8": ("http://hdl.handle.net/11234/1-3687", OLD / "ud-treebanks-v2.8"),
    "2.11": ("http://hdl.handle.net/11234/1-4923", OLD / "ud-treebanks-v2.11"),
    "2.17": ("http://hdl.handle.net/11234/1-6036", OLD / "ud-treebanks-v2.17"),
    "2.18": ("http://hdl.handle.net/11234/1-6149", OLD / "ud-treebanks-v2.18"),
    "2.19": (
        "http://hdl.handle.net/11234/1-6316",
        Path(f"{ORDER_RUN}/inputs/ud-treebanks-v2.19"),
    ),
}


def conllu(path):
    """Word forms of each sentence, the rows experiments/evaluate.py reads."""
    forms = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                if forms:
                    yield forms
                forms = []
            elif not line.startswith("#"):
                row = line.rstrip("\n\r").split("\t")
                if row[0].isdecimal():
                    forms.append(row[1])
    if forms:
        yield forms


def keys(forms):
    text = "".join("".join(t.split()) for t in forms).casefold()
    return (
        hashlib.sha256("\t".join(forms).encode()).digest()[:12],
        hashlib.sha256(unicodedata.normalize("NFC", text).encode()).digest()[:12],
    )


def index(root, code):
    """Per treebank of one language code: key sets of train+dev and of test."""
    found = {}
    for path in sorted(root.glob(f"UD_*/{code}_*-ud-*.conllu")):
        config, split = path.name.removesuffix(".conllu").split("-ud-")
        row = found.setdefault(
            path.parent.name,
            {"config": config, "train_dev": (set(), set()), "test": (set(), set())},
        )
        exact, loose = row["test" if split == "test" else "train_dev"]
        for forms in conllu(path):
            e, f = keys(forms)
            exact.add(e)
            loose.add(f)
    return found


def count(test, sets):
    return {
        "exact": sum(e in sets[0] for e, _ in test),
        "loose": sum(f in sets[1] for _, f in test),
    }


def main(output):
    inv = json.loads((C / "census/inputs/order/inventory.json").read_text())
    report = {
        "releases": {name: handle for name, (handle, _) in RELEASES.items()},
        "languages": {},
    }
    for iso, row in sorted(inv["languages"].items()):
        if row.get("status") != "paired":
            continue
        config = row["config"]
        with (C / "census/inputs/order" / iso / "test.jsonl").open() as fh:
            test = [keys(json.loads(line)["tokens"]) for line in fh]
        entry = {"config": config, "treebank": row["treebank"], "sentences": len(test)}
        for name, (_, root) in RELEASES.items():
            found = index(root, config.split("_")[0])
            union = {
                part: tuple(
                    set().union(*(t[part][k] for t in found.values())) for k in (0, 1)
                )
                for part in ("train_dev", "test")
            }
            entry[name] = {
                "same": next(
                    (tb for tb, t in found.items() if t["config"] == config), None
                ),
                "language": {p: count(test, union[p]) for p in union},
                "treebanks": {
                    tb: {
                        "config": t["config"],
                        "train_dev": count(test, t["train_dev"]),
                        "test": count(test, t["test"]),
                    }
                    for tb, t in found.items()
                },
            }
        # Self-check of the reader: every test sentence comes from the 2.19 test file.
        now = entry["2.19"]
        if now["treebanks"][now["same"]]["test"]["exact"] != len(test):
            raise ValueError(f"{iso}: test.jsonl is not a subset of the UD 2.19 test")
        report["languages"][iso] = entry
        print(
            iso,
            config,
            len(test),
            *(
                f"{name}:{entry[name]['language']['train_dev']['exact']}/{entry[name]['language']['test']['exact']}"
                for name in RELEASES
            ),
            flush=True,
        )
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
