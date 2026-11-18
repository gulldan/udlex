"""Exploratory: the cascade and other engines on every catalog language, by stratum of test words.

usage: census.py CENSUS PREDICTIONS OUTPUT [ENGINE...]

CENSUS holds inputs/order, inputs/cv/{0,1}, results and cv-results/{0,1} of a diagnostic run
(prepare_order_native.py --examples); PREDICTIONS holds ENGINE/<iso>.json.gz from compare_engines.py.
"""

import collections
import gzip
import json
import sys
from pathlib import Path

STRATA = (
    "seen, one lemma",
    "seen, several lemmas",
    "unseen, corpus lowercase",
    "unseen, in a dictionary",
    "unseen, rule or copy",
)


def rows(path):
    with open(path) as f:
        for line in f:
            s = json.loads(line)
            yield s["tokens"], s["lemmas"]


def language(inputs, result, theirs):
    """Counts per stratum: words, cascade right, and per engine right / either right."""
    answers = {a[0]: a[1:] for a in result["answers"]}
    seen = collections.defaultdict(set)
    for tokens, gold in rows(inputs / "train.jsonl"):
        for f, y in zip(tokens, gold, strict=True):
            if y is not None:
                seen[f].add(y)
    n, ours = [0] * 5, [0] * 5
    engines = {e: {"ok": [0] * 5, "either": [0] * 5} for e in theirs}
    votes = collections.defaultdict(collections.Counter)
    for s, (tokens, gold) in enumerate(rows(inputs / "test.jsonl")):
        for i, (f, y) in enumerate(zip(tokens, gold, strict=True)):
            if y is None:
                continue
            a = answers[f]
            text, stage = (a[2], a[3]) if i == 0 else (a[0], a[1])
            if f in seen:
                k = 0 if len(seen[f]) == 1 else 1
            elif stage in (2, 3, 4):
                k = 2
            elif stage in (5, 6):
                k = 3
            else:
                k = 4
            n[k] += 1
            ours[k] += text == y
            votes[f][y] += 1
            for e, sentences in theirs.items():
                engines[e]["ok"][k] += sentences[s][i] == y
                engines[e]["either"][k] += text == y or sentences[s][i] == y
    if sum(ours) != result["test"]["counts"][0]["correct"]:
        raise ValueError(f"{inputs}: answers disagree with the scored run")
    ceiling = sum(max(c.values()) for c in votes.values())
    return {"n": n, "cascade": ours, "ceiling": ceiling, "engines": engines}


def add(a, b):
    if not a:
        return b
    return {
        "n": [x + y for x, y in zip(a["n"], b["n"], strict=True)],
        "cascade": [x + y for x, y in zip(a["cascade"], b["cascade"], strict=True)],
        "ceiling": a["ceiling"] + b["ceiling"],
        "engines": {},
    }


def acc(right, words):
    return 100 * right / words if words else float("nan")


def group(name, cells):
    if not cells:
        return
    words = sum(sum(c["n"]) for c in cells)
    macro = sum(acc(sum(c["cascade"]), sum(c["n"])) for c in cells) / len(cells)
    top = sum(acc(c["ceiling"], sum(c["n"])) for c in cells) / len(cells)
    unseen = sum(acc(sum(c["n"][2:]), sum(c["n"])) for c in cells) / len(cells)
    print(
        f"{name:28s} langs={len(cells):3d} words={words:9,d} cascade macro={macro:6.2f}"
        f" pooled={acc(sum(sum(c['cascade']) for c in cells), words):6.2f}"
        f" ceiling macro={top:6.2f} unseen share macro={unseen:5.2f}"
    )


def main():
    census, predictions, output, *names = sys.argv[1:]
    census, predictions = Path(census), Path(predictions)
    inv = json.loads((census / "inputs/order/inventory.json").read_text())["languages"]
    report = {}
    for path in sorted((census / "results").glob("*.json")):
        iso = path.stem
        if iso == "run":
            continue
        theirs = {}
        for e in names:
            file = predictions / e / f"{iso}.json.gz"
            if file.exists():
                theirs[e] = json.loads(gzip.decompress(file.read_bytes()))
        cell = language(
            census / "inputs/order" / iso, json.loads(path.read_text()), theirs
        )
        report[iso] = {
            **cell,
            "split": "official",
            "sources": len(inv[iso]["sources"]),
        }
    for path in sorted((census / "cv-results/0").glob("*.json")):
        iso = path.stem
        if iso == "run":
            continue
        cell = None
        for k in ("0", "1"):
            result = json.loads((census / "cv-results" / k / f"{iso}.json").read_text())
            cell = add(cell, language(census / "inputs/cv" / k / iso, result, {}))
        report[iso] = {**cell, "split": "two-fold", "sources": len(inv[iso]["sources"])}
    Path(output).write_text(json.dumps(report, sort_keys=True))

    home = [c for c in report.values() if c["split"] == "official"]
    group("official test, all", home)
    for label, keep in (
        ("  no dictionary", lambda s: s == 0),
        ("  one dictionary", lambda s: s == 1),
        ("  two or more", lambda s: s >= 2),
    ):
        group(label, [c for c in home if keep(c["sources"])])
    away = [c for c in report.values() if c["split"] == "two-fold"]
    group("two-fold, all", away)
    group("  no dictionary", [c for c in away if c["sources"] == 0])
    group("  with a dictionary", [c for c in away if c["sources"] > 0])
    group("every language with gold", home + away)

    for label, cells in (("official", home), ("two-fold", away)):
        words = sum(sum(c["n"]) for c in cells)
        print(f"\nstrata, {label} ({len(cells)} languages, {words:,} words)")
        for k, name in enumerate(STRATA):
            a = sum(c["n"][k] for c in cells)
            print(
                f"  {name:26s} share={acc(a, words):6.2f}"
                f" cascade={acc(sum(c['cascade'][k] for c in cells), a):6.2f}"
            )

    for e in names:
        cells = [c for c in home if e in c["engines"]]
        if not cells:
            continue
        mine = [acc(sum(c["cascade"]), sum(c["n"])) for c in cells]
        other = [acc(sum(c["engines"][e]["ok"]), sum(c["n"])) for c in cells]
        words = sum(sum(c["n"]) for c in cells)
        print(
            f"\n{e}: {len(cells)} languages, {words:,} words; macro {sum(other) / len(cells):.2f}"
            f" vs cascade {sum(mine) / len(cells):.2f}; above the cascade in"
            f" {sum(o > m for o, m in zip(other, mine, strict=True))}, below in"
            f" {sum(o < m for o, m in zip(other, mine, strict=True))}"
        )
        for k, name in enumerate(STRATA):
            a = sum(c["n"][k] for c in cells)
            print(
                f"  {name:26s} share={acc(a, words):6.2f}"
                f" cascade={acc(sum(c['cascade'][k] for c in cells), a):6.2f}"
                f" {e}={acc(sum(c['engines'][e]['ok'][k] for c in cells), a):6.2f}"
                f" either={acc(sum(c['engines'][e]['either'][k] for c in cells), a):6.2f}"
            )


if __name__ == "__main__":
    main()
