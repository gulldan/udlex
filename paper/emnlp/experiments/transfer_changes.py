"""List every prediction that the learned order changes, from a diagnostic build of the same engine."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from verify_snapshot import read, save

TOP = 10


def main(a):
    corpora = read(a.evidence / "corpora.json")
    primary = {
        i for i, v in read(a.evidence / "languages.json").items() if v["primary"]
    }
    kinds = defaultdict(Counter)
    for cfg, row in sorted(corpora.items()):
        if row["group"] != "gold" or row["iso"] not in primary:
            continue
        frozen = read(a.results / cfg / (row["iso"] + ".json"))
        listed = read(a.diagnostic / cfg / (row["iso"] + ".json"))
        if any(listed[k] != frozen[k] for k in ("order", "train_correct")) or any(
            listed["test"][k] != frozen["test"][k] for k in ("counts", "hist")
        ):
            raise ValueError(f"The diagnostic build predicts differently on {cfg}")
        rows = listed["test"]["examples"]
        if len(rows) != row["changed"]:
            raise ValueError(f"Incomplete list of changed predictions for {cfg}")
        for e in rows:
            outcome = (
                "win"
                if e["learned"] == e["gold"]
                else "loss"
                if e["declared"] == e["gold"]
                else "both_wrong"
            )
            kinds[row["iso"]][
                (outcome, e["form"], e["gold"], e["declared"], e["learned"])
            ] += 1
    report = {}
    for iso, types in sorted(kinds.items()):
        entry = report[iso] = {}
        for outcome in ("win", "loss", "both_wrong"):
            mine = Counter({k[1:]: n for k, n in types.items() if k[0] == outcome})
            top = mine.most_common(TOP)
            entry[outcome] = {
                "tokens": sum(mine.values()),
                "types": len(mine),
                "case_only": sum(
                    n for (_, _, d, l), n in mine.items() if d.lower() == l.lower()
                ),
                "top_type_tokens": top[0][1] if top else 0,
                "top": [
                    dict(
                        zip(("form", "gold", "declared", "learned"), k, strict=True),
                        tokens=n,
                    )
                    for k, n in top
                ],
            }
    save(a.output, report)
    for iso, v in report.items():
        w, l = v["win"], v["loss"]
        if w["tokens"] + l["tokens"] >= 200:
            print(
                iso,
                "wins",
                w["tokens"],
                w["types"],
                "top",
                w["top_type_tokens"],
                "| losses",
                l["tokens"],
                l["types"],
                "top",
                l["top_type_tokens"],
                (l["top"] or [{}])[0].get("form"),
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("evidence", "results", "diagnostic", "output"):
        p.add_argument(key, type=Path)
    main(p.parse_args())
