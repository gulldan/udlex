"""List and classify every prediction the learned order changes, from a diagnostic build."""

import argparse
import csv
import gzip
import hashlib
import json
import os
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from verify_snapshot import read, save

TOP = 10
SAMPLE = 12
KINDS = ("case", "diacritics", "copy", "nested", "ending", "other")
OUTCOMES = ("win", "loss", "both_wrong")
ROOM = ("external", "disputed", "declared_correct", "learned_correct", "oracle_correct")


def plain(text):
    marks = unicodedata.normalize("NFD", text.casefold())
    return "".join(c for c in marks if unicodedata.category(c) != "Mn")


def kind(form, declared, learned):
    """How the two answers differ; the first matching description wins."""
    a, b = declared.casefold(), learned.casefold()
    if a == b:
        return "case"
    if plain(declared) == plain(learned):
        return "diacritics"
    if form.casefold() in (a, b):
        return "copy"  # one source returns the word itself
    short, long = sorted((a, b), key=len)
    if long.startswith(short):
        return "nested"  # a shorter and a longer base, such as stem and infinitive
    shared = len(os.path.commonprefix([short, long]))
    if shared >= 3 and 2 * shared >= len(long):
        return "ending"
    return "other"


def pairs(a):
    """Yield (language, cohort, frozen run, diagnostic run) for one setting."""
    if a.setting == "anchor":
        released = json.loads(
            gzip.decompress((a.evidence / "pairs.json.gz").read_bytes())
        )
        for iso, runs in released["official"].items():
            yield (
                iso,
                "official",
                runs[0],
                read(a.diagnostic / "results" / f"{iso}.json"),
            )
        for iso, runs in released["cross_validation"].items():
            for k, run in enumerate(runs):
                listed = read(a.diagnostic / "cv-results" / str(k) / f"{iso}.json")
                yield iso, "cross_validation", run, listed
    else:
        primary = {
            i for i, v in read(a.evidence / "languages.json").items() if v["primary"]
        }
        for cfg, row in sorted(read(a.evidence / "corpora.json").items()):
            if row["group"] == "gold" and row["iso"] in primary:
                iso = row["iso"]
                cohort = row["fit"] + "_fit"
                frozen = read(a.results / cfg / f"{iso}.json")
                yield iso, cohort, frozen, read(a.diagnostic / cfg / f"{iso}.json")


def main(a):
    types = defaultdict(Counter)
    room = defaultdict(Counter)
    cohorts = {}
    for iso, cohort, frozen, listed in pairs(a):
        cohorts[iso] = cohort
        same = all(listed[k] == frozen[k] for k in ("order", "train_correct"))
        if not same or any(
            listed["test"][k] != frozen["test"][k] for k in ("counts", "hist")
        ):
            raise ValueError(f"The diagnostic build predicts differently for {iso}")
        c = frozen["test"]["counts"][1]
        h = listed["headroom"]["test"]
        if len(listed["test"]["examples"]) != c["changed"] or h["unmatched"]:
            raise ValueError(f"Incomplete diagnostic output for {iso}")
        if h["learned_correct"] - h["declared_correct"] != c["wins"] - c["losses"]:
            raise ValueError(f"A change outside the disputed keys for {iso}")
        room[iso].update({k: h[k] for k in ROOM})
        for e in listed["test"]["examples"]:
            outcome = (
                "win"
                if e["learned"] == e["gold"]
                else "loss"
                if e["declared"] == e["gold"]
                else "both_wrong"
            )
            types[iso][
                (outcome, e["form"], e["gold"], e["declared"], e["learned"])
            ] += 1
    languages, totals, rows = {}, {}, []
    for iso in sorted(cohorts):
        entry = languages[iso] = {"cohort": cohorts[iso], "headroom": dict(room[iso])}
        kinds = {k: dict.fromkeys(OUTCOMES, 0) for k in KINDS}
        for outcome in OUTCOMES:
            mine = Counter({k[1:]: n for k, n in types[iso].items() if k[0] == outcome})
            for (form, gold, declared, learned), n in mine.items():
                label = kind(form, declared, learned)
                kinds[label][outcome] += n
                rows.append((iso, label, outcome, form, gold, declared, learned, n))
            top = mine.most_common(TOP)
            entry[outcome] = {
                "tokens": sum(mine.values()),
                "types": len(mine),
                "top": [
                    dict(
                        zip(("form", "gold", "declared", "learned"), k, strict=True),
                        tokens=n,
                    )
                    for k, n in top
                ],
            }
        entry["kinds"] = kinds
        total = totals.setdefault(
            cohorts[iso],
            {
                "languages": 0,
                "headroom": Counter(),
                "kinds": {k: Counter() for k in KINDS},
            },
        )
        total["languages"] += 1
        total["headroom"].update(room[iso])
        for k in KINDS:
            total["kinds"][k].update(kinds[k])
    save(a.output / f"{a.setting}.json", {"totals": totals, "languages": languages})
    # A fixed sample of types for annotators: per outcome and kind, in digest order.
    strata = defaultdict(list)
    for row in rows:
        if row[2] != "both_wrong":
            digest = hashlib.sha256("\t".join(map(str, row[:7])).encode()).digest()
            strata[row[1], row[2]].append((digest, row))
    with (a.output / f"sample-{a.setting}.tsv").open("w", newline="") as fh:
        out = csv.writer(fh, delimiter="\t", lineterminator="\n")
        out.writerow(
            [
                "language",
                "kind",
                "outcome",
                "form",
                "gold",
                "declared",
                "learned",
                "tokens",
                "judgment",
            ]
        )
        for key in sorted(strata):
            for _, row in sorted(strata[key])[:SAMPLE]:
                out.writerow([*row, ""])
    for cohort, total in totals.items():
        print(cohort, total["languages"], dict(total["headroom"]))
        for k in KINDS:
            print("  ", k, dict(total["kinds"][k]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("setting", choices=("anchor", "transfer"))
    p.add_argument(
        "evidence", type=Path, help="released order evidence or transfer evidence"
    )
    p.add_argument("diagnostic", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--results", type=Path, help="frozen transfer results")
    main(p.parse_args())
