"""Aggregate aligned per-language contrasts, with a paired fixed-language block bootstrap."""

import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from verify_snapshot import read, save, sha

LABELS = ["declared", "learned", "single", "greedy_subset", "exact_subset", "dev_gated"]
SEED = 20261118
DRAWS = 2000


def gate(dev):
    c = dev["counts"][1]
    w, lo, n = c["wins"], c["losses"], c["tokens"]
    if w - lo < max(5, math.ceil(n / 10000)):
        return False
    m = w + lo
    term = math.exp(
        math.lgamma(m + 1) - math.lgamma(lo + 1) - math.lgamma(w + 1) - m * math.log(2)
    )
    tail = term
    for i in range(lo, 0, -1):
        term *= i / (m - i + 1)
        tail += term
        if term < tail * 1e-15:
            break
    return 2 * tail <= 0.05


def summarize(groups, bootstrap):
    rng = np.random.default_rng(SEED)
    totals = [Counter() for _ in LABELS]
    macro = np.zeros(len(LABELS))
    signs = [Counter() for _ in LABELS]
    delta_draws = np.zeros((DRAWS, len(LABELS)))
    reordered = 0
    changed_tests = 0
    passed = 0
    ext = Counter()
    for iso, runs in sorted(groups.items()):
        cs = [Counter() for _ in LABELS]
        res = np.zeros((DRAWS, 1 + len(LABELS)), dtype=np.int64)
        reordered += any(row["order"][0] != row["order"][1] for row in runs)
        for row in runs:
            trained = gate(row["dev"])
            passed += trained
            metrics = row["test"]["counts"]
            used = [0, 1, 2, 3, 4, int(trained)]
            for i, a in enumerate(used):
                cs[i].update(metrics[a])
            ext.update(row["test"]["external_hits"][1])
            if bootstrap:
                h = row["test"]["hist"]
                vals = np.array(
                    [[*v, v[1 + int(trained)]] for v, _ in h], dtype=np.int64
                )
                multiplicity = np.array([f for _, f in h], dtype=np.int64)
                if list(multiplicity @ vals) != [metrics[0]["tokens"]] + [
                    metrics[a]["wins"] - metrics[a]["losses"] for a in used
                ]:
                    raise ValueError(
                        f"Paired blocks disagree with aggregate counts for {iso}"
                    )
                for start in range(0, DRAWS, 40):
                    sample = rng.multinomial(
                        int(multiplicity.sum()),
                        multiplicity / multiplicity.sum(),
                        size=40,
                    )
                    res[start : start + 40] += sample @ vals
        n = cs[0]["tokens"]
        changed_tests += cs[1]["changed"] > 0
        for i, c in enumerate(cs):
            if (
                c["tokens"] != n
                or c["correct"] - cs[0]["correct"] != c["wins"] - c["losses"]
            ):
                raise ValueError(f"Inconsistent pairing for {iso}")
            totals[i].update(c)
            macro[i] += 100 * c["correct"] / n
            signs[i][int(np.sign(c["wins"] - c["losses"]))] += 1
        if bootstrap:
            delta_draws += 100 * res[:, 1:] / res[:, :1]
    macro /= len(groups)
    delta_draws /= len(groups)
    return {
        "languages": len(groups),
        "reordered_languages": reordered,
        "languages_with_changed_predictions": changed_tests,
        "gated_acceptances": passed,
        "external_hits_learned": dict(ext),
        "arms": {
            label: dict(
                totals[i],
                macro_accuracy=macro[i],
                micro_accuracy=100 * totals[i]["correct"] / totals[i]["tokens"],
                macro_delta_pp=macro[i] - macro[0],
                micro_delta_pp=100
                * (totals[i]["wins"] - totals[i]["losses"])
                / totals[i]["tokens"],
                ci95_pp=np.percentile(delta_draws[:, i], [2.5, 97.5]).tolist()
                if bootstrap
                else None,
                improved_tied_worse=[signs[i][k] for k in (1, 0, -1)],
            )
            for i, label in enumerate(LABELS)
        },
    }


def main(src, dest):
    m = read(src / "inputs/order/inventory.json")
    cv_ledger = read(src / "inputs/cv/ledger.json")
    groups = {"official": {}, "cross_validation": {}}
    ledger = {}
    for iso, row in m["languages"].items():
        cnt = len(row["sources"])
        status = "fewer than two sources"
        runs = []
        if row.get("status") == "paired":
            status = "official"
            runs = [read(src / "results" / (iso + ".json"))]
        elif iso in cv_ledger and min(cv_ledger[iso]["fold_tokens"]):
            status = "cross_validation"
            runs = [
                read(src / "cv-results" / str(k) / (iso + ".json")) for k in range(2)
            ]
        elif cnt >= 2:
            status = "no scorable independent pair"
        ledger[iso] = {
            "sources": cnt,
            "cohort": status,
            "treebank": row.get("treebank"),
        }
        if not runs:
            continue
        for r in runs:
            if (
                r["arms"] != LABELS[:5]
                or sorted(r["order"][0]) != sorted(r["order"][1])
                or len(set(r["order"][1])) != cnt
            ):
                raise ValueError(
                    f"Source membership changed in the primary contrast for {iso}"
                )
        groups[status][iso] = runs
        c = sum((Counter(r["test"]["counts"][1]) for r in runs), Counter())
        ledger[iso].update(
            tokens=c["tokens"],
            learned_correct=c["correct"],
            declared_correct=c["correct"] - c["wins"] + c["losses"],
            wins=c["wins"],
            losses=c["losses"],
            changed=c["changed"],
            fold_unit=cv_ledger.get(iso, {}).get("unit"),
        )
    results = {
        "official": summarize(groups["official"], bootstrap=True),
        "cross_validation": summarize(groups["cross_validation"], bootstrap=False),
        "combined": summarize(
            groups["official"] | groups["cross_validation"], bootstrap=False
        ),
        "source_inventory_sha256": sha(src / "inputs/order/inventory.json"),
        "binary": read(src / "results/run.json"),
        "cohorts": dict(Counter(v["cohort"] for v in ledger.values())),
        "bootstrap": {"seed": SEED, "replicates": DRAWS, "numpy": np.__version__},
    }
    save(dest / "summary.json", results)
    save(dest / "languages.json", ledger)
    save(dest / "cv-folds.json", cv_ledger)
    (dest / "pairs.json.gz").write_bytes(
        gzip.compress(
            json.dumps(
                groups, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode(),
            mtime=0,
        )
    )
    save(dest / "inventory.json", m)
    for name in ("official", "cross_validation", "combined"):
        value = results[name]
        print(name, value["languages"], value["arms"]["learned"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    main(a.run, a.output)
