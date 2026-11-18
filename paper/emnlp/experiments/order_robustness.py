"""Stress the order contrasts: flagged sources, languages as the sampling unit, the confirmation rule."""

import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from order_summary import SEED, summarize
from verify_snapshot import read, save

DRAWS = 10000
ALPHAS = (0.01, 0.05, 0.10, 1.0)


def binomial(wins, losses):
    """Two-sided exact p for an even split of wins and losses."""
    m, k = wins + losses, min(wins, losses)
    if not m:
        return 1.0
    tail = sum(math.comb(m, i) for i in range(k + 1)) / 2**m
    return min(1.0, 2 * tail)


def counts(runs, arm=1):
    return sum((Counter(r["test"]["counts"][arm]) for r in runs), Counter())


def accepts(dev, alpha, floor):
    c = dev["counts"][1]
    least = max(5, math.ceil(c["tokens"] / 10000)) if floor else 1
    return (
        c["wins"] - c["losses"] >= least and binomial(c["wins"], c["losses"]) <= alpha
    )


def by_language(groups, choose=None):
    """Per-language change in accuracy (pp) and net corrections for one policy."""
    rows = {}
    for iso, runs in groups.items():
        used = [r for r in runs if choose is None or choose(r["dev"])]
        c = counts(used)
        n = counts(runs)["tokens"]
        rows[iso] = (100 * (c["wins"] - c["losses"]) / n, c["wins"], c["losses"])
    return rows


def inference(rows, rng):
    change = np.array([v[0] for v in rows.values()])
    up, down = int((change > 0).sum()), int((change < 0).sum())
    draws = rng.choice(change, size=(DRAWS, len(change))).mean(axis=1)
    return {
        "languages": len(change),
        "macro_delta_pp": float(change.mean()),
        "improved_tied_worse": [up, len(change) - up - down, down],
        "sign_test_p": binomial(up, down),
        "language_bootstrap_ci95_pp": np.percentile(draws, [2.5, 97.5]).tolist(),
        "corrections": sum(v[1] for v in rows.values()),
        "regressions": sum(v[2] for v in rows.values()),
    }


def stress(groups, flagged):
    rng = np.random.default_rng(SEED)
    confirmed = lambda dev: accepts(dev, 0.05, True)  # noqa: E731
    clean = {i: v for i, v in groups.items() if i not in flagged}
    return {
        "learned": inference(by_language(groups), rng),
        "confirmed": inference(by_language(groups, confirmed), rng),
        "flagged_languages": sorted(set(groups) & flagged),
        "without_flagged": summarize(clean, True)["arms"] if clean else None,
        "confirmation_grid": [
            dict(
                alpha=alpha,
                net_floor=floor,
                accepted=sum(
                    any(accepts(r["dev"], alpha, floor) for r in v)
                    for v in groups.values()
                ),
                **inference(
                    by_language(
                        groups,
                        lambda dev, alpha=alpha, floor=floor: accepts(
                            dev, alpha, floor
                        ),
                    ),
                    rng,
                ),
            )
            for alpha in ALPHAS
            for floor in (True, False)
        ],
    }


def main(a):
    proof = read(a.provenance)["pairs"]
    flagged = {
        v["iso"] for k, v in proof.items() if v["flagged"] and ":" in k.split("|")[0]
    }
    home = json.loads(gzip.decompress((a.order / "pairs.json.gz").read_bytes()))
    away = json.loads(gzip.decompress((a.transfer / "pairs.json.gz").read_bytes()))[
        "gold"
    ]
    primary = {
        i for i, v in read(a.transfer / "languages.json").items() if v["primary"]
    }
    official = {i: v for i, v in away.items() if i in primary and i in home["official"]}
    report = {
        "seed": SEED,
        "language_draws": DRAWS,
        "screened_splits": sorted(
            {k.split("|")[0].split(":")[1] for k in proof if ":" in k.split("|")[0]}
        ),
        "flagged_anchor_pairs": sorted(
            k for k, v in proof.items() if v["flagged"] and ":" in k.split("|")[0]
        ),
        "anchor_official": stress(home["official"], flagged),
        "anchor_cross_validation": stress(home["cross_validation"], flagged),
        "transfer_official_fit": stress(official, flagged),
        "anchor_same_languages": stress(
            {i: home["official"][i] for i in official}, flagged
        ),
    }
    save(a.output, report)
    for name in ("anchor_official", "transfer_official_fit"):
        for arm in ("learned", "confirmed"):
            v = report[name][arm]
            print(
                name,
                arm,
                v["improved_tied_worse"],
                round(v["sign_test_p"], 4),
                [round(x, 3) for x in v["language_bootstrap_ci95_pp"]],
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("order", "transfer", "provenance", "output"):
        p.add_argument(key, type=Path)
    main(p.parse_args())
