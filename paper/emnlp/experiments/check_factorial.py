"""Cross-check clean fits and compute the two-by-two table on matched test populations."""

import argparse
import gzip
import json
from pathlib import Path

from order_summary import summarize
from verify_snapshot import read, save, sha


def compare(enabled, disabled):
    data = []
    for directory in (enabled, disabled):
        groups = json.loads(gzip.decompress((directory / "pairs.json.gz").read_bytes()))
        summary = read(directory / "summary.json")
        for cohort, runs in groups.items():
            if summarize(runs, cohort == "official") != summary[cohort]:
                raise ValueError(
                    f"Released per-block counts do not reproduce {directory.name} {cohort}"
                )
        data.append((groups, summary))
    (on, a), (off, b) = data
    if (
        a["source_inventory_sha256"] != b["source_inventory_sha256"]
        or a["binary"]["executable"] == b["binary"]["executable"]
    ):
        raise ValueError("Expected identical input data and distinct rule settings")
    checked = 0
    cohorts = {}
    for cohort, runs in on.items():
        if set(runs) != set(off[cohort]):
            raise ValueError("Different language coverage")
        for iso, folds in runs.items():
            for x, y in zip(folds, off[cohort][iso], strict=True):
                if any(
                    x[k] != y[k]
                    for k in ("iso", "order", "train_tokens", "train_correct", "arms")
                ):
                    raise ValueError(f"Fit changed for {iso}")
                for split in ("dev", "test"):
                    for k in ("tokens", "wins", "losses", "changed"):
                        if x[split]["counts"][1][k] != y[split]["counts"][1][k]:
                            raise ValueError(
                                f"Order contrast depends on rules for {cohort} {iso} {k}"
                            )
                    if x[split]["external_hits"][:2] != y[split]["external_hits"][:2]:
                        raise ValueError(f"External predictions differ for {iso}")
                checked += 1
        cells = {}
        for label, base in (("rules_on", a), ("rules_off", b)):
            cells[label] = {}
            for n in ("declared", "learned"):
                arm = base[cohort]["arms"][n]
                cells[label][n] = {
                    k: arm[k]
                    for k in (
                        "correct",
                        "tokens",
                        "macro_accuracy",
                        "micro_accuracy",
                        "wins",
                        "losses",
                    )
                }
        cohorts[cohort] = cells
    return {
        "fits_checked": checked,
        "source_sha256": a["source_inventory_sha256"],
        "binary_sha256": {
            "on": a["binary"]["executable"],
            "off": b["binary"]["executable"],
        },
        "on_pairs_sha256": sha(enabled / "pairs.json.gz"),
        "off_pairs_sha256": sha(disabled / "pairs.json.gz"),
        "cells": cohorts,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("enabled", "disabled", "output"):
        p.add_argument(n, type=Path)
    a = p.parse_args()
    save(a.output, compare(a.enabled, a.disabled))
