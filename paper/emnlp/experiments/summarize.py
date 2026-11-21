"""Summarize all paired evaluation rows and bootstrap within fixed treebanks."""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
from verify_snapshot import read, save, sha

ARMS = ("copy", "lookup", "full")
DRAWS = 2000
SEED = 20261118


def aggregate(languages, replicates):
    data = [v["counts"] for v in languages.values() if v["counts"]["tokens"]]
    total = sum((Counter(c) for c in data), Counter())
    macro = {a: 100 * np.mean([c[a] / c["tokens"] for c in data]) for a in ARMS}
    micro = {a: 100 * total[a] / total["tokens"] for a in ARMS}
    result = {
        "scored_languages": len(data),
        "counts": dict(total),
        "macro_accuracy": macro,
        "micro_accuracy": micro,
    }
    for pos, arm in enumerate(("copy", "lookup")):
        diffs = [c["full"] - c[arm] for c in data]
        draws = np.mean(
            [replicates[iso][:, pos] for iso in sorted(languages) if iso in replicates],
            axis=0,
        )
        result[f"full_minus_{arm}"] = {
            "macro_pp": macro["full"] - macro[arm],
            "micro_pp": micro["full"] - micro[arm],
            "macro_ci95_pp": np.percentile(draws, [2.5, 97.5]).tolist(),
            "language_wins_ties_losses": [
                sum(x > 0 for x in diffs),
                sum(x == 0 for x in diffs),
                sum(x < 0 for x in diffs),
            ],
        }
    return result


def report(paired, snapshot, dest):
    compact = (
        json.loads(gzip.decompress(paired.read_bytes())) if paired.is_file() else None
    )
    run = compact["run"] if compact else read(paired / "run.json")
    if run["snapshot_sha256"] != sha(snapshot):
        raise ValueError("Run and source snapshot differ")
    rng = np.random.default_rng(SEED)
    languages, replicates, treebanks = {}, {}, {}
    for iso in run["languages"]:
        row = compact["rows"][iso] if compact else read(paired / f"{iso}.json")
        counts, omissions, block_types = Counter(), Counter(), Counter()
        boot = np.zeros((DRAWS, 3), dtype=np.int64)
        for file, test in sorted(row["test_files"].items()):
            c = test["counts"]
            if (
                c["full"] - c["copy"] != c["copy_wins"] - c["copy_losses"]
                or c["full"] - c["lookup"] != c["rules_wins"] - c["rules_losses"]
            ):
                raise ValueError(f"Inconsistent paired counts for {file}")
            counts.update(c)
            omissions.update(test["excluded"])
            block_types.update(test["block_types"])
            treebanks[file] = dict(
                iso=iso, **{k: v for k, v in test.items() if k != "block_histogram"}
            )
            if c["tokens"]:
                hist = np.asarray(test["block_histogram"], dtype=np.int64)
                values, freq = hist[:, :3], hist[:, 3]
                if list(freq @ values) != [
                    c["tokens"],
                    c["full"] - c["copy"],
                    c["full"] - c["lookup"],
                ]:
                    raise ValueError(f"Block and token totals differ for {file}")
                for offset in range(0, DRAWS, 50):
                    weights = rng.multinomial(
                        int(freq.sum()), freq / freq.sum(), size=min(50, DRAWS - offset)
                    )
                    boot[offset : offset + len(weights)] += weights @ values
        if counts["tokens"]:
            status = "scored"
            replicates[iso] = 100 * boot[:, 1:] / boot[:, :1]
        elif not row["test_files"]:
            status = "no official test file for code"
        elif len(row["test_files"]) == omissions["declared_dictionary_input_file"]:
            status = "all available tests used in dictionary"
        else:
            status = "no scorable test tokens after exclusions"
        languages[iso] = {
            "counts": {
                k: counts[k]
                for k in (
                    "tokens",
                    *ARMS,
                    "copy_wins",
                    "copy_losses",
                    "rules_wins",
                    "rules_losses",
                )
            },
            "status": status,
            "test_files": len(row["test_files"]),
            "source_conllu": row["source_conllu"],
            "scored_files": sum(
                t["counts"]["tokens"] > 0 for t in row["test_files"].values()
            ),
            "resource_type": row["resource_type"],
            "excluded": dict(omissions),
            "block_types": dict(block_types),
        }
    excluded = sum((Counter(r["excluded"]) for r in languages.values()), Counter())
    block_types = sum(
        (Counter(r["block_types"]) for r in languages.values()), Counter()
    )
    result = {
        "run": run,
        "bootstrap": {
            "replicates": DRAWS,
            "seed": SEED,
            "numpy": np.__version__,
            "unit": "document if available, otherwise sentence; fixed treebanks and languages",
        },
        "all": aggregate(languages, replicates),
        "language_status": dict(Counter(r["status"] for r in languages.values())),
        "excluded": dict(excluded),
        "block_types": dict(block_types),
        "test_files_in_scope": len(treebanks),
        "test_files_scored": sum(t["counts"]["tokens"] > 0 for t in treebanks.values()),
    }
    for kind in ("UD", "external"):
        result[kind] = aggregate(
            {k: v for k, v in languages.items() if v["resource_type"] == kind},
            replicates,
        )
    save(dest / "summary.json", result)
    save(dest / "languages.json", languages)
    save(dest / "treebanks.json", treebanks)
    if compact is None:
        archive = {
            "run": run,
            "rows": {iso: read(paired / f"{iso}.json") for iso in run["languages"]},
        }
        data = json.dumps(archive, sort_keys=True, separators=(",", ":")).encode()
        (dest / "paired-counts.json.gz").write_bytes(gzip.compress(data, mtime=0))
    print(result["all"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("paired", "snapshot", "dest"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    report(a.paired, a.snapshot, a.dest)
