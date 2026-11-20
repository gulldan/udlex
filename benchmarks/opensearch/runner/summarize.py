"""Summarize verified per-profile measurements from the OpenSearch 3.10 stand."""

# ruff: noqa: INP001
import argparse
import csv
import hashlib
import json
import statistics
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any

MAIN = ("OpenSearch", "Native", "Udlex")
EXTRA = ("Native-kuromoji-smartcn", "Udlex-kuromoji-icu")


def read(path: Path) -> dict[str, Any]:
    """Read a JSON measurement or manifest."""
    return json.loads(path.read_bytes())


def combine(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate lexical counts across test folds before calculating percentages."""
    pairs = [s["metrics"]["pairs"] for s in cells]
    lexical = [s["metrics"]["lexical_tokens"] for s in cells]
    tp = sum(x["true_positive_pairs"] for x in pairs)
    gold = sum(x["gold_pairs"] for x in pairs)
    pred = sum(x["predicted_pairs"] for x in pairs)
    total = sum(x["tokens"] for x in lexical)
    return {
        "gold_pairs": gold,
        "predicted_pairs": pred,
        "true_positive_pairs": tp,
        "pair_f1_pct": 200 * tp / (gold + pred) if gold + pred else None,
        "lexical_tokens": total,
        "lexical_casefold_pct": 100
        * sum(x["casefold_matches"] for x in lexical)
        / total
        if total
        else None,
        "test_tokens": sum(s["metrics"]["all_tokens"]["tokens"] for s in cells),
    }


def extract(
    folder: Path, roster: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Read a profile and verify that timing and quality inputs belong to the run."""
    env = read(folder / "environment.json")
    complete = read(folder / "complete.json")
    assert complete["fingerprint"] == env["fingerprint"]
    results = {}
    for iso in complete["languages"]:
        rec = read(folder / "shipped" / (iso + ".json"))
        assert rec["fingerprint"] == env["fingerprint"]
        assert rec["iso"] == iso
        r = rec["cells"][0]
        speed = r["speed"]
        assert speed["requests"]
        assert speed["input_forms"]
        assert len(speed["sample_sha256"]) == 64
        timing = folder / "timings" / (iso + ".jsonl")
        with timing.open("rb") as stream:
            assert (
                hashlib.file_digest(stream, "sha256").hexdigest()
                == speed["timing_sha256"]
            ), (folder, iso)
        mode = roster[iso]["evaluation"]
        assert rec["mode"] == mode
        cells = []
        if mode == "test":
            cells = [r]
        elif mode == "two-fold":
            held = read(folder / "heldout" / (iso + ".json"))
            assert held["fingerprint"] == env["fingerprint"]
            assert len(held["cells"]) == 2
            cells = held["cells"]
        total = combine(cells) if cells else {}
        parity = sum(
            c.get("parity_forms", 0)
            for c in ([r, *cells] if mode == "two-fold" else [r])
        )
        if folder.name.startswith("Udlex"):
            assert parity >= speed["sample_forms"] > 0, (folder, iso)
        results[iso] = {
            **total,
            "forms_per_sec": speed["forms_per_second"],
            "cpu_s_per_million": speed["cpu_seconds_per_million_forms"],
            "p95_ms": speed["request_ms_p95"],
            "sample_sha256": speed["sample_sha256"],
            "gold_sha256": [c["gold_sha256"] for c in cells],
            "mode": mode,
            "parity_forms": parity,
        }
    return results


def summarize(
    systems: dict[str, dict[str, dict[str, Any]]], labels: Sequence[str]
) -> dict[str, Any]:
    """Compare speed and quality for the languages shared by the given profiles."""
    shared = sorted(set.intersection(*(set(systems[x]) for x in labels)))
    for iso in shared:
        assert len({systems[x][iso]["sample_sha256"] for x in labels}) == 1, iso
        assert len({tuple(systems[x][iso]["gold_sha256"]) for x in labels}) == 1, iso
        assert len({systems[x][iso].get("gold_pairs") for x in labels}) == 1, iso
    groups = {}
    for mode in ("all", "test", "two-fold"):
        cohort = [
            iso
            for iso in shared
            if systems[labels[0]][iso].get("gold_pairs", 0) > 0
            and (mode == "all" or systems[labels[0]][iso]["mode"] == mode)
        ]
        groups[mode] = {
            "n": len(cohort),
            "languages": cohort,
            "macro_pair_f1_pct": {
                x: statistics.mean(systems[x][i]["pair_f1_pct"] for i in cohort)
                if cohort
                else None
                for x in labels
            },
        }
    return {
        "languages": shared,
        "speed_n": len(shared),
        "quality": groups,
        "speed": {
            x: {
                k: statistics.median(systems[x][iso][k] for iso in shared)
                for k in ("forms_per_sec", "cpu_s_per_million", "p95_ms")
            }
            for x in labels
        },
    }


def main() -> None:
    """Write the per-language tables and the cross-profile summary."""
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    p.add_argument("catalog", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--languages", nargs="+")
    args = p.parse_args()
    roster = {r["iso639_3"]: r for r in read(args.catalog)["dictionaries"]}
    if args.languages:
        roster = {iso: roster[iso] for iso in args.languages}
    systems = {name: extract(args.results / name, roster) for name in (*MAIN, *EXTRA)}
    assert set(systems["Udlex"]) == set(systems["OpenSearch"]) == set(roster)
    assert (
        set(systems[EXTRA[0]])
        == set(systems[EXTRA[1]])
        == (set(roster) & {"jpn", "cmn", "yue", "kor", "tha"})
    )

    def table(name: str, labels: Sequence[str], languages: Collection[str]) -> None:
        fields = ["iso", "language", "evaluation", "dictionary_package", "gold_pairs"]
        metrics = [
            "pair_f1_pct",
            "forms_per_sec",
            "cpu_s_per_million",
            "p95_ms",
            "lexical_casefold_pct",
        ]
        fields += [f"{label}_{key}" for label in labels for key in metrics]
        with (args.out / name).open("w", newline="", encoding="utf-8") as stream:
            wr = csv.DictWriter(stream, fieldnames=fields)
            wr.writeheader()
            for iso in sorted(languages):
                row = {
                    "iso": iso,
                    "language": roster[iso]["language"],
                    "evaluation": roster[iso]["evaluation"] or "no gold",
                    "dictionary_package": roster[iso]["distribution"],
                    "gold_pairs": systems["Udlex"][iso].get("gold_pairs"),
                }
                for label in labels:
                    for key in metrics:
                        v = systems[label].get(iso, {}).get(key)
                        row[f"{label}_{key}"] = None if v is None else round(v, 5)
                wr.writerow(row)

    table("by-language.csv", MAIN, roster)
    table("cjk.csv", (*MAIN, *EXTRA), systems[EXTRA[0]])
    scopes = {
        "three_way": summarize(systems, MAIN),
        "standard_and_udlex": summarize(systems, ["OpenSearch", "Udlex"]),
        "official_tokenizers": summarize(systems, EXTRA),
    }
    resources = {}
    for label in MAIN:
        mem = read(args.results / label / "memory-common.json")
        assert sorted(mem["completed_languages"]) == scopes["three_way"]["languages"], (
            label
        )
        resources[label] = {
            "pss_mib_after_common": mem["process"]["pss_bytes"] / 2**20,
            "rss_mib_after_common": mem["process"]["rss_bytes"] / 2**20,
        }
    russian = {s: systems[s]["rus"] for s in MAIN}
    recap = {
        "opensearch": "3.10.0",
        "scope": "smoke" if args.languages else "full",
        "catalog_sha256": hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
        "coverage": {s: len(systems[s]) for s in systems},
        "scored_by_mode": {
            str(k): sum(r["evaluation"] == k for r in roster.values())
            for k in ("test", "two-fold", None)
        },
        "comparisons": scopes,
        "russian": russian,
        "common_memory": resources,
        "parity_checks": {
            s: sum(row["parity_forms"] for row in systems[s].values())
            for s in systems
            if s.startswith("Udlex")
        },
        "russian_search_smoke": read(args.results / "Udlex/search-smoke.json"),
    }
    (args.out / "summary.json").write_text(
        json.dumps(recap, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: recap[k]
                for k in [
                    "coverage",
                    "scored_by_mode",
                    "parity_checks",
                    "common_memory",
                ]
            },
            ensure_ascii=False,
        )
    )
    print("Common speed", scopes["three_way"]["speed"])
    print("Common F1", scopes["three_way"]["quality"]["all"])
    print("Russian F1", {k: v["pair_f1_pct"] for k, v in russian.items()})


if __name__ == "__main__":
    main()
