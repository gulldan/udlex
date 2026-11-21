"""Score the new packages, the 1.0.0 dictionaries and copying on the same tokens.

Gold tokenization, exact lemma match after NFC. Official-split languages are
scored on test with the shipped archive; two-fold languages with the fold
builds, each on the fold it did not learn from. The 1.0.0 dictionaries are
flagged where their catalog says they learned from the scored treebank's dev or
test. Dev is reported too, against the train-only build the gate measured.
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path
from statistics import mean

from common import jsonl, nfc, read_json, write_json

import udlex


def correct(lexicon: udlex.Lexicon | None, sentences: list[dict]) -> tuple[int, int]:
    """Correct and scored tokens; ``None`` scores copying the form."""
    tokens = [s["tokens"] for s in sentences]
    answers = lexicon.lemmatize_many(tokens) if lexicon else tokens
    hits = total = 0
    for sentence, row in zip(sentences, answers, strict=True):
        for answer, gold in zip(row, sentence["lemmas"], strict=True):
            if gold is not None:
                total += 1
                hits += nfc(answer) == nfc(gold)
    return hits, total


def unpack(archive: Path, dest: Path) -> Path:
    """Extract a portable archive once; return its bundle directory."""
    if not (dest / "bundle").is_dir():
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as packed:
            packed.extractall(dest, filter="data")
    return dest / "bundle"


def old_lexicon(
    iso: str, catalog: dict, packages: Path, scratch: Path
) -> udlex.Lexicon | None:
    """The 1.0.0 dictionary of a language, when it has one."""
    row = catalog.get(iso)
    if not row:
        return None
    bundle = unpack(packages / row["archive"]["name"], scratch / "old" / iso)
    return udlex.load(str(bundle), iso, use_user=False)


def saw(row: dict | None, treebank: str | None, mode: str) -> bool:
    """Whether the 1.0.0 dictionary learned from the scored sentences."""
    if not row or row.get("ud_repository") != treebank:
        return False
    if mode == "official":
        return row["source_split"] in {"dev", "test"}
    return row["source_split"] in {"train", "dev", "test"}


def score_language(
    iso: str, row: dict, args: argparse.Namespace, catalog: dict
) -> dict | None:
    """All three systems on one language's evaluation tokens."""
    work, release, packages = args.work, args.release, args.old / "packages"
    decision_path = work / "build" / iso / "decision.json"
    if row["mode"] not in {"official", "twofold"} or not decision_path.is_file():
        return None
    decision = read_json(decision_path)
    entry = next(
        e
        for e in read_json(release / "DICTIONARIES.json")["dictionaries"]
        if e["iso639_3"] == iso
    )
    old = old_lexicon(iso, catalog, packages, work / "eval")
    splits = work / "splits" / iso
    result: dict = {
        "iso639_3": iso,
        "language": row["language"],
        "mode": row["mode"],
        "treebank": row["treebank"],
        "distribution": entry["distribution"],
        "dictionaries": decision["order"],
        "old_available": old is not None,
        "old_saw_evaluation": saw(catalog.get(iso), row["treebank"], row["mode"]),
        "old_source": catalog[iso]["source"] if iso in catalog else None,
    }
    if row["mode"] == "official":
        archive = entry["archive"]
        # Keyed by digest: a rebuilt package must never reuse a stale extraction.
        scratch = work / "eval" / "new" / iso / archive["sha256"][:16]
        bundle = unpack(release / "packages" / archive["name"], scratch)
        new = udlex.load(str(bundle), iso, use_user=False)
        test = jsonl(splits / "test.jsonl")
        parts = [(new, test)]
        dev = jsonl(splits / "dev.jsonl")
        if dev and "dev_chosen" in decision:
            result["dev"] = {
                "tokens": decision["dev_chosen"]["tokens"],
                "new_train_only": decision["dev_chosen"]["correct"],
                "declared_train_only": decision["dev_declared"]["correct"],
                "old": correct(old, dev)[0] if old else None,
                "copy": correct(None, dev)[0],
            }
    else:
        parts = [
            (
                udlex.Lexicon(
                    str(work / "build" / iso / f"fold{k}" / iso), use_user=False
                ),
                jsonl(splits / f"fold{k}.jsonl"),
            )
            for k in (0, 1)
        ]
    totals = {"new": 0, "old": 0, "copy": 0, "tokens": 0}
    for lexicon, sentences in parts:
        hits, total = correct(lexicon, sentences)
        totals["new"] += hits
        totals["tokens"] += total
        totals["copy"] += correct(None, sentences)[0]
        if old:
            totals["old"] += correct(old, sentences)[0]
    result |= totals
    if not old:
        result["old"] = None
    return result


def summarize(rows: list[dict]) -> dict:
    """Macro and pooled accuracy per group, each on one common language set."""
    groups = {
        "official": lambda r: r["mode"] == "official",
        "official, with dictionaries": lambda r: (
            r["mode"] == "official" and r["dictionaries"]
        ),
        "official, without dictionaries": lambda r: (
            r["mode"] == "official" and not r["dictionaries"]
        ),
        "two-fold": lambda r: r["mode"] == "twofold",
        "two-fold, with dictionaries": lambda r: (
            r["mode"] == "twofold" and r["dictionaries"]
        ),
        "two-fold, without dictionaries": lambda r: (
            r["mode"] == "twofold" and not r["dictionaries"]
        ),
        "official, 1.0.0 did not see test": lambda r: (
            r["mode"] == "official"
            and r["old"] is not None
            and not r["old_saw_evaluation"]
        ),
        "two-fold, 1.0.0 did not see the folds": lambda r: (
            r["mode"] == "twofold"
            and r["old"] is not None
            and not r["old_saw_evaluation"]
        ),
    }
    summary = {}
    for name, keep in groups.items():
        members = [r for r in rows if keep(r) and r["tokens"]]
        both = [r for r in members if r["old"] is not None]
        if not members:
            continue
        summary[name] = {
            "languages": len(members),
            "tokens": sum(r["tokens"] for r in members),
            "macro": {
                s: 100 * mean(r[s] / r["tokens"] for r in members)
                for s in ("new", "copy")
            },
            "pooled": {
                s: 100 * sum(r[s] for r in members) / sum(r["tokens"] for r in members)
                for s in ("new", "copy")
            },
            "with_1.0.0": {
                "languages": len(both),
                "macro": {
                    s: 100 * mean(r[s] / r["tokens"] for r in both)
                    for s in ("new", "old", "copy")
                }
                if both
                else None,
                "new_better": sum(r["new"] > r["old"] for r in both),
                "new_worse": sum(r["new"] < r["old"] for r in both),
            },
        }
    return summary


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("release", type=Path, help="output of package.py")
    parser.add_argument(
        "old",
        type=Path,
        help="directory with the 1.0.0 DICTIONARIES.json and packages/",
    )
    args = parser.parse_args()
    roster = read_json(args.work / "roster.json")
    catalog = {
        r["iso639_3"]: r
        for r in read_json(args.old / "DICTIONARIES.json")["dictionaries"]
    }
    rows = []
    for iso, row in sorted(roster.items()):
        result = score_language(iso, row, args, catalog)
        if result:
            rows.append(result)
            print(
                iso,
                result["mode"],
                result["new"],
                result["old"],
                result["copy"],
                result["tokens"],
                flush=True,
            )
    write_json(
        args.release / "evaluation.json",
        {"languages": rows, "summary": summarize(rows)},
    )


if __name__ == "__main__":
    main()
