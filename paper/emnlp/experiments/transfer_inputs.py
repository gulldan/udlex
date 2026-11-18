"""Freeze transfer tests from corpora outside each anchor treebank, screened against the anchor."""

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Run with -I: corpora are untrusted input, so only this directory joins the path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from order_inputs import fingerprint, sentences, valid  # noqa: E402
from verify_snapshot import EVIDENCE, read, save, sha  # noqa: E402

GOLD = {
    "manual native",
    "converted from manual",
    "converted with corrections",
    "automatic with corrections",
}
WINDOW = 32
KEEP = 4
DOCUMENT = 0.10


def squash(forms):
    text = unicodedata.normalize("NFKC", "".join(forms)).casefold()
    return "".join(c for c in text if c.isalnum())


def windows(text):
    for i in range(len(text) - WINDOW + 1):
        h = hashlib.blake2b(text[i : i + WINDOW].encode(), digest_size=8).digest()
        if h[0] % KEEP == 0:
            yield h


def config(bank):
    return min(f.name for f in bank.glob("*.conllu")).split("-ud-")[0]


def contributors(bank):
    text = min(bank.glob("README*")).read_text(errors="replace")
    found = re.search(r"^Contributors:[ \t]*(.*)$", text, re.M)
    return (
        {n.strip().casefold() for n in found[1].split(";") if n.strip()}
        if found
        else set()
    )


def screen(bank, exact, near, seen):
    cfg = config(bank)
    counts = Counter()
    docs = {}
    for path in sorted(bank.glob("*.conllu")):
        for n, (doc, _, forms, gold) in enumerate(sentences(path), 1):
            counts["sentences_read"] += 1
            lemmas = [
                g if valid(cfg, f, g) else None
                for f, g in zip(forms, gold, strict=True)
            ]
            if not any(lemmas):
                counts["no_scorable_word"] += 1
                continue
            mark = fingerprint(forms)
            flag = (
                "exact_overlap"
                if mark in exact
                else "near_overlap"
                if any(w in near for w in windows(squash(forms)))
                else None
            )
            block = (
                f"d:{path.name}:{doc}"
                if doc and not doc.startswith("@line-")
                else f"s:{path.name}:{n}"
            )
            item = {"tokens": forms, "lemmas": lemmas, "block": block}
            docs.setdefault(block, []).append((flag, mark, item))
    kept = []
    for block, rows in docs.items():
        flags = [flag for flag, _, _ in rows if flag]
        counts.update(flags)
        if block.startswith("d:") and flags and len(flags) >= DOCUMENT * len(rows):
            counts["documents_removed"] += 1
            counts["document_sentences_removed"] += len(rows) - len(flags)
            continue
        for flag, mark, item in rows:
            if flag:
                continue
            if mark in seen:
                counts["repeated_sentence"] += 1
                continue
            seen.add(mark)
            kept.append(item)
    counts["sentences"] = len(kept)
    counts["tokens"] = sum(sum(y is not None for y in s["lemmas"]) for s in kept)
    return kept, dict(counts)


def language(job):
    iso, code, row, fit, splits, cohort, roots, out = job
    exact, near = set(), set()
    for path in sorted((roots[0] / row["treebank"]).glob("*.conllu")):
        stream = []
        for _, _, forms, _ in sentences(path):
            exact.add(fingerprint(forms))
            stream.append(squash(forms))
        near.update(windows("".join(stream)))
    team = contributors(roots[0] / row["treebank"])
    banks = []
    for root in roots:
        for bank in root.iterdir():
            if bank.name == row["treebank"] or not bank.is_dir():
                continue
            if any(f.name.startswith(code + "_") for f in bank.glob("*.conllu")):
                text = min(bank.glob("README*")).read_text(errors="replace")
                found = re.search(r"^Lemmas:[ \t]*(.*)$", text, re.M)
                status = found[1].strip() if found else "not available"
                tier = (
                    "gold"
                    if status in GOLD
                    else "silver"
                    if status == "automatic"
                    else None
                )
                banks.append((tier != "gold", config(bank), bank, status, tier))
    ledger = {}
    seen = set()
    for _, cfg, bank, status, tier in sorted(banks):
        entry = ledger[cfg] = {
            "iso": iso,
            "treebank": bank.name,
            "anchor": row["treebank"],
            "fit": cohort,
            "lemmas": status,
            "tier": tier,
            "ud": bank.parent == roots[0],
            "shared_contributors": bool(team & contributors(bank)),
            "files": {f.name: sha(f) for f in sorted(bank.glob("*.conllu"))},
        }
        if not tier:
            continue
        kept, entry["screen"] = screen(bank, exact, near, seen)
        if not entry["screen"]["tokens"]:
            continue
        dest = out / cfg / iso
        dest.mkdir(parents=True)
        for name in ("train", "dev"):
            os.link(fit / (name + ".jsonl"), dest / (name + ".jsonl"))
        with (dest / "test.jsonl").open("w") as fh:
            for s in kept:
                line = json.dumps(s, ensure_ascii=False, separators=(",", ":"))
                fh.write(line + "\n")
        test = {
            "jsonl_sha256": sha(dest / "test.jsonl"),
            "sentences": len(kept),
            "tokens": entry["screen"]["tokens"],
        }
        save(
            out / cfg / "inventory.json",
            {
                "languages": {
                    iso: dict(
                        row,
                        status="paired",
                        config=cfg,
                        treebank=bank.name,
                        splits=dict(splits, test=test),
                    )
                }
            },
        )
    return ledger


def main(a):
    inv = read(a.order / "inventory.json")["languages"]
    folds = read(a.cv / "ledger.json")
    codes = read(EVIDENCE / "language-codes.json")
    roots = [a.ud, *a.external]
    jobs = []
    for iso, row in sorted(inv.items()):
        if len(row["sources"]) < 2:
            continue
        if row.get("status") == "paired":
            fit, cohort = a.order / iso, "official"
            splits = {k: row["splits"][k] for k in ("train", "dev")}
        elif iso in folds and min(folds[iso]["fold_tokens"]):
            # Both folds together are the whole deduplicated anchor.
            fit, cohort = a.output / ".fit" / iso, "pooled"
            fit.mkdir(parents=True)
            parts = [a.cv / "0" / iso / (n + ".jsonl") for n in ("train", "test")]
            (fit / "train.jsonl").write_bytes(b"".join(p.read_bytes() for p in parts))
            (fit / "dev.jsonl").write_bytes(b"")
            splits = {
                "train": {
                    "jsonl_sha256": sha(fit / "train.jsonl"),
                    "tokens": sum(folds[iso]["fold_tokens"]),
                },
                "dev": {"jsonl_sha256": sha(fit / "dev.jsonl")},
            }
        else:
            continue
        jobs.append((iso, codes[iso], row, fit, splits, cohort, roots, a.output))
    ledger = {}
    with ProcessPoolExecutor(a.jobs) as pool:
        for part in pool.map(language, jobs):
            ledger.update(part)
    save(
        a.output / "ledger.json",
        {
            "protocol_sha256": sha(Path(__file__).with_name("transfer-protocol.md")),
            "order_inventory_sha256": sha(a.order / "inventory.json"),
            "candidate_languages": len(jobs),
            "corpora": ledger,
        },
    )
    print(len(ledger), "candidate corpora", Counter(v["tier"] for v in ledger.values()))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("order", "cv", "ud", "output"):
        p.add_argument(key, type=Path)
    p.add_argument("--external", type=Path, action="append", default=[])
    p.add_argument("--jobs", type=int, default=8)
    main(p.parse_args())
