"""Screen lexical sources for derivation from an evaluation corpus."""

import argparse
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_snapshot import read, save

TYPES, SHARE = 200, 0.98
MARKED, MARKED_SHARE = 30, 0.50


def gold(path):
    votes = defaultdict(Counter)
    with path.open() as fh:
        for line in fh:
            s = json.loads(line)
            for form, lemma in zip(s["tokens"], s["lemmas"], strict=True):
                if lemma is not None:
                    votes[unicodedata.normalize("NFC", form)][lemma] += 1
    return votes


def language(job):
    root, iso, sources, corpora = job
    seen = set(gold(corpora[0][1]))  # the fit's training forms
    unseen = {}
    for name, path in corpora[1:]:
        votes = gold(path)
        unseen[name] = {
            f: v.most_common(1)[0][0] for f, v in votes.items() if f not in seen
        }
    wanted = {k for forms in unseen.values() for f in forms for k in (f, f.lower())}
    report = {}
    for src in sources:
        table = defaultdict(set)
        with (root / src["path"]).open(encoding="utf-8") as fh:
            for line in fh:
                cells = line.rstrip("\n").split("\t")
                if len(cells) < 2:
                    continue
                form, lemma = cells[::-1][-2:] if src["lemma_first"] else cells[:2]
                if form in wanted:
                    table[form].add(lemma)
        for name, forms in unseen.items():
            c = Counter(types=len(forms))
            for form, lemma in forms.items():
                marked = len(form) > 3 and any(
                    unicodedata.category(ch)[0] in "NSP" and ch not in "-'’.‐"
                    for ch in form
                )
                found = table.get(form) or table.get(form.lower())
                c["marked"] += marked
                if found:
                    c["covered"] += 1
                    c["agree"] += lemma in found
                    c["marked_covered"] += marked
            flag = (c["types"] >= TYPES and c["covered"] >= SHARE * c["types"]) or (
                c["marked"] >= MARKED
                and c["marked_covered"] >= MARKED_SHARE * c["marked"]
            )
            report[f"{name}|{src['id']}"] = dict(c, iso=iso, flagged=flag)
    return report


def main(a):
    ledger = read(a.transfer / "ledger.json")["corpora"]
    by = defaultdict(list)
    for cfg, row in sorted(ledger.items()):
        if (a.transfer / cfg / "inventory.json").exists():
            by[row["iso"]].append(cfg)
    inv = read(a.order / "inventory.json")["languages"]
    jobs = []
    for iso, row in sorted(inv.items()):
        if len(row["sources"]) < 2:
            continue
        corpora = []
        for k in ("0", "1") if a.cv else ():
            # A two-fold language: each fold is screened against its own training fold.
            fold = a.cv / k / iso
            if (fold / "test.jsonl").exists():
                tests = [
                    ("train", fold / "train.jsonl"),
                    (f"{iso}:fold-{k}", fold / "test.jsonl"),
                ]
                jobs.append((a.research, iso, row["sources"], tests))
        if row.get("status") == "paired":
            fit = a.order / iso
            corpora.append((iso + ":anchor-test", fit / "test.jsonl"))
            if a.cv and row["splits"]["dev"].get("tokens"):
                corpora.append((iso + ":anchor-dev", fit / "dev.jsonl"))
        elif by[iso]:
            fit = a.transfer / ".fit" / iso
        else:
            continue
        corpora = [("train", fit / "train.jsonl"), *corpora]
        corpora += [(cfg, a.transfer / cfg / iso / "test.jsonl") for cfg in by[iso]]
        jobs.append((a.research, iso, row["sources"], corpora))
    report = {}
    with ProcessPoolExecutor(a.jobs) as pool:
        for part in pool.map(language, jobs):
            report.update(part)
    save(
        a.output, {"thresholds": [TYPES, SHARE, MARKED, MARKED_SHARE], "pairs": report}
    )
    for key, v in report.items():
        if v["flagged"]:
            print(
                key,
                {
                    k: v.get(k, 0)
                    for k in ("types", "covered", "agree", "marked", "marked_covered")
                },
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("research", "order", "transfer", "output"):
        p.add_argument(key, type=Path)
    p.add_argument(
        "--cv", type=Path, help="also screen development splits and both folds"
    )
    p.add_argument("--jobs", type=int, default=8)
    main(p.parse_args())
