"""Form disjoint two-fold diagnostics only for entries without an official pair."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from order_inputs import fingerprint, sentences, valid
from verify_snapshot import read, save, sha


def split_language(row, root):
    cfg = row["config"]
    folder = root / row["treebank"]
    unique, groups = set(), defaultdict(list)
    for split in ("train", "dev", "test"):
        path = folder / f"{cfg}-ud-{split}.conllu"
        if not path.exists():
            continue
        for n, (doc, _, forms, gold) in enumerate(sentences(path), 1):
            key = fingerprint(forms)
            lemmas = [
                y if valid(cfg, f, y) else None
                for f, y in zip(forms, gold, strict=True)
            ]
            if key in unique or not any(lemmas):
                continue
            unique.add(key)
            ident = (
                "d:" + doc if doc and not doc.startswith("@line-") else f"s:{split}:{n}"
            )
            groups[ident].append({"tokens": forms, "lemmas": lemmas, "block": ident})
    if len(groups) == 1:
        group = next(iter(groups.values()))
        groups = {f"s:{n}": [dict(s, block=f"s:{n}")] for n, s in enumerate(group, 1)}
        mode = "sentence within one original group"
    else:
        mode = (
            "documents and ungrouped sentences"
            if any(k.startswith("d:") for k in groups)
            else "sentences"
        )
    weights = {
        k: sum(sum(y is not None for y in s["lemmas"]) for s in vs)
        for k, vs in groups.items()
    }
    folds = [[], []]
    loads = [0, 0]
    for k in sorted(
        groups, key=lambda x: (-weights[x], hashlib.sha256(x.encode()).digest())
    ):
        i = int(loads[1] < loads[0])
        loads[i] += weights[k]
        folds[i].extend(groups[k])
    return folds, loads, mode


def main(a):
    inv = read(a.inputs / "inventory.json")
    manifests = [{}, {}]
    ledger = {}
    for iso, row in inv["languages"].items():
        if row.get("status") != "no train or test gold":
            continue
        folds, loads, mode = split_language(row, a.ud)
        ledger[iso] = {"fold_tokens": loads, "unit": mode}
        if not min(loads):
            continue
        for k in range(2):
            folder = a.output / str(k) / iso
            folder.mkdir(parents=True, exist_ok=True)
            splits = {}
            for name, sents in (
                ("train", folds[1 - k]),
                ("dev", []),
                ("test", folds[k]),
            ):
                p = folder / (name + ".jsonl")
                with p.open("w") as f:
                    for s in sents:
                        f.write(
                            json.dumps(s, ensure_ascii=False, separators=(",", ":"))
                            + "\n"
                        )
                splits[name] = {
                    "jsonl_sha256": sha(p),
                    "tokens": sum(
                        sum(y is not None for y in s["lemmas"]) for s in sents
                    ),
                }
            manifests[k][iso] = dict(row, status="paired", splits=splits)
    for k, rows in enumerate(manifests):
        save(
            a.output / str(k) / "inventory.json",
            {"languages": rows, "parent_sha256": sha(a.inputs / "inventory.json")},
        )
    save(a.output / "ledger.json", ledger)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ("inputs", "ud", "output"):
        p.add_argument(n, type=Path)
    main(p.parse_args())
