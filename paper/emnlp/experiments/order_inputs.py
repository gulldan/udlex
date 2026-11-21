"""Freeze the research source registry and build disjoint, same-treebank train/dev/test inputs."""

import argparse
import json
from collections import Counter
from pathlib import Path

from evaluate import UD218, fingerprint, sentences
from verify_snapshot import EVIDENCE, read, save, sha

from udlex.language_codes import source_language_candidates

MARKERS = {
    "de_hdt": {"unknown", "-", "NULL"},
    "pcm_nsc": {"X"},
    "qaf_arabizi": {"X"},
    "say_autogramm": {"X"},
    "cu_proiel": {"?"},
    "sk_snk": {"?"},
    "myv_jr": {"-"},
    "cop_scriptorium": {"unknown"},
}

RELEASES = {
    "2.18": (UD218, 353, "order-protocol.md"),
    "2.19": (
        "6bbe2f0234dcc2d2367cc912848259c94c278ea69932d228d744f265a19a4a31",
        402,
        "order-219-protocol.md",
    ),
}


def sources(root, iso, specs):
    found, excluded = [], []
    names = source_language_candidates(iso)
    for spec in specs:
        directory = root / "data" / spec["folder"]
        physical = next(
            (
                n
                for n in names
                if (directory / (n + ".tsv")).exists()
                or (directory / (n + ".manifest.json")).exists()
            ),
            iso,
        )
        tab = directory / (physical + ".tsv")
        if not tab.is_file() or not tab.stat().st_size:
            continue
        side = tab.with_suffix(".manifest.json")
        meta = read(side) if side.is_file() else {}
        reason = None
        if not spec.get("enabled", True) or set(names).intersection(
            spec.get("exclude", [])
        ):
            reason = "registry exclusion"
        elif spec.get("manifest_required") and not meta:
            reason = "missing required source manifest"
        elif meta.get("benchmark_eligible") is False or meta.get("transductive_refit"):
            reason = "benchmark quarantine"
        if reason:
            excluded.append([spec["id"], reason])
            continue
        digest = sha(tab)
        if meta.get("artifact", {}).get("sha256", digest) != digest:
            raise ValueError(f"Source hash differs from its manifest: {tab}")
        overrides = spec.get("priority_by_language", {})
        priority = next(
            (overrides[n] for n in names if n in overrides), spec["priority"]
        )
        found.append(
            {
                "id": spec["id"],
                "path": str(tab.relative_to(root)),
                "sha256": digest,
                "bytes": tab.stat().st_size,
                "manifest_sha256": sha(side) if meta else None,
                "priority": priority,
                "unique": spec["folder"].startswith("imported/"),
                "lemma_first": spec.get("lemma_first", False),
                "license": meta.get("license"),
                "version": meta.get("source_version"),
            }
        )
    found.sort(
        key=lambda x: (x["priority"], ("imported/" if x["unique"] else "") + x["id"])
    )
    return found, excluded


def valid(config, form, gold):
    return bool(
        form
        and gold
        and form != "_"
        and gold != "_"
        and (form == gold or gold not in MARKERS.get(config, ()))
    )


def prepare_splits(config, prefix, output):
    past_docs, past_ids, seen = set(), set(), set()
    results = {}
    for split in ("train", "dev", "test"):
        path = prefix.with_name(config + "-ud-" + split + ".conllu")
        out = output / (split + ".jsonl")
        docs, ids, counts = set(), set(), Counter()
        with out.open("w") as fh:
            if path.is_file():
                for doc, sid, forms, gold in sentences(path):
                    fprint = fingerprint(forms)
                    if (
                        (doc and not doc.startswith("@line-") and doc in past_docs)
                        or (sid and sid in past_ids)
                        or fprint in seen
                    ):
                        counts["duplicate_or_shared_sentences"] += 1
                        continue
                    lemmas = [
                        g if valid(config, f, g) else None
                        for f, g in zip(forms, gold, strict=True)
                    ]
                    n = sum(g is not None for g in lemmas)
                    if not n:
                        continue
                    seen.add(fprint)
                    if doc and not doc.startswith("@line-"):
                        docs.add(doc)
                    if sid:
                        ids.add(sid)
                    counts["sentences"] += 1
                    counts["tokens"] += n
                    item = {
                        "tokens": forms,
                        "lemmas": lemmas,
                        "block": "d:" + doc if doc else "s:" + str(counts["sentences"]),
                    }
                    fh.write(
                        json.dumps(item, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )
        past_docs.update(docs)
        past_ids.update(ids)
        results[split] = {
            "file": path.name if path.exists() else None,
            "source_sha256": sha(path) if path.exists() else None,
            "jsonl_sha256": sha(out),
            **counts,
        }
    return results


def main(a):
    a.output.mkdir(parents=True, exist_ok=True)
    archive_hash, treebanks, protocol = RELEASES[a.release]
    if sha(a.udtar) != archive_hash:
        raise ValueError(f"UD {a.release} archive hash mismatch")
    regfile = a.research / "registry/imported_sources.json"
    specs = [
        {
            "id": "dictionary",
            "folder": "dictionaries",
            "priority": 0,
            "manifest_required": True,
            "exclude": ["hun"],
        },
        {"id": "unimorph", "folder": "unimorph", "priority": 200, "lemma_first": True},
        {"id": "wiktionary", "folder": "kaikki", "priority": 210},
    ]
    specs.extend(
        dict(v, folder="imported/" + v["directory"]) for v in read(regfile)["sources"]
    )
    codes = read(a.codes)
    catalog = read(a.research / "data/wide/index.json")["languages"]
    tests = sorted(a.ud.glob("UD_*/*-ud-test.conllu"))
    if len(tests) != treebanks:
        raise ValueError(f"Incomplete UD {a.release} distribution")
    rows = {}
    for iso, code in sorted(codes.items()):
        src, exc = sources(a.research, iso, specs)
        row = rows[iso] = {"sources": src, "excluded_sources": exc}
        if len(src) < a.min_sources:
            continue
        matches = [v for v in tests if v.name.split("_")[0] == code]
        # A language outside the catalog has no preferred treebank.
        preferred = catalog.get(iso, {}).get("main", {}).get("config", "\0")

        def priority(path, preferred=preferred):
            train = path.with_name(path.name.replace("-test.conllu", "-train.conllu"))
            return (
                -int(train.is_file()),
                -int(path.name.startswith(preferred + "-ud-")),
                -train.stat().st_size if train.is_file() else 0,
                str(path),
            )

        dest = a.output / iso
        dest.mkdir(exist_ok=True)
        for test in sorted(matches, key=priority):
            config = test.name.removesuffix("-ud-test.conllu")
            splits = prepare_splits(config, test, dest)
            if splits["train"].get("tokens") and splits["test"].get("tokens"):
                break
        else:
            if not matches:
                row["status"] = "no official test file"
                continue
            test = min(matches, key=priority)
            config = test.name.removesuffix("-ud-test.conllu")
            splits = prepare_splits(config, test, dest)
        row.update(
            config=config,
            treebank=test.parent.name,
            splits=splits,
            status="paired"
            if splits["train"].get("tokens") and splits["test"].get("tokens")
            else "no train or test gold",
        )
        print(iso, len(src), row["status"], flush=True)
    save(
        a.output / "inventory.json",
        {
            "registry_sha256": sha(regfile),
            "codes_sha256": sha(a.codes),
            "ud" + a.release.replace(".", "") + "_sha256": archive_hash,
            "protocol_sha256": sha(Path(__file__).with_name(protocol)),
            "languages": rows,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("research", "ud", "udtar", "output"):
        p.add_argument(key, type=Path)
    p.add_argument("--release", choices=RELEASES, default="2.18")
    p.add_argument(
        "--min-sources", type=int, default=2, help="0 covers every catalog language"
    )
    p.add_argument(
        "--codes",
        type=Path,
        default=EVIDENCE / "language-codes.json",
        help="ISO 639-3 to UD code; another file adds languages outside the catalog",
    )
    main(p.parse_args())
