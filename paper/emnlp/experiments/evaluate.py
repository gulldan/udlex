"""Evaluate verified 1.0.0 dictionaries on UD with a declared-source overlap screen."""

import argparse
import hashlib
import os
import re
import shutil
import tarfile
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from verify_snapshot import CATALOG, EVIDENCE, read, save, sha

import udlex

UD218 = "a93fe8520bc4c5ff34670d9a93a5a7689c018c1e59643fa27e03036717841b8a"
ARMS = (
    "tokens",
    "copy",
    "lookup",
    "full",
    "copy_wins",
    "copy_losses",
    "rules_wins",
    "rules_losses",
)


def sentences(path):
    doc = None
    sid = None
    forms, lemmas = [], []
    with path.open(encoding="utf-8") as stream:
        for n, line in enumerate(stream, 1):
            if not line.strip():
                if forms:
                    yield doc, sid, forms, lemmas
                sid, forms, lemmas = None, [], []
            elif line.startswith("#"):
                if line.startswith("# newdoc id = "):
                    doc = line.split(" = ", 1)[1].strip()
                elif line.rstrip() == "# newdoc":
                    doc = f"@line-{n}"
                elif line.startswith("# sent_id = "):
                    sid = line.split(" = ", 1)[1].strip()
            else:
                row = line.rstrip("\n\r").split("\t")
                if len(row) != 10:
                    raise ValueError(f"Malformed CoNLL-U row in {path} at line {n}")
                if row[0].isdecimal():
                    forms.append(row[1])
                    lemmas.append(row[2])
        if forms:
            yield doc, sid, forms, lemmas


def fingerprint(forms):
    text = "".join("".join(t.split()) for t in forms).casefold()
    return hashlib.sha256(unicodedata.normalize("NFC", text).encode()).digest()


def dictionary_sources(row, release, cache):
    inputs = [
        f
        for f in (row.get("source_files") or row.get("inputs") or [])
        if f.get("url", "").endswith(".conllu")
    ]
    if not inputs:
        inputs = [{"url": row["download"], "sha256": row["source_sha256"]}]
    pinned, docs, ids, texts = {}, set(), set(), set()
    for source in inputs:
        url = source["url"]
        repo, name = url.split("/")[4], url.rsplit("/", 1)[-1]
        archive_file = release / repo / name
        path = archive_file
        if not path.is_file() or sha(path) != source["sha256"]:
            path = cache / f"{source['sha256']}.conllu"
            if not path.exists():
                tmp = path.with_suffix(f".{os.getpid()}.tmp")
                with (
                    urllib.request.urlopen(url, timeout=60) as remote,  # noqa: S310 - pinned https source
                    tmp.open("wb") as out,
                ):
                    shutil.copyfileobj(remote, out)
                if sha(tmp) != source["sha256"]:
                    raise ValueError(f"Declared dictionary input has changed at {url}")
                tmp.replace(path)
        if sha(path) != source["sha256"]:
            raise ValueError(f"Invalid cached source for {url}")
        pinned[(repo, name)] = source["sha256"]
        for doc, sid, forms, _ in sentences(path):
            if doc and not doc.startswith("@line-"):
                docs.add((repo, doc))
            if sid:
                ids.add((repo, sid))
            texts.add(fingerprint(forms))
    return pinned, docs, ids, texts


def score_language(job):
    iso, row, snap, paths, release, bundle, output = job
    cache = output / "sources"
    cache.mkdir(exist_ok=True)
    pinned, docs, ids, texts = ({}, set(), set(), set())
    if snap["ud_repository"]:
        pinned, docs, ids, texts = dictionary_sources(row, release, cache)
    lex = udlex.Lexicon(str(bundle / iso), use_user=False)
    fixed = udlex.Lexicon(str(bundle / iso), use_rules=False, use_user=False)
    result, seen = {}, set()
    for path in paths:
        repo = path.parent.name
        reason = Counter()
        scores = Counter()
        clusters = defaultdict(Counter)
        kinds = Counter()
        if (repo, path.name) in pinned:
            reason["declared_dictionary_input_file"] = 1
        else:
            for n, (doc, sid, forms, lemmas) in enumerate(sentences(path)):
                if (repo, doc) in docs or (repo, sid) in ids:
                    reason["source_identifier_sentences"] += 1
                    continue
                key = fingerprint(forms)
                if key in texts or key in seen:
                    reason[
                        "source_text_sentences"
                        if key in texts
                        else "repeated_evaluation_sentences"
                    ] += 1
                    continue
                valid = [
                    i
                    for i, (f, g) in enumerate(zip(forms, lemmas, strict=True))
                    if f not in ("", "_") and g not in ("", "_")
                ]
                reason["masked_forms"] += sum(f in ("", "_") for f in forms)
                reason["unknown_lemmas_with_visible_form"] += sum(
                    f not in ("", "_") and g in ("", "_")
                    for f, g in zip(forms, lemmas, strict=True)
                )
                if not valid:
                    reason["no_scorable_words_sentences"] += 1
                    continue
                seen.add(key)
                full = lex.lemmatize(forms)
                lookup = fixed.lemmatize(forms)
                block = ("doc", doc) if doc is not None else ("sent", n)
                counts = clusters[block]
                kinds[block[0]] += 1
                for i in valid:
                    a, b, c = (
                        forms[i] == lemmas[i],
                        lookup[i] == lemmas[i],
                        full[i] == lemmas[i],
                    )
                    counts["tokens"] += 1
                    counts["copy"] += a
                    counts["lookup"] += b
                    counts["full"] += c
                    counts["copy_wins"] += c and not a
                    counts["copy_losses"] += a and not c
                    counts["rules_wins"] += c and not b
                    counts["rules_losses"] += b and not c
        histogram = Counter()
        for counts in clusters.values():
            scores.update(counts)
            histogram[
                (
                    counts["tokens"],
                    counts["full"] - counts["copy"],
                    counts["full"] - counts["lookup"],
                )
            ] += 1
        result[f"{repo}/{path.name}"] = {
            "sha256": sha(path),
            "excluded": dict(reason),
            "counts": {a: scores[a] for a in ARMS},
            "block_types": dict(Counter(b[0] for b in clusters)),
            "sentences_by_block_type": dict(kinds),
            "block_histogram": [[*list(k), m] for k, m in sorted(histogram.items())],
        }
    value = {
        "iso": iso,
        "resource_type": "UD" if snap["ud_repository"] else "external",
        "source_conllu": {f"{r}/{n}": v for (r, n), v in pinned.items()},
        "test_files": result,
    }
    save(output / f"{iso}.json", value)
    return iso


def main(args):
    snapshot = read(args.snapshot)
    if (
        snapshot["catalog_sha256"] != CATALOG
        or udlex.__version__ != snapshot["version"]
    ):
        raise ValueError("Runtime or catalog differs from the verified snapshot")
    if sha(args.ud_archive) != UD218:
        raise ValueError("Unexpected UD 2.18 distribution")
    if sha(args.assets / "DICTIONARIES.json") != CATALOG:
        raise ValueError("Dictionary catalog changed")
    args.output.mkdir(parents=True, exist_ok=True)
    mapping = read(EVIDENCE / "language-codes.json")
    registry = {
        r["iso639_3"]: r
        for r in read(args.assets / "DICTIONARIES.json")["dictionaries"]
    }
    if set(registry) != set(mapping) or len(set(mapping.values())) != len(mapping):
        raise ValueError("Language correspondence is incomplete or ambiguous")
    by_code = defaultdict(list)
    files = sorted(args.ud_directory.glob("UD_*/*-ud-test.conllu"))
    if len(files) != 353:
        raise ValueError("Unexpected UD 2.18 test-file inventory")
    expected = {}
    with tarfile.open(args.ud_archive, "r|gz") as packed:
        for entry in packed:
            if entry.isfile() and entry.name.endswith("-ud-test.conllu"):
                with packed.extractfile(entry) as stream:
                    expected[
                        str(Path(entry.name).relative_to("ud-treebanks-v2.18"))
                    ] = hashlib.file_digest(stream, "sha256").hexdigest()
    if {str(p.relative_to(args.ud_directory)): sha(p) for p in files} != expected:
        raise ValueError("Extracted test files differ from the pinned archive")
    for path in files:
        by_code[re.split(r"[_-]", path.name, maxsplit=1)[0]].append(path)
    jobs = [
        (
            iso,
            registry[iso],
            snapshot["languages"][iso],
            by_code[code],
            args.ud_directory,
            args.bundle,
            args.output,
        )
        for iso, code in sorted(mapping.items())
    ]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for n, iso in enumerate(pool.map(score_language, jobs), 1):
            if n % 20 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} languages ({iso})", flush=True)
    save(
        args.output / "run.json",
        {
            "version": snapshot["version"],
            "catalog_sha256": CATALOG,
            "ud218_sha256": UD218,
            "snapshot_sha256": sha(args.snapshot),
            "language_codes_sha256": sha(EVIDENCE / "language-codes.json"),
            "protocol_sha256": sha(Path(__file__).with_name("protocol.md")),
            "program_sha256": sha(Path(__file__)),
            "release_test_files": len(files),
            "languages": sorted(registry),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "assets",
        "bundle",
        "snapshot",
        "ud_archive",
        "ud_directory",
        "output",
    ):
        parser.add_argument(name, type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    main(parser.parse_args())
