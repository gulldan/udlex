"""Acquire pinned upstream input files for a clean catalog build."""

import argparse
import hashlib
import json
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(item):
    url, digest, path = item
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and sha(path) == digest:
        return
    for attempt in range(4):
        try:
            with (
                urllib.request.urlopen(
                    urllib.request.Request(
                        url, headers={"User-Agent": "udlex-research/1.0"}
                    ),
                    timeout=120,
                ) as response,
                path.with_suffix(".partial").open("wb") as dest,
            ):
                shutil.copyfileobj(response, dest)
            if sha(path.with_suffix(".partial")) != digest:
                raise ValueError(f"Source checksum mismatch: {url}")
            path.with_suffix(".partial").replace(path)
            return
        except (OSError, ValueError) as error:
            if attempt == 3:
                raise
            time.sleep(1 + attempt)


def inputs(row):
    return [v for v in row.get("inputs", []) if "url" in v] or [
        {"url": row["download"], "sha256": row["source_sha256"]}
    ]


def main(catalog, root):
    rows = json.loads(catalog.read_bytes())["dictionaries"]
    jobs = {}
    for row in rows:
        iso = row["iso639_3"]
        files = []
        if row["source"].startswith("UniMorph") or iso == "cmn":
            files = inputs(row)
        elif iso == "tuk":
            files = [v for v in row["source_files"] if v["url"].endswith(".conllu")]
        elif iso == "rus":
            with urllib.request.urlopen(
                "https://pypi.org/pypi/mawo-pymorphy3/1.0.4/json"
            ) as doc:
                asset = next(
                    a
                    for a in json.load(doc)["urls"]
                    if a["filename"] == row["source_artifact"]
                )
            files = [{"url": asset["url"], "sha256": row["source_sha256"]}]
        for entry in files:
            digest = entry["sha256"]
            jobs[digest] = (entry["url"], digest, root / digest)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for job, _ in zip(jobs.values(), pool.map(fetch, jobs.values()), strict=True):
            print(job[1], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("cache", type=Path)
    args = parser.parse_args()
    main(args.catalog, args.cache)
