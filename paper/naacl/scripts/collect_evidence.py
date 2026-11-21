"""Copy the C9 speed receipts and the dictionary release manifest into the repository.

usage: python -I collect_evidence.py SPEED_RECEIPTS REUSED_RECEIPTS RELEASE OUT

SPEED_RECEIPTS is a `bench.py run` output directory, REUSED_RECEIPTS the earlier run
whose competitor cells it reused (only its environment snapshots are copied), and
RELEASE the dictionary release directory (with evaluation.json and DICTIONARIES.json),
all as fetched from the measuring machine. The copies keep every measured value. They drop what describes the
machine's other users rather than the measurement: the names of foreign processes
(their CPU share stays), free-text operator notes, and absolute paths, which become
relative to a placeholder.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

PATHS = [
    (re.compile(r"/mnt/[\w.-]+/codex-runs/"), "<runs>/"),
    (re.compile(r"/mnt/[\w.-]+/"), "<disk>/"),
    (re.compile(r"/home/[\w.-]+/"), "<home>/"),
]


def scrub(value: Any) -> Any:
    """Return a JSON value with foreign process names and absolute paths removed."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key in {"foreign_top", "note"}:  # free-text operator notes name services
                continue
            if key == "foreign" and isinstance(item, list) and len(item) == 2:
                out[key] = [item[0], []]  # CPU share kept, process names dropped
            else:
                out[key] = scrub(item)
        return out
    if isinstance(value, list):
        return [scrub(item) for item in value]
    if isinstance(value, str):
        for pattern, placeholder in PATHS:
            value = pattern.sub(placeholder, value)
    return value


def sha256(path: Path) -> str:
    """Return the hex SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    """Write indented UTF-8 JSON with a trailing newline."""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )


def write_gzip(path: Path, data: bytes) -> None:
    """Write gzip with a zero timestamp, so the same input gives the same bytes."""
    with (
        path.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as packed,
    ):
        packed.write(data)


def main(speed: Path, reused: Path, release: Path, out: Path) -> None:
    """Write the scrubbed copies and a manifest of the originals' digests."""
    (out / "speed").mkdir(parents=True, exist_ok=True)
    (out / "release").mkdir(parents=True, exist_ok=True)
    sources = {}
    # Every start of a run snapshots the environment, harness hashes included.
    snapshots = [(speed, "speed"), (reused, "speed/reused")]
    for src, dst in snapshots:
        (out / dst).mkdir(parents=True, exist_ok=True)
        for path in sorted(src.glob("environment*.json")):
            write_json(out / dst / path.name, scrub(json.loads(path.read_text())))
            sources[f"{dst}/{path.name}"] = sha256(path)
    write_json(
        out / "speed" / "summary.json",
        scrub(json.loads((speed / "summary.json").read_text())),
    )
    sources["speed/summary.json"] = sha256(speed / "summary.json")
    with (speed / "cells.jsonl").open(encoding="utf-8") as src:
        lines = (
            json.dumps(scrub(json.loads(line)), ensure_ascii=False) + "\n"
            for line in src
        )
        write_gzip(out / "speed" / "cells.jsonl.gz", "".join(lines).encode())
    sources["speed/cells.jsonl"] = sha256(speed / "cells.jsonl")
    for name in ("languages.txt", "REUSED.md"):
        shutil.copyfile(speed / name, out / "speed" / name)
    for name in ("evaluation.json", "EVALUATION.md"):
        shutil.copyfile(release / name, out / "release" / name)
        sources[f"release/{name}"] = sha256(release / name)
    write_gzip(
        out / "release" / "DICTIONARIES.json.gz",
        (release / "DICTIONARIES.json").read_bytes(),
    )
    sources["release/DICTIONARIES.json"] = sha256(release / "DICTIONARIES.json")
    write_json(out / "sources.json", {"sha256_of_originals": sources})


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    main(*(Path(arg) for arg in sys.argv[1:]))
