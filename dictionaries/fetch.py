"""Download every pinned input into a content-addressed cache.

``--pin`` downloads entries that have no SHA-256 yet and writes the digest,
size and upstream ``Last-Modified`` back into ``sources.json``; a normal run
refuses any byte that differs from the pin. An entry with ``mirror`` parts
falls back to them, concatenated, when its upstream URL fails or has changed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from common import SOURCES, read_json, sha256, write_json

AGENT = "udlex-dictionary-build/1.0"
PAUSE = 1.0


def download(url: str, target: Path) -> str | None:
    """Fetch one URL with retries and backoff; return ``Last-Modified``."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": AGENT})  # noqa: S310
            with (
                urllib.request.urlopen(request, timeout=300) as response,  # noqa: S310
                partial.open("wb") as out,
            ):
                shutil.copyfileobj(response, out, 1 << 20)
                modified = response.headers.get("Last-Modified")
            partial.replace(target)
        except OSError as error:
            if attempt == 4:
                raise
            print(f"retry {url}: {error}", file=sys.stderr, flush=True)
            time.sleep(5 * 3**attempt)
        else:
            return modified
    return None


def retrieve(item: dict[str, Any], target: Path) -> str | None:
    """Upstream first; the pinned mirror when upstream is gone or has changed."""
    try:
        modified = download(item["url"], target)
        if not item.get("sha256") or sha256(target) == item["sha256"]:
            return modified
        print(f"upstream changed: {item['url']}", file=sys.stderr, flush=True)
    except OSError:
        if not item.get("mirror"):
            raise
        print(f"upstream failed: {item['url']}", file=sys.stderr, flush=True)
    if not item.get("mirror"):
        return None
    with target.open("wb") as out:
        for n, url in enumerate(item["mirror"]):
            part = target.with_name(f"{target.name}.part{n}")
            download(url, part)
            with part.open("rb") as stream:
                shutil.copyfileobj(stream, out, 1 << 20)
            part.unlink()
    return None


def fetch(cache: Path, *, pin: bool, only: set[str] | None) -> None:
    """Fill the cache; with ``pin`` record digests of unpinned entries."""
    config = read_json(SOURCES)
    changed = False
    sources: list[dict[str, Any]] = [{"id": "ud", "files": [config["ud"]]}]
    for source in sources + config["sources"]:
        if only and source["id"] not in only:
            continue
        for item in source["files"]:
            digest = item.get("sha256")
            if digest and (cache / digest).is_file():
                continue
            if not digest and not pin:
                msg = f"{source['id']}: unpinned input {item['url']}"
                raise ValueError(msg)
            pending = cache / "pending" / Path(item["url"]).name
            pending.parent.mkdir(parents=True, exist_ok=True)
            modified = retrieve(item, pending)
            found = sha256(pending)
            if digest and found != digest:
                msg = f"{item['url']}: SHA-256 {found}, pinned {digest}"
                raise ValueError(msg)
            pending.replace(cache / found)
            if not digest or not item.get("size"):
                item.update(sha256=found, size=(cache / found).stat().st_size)
                if modified and not digest:
                    item["last_modified"] = modified
                changed = True
                write_json(SOURCES, config)
            print(source["id"], found, item["url"], flush=True)
            time.sleep(PAUSE)
    if changed:
        write_json(SOURCES, config)


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--pin", action="store_true")
    parser.add_argument("--only", nargs="*", help="source ids to fetch")
    args = parser.parse_args()
    fetch(args.cache, pin=args.pin, only=set(args.only) if args.only else None)


if __name__ == "__main__":
    main()
