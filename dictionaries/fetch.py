"""Download every pinned input into a content-addressed cache.

``--pin`` downloads entries that have no SHA-256 yet and writes the digest,
size and upstream ``Last-Modified`` back into ``sources.json``; a normal run
refuses any byte that differs from the pin. A source with ``mirror`` archives
of its extracted pairs falls back to them when an upstream file is gone or has
changed.
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


def fetch_item(item: dict[str, Any], cache: Path) -> bool:
    """Download one upstream file into the cache; return whether its pin changed."""
    digest = item.get("sha256")
    pending = cache / "pending" / Path(item["url"]).name
    modified = download(item["url"], pending)
    found = sha256(pending)
    if digest and found != digest:
        msg = f"{item['url']}: SHA-256 {found}, pinned {digest}"
        raise ValueError(msg)
    pending.replace(cache / found)
    print(found, item["url"], flush=True)
    time.sleep(PAUSE)
    if digest and item.get("size"):
        return False
    item.update(sha256=found, size=(cache / found).stat().st_size)
    if modified and not digest:
        item["last_modified"] = modified
    return True


def fetch_source(source: dict[str, Any], cache: Path, *, pin: bool) -> bool:
    """Fill the cache for one source; return whether ``sources.json`` changed.

    A source with ``mirror`` archives of its extracted pairs falls back to them
    when an upstream file is gone or no longer matches its pin; extraction then
    starts from the pairs.
    """
    changed = False
    missing = False
    for item in source["files"]:
        if item.get("sha256") and (cache / item["sha256"]).is_file():
            continue
        if not item.get("sha256") and not pin:
            msg = f"{source['id']}: unpinned input {item['url']}"
            raise ValueError(msg)
        try:
            changed |= fetch_item(item, cache)
        except (OSError, ValueError) as error:
            if pin or not source.get("mirror"):
                raise
            print(f"{source['id']}: {error}; using the mirror", file=sys.stderr)
            missing = True
    for part in source.get("mirror", []) if missing else []:
        if not (cache / part["sha256"]).is_file():
            fetch_item(part, cache)
    return changed


def fetch(cache: Path, *, pin: bool, only: set[str] | None) -> None:
    """Fill the cache; with ``pin`` record digests of unpinned entries."""
    config = read_json(SOURCES)
    sources: list[dict[str, Any]] = [{"id": "ud", "files": [config["ud"]]}]
    for source in sources + config["sources"]:
        if (not only or source["id"] in only) and fetch_source(source, cache, pin=pin):
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
