"""Lay out the unversioned upstream inputs as release assets and pin their URLs.

Each listed source's cached files become assets named ``<source>-<file>``,
split below GitHub's 2 GiB asset limit; ``sources.json`` gains the asset URLs
as ``mirror`` next to the upstream URL. Uploading the directory is left to the
release owner.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import POLICY, SOURCES, read_json, sha256, write_json

PART = 1900 << 20


def asset_name(source: str, url: str) -> str:
    """A stable, URL-safe asset name for one upstream file."""
    name = url.rstrip("/").rsplit("/", 1)[-1].split("?file=")[-1].split("?")[0]
    return f"{source}-{name}"


def mirror(cache: Path, out: Path) -> None:
    """Write the assets and their SHA256SUMS; record mirror URLs in sources.json."""
    policy = read_json(POLICY)["mirror"]
    config = read_json(SOURCES)
    out.mkdir(parents=True, exist_ok=True)
    sums = []
    for source in config["sources"]:
        if source["id"] not in policy["sources"]:
            continue
        for item in source["files"]:
            name = asset_name(source["id"], item["url"])
            data = cache / item["sha256"]
            count = -(-data.stat().st_size // PART)
            names = (
                [name] if count == 1 else [f"{name}.part{n + 1}" for n in range(count)]
            )
            with data.open("rb") as stream:
                for part in names:
                    target = out / part
                    with target.open("wb") as dest:
                        left = PART
                        while left and (chunk := stream.read(min(left, 1 << 24))):
                            dest.write(chunk)
                            left -= len(chunk)
                    sums.append(f"{sha256(target)}  {part}\n")
            item["mirror"] = [policy["base"] + part for part in names]
    (out / "SHA256SUMS").write_text("".join(sums), encoding="utf-8")
    write_json(SOURCES, config)


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    mirror(args.cache, args.out)


if __name__ == "__main__":
    main()
