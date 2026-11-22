"""Pack the extracted pairs of unversioned upstream inputs as release assets.

Each listed source's ``work/pairs/<id>`` (one ``<iso>.tsv.gz`` per language
and the manifest naming the upstream digests) becomes ``pairs-<id>.tar``, split
below GitHub's 2 GiB asset limit, and ``sources.json`` records the archive URLs
and digests as the source's ``mirror``. Uploading is left to the release owner.
"""

from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path

from common import POLICY, SOURCES, read_json, sha256, write_json

PART = 1900 << 20


def pack(folder: Path, target: Path) -> None:
    """A byte-reproducible tar of one pairs directory."""
    with (
        target.open("wb") as raw,
        tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as tar,
    ):
        for path in sorted(folder.iterdir()):
            data = path.read_bytes()
            member = tarfile.TarInfo(path.name)
            member.size, member.mode = len(data), 0o644
            tar.addfile(member, io.BytesIO(data))


def split(archive: Path) -> list[Path]:
    """The archive itself, or numbered parts of at most ``PART`` bytes."""
    if archive.stat().st_size <= PART:
        return [archive]
    parts = []
    with archive.open("rb") as stream:
        while chunk := stream.read(PART):
            part = archive.with_name(f"{archive.name}.part{len(parts) + 1}")
            part.write_bytes(chunk)
            parts.append(part)
    archive.unlink()
    return parts


def mirror(work: Path, out: Path) -> None:
    """Write the archives and their SHA256SUMS; record them in sources.json."""
    policy = read_json(POLICY)["mirror"]
    config = read_json(SOURCES)
    out.mkdir(parents=True, exist_ok=True)
    sums = []
    for source in (s for s in config["sources"] if s["id"] in policy["sources"]):
        folder = work / "pairs" / source["id"]
        inputs = [f["sha256"] for f in source["files"] if "notice" not in f]
        if read_json(folder / "manifest.json")["inputs"] != inputs:
            msg = f"{folder}: pairs extracted from other inputs than the pins"
            raise ValueError(msg)
        archive = out / f"pairs-{source['id']}.tar"
        pack(folder, archive)
        parts = split(archive)
        source["mirror"] = [
            {
                "url": policy["base"] + p.name,
                "sha256": sha256(p),
                "size": p.stat().st_size,
            }
            for p in parts
        ]
        sums += [
            f"{m['sha256']}  {p.name}\n"
            for m, p in zip(source["mirror"], parts, strict=True)
        ]
        for item in source["files"]:
            item.pop("mirror", None)
    (out / "SHA256SUMS").write_text("".join(sums), encoding="utf-8")
    write_json(SOURCES, config)


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    mirror(args.work, args.out)


if __name__ == "__main__":
    main()
