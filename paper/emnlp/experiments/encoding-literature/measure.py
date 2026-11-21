#!/usr/bin/env python3
"""Measure lossless encodings of a catalog's long exact.hybrid records."""

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _measure(catalog_dir: Path, binary: Path) -> dict[str, object]:
    catalog_bytes = (catalog_dir / "DICTIONARIES.json").read_bytes()
    catalog = json.loads(catalog_bytes)
    totals = Counter()
    rows = {}
    for entry in catalog["dictionaries"]:
        iso = entry["iso639_3"]
        wheel = catalog_dir / "packages" / entry["wheel"]["name"]
        if _sha256(wheel.read_bytes()) != entry["wheel"]["sha256"]:
            msg = f"{iso}: wheel digest mismatch"
            raise ValueError(msg)
        with zipfile.ZipFile(wheel) as archive:
            (meta_name,) = (p for p in archive.namelist() if p.endswith("/meta.json"))
            meta = json.loads(archive.read(meta_name))
            if (
                meta["generation"] != entry["compiled_generation"]
                or meta["format_version"] != 5
            ):
                msg = f"{iso}: unexpected generation or format"
                raise ValueError(msg)
            root = meta_name.removesuffix("meta.json")
            sizes = {name: info["size"] for name, info in meta["files"].items()}
            if any(
                archive.getinfo(root + name).file_size != size
                for name, size in sizes.items()
            ):
                msg = f"{iso}: unexpected member size"
                raise ValueError(msg)
            row = {
                "runtime": sum(sizes.values()),
                "exact": sizes.get("exact.hybrid", 0),
                "lower": sizes.get("lower.hybrid", 0),
                "strings_txt": sizes.get("strings.txt", 0),
                "strings_idx": sizes.get("strings.idx", 0),
                "external_entries": meta["entries"]["external"],
            }
            if row["exact"]:
                image = archive.read(root + "exact.hybrid")
                if _sha256(image) != meta["files"]["exact.hybrid"]["sha256"]:
                    msg = f"{iso}: exact.hybrid digest mismatch"
                    raise ValueError(msg)
                # The caller supplies the locally compiled measurement binary.
                result = subprocess.run(
                    [binary],
                    input=image,
                    stdout=subprocess.PIPE,
                    check=True,
                )
                parts = dict(item.split("=") for item in result.stdout.decode().split())
                row.update({key: int(value) for key, value in parts.items()})
                if row["prefix"] + row["raw"] != row["exact"]:
                    msg = f"{iso}: incomplete image accounting"
                    raise ValueError(msg)
            elif meta["entries"]["exact"]:
                msg = f"{iso}: an exact table needs a different reader"
                raise ValueError(msg)
        rows[iso] = row
        totals.update(row)
    return {
        "catalog_sha256": _sha256(catalog_bytes),
        "dictionary_version": catalog["dictionary_version"],
        "engine_version": catalog["engine_version"],
        "optfsst_commit": "be06199760288005e2ebdc11fbd0f087903c0476",
        "totals": dict(totals),
        "languages": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog_dir", type=Path)
    parser.add_argument("binary", type=Path)
    args = parser.parse_args()
    json.dump(_measure(args.catalog_dir, args.binary.resolve()), sys.stdout, indent=2)
    print()
