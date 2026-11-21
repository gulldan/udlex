#!/usr/bin/env python3
"""Profile long-slot byte edits in the same verified catalog as the size pilot."""

import argparse
import hashlib
import json
import math
import struct
import sys
import zipfile
from collections import Counter
from pathlib import Path

LENGTHS = struct.Struct("<HH")


def _profile(image: bytes, expected: dict[str, int]) -> dict[str, int | float]:
    if image[:8] != b"UDLXHYB1" or len(image) != expected["exact"]:
        msg = "Unexpected hybrid image"
        raise ValueError(msg)
    buckets, slots, _, blob_bytes = struct.unpack_from("<4Q", image, 40)
    start = len(image) - blob_bytes
    if blob_bytes != expected["raw"] or start != expected["prefix"]:
        msg = "Record blob disagrees with the size pilot"
        raise ValueError(msg)
    counts: Counter[tuple[int, bytes]] = Counter()
    # Occupied records are contiguous. An empty slot occupies zero blob bytes.
    pos = start
    while pos < len(image):
        form_size, lemma_size = LENGTHS.unpack_from(image, pos)
        pos += LENGTHS.size
        form = image[pos : pos + form_size]
        pos += form_size
        lemma = image[pos : pos + lemma_size]
        pos += lemma_size
        if form == lemma:
            counts[0, b""] += 1
        else:
            limit = min(form_size, lemma_size)
            keep = 0
            while keep < limit and form[keep] == lemma[keep]:
                keep += 1
            counts[form_size - keep, lemma[keep:]] += 1
    n = counts.total()
    if pos != len(image) or n != expected["long_keys"]:
        msg = "Decoded record count or length disagrees with the size pilot"
        raise ValueError(msg)
    entropy = sum(v * math.log2(n / v) for v in counts.values()) / n if n else 0.0
    return {
        "long_keys": n,
        "distinct_edits": len(counts),
        "identity_keys": counts[0, b""],
        "most_frequent_edit_keys": max(counts.values(), default=0),
        "h0_bits_per_key": round(entropy, 6),
        "pilot_array_bytes": 2 * buckets,
        "slot_offset_bytes": 4 * (slots + 1),
    }


def _run(catalog_dir: Path) -> dict[str, object]:
    pilot = json.loads(Path(__file__).with_name("pilot.json").read_bytes())
    catalog = (catalog_dir / "DICTIONARIES.json").read_bytes()
    digest = hashlib.sha256(catalog).hexdigest()
    if digest != pilot["catalog_sha256"]:
        msg = "Catalog differs from the size pilot"
        raise ValueError(msg)
    rows = {}
    for entry in json.loads(catalog)["dictionaries"]:
        iso = entry["iso639_3"]
        expected = pilot["languages"][iso]
        if expected["exact"]:
            wheel = catalog_dir / "packages" / entry["wheel"]["name"]
            wheel_digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
            if wheel_digest != entry["wheel"]["sha256"]:
                msg = f"{iso}: wheel checksum mismatch"
                raise ValueError(msg)
            with zipfile.ZipFile(wheel) as archive:
                (name,) = (p for p in archive.namelist() if p.endswith("/exact.hybrid"))
                rows[iso] = _profile(archive.read(name), expected)
    return {"catalog_sha256": digest, "languages": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog_dir", type=Path)
    args = parser.parse_args()
    json.dump(_run(args.catalog_dir), sys.stdout, indent=2)
    print()
