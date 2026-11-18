"""Independently verify all wheels, portable archives, manifests, bytes and engine lookups."""

import argparse
import base64
import csv
import hashlib
import io
import json
import tarfile
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import udlex

from fetch_sources import sha


def verify(catalog, report):
    meta = json.loads((catalog / "DICTIONARIES.json").read_bytes())
    if udlex.__version__ != "1.0.0" or len(meta["language_codes"]) != 172:
        raise ValueError("Unexpected engine or roster")
    counts = Counter()
    roots = set()
    license_review = []
    with tempfile.TemporaryDirectory() as tmp:
        installation = Path(tmp) / "udlex_bundle"
        for row in meta["dictionaries"]:
            iso = row["iso639_3"]
            paths = {}
            for kind in ("wheel", "archive"):
                info = row[kind]
                path = catalog / "packages" / info["name"]
                if sha(path) != info["sha256"] or path.stat().st_size != info["size"]:
                    raise ValueError(f"Bad {iso} {kind}")
                paths[kind] = path
            with (
                zipfile.ZipFile(paths["wheel"]) as wheel,
                tarfile.open(paths["archive"]) as parcel,
            ):
                if wheel.testzip():
                    raise ValueError(f"Damaged wheel for {iso}")
                names = wheel.namelist()
                record = next(n for n in names if n.endswith(".dist-info/RECORD"))
                entries = {
                    name: (digest, size)
                    for name, digest, size in csv.reader(
                        io.StringIO(wheel.read(record).decode())
                    )
                }
                if set(names) != set(entries) or len(names) != len(set(names)):
                    raise ValueError(f"Wheel inventory mismatch for {iso}")
                expected = set()
                for name in names:
                    if name == record:
                        continue
                    data = wheel.read(name)
                    encoded = (
                        base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                        .decode()
                        .rstrip("=")
                    )
                    if entries[name] != ("sha256=" + encoded, str(len(data))):
                        raise ValueError(f"Invalid wheel RECORD for {name}")
                    if name.startswith("udlex_bundle/"):
                        relative = Path(name.removeprefix("udlex_bundle/"))
                        if relative.is_absolute() or ".." in relative.parts:
                            raise ValueError("Invalid path")
                        dest = installation / relative
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(data)
                        tarname = "bundle/" + str(relative)
                    elif name.endswith("/SOURCES.json"):
                        tarname = "bundle/" + iso + "/SOURCES.json"
                    elif "/licenses/" in name:
                        tarname = (
                            "bundle/"
                            + iso
                            + "/LICENSES/"
                            + name.split("/licenses/", 1)[1]
                        )
                    else:
                        continue
                    expected.add(tarname)
                    with parcel.extractfile(tarname) as stream:
                        if stream.read() != data:
                            raise ValueError(
                                f"Portable and wheel payload differ for {name}"
                            )
                if set(parcel.getnames()) != expected:
                    raise ValueError(f"Unexpected portable content for {iso}")
            folder = installation / iso
            gen = (folder / "current").read_text().strip()
            if gen != row["compiled_generation"] or iso in roots:
                raise ValueError(f"Generation or roster mismatch for {iso}")
            roots.add(iso)
            payload = folder / ".generations" / gen
            m = json.loads((payload / "meta.json").read_bytes())
            if (
                m["iso639_3"] != iso
                or m["generation"] != gen
                or Counter(m["entries"]) != Counter(row["entries"])
            ):
                raise ValueError(f"Metadata mismatch for {iso}")
            for name, r in m["files"].items():
                if (
                    sha(payload / name) != r["sha256"]
                    or (payload / name).stat().st_size != r["size"]
                ):
                    raise ValueError(f"Corrupt table for {iso}")
            lex = udlex.load(str(installation), iso, use_user=False)
            for x, y in row["examples"].items():
                if lex.lemmatize([x])[0] != y:
                    raise ValueError(f"Lookup mismatch for {iso}")
                counts["examples"] += 1
            counts["dictionaries"] += 1
            counts["wheels"] += 1
            counts["archives"] += 1
            counts["with_suffix_rules"] += bool(m["entries"]["rules"])
            counts["rule_entries"] += m["entries"]["rules"]
            counts["training_observations"] += m["training_tokens"]
            if row["requires_license_review"]:
                license_review.append(iso)
    if roots != set(meta["language_codes"]):
        raise ValueError("Missing language")
    result = {
        "catalog_sha256": sha(catalog / "DICTIONARIES.json"),
        "engine": udlex.__version__,
        "checks": dict(counts),
        "license_review": license_review,
        "source_modes": dict(Counter(r["input_mode"] for r in meta["dictionaries"])),
    }
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("catalog", type=Path)
    p.add_argument("report", type=Path)
    a = p.parse_args()
    verify(a.catalog, a.report)
