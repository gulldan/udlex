"""Package every built lexicon as a wheel and a portable archive with its licenses.

The package license is CC BY-NC-SA 4.0 with a ``-nc`` distribution suffix when
the treebank base or any dictionary is non-commercial; otherwise it is the
strictest license among the base and the dictionaries used. ``DICTIONARIES.json``
records, per language, every input digest, the selection evidence and a recipe.
"""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import io
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Any

from common import (
    HERE,
    POLICY,
    SOURCES,
    STRICTNESS,
    digest_bytes,
    json_bytes,
    license_class,
    read_json,
    sha256,
    write_json,
)

import udlex

LICENSE_URLS = {
    "CC-BY-SA-3.0": "https://creativecommons.org/licenses/by-sa/3.0/",
    "CC-BY-SA-4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
    "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "CC-BY-NC-SA-4.0": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
}
DATE = (2026, 11, 19, 0, 0, 0)


def package_license(components: list[str]) -> tuple[str, bool]:
    """Strictest compatible license of the components and whether it is NC."""
    if any(license_class(c) == "nc" for c in components):
        return "CC-BY-NC-SA-4.0", True
    known = [c for c in components if c in STRICTNESS]
    if len(known) != len(components):
        msg = f"license outside the strictness order: {components}"
        raise ValueError(msg)
    return max(known, key=STRICTNESS.index), False


def runtime_files(lexicon: Path, stage: Path, iso: str) -> tuple[str, dict[str, bytes]]:
    """Prune a copy of the lexicon to its runtime files; return generation and files."""
    folder = stage / iso
    if folder.exists():
        shutil.rmtree(folder)
    shutil.copytree(lexicon, folder, symlinks=True)
    udlex.prune_bundle(str(folder))
    generation = (folder / "current").read_text().strip()
    root = folder / ".generations" / generation
    files = {f"{iso}/current": (generation + "\n").encode()}
    files.update(
        {
            f"{iso}/.generations/{generation}/{f.name}": f.read_bytes()
            for f in sorted(root.iterdir())
            if f.is_file() and (f.name == "meta.json" or not f.name.endswith(".tsv"))
        }
    )
    return generation, files


def notice(entry: dict, conditions: str | None) -> bytes:
    """Attribution for the treebank base and every dictionary used."""
    lines = [f"udlex dictionary for {entry['language']} ({entry['iso639_3']}).", ""]
    corpus = entry["corpus"]
    if corpus:
        lines += [
            (
                f"Base: Universal Dependencies 2.19, {corpus['treebank']} "
                f"({corpus['readme_license']}; package base {corpus['license']})."
            ),
            f"https://github.com/UniversalDependencies/{corpus['treebank']}/tree/r2.19",
            "Its README and LICENSE are included as TREEBANK-README, TREEBANK-LICENSE.",
        ]
        if conditions:
            lines.append(f"Special conditions: {conditions}")
        lines.append("")
    for item in entry["sources"]:
        for member in item["members"]:
            lines += [
                f"{member['id']}: {member['attribution']}.",
                f"License: {item['license']} {LICENSE_URLS.get(item['license'], '')}",
                f"Revision: {member['revision']}. Home: {member['home']}",
                "",
            ]
    lines.append(
        "Lookup tables and suffix rules were learned with udlex from the inputs above;"
        " see SOURCES.json for digests and the selection evidence."
    )
    return ("\n".join(lines) + "\n").encode()


def write_wheel(
    out: Path, entry: dict, runtime: dict[str, bytes], docs: dict[str, bytes]
) -> dict:
    """Wheel and portable archive, both byte-reproducible."""
    iso, name, version = (
        entry["iso639_3"],
        entry["distribution"],
        entry["dictionary_version"],
    )
    dist = name.replace("-", "_")
    info = f"{dist}-{version}.dist-info"
    manifest = {
        "distribution": dist,
        "format_version": 5,
        "language_aliases": {},
        "lexicons": [
            {"config": iso, "generation": entry["compiled_generation"], "iso639_3": iso}
        ],
        "schema": "udlex-dictionary-wheel-manifest/v1",
    }
    runtime = dict(runtime)
    runtime[f".manifests/udlex_dict_{iso}.json"] = json_bytes(manifest)
    provenance = json_bytes(entry)
    data = {f"udlex_bundle/{k}": v for k, v in runtime.items()}
    data.update({f"{info}/licenses/{k}": v for k, v in docs.items()})
    data[f"{info}/SOURCES.json"] = provenance
    data[f"{info}/METADATA"] = (
        f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n"
        f"Summary: {entry['language']} dictionary for udlex\nRequires-Python: >=3.11\n"
        f"Requires-Dist: udlex (>=1.0.0,<2)\nLicense-Expression: {entry['license']}\n"
        + "".join(f"License-File: {f}\n" for f in sorted(docs))
        + "\n"
    ).encode()
    data[f"{info}/WHEEL"] = (
        b"Wheel-Version: 1.0\nGenerator: udlex-dictionaries\n"
        b"Root-Is-Purelib: true\nTag: py3-none-any\n"
    )
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for path, blob in sorted(data.items()):
        hashed = (
            base64.urlsafe_b64encode(hashlib.sha256(blob).digest()).decode().rstrip("=")
        )
        writer.writerow([path, "sha256=" + hashed, str(len(blob))])
    writer.writerow([f"{info}/RECORD", "", ""])
    data[f"{info}/RECORD"] = record.getvalue().encode()
    wheel = out / f"{dist}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, blob in sorted(data.items()):
            member = zipfile.ZipInfo(path, date_time=DATE)
            member.external_attr = 0o100644 << 16
            archive.writestr(member, blob, compress_type=zipfile.ZIP_DEFLATED)
    portable = {f"bundle/{k}": v for k, v in runtime.items()}
    portable.update({f"bundle/{iso}/LICENSES/{k}": v for k, v in docs.items()})
    portable[f"bundle/{iso}/SOURCES.json"] = provenance
    bundle = out / f"{name}-{version}-bundle.tar.gz"
    with (
        bundle.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as gz,
        tarfile.open(fileobj=gz, mode="w|") as tar,
    ):
        for path, blob in sorted(portable.items()):
            member = tarfile.TarInfo(path)
            member.size, member.mode = len(blob), 0o644
            tar.addfile(member, io.BytesIO(blob))
    return {
        k: {"name": p.name, "size": p.stat().st_size, "sha256": sha256(p)}
        for k, p in (("wheel", wheel), ("archive", bundle))
    }


def scripts_digest() -> dict[str, str]:
    """Digests of the build code, part of every recipe."""
    return {p.name: sha256(p) for p in sorted(HERE.glob("*.py"))} | {
        p.name: sha256(p) for p in sorted(HERE.glob("*.json"))
    }


def package_one(iso: str, row: dict, work: Path, out: Path, commit: str) -> dict | None:
    """Assemble one language's entry, documents and artifacts."""
    folder = work / "build" / iso
    if not (folder / "decision.json").is_file():
        return None
    decision = read_json(folder / "decision.json")
    policy = read_json(POLICY)
    config = read_json(SOURCES)
    sources = {s["id"]: s for s in config["sources"]}
    used = [s for s in decision["sources"] if s["id"] in decision["order"]]
    used.sort(key=lambda s: decision["order"].index(s["id"]))
    unused = [
        s | {"reason": "not in the dev-selected order"}
        for s in decision["sources"]
        if s["id"] not in decision["order"]
    ]
    corpus: dict[str, Any] | None = None
    docs: dict[str, bytes] = {}
    if row["treebank"]:
        bank = work / "ud" / "ud-treebanks-v2.19" / row["treebank"]
        lic = row["license"]
        corpus = {
            "treebank": row["treebank"],
            "config": row["config"],
            "mode": row["mode"],
            "readme_license": lic["readme_license"],
            "license_file_expression": lic["license_file_expression"],
            "license": lic["license"],
            "files": [
                {"name": p.name, "sha256": sha256(p)}
                for p in sorted(bank.glob(f"{row['config']}-ud-*.conllu"))
            ],
            "splits": row.get("splits") or row.get("folds"),
        }
        docs["TREEBANK-README"] = (bank / lic["readme"]).read_bytes()
        if lic["license_file"]:
            docs["TREEBANK-LICENSE"] = (bank / lic["license_file"]).read_bytes()
    components: list[str] = [corpus["license"]] if corpus else []
    components += [s["license"] for s in used]
    spdx, noncommercial = package_license(components)
    conditions = policy["special_conditions"].get(row["config"] or "")
    generation, runtime = runtime_files(folder / "lexicon" / iso, work / "stage", iso)
    source_entries = [
        {
            "id": s["id"],
            "family": s["family"],
            "license": s["license"],
            "rows": s.get("rows"),
            "members": [
                {
                    "id": m,
                    "revision": sources[m]["revision"],
                    "attribution": sources[m]["attribution"],
                    "home": sources[m]["home"],
                    "inputs": [
                        {"url": f["url"], "sha256": f["sha256"]}
                        for f in sources[m]["files"]
                        if iso in f.get("languages", [iso])
                    ],
                }
                for m in s["members"]
            ],
            "table_sha256": s["pairs_sha256"],
        }
        for s in used
    ]
    recipe = {
        "engine_version": udlex.__version__,
        "build_commit": commit,
        "scripts": scripts_digest(),
        "ud_archive_sha256": config["ud"]["sha256"],
        "corpus_files": corpus["files"] if corpus else [],
        "inputs": {
            m["id"]: [f["sha256"] for f in m["inputs"]]
            for s in source_entries
            for m in s["members"]
        },
        "tables": {s["id"]: s["table_sha256"] for s in source_entries},
        "order": decision["order"],
        "options": decision["options"],
    }
    entry = {
        "iso639_3": iso,
        "language": row["language"],
        "distribution": f"udlex-dict-{iso}" + ("-nc" if noncommercial else ""),
        "dictionary_version": policy["dictionary_version"],
        "bundle_format": 5,
        "license": spdx,
        "noncommercial": noncommercial,
        "requires_license_review": bool(
            corpus and row["license"]["requires_license_review"]
        ),
        "special_conditions": conditions,
        "corpus": corpus,
        "evaluation": {"official": "test", "twofold": "two-fold", "no-lemmas": None}[
            row["mode"]
        ],
        "sources": source_entries,
        "excluded_sources": [
            {k: s[k] for k in ("id", "license", "reason")}
            for s in decision["excluded"] + unused
        ],
        "order": decision["order"],
        "selection": {
            k: decision[k]
            for k in (
                "policies",
                "order_gate",
                "option_gate",
                "dev_declared",
                "dev_chosen",
                "copyleft_cost",
            )
            if k in decision
        },
        "options": decision["options"],
        "build": decision["outcome"],
        "compiled_generation": generation,
        "recipe": recipe,
        "recipe_sha256": digest_bytes(json_bytes(recipe)),
    }
    docs["NOTICE"] = notice(entry, conditions)
    for member in (m for s in used for m in s["members"]):
        # BSD and Apache terms travel with the data: upstream notices, verbatim.
        block = work / "pairs" / member / "NOTICE.txt"
        if block.is_file():
            docs[f"{member}-NOTICE"] = block.read_bytes()
        for item in sources[member]["files"]:
            if "notice" in item:
                path = work.parent / "cache" / item["sha256"]
                docs[f"{member}-{item['notice']}"] = path.read_bytes()
    entry |= write_wheel(out / "packages", entry, runtime, docs)
    return entry


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    commit = (
        subprocess.run(  # noqa: S603
            ["git", "-C", str(HERE), "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        or "unknown"
    )
    roster = read_json(args.work / "roster.json")
    (args.output / "packages").mkdir(parents=True, exist_ok=True)
    entries = []
    for iso, row in sorted(roster.items()):
        entry = package_one(iso, row, args.work, args.output, commit)
        if entry:
            entries.append(entry)
            print(iso, entry["distribution"], entry["license"], flush=True)
    policy = read_json(POLICY)
    write_json(
        args.output / "DICTIONARIES.json",
        {
            "release": policy["release"],
            "dictionary_version": policy["dictionary_version"],
            "engine_version": udlex.__version__,
            "ud_revision": "2.19",
            "ud_archive_sha256": read_json(SOURCES)["ud"]["sha256"],
            "languages_packaged": len(entries),
            "noncommercial_languages": [
                e["iso639_3"] for e in entries if e["noncommercial"]
            ],
            "language_codes": [e["iso639_3"] for e in entries],
            "dictionaries": entries,
        },
    )


if __name__ == "__main__":
    main()
