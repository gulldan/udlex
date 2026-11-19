"""Package every built lexicon as a wheel and a portable archive with its licenses.

Each table keeps the license of its source: the corpus tables (exact, lower,
rules, entities) that of the treebank, every row of the merged external table
that of the dictionary it came from. The package as a whole is CC BY-NC-SA 4.0
with a ``-nc`` distribution suffix when any part is non-commercial; otherwise it
is the strictest license of its parts. ``DICTIONARIES.json`` records, per
language, every input digest, the part licenses, the selection evidence and a
recipe.
"""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import io
import itertools
import re
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from collections.abc import Iterator


def license_url(spdx: str) -> str:
    """Canonical deed URL of a Creative Commons license, empty for others."""
    if spdx == "CC0-1.0":
        return "https://creativecommons.org/publicdomain/zero/1.0/"
    if not spdx.startswith("CC-"):
        return ""
    *kind, version = spdx.removeprefix("CC-").split("-")
    return f"https://creativecommons.org/licenses/{'-'.join(kind).lower()}/{version}/"


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


def lines_of(path: Path) -> Iterator[tuple[str, str]]:
    """``(form, lemma)`` rows of a resolved, form-sorted dictionary table."""
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            form, lemma = line.rstrip("\n").split("\t")
            yield form, lemma


def external_provenance(lexicon: Path, tables: Path, order: list[str]) -> dict:
    """Which dictionary every external row came from, and whether it changed.

    The external table is sorted like the resolved dictionaries, so one merge
    finds, for each row, the first dictionary of the order holding its form:
    the one the build took it from. The runtime gives the shipped lemma; one
    that differs from the dictionary's was respelled from treebank lemmas by
    the lemma adapter or the compound marker, or kept as the form by the
    engine's guard against a lemma in another script.
    """
    generation = lexicon / ".generations" / (lexicon / "current").read_text().strip()
    lex = udlex.Lexicon(
        str(lexicon),
        use_user=False,
        use_exact=False,
        use_lower=False,
        use_exact_lower=False,
        use_rules=False,
        use_external_lower=False,
    )
    streams = [lines_of(tables / f"{name}.tsv") for name in order]
    heads = [next(s, None) for s in streams]
    counts = {name: {"rows": 0, "changed": 0} for name in order}
    unattributed = 0
    with (generation / "external.tsv").open(encoding="utf-8") as rows:
        while chunk := [
            line.split("\t", 1)[0] for line in itertools.islice(rows, 100_000)
        ]:
            shipped = lex.lemmatize(chunk, sentence_initial=[False] * len(chunk))
            for form, lemma in zip(chunk, shipped, strict=True):
                owner = None
                for j, stream in enumerate(streams):
                    while (head := heads[j]) is not None and head[0] < form:
                        heads[j] = next(stream, None)
                    head = heads[j]
                    if owner is None and head is not None and head[0] == form:
                        owner = (order[j], head[1])
                if owner is None:
                    unattributed += 1
                    continue
                count = counts[owner[0]]
                count["rows"] += 1
                count["changed"] += owner[1] != lemma
    return {"sources": counts, "unattributed": unattributed}


def table_parts(
    entry: dict, runtime: dict[str, bytes], provenance: dict | None
) -> list[dict]:
    """License of every part of the generation, with the files that hold it."""
    files: dict[str, list[str]] = {}
    for path in runtime:
        name = path.rsplit("/", 1)[-1]
        if "/.generations/" in path:
            files.setdefault(name.split(".", 1)[0], []).append(name)
    corpus, sources = entry["corpus"], entry["dictionaries"]
    learned = [k for k in ("exact", "lower", "rules", "entities") if k in files]
    parts: list[dict[str, Any]] = []
    if corpus:
        parts.append(
            {
                "part": "corpus",
                "tables": learned,
                "files": sorted(f for k in learned for f in files[k]),
                "source": corpus["treebank"],
                "license": corpus["part_license"],
                "rows": {k: entry["build"]["entries"].get(k) for k in learned},
            }
        )
        dictionaries = sources
    elif sources:
        parts.append(
            {
                "part": "training dictionary",
                "tables": learned,
                "files": sorted(f for k in learned for f in files[k]),
                "source": sources[0]["id"],
                "license": sources[0]["license"],
                "rows": {k: entry["build"]["entries"].get(k) for k in learned},
            }
        )
        dictionaries = sources[1:]
    else:
        dictionaries = []
    # A dictionary whose rows the chosen variant does not ship is no part of it.
    counts = (provenance or {}).get("sources", {})
    parts += [
        {
            "part": "external",
            "tables": ["external"],
            "files": files["external"],
            "source": s["id"],
            "license": s["license"],
            "rows": counts[s["id"]]["rows"],
            "changed_lemmas": counts[s["id"]]["changed"],
        }
        for s in dictionaries
        if "external" in files and counts.get(s["id"], {}).get("rows")
    ]
    shared = sorted({p["license"] for p in parts})
    parts.append(
        {
            "part": "shared",
            "tables": ["strings", "meta"],
            "files": sorted(files.get("strings", []) + files.get("meta", [])),
            "source": "all parts above",
            "license": entry["license"],
            "licenses_combined": shared,
        }
    )
    return parts


def source_list(entry: dict, treebank_readme: bytes) -> list[dict]:
    """Every source as spaCy's ``meta.json`` lists them, plus the files it fills."""
    parts = {(p["part"], p["source"]): p for p in entry["tables"]}
    listed = []
    corpus = entry["corpus"]
    if corpus:
        found = re.search(rb"^Contributors:\s*(.+)$", treebank_readme, re.MULTILINE)
        part = parts[("corpus", corpus["treebank"])]
        listed.append(
            {
                "name": f"{corpus['treebank']} (Universal Dependencies 2.19)",
                "url": f"https://github.com/UniversalDependencies/{corpus['treebank']}",
                "version": "r2.19",
                "author": found[1].decode().strip() if found else "UD contributors",
                "license": part["license"],
                "part": "corpus",
                "tables": part["files"],
                "rows": part["rows"],
            }
        )
    for group in entry["dictionaries"]:
        part = parts.get(("external", group["id"])) or parts.get(
            ("training dictionary", group["id"])
        )
        if part is None:
            continue
        listed += [
            {
                "name": member["id"],
                "url": member["home"],
                "version": member["revision"],
                "author": member["attribution"],
                "license": group["license"],
                "part": part["part"],
                "tables": part["files"],
            }
            | (
                {"merged_as": group["id"], "merged_rows": part["rows"]}
                if len(group["members"]) > 1
                else {"rows": part["rows"]}
            )
            for member in group["members"]
        ]
    return listed


def notice(entry: dict, conditions: str | None) -> bytes:
    """Attribution for the treebank base and every dictionary, one section per part."""
    lines = [
        f"udlex dictionary for {entry['language']} ({entry['iso639_3']}).",
        f"Package license: {entry['license']}. Each part below keeps its own license.",
        "",
    ]
    corpus = entry["corpus"]
    members = {m["id"]: (s, m) for s in entry["dictionaries"] for m in s["members"]}
    for part in entry["tables"]:
        lines.append(f"== {part['part']}: {', '.join(part['files']) or '(none)'} ==")
        lines.append(
            f"License: {part['license']} {license_url(part['license'])}".rstrip()
        )
        if part["part"] == "corpus":
            lines += [
                (
                    f"Learned from Universal Dependencies 2.19, {corpus['treebank']}"
                    f" (README: {corpus['readme_license']};"
                    f" LICENSE file: {corpus['license_file_expression'] or 'none'})."
                ),
                f"https://github.com/UniversalDependencies/{corpus['treebank']}/tree/r2.19",
                "README and LICENSE: TREEBANK-README, TREEBANK-LICENSE.",
            ]
            if conditions:
                lines.append(f"Special conditions: {conditions}")
        elif part["part"] == "shared":
            lines.append(
                "The string pool holds the lemmas of every part; meta.json holds"
                " table counts and the selection evidence. Combined licenses: "
                + ", ".join(part["licenses_combined"])
                + "."
            )
        else:
            if part.get("rows") is not None:
                lines.append(f"{part['rows']} external rows come from this source.")
            if part.get("changed_lemmas"):
                why = "the script guard kept the form"
                if entry["options"]:
                    why = "respelled with edits learned from the treebank, or " + why
                lines.append(
                    f"{part['changed_lemmas']} of them answer differently: {why}."
                )
            source = next(s for s in entry["dictionaries"] if s["id"] == part["source"])
            for member in source["members"]:
                _, info = members[member["id"]]
                lines += [
                    f"{info['id']}: {info['attribution']}.",
                    f"Revision: {info['revision']}. Home: {info['home']}",
                ]
        lines.append("")
    lines.append(
        "Tables and suffix rules were learned with udlex from the inputs above;"
        " SOURCES.json lists digests, the part licenses and the selection evidence."
    )
    return ("\n".join(lines) + "\n").encode()


def readme(entry: dict, conditions: str | None) -> str:
    """Package description: licenses of the parts and the treebank's own terms."""
    lines = [
        f"# udlex dictionary for {entry['language']} ({entry['iso639_3']})",
        "",
        f'Install next to `udlex` and call `udlex.load("{entry["iso639_3"]}")`.',
        "",
        f"Package license: {entry['license']}. Each part keeps its own license:",
        "",
        "| Part | Files | Source | License |",
        "| --- | --- | --- | --- |",
    ]
    lines += [
        f"| {p['part']} | {', '.join(p['files'])} | {p['source']} | {p['license']} |"
        for p in entry["tables"]
    ]
    if conditions:
        lines += ["", f"**Treebank terms.** {conditions}"]
    corpus = entry["corpus"]
    if corpus:
        lines += [
            "",
            (
                f"Base treebank: UD 2.19 {corpus['treebank']}. Evaluation:"
                f" {entry['evaluation']}. Attribution of every input is in NOTICE."
            ),
        ]
    return "\n".join(lines) + "\n"


def write_wheel(
    out: Path,
    entry: dict,
    runtime: dict[str, bytes],
    docs: dict[str, bytes],
    description: str,
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
    terms = (
        "; treebank terms apply, see description" if entry["special_conditions"] else ""
    )
    data[f"{info}/METADATA"] = (
        f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n"
        f"Summary: {entry['language']} dictionary for udlex{terms}\n"
        f"Requires-Python: >=3.11\nRequires-Dist: udlex (>=1.0.0,<2)\n"
        f"License-Expression: {entry['license']}\n"
        + "".join(f"License-File: {f}\n" for f in sorted(docs))
        + "Description-Content-Type: text/markdown\n\n"
        + description
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
    portable[f"bundle/{iso}/README.md"] = description.encode()
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
            # The treebank's own license; review cases count as non-commercial.
            "part_license": lic["license"]
            if lic["requires_license_review"]
            else lic["readme_spdx"],
        }
        docs["TREEBANK-README"] = (bank / lic["readme"]).read_bytes()
        if lic["license_file"]:
            docs["TREEBANK-LICENSE"] = (bank / lic["license_file"]).read_bytes()
    external = decision["order"] if corpus else decision["order"][1:]
    tables = external_provenance(folder / "lexicon" / iso, folder / "tables", external)
    shipped = sum(v["rows"] for v in tables["sources"].values())
    if tables["unattributed"] or shipped != decision["outcome"]["entries"].get(
        "external", 0
    ):
        msg = f"{iso}: external rows not attributable to one dictionary: {tables}"
        raise ValueError(msg)
    # Only parts that ship data license the package; a dropped external table does not.
    rows = tables["sources"]
    components: list[str] = [corpus["license"]] if corpus else [used[0]["license"]]
    components += [s["license"] for s in used if rows.get(s["id"], {}).get("rows")]
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
                        {k: f[k] for k in ("url", "sha256") if k in f}
                        for f in sources[m]["files"]
                        if iso in f.get("languages", [iso])
                    ],
                }
                | (
                    {"pairs_mirror": sources[m]["mirror"]}
                    if sources[m].get("mirror")
                    else {}
                )
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
        "dictionaries": source_entries,
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
    entry["tables"] = table_parts(entry, runtime, tables)
    entry["part_licenses"] = sorted({p["license"] for p in entry["tables"]})
    entry["sources"] = source_list(entry, docs.get("TREEBANK-README", b""))
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
    entry |= write_wheel(
        out / "packages", entry, runtime, docs, readme(entry, conditions)
    )
    return entry


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--commit", help="build commit, when the checkout has no .git")
    args = parser.parse_args()
    commit = (
        args.commit
        or subprocess.run(  # noqa: S603
            ["git", "-C", str(HERE), "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    )
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        msg = f"build commit must be a full git hash, got {commit!r}"
        raise ValueError(msg)
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
