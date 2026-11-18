"""Turn each pinned source into per-language ``form, lemma, count`` tables.

Only lexical rows survive: one token on each side, no control characters,
NFC. Counts keep how many analyses support a pair, which the later resolution
of one lemma per form needs. Nothing here looks at any treebank.
"""

from __future__ import annotations

import argparse
import bz2
import csv
import gzip
import io
import itertools
import json
import lzma
import re
import shutil
import tarfile
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from common import (
    LANGUAGES,
    POLICY,
    SOURCES,
    jobs,
    lexical,
    nfc,
    read_json,
    sha256,
    write_json,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

# Wiktextract form tags naming a table, template or transliteration, not a form.
SKIP_TAGS = {
    "table-tags",
    "inflection-template",
    "class",
    "romanization",
    "transliteration",
    "canonical",
    "error-unknown-tag",
}


def language_codes() -> dict[str, list[str]]:
    """Wiktionary language code to roster ISO 639-3 codes."""
    codes: dict[str, list[str]] = {}
    for iso, spec in read_json(LANGUAGES).items():
        for code in {iso, spec["ud"]}:
            codes.setdefault(code, []).append(iso)
    for code, isos in read_json(POLICY)["wiktionary_codes"].items():
        codes.setdefault(code, []).extend(i for i in isos if i not in codes[code])
    return codes


def wiktextract(path: Path) -> Iterator[tuple[str, str, str]]:
    """Yield ``(code, form, lemma)`` from inflection tables and form-of senses."""
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            entry = json.loads(line)
            code, word = entry.get("lang_code"), entry.get("word")
            if not code or not word:
                continue
            senses = entry.get("senses") or []
            targets = [
                link["word"]
                for sense in senses
                for link in sense.get("form_of") or ()
                if link.get("word")
            ]
            if targets:
                for lemma in targets:
                    yield code, word, lemma
            # A headword whose senses all point elsewhere is not a lemma itself.
            if not senses or any(not s.get("form_of") for s in senses):
                yield code, word, word
                for form in entry.get("forms") or ():
                    text = form.get("form")
                    if text and not SKIP_TAGS.intersection(form.get("tags") or ()):
                        yield code, text, word


def unimorph(path: Path) -> Iterator[tuple[str, str]]:
    """Yield ``(form, lemma)`` from a UniMorph ``lemma, form, features`` file."""
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) >= 2 and not line.startswith("#"):
                yield fields[1], fields[0]


def cedict(path: Path) -> Iterator[tuple[str, str]]:
    """Traditional to simplified where unambiguous, simplified to itself."""
    traditional: dict[str, str | None] = {}
    simplified = set()
    with gzip.open(path, "rt", encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip() or line.startswith("#"):
                continue
            trad, simp, *_ = line.split()
            simplified.add(simp)
            traditional[trad] = simp if traditional.get(trad, simp) == simp else None
    yield from ((t, s) for t, s in traditional.items() if s is not None)
    yield from ((s, s) for s in simplified)


def pymorphy(path: Path) -> Iterator[tuple[str, str]]:
    """Every OpenCorpora form with its most frequent normal form."""
    from pymorphy3.opencorpora_dict.wrapper import Dictionary  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(path) as wheel:
        for name in wheel.namelist():
            if "/data/" in name and not name.endswith("/"):
                Path(tmp, Path(name).name).write_bytes(wheel.read(name))
        dictionary = Dictionary(tmp)
        for form, group in itertools.groupby(
            dictionary.words.iteritems(), key=lambda p: p[0]
        ):
            labels = Counter(
                dictionary.build_normal_form(idx, slot, form)
                for _, (idx, slot) in group
            )
            best = min(labels, key=lambda y: (-labels[y], y != form, y))
            yield form, best


def members(path: Path, spec: dict) -> Iterator[io.TextIOBase]:
    """Text streams of a plain, compressed or archived table."""
    container = spec.get("container")
    encoding = spec.get("encoding", "utf-8-sig")
    if container in {None, "gz", "xz", "bz2"}:
        opener = {None: open, "gz": gzip.open, "xz": lzma.open, "bz2": bz2.open}[
            container
        ]
        with opener(path, "rt", encoding=encoding) as stream:
            yield stream
    elif container == "zip":
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                if re.fullmatch(spec["member"], name):
                    with archive.open(name) as raw:
                        yield io.TextIOWrapper(raw, encoding=encoding)
    elif container == "tar":
        with tarfile.open(path) as archive:
            for member in archive:
                if member.isfile() and re.fullmatch(spec["member"], member.name):
                    raw = archive.extractfile(member)
                    if raw:
                        yield io.TextIOWrapper(raw, encoding=encoding)
    else:
        msg = f"unknown container {container}"
        raise ValueError(msg)


def table(path: Path, spec: dict, notices: list[str]) -> Iterator[tuple[str, str]]:
    """Rows of a delimited full-form list; column names resolve against a header."""
    subs = [(re.compile(a), b) for a, b in spec.get("lemma_sub", [])]
    skip = re.compile(spec["skip"]) if "skip" in spec else None
    for stream in members(path, spec):
        if spec["delimiter"] == "csv":
            rows: Iterator[list[str]] = csv.reader(stream)
        else:
            rows = (line.rstrip("\r\n").split(spec["delimiter"]) for line in stream)
        header = [next(rows) for _ in range(spec.get("header", 0))]
        form_at, lemma_at = (
            header[0].index(spec[k]) if isinstance(spec[k], str) else spec[k]
            for k in ("form", "lemma")
        )
        block = None
        for fields in rows:
            first = fields[0] if fields else ""
            if "block" in spec and first.startswith(spec["block"][0]):
                block = []
                continue
            if block is not None:
                if first.startswith(spec["block"][1]):
                    notices.append("\n".join(block))
                    block = None
                else:
                    block.append(spec["delimiter"].join(fields))
                continue
            if len(fields) <= max(form_at, lemma_at) or first.startswith(
                spec.get("comment", "\0")
            ):
                continue
            form, lemma = fields[form_at], fields[lemma_at]
            for pattern, repl in subs:
                lemma = pattern.sub(repl, lemma)
            if skip and (skip.search(form) or skip.search(lemma)):
                continue
            yield form, lemma


def sudachi(path: Path) -> Iterator[tuple[str, str]]:
    """SudachiDict raw lexicon: column 13 starts with the dictionary form."""
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name.endswith(".csv"):
                with archive.open(name) as raw:
                    for fields in csv.reader(io.TextIOWrapper(raw, encoding="utf-8")):
                        if len(fields) > 13 and fields[0] != "IndexForm":
                            target = fields[13].split(",")[0]
                            yield (
                                fields[0],
                                target if target not in {"", "*"} else fields[0],
                            )


def xml_entries(path: Path, member: str, tag: str) -> Iterator[ET.Element]:
    """Completed ``tag`` elements of every matching XML member, memory-bounded."""
    opener = zipfile.ZipFile(path) if zipfile.is_zipfile(path) else None
    names = (
        sorted(n for n in opener.namelist() if re.fullmatch(member, n))
        if opener
        else [None]
    )
    for name in names:
        with opener.open(name) if opener and name else path.open("rb") as raw:
            # Inputs are pinned by SHA-256 before any parser sees them.
            for _, element in ET.iterparse(raw, events=("end",)):  # noqa: S314
                if element.tag.rsplit("}", 1)[-1] == tag:
                    yield element
                    element.clear()


def local(element: ET.Element, name: str) -> Iterator[ET.Element]:
    """Descendants by local name, ignoring XML namespaces."""
    return (e for e in element.iter() if e.tag.rsplit("}", 1)[-1] == name)


def saldom(path: Path) -> Iterator[tuple[str, str]]:
    """SALDO morphology: word forms of each lexical entry, without compound parts."""
    for entry in xml_entries(path, "", "LexicalEntry"):
        feats = lambda node: {f.get("att"): f.get("val") for f in local(node, "feat")}  # noqa: E731
        lemma = next((feats(n).get("writtenForm") for n in local(entry, "Lemma")), None)
        for form in local(entry, "WordForm"):
            values = feats(form)
            text = values.get("writtenForm") or ""
            if (
                lemma
                and values.get("msd") not in {"c", "ci", "cm", "sms"}
                and not text.endswith("-")
            ):
                yield text, lemma


def grammardb(path: Path) -> Iterator[tuple[str, str]]:
    """Belarusian GrammarDB: current-orthography variants, stress marks removed."""
    for paradigm in xml_entries(path, r"[^/]*\.xml", "Paradigm"):
        for variant in local(paradigm, "Variant"):
            if "A2008" not in (variant.get("pravapis") or "A2008"):
                continue
            lemma = (variant.get("lemma") or paradigm.get("lemma") or "").replace(
                "+", ""
            )
            for form in local(variant, "Form"):
                if form.get("type") not in {"potential", "nonstandard"} and form.text:
                    yield form.text.replace("+", ""), lemma


def sloleks(path: Path) -> Iterator[tuple[str, str]]:
    """Sloleks 3.0: every orthographic word form of each entry."""
    for entry in xml_entries(path, r".*\.xml", "entry"):
        lemma = next((e.text for e in local(entry, "lemma") if e.text), None)
        for form in local(entry, "form"):
            if lemma and form.text:
                yield form.text, lemma


def tezaurs(path: Path) -> Iterator[tuple[str, str]]:
    """Tēzaurs.lv TEI word forms: the lemma form and every inflected form."""
    for entry in xml_entries(path, r".*\.xml", "entry"):
        forms = list(local(entry, "form"))
        lemma = next(
            (
                o.text
                for f in forms
                if f.get("type") == "lemma"
                for o in local(f, "orth")
                if o.text
            ),
            None,
        )
        for form in forms:
            if lemma and form.get("type") in {"lemma", "inflection"}:
                for orth in local(form, "orth"):
                    if orth.text:
                        yield orth.text, lemma


def eesthetic(path: Path) -> Iterator[tuple[str, str]]:
    """Eesthetic paradigms joined to their lexemes' lemmas."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()

        def rows(suffix: str) -> Iterator[dict[str, str]]:
            name = next(n for n in names if n.endswith(suffix))
            with archive.open(name) as raw:
                yield from csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"))

        lemmas = {r["lexeme_id"]: r["lemma"] for r in rows("estonian_lexemes.csv")}
        for row in rows("estonian_paradigms.csv"):
            if row["lexeme"] in lemmas and row.get("orth_form"):
                yield row["orth_form"], lemmas[row["lexeme"]]


PARSERS: dict[str, Callable[[Path], Iterator[tuple[str, str]]]] = {
    "unimorph": unimorph,
    "cedict": cedict,
    "pymorphy": pymorphy,
    "sudachi": sudachi,
    "saldom": saldom,
    "grammardb": grammardb,
    "sloleks": sloleks,
    "tezaurs": tezaurs,
    "eesthetic": eesthetic,
}


def flush(out: Path, iso: str, pairs: Counter[tuple[str, str]]) -> dict:
    """Write one language table sorted by form and lemma."""
    path = out / f"{iso}.tsv.gz"
    with (
        path.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz,
    ):
        for (form, lemma), count in sorted(pairs.items()):
            gz.write(f"{form}\t{lemma}\t{count}\n".encode())
    return {
        "pairs": len(pairs),
        "forms": len({f for f, _ in pairs}),
        "sha256": sha256(path),
    }


def clean(form: str, lemma: str) -> tuple[str, str] | None:
    """NFC pair, or ``None`` for anything that is not one token on each side."""
    form, lemma = nfc(form.strip()), nfc(lemma.strip())
    return (form, lemma) if lexical(form) and lexical(lemma) else None


def spool_wiktextract(path: Path, out: Path) -> dict:
    """One pass over a Wiktextract dump, spooling pairs per language to disk."""
    codes = language_codes()
    spool = out / ".spool"
    spool.mkdir()
    handles = {}
    try:
        for code, form, lemma in wiktextract(path):
            pair = clean(form, lemma)
            if pair is None:
                continue
            for iso in codes.get(code, ()):
                if iso not in handles:
                    handles[iso] = (spool / iso).open("a", encoding="utf-8")
                handles[iso].write(f"{pair[0]}\t{pair[1]}\n")
    finally:
        for handle in handles.values():
            handle.close()
    languages = {}
    for part in sorted(spool.iterdir()):
        with part.open(encoding="utf-8") as stream:
            counts: Counter[tuple[str, str]] = Counter()
            for line in stream:
                form, lemma = line.rstrip("\n").split("\t")
                counts[form, lemma] += 1
        languages[part.name] = flush(out, part.name, counts)
        part.unlink()
    spool.rmdir()
    return languages


def extract(source: dict, cache: Path, root: Path) -> dict:
    """Extract one source into ``root/<id>/<iso>.tsv.gz``; resumable per source."""
    out = root / source["id"]
    manifest = out / "manifest.json"
    inputs = [f["sha256"] for f in source["files"] if "notice" not in f]
    if manifest.is_file() and read_json(manifest).get("inputs") == inputs:
        return read_json(manifest)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    if source["parser"] == "wiktextract":
        languages = spool_wiktextract(cache / source["files"][0]["sha256"], out)
    else:
        notices: list[str] = []
        tables: dict[str, Counter[tuple[str, str]]] = {}
        for item in (f for f in source["files"] if "notice" not in f):
            path = cache / item["sha256"]
            if source["parser"] == "table":
                reader = table(path, item.get("format") or source["format"], notices)
            else:
                reader = PARSERS[source["parser"]](path)
            rows = [p for x, y in reader if (p := clean(x, y))]
            for iso in item.get("languages") or source["languages"]:
                tables.setdefault(iso, Counter()).update(rows)
        languages = {
            iso: flush(out, iso, pairs) for iso, pairs in sorted(tables.items())
        }
        if notices:
            (out / "NOTICE.txt").write_text(notices[0] + "\n", encoding="utf-8")
    result = {"id": source["id"], "inputs": inputs, "languages": languages}
    write_json(manifest, result)
    print(source["id"], len(languages), "languages", flush=True)
    return result


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    sources = [
        s
        for s in read_json(SOURCES)["sources"]
        if not args.only or s["id"] in args.only
    ]
    # Largest inputs first so the long Wiktionary pass starts immediately.
    sources.sort(key=lambda s: -sum(f.get("size") or 0 for f in s["files"]))
    with ProcessPoolExecutor(max_workers=jobs()) as pool:
        futures = [
            pool.submit(extract, s, args.cache, args.work / "pairs") for s in sources
        ]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
