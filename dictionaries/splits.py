"""Pick one anchor treebank per language and write leak-free JSONL splits.

A language with a train and a test split keeps the official partition; dev and
test lose every sentence, document or text already seen in an earlier split.
A language with gold lemmas but no usable train split is pooled and cut into
two folds of whole documents for evaluation, and its package learns from the
whole pool. Languages without gold lemmas are listed for dictionary-only builds.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import tarfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from common import (
    LANGUAGES,
    POLICY,
    SOURCES,
    STRICTNESS,
    license_class,
    read_json,
    sha256,
    write_json,
    write_jsonl,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

SPLITS = ("train", "dev", "test")
# Treebank-specific placeholders for a missing lemma, as the engine masks them.
MARKERS = {
    "de_hdt": {"unknown", "-", "NULL"},
    "pcm_nsc": {"X"},
    "qaf_arabizi": {"X"},
    "say_autogramm": {"X"},
    "cu_proiel": {"?"},
    "sk_snk": {"?"},
    "myv_jr": {"-"},
    "cop_scriptorium": {"unknown"},
}
README_LICENSES = {
    "CC BY-SA 4.0": "CC-BY-SA-4.0",
    "CC BY-SA 3.0": "CC-BY-SA-3.0",
    "CC BY 4.0": "CC-BY-4.0",
    "CC BY-NC-SA 4.0": "CC-BY-NC-SA-4.0",
    "CC BY-NC-SA 3.0": "CC-BY-NC-SA-3.0",
    "CC BY-NC-SA 2.5": "CC-BY-NC-SA-2.5",
    "CC BY-NC 4.0": "CC-BY-NC-4.0",
    "CC BY-NC-ND 4.0": "CC-BY-NC-ND-4.0",
    "CC0 1.0": "CC0-1.0",
    "PD": "LicenseRef-Public-Domain",
    "LGPL-LR": "LicenseRef-LGPL-LR",
    "GNU GPL 3.0": "GPL-3.0-only",
    "C-UDA 1.0": "LicenseRef-C-UDA-1.0",
}


def unpack(archive: Path, root: Path) -> Path:
    """Extract CoNLL-U, README and LICENSE files of the pinned UD archive."""
    pinned = read_json(SOURCES)["ud"]["sha256"]
    done = root / ".unpacked"
    if done.is_file() and done.read_text().strip() == pinned:
        return root / "ud-treebanks-v2.19"
    if sha256(archive) != pinned:
        msg = f"{archive}: not the pinned UD 2.19 archive"
        raise ValueError(msg)
    with tarfile.open(archive, "r:gz") as packed:
        members = [
            m
            for m in packed.getmembers()
            if m.isfile()
            and re.fullmatch(r"ud-treebanks-v2\.19/UD_[^/]+/[^/]+", m.name)
            and (
                m.name.endswith(".conllu")
                or Path(m.name).name.startswith(("README", "LICENSE"))
            )
        ]
        packed.extractall(root, members=members, filter="data")
    done.write_text(pinned + "\n")
    return root / "ud-treebanks-v2.19"


def sentences(path: Path) -> Iterator[tuple[str | None, str | None, list[list[str]]]]:
    """Yield ``(document, sentence id, rows)`` with syntactic words only."""
    doc = sid = None
    rows: list[list[str]] = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                if rows:
                    yield doc, sid, rows
                sid, rows = None, []
            elif line.startswith("# sent_id = "):
                sid = line.split(" = ", 1)[1].strip()
            elif line.startswith("#"):
                doc = document(line, number) or doc
            else:
                row = line.rstrip("\r\n").split("\t")
                if len(row) != 10:
                    msg = f"{path}:{number}: malformed CoNLL-U row"
                    raise ValueError(msg)
                if row[0].isdecimal():
                    rows.append(row)
    if rows:
        yield doc, sid, rows


def document(line: str, number: int) -> str | None:
    """Document identity a comment line opens, if any."""
    if line.startswith("# newdoc id = "):
        return line.split(" = ", 1)[1].strip()
    return f"@line-{number}" if line.rstrip() == "# newdoc" else None


def fingerprint(forms: list[str]) -> bytes:
    """Text identity of a sentence, blind to spacing and case."""
    text = "".join("".join(t.split()) for t in forms).casefold()
    return hashlib.sha256(unicodedata.normalize("NFC", text).encode()).digest()


def valid(config: str, form: str, gold: str) -> bool:
    """Whether a gold lemma is an annotation rather than a placeholder."""
    return bool(
        form
        and gold
        and form != "_"
        and gold != "_"
        and (form == gold or gold not in MARKERS.get(config, ()))
    )


def item(config: str, rows: list[list[str]], block: str) -> dict:
    """One JSONL sentence: tokens, masked gold lemmas, UPOS, grouping block."""
    return {
        "tokens": [r[1] for r in rows],
        "lemmas": [r[2] if valid(config, r[1], r[2]) else None for r in rows],
        "upos": [r[3] if r[3] != "_" else None for r in rows],
        "block": block,
    }


def scored(sentence: dict) -> int:
    """Number of tokens with a gold lemma."""
    return sum(y is not None for y in sentence["lemmas"])


def treebank_license(bank: Path, iso: str) -> dict:
    """README and LICENSE declarations; an NC side or the owner's list makes NC."""
    readme = next(iter(sorted(bank.glob("README*"))))
    text = readme.read_text(encoding="utf-8")
    declared = re.search(r"^License:\s*([^\r\n]+)", text, re.MULTILINE)
    stated = declared[1].strip() if declared else None
    spdx = README_LICENSES.get(stated or "", "LicenseRef-Unknown")
    license_file = next((p for p in sorted(bank.glob("LICENSE*")) if p.is_file()), None)
    named = None
    if license_file:
        found = re.search(
            r"creativecommons\.org/licenses/([a-z-]+)/(\d\.\d)",
            license_file.read_text(encoding="utf-8", errors="replace"),
            re.IGNORECASE,
        )
        named = f"CC-{found[1].upper()}-{found[2]}" if found else None
    listed = iso in read_json(POLICY)["noncommercial_review"]
    classes = {license_class(spdx)} | ({license_class(named)} if named else set())
    noncommercial = listed or "nc" in classes
    effective = spdx
    if noncommercial:
        effective = "CC-BY-NC-SA-4.0"
    elif named and named in STRICTNESS and spdx in STRICTNESS:
        effective = max(spdx, named, key=STRICTNESS.index)
    return {
        "readme_license": stated,
        "readme_spdx": spdx,
        "license_file_expression": named,
        "license": effective,
        "license_class": license_class(spdx),
        "requires_license_review": listed or len(classes) > 1,
        "noncommercial": noncommercial,
        "readme": readme.name,
        "license_file": license_file.name if license_file else None,
    }


def write_official(config: str, bank: Path, out: Path) -> dict:
    """Write train/dev/test, removing later sentences seen in earlier splits."""
    past_docs: set[str] = set()
    past_ids: set[str] = set()
    seen: set[bytes] = set()
    result = {}
    for split in SPLITS:
        path = bank / f"{config}-ud-{split}.conllu"
        docs: set[str] = set()
        ids: set[str] = set()
        counts: Counter[str] = Counter()
        kept = []
        for doc, sid, rows in sentences(path) if path.is_file() else ():
            mark = fingerprint([r[1] for r in rows])
            named_doc = doc if doc and not doc.startswith("@line-") else None
            if (
                (named_doc and named_doc in past_docs)
                or (sid and sid in past_ids)
                or mark in seen
            ):
                counts["dropped_shared"] += 1
                continue
            sentence = item(config, rows, f"d:{doc}" if doc else f"s:{len(kept) + 1}")
            if not scored(sentence):
                continue
            seen.add(mark)
            if named_doc:
                docs.add(named_doc)
            if sid:
                ids.add(sid)
            counts["sentences"] += 1
            counts["tokens"] += scored(sentence)
            kept.append(sentence)
        past_docs |= docs
        past_ids |= ids
        write_jsonl(out / f"{split}.jsonl", kept)
        result[split] = {
            "file": path.name if path.is_file() else None,
            "conllu_sha256": sha256(path) if path.is_file() else None,
            "jsonl_sha256": sha256(out / f"{split}.jsonl"),
            **counts,
        }
    return result


def write_twofold(config: str, bank: Path, out: Path) -> dict:
    """Pool unique sentences, group by document, deal groups into two folds."""
    unique: set[bytes] = set()
    groups: dict[str, list[dict]] = defaultdict(list)
    for split in SPLITS:
        path = bank / f"{config}-ud-{split}.conllu"
        if not path.is_file():
            continue
        for n, (doc, _, rows) in enumerate(sentences(path), 1):
            mark = fingerprint([r[1] for r in rows])
            ident = (
                f"d:{doc}" if doc and not doc.startswith("@line-") else f"s:{split}:{n}"
            )
            sentence = item(config, rows, ident)
            if mark in unique or not scored(sentence):
                continue
            unique.add(mark)
            groups[ident].append(sentence)
    if len(groups) == 1:
        (only,) = groups.values()
        groups = {f"s:{n}": [dict(s, block=f"s:{n}")] for n, s in enumerate(only, 1)}
        unit = "sentence within one original group"
    elif any(k.startswith("d:") for k in groups):
        unit = "documents and ungrouped sentences"
    else:
        unit = "sentences"
    weight = {k: sum(scored(s) for s in v) for k, v in groups.items()}
    folds: list[list[dict]] = [[], []]
    loads = [0, 0]
    for key in sorted(
        groups, key=lambda k: (-weight[k], hashlib.sha256(k.encode()).digest())
    ):
        side = int(loads[1] < loads[0])
        loads[side] += weight[key]
        folds[side].extend(groups[key])
    write_jsonl(out / "all.jsonl", folds[0] + folds[1])
    for k in (0, 1):
        write_jsonl(out / f"fold{k}.jsonl", folds[k])
    return {
        "unit": unit,
        "fold_tokens": loads,
        "all_sha256": sha256(out / "all.jsonl"),
        "fold_sha256": [sha256(out / f"fold{k}.jsonl") for k in (0, 1)],
    }


def lemma_tokens(config: str, bank: Path) -> dict[str, int]:
    """Gold-lemma tokens per split, before any de-duplication."""
    found = {}
    for split in SPLITS:
        path = bank / f"{config}-ud-{split}.conllu"
        if path.is_file():
            found[split] = sum(
                valid(config, r[1], r[2])
                for _, _, rows in sentences(path)
                for r in rows
            )
    return found


def plan(iso: str, spec: dict, banks: list[Path], out: Path) -> dict:
    """Choose the anchor among treebanks a base may come from; write its splits."""
    excluded = set(read_json(POLICY)["excluded_license_classes"])
    licenses = {b: treebank_license(b, iso) for b in banks}
    configs = {
        b: next(iter(sorted(b.glob("*-ud-*.conllu")))).name.split("-ud-")[0]
        for b in banks
    }
    gold = {b: lemma_tokens(configs[b], b) for b in banks}
    preferred = spec.get("preferred")
    row = {
        "iso639_3": iso,
        "language": spec["name"],
        "ud_code": spec["ud"],
        "group": spec["group"],
    }
    row["treebanks"] = {
        b.name: {
            "config": configs[b],
            "lemma_tokens": gold[b],
            "license": licenses[b]["license"],
        }
        for b in banks
    }
    usable = [b for b in banks if licenses[b]["license_class"] not in excluded]

    def train_bytes(bank: Path) -> int:
        path = bank / f"{configs[bank]}-ud-train.conllu"
        return path.stat().st_size if path.is_file() else 0

    trained = [b for b in usable if gold[b].get("train") and gold[b].get("test")]
    for bank in sorted(
        trained, key=lambda b: (b.name != preferred, -train_bytes(b), b.name)
    ):
        splits = write_official(configs[bank], bank, out / iso)
        if splits["train"].get("tokens") and splits["test"].get("tokens"):
            row.update(
                mode="official", treebank=bank.name, config=configs[bank], splits=splits
            )
            break
    else:
        lemmatized = [b for b in usable if sum(gold[b].values())]
        if lemmatized:
            bank = min(
                lemmatized,
                key=lambda b: (b.name != preferred, -sum(gold[b].values()), b.name),
            )
            folds = write_twofold(configs[bank], bank, out / iso)
            row.update(
                mode="twofold", treebank=bank.name, config=configs[bank], folds=folds
            )
        else:
            row.update(mode="no-lemmas", treebank=None, config=None)
    if row["treebank"]:
        row["license"] = next(
            v for b, v in licenses.items() if b.name == row["treebank"]
        )
    return row


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="pinned UD 2.19 tgz")
    parser.add_argument("work", type=Path)
    args = parser.parse_args()
    ud = unpack(args.archive, args.work / "ud")
    languages = read_json(LANGUAGES)
    by_code: dict[str, list[Path]] = defaultdict(list)
    for bank in sorted(ud.glob("UD_*")):
        files = sorted(bank.glob("*-ud-*.conllu"))
        if files:
            by_code[files[0].name.split("_")[0]].append(bank)
    roster = {}
    for iso, spec in sorted(languages.items()):
        roster[iso] = plan(iso, spec, by_code.get(spec["ud"], []), args.work / "splits")
        print(iso, roster[iso]["mode"], roster[iso]["treebank"], flush=True)
    write_json(args.work / "roster.json", roster)


if __name__ == "__main__":
    main()
