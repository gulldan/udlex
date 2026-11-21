"""Write the numbers and tables of the system paper from the committed evidence.

usage: python -I make_tables.py

Reads paper/naacl/evidence (speed receipts and the dictionary release) and
paper/emnlp/evidence/ud219/census (other lemmatizers on the same test words), and
writes paper/naacl/generated/numbers.tex. Every number in the paper's text and tables comes
from here; the comparison rules are stated next to the code that applies them.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
EVIDENCE = PAPER / "evidence"
CENSUS = PAPER.parent / "emnlp" / "evidence" / "ud219" / "census"
OUT = PAPER / "generated"

# Timed systems, as `bench.py` names them, and how the paper names them.
TIMED = {
    "simplemma": "simplemma",
    "lemmagen3": "LemmaGen3",
    "udpipe1": "UDPipe 1",
    "spacy-sm": "spaCy sm",
    "spacy-lg": "spaCy lg",
    "corenlp": "CoreNLP",
    "stanza": "Stanza",
    "trankit": "Trankit",
    "spacy-trf": "spaCy trf",
}
NEURAL = {"stanza", "trankit", "spacy-trf"}
# Scored on the census test words but not timed (UDPipe 2 runs as a service).
SCORED = {**TIMED, "udpipe2": "UDPipe 2"}
# Rows of the split-drift report name a model family; spaCy's sm, lg and trf
# pipelines were trained on the same UD release, so one row covers all three.
DRIFT_FAMILY = {"spacy-sm": "spacy", "spacy-lg": "spacy", "spacy-trf": "spacy"}
SIXTEEN = ("proc16", "thr16")


def read(path: Path) -> dict:
    """Return a JSON document."""
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Pair:
    """Accuracy of udlex and one other system on the same test words of one language."""

    iso: str
    ours: float
    theirs: float


def release_scores() -> dict[str, dict]:
    """Return the release evaluation rows by ISO 639-3 code."""
    rows = read(EVIDENCE / "release" / "evaluation.json")["languages"]
    return {row["iso639_3"]: row for row in rows}


def leaked() -> set[tuple[str, str]]:
    """Return (engine family, language) pairs whose model was trained on our test.

    The drift report lists languages where more than 1 % of the UD 2.19 test sentences
    occur in the train or dev data of the UD release the model was trained on.
    """
    with (CENSUS / "drift-report.tsv").open(encoding="utf-8") as fh:
        return {
            (row["engine"], row["iso"]) for row in csv.DictReader(fh, delimiter="\t")
        }


def stanza_excluded() -> tuple[set[str], set[str]]:
    """Return Stanza languages without a lemma model, and those whose model saw our test.

    Without a lemma model Stanza copies the form. Its lemma dictionary knowing at least
    half of the test word types that UD 2.19 train lacks means it was trained on our
    test (the rule of compare_engines.py, which wrote stanza-coverage.json).
    """
    coverage = read(CENSUS / "stanza-coverage.json")
    copy = {iso for iso, row in coverage.items() if row.get("lemma_model") is None}
    seen = {
        iso
        for iso, row in coverage.items()
        if row.get("lemma_model") and 2 * row["covered"] >= row["unseen_types"]
    }
    return copy, seen


def pairs(engine: str, scores: dict[str, dict]) -> tuple[list[Pair], dict[str, int]]:
    """Return per-language accuracies of udlex and `engine` on identical test words.

    A language counts only when the engine scored it, the release scored the same
    official test split (equal token totals; six census languages used another
    anchor treebank), the engine's model did not see that test, and, for Stanza,
    the package has a lemmatizer (see `stanza_excluded`).
    """
    cells = read(CENSUS / f"{engine}.json")["cells"]
    family = DRIFT_FAMILY.get(engine, engine)
    seen = {iso for fam, iso in leaked() if fam == family}
    copy: set[str] = set()
    if engine == "stanza":
        copy, covered = stanza_excluded()
        seen |= covered
    out, dropped = [], {"other test": 0, "saw test": 0, "no lemmatizer": 0}
    for iso, cell in sorted(cells.items()):
        if cell["status"] != "scored":
            continue
        row = scores.get(iso)
        if row is None or row["mode"] != "official" or row["tokens"] != cell["tokens"]:
            dropped["other test"] += 1
        elif iso in seen:
            dropped["saw test"] += 1
        elif iso in copy:
            dropped["no lemmatizer"] += 1
        else:
            out.append(
                Pair(
                    iso,
                    100 * row["new"] / row["tokens"],
                    100 * cell["correct"] / cell["tokens"],
                )
            )
    return out, dropped


def geomean(values: list[float]) -> float:
    """Return the geometric mean of positive numbers."""
    return math.exp(statistics.fmean(math.log(v) for v in values))


def speed() -> dict[str, dict]:
    """Return the speed summary keyed by `iso/system/mode`; every cell must be complete."""
    summary = read(EVIDENCE / "speed" / "summary.json")
    broken = [
        key
        for key, row in summary.items()
        if row.get("failed") or row.get("mismatch") is None
    ]
    if broken:
        msg = f"incomplete speed cells: {broken[:5]}"
        raise SystemExit(msg)
    return summary


def best16(summary: dict[str, dict], iso: str, system: str) -> float | None:
    """Return a system's faster 16-core rate (16 processes or 16 threads), if timed."""
    rates = [
        summary[k]["wps"] for m in SIXTEEN if (k := f"{iso}/{system}/{m}") in summary
    ]
    return max(rates) if rates else None


def rate(value: float) -> str:
    """Format words per second compactly: 54.4M, 731k, 93."""
    if value >= 1e6:
        return f"{value / 1e6:.1f}M" if value < 1e8 else f"{value / 1e6:.0f}M"
    if value >= 1e4:
        return f"{value / 1e3:.0f}k"
    if value >= 1e3:
        return f"{value / 1e3:.1f}k"
    return f"{value:.0f}"


def ratio(value: float) -> str:
    """Format a speed-up factor."""
    return f"{value:,.0f}" if value >= 100 else f"{value:.1f}"


DIGITS = dict(
    zip(
        "0123456789",
        [
            "Zero",
            "One",
            "Two",
            "Three",
            "Four",
            "Five",
            "Six",
            "Seven",
            "Eight",
            "Nine",
        ],
        strict=True,
    )
)


def tex_key(name: str) -> str:
    """Turn a system name into letters only, as LaTeX macro names require."""
    return "".join(DIGITS.get(c, c) for c in name if c.isalnum())


def macro(name: str, value: str) -> str:
    """Return one LaTeX macro definition."""
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def accuracy_rows(scores: dict[str, dict]) -> tuple[list[str], list[str]]:
    """Return the accuracy table rows and the macros summarizing them."""
    rows, macros = [], []
    for engine, label in SCORED.items():
        got, _ = pairs(engine, scores)
        ours = statistics.fmean(p.ours for p in got)
        theirs = statistics.fmean(p.theirs for p in got)
        wins = sum(p.ours > p.theirs for p in got)
        losses = sum(p.ours < p.theirs for p in got)
        rows.append(
            f"{label} & {len(got)} & {theirs:.2f} & {ours:.2f} & {ours - theirs:+.2f} & "
            f"{wins} / {losses} \\\\\n"
        )
        key = tex_key(engine)
        macros += [
            macro(f"acc{key}N", str(len(got))),
            macro(f"acc{key}Theirs", f"{theirs:.2f}"),
            macro(f"acc{key}Ours", f"{ours:.2f}"),
        ]
    return rows, macros


def speed_rows(summary: dict[str, dict]) -> tuple[list[str], list[str]]:
    """Return the speed table rows (one per other system) and their macros.

    Each row compares on the languages that system was timed on; speed-ups are
    geometric means of per-language ratios, so no language set is averaged with
    another.
    """
    rows, macros = [], []
    for system, label in TIMED.items():
        langs = sorted({k.split("/")[0] for k in summary if k.split("/")[1] == system})
        one = [summary[f"{i}/{system}/1c"]["wps"] for i in langs]
        ours = [summary[f"{i}/udlex-rs/1c"]["wps"] for i in langs]
        py = [summary[f"{i}/udlex-py/1c"]["wps"] for i in langs]
        up_rs = geomean([o / t for o, t in zip(ours, one, strict=True)])
        up_py = geomean([o / t for o, t in zip(py, one, strict=True)])
        key = tex_key(system)
        if system in NEURAL:
            gpu = [summary[f"{i}/{system}/gpu"]["wps"] for i in langs]
            wide = f"{rate(statistics.median(gpu))} (GPU)"
            up_wide = geomean(
                [
                    summary[f"{i}/udlex-rs/thr16"]["wps"] / g
                    for i, g in zip(langs, gpu, strict=True)
                ]
            )
            macros += [
                macro(f"speed{key}Gpu", rate(statistics.median(gpu))),
                macro(
                    f"speed{key}GpuOneUp",
                    ratio(geomean([o / g for o, g in zip(ours, gpu, strict=True)])),
                ),
            ]
        else:
            theirs16 = [best16(summary, i, system) or math.nan for i in langs]
            wide = rate(statistics.median(theirs16))
            up_wide = geomean(
                [
                    summary[f"{i}/udlex-rs/thr16"]["wps"] / t
                    for i, t in zip(langs, theirs16, strict=True)
                ]
            )
        rss = statistics.median(
            summary[f"{i}/{system}/1c"]["peak_rss_mb"] for i in langs
        )
        rss_ours = statistics.median(
            summary[f"{i}/udlex-rs/1c"]["peak_rss_mb"] for i in langs
        )
        rows.append(
            f"{label} & {len(langs)} & {rate(statistics.median(one))} & "
            f"{ratio(up_rs)} & {ratio(up_py)} & {wide} & {ratio(up_wide)} & "
            f"{rss:,.0f} / {rss_ours:,.0f} \\\\\n"
        )
        macros += [
            macro(f"speed{key}N", str(len(langs))),
            macro(f"speed{key}Up", ratio(up_rs)),
        ]
    return rows, macros


def udlex_macros(summary: dict[str, dict]) -> list[str]:
    """Return macros for udlex's own rates over every timed language."""
    out = []
    langs = sorted({k.split("/")[0] for k in summary if "/udlex-rs/" in k})
    for system, mode, name in [
        ("udlex-rs", "1c", "RsOne"),
        ("udlex-rs", "thr16", "RsSixteen"),
        ("udlex-py", "1c", "PyOne"),
    ]:
        values = [summary[f"{i}/{system}/{mode}"]["wps"] for i in langs]
        out += [
            macro(f"speed{name}Median", rate(statistics.median(values))),
            macro(f"speed{name}Min", rate(min(values))),
            macro(f"speed{name}Max", rate(max(values))),
        ]
    py16 = [best16(summary, i, "udlex-py") or math.nan for i in langs]
    out.append(macro("speedPySixteenMedian", rate(statistics.median(py16))))
    load = [summary[f"{i}/udlex-rs/1c"]["ttfa_s"] * 1e3 for i in langs]
    out.append(macro("loadRsMedianMs", f"{statistics.median(load):.1f}"))
    out.append(macro("loadRsMaxMs", f"{max(load):.0f}"))
    checked = sum(
        summary[f"{i}/{s}/{m}"]["checked"]
        for i in langs
        for s, m in [
            ("udlex-rs", "1c"),
            ("udlex-rs", "thr16"),
            ("udlex-py", "1c"),
            ("udlex-py", "thr16"),
            ("udlex-py", "proc16"),
        ]
    )
    mism = sum(summary[k]["mismatch"] for k in summary if "/udlex-" in k)
    out += [
        macro("speedLangs", str(len(langs))),
        macro("speedChecked", f"{checked:,}"),
        macro("speedMismatch", str(mism)),
    ]
    rivals = {
        i: [
            summary[k]["wps"]
            for k in summary
            if k.startswith(f"{i}/") and k.endswith("/1c") and "/udlex-" not in k
        ]
        for i in langs
    }
    timed = [i for i in langs if rivals[i]]
    slower = [i for i in timed if summary[f"{i}/udlex-rs/1c"]["wps"] <= max(rivals[i])]
    out += [
        macro("speedRivalLangs", str(len(timed))),
        macro("speedRsFastestLangs", str(len(timed) - len(slower))),
        macro("speedRsSlowerLangs", ", ".join(slower) or "none"),
    ]
    return out


def release_macros(scores: dict[str, dict]) -> list[str]:
    """Return macros for the dictionary release and its own evaluation."""
    manifest = json.loads(
        gzip.decompress((EVIDENCE / "release" / "DICTIONARIES.json.gz").read_bytes())
    )
    dicts = manifest["dictionaries"]
    sizes = sorted(d["archive"]["size"] for d in dicts)
    summary = read(EVIDENCE / "release" / "evaluation.json")["summary"]
    off, two = summary["official"], summary["two-fold"]
    noncommercial = sum(d["noncommercial"] for d in dicts)
    sources = {s["id"] for d in dicts for s in d["dictionaries"]}
    national = {
        s["id"] for d in dicts for s in d["dictionaries"] if s["family"] == "national"
    }
    modes = [d["corpus"]["mode"] if d.get("corpus") else "none" for d in dicts]
    return [
        macro("relSources", str(len(sources))),
        macro("relNational", str(len(national))),
        macro("relNoCorpus", str(modes.count("none"))),
        macro("relLangs", str(len(dicts))),
        macro("relNoncommercial", str(noncommercial)),
        macro("relArchiveGB", f"{sum(sizes) / 1e9:.2f}"),
        macro("relArchiveMedianKB", f"{statistics.median(sizes) / 1e3:.0f}"),
        macro("relArchiveMaxMB", f"{sizes[-1] / 1e6:.0f}"),
        macro("relScored", str(off["languages"] + two["languages"])),
        macro("accOffN", str(off["languages"])),
        macro("accOffMacro", f"{off['macro']['new']:.2f}"),
        macro("accOffCopy", f"{off['macro']['copy']:.2f}"),
        macro("accOffPooled", f"{off['pooled']['new']:.2f}"),
        macro("accOffTokens", f"{off['tokens']:,}"),
        macro("accTwoN", str(two["languages"])),
        macro("accTwoMacro", f"{two['macro']['new']:.2f}"),
        macro("accTwoCopy", f"{two['macro']['copy']:.2f}"),
        macro(
            "accScoredOfficial",
            str(sum(r["mode"] == "official" for r in scores.values())),
        ),
    ]


def load_macros() -> list[str]:
    """Return the median foreign CPU load (cores) of new and of reused timing cells."""
    new, reused = [], []
    with gzip.open(EVIDENCE / "speed" / "cells.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            cell = json.loads(line)
            if cell["status"] == "ok":
                (reused if "reused_from" in cell else new).append(cell["foreign_cores"])
    return [
        macro("cellsNew", f"{len(new):,}"),
        macro("cellsReused", f"{len(reused):,}"),
        macro("foreignNew", f"{statistics.median(new):.2f}"),
        macro("foreignReused", f"{statistics.median(reused):.2f}"),
    ]


def main() -> None:
    """Write generated/numbers.tex: every number and table body of the paper."""
    OUT.mkdir(exist_ok=True)
    scores = release_scores()
    summary = speed()
    acc_rows, acc_macros = accuracy_rows(scores)
    speed_table, speed_macros = speed_rows(summary)
    numbers = (
        release_macros(scores)
        + udlex_macros(summary)
        + load_macros()
        + acc_macros
        + speed_macros
    )
    # Table bodies are macros: \input inside a tabular breaks \bottomrule.
    numbers += [
        macro("accuracyRows", "".join(acc_rows)),
        macro("speedRows", "".join(speed_table)),
    ]
    header = "% Generated by scripts/make_tables.py; do not edit.\n"
    (OUT / "numbers.tex").write_text(header + "".join(numbers), encoding="utf-8")


if __name__ == "__main__":
    main()
