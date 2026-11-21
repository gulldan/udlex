"""Per model: census languages whose UD 2.19 test lies in the data the model was trained on.

usage: python -I drift_report.py [THRESHOLD_PERCENT]

Reads split-drift.json, the census results of the engines, Stanza's resources.json and
Trankit's model table; prints one row per language above the threshold.
"""

import json
import os
import sys
from pathlib import Path

COMPARE_DIR = os.environ["UDLEX_COMPARE"]

C = Path(COMPARE_DIR)
STANZA_CODE = {"no_nynorsk": "nn", "no_bokmaal": "nb"}
# Trankit's own names of two UD 2.5 treebanks.
ALIAS = {
    "UD_Norwegian_Nynorsk-Nynorsk": "UD_Norwegian-Nynorsk",
    "UD_Simplified_Chinese-GSDSimp": "UD_Chinese-GSDSimp",
}
# UD sources named in meta.json of the spaCy 3.8.0 pipelines (sm, lg and trf share them);
# None is every treebank of the language. German, English, Croatian, Russian, Ukrainian
# and Chinese pipelines name no UD treebank.
SPACY_UD = {
    "cat": ("2.8", "ca_ancora"),
    "dan": ("2.8", "da_ddt"),
    "ell": ("2.8", "el_gdt"),
    "spa": ("2.8", "es_ancora"),
    "fin": ("2.8", "fi_tdt"),
    "fra": ("2.8", "fr_sequoia"),
    "ita": ("2.8", "it_isdt"),
    "kor": ("2.8", "ko_kaist"),
    "lit": ("2.8", "lt_alksnis"),
    "nld": ("2.8", None),
    "pol": ("2.8", "pl_pdb"),
    "por": ("2.8", "pt_bosque"),
    "ron": ("2.8", "ro_rrt"),
    "swe": ("2.8", "sv_talbanken"),
    "slv": ("2.11", "sl_ssj"),
}


def read(path):
    return json.loads(path.read_text())


def scored(engine):
    path = C / "results-census" / f"{engine}.json"
    if not path.exists():
        return None
    return {i: c for i, c in read(path)["cells"].items() if c["status"] == "scored"}


def stanza_package(row, resources):
    """The package compare_engines.stanza_pipeline picks, and the treebank behind it if one."""
    config = row["config"]
    lang, suffix = config.split("_", 1)
    code = STANZA_CODE.get(config, lang)
    entry = resources.get(code)
    while isinstance(entry, dict) and "alias" in entry:
        entry = resources.get(entry["alias"])
    if not entry:
        return None, None
    if suffix in entry.get("packages", {}):
        return f"{code}/{suffix}", config
    lemma = entry.get("default_processors", {}).get("lemma")
    if not lemma:
        return f"{code}/default (no lemmatizer)", None
    return f"{code}/default={lemma}", f"{lang}_{lemma.split('_')[0]}"


def main(threshold):
    drift = read(C / "split-drift.json")["languages"]
    resources = read(C / "stanza/resources.json")
    trankit = read(C / "mine/trankit-treebanks.json")
    plans = []  # (engine, release, iso, model, treebank directory or None for the whole language)
    for engine, release in (("udpipe2", "2.17"), ("udpipe1", "2.5")):
        for iso, c in (scored(engine) or {}).items():
            same = drift[iso][release]["same"] or "?"  # reported as absent
            plans.append((engine, release, iso, c["model"], same))
    for iso, c in (scored("trankit") or {}).items():
        tb = trankit[c["model"].split("/")[0]]
        plans.append(("trankit", "2.5", iso, c["model"], ALIAS.get(tb, tb)))
    for iso, c in (scored("spacy-lg") or {}).items():
        if iso in SPACY_UD:
            release, config = SPACY_UD[iso]
            tbs = drift[iso][release]["treebanks"]
            tb = next((t for t, v in tbs.items() if v["config"] == config), None)
            if config and not tb:
                tb = config  # reported as absent
            plans.append(("spacy", release, iso, c["model"], tb))
    done = scored("stanza")
    for iso, row in drift.items():
        if done is not None and iso not in done:
            continue
        model, config = stanza_package(row, resources)
        if model is None:
            continue
        tbs = row["2.18"]["treebanks"]
        tb = next((t for t, v in tbs.items() if v["config"] == config), None)
        plans.append(("stanza", "2.18", iso, model, tb))
    print(
        "engine\trelease\tiso\ttest\tmodel\ttrained on\ttrain+dev exact %\ttrain+dev loose %"
        "\ttest exact %\ttest loose %"
    )
    seen = {}
    for engine, release, iso, model, tb in plans:
        row = drift[iso]
        part = row[release]["treebanks"].get(tb) if tb else row[release]["language"]
        seen.setdefault(engine, [0, 0, 0])[0] += 1
        if part is None:
            seen[engine][2] += 1
            print(
                f"{engine}\t{release}\t{iso}\t{row['sentences']}\t{model}\t{tb} ABSENT"
            )
            continue
        n = row["sentences"]
        cells = [
            100 * part[p][k] / n
            for p in ("train_dev", "test")
            for k in ("exact", "loose")
        ]
        if max(cells[:2]) > threshold:
            seen[engine][1] += 1
            print(
                f"{engine}\t{release}\t{iso}\t{n}\t{model}\t{tb or 'all treebanks of the language'}\t"
                + "\t".join(f"{x:.2f}" for x in cells)
            )
    for engine, (total, above, absent) in seen.items():
        print(
            f"# {engine}: {total} languages, {above} above {threshold} %, {absent} without the treebank in the release"
        )


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 1.0)
