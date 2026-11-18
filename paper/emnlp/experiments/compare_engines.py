"""Score other lemmatizers on the official test words of the order experiment."""

import argparse
import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from order_summary import gate
from verify_snapshot import read, save

# Adapters of the project benchmark, one per released tool.
ADAPTERS = {
    "simplemma": "simplemma",
    "simplemma-greedy": "simplemma",
    "spacy-sm": "spacy",
    "spacy-lookup": "spacy-lookups-data",
    "lemmagen3": "lemmagen3",
    "pymorphy3": "pymorphy3",
    "kiwi": "kiwipiepy",
    "sudachi": "sudachipy",
    "hazm": "hazm",
    "zeyrek": "zeyrek",
    "uzmorph": "UzMorphAnalyser",
    "nltk-wordnet": "nltk",
}
STANZA_CODE = {"no_nynorsk": "nn", "no_bokmaal": "nb"}
CHUNK = 2000


def sentences(path):
    with path.open() as fh:
        for line in fh:
            row = json.loads(line)
            yield row["tokens"], row["lemmas"]


def stanza_pipeline(lang, config, models):
    """The package trained on the anchor treebank if Stanza ships one, else its default."""
    import stanza

    code = STANZA_CODE.get(config, lang)
    last = None
    for package in (config.split("_", 1)[1], "default"):
        try:
            stanza.download(
                code,
                model_dir=str(models),
                package=package,
                processors="tokenize,pos,lemma",
                logging_level="ERROR",
            )
            nlp = stanza.Pipeline(
                code,
                dir=str(models),
                package=package,
                processors="tokenize,pos,lemma",
                tokenize_pretokenized=True,
                use_gpu=False,
                download_method=None,
                logging_level="ERROR",
            )
        except Exception as error:  # noqa: BLE001 - a missing package is a property of the tool
            last = error
            continue
        return nlp, f"{code}/{package}"
    raise LookupError(str(last)[:200])


def spacy_pipeline(lang):
    """Every installed small pipeline with a lemmatizer, rule-based or trained."""
    import spacy
    from spacy.tokens import Doc

    name = "en_core_web_sm" if lang == "en" else f"{lang}_core_news_sm"
    nlp = spacy.load(name, exclude=["parser", "ner", "senter"])
    if not {"lemmatizer", "trainable_lemmatizer"} & set(nlp.pipe_names):
        raise LookupError(f"{name} runs no lemmatizer")

    def run(batch):
        docs = (Doc(nlp.vocab, words=[t or "_" for t in s]) for s in batch)
        return [[t.lemma_ for t in d] for d in nlp.pipe(docs, batch_size=256)]

    return run, f"{name}-{nlp.meta['version']}"


def cell(job):
    engine, iso, lang, config, test, repo, models, keep = job
    data = list(sentences(test))
    try:
        if engine == "stanza":
            import torch

            torch.set_num_threads(6)
            nlp, model = stanza_pipeline(lang, config, models)

            def run(batch):
                doc = nlp([[t or "_" for t in s] for s in batch])
                return [
                    [w.lemma if w.lemma is not None else w.text for w in s.words]
                    for s in doc.sentences
                ]
        elif engine == "spacy-sm":
            run, model = spacy_pipeline(lang)
        else:
            sys.path.insert(0, str(repo))
            from bench import adapters

            adapters.configure_project_root(repo)
            adapter = adapters.create_adapter(engine)
            adapter.load(lang, config)
            model, run = str(getattr(adapter, "model", None) or engine), adapter.run
    except Exception as error:  # noqa: BLE001 - recorded as an unsupported language
        return (
            engine,
            iso,
            {
                "status": "unsupported",
                "reason": f"{type(error).__name__}: {error}"[:200],
            },
        )
    tokens = correct = 0
    kept = []
    forms = [s for s, _ in data]
    for start in range(0, len(forms), CHUNK):
        batch = forms[start : start + CHUNK]
        answers = run(batch)
        kept.extend(answers)
        for (_, gold), words, lemmas in zip(
            data[start : start + CHUNK], batch, answers, strict=True
        ):
            if len(lemmas) != len(words):
                raise ValueError(f"{engine} changed the tokenization of {config}")
            for y, lemma in zip(gold, lemmas, strict=True):
                if y is not None:
                    tokens += 1
                    correct += lemma == y
    if keep:
        # Every answer, in the order of the test file, for error analysis.
        keep.mkdir(parents=True, exist_ok=True)
        packed = gzip.compress(json.dumps(kept, ensure_ascii=False).encode(), mtime=0)
        (keep / f"{iso}.json.gz").write_bytes(packed)
    return (
        engine,
        iso,
        {"status": "scored", "model": model, "tokens": tokens, "correct": correct},
    )


def run(a):
    inv = read(a.order / "inventory.json")["languages"]
    if a.transfer:
        # Corpora of the transfer test; each is scored with the model of its anchor treebank.
        ledger = read(a.transfer / "ledger.json")["corpora"]
        targets = {
            cfg: (
                inv[row["iso"]]["config"],
                a.transfer / cfg / row["iso"] / "test.jsonl",
            )
            for cfg, row in ledger.items()
            if (a.transfer / cfg / "inventory.json").exists()
        }
    else:
        targets = {
            iso: (row["config"], a.order / iso / "test.jsonl")
            for iso, row in inv.items()
            if row.get("status") == "paired" and len(row["sources"]) >= 2
        }
    for engine in a.engines:
        out = a.output / f"{engine}.json"
        if out.exists():
            continue
        jobs = [
            (
                engine,
                name,
                config.split("_")[0],
                config,
                test,
                a.repo,
                a.models,
                a.predictions / engine if a.predictions else None,
            )
            for name, (config, test) in sorted(targets.items())
        ]
        cells = {}
        with ProcessPoolExecutor(
            a.stanza_jobs if engine == "stanza" else a.jobs
        ) as pool:
            for _, iso, record in pool.map(cell, jobs):
                cells[iso] = record
        try:
            release = version(ADAPTERS.get(engine, engine))
        except PackageNotFoundError:
            release = None
        save(out, {"engine": engine, "version": release, "cells": cells})
        print(
            engine,
            release,
            sum(c["status"] == "scored" for c in cells.values()),
            "languages",
            flush=True,
        )


def screen(a):
    """Does a Stanza lemma dictionary hold test words that the official training split lacks?"""
    import stanza.models.lemma.trainer as trainer
    import torch

    unpack = next(getattr(trainer, n) for n in dir(trainer) if n.startswith("_unpack"))
    report = {}
    for iso, c in sorted(read(a.output / "stanza.json")["cells"].items()):
        if c["status"] != "scored":
            continue
        code, package = c["model"].split("/")
        files = sorted((a.models / code / "lemma").glob(f"{package}*.pt"))
        if not files:
            report[iso] = {"lemma_model": None}  # the pipeline copies the word
            continue
        packed = torch.load(files[0], map_location="cpu", weights_only=False)["dicts"]
        words = set()
        for key, value in (
            unpack(packed) if isinstance(packed, bytes) else packed
        ).items():
            words |= (
                set(value)
                if isinstance(value, dict)
                else {key[0] if isinstance(key, tuple) else key}
            )
        forms = {}
        for split in ("train", "test"):
            forms[split] = {
                f
                for tokens, gold in sentences(a.order / iso / f"{split}.jsonl")
                for f, y in zip(tokens, gold, strict=True)
                if y is not None
            }
        seen = forms["train"] | {f.lower() for f in forms["train"]}
        unseen = forms["test"] - seen
        covered = sum(f in words or f.lower() in words for f in unseen)
        report[iso] = {
            "lemma_model": files[0].name,
            "unseen_types": len(unseen),
            "covered": covered,
        }
    save(a.output / "stanza-coverage.json", report)


def summarize(a):
    home = json.loads(gzip.decompress((a.released / "pairs.json.gz").read_bytes()))[
        "official"
    ]
    ours = {}
    for iso, runs in home.items():
        c = runs[0]["test"]["counts"]
        ours[iso] = (
            c[0]["tokens"],
            c[0]["correct"],
            c[1 if gate(runs[0]["dev"]) else 0]["correct"],
        )
    report = {}
    for path in sorted(a.output.glob("*.json")):
        if path.name == "summary.json":
            continue
        if path.name == "stanza-coverage.json":
            continue
        result = read(path)
        scored = {i: c for i, c in result["cells"].items() if c["status"] == "scored"}
        entries = [(result["engine"], scored)]
        if result["engine"] == "stanza":
            # Drop models without a lemmatizer and models whose dictionary holds
            # at least half of the test words absent from the official training split.
            cover = read(a.output / "stanza-coverage.json")
            clean = {
                i: c
                for i, c in scored.items()
                if cover[i]["lemma_model"]
                and 2 * cover[i]["covered"] < cover[i]["unseen_types"]
            }
            entries.append(("stanza-screened", clean))
        for name, scored in entries:
            report.update(compare(name, result["version"], scored, ours))
    save(a.output / "summary.json", report)
    for name, v in sorted(report.items(), key=lambda kv: -kv[1]["languages"]):
        print(
            f"{name:18s} {str(v['version']):10s} langs={v['languages']:2d} theirs={v['macro_accuracy']:.2f} "
            f"declared={v['declared_macro_accuracy']:.2f} confirmed={v['confirmed_macro_accuracy']:.2f} "
            f"above={v['languages_above_declared']}"
        )


def compare(name, release, scored, ours):
    for iso, c in scored.items():
        if c["tokens"] != ours[iso][0]:
            raise ValueError(f"{name} was scored on other words for {iso}")
    if not scored:
        return {}
    n = len(scored)
    theirs = [100 * c["correct"] / c["tokens"] for c in scored.values()]
    declared = [100 * ours[i][1] / ours[i][0] for i in scored]
    confirmed = [100 * ours[i][2] / ours[i][0] for i in scored]
    words = sum(c["tokens"] for c in scored.values())
    return {
        name: {
            "version": release,
            "languages": n,
            "tokens": words,
            "macro_accuracy": sum(theirs) / n,
            "declared_macro_accuracy": sum(declared) / n,
            "confirmed_macro_accuracy": sum(confirmed) / n,
            "micro_accuracy": 100 * sum(c["correct"] for c in scored.values()) / words,
            "declared_micro_accuracy": 100 * sum(ours[i][1] for i in scored) / words,
            "languages_above_declared": sum(
                t > d for t, d in zip(theirs, declared, strict=True)
            ),
            "models": {i: c["model"] for i, c in scored.items()},
        }
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run")
    for key in ("order", "repo", "models", "output"):
        r.add_argument(key, type=Path)
    r.add_argument("--engines", nargs="+", default=["stanza", *ADAPTERS])
    r.add_argument("--jobs", type=int, default=8)
    r.add_argument("--stanza-jobs", type=int, default=4)
    r.add_argument("--predictions", type=Path, help="also store every answer")
    r.add_argument("--transfer", type=Path, help="score transfer corpora instead")
    c = sub.add_parser("screen")
    for key in ("order", "models", "output"):
        c.add_argument(key, type=Path)
    s = sub.add_parser("summary")
    for key in ("released", "output"):
        s.add_argument(key, type=Path)
    args = p.parse_args()
    {"run": run, "screen": screen, "summary": summarize}[args.mode](args)
