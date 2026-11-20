"""Compare deployed analysis chains on a frozen 224-language public udlex build."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent / "helpers"))
import benchmark as bench
import benchmark_threeway as three
import resource_metrics as resources

CJK = {"cmn", "yue", "jpn", "kor", "tha"}
MOD = SimpleNamespace(bench=bench, resources=resources)

def json_read(path):
    return json.loads(path.read_bytes())

def read_corpus(inputs, part):
    source = inputs / part["gold"]
    if bench.file_sha(source) != part["gold_sha256"]:
        raise ValueError(f"Gold digest differs: {source}")
    corpus = {"forms": [], "lemmas": [], "upos": []}
    with source.open() as stream:
        for line in stream:
            row = json.loads(line)
            for word, lemma, upos in zip(row["tokens"], row["lemmas"], row["upos"], strict=True):
                if word and lemma and (lemma != "_" or word == "_"):
                    corpus["forms"].append(word)
                    corpus["lemmas"].append(lemma)
                    corpus["upos"].append(upos)
    return corpus

def direct_answers(library, lexicon_dir, forms):
    """Use the original public single-lexicon entry point, independently of JNI."""
    lib = ctypes.CDLL(str(library))
    pointer = ctypes.c_void_p
    size = ctypes.c_size_t
    lib.udlex_native_open.argtypes = [ctypes.c_char_p, size, ctypes.POINTER(pointer)]
    lib.udlex_native_lemma.argtypes = [pointer, ctypes.c_char_p, size, ctypes.c_uint8, pointer, size, ctypes.POINTER(size)]
    lib.udlex_native_close.argtypes = [pointer]
    path = os.fsencode(lexicon_dir)
    handle = pointer()
    if lib.udlex_native_open(path, len(path), ctypes.byref(handle)):
        raise ValueError(f"Direct public reader failed to open {lexicon_dir}")
    answers = []
    try:
        buf = ctypes.create_string_buffer(256)
        for word in forms:
            token = word.encode()
            length = size()
            while True:
                status = lib.udlex_native_lemma(handle, token, len(token), 0, buf, len(buf), ctypes.byref(length))
                if status == 0:
                    answers.append(buf.raw[:length.value].decode())
                    break
                if status != 1 or length.value <= len(buf):
                    raise ValueError(f"Direct lookup failed for {word!r}: {status}")
                buf = ctypes.create_string_buffer(length.value)
    finally:
        if lib.udlex_native_close(handle):
            raise ValueError("Direct public reader failed to close")
    return answers

def eval_cell(ctx, entry, part, forms, corpus, timed):
    iso = entry["iso"]
    key = part["bundle"]
    profile = dict(ctx.profile, plugin_bundle=key)
    spec = three.analyzer_for(bench, profile, iso)
    identity = bench.digest([ctx.fingerprint, iso, key, part.get("gold_sha256"), bench.digest(forms)])
    work = ctx.args.output
    specs = [spec]
    udlex = profile["name"] in three.UDLEX_PROFILES
    if udlex:
        keyword = bench.Analyzer("reference", "isolated", {"tokenizer": "keyword", "filter": [spec.body["filter"][0]]}, "dictionary", True)
        specs.append(keyword)
    specs = bench.configure_indices({"isolated": ctx.client}, specs, iso, identity)
    try:
        sample, _ = three.select_sample(bench, iso, forms, ctx.args.sample_forms)
        predictions, receipt = bench.predict(ctx.client, specs[0], forms, work / "cache" / f"{iso}-{key}.jsonl", identity, 256)
        row = {"bundle": key, "analyzer": spec.body, "forms": len(forms), "alignment": receipt["alignment"]["mode"]}
        if corpus is not None:
            row["metrics"] = bench.score(corpus, predictions)
            row["gold_sha256"] = part["gold_sha256"]
        if timed:
            perf = three.measure(MOD, ctx.args, profile, ctx.client, specs[0], forms, predictions, identity, iso)
            row["speed"] = {**perf["summary"], "sample_forms": len(sample), "sample_sha256": perf["sample_sha256"],
                            "timing_sha256": perf["timing_file_sha256"]}
        if udlex:
            reference = direct_answers(ctx.lib, ctx.inputs / key / iso, sample)
            tokens, _ = bench.analyze(ctx.client, specs[1], sample)
            actual = bench.align_array(sample, tokens)
            if actual != [[answer] for answer in reference]:
                raise ValueError(f"{iso}/{key}: deployed keyword filter differs from the public core")
            row["parity_forms"] = len(sample)
        return row
    finally:
        ctx.client.request("DELETE", f"/{specs[0].index}")

def cell(ctx, entry, stage):
    iso = entry["iso"]
    path = ctx.args.output / stage / (iso + ".json")
    if path.exists():
        saved = json_read(path)
        if saved["fingerprint"] != ctx.fingerprint:
            raise ValueError(f"Cannot resume result {path} from a different run")
        return
    parts = entry["parts"]
    corpora = [read_corpus(ctx.inputs, p) for p in parts]
    record = {"iso": iso, "mode": entry["mode"], "fingerprint": ctx.fingerprint, "cells": []}
    if stage == "shipped":
        if entry["mode"] == "test":
            forms = sorted(set(corpora[0]["forms"]))
            part = parts[0]
            corpus = corpora[0]
        else:
            available = [f for corpus in corpora for f in corpus["forms"]] if corpora else entry["source_forms"]
            forms = sorted(set(available))
            # Only timing is measured with the shipped, full-corpus two-fold dictionaries.
            # Use exactly the same workload selection as the timed measurement.
            forms, _ = three.select_sample(bench, iso, forms, ctx.args.sample_forms)
            part, corpus = {"bundle": "bundle"}, None
        record["cells"].append(eval_cell(ctx, entry, part, forms, corpus, True))
    else:
        for part, corpus in zip(parts, corpora, strict=True):
            record["cells"].append(eval_cell(ctx, entry, part, sorted(set(corpus["forms"])), corpus, False))
    bench.atomic_json(path, record)
    print(f"{stage} {iso}", flush=True)

def main():
    pr = argparse.ArgumentParser()
    pr.add_argument("root", type=Path)
    pr.add_argument("profile")
    pr.add_argument("--languages", nargs="+")
    args = pr.parse_args()
    root = args.root.resolve()
    stand = root / "stand"
    inputs = root / "inputs"
    profile = json_read(stand / "profiles" / (args.profile + ".json"))
    profile["plugin_bundle"] = "bundle"
    args.output = stand / "results" / args.profile
    args.output.mkdir(parents=True, exist_ok=True)
    args.sample_forms, args.warmup, args.blocks, args.block_seconds, args.resource_interval = 256, 5, 3, 0.50, 0.10
    client = three.make_client(bench, profile["url"], 120)
    process = resources.read_process_identity(profile["pid"])
    server = client.info()
    if server["version"]["number"] != "3.10.0":
        raise ValueError("Unexpected OpenSearch version")
    three.validate_runtime(profile, process, server)
    resources.jvm_snapshot(client, profile["pid"])
    manifest = json_read(inputs / "inputs.json")
    if bench.file_sha(inputs / "DICTIONARIES.json") != manifest["catalog_sha256"]:
        raise ValueError("Catalog changed")
    here = Path(__file__).resolve().parent
    lib = here / "artifacts/libudlex_capi.so"
    receipt = {"catalog_sha256": manifest["catalog_sha256"], "inputs_sha256": bench.file_sha(inputs/"inputs.json"),
               "server": server, "profile": profile, "process": process, "parameters": {k:v for k,v in vars(args).items() if k not in {"output","root","profile"}},
               "build_core_commit": json_read(here / "benchmark.lock.json")["core_commit"], "runner_sha256": bench.file_sha(Path(__file__)),
               "helpers": {p.name:bench.file_sha(p) for p in sorted((here/"helpers").glob("*.py"))},
               "plugin_zip_sha256": bench.file_sha(here / "artifacts/analysis-udlex.zip"),
               "direct_reference_sha256": bench.file_sha(lib)}
    fingerprint = bench.digest(receipt)
    receipt["fingerprint"] = fingerprint
    dest = args.output / "environment.json"
    if dest.exists() and json_read(dest) != receipt:
        raise ValueError("JVM, inputs or benchmark changed during resume")
    bench.atomic_json(dest, receipt)
    ctx = SimpleNamespace(profile=profile, fingerprint=fingerprint, args=args, client=client, inputs=inputs, lib=lib)
    roster = [r for r in manifest["languages"] if args.profile not in three.OFFICIAL_PLUGIN_LABELS or r["iso"] in CJK]
    if args.languages:
        roster = [r for r in roster if r["iso"] in args.languages]
    common = sorted(r["iso"] for r in roster if r["iso"] in bench.LANGUAGE_ANALYZERS)
    roster.sort(key=lambda row:(row["iso"] not in common,row["iso"]))
    roster = [row for row in roster if three.analyzer_for(bench, profile, row["iso"]) is not None]
    def snapshot(name, done):
        target = args.output / ("memory-" + name + ".json")
        if not target.exists():
            bench.atomic_json(target, three.resource_snapshot(MOD, client, profile, done))
    snapshot("start", [])
    completed = []
    for row in roster:
        cell(ctx,row,"shipped")
        completed.append(row["iso"])
        if completed == common:
            snapshot("common",completed)
    snapshot("shipped",completed)
    for row in roster:
        if row["mode"] == "two-fold":
            cell(ctx,row,"heldout")
    snapshot("final",completed)
    bench.atomic_json(args.output/"complete.json",{"fingerprint":fingerprint,"languages":completed})
    print("complete", args.profile, len(completed), flush=True)

if __name__ == "__main__":
    main()
