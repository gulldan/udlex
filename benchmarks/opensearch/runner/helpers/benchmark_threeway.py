"""Measure three explicitly named OpenSearch pipelines on isolated JVM profiles.

Run once per profile, with the same frozen source, bundle, gold and exclusions.
Existing benchmark code and results are read only. Timings include persistent
HTTP transport and JSON encoding/decoding; they are not isolated core timings.
"""

# Standalone orchestration deliberately retains detailed JSON evidence.
# ruff: noqa: ANN401, C901, E501, EM101, EM102, INP001, PLR0912, PLR0913, PLR0915, PLR0917, TRY003

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import http.client
import importlib
import importlib.util
import json
import math
import os
import platform
import random
import statistics
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SCHEMA = "udlex-opensearch-threeway-suite/v1"
PROFILE_SCHEMA = "udlex-opensearch-threeway-profile/v1"
LABELS = ("OpenSearch", "Native", "Udlex")
# Supplementary measurement rows, not library variants: the same udlex plugin
# and bundles, only the tokenizer / official analysis plugins differ. The
# distribution does not ship them; CJK subset only, never merged into the main table.
OFFICIAL_PLUGIN_LABELS = ("Native-kuromoji-smartcn", "Udlex-kuromoji-icu")
UDLEX_PROFILES = ("Udlex", "Udlex-kuromoji-icu")
OFFICIAL_PLUGIN_ANALYZERS = {"jpn": "kuromoji", "cmn": "smartcn", "yue": "smartcn"}
OFFICIAL_PLUGIN_TOKENIZERS = {
    "jpn": "kuromoji_tokenizer",
    "cmn": "icu_tokenizer",
    "yue": "icu_tokenizer",
}
OFFICIAL_PLUGINS = {
    "Native-kuromoji-smartcn": {"analysis-kuromoji", "analysis-smartcn"},
    "Udlex-kuromoji-icu": {"analysis-kuromoji", "analysis-icu"},
}
ACCEPTED = {"ok", "no_scorable_gold", "unsupported_analyzer"}


def read_json(path: Path) -> dict[str, Any]:
    """Read an object without modifying its source."""
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected JSON object")
    return value


def load_helpers(source_root: Path) -> SimpleNamespace:
    """Import the explicitly pinned harness, never an ambient package."""
    source_root = source_root.resolve()
    sys.path.insert(0, str(source_root / "src"))
    sys.path.insert(0, str(source_root))
    modules = {}
    for key, name in (
        ("bench", "integrations.opensearch.benchmark"),
        ("datasets", "integrations.opensearch.datasets"),
        ("repair", "integrations.opensearch.compare_dictionary_repair"),
        ("bundle", "bench.adapter_bundle"),
    ):
        module = importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(source_root):
            raise ValueError(f"{name} did not load from --source-root")
        modules[key] = module
    resource_path = HERE / "resource_metrics.py"
    spec = importlib.util.spec_from_file_location(
        "threeway_resource_metrics", resource_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {resource_path}")
    resource = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = resource
    spec.loader.exec_module(resource)
    modules["resources"] = resource
    return SimpleNamespace(**modules)


def make_client(bench: Any, endpoint: str, timeout: float) -> Any:
    """Disable hidden retries for auditable latency and explicit failures."""

    class StrictHTTP(bench.HTTP):
        def request(
            self,
            method: str,
            path: str,
            body: Any = None,
            allowed: tuple[int, ...] = (),
        ) -> tuple[Any, float]:
            started = time.perf_counter()
            raw = None if body is None else bench.canonical(body)
            try:
                self.connection.request(
                    method,
                    self.prefix + path,
                    body=raw,
                    headers={"Content-Type": "application/json"},
                )
                response = self.connection.getresponse()
                data = response.read()
                parsed = json.loads(data) if data else {}
                if response.status >= 400 and response.status not in allowed:
                    raise bench.HTTPError(response.status, method, path, parsed)
                return parsed, (time.perf_counter() - started) * 1000
            except (OSError, http.client.HTTPException):
                self.connection.close()
                raise

    return StrictHTTP(endpoint, timeout)


def validate_profile(profile: dict[str, Any]) -> None:
    """Require explicit names, host PID and verifiable resource limits."""
    names = LABELS + OFFICIAL_PLUGIN_LABELS
    if profile.get("schema") != PROFILE_SCHEMA or profile.get("name") not in names:
        raise ValueError(f"profile must use {PROFILE_SCHEMA} and a name in {names}")
    pid = profile.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("profile.pid must be the positive host JVM PID")
    limits = profile.get("runtime_limits", {})
    cpus = limits.get("allowed_cpus")
    if (
        not isinstance(cpus, list)
        or not cpus
        or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in cpus)
        or len(set(cpus)) != len(cpus)
    ):
        raise ValueError("runtime_limits.allowed_cpus must be explicit unique CPU IDs")
    heap = limits.get("heap_max_bytes")
    if isinstance(heap, bool) or not isinstance(heap, int) or heap <= 0:
        raise ValueError("runtime_limits.heap_max_bytes must be a positive integer")
    if profile["name"] in UDLEX_PROFILES:
        for key in ("plugin_bundle", "plugin_bundle_dir", "installed_plugin_dir"):
            if not isinstance(profile.get(key), str) or not profile[key]:
                raise ValueError(f"Udlex profile requires {key}")


def validate_runtime(
    profile: dict[str, Any], process: dict[str, Any], server: dict[str, Any]
) -> dict[str, Any]:
    """Compare declarations with actual affinity, JVM heap and plugin inventory."""
    limits = profile["runtime_limits"]
    if sorted(process["allowed_cpus"]) != sorted(limits["allowed_cpus"]):
        raise ValueError("actual server CPU affinity differs from runtime_limits")
    nodes = server["runtime_per_node"]
    if len(nodes) != 1 or len(server["plugins_per_node"]) != 1:
        raise ValueError("each profile must be a single isolated OpenSearch node")
    node = nodes[0]
    if node["jvm_memory"]["heap_max_in_bytes"] != limits["heap_max_bytes"]:
        raise ValueError("actual JVM maximum heap differs from runtime_limits")
    if (
        "heap_init_bytes" in limits
        and node["jvm_memory"]["heap_init_in_bytes"] != limits["heap_init_bytes"]
    ):
        raise ValueError("actual JVM initial heap differs from runtime_limits")
    if (
        "allocated_processors" in limits
        and node["os"]["allocated_processors"] != limits["allocated_processors"]
    ):
        raise ValueError(
            "actual OpenSearch allocated_processors differs from runtime_limits"
        )
    names = {p["name"] for p in server["plugins_per_node"][0]}
    if any("udlex" in name for name in names) != (profile["name"] in UDLEX_PROFILES):
        raise ValueError(
            "Udlex must be installed only in the Udlex and Udlex-kuromoji-icu profiles"
        )
    required = OFFICIAL_PLUGINS.get(profile["name"], set())
    installed = names & set().union(*OFFICIAL_PLUGINS.values())
    if installed != required:
        raise ValueError(
            f"{profile['name']} requires analysis plugins {sorted(required)}, node has {sorted(installed)}"
        )
    client_cpus = sorted(os.sched_getaffinity(0))
    if "client_allowed_cpus" in limits and client_cpus != sorted(
        limits["client_allowed_cpus"]
    ):
        raise ValueError("actual client CPU affinity differs from runtime_limits")
    return {
        "verified": True,
        "client_allowed_cpus": client_cpus,
        "server_allowed_cpus": process["allowed_cpus"],
    }


def analyzer_for(bench: Any, profile: dict[str, Any], language: str) -> Any | None:
    """Return the exact requested deployed chain, with no Native fallback."""
    name = profile["name"]
    if name == "OpenSearch":
        body, kind = {"analyzer": "standard"}, "shipped_default_standard"
    elif name in ("Native", "Native-kuromoji-smartcn"):
        plugin = (
            OFFICIAL_PLUGIN_ANALYZERS.get(language)
            if name == "Native-kuromoji-smartcn"
            else None
        )
        builtin = plugin or bench.LANGUAGE_ANALYZERS.get(language)
        if builtin is None:
            return None
        body = {"analyzer": builtin}
        kind = (
            "official_plugin_language_analyzer"
            if plugin
            else "shipped_builtin_language_analyzer"
        )
    else:
        tokenizer = (
            OFFICIAL_PLUGIN_TOKENIZERS.get(language, "standard")
            if name == "Udlex-kuromoji-icu"
            else "standard"
        )
        body = {
            "tokenizer": tokenizer,
            "filter": [
                {
                    "type": "udlex",
                    "language": language,
                    "bundle_path": profile["plugin_bundle"],
                    "use_rules": True,
                },
                "lowercase",
            ],
        }
        kind = (
            "standard_tokenizer" if tokenizer == "standard" else tokenizer
        ) + "_then_udlex_then_lowercase"
    return bench.Analyzer(name.lower(), "isolated", body, kind)


def ordered_entries(
    bench: Any, entries: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run the identical supported cohort first for comparable accumulated RAM."""
    common = sorted(
        entry["language"]
        for entry in entries
        if entry["status"] == "available"
        and entry["language"] in bench.LANGUAGE_ANALYZERS
    )
    membership = set(common)
    return sorted(
        entries, key=lambda e: (e["language"] not in membership, e["language"])
    ), common


def common_snapshot_context(
    common: list[str], rows: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Refuse a common-cohort RAM claim after later languages were attempted."""
    later = sorted(
        language
        for language, row in rows.items()
        if language not in common and row.get("status") in {"ok", "failed"}
    )
    return {
        "comparable": bool(common) and not later,
        "earlier_noncommon_attempts": later,
        "reason": "Exact common cohort, before remaining languages"
        if common and not later
        else "No common cohort, or resume followed attempts outside the common cohort; RAM is not comparable as common-only state",
    }


def validate_output(output: Path, inputs: list[Path]) -> None:
    """Keep all checkpoints outside immutable source and input trees."""
    if any(output.resolve().is_relative_to(root.resolve()) for root in inputs):
        raise ValueError("--output must be outside every source, gold and bundle tree")


def select_sample(
    bench: Any, language: str, forms: list[str], count: int
) -> tuple[list[str], str]:
    """Shuffle a complete unique-form inventory without using labels or profile."""
    unique = sorted(set(forms))
    seed = bench.digest(
        {"policy": "unique-surface-shuffle/v1", "language": language, "forms": unique}
    )
    random.Random(int(seed, 16)).shuffle(unique)  # noqa: S311 -- reproducible workload, not cryptography
    return unique[:count], seed


def timing_summary(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate totals; throughput is not the reciprocal of median latency."""
    request_ms = [value for block in blocks for value in block["request_ms"]]
    elapsed = sum(block["elapsed_seconds"] for block in blocks)
    input_forms = sum(block["input_forms"] for block in blocks)
    cpu_seconds = sum(block["resources"]["cpu_seconds_delta"] for block in blocks)
    cpu_wall = sum(block["resources"]["wall_seconds"] for block in blocks)
    output = {
        "elapsed_seconds": elapsed,
        "http_elapsed_seconds": sum(request_ms) / 1000,
        "requests": len(request_ms),
        "input_forms": input_forms,
        "output_tokens": sum(block["output_tokens"] for block in blocks),
        "input_characters": sum(block["input_characters"] for block in blocks),
        "input_utf8_bytes": sum(block["input_utf8_bytes"] for block in blocks),
        "forms_per_second": input_forms / elapsed if elapsed else None,
        "http_forms_per_second": input_forms / (sum(request_ms) / 1000)
        if request_ms
        else None,
        "request_ms_median": statistics.median(request_ms) if request_ms else None,
        "request_ms_p95": sorted(request_ms)[math.ceil(0.95 * len(request_ms)) - 1]
        if request_ms
        else None,
        "cpu_seconds": cpu_seconds,
        "cpu_measurement_seconds": cpu_wall,
        "cpu_cores": cpu_seconds / cpu_wall if cpu_wall else None,
        "cpu_seconds_per_million_forms": cpu_seconds * 1_000_000 / input_forms
        if input_forms
        else None,
        "cpu_percent_one_core": 100 * cpu_seconds / cpu_wall if cpu_wall else None,
        "cpu_percent_allocated": sum(
            block["resources"]["cpu_percent_allocated"]
            * block["resources"]["wall_seconds"]
            for block in blocks
        )
        / cpu_wall
        if cpu_wall
        else None,
    }
    for key in ("rss_bytes", "pss_bytes"):
        peaks = [
            block["resources"][key]["peak"]
            for block in blocks
            if block["resources"][key]["peak"] is not None
        ]
        measured = [
            (
                block["resources"][key]["mean"],
                block["resources"]["wall_seconds"]
                * block["resources"][key]["coverage_fraction"],
            )
            for block in blocks
            if block["resources"][key]["mean"] is not None
            and block["resources"][key]["coverage_fraction"] > 0
        ]
        observed_seconds = sum(seconds for _, seconds in measured)
        output[key] = {
            "peak": max(peaks) if peaks else None,
            "mean": sum(value * seconds for value, seconds in measured)
            / observed_seconds
            if measured
            else None,
            "coverage_fraction": observed_seconds / cpu_wall if cpu_wall else 0,
            "observed_seconds": observed_seconds,
            "mean_available_blocks": len(measured),
            "missing_mean_blocks": len(blocks) - len(measured),
        }
    return output


def load_timing_blocks(path: Path, fingerprint: str) -> list[dict[str, Any]]:
    """Recover completed timing blocks, discarding only an incomplete last line."""
    if not path.exists():
        return []
    records = []
    valid_bytes = 0
    with path.open("rb") as stream:
        for line_no, raw in enumerate(stream):
            if not raw.endswith(b"\n"):
                break
            item = json.loads(raw)
            if line_no == 0:
                if item != {
                    "schema": "udlex-threeway-timings/v1",
                    "fingerprint": fingerprint,
                }:
                    raise ValueError(f"timing fingerprint mismatch: {path}")
            else:
                if item.get("block") != len(records) or not item.get("request_ms"):
                    raise ValueError(f"timing block order/content mismatch: {path}")
                records.append(item)
            valid_bytes += len(raw)
    if path.stat().st_size != valid_bytes:
        with path.open("r+b") as stream:
            stream.truncate(valid_bytes)
    return records


def measure(
    modules: SimpleNamespace,
    args: argparse.Namespace,
    profile: dict[str, Any],
    client: Any,
    spec: Any,
    forms: list[str],
    predictions: dict[str, list[str]],
    fingerprint: str,
    language: str,
) -> dict[str, Any]:
    """Measure complete requests in bounded blocks, preserving every duration."""
    bench, resources = modules.bench, modules.resources
    sample, seed = select_sample(bench, language, forms, args.sample_forms)
    expected = [token for form in sample for token in predictions[form]]
    expected_sha = bench.digest(expected)
    key = bench.digest(
        {"cell": fingerprint, "sample": sample, "expected": expected_sha}
    )
    path = args.output / "timings" / f"{language}.jsonl"
    blocks = load_timing_blocks(path, key)
    if len(blocks) > args.blocks:
        raise ValueError("timing checkpoint contains too many blocks")
    # These requests also validate that the timed multi-value call has the same
    # ordered terms as independently aligned full-corpus quality predictions.
    for _ in range(args.warmup):
        tokens, _ = bench.analyze(client, spec, sample)
        if [token["token"] for token in tokens] != expected:
            raise ValueError("timed batch differs from aligned quality predictions")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        if stream.tell() == 0:
            stream.write(
                bench.canonical(
                    {"schema": "udlex-threeway-timings/v1", "fingerprint": key}
                )
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        for number in range(len(blocks), args.blocks):
            jvm_before = resources.jvm_snapshot(client, expected_pid=profile["pid"])
            sampler = resources.ProcessSampler(
                profile["pid"], interval_seconds=args.resource_interval
            )
            sampler.start()
            durations = []
            started_at = bench.utc_now()
            started = time.perf_counter()
            try:
                while (
                    not durations or time.perf_counter() - started < args.block_seconds
                ):
                    tokens, elapsed_ms = bench.analyze(client, spec, sample)
                    durations.append(elapsed_ms)
                    if [token["token"] for token in tokens] != expected:
                        raise ValueError(
                            "timed request changed its predicted token sequence"
                        )
                ended = time.perf_counter()
            finally:
                resource_result = sampler.stop()
            jvm_after = resources.jvm_snapshot(client, expected_pid=profile["pid"])
            elapsed = ended - started
            block = {
                "block": number,
                "started_at": started_at,
                "elapsed_seconds": elapsed,
                "http_elapsed_seconds": sum(durations) / 1000,
                "resource_extra_seconds": resource_result["wall_seconds"] - elapsed,
                "requests": len(durations),
                "input_forms": len(sample) * len(durations),
                "output_tokens": len(expected) * len(durations),
                "input_characters": sum(map(len, sample)) * len(durations),
                "input_utf8_bytes": sum(len(value.encode()) for value in sample)
                * len(durations),
                "request_ms": durations,
                "resources": resource_result,
                "jvm_before": jvm_before,
                "jvm_after": jvm_after,
                "jvm_delta": resources.jvm_delta(jvm_before, jvm_after),
            }
            stream.write(bench.canonical(block) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
            blocks.append(block)
    return {
        "status": "ok",
        "kind": "persistent_HTTP_analyze_encode_through_decode",
        "sample_forms": len(sample),
        "sample": sample,
        "sample_sha256": bench.digest(sample),
        "sample_seed_sha256": seed,
        "sample_policy": "sorted unique forms, deterministic seeded shuffle, no gold labels",
        "expected_output_tokens_per_request": len(expected),
        "expected_output_sha256": expected_sha,
        "warmup_requests": args.warmup,
        "minimum_block_seconds": args.block_seconds,
        "concurrency": 1,
        "automatic_retries": 0,
        "blocks": blocks,
        "summary": timing_summary(blocks),
        "timing_file_sha256": bench.file_sha(path),
        "note": "Wall throughput includes per-response validation and sampler scheduling. HTTP latencies include JSON encode/decode, exclude validation. CPU interval is slightly broader and its extra time is recorded. Memory is the whole warmed JVM with dictionaries retained across prior languages.",
    }


def resource_snapshot(
    modules: SimpleNamespace, client: Any, profile: dict[str, Any], completed: list[str]
) -> dict[str, Any]:
    """Record node-wide state at a comparable cohort boundary."""
    return {
        "captured_at": modules.bench.utc_now(),
        "completed_languages": completed,
        "process": modules.resources.process_snapshot(profile["pid"]),
        "jvm": modules.resources.jvm_snapshot(client, expected_pid=profile["pid"]),
    }


def run_language(
    modules: SimpleNamespace,
    args: argparse.Namespace,
    profile: dict[str, Any],
    client: Any,
    entry: dict[str, Any],
    fingerprint: str,
) -> dict[str, Any]:
    """Score complete held-out data, then measure the same deployed pipeline."""
    bench = modules.bench
    language = entry["language"]
    result = {
        "language": language,
        "language_name": entry.get("language_name", language),
        "entry": entry,
        "profile": profile["name"],
        "manifest_fingerprint": fingerprint,
        "metrics": None,
        "performance": None,
    }
    if entry["status"] != "available":
        result.update(
            status=entry["status"], reason="No scored held-out gold is available"
        )
        return result
    corpus = modules.datasets.load_corpus(entry)
    clean, evidence = modules.repair.clean_gold(corpus)
    result.update(
        corpus={
            "gold_sha256": corpus["gold_sha256"],
            "selection_sha256": corpus["selection_sha256"],
            "grid_sha256": modules.repair.grid_sha(corpus),
            "tokens": len(corpus["forms"]),
            "lexical_tokens": sum(corpus["lexical_mask"]),
            "sentences": corpus["selected_sentences"],
            **{
                key: corpus[key]
                for key in (
                    "excluded_sentences",
                    "excluded_tokens",
                    "excluded_scorable_tokens",
                )
            },
        },
        clean_policy=evidence,
        quality_support="lexical_gold"
        if any(corpus["lexical_mask"])
        else "punctuation_only",
    )
    spec = analyzer_for(bench, profile, language)
    if spec is None:
        result.update(
            status="unsupported_analyzer",
            reason="OpenSearch has no shipped built-in analyzer for this language",
        )
        return result
    forms = list(dict.fromkeys(corpus["forms"]))
    cell = bench.digest(
        {
            "manifest": fingerprint,
            "language": language,
            "grid": result["corpus"]["grid_sha256"],
            "forms": forms,
        }
    )
    result["fingerprint"] = cell
    spec = bench.configure_indices({"isolated": client}, [spec], language, fingerprint)[
        0
    ]
    result["analyzer"] = dataclasses.asdict(spec)
    try:
        cache = args.output / "caches" / f"{language}.jsonl"
        predictions, collection = bench.predict(
            client, spec, forms, cache, cell, args.batch_size
        )
        result.update(
            metrics={
                "raw": bench.score(corpus, predictions),
                "clean": bench.score(clean, predictions),
            },
            collection={**collection, "prediction_cache_sha256": bench.file_sha(cache)},
        )
        result["performance"] = measure(
            modules, args, profile, client, spec, forms, predictions, cell, language
        )
        result["status"] = "ok"
        return result
    finally:
        client.request("DELETE", f"/{spec.index}", allowed=(404,))


def make_parser() -> argparse.ArgumentParser:
    """Expose a stable launch contract suitable for nohup and resumption."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument(
        "--root", required=True, type=Path, help="Frozen gold/input snapshot"
    )
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--exclusions", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--languages", help="Explicit pilot subset; omit for every inventory language"
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--sample-forms", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--block-seconds", type=float, default=1.0)
    parser.add_argument("--resource-interval", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Freeze inputs, resume one JVM session, and publish a complete suite."""
    args = make_parser().parse_args(argv)
    for key in (
        "batch_size",
        "sample_forms",
        "warmup",
        "blocks",
        "block_seconds",
        "resource_interval",
        "timeout",
    ):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f"{key} must be positive and finite")
    for key in ("source_root", "root", "bundle", "exclusions", "profile", "output"):
        setattr(args, key, getattr(args, key).resolve())
    profile = read_json(args.profile)
    validate_profile(profile)
    protected = [args.source_root, args.root, args.bundle]
    if profile["name"] in UDLEX_PROFILES:
        protected.extend(
            Path(profile[key]) for key in ("plugin_bundle_dir", "installed_plugin_dir")
        )
    validate_output(args.output, protected)
    modules = load_helpers(args.source_root)
    bench = modules.bench
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    client = make_client(bench, profile["url"], args.timeout)
    process_identity = modules.resources.read_process_identity(profile["pid"])
    server = client.info()
    runtime_validation = validate_runtime(profile, process_identity, server)
    # Also bind the endpoint's process to the host PID (including PID namespaces).
    modules.resources.jvm_snapshot(client, expected_pid=profile["pid"])
    bench.event(args.output, "preflight_started", profile=profile["name"])
    data = modules.datasets.inventory(
        args.root, bundle=args.bundle, exclusions=args.exclusions
    )
    if (
        data["bundle"].get("benchmark_eligible") is not True
        or data["bundle"].get("transductive_refit") is True
    ):
        raise ValueError(
            "bundle must explicitly be benchmark eligible without transductive refitting"
        )
    bundle = modules.bundle.bundle_identity(args.bundle)
    installed = None
    if profile["name"] in UDLEX_PROFILES:
        deployed = modules.bundle.bundle_identity(Path(profile["plugin_bundle_dir"]))
        if deployed != bundle:
            raise ValueError(
                "deployed plugin bundle bytes differ from the scored bundle"
            )
        installed = bench.installed_plugin_identity(
            Path(profile["installed_plugin_dir"])
        )
    selected = set(args.languages.split(",")) if args.languages else None
    entries = [
        entry
        for entry in data["entries"]
        if selected is None or entry["language"] in selected
    ]
    if selected and selected != {entry["language"] for entry in entries}:
        raise ValueError("--languages contains unknown inventory entries")
    entries, common = ordered_entries(bench, entries)
    parameters = {
        key: getattr(args, key)
        for key in (
            "batch_size",
            "sample_forms",
            "warmup",
            "blocks",
            "block_seconds",
            "resource_interval",
            "timeout",
            "languages",
        )
    }
    source = bench.source_identity(args.source_root)
    identity = {
        "schema": SCHEMA,
        "profile": profile,
        "profile_sha256": bench.file_sha(args.profile),
        "server": server,
        "process_identity": process_identity,
        "runtime_validation": runtime_validation,
        "source": source,
        "runner_sha256": bench.file_sha(Path(__file__)),
        "resource_sampler_sha256": bench.file_sha(HERE / "resource_metrics.py"),
        "bundle": bundle,
        "installed_plugin": installed,
        "inventory": [
            {
                key: value
                for key, value in entry.items()
                if key not in {"path", "bundle_path"}
            }
            for entry in entries
        ],
        "exclusion_audit_sha256": bench.file_sha(args.exclusions),
        "annotation_policy": modules.repair.POLICY,
        "annotation_policy_sha256": bench.digest(modules.repair.POLICY),
        "parameters": parameters,
        "common_languages": common,
        "language_order": [entry["language"] for entry in entries],
        "python": platform.python_version(),
    }
    fingerprint = bench.digest(identity)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        old = read_json(manifest_path)
        if (
            old.get("fingerprint") != fingerprint
            or bench.digest(old.get("identity")) != fingerprint
        ):
            raise ValueError(
                "profile/PID/input fingerprint changed: use another --output; old results are preserved"
            )
    else:
        bench.atomic_json(
            manifest_path,
            {
                "schema": "udlex-opensearch-threeway-manifest/v1",
                "fingerprint": fingerprint,
                "identity": identity,
                "started_at": bench.utc_now(),
                "inventory": data,
                "methodology": {
                    "systems": {
                        "OpenSearch": "shipped standard analyzer",
                        "Native": "shipped built-in language analyzer, no unsupported fallback",
                        "Udlex": "standard tokenizer -> native udlex -> lowercase",
                        "Native-kuromoji-smartcn": "Native row with official analysis plugins: jpn kuromoji analyzer, cmn/yue smartcn analyzer, other languages as Native; CJK subset only",
                        "Udlex-kuromoji-icu": "Same udlex plugin and bundle as Udlex, only the tokenizer differs: jpn kuromoji_tokenizer, cmn/yue icu_tokenizer, other languages standard; CJK subset only",
                    },
                    "quality": "Complete held-out gold with pinned overlap exclusions; per-form _analyze and live UTF-16 multi-value alignment checks; split and dropped outputs retained.",
                    "pair_f1": "Distinct lexical surface-type conflation with unambiguous casefold gold lemma; matching nonempty whole token sequences. This is not ranked retrieval and has no qrels.",
                    "clean": "Separate post-hoc source-specific invalid-annotation sensitivity; original raw gold metrics retained.",
                    "speed": "Same deterministic shuffled unique forms; serial persistent local HTTP including encode/decode. Warmups follow complete quality collection. No retry or concurrency.",
                    "resources": "Whole JVM process CPU and RSS/PSS sampled during blocks; JVM stats queried outside timing. Baseline and common-cohort snapshots precede remaining languages. Retained dictionaries and JVM caches make later per-language RAM cumulative.",
                    "gpu": "No GPU work; OpenSearch and native morphology run on CPU.",
                    "resume": "Only identical host JVM PID/starttime/boot, verified limits and all input fingerprints may reuse checkpoints. Restart requires a new directory.",
                },
            },
        )
    rows: dict[str, dict[str, Any]] = {}
    for entry in entries:
        path = args.output / "languages" / f"{entry['language']}.json"
        if path.exists():
            previous = read_json(path)
            if previous.get("manifest_fingerprint") != fingerprint:
                raise ValueError(f"language manifest mismatch: {path}")
            rows[entry["language"]] = previous
    snapshots_path = args.output / "resources.json"
    if snapshots_path.exists():
        snapshot_state = read_json(snapshots_path)
        if snapshot_state.get("fingerprint") != fingerprint:
            raise ValueError("resource snapshot fingerprint mismatch")
        snapshots = snapshot_state["snapshots"]
    else:
        snapshots = {"startup": resource_snapshot(modules, client, profile, [])}
        bench.atomic_json(
            snapshots_path, {"fingerprint": fingerprint, "snapshots": snapshots}
        )

    def publish(status: str, integrity: dict[str, Any] | None = None) -> None:
        bench.atomic_json(
            args.output / "suite.json",
            {
                "schema": SCHEMA,
                "fingerprint": fingerprint,
                "profile": profile,
                "manifest_sha256": bench.file_sha(manifest_path),
                "status": status,
                "updated_at": bench.utc_now(),
                "common_languages": common,
                "expected_languages": identity["language_order"],
                "resource_snapshots": snapshots,
                "input_integrity": integrity,
                "results": [
                    rows[entry["language"]]
                    for entry in entries
                    if entry["language"] in rows
                ],
            },
        )

    publish("running")
    for entry in entries:
        language = entry["language"]
        if rows.get(language, {}).get("status") in ACCEPTED:
            bench.event(args.output, "language_resumed_complete", language=language)
        else:
            bench.event(
                args.output,
                "language_started",
                language=language,
                profile=profile["name"],
            )
            try:
                row = run_language(modules, args, profile, client, entry, fingerprint)
            except Exception as exc:  # noqa: BLE001 -- record failures without inventing scores
                row = {
                    "language": language,
                    "language_name": entry.get("language_name", language),
                    "entry": entry,
                    "profile": profile["name"],
                    "status": "failed",
                    "metrics": None,
                    "performance": None,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            row.update(manifest_fingerprint=fingerprint, finished_at=bench.utc_now())
            rows[language] = row
            bench.atomic_json(args.output / "languages" / f"{language}.json", row)
            bench.event(
                args.output,
                "language_completed",
                language=language,
                status=row["status"],
            )
        if "post_common_cohort" not in snapshots and all(
            rows.get(lang, {}).get("status") == "ok" for lang in common
        ):
            snapshots["post_common_cohort"] = resource_snapshot(
                modules, client, profile, common
            )
            snapshots["post_common_cohort"].update(
                common_snapshot_context(common, rows)
            )
            bench.atomic_json(
                snapshots_path, {"fingerprint": fingerprint, "snapshots": snapshots}
            )
        publish("running")
    if "final" in snapshots:
        snapshots.setdefault("final_history", []).append(snapshots["final"])
    snapshots["final"] = resource_snapshot(
        modules,
        client,
        profile,
        [
            entry["language"]
            for entry in entries
            if rows[entry["language"]]["status"] == "ok"
        ],
    )
    bench.atomic_json(
        snapshots_path, {"fingerprint": fingerprint, "snapshots": snapshots}
    )
    # Rehash exact inputs after work; do not accept a mid-run mutation.
    checks = {
        "source_unchanged": bench.source_identity(args.source_root) == source,
        "runner_unchanged": bench.file_sha(Path(__file__)) == identity["runner_sha256"],
        "resource_sampler_unchanged": bench.file_sha(HERE / "resource_metrics.py")
        == identity["resource_sampler_sha256"],
        "bundle_unchanged": modules.bundle.bundle_identity(args.bundle) == bundle,
        "exclusions_unchanged": bench.file_sha(args.exclusions)
        == identity["exclusion_audit_sha256"],
        "profile_unchanged": bench.file_sha(args.profile) == identity["profile_sha256"],
        "process_unchanged": modules.resources.read_process_identity(profile["pid"])
        == process_identity,
        "gold_unchanged": all(
            bench.file_sha(Path(entry["path"])) == entry["gold_sha256"]
            for entry in entries
            if entry.get("gold_sha256")
        ),
    }
    if profile["name"] in UDLEX_PROFILES:
        checks.update(
            plugin_bundle_unchanged=modules.bundle.bundle_identity(
                Path(profile["plugin_bundle_dir"])
            )
            == bundle,
            installed_plugin_unchanged=bench.installed_plugin_identity(
                Path(profile["installed_plugin_dir"])
            )
            == installed,
        )
    integrity = {
        "checked_at": bench.utc_now(),
        "checks": checks,
        "passed": all(checks.values()),
    }
    failures = sum(row["status"] not in ACCEPTED for row in rows.values())
    passed = not failures and integrity["passed"]
    publish("complete" if passed else "failed", integrity)
    bench.event(
        args.output,
        "run_finished",
        profile=profile["name"],
        languages=len(entries),
        failures=failures,
        input_integrity=integrity["passed"],
    )
    client.connection.close()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
