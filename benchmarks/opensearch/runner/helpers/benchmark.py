"""Resumable HTTP evaluation of stock OpenSearch and the native udlex filter.

Run on the same host as two disposable OpenSearch instances.  No gold labels
are sent to OpenSearch or to udlex.  Gold UD tokens measure morphology; sentence
smoke checks exercise actual tokenization, indexing, queries and highlighting.
Pair scores are lexical conflation proxies, not ranked retrieval/qrels scores.
"""

# This standalone stdlib runner exchanges heterogeneous JSON and reports explicit
# boundary failures. Orchestration is kept here to preserve one launchable file.
# Long diagnostic strings and positional flags mirror the native analysis API.
# ruff: noqa: ANN401, C901, E501, EM101, EM102, FBT001, FBT003, INP001, PLR0912, PLR0913, PLR0915, PLR0917, TRY003

from __future__ import annotations

import argparse
import bisect
import collections
import dataclasses
import datetime as dt
import fcntl
import hashlib
import http.client
import importlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import urllib.parse
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCHEMA = "udlex-opensearch-benchmark/v1"
SOURCES = {
    "default": "https://docs.opensearch.org/latest/analyzers/supported-analyzers/",
    "languages": "https://docs.opensearch.org/latest/analyzers/language-analyzers/",
    "stemmer": "https://docs.opensearch.org/latest/analyzers/token-filters/stemmer/",
    "analyze": "https://docs.opensearch.org/latest/api-reference/analyze-apis/",
}
# These are installed-in-core analyzers, not optional analysis plugins.
LANGUAGE_ANALYZERS = {
    "ara": "arabic",
    "arb": "arabic",
    "hye": "armenian",
    "eus": "basque",
    "ben": "bengali",
    "bul": "bulgarian",
    "cat": "catalan",
    "ces": "czech",
    "dan": "danish",
    "nld": "dutch",
    "eng": "english",
    "est": "estonian",
    "fin": "finnish",
    "fra": "french",
    "glg": "galician",
    "deu": "german",
    "ell": "greek",
    "hin": "hindi",
    "hun": "hungarian",
    "ind": "indonesian",
    "gle": "irish",
    "ita": "italian",
    "lav": "latvian",
    "lit": "lithuanian",
    "nob": "norwegian",
    "nor": "norwegian",
    "fas": "persian",
    "pes": "persian",
    "por": "portuguese",
    "ron": "romanian",
    "rus": "russian",
    "ckb": "sorani",
    "spa": "spanish",
    "swe": "swedish",
    "tha": "thai",
    "tur": "turkish",
    "zho": "cjk",
    "cmn": "cjk",
    "yue": "cjk",
    "jpn": "cjk",
    "kor": "cjk",
}
STEMMERS = {
    key: value
    for key, value in LANGUAGE_ANALYZERS.items()
    if value not in {"thai", "cjk"}
}
STEMMERS["nno"] = "light_nynorsk"


def canonical(value: Any) -> bytes:
    """Return stable UTF-8 JSON suitable for fingerprints."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def digest(value: Any) -> str:
    """Hash a JSON-compatible value without depending on mapping order."""
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: Path) -> str:
    """Stream a runtime artifact into a SHA-256 digest."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """Publish one complete checkpoint after fsync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def utc_now() -> str:
    """Return a timezone-aware timestamp for event records."""
    return dt.datetime.now(dt.UTC).isoformat()


def event(output: Path, kind: str, **values: Any) -> None:
    """Persist and print one durable progress event."""
    record = {"time": utc_now(), "event": kind, **values}
    with (output / "events.jsonl").open("ab") as stream:
        stream.write(canonical(record) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(record, ensure_ascii=False), flush=True)


class HTTPError(RuntimeError):
    """Preserve the status and bounded server error for failed requests."""

    def __init__(self, status: int, method: str, path: str, payload: Any) -> None:
        """Initialize one connection or structured failure."""
        self.status = status
        super().__init__(f"HTTP {status} {method} {path}: {str(payload)[:2000]}")


class HTTP:
    """One reusable stdlib connection; elapsed time includes encode/decode."""

    def __init__(self, endpoint: str, timeout: float = 120) -> None:
        """Initialize one connection or structured failure."""
        url = urllib.parse.urlsplit(endpoint)
        if url.scheme not in {"http", "https"} or not url.hostname:
            raise ValueError("endpoint must be an http(s) URL")
        if url.username or url.password or url.query or url.fragment:
            raise ValueError(
                "use an isolated local endpoint without embedded credentials"
            )
        self.endpoint = endpoint.rstrip("/")
        self.prefix = url.path.rstrip("/")
        cls = (
            http.client.HTTPSConnection
            if url.scheme == "https"
            else http.client.HTTPConnection
        )
        self.connection = cls(url.hostname, url.port, timeout=timeout)

    def request(
        self, method: str, path: str, body: Any = None, allowed: tuple[int, ...] = ()
    ) -> tuple[Any, float]:
        """Make a replay-safe request and measure the full HTTP round trip."""
        started = time.perf_counter()
        raw = None if body is None else canonical(body)
        # Retry only replay-safe benchmark operations. Index bodies and document
        # IDs are fixed; a retry never creates another document.
        for attempt in range(4):
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
                if response.status in {429, 502, 503, 504} and attempt < 3:
                    self.connection.close()
                    time.sleep(0.5 * 2**attempt)
                    continue
                if response.status >= 400 and response.status not in allowed:
                    raise HTTPError(response.status, method, path, parsed)
                return parsed, (time.perf_counter() - started) * 1000
            except (OSError, http.client.HTTPException):
                self.connection.close()
                if attempt == 3:
                    raise
                time.sleep(0.5 * 2**attempt)
        raise AssertionError("unreachable retry state")

    def info(self) -> dict[str, Any]:
        """Collect stable server version, plugin and resource configuration."""
        root, _ = self.request("GET", "/")
        nodes, _ = self.request("GET", "/_nodes/plugins")
        runtime, _ = self.request("GET", "/_nodes/os,jvm")
        plugins = [
            sorted(
                [
                    {key: p.get(key) for key in ("name", "version", "classname")}
                    for p in node.get("plugins", [])
                ],
                key=lambda p: p["name"],
            )
            for node in nodes.get("nodes", {}).values()
        ]
        environments = []
        for node in runtime.get("nodes", {}).values():
            jvm, operating_system = node.get("jvm", {}), node.get("os", {})
            environments.append(
                {
                    "os": {
                        key: operating_system.get(key)
                        for key in (
                            "name",
                            "arch",
                            "version",
                            "available_processors",
                            "allocated_processors",
                        )
                    },
                    "jvm": {
                        key: jvm.get(key)
                        for key in ("version", "vm_name", "vm_version", "vm_vendor")
                    },
                    "jvm_memory": {
                        key: jvm.get("mem", {}).get(key)
                        for key in ("heap_init_in_bytes", "heap_max_in_bytes")
                    },
                    "jvm_resource_options": sorted(
                        arg
                        for arg in jvm.get("input_arguments", [])
                        if arg.startswith(
                            (
                                "-Xms",
                                "-Xmx",
                                "-XX:ActiveProcessorCount=",
                                "-XX:ParallelGCThreads=",
                                "-XX:ConcGCThreads=",
                            )
                        )
                    ),
                }
            )
        return {
            "version": root.get("version"),
            "plugins_per_node": plugins,
            "runtime_per_node": sorted(environments, key=digest),
        }


@dataclasses.dataclass(frozen=True)
class Analyzer:
    """Describe an exact analysis pipeline and its runtime index."""

    name: str
    endpoint: str
    body: dict[str, Any]
    kind: str
    single_token: bool = False
    index: str | None = None


def configure_indices(
    clients: dict[str, HTTP], specs: list[Analyzer], language: str, run_key: str
) -> list[Analyzer]:
    """Install named analyzers, reusing factories and native dictionary handles."""
    configured = []
    for endpoint, client in clients.items():
        index = f"udlex-analysis-{run_key[:10]}-{language}-{endpoint}"
        filters: dict[str, Any] = {}
        definitions: dict[str, Any] = {}
        for spec in specs:
            if spec.endpoint != endpoint:
                continue
            if "analyzer" in spec.body:
                definition = {"type": spec.body["analyzer"]}
            else:
                names = []
                for number, item in enumerate(spec.body.get("filter", [])):
                    if isinstance(item, str):
                        names.append(item)
                    else:
                        name = f"{spec.name}_filter_{number}"
                        filters[name] = item
                        names.append(name)
                definition = {
                    "type": "custom",
                    "tokenizer": spec.body["tokenizer"],
                    "filter": names,
                }
            definitions[f"bench_{spec.name}"] = definition
            configured.append(dataclasses.replace(spec, index=index))
        settings = {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "index.analyze.max_token_count": 100000,
            "analysis": {"analyzer": definitions, "filter": filters},
        }
        client.request("DELETE", f"/{index}", allowed=(404,))
        client.request("PUT", f"/{index}", {"settings": settings})
    return configured


def analyzers(
    language: str, plugin_bundle: str, use_rules: bool, *, config: str | None = None
) -> list[Analyzer]:
    """Select native controls and udlex pipelines for one canonical ISO code."""
    udlex = {
        "type": "udlex",
        "language": language,
        "bundle_path": plugin_bundle,
        "use_rules": use_rules,
    }
    result = [
        Analyzer("standard", "stock", {"analyzer": "standard"}, "default_analyzer"),
        Analyzer(
            "lowercase",
            "stock",
            {"tokenizer": "keyword", "filter": ["lowercase"]},
            "morphology_control",
            True,
        ),
    ]
    if language in LANGUAGE_ANALYZERS:
        result.append(
            Analyzer(
                "builtin",
                "stock",
                {"analyzer": LANGUAGE_ANALYZERS[language]},
                "language_analyzer",
            )
        )
    if language in STEMMERS:
        stemmer = (
            "light_nynorsk"
            if language == "nor" and config in {"no_nynorsk", "no_nynorsklia"}
            else STEMMERS[language]
        )
        lower = (
            {
                "type": "lowercase",
                "language": {"ell": "greek", "gle": "irish", "tur": "turkish"}[
                    language
                ],
            }
            if language in {"ell", "gle", "tur"}
            else "lowercase"
        )
        # Explicit morphology-only algorithm control, NOT a clone of the
        # built-in analyzer: tokenization, stopwords and normalizers differ.
        result.append(
            Analyzer(
                "keyword_stemmer",
                "stock",
                {
                    "tokenizer": "keyword",
                    "filter": [
                        lower,
                        {"type": "stemmer", "language": stemmer},
                    ],
                },
                "stemmer_algorithm_no_stopwords",
                True,
            )
        )
    result.extend(
        [
            Analyzer(
                "udlex",
                "plugin",
                {"tokenizer": "keyword", "filter": [udlex]},
                "lemma",
                True,
            ),
            Analyzer(
                "udlex_lowercase",
                "plugin",
                {"tokenizer": "keyword", "filter": [udlex, "lowercase"]},
                "lemma_search_normalized",
                True,
            ),
        ]
    )
    return result


def utf16_length(text: str) -> int:
    """Count the Java/Lucene offset units in a string."""
    return len(text.encode("utf-16-le")) // 2


def align_array(forms: list[str], tokens: list[dict[str, Any]]) -> list[list[str]]:
    """Align original UTF-16 offsets, including stopword/empty-token gaps.

    This convention is never trusted without verify_alignment comparing a
    multi-value request with independent requests on the live server.
    """
    starts, ends, offset = [], [], 0
    for form in forms:
        starts.append(offset)
        offset += utf16_length(form)
        ends.append(offset)
        offset += 1
    outputs: list[list[str]] = [[] for _ in forms]
    for token in tokens:
        start, end = token["start_offset"], token["end_offset"]
        index = bisect.bisect_right(starts, start) - 1
        if index < 0 or end > ends[index] or start > end or start > ends[index]:
            raise ValueError(f"token offset does not fit one input: {token}")
        outputs[index].append(token["token"])
    return outputs


def analyze(
    client: HTTP, spec: Analyzer, text: str | list[str]
) -> tuple[list[dict[str, Any]], float]:
    """Execute the same named-analyzer HTTP path for every system."""
    body = {"analyzer": f"bench_{spec.name}"} if spec.index else spec.body
    path = f"/{spec.index}/_analyze" if spec.index else "/_analyze"
    response, elapsed = client.request("POST", path, {**body, "text": text})
    if not isinstance(response.get("tokens"), list):
        raise TypeError("_analyze did not return a tokens array")
    return response["tokens"], elapsed


def verify_alignment(client: HTTP, spec: Analyzer, forms: list[str]) -> dict[str, Any]:
    """Validate batching against independent requests on the live server."""
    probes = list(
        dict.fromkeys(
            [
                "Axes",
                "the",
                "Мышами",
                "ice-cream",
                "can't",
                "東京",
                "😀foo",
                "!",
                "a b",
                "A\u0301",
                *forms[:12],
            ]
        )
    )
    batched, _ = analyze(client, spec, probes)
    singles = [[t["token"] for t in analyze(client, spec, form)[0]] for form in probes]
    try:
        matched = align_array(probes, batched) == singles
    except (KeyError, TypeError, ValueError):
        matched = False
    if spec.single_token and any(len(tokens) != 1 for tokens in singles):
        raise ValueError(
            f"{spec.name} violated one-input/one-output on alignment probes"
        )
    return {
        "mode": "utf16_offsets" if matched else "individual_requests",
        "probes": len(probes),
        "validated_against_single_requests": True,
    }


def read_cache(
    path: Path, fingerprint: str, forms: list[str]
) -> tuple[list[list[str]], list[float]]:
    """Recover complete chunks, truncating only an interrupted final write."""
    if not path.exists():
        return [], []
    predictions: list[list[str]] = []
    durations: list[float] = []
    good_bytes = 0
    with path.open("rb") as stream:
        for line_index, raw in enumerate(stream):
            if not raw.endswith(b"\n"):
                break
            item = json.loads(raw)
            if line_index == 0:
                if item != {"schema": SCHEMA, "fingerprint": fingerprint}:
                    raise ValueError(f"cache fingerprint mismatch: {path}")
            else:
                start, count = item["start"], len(item["predictions"])
                if start != len(predictions) or item["forms_sha256"] != digest(
                    forms[start : start + count]
                ):
                    raise ValueError(f"cache order or input mismatch: {path}")
                if not all(
                    isinstance(pred, list) and all(isinstance(s, str) for s in pred)
                    for pred in item["predictions"]
                ):
                    raise ValueError(f"invalid predictions cache: {path}")
                predictions.extend(item["predictions"])
                durations.extend(item["request_ms"])
            good_bytes += len(raw)
    if path.stat().st_size != good_bytes:
        with path.open("r+b") as stream:
            stream.truncate(good_bytes)
    if len(predictions) > len(forms):
        raise ValueError(f"cache has too many predictions: {path}")
    return predictions, durations


def predict(
    client: HTTP,
    spec: Analyzer,
    forms: list[str],
    path: Path,
    fingerprint: str,
    batch_size: int,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Resume durable unique-form chunks without discarding token multiplicity."""
    alignment = verify_alignment(client, spec, forms)
    key = digest(
        {
            "cell": fingerprint,
            "analyzer": dataclasses.asdict(spec),
            "alignment": alignment,
        }
    )
    predictions, durations = read_cache(path, key, forms)
    resumed = len(predictions)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        if stream.tell() == 0:
            stream.write(canonical({"schema": SCHEMA, "fingerprint": key}) + b"\n")
        for start in range(resumed, len(forms), batch_size):
            chunk = forms[start : start + batch_size]
            if alignment["mode"] == "utf16_offsets":
                tokens, elapsed = analyze(client, spec, chunk)
                values, elapsed_values = align_array(chunk, tokens), [elapsed]
            else:
                responses = [analyze(client, spec, form) for form in chunk]
                values = [[t["token"] for t in tokens] for tokens, _ in responses]
                elapsed_values = [elapsed for _, elapsed in responses]
            if spec.single_token and any(len(value) != 1 for value in values):
                raise ValueError(
                    f"{spec.name} violated one-input/one-output at {start}"
                )
            record = {
                "start": start,
                "forms_sha256": digest(chunk),
                "predictions": values,
                "request_ms": elapsed_values,
            }
            stream.write(canonical(record) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
            predictions.extend(values)
            durations.extend(elapsed_values)
    return dict(zip(forms, predictions, strict=True)), {
        "alignment": alignment,
        "unique_forms": len(forms),
        "resumed_forms": resumed,
        "http_requests": len(durations),
        "http_total_ms": sum(durations),
        "note": "collection timings include first calls and retries; use performance for warmed measurements",
    }


def choose2(count: int) -> int:
    """Count unordered pairs of distinct members."""
    return count * (count - 1) // 2


def pair_metrics(
    forms: list[str],
    lemmas: list[str],
    lexical: list[bool],
    predictions: dict[str, list[str]],
) -> dict[str, Any]:
    """Count all unordered type pairs algebraically without sampling negatives.

    Each exact surface form occurs once; exclude forms with multiple casefolded
    gold lemmas. Distinct forms are positive iff their casefolded gold lemma is
    equal. Predicted positives require an identical, nonempty token sequence.
    Do not casefold predictions: the actual analyzer defines search equality.
    """
    gold: dict[str, set[str]] = collections.defaultdict(set)
    for form, lemma, keep in zip(forms, lemmas, lexical, strict=True):
        if keep:
            gold[form].add(lemma.casefold())
    selected = {
        form: next(iter(values)) for form, values in gold.items() if len(values) == 1
    }
    gold_counts = collections.Counter(selected.values())
    pred_counts: collections.Counter[tuple[str, ...]] = collections.Counter()
    joint: collections.Counter[tuple[str, tuple[str, ...]]] = collections.Counter()
    dropped = 0
    for form, lemma in selected.items():
        key = tuple(predictions[form])
        if not key:
            dropped += 1
            continue  # absent terms never form a false "all stopwords" cluster
        pred_counts[key] += 1
        joint[(lemma, key)] += 1
    gold_pairs = sum(choose2(n) for n in gold_counts.values())
    predicted_pairs = sum(choose2(n) for n in pred_counts.values())
    tp = sum(choose2(n) for n in joint.values())
    precision = tp / predicted_pairs if predicted_pairs else None
    recall = tp / gold_pairs if gold_pairs else None
    f1 = (
        2 * tp / (predicted_pairs + gold_pairs)
        if predicted_pairs + gold_pairs
        else None
    )
    return {
        "forms": len(selected),
        "ambiguous_forms_excluded": len(gold) - len(selected),
        "dropped_forms": dropped,
        "gold_pairs": gold_pairs,
        "predicted_pairs": predicted_pairs,
        "true_positive_pairs": tp,
        "false_positive_pairs": predicted_pairs - tp,
        "false_negative_pairs": gold_pairs - tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def score(corpus: dict[str, Any], predictions: dict[str, list[str]]) -> dict[str, Any]:
    """Report lemma agreement, coverage and type-pair conflation separately."""

    def subset(mask: list[bool]) -> dict[str, Any]:
        total = singleton = exact = folded = dropped = split = 0
        for form, lemma, keep in zip(
            corpus["forms"], corpus["lemmas"], mask, strict=True
        ):
            if not keep:
                continue
            total += 1
            value = predictions[form]
            if len(value) == 1:
                singleton += 1
                exact += value[0] == lemma
                folded += value[0].casefold() == lemma.casefold()
            elif not value:
                dropped += 1
            else:
                split += 1
        return {
            "tokens": total,
            "single_output_tokens": singleton,
            "dropped_tokens": dropped,
            "split_tokens": split,
            "exact_matches": exact,
            "casefold_matches": folded,
            "exact_all": exact / total if total else None,
            "casefold_all": folded / total if total else None,
            "exact_single_output": exact / singleton if singleton else None,
            "casefold_single_output": folded / singleton if singleton else None,
            "single_output_coverage": singleton / total if total else None,
        }

    lexical = corpus.get(
        "lexical_mask", [p not in {"PUNCT", "SYM"} for p in corpus["upos"]]
    )
    return {
        "all_tokens": subset([True] * len(corpus["forms"])),
        "lexical_tokens": subset(lexical),
        "pairs": pair_metrics(corpus["forms"], corpus["lemmas"], lexical, predictions),
    }


def performance(
    client: HTTP, spec: Analyzer, forms: list[str], repeats: int
) -> dict[str, Any]:
    """Measure a bounded, warmed HTTP workload on the same input forms."""
    sample = forms[: min(256, len(forms))]
    if not sample or repeats == 0:
        return {"status": "not_requested"}
    for _ in range(2):
        analyze(client, spec, sample)
    timings, counts = [], []
    for _ in range(repeats):
        tokens, elapsed = analyze(client, spec, sample)
        timings.append(elapsed)
        counts.append(len(tokens))
    ordered = sorted(timings)
    median = statistics.median(timings)
    return {
        "status": "ok",
        "kind": "HTTP_analyze_roundtrip_encode_to_decode",
        "warmup_requests": 2,
        "repeats": repeats,
        "input_forms_per_request": len(sample),
        "output_tokens_per_request": counts,
        "request_ms": timings,
        "median_ms": median,
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "input_forms_per_second": len(sample) * 1000 / median,
        "concurrency": 1,
        "note": "batched local HTTP; not isolated core throughput or capacity",
    }


def python_parity(
    bundle: Path,
    language: str,
    forms: list[str],
    predictions: dict[str, list[str]],
    use_rules: bool,
    required: bool,
) -> dict[str, Any]:
    """Check every unique form against the matching Python core flags."""
    try:
        udlex = importlib.import_module("udlex")
    except ImportError as exc:
        if required:
            raise RuntimeError(
                "Python extension unavailable for required native parity"
            ) from exc
        return {"status": "unavailable", "reason": str(exc)}
    lexicon = udlex.load(
        str(bundle), language, use_rules=use_rules, use_entities=False, use_user=False
    )
    mismatches, examples = 0, []
    started = time.perf_counter()
    for start in range(0, len(forms), 1024):
        chunk = forms[start : start + 1024]
        expected = lexicon.lemmatize(
            chunk, sentence_initial=[False] * len(chunk), context=False
        )
        for form, value in zip(chunk, expected, strict=True):
            if predictions[form] != [value]:
                mismatches += 1
                if len(examples) < 20:
                    examples.append(
                        {
                            "form": form,
                            "native_plugin": predictions[form],
                            "python": value,
                        }
                    )
    return {
        "status": "ok" if not mismatches else "failed",
        "unique_forms": len(forms),
        "mismatches": mismatches,
        "examples": examples,
        "python_core_version": udlex.__version__,
        "flags": {
            "sentence_initial": False,
            "context": False,
            "entities": False,
            "use_rules": use_rules,
            "use_user": False,
        },
        "elapsed_ms": (time.perf_counter() - started) * 1000,
    }


def sentence_smoke(
    clients: dict[str, HTTP],
    language: str,
    corpus: dict[str, Any],
    run_key: str,
    plugin_bundle: str,
    use_rules: bool,
    count: int,
    keep_indices: bool,
) -> dict[str, Any]:
    """Test real index/search/highlight behavior, preserving raw source text."""
    sentences = [
        s
        for s in corpus["sentences"]
        if len(s["tokens"]) >= 2
        and any(any(c.isalpha() for c in t) for t in s["tokens"])
    ][:count]
    if not sentences:
        return {"status": "not_requested_or_no_sentence"}
    results = []
    udlex_filter = {
        "type": "udlex",
        "language": language,
        "bundle_path": plugin_bundle,
        "use_rules": use_rules,
    }
    for endpoint, client in clients.items():
        index = f"udlex-bench-{run_key[:10]}-{language}-{endpoint}"
        analysis: dict[str, Any] = {"analyzer": {}}
        fields = {"standard": {"type": "text", "analyzer": "standard"}}
        if language in LANGUAGE_ANALYZERS:
            fields["builtin"] = {
                "type": "text",
                "analyzer": LANGUAGE_ANALYZERS[language],
            }
        if endpoint == "plugin":
            analysis = {
                "filter": {"lemma": udlex_filter},
                "analyzer": {
                    "lemma_text": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lemma", "lowercase"],
                    }
                },
            }
            fields["udlex"] = {"type": "text", "analyzer": "lemma_text"}
        settings = {"number_of_shards": 1, "number_of_replicas": 0}
        if analysis["analyzer"]:
            settings["analysis"] = analysis
        client.request("DELETE", f"/{index}", allowed=(404,))
        client.request(
            "PUT",
            f"/{index}",
            {"settings": settings, "mappings": {"properties": fields}},
        )
        try:
            for number, sentence in enumerate(sentences):
                raw = sentence.get("text") or " ".join(sentence["tokens"])
                _, index_ms = client.request(
                    "PUT",
                    f"/{index}/_doc/{number}?refresh=true",
                    dict.fromkeys(fields, raw),
                )
                for field in fields:
                    analyzed, analyze_ms = client.request(
                        "POST", f"/{index}/_analyze", {"field": field, "text": raw}
                    )
                    tokens = analyzed["tokens"]
                    tests = {}
                    for query_type in ("match", "match_phrase"):
                        query = {
                            "query": {query_type: {field: raw}},
                            "size": len(sentences) + 1,
                            "highlight": {
                                "fields": {field: {"number_of_fragments": 0}}
                            },
                        }
                        response, search_ms = client.request(
                            "POST", f"/{index}/_search", query
                        )
                        hits = response["hits"]["hits"]
                        own = next(
                            (hit for hit in hits if hit["_id"] == str(number)), None
                        )
                        tests[query_type] = {
                            "found_document": own is not None,
                            "highlight": own.get("highlight", {}).get(field, [])
                            if own
                            else [],
                            "latency_ms": search_ms,
                        }
                        if tokens and (
                            own is None or not tests[query_type]["highlight"]
                        ):
                            raise ValueError(
                                f"{endpoint}/{language}/{field}: {query_type}/highlight smoke failed"
                            )
                    # The plugin may change only terms, never original offsets or positions.
                    offset_check = None
                    if field == "udlex":
                        before, _ = client.request(
                            "POST",
                            f"/{index}/_analyze",
                            {"tokenizer": "standard", "text": raw},
                        )

                        def attributes(
                            ts: list[dict[str, Any]],
                        ) -> list[tuple[int, int, int]]:
                            return [
                                (t["start_offset"], t["end_offset"], t["position"])
                                for t in ts
                            ]

                        offset_check = attributes(before["tokens"]) == attributes(
                            tokens
                        )
                        if not offset_check:
                            raise ValueError(
                                f"{language}: udlex changed offsets/positions"
                            )
                    results.append(
                        {
                            "endpoint": endpoint,
                            "field": field,
                            "sentence_id": sentence.get("id", sentence.get("sent_id")),
                            "text_source": sentence.get("text_source", "unknown"),
                            "text": raw,
                            "tokens": tokens,
                            "index_ms": index_ms,
                            "analyze_ms": analyze_ms,
                            "queries": tests,
                            "offset_position_parity": offset_check,
                            "no_search_terms": not tokens,
                        }
                    )
        finally:
            if not keep_indices:
                client.request("DELETE", f"/{index}", allowed=(404,))
    return {
        "status": "ok",
        "sentences": len(sentences),
        "cases": results,
        "note": "same-text functional smoke; not evidence of relevance improvement",
    }


def source_identity(root: Path) -> dict[str, Any]:
    """Fingerprint core, wrapper and evaluation code including native bridges."""
    files = {}
    for folder, suffixes in (
        ("rust", {".rs", ".toml", ".lock"}),
        (
            "integrations/opensearch",
            {".py", ".rs", ".java", ".cpp", ".h", ".sh", ".policy", ".toml", ".xml"},
        ),
        ("integrations/go/capi", {".rs", ".toml", ".lock"}),
        ("src/udlex", {".py", ".so", ".pyd"}),
    ):
        for path in sorted((root / folder).rglob("*")):
            if (
                path.is_file()
                and path.suffix in suffixes
                and not {"target", "build", "__pycache__"}.intersection(path.parts)
            ):
                files[str(path.relative_to(root))] = file_sha(path)
    git = shutil.which("git")
    commit = (
        subprocess.run(  # noqa: S603 -- resolved git executable and fixed arguments
            [git, "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip()
        if git
        else ""
    )
    if not commit and (root / "SOURCE_COMMIT").exists():
        commit = (root / "SOURCE_COMMIT").read_text().strip()
    return {"commit": commit, "files_sha256": digest(files), "files": files}


def installed_plugin_identity(path: Path | None) -> dict[str, Any] | None:
    """Hash the exact installed files independently of plugin version labels."""
    if path is None:
        return None
    if not path.is_dir():
        raise ValueError("--installed-plugin-dir must be a readable directory")
    files = {
        str(file.relative_to(path)): file_sha(file)
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }
    if not files:
        raise ValueError("--installed-plugin-dir contains no files")
    return {"files": files, "sha256": digest(files)}


def load_datasets_module() -> Any:
    # Direct script invocation must not accidentally import Hugging Face datasets.
    """Import this repository's reader without a third-party name collision."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    return importlib.import_module("integrations.opensearch.datasets")


def run_language(
    args: argparse.Namespace,
    entry: dict[str, Any],
    datasets: Any,
    clients: dict[str, HTTP],
    manifest_key: str,
) -> dict[str, Any]:
    """Evaluate and checkpoint one language, with explicit unsupported cases."""
    language = entry["language"]
    if entry.get("status") == "no_scorable_gold":
        return {"language": language, "status": "no_scorable_gold", "entry": entry}
    if entry.get("status") != "available":
        if entry.get("status") in {
            "unsupported_bundle",
            "no_gold_metadata",
            "no_test_metadata",
        }:
            return {
                "language": language,
                "status": "unavailable",
                "entry": entry,
                "reason": entry.get("status"),
            }
        raise ValueError(f"{language}: inventory status {entry.get('status')}")
    corpus = datasets.load_corpus(entry, max_tokens=args.max_tokens)
    forms = list(dict.fromkeys(corpus["forms"]))
    if not forms:
        return {
            "language": language,
            "status": "no_scorable_gold",
            "entry": entry,
            "gold_tokens": 0,
            "original_gold_tokens": corpus.get("corpus_counts", {}).get(
                "scorable_tokens"
            ),
            "exclusions": {
                key: corpus.get(key, 0)
                for key in (
                    "excluded_sentences",
                    "excluded_tokens",
                    "excluded_scorable_tokens",
                )
            },
            "lexical_gold_tokens": 0,
            "quality_scope": "no_lexical_gold_after_exclusions",
        }
    fingerprint = digest(
        {
            "manifest": manifest_key,
            "language": language,
            "gold": corpus.get("gold_sha256", corpus.get("sha256")),
            "selection": corpus.get("selection_sha256"),
            "forms": digest(forms),
        }
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "language": language,
        "entry": entry,
        "fingerprint": fingerprint,
        "status": "running",
        "started_at": utc_now(),
        "gold_tokens": len(corpus["forms"]),
        "original_gold_tokens": corpus.get("corpus_counts", {}).get("scorable_tokens"),
        "exclusions": {
            key: corpus.get(key, 0)
            for key in (
                "excluded_sentences",
                "excluded_tokens",
                "excluded_scorable_tokens",
            )
        },
        "unique_forms": len(forms),
        "lexical_gold_tokens": sum(corpus["lexical_mask"]),
        "quality_scope": "morphology"
        if any(corpus["lexical_mask"])
        else "punctuation_only_no_lexical_gold",
        "sentences": len(corpus["sentences"]),
        "systems": {},
        "normalization_notes": [
            (
                "nor: the corpus is Nynorsk; shipped builtin norwegian uses Bokmål; "
                "keyword_stemmer uses light_nynorsk to provide a language-matched algorithm control."
            )
        ]
        if language == "nor" and entry.get("config") in {"no_nynorsk", "no_nynorsklia"}
        else [],
    }
    identity = {form: [form] for form in forms}
    result["systems"]["identity"] = {
        "status": "ok",
        "kind": "identity",
        "metrics": score(corpus, identity),
    }
    specs = configure_indices(
        clients,
        analyzers(
            language, args.plugin_bundle, not args.no_rules, config=entry.get("config")
        ),
        language,
        manifest_key,
    )
    if language not in LANGUAGE_ANALYZERS:
        result["systems"]["builtin"] = {
            "status": "unavailable",
            "reason": "no matching core language analyzer; optional plugins were not installed",
        }
    if language not in STEMMERS:
        result["systems"]["keyword_stemmer"] = {
            "status": "unavailable",
            "reason": "no corresponding core stemmer algorithm configured",
        }
    try:
        for spec in specs:
            event(args.output, "system_started", language=language, system=spec.name)
            path = args.output / "cache" / language / f"{spec.name}.jsonl"
            values, collection = predict(
                clients[spec.endpoint], spec, forms, path, fingerprint, args.batch_size
            )
            system = {
                "status": "ok",
                "kind": spec.kind,
                "analyzer": spec.body,
                "metrics": score(corpus, values),
                "collection": collection,
                "performance": performance(
                    clients[spec.endpoint], spec, forms, args.repeats
                ),
            }
            result["systems"][spec.name] = system
            if spec.name == "udlex":
                result["parity"] = python_parity(
                    args.bundle,
                    language,
                    forms,
                    values,
                    not args.no_rules,
                    args.require_parity,
                )
            event(
                args.output,
                "system_completed",
                language=language,
                system=spec.name,
                pair_f1=system["metrics"]["pairs"]["f1"],
                exact=system["metrics"]["all_tokens"]["exact_all"],
            )
    finally:
        if not args.keep_indices:
            for endpoint, index in {(spec.endpoint, spec.index) for spec in specs}:
                clients[endpoint].request("DELETE", f"/{index}", allowed=(404,))
    result["smoke"] = sentence_smoke(
        clients,
        language,
        corpus,
        manifest_key,
        args.plugin_bundle,
        not args.no_rules,
        args.smoke_sentences,
        args.keep_indices,
    )
    result["status"] = (
        "ok" if result.get("parity", {}).get("status") != "failed" else "parity_failed"
    )
    result["finished_at"] = utc_now()
    return result


def make_parser() -> argparse.ArgumentParser:
    """Build the reproducible stand runner's command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=ROOT, help="repository/data snapshot root"
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="same frozen bundle mounted into the plugin",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclusions",
        type=Path,
        help="verified fit/test overlap audit receipt; original gold stays unchanged",
    )
    parser.add_argument("--stock-url", default="http://127.0.0.1:19200")
    parser.add_argument("--plugin-url", default="http://127.0.0.1:19201")
    parser.add_argument(
        "--plugin-bundle",
        default="udlex-bundle",
        help="bundle_path relative to OpenSearch config",
    )
    parser.add_argument(
        "--plugin-artifact",
        type=Path,
        help="installed plugin zip/native artifact to hash",
    )
    parser.add_argument(
        "--installed-plugin-dir",
        type=Path,
        help="readable exact installed plugin directory to hash",
    )
    parser.add_argument(
        "--languages", help="comma-separated ISO codes; default all inventory entries"
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--smoke-sentences", type=int, default=2)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="optional deterministic sentence-prefix pilot only",
    )
    parser.add_argument("--no-rules", action="store_true")
    parser.add_argument("--require-parity", action="store_true")
    parser.add_argument("--keep-indices", action="store_true")
    parser.add_argument("--timeout", type=float, default=120)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate inputs, resume complete languages and regenerate report artifacts."""
    args = make_parser().parse_args(argv)
    if (
        args.batch_size < 1
        or args.repeats < 0
        or args.smoke_sentences < 0
        or (args.max_tokens is not None and args.max_tokens < 1)
    ):
        raise ValueError(
            "batch-size/max-tokens must be positive; repeats/smoke-sentences nonnegative"
        )
    args.root, args.bundle, args.output = (
        args.root.resolve(),
        args.bundle.resolve(),
        args.output.resolve(),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    bundle_identity = importlib.import_module("bench.adapter_bundle").bundle_identity

    datasets = load_datasets_module()
    event(args.output, "preflight_started")
    data = datasets.inventory(args.root, bundle=args.bundle, exclusions=args.exclusions)
    if (
        data["bundle"].get("benchmark_eligible") is not True
        or data["bundle"].get("transductive_refit") is True
    ):
        raise ValueError(
            "bundle must explicitly declare benchmark_eligible=true and cannot be transductively refit"
        )
    clients = {
        "stock": HTTP(args.stock_url, args.timeout),
        "plugin": HTTP(args.plugin_url, args.timeout),
    }
    servers = {name: client.info() for name, client in clients.items()}
    if servers["stock"]["version"] != servers["plugin"]["version"]:
        raise ValueError("stock/plugin OpenSearch builds differ")
    if any(
        any("udlex" in p["name"] for p in node)
        for node in servers["stock"]["plugins_per_node"]
    ):
        raise ValueError("stock server unexpectedly contains udlex")
    if not any(
        any("udlex" in p["name"] for p in node)
        for node in servers["plugin"]["plugins_per_node"]
    ):
        raise ValueError("plugin server does not advertise a udlex plugin")
    selected = set(args.languages.split(",")) if args.languages else None
    entries = [
        entry
        for entry in data["entries"]
        if selected is None or entry["language"] in selected
    ]
    if selected and selected != {entry["language"] for entry in entries}:
        raise ValueError(
            f"unknown languages: {selected - {entry['language'] for entry in entries}}"
        )
    parameters = {
        key: getattr(args, key)
        for key in (
            "plugin_bundle",
            "batch_size",
            "repeats",
            "smoke_sentences",
            "max_tokens",
            "no_rules",
            "require_parity",
        )
    }
    stable_entries = [
        {
            key: value
            for key, value in entry.items()
            if key not in {"path", "bundle_path"}
        }
        for entry in entries
    ]
    identity = {
        "schema": SCHEMA,
        "source": source_identity(ROOT),
        "bundle": bundle_identity(args.bundle),
        "servers": servers,
        "parameters": parameters,
        "inventory": stable_entries,
        "python": platform.python_version(),
        "plugin_artifact_sha256": file_sha(args.plugin_artifact)
        if args.plugin_artifact
        else None,
        "installed_plugin": installed_plugin_identity(args.installed_plugin_dir),
        "exclusion_audit_sha256": file_sha(args.exclusions)
        if args.exclusions
        else None,
    }
    fingerprint = digest(identity)
    manifest_path = args.output / "manifest.json"
    if (
        manifest_path.exists()
        and json.loads(manifest_path.read_text())["fingerprint"] != fingerprint
    ):
        raise ValueError(
            "run fingerprint changed: use another --output directory; old results are preserved"
        )
    if not manifest_path.exists():
        atomic_json(
            manifest_path,
            {
                "fingerprint": fingerprint,
                "identity": identity,
                "started_at": utc_now(),
                "endpoints": {"stock": args.stock_url, "plugin": args.plugin_url},
                "inventory": data,
                "methodology": {
                    "sources": SOURCES,
                    "pair_unit": "distinct lexical surface types, unambiguous casefold gold lemma",
                    "pair_equality": "identical nonempty analyzer token sequence, no extra folding",
                    "gold_lexical_excludes": ["PUNCT", "SYM"],
                    "stopwords": "dropped separately; never clustered together",
                    "lemma_metrics": "all and single-output denominators explicit; stemming is not lemmatization",
                    "context": "sentence_initial=false, context=false, no gold POS passed",
                    "scope": "UD morphology/conflation proxy and functional search smoke, not a qrels IR benchmark",
                },
            },
        )
    failed = 0
    for entry in entries:
        language = entry["language"]
        path = args.output / "languages" / f"{language}.json"
        if path.exists():
            previous = json.loads(path.read_text())
            if previous.get("manifest_fingerprint") != fingerprint:
                raise ValueError(f"language manifest mismatch: {path}")
            if previous.get("status") in {"ok", "no_scorable_gold", "unavailable"}:
                event(args.output, "language_resumed_complete", language=language)
                continue
        event(
            args.output,
            "language_started",
            language=language,
            inventory_status=entry.get("status"),
        )
        try:
            result = run_language(args, entry, datasets, clients, fingerprint)
        except Exception as exc:  # noqa: BLE001 -- persist failures and continue other languages
            failed += 1
            result = {
                "schema": SCHEMA,
                "language": language,
                "entry": entry,
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "finished_at": utc_now(),
            }
            event(args.output, "language_failed", language=language, error=str(exc))
        else:
            if result["status"] not in {"ok", "no_scorable_gold", "unavailable"}:
                failed += 1
            event(
                args.output,
                "language_completed",
                language=language,
                status=result["status"],
            )
        result["manifest_fingerprint"] = fingerprint
        atomic_json(path, result)
        report = importlib.import_module("integrations.opensearch.report")
        report.generate(args.output)
    event(args.output, "run_finished", languages=len(entries), failures=failed)
    importlib.import_module("integrations.opensearch.report").generate(args.output)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
