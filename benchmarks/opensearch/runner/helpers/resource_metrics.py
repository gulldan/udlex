"""Read-only Linux/JVM measurements for sequential OpenSearch benchmark profiles.

CPU is the server PID's user+system time across its threads, excluding children
and the benchmark client. 100% means one logical CPU; allocated percent divides
by the PID's observed affinity size, not host CPU count or a cgroup quota.

RSS includes resident file mappings; PSS apportions shared pages. Neither is JVM
heap. Sampled peaks are lower bounds; means use trapezoidal time weighting.
Loaded dictionaries accumulate across languages. Compare startup/post-common/
final suite snapshots, not independent per-language memory costs. This module
never changes affinity, invokes GC, drops caches, or reclaims kernel page cache.

References: https://docs.kernel.org/filesystems/proc.html and
https://docs.opensearch.org/latest/api-reference/nodes-apis/nodes-stats/
"""

# Heterogeneous receipts are JSON; explicit errors describe failed boundaries.
# ruff: noqa: ANN401, EM101, EM102, INP001, TRY003

from __future__ import annotations

import math
import os
import threading
import time
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

PROC = Path("/proc")
MEMORY_FIELDS = (
    "rss_bytes",
    "pss_bytes",
    "rss_anon_bytes",
    "rss_file_bytes",
    "pss_anon_bytes",
    "pss_file_bytes",
    "swap_bytes",
)


class ProcessChangedError(RuntimeError):
    """The measured process exited, was replaced, or changed its CPU affinity."""


class HTTPClient(Protocol):
    """The existing benchmark HTTP client returns payload plus elapsed time."""

    def request(self, method: str, path: str) -> tuple[dict[str, Any], float]:
        """Return one JSON response and its request duration."""
        ...


def parse_proc_stat(text: str) -> dict[str, Any]:
    """Parse /proc/PID/stat without splitting a command containing spaces or ')'."""
    opening, closing = text.find("("), text.rfind(")")
    if opening < 1 or closing <= opening:
        raise ValueError("Malformed /proc/PID/stat command field")
    fields = text[closing + 1 :].split()
    if len(fields) < 22 or len(fields[0]) != 1:
        raise ValueError("Truncated /proc/PID/stat")
    parsed = {"pid": int(text[:opening].strip()), "state": fields[0]}
    for name, offset in (
        ("user_ticks", 11),
        ("system_ticks", 12),
        ("starttime_ticks", 19),
        ("rss_pages", 21),
        ("minor_faults", 7),
        ("major_faults", 9),
    ):
        value = int(fields[offset])
        if value < 0:
            raise ValueError(f"Negative /proc/PID/stat {name}")
        parsed[name] = value
    return parsed


def parse_cpu_list(text: str) -> list[int]:
    """Expand Linux's comma-separated CPU list with inclusive ranges."""
    cpus: set[int] = set()
    for part in text.strip().split(","):
        ends = part.split("-")
        if len(ends) not in {1, 2} or not all(end.isdecimal() for end in ends):
            raise ValueError("Malformed Cpus_allowed_list")
        first, last = int(ends[0]), int(ends[-1])
        if first > last:
            raise ValueError("Reversed Cpus_allowed_list range")
        cpus.update(range(first, last + 1))
    return sorted(cpus)


def _status(path: Path) -> dict[str, str]:
    return {
        key: value.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if ":" in line
        for key, value in [line.split(":", 1)]
    }


def _kb(values: Mapping[str, str], key: str) -> int | None:
    value = values.get(key)
    if value is None:
        return None
    parts = value.split()
    if len(parts) != 2 or parts[1] != "kB" or not parts[0].isdecimal():
        raise ValueError(f"Malformed {key} memory field")
    return int(parts[0]) * 1024


def _stat(pid: int, proc_root: Path) -> dict[str, Any]:
    row = parse_proc_stat((proc_root / str(pid) / "stat").read_text(encoding="utf-8"))
    if row["pid"] != pid or row["state"] in {"Z", "X", "x"}:
        raise ProcessChangedError("Measured process exited or PID differs")
    return row


def read_process_identity(pid: int, *, proc_root: Path = PROC) -> dict[str, Any]:
    """Bind PID, boot, start time, executable inode and actual main-thread affinity.

    No argv, environment, executable path or command-line credentials are read.
    NSpid records bind a host PID to OpenSearch's PID inside a container.
    """
    if isinstance(pid, bool) or pid <= 0:
        raise ValueError("pid must be positive")
    directory = proc_root / str(pid)
    before = _stat(pid, proc_root)
    status = _status(directory / "status")
    executable = directory / "exe"
    info = executable.stat()
    name = executable.readlink().name
    cpus = parse_cpu_list(status["Cpus_allowed_list"])
    namespace_pids = [int(value) for value in status.get("NSpid", str(pid)).split()]
    if not namespace_pids or namespace_pids[0] != pid:
        raise ProcessChangedError("/proc status PID namespace identity differs")
    boot_id = (proc_root / "sys/kernel/random/boot_id").read_text().strip()
    after = _stat(pid, proc_root)
    if not boot_id or before["starttime_ticks"] != after["starttime_ticks"]:
        raise ProcessChangedError("PID was reused while reading its identity")
    return {
        "pid": pid,
        "starttime_ticks": before["starttime_ticks"],
        "boot_id": boot_id,
        "exe_name": name,
        "exe_device": info.st_dev,
        "exe_inode": info.st_ino,
        "allowed_cpus": cpus,
        "allowed_cpu_count": len(cpus),
        "namespace_pids": namespace_pids,
        "affinity_scope": "main thread; taskset affinity is inherited by JVM threads",
        "clock_ticks_per_second": os.sysconf("SC_CLK_TCK"),
    }


def _host_cpu_counters(text: str) -> dict[str, tuple[int, int]]:
    counters = {}
    for line in text.splitlines():
        values = line.split()
        if values and (
            values[0] == "cpu"
            or (values[0].startswith("cpu") and values[0][3:].isdecimal())
        ):
            ticks = [int(value) for value in values[1:9]]
            if len(ticks) < 5 or any(value < 0 for value in ticks):
                raise ValueError("Invalid host CPU counters")
            counters[values[0]] = (sum(ticks), sum(ticks) - ticks[3] - ticks[4])
    return counters


def _host_snapshot(proc_root: Path, cpus: Sequence[int]) -> dict[str, Any]:
    """Capture optional shared-host context; busy time excludes idle and iowait."""
    try:
        load = [
            float(value) for value in (proc_root / "loadavg").read_text().split()[:3]
        ]
        counters = _host_cpu_counters((proc_root / "stat").read_text())
        total, busy = counters["cpu"]
        allowed = [counters[f"cpu{cpu}"] for cpu in cpus]
        memory = _status(proc_root / "meminfo")
        return {
            "status": "ok",
            "load_1m": load[0],
            "load_5m": load[1],
            "load_15m": load[2],
            "cpu_total_ticks": total,
            "cpu_busy_ticks": busy,
            "allowed_cpu_total_ticks": sum(pair[0] for pair in allowed),
            "allowed_cpu_busy_ticks": sum(pair[1] for pair in allowed),
            "mem_available_bytes": _kb(memory, "MemAvailable"),
            "cached_bytes": _kb(memory, "Cached"),
        }
    except (OSError, ValueError, KeyError, IndexError) as error:
        return {"status": "unavailable", "reason": type(error).__name__}


def process_snapshot(pid: int, *, proc_root: Path = PROC) -> dict[str, Any]:
    """Read one process snapshot, refusing identity changes across its reads."""
    identity = read_process_identity(pid, proc_root=proc_root)
    directory = proc_root / str(pid)
    status = _status(directory / "status")
    try:
        rollup = _status(directory / "smaps_rollup")
        if "Rss" not in rollup or "Pss" not in rollup:
            raise ValueError("smaps_rollup lacks RSS/PSS")
        rollup_status = "ok"
    except PermissionError:
        rollup, rollup_status = {}, "permission_denied"
    except FileNotFoundError:
        rollup, rollup_status = {}, "unavailable"
    stat = _stat(pid, proc_root)
    sampled_at = time.monotonic()
    if identity != read_process_identity(pid, proc_root=proc_root):
        raise ProcessChangedError("Process identity or affinity changed during sample")
    rss = _kb(rollup, "Rss") if rollup else _kb(status, "VmRSS")
    source = "smaps_rollup" if rollup else "status_approximate"
    if rss is None:
        rss, source = stat["rss_pages"] * os.sysconf("SC_PAGE_SIZE"), "stat_approximate"
    ticks = identity["clock_ticks_per_second"]
    return {
        "identity": identity,
        "monotonic_seconds": sampled_at,
        "cpu_seconds": (stat["user_ticks"] + stat["system_ticks"]) / ticks,
        "user_cpu_seconds": stat["user_ticks"] / ticks,
        "system_cpu_seconds": stat["system_ticks"] / ticks,
        "rss_bytes": rss,
        "rss_source": source,
        "pss_bytes": _kb(rollup, "Pss"),
        "rss_anon_bytes": _kb(status, "RssAnon"),
        "rss_file_bytes": _kb(status, "RssFile"),
        "pss_anon_bytes": _kb(rollup, "Pss_Anon"),
        "pss_file_bytes": _kb(rollup, "Pss_File"),
        "swap_bytes": _kb(rollup, "Swap") if rollup else _kb(status, "VmSwap"),
        "rss_lifetime_peak_bytes": _kb(status, "VmHWM"),
        "smaps_rollup_status": rollup_status,
        "minor_faults": stat["minor_faults"],
        "major_faults": stat["major_faults"],
        "host": _host_snapshot(proc_root, identity["allowed_cpus"]),
    }


def _memory_summary(samples: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [row[key] for row in samples if row[key] is not None]
    area, duration = 0.0, 0.0
    for before, after in pairwise(samples):
        if before[key] is not None and after[key] is not None:
            interval = after["monotonic_seconds"] - before["monotonic_seconds"]
            duration += interval
            area += interval * (before[key] + after[key]) / 2
    wall = samples[-1]["monotonic_seconds"] - samples[0]["monotonic_seconds"]
    return {
        "start": samples[0][key],
        "end": samples[-1][key],
        "peak": max(values) if values else None,
        "mean": area / duration if duration > 0 else None,
        "available_samples": len(values),
        "coverage_fraction": duration / wall,
    }


def _busy_percent(before: dict, after: dict, prefix: str = "cpu") -> float | None:
    if before.get("status") != "ok" or after.get("status") != "ok":
        return None
    total = after[f"{prefix}_total_ticks"] - before[f"{prefix}_total_ticks"]
    busy = after[f"{prefix}_busy_ticks"] - before[f"{prefix}_busy_ticks"]
    return 100 * busy / total if total > 0 and 0 <= busy <= total else None


def summarize_samples(
    samples: Sequence[dict[str, Any]], interval: float
) -> dict[str, Any]:
    """Summarize an interval while retaining absolute startup/end snapshots."""
    if len(samples) < 2:
        raise ValueError("At least two process samples are required")
    start, end = samples[0], samples[-1]
    for before, after in pairwise(samples):
        if after["identity"] != start["identity"]:
            raise ProcessChangedError("PID, executable or affinity changed")
        if after["cpu_seconds"] < before["cpu_seconds"]:
            raise ProcessChangedError("Process CPU counter moved backwards")
        if after["monotonic_seconds"] <= before["monotonic_seconds"]:
            raise ValueError("Sample timestamps must strictly increase")
    wall = end["monotonic_seconds"] - start["monotonic_seconds"]
    cpu = end["cpu_seconds"] - start["cpu_seconds"]
    cores = cpu / wall
    return {
        "schema": "udlex-process-resources/v1",
        "identity": start["identity"],
        "wall_seconds": wall,
        "cpu_seconds_delta": cpu,
        "cpu_cores": cores,
        "cpu_percent_one_core": 100 * cores,
        "cpu_percent_allocated": 100 * cores / start["identity"]["allowed_cpu_count"],
        "samples": len(samples),
        "interval_seconds": interval,
        "max_observed_interval_seconds": max(
            b["monotonic_seconds"] - a["monotonic_seconds"]
            for a, b in pairwise(samples)
        ),
        **{key: _memory_summary(samples, key) for key in MEMORY_FIELDS},
        "start_snapshot": start,
        "end_snapshot": end,
        "host_cpu_busy_percent": _busy_percent(start["host"], end["host"]),
        "affinity_cpu_busy_percent": _busy_percent(
            start["host"], end["host"], "allowed_cpu"
        ),
        "memory_mean_method": (
            "trapezoidal time weighting over available adjacent samples"
        ),
        "sampled_peak_is_lower_bound": True,
        "scope": (
            "One server JVM process including JNI, excluding child processes and client"
        ),
    }


class ProcessSampler:
    """Sample one PID in a daemon thread and propagate every thread failure."""

    def __init__(
        self, pid: int, interval_seconds: float = 0.25, *, proc_root: Path = PROC
    ) -> None:
        """Configure a sampler without touching the measured process."""
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be finite and positive")
        self.pid, self.interval_seconds, self.proc_root = (
            pid,
            interval_seconds,
            proc_root,
        )
        self.identity: dict[str, Any] | None = None
        self._samples: list[dict[str, Any]] = []
        self._event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._result: dict[str, Any] | None = None

    def _sample(self) -> None:
        row = process_snapshot(self.pid, proc_root=self.proc_root)
        if self.identity is None:
            self.identity = row["identity"]
        elif row["identity"] != self.identity:
            raise ProcessChangedError("Measured PID or affinity changed")
        self._samples.append(row)

    def _run(self) -> None:
        while not self._event.wait(self.interval_seconds):
            try:
                self._sample()
            except Exception as error:  # noqa: BLE001
                # All failures must cross the thread boundary to stop().
                self._error = error
                self._event.set()
                return

    def start(self) -> ProcessSampler:
        """Take a synchronous baseline, then begin background sampling."""
        if self._thread is not None:
            raise RuntimeError("Sampler has already been started")
        self._sample()
        self._thread = threading.Thread(
            target=self._run, name="opensearch-proc-sampler", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        """Join the sampler, fail on background errors and capture the final sample."""
        if self._thread is None:
            raise RuntimeError("Sampler has not been started")
        if self._result is not None:
            return self._result
        self._event.set()
        self._thread.join(timeout=max(2.0, self.interval_seconds * 4))
        if self._thread.is_alive():
            raise RuntimeError("Process sampler failed to stop")
        if self._error is not None:
            raise self._error
        self._sample()
        self._result = summarize_samples(self._samples, self.interval_seconds)
        return self._result


def _single_node(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if (
        payload.get("_nodes", {}).get("failed") != 0
        or len(payload.get("nodes", {})) != 1
    ):
        raise ValueError("Resource measurement requires one healthy OpenSearch node")
    return next(iter(payload["nodes"].items()))


def _available(value: Any) -> int | None:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else None
    )


def jvm_snapshot(client: HTTPClient, expected_pid: int | None = None) -> dict[str, Any]:
    """Read JVM stats outside timing, optionally binding a host PID via NSpid."""
    information, _ = client.request("GET", "/_nodes/process")
    node_id, node = _single_node(information)
    pid = node["process"]["id"]
    binding = None
    if expected_pid is not None:
        binding = read_process_identity(expected_pid)
        if pid not in binding["namespace_pids"]:
            raise ProcessChangedError("OpenSearch endpoint PID does not match host PID")
    payload, _ = client.request("GET", "/_nodes/stats/jvm,process")
    stats_id, stats = _single_node(payload)
    if stats_id != node_id:
        raise ProcessChangedError("OpenSearch node changed during JVM snapshot")
    if binding is not None and binding != read_process_identity(expected_pid):
        raise ProcessChangedError("Host JVM PID changed during HTTP snapshot")
    jvm, process = stats["jvm"], stats["process"]
    memory = jvm["mem"]
    return {
        "node_id": node_id,
        "pid": pid,
        "host_pid": expected_pid,
        "process_identity": binding,
        "timestamp_ms": jvm["timestamp"],
        "uptime_ms": jvm["uptime_in_millis"],
        **{
            name: _available(memory.get(source))
            for name, source in (
                ("heap_used_bytes", "heap_used_in_bytes"),
                ("heap_committed_bytes", "heap_committed_in_bytes"),
                ("heap_max_bytes", "heap_max_in_bytes"),
                ("non_heap_used_bytes", "non_heap_used_in_bytes"),
            )
        },
        "gc": {
            name: {
                "count": _available(row.get("collection_count")),
                "time_ms": _available(row.get("collection_time_in_millis")),
            }
            for name, row in jvm["gc"]["collectors"].items()
        },
        "process_cpu_total_ms": _available(
            process.get("cpu", {}).get("total_in_millis")
        ),
    }


def _counter_delta(before: int | None, after: int | None) -> int | None:
    if before is None or after is None:
        return None
    if after < before:
        raise ProcessChangedError(
            "JVM counter moved backwards; process may have restarted"
        )
    return after - before


def jvm_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Keep heap snapshots distinct from cumulative GC/CPU counter deltas."""
    if any(
        before[key] != after[key]
        for key in ("node_id", "pid", "host_pid", "process_identity")
    ):
        raise ProcessChangedError("JVM identity changed across interval")
    _counter_delta(before["uptime_ms"], after["uptime_ms"])
    if set(before["gc"]) != set(after["gc"]):
        raise ProcessChangedError("JVM garbage collectors changed")
    return {
        "before": before,
        "after": after,
        "process_cpu_ms_delta": _counter_delta(
            before["process_cpu_total_ms"], after["process_cpu_total_ms"]
        ),
        "gc": {
            name: {
                key: _counter_delta(before["gc"][name][key], after["gc"][name][key])
                for key in ("count", "time_ms")
            }
            for name in before["gc"]
        },
    }
