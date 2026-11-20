"""Run the frozen OpenSearch comparison in a fresh, network-isolated container."""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from compare import bench, three

HERE = Path(__file__).resolve().parent
ENGINE = Path("/opt/opensearch")
OPTIONAL = Path("/opt/optional")
ASSETS = Path("/inputs")
OUTPUT = Path("/results")
PROFILES = ("Udlex", "OpenSearch", "Native", *three.OFFICIAL_PLUGIN_LABELS)
LOCK = json.loads((HERE / "benchmark.lock.json").read_bytes())


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"SHA-256 mismatch for {path}")


def prepare_inputs(root, languages):
    config = root / "inputs"
    shutil.copytree(ENGINE / "config", config)
    catalog = ASSETS / "DICTIONARIES.json"
    verify(catalog, LOCK["dictionary_catalog_sha256"])
    shutil.copyfile(catalog, config / catalog.name)
    entries = json.loads(catalog.read_bytes())["dictionaries"]
    if len({entry["iso639_3"] for entry in entries}) != LOCK["dictionary_count"]:
        raise ValueError("Unexpected dictionary inventory")
    frozen = ASSETS / "benchmark-inputs.tar.zst"
    verify(frozen, LOCK["benchmark_inputs_sha256"])
    with tarfile.open(frozen, "r|zst") as archive:
        archive.extractall(config, filter="data")
    verify(config / "inputs.json", LOCK["input_manifest_sha256"])
    for entry in entries:
        iso = entry["iso639_3"]
        if languages and iso not in languages:
            continue
        item = entry["archive"]
        path = ASSETS / "dictionaries" / item["name"]
        verify(path, item["sha256"])
        with tarfile.open(path, "r|gz") as archive:
            archive.extractall(config, filter="data")
        if (config / "bundle" / iso / "current").read_text().strip() != entry["compiled_generation"]:
            raise ValueError(f"Dictionary generation mismatch for {iso}")
    return config


def cpu_layout():
    allowed = sorted(os.sched_getaffinity(0))
    # Docker may impose a CPU quota without shrinking the affinity mask.
    quota = Path("/sys/fs/cgroup/cpu.max")
    bandwidth = quota.read_text().split() if quota.is_file() else []
    old_quota = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    if not bandwidth and old_quota.is_file():
        limit = old_quota.read_text().strip()
        bandwidth = ["max" if limit == "-1" else limit, old_quota.with_name("cpu.cfs_period_us").read_text().strip()]
    count = len(allowed)
    if bandwidth and bandwidth[0] != "max":
        count = min(count, max(1, math.ceil(int(bandwidth[0]) / int(bandwidth[1]))))
    available = allowed[:count]
    server = available[:min(4, max(1, count - 1))]
    client = available[len(server):len(server) + 2] or server
    return {"allowed": allowed, "server": server, "client": client, "cpu_max": bandwidth}


def start_on(cpus, command, **kwargs):
    previous = os.sched_getaffinity(0)
    try:
        os.sched_setaffinity(0, cpus)
        return subprocess.Popen(command, **kwargs)
    finally:
        os.sched_setaffinity(0, previous)


def wait_for_node(process, url, log):
    health = url + "/_cluster/health?wait_for_status=yellow&timeout=1s"
    deadline = time.monotonic() + 300
    while process.poll() is None and time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=3) as response:
                if json.load(response)["status"] in {"yellow", "green"}:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(1)
    tail = "\n".join(log.read_text(errors="replace").splitlines()[-25:])
    raise RuntimeError(f"OpenSearch startup failed. See {log.name}\n{tail}")


def run_profile(name, root, report, cpus, selection):
    config = root / "inputs"
    plugins = set(three.OFFICIAL_PLUGINS.get(name, ()))
    if name in three.UDLEX_PROFILES:
        plugins.add("analysis-udlex")
    for plugin in sorted(plugins):
        shutil.copytree(OPTIONAL / plugin, ENGINE / "plugins" / plugin)
    url = "http://127.0.0.1:19200"
    node = root / "nodes" / name
    node.mkdir(parents=True)
    log = report / "logs" / f"{name}.log"
    env = dict(os.environ, OPENSEARCH_JAVA_HOME=str(ENGINE / "jdk"), OPENSEARCH_PATH_CONF=str(config),
               OPENSEARCH_JAVA_OPTS=f"-Xms1g -Xmx1g -XX:ActiveProcessorCount={len(cpus['server'])} --enable-native-access=ALL-UNNAMED")
    command = [str(ENGINE / "bin/opensearch"), f"-Ecluster.name=bench-{name.lower()}",
               f"-Enode.name=bench-{name.lower()}", f"-Epath.data={node}/data", f"-Epath.logs={node}/logs"]
    try:
        with log.open("w") as console:
            jvm = start_on(cpus["server"], command, env=env, stdout=console, stderr=subprocess.STDOUT)
            try:
                wait_for_node(jvm, url, log)
                profile = {"schema": three.PROFILE_SCHEMA, "name": name, "url": url, "pid": jvm.pid,
                           "runtime_limits": {"allowed_cpus": cpus["server"], "client_allowed_cpus": cpus["client"],
                                             "heap_init_bytes": 2**30, "heap_max_bytes": 2**30,
                                             "allocated_processors": len(cpus["server"]), "global_page_cache_reset": False}}
                if name in three.UDLEX_PROFILES:
                    profile.update(plugin_bundle="bundle", plugin_bundle_dir=str(config / "bundle"),
                                   installed_plugin_dir=str(ENGINE / "plugins/analysis-udlex"))
                three.validate_profile(profile)
                bench.atomic_json(root / "stand/profiles" / f"{name}.json", profile)
                measure = [sys.executable, str(HERE / "compare.py"), str(root), name, *selection]
                worker = start_on(cpus["client"], measure)
                try:
                    if worker.wait():
                        raise RuntimeError(f"{name} measurement failed")
                finally:
                    if worker.poll() is None:
                        worker.terminate()
                        worker.wait()
                if name == "Udlex":
                    subprocess.run([sys.executable, HERE / "smoke.py", root], check=True)
            finally:
                jvm.terminate()
                try:
                    jvm.wait(timeout=35)
                except subprocess.TimeoutExpired:
                    jvm.kill()
                    jvm.wait()
    finally:
        for plugin in plugins:
            shutil.rmtree(ENGINE / "plugins" / plugin)
        raw = root / "stand/results" / name
        if raw.is_dir():
            shutil.copytree(raw, report / "measurements" / name)


def verify_quality(report, languages):
    checks = {}
    for table in ("by-language.csv", "cjk.csv"):
        reference = HERE / "reference" / table
        verify(reference, LOCK["reference_sha256"][table])
        with reference.open() as old, (report / table).open() as new:
            expected = {r["iso"]: r for r in csv.DictReader(old) if not languages or r["iso"] in languages}
            actual = {r["iso"]: r for r in csv.DictReader(new)}
        if actual.keys() != expected.keys():
            raise ValueError(f"Language coverage differs from the reference in {table}")
        for iso, row in actual.items():
            for key, value in row.items():
                if key in {"evaluation", "gold_pairs"} or key.endswith(("_pair_f1_pct", "_lexical_casefold_pct")):
                    if value != expected[iso][key]:
                        raise ValueError(f"{table}, {iso}, {key}: expected {expected[iso][key]}, got {value}")
        checks[table] = {"matched_rows": len(actual), "reference_sha256": sha256(reference)}
    bench.atomic_json(report / "quality-check.json", checks)


def run(mode):
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    report = Path(tempfile.mkdtemp(prefix=f"{stamp}-{mode}-", dir=OUTPUT))
    if os.getuid() == 0:
        owner = OUTPUT.stat()
        uid = owner.st_uid or 1000
        gid = owner.st_gid or 1000
        os.chown(report, uid, gid)
        os.setgroups([])
        os.setgid(gid)
        os.setuid(uid)
    report.chmod(0o755)
    (report / "logs").mkdir()
    cpus = cpu_layout()
    languages = LOCK["smoke_languages"] if mode == "smoke" else []
    selection = ["--languages", *languages] if languages else []
    receipt = {"scope": mode, "state": "running", "started_utc": stamp, "architecture": platform.machine(),
               "kernel": platform.release(), "python": platform.python_version(), "cpu_layout": cpus,
               "lock": LOCK, "harness_sha256": sha256(Path(__file__)), "completed_profiles": []}
    (report / "cpuinfo.txt").write_bytes(Path("/proc/cpuinfo").read_bytes())
    status = report / "run.json"
    bench.atomic_json(status, receipt)
    print(f"Results: results/{report.name}", flush=True)
    try:
        with tempfile.TemporaryDirectory(prefix="opensearch-bench-") as work:
            root = Path(work)
            os.environ["HOME"] = str(root)
            config = prepare_inputs(root, languages)
            (config / "opensearch.yml").write_text(
                "discovery.type: single-node\nnetwork.host: 127.0.0.1\nhttp.port: 19200\n"
                "transport.port: 19300\nplugins.security.disabled: true\nbootstrap.memory_lock: false\n"
                "cluster.routing.allocation.disk.watermark.low: 10gb\n"
                "cluster.routing.allocation.disk.watermark.high: 5gb\n"
                "cluster.routing.allocation.disk.watermark.flood_stage: 2gb\n")
            for name in PROFILES:
                print(f"Starting {name}", flush=True)
                run_profile(name, root, report, cpus, selection)
                receipt["completed_profiles"].append(name)
                bench.atomic_json(status, receipt)
            subprocess.run([sys.executable, HERE / "summarize.py", report / "measurements",
                            config / "DICTIONARIES.json", report, *selection], check=True)
            verify_quality(report, languages)
        receipt["state"] = "complete"
    except BaseException as exc:
        receipt.update(state="failed", error=str(exc))
        raise
    finally:
        receipt["finished_utc"] = dt.datetime.now(dt.UTC).isoformat()
        bench.atomic_json(status, receipt)
    print(f"Verified {mode} results: results/{report.name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "full"), nargs="?", default="full")
    args = parser.parse_args()
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    run(args.mode)
