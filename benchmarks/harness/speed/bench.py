"""Speed and memory of udlex 1.0.0 and the compared lemmatizers on UD 2.19 test words (C9).

    bench.py prepare RUN LANG...     unpack the dictionary packages, write udlex reference
                                     answers; run with the Python that scored them
    bench.py run RUN OUT [options]   measure; one language is one session for every system
    bench.py summary OUT             tables of the receipts

A cell is (language, system, mode, round): fresh worker processes pinned with taskset load
the system, warm up, then time windows of at least a second, each on a fresh random sample
of test sentences. Within a language the cells of a round run in a random order. Answers of
every window are checked against the predictions the quality was scored from.

Directories of the stand come from the environment (prepare, run and verify need them):
UDLEX_SPEED_COMPARE the quality comparison, UDLEX_SPEED_DICTS the dictionary build,
UDLEX_SPEED_INPUTS the timing inputs (default: the census inputs of the comparison).
"""

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import select
import statistics
import subprocess
import threading
import time
import zlib
from pathlib import Path

LANGS = [
    "eng",
    "rus",
    "deu",
    "fin",
    "tur",
    "arb",
    "hin",
    "kor",
    "jpn",
    "ces",
    "hun",
    "gla",
]
HERE = Path(__file__).resolve().parent
# 1c: one pinned CPU; thr16: one process, 16 threads; proc16: 16 single-thread processes.
MODES = {
    "udlex-rs": ["1c", "thr16"],
    "udlex-py": ["1c", "thr16", "proc16"],
    "stanza": ["1c", "thr16", "proc16", "gpu"],
    "trankit": [
        "1c",
        "thr16",
        "gpu",
    ],  # 16 copies of XLM-R large do not fit beside the owner's jobs
    "spacy-sm": ["1c", "proc16"],
    "spacy-lg": ["1c", "proc16"],
    "spacy-trf": ["1c", "thr16", "proc16", "gpu"],
    "simplemma": ["1c", "proc16"],
    "lemmagen3": ["1c", "proc16"],
    "udpipe1": ["1c", "proc16"],
    "corenlp": ["1c", "thr16"],
}
NEURAL = {"stanza", "trankit", "spacy-trf"}
SIXTEEN = list(
    range(16)
)  # one hardware thread on each physical core; siblings are 16-31
TICK = os.sysconf("SC_CLK_TCK")


def stand(name):
    """A directory of the stand, from the environment: the harness carries no paths."""
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set (see run.sh)")
    return Path(value)


def compare():
    """The quality comparison: census inputs, predictions and environments."""
    return stand("UDLEX_SPEED_COMPARE")


def dicts():
    """The build whose release and scored extractions are timed; a rebuild overrides it."""
    return stand("UDLEX_SPEED_DICTS")


def inputs():
    """Census inputs; fill_inputs.py builds an overlay with text for twofold languages."""
    if os.environ.get("UDLEX_SPEED_INPUTS"):
        return stand("UDLEX_SPEED_INPUTS")
    return compare() / "census/inputs/order"


def read(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def size(*paths):
    """Bytes of the files under the paths, each real file counted once."""
    real = set()
    for path in paths:
        p = Path(path)
        for f in p.rglob("*") if p.is_dir() else [p]:
            if f.is_file():
                real.add(os.path.realpath(f))
    return sum(Path(f).stat().st_size for f in real)


# ----------------------------------------------------------------------------- prepare


def digest_tree(root):
    """One digest over every file of a bundle, by relative path."""
    h = hashlib.sha256()
    for f in sorted(p for p in Path(root).rglob("*") if p.is_file()):
        h.update(f"{f.relative_to(root)}\0{sha256(f)}\0".encode())
    return h.hexdigest()


def prepare(a):
    """Bundles of the dictionary release and the answers its evaluation counted.

    With ``--evaluated`` the bundles are the extractions evaluate.py scored (for a
    smoke test while the release is being repackaged); either way a bundle must give
    the evaluated counts on the census test, and its lexicon files are compared with
    the evaluated extraction.
    """
    import shutil
    import tarfile
    import unicodedata

    import udlex

    release = dicts() / "release"
    evaluation = release / "evaluation.json"
    if not evaluation.exists():
        evaluation = dicts() / "work/evaluation.final.json"
    scored = {r["iso639_3"]: r for r in read(evaluation)["languages"]}
    entries = [] if a.evaluated else read(release / "DICTIONARIES.json")["dictionaries"]
    out = a.run / "ref" / "udlex"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "udlex": udlex.__version__,
        "udlex_file": udlex.__file__,
        "source": "evaluated extractions" if a.evaluated else str(release),
        "evaluation": str(evaluation),
        "languages": {},
    }
    for iso in a.langs:
        # Only officially split languages have a scored extraction; twofold ones do not.
        scored_dir = dicts() / "work/eval/new" / iso
        evaluated = (
            next(scored_dir.iterdir()) / "bundle" if scored_dir.is_dir() else None
        )
        row = {}
        if a.evaluated:
            if evaluated is None:
                raise SystemExit(f"{iso}: no evaluated extraction to time")
            dest = a.run / "bundles" / iso / f"evaluated-{evaluated.parent.name}"
            if not dest.is_dir():
                shutil.copytree(evaluated, dest / "bundle")
        else:
            entry = next(e for e in entries if e["iso639_3"] == iso)  # as evaluate.py
            archive = release / "packages" / entry["archive"]["name"]
            digest = sha256(archive)
            if digest != entry["archive"]["sha256"]:
                raise SystemExit(f"{archive} does not match DICTIONARIES.json")
            dest = a.run / "bundles" / iso / digest[:16]
            if not dest.is_dir():
                dest.mkdir(parents=True)
                with tarfile.open(archive) as packed:
                    packed.extractall(dest, filter="data")
            row = {
                "archive": entry["archive"]["name"],
                "archive_sha256": digest,
                "archive_bytes": archive.stat().st_size,
            }
        bundle = dest / "bundle"
        lexicon = udlex.load(str(bundle), iso, use_user=False)
        with (inputs() / iso / "test.jsonl").open(encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh]
        answers = lexicon.lemmatize_many([r["tokens"] for r in rows])

        def nfc(s):
            return unicodedata.normalize("NFC", s)  # evaluate.py's scoring

        pairs = [
            (x, y)
            for r, row_ in zip(rows, answers, strict=True)
            for x, y in zip(row_, r["lemmas"], strict=True)
            if y is not None
        ]
        hits, total = sum(nfc(x) == nfc(y) for x, y in pairs), len(pairs)
        want = scored[iso]
        # A twofold language ships a dictionary trained on all its data, while the
        # evaluation scored two held-out folds: there are no counts to match.
        # Six languages were scored on another anchor treebank: totals differ, and
        # only an equal total can check the bundle.
        same_test = want.get("mode") != "twofold" and total == want["tokens"]
        if same_test and hits != want["new"]:
            raise SystemExit(
                f"{iso}: {hits}/{total} here, {want['new']}/{want['tokens']} evaluated"
            )
        (out / f"{iso}.json").write_text(
            json.dumps(answers, ensure_ascii=False), encoding="utf-8"
        )
        tables = Path(
            iso, ".generations"
        )  # the compiled tables; recipes and notices aside
        lexicons = digest_tree(bundle / tables) if (bundle / tables).is_dir() else None
        manifest["languages"][iso] = row | {
            "bundle": str(bundle),
            "bundle_bytes": size(bundle),
            "bundle_sha256": digest_tree(bundle),
            "same_bundle_as_evaluated": evaluated is not None
            and digest_tree(bundle) == digest_tree(evaluated),
            "same_lexicon_as_evaluated": evaluated is not None
            and lexicons is not None
            and lexicons == digest_tree(evaluated / tables),
            "evaluation_mode": want.get("mode"),
            "checked_against_evaluation": same_test,
            "correct": hits,
            "scored": total,
            # some tests carry no lemmas
            "accuracy": 100 * hits / total if total else None,
        }
        print(iso, hits, total, flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------- machine load


def cpu_counters():
    out = {}
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith("cpu") and line[3].isdigit():
            fields = line.split()
            values = list(map(int, fields[1:]))
            out[int(fields[0][3:])] = (sum(values), values[3] + values[4])
    return out


def busy(before, after):
    return {
        c: 100 * (1 - (after[c][1] - before[c][1]) / max(1, after[c][0] - before[c][0]))
        for c in after
    }


def processes():
    out = {}
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            stat = Path(entry.path, "stat").read_text()
        except OSError:
            continue
        rest = stat[stat.rindex(")") + 2 :].split()
        out[int(entry.name)] = (
            stat[stat.index("(") + 1 : stat.rindex(")")],
            int(rest[11]) + int(rest[12]),
            int(rest[1]),
        )
    return out


def foreign(before, after, seconds):
    """CPU of processes that are not this driver or its children, in cores, and the top five."""
    ours, grew = {os.getpid()}, True
    while grew:
        grew = False
        for pid, (_, _, parent) in after.items():
            if parent in ours and pid not in ours:
                ours.add(pid)
                grew = True
    used = [
        (name, (ticks - before[pid][1]) / TICK / max(seconds, 1e-9))
        for pid, (name, ticks, _) in after.items()
        if pid in before and pid not in ours
    ]
    used.sort(key=lambda x: -x[1])
    return round(sum(u for _, u in used), 3), [
        [n, round(u, 3)] for n, u in used[:5] if u >= 0.01
    ]


def gpu_state():
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        ).stdout
        util, used, total, power = (float(x) for x in out.strip().split(","))
    except (OSError, subprocess.SubprocessError, ValueError):
        return None  # no GPU reading is recorded as none
    return {
        "util": util,
        "mem_used_mb": used,
        "mem_total_mb": total,
        "power_w": power,
    }


class Sampler(threading.Thread):
    """Machine load once a second (GPU every five), tagged with the running cell."""

    def __init__(self, path):
        super().__init__(daemon=True)
        self.path, self.tag, self.rows, self.stop = path, "start", [], threading.Event()

    def run(self):
        last, k = cpu_counters(), 0
        with self.path.open("a", encoding="utf-8") as fh:
            while not self.stop.wait(1.0):
                now = cpu_counters()
                load1 = float(Path("/proc/loadavg").read_text().split()[0])
                avail = next(
                    int(line.split()[1])
                    for line in Path("/proc/meminfo").read_text().splitlines()
                    if line.startswith("MemAvailable:")
                )
                row = {
                    "t": time.time(),
                    "tag": self.tag,
                    "load1": load1,
                    "mem_avail_kb": avail,
                    "busy": [round(v, 1) for _, v in sorted(busy(last, now).items())],
                }
                if k % 5 == 0:
                    row["gpu"] = gpu_state()
                self.rows.append(row)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                last, k = now, k + 1

    def summary(self, start, end, pinned):
        rows = [r for r in self.rows if start <= r["t"] <= end + 0.5]
        if not rows:
            return {}
        others = [c for c in range(len(rows[0]["busy"])) if c not in pinned]
        gpus = [r["gpu"] for r in rows if r.get("gpu")]
        return {
            "samples": len(rows),
            "pinned_busy_pct": round(
                statistics.mean(
                    statistics.mean(r["busy"][c] for c in pinned) for r in rows
                ),
                1,
            )
            if pinned
            else None,
            "other_cpus_busy_cores": round(
                statistics.mean(sum(r["busy"][c] for c in others) / 100 for r in rows),
                2,
            ),
            "load1_max": max(r["load1"] for r in rows),
            "mem_avail_min_gb": round(min(r["mem_avail_kb"] for r in rows) / 2**20, 1),
            "gpu_util_mean": round(statistics.mean(g["util"] for g in gpus), 1)
            if gpus
            else None,
            "gpu_util_max": max(g["util"] for g in gpus) if gpus else None,
        }


# ----------------------------------------------------------------------------- cells


def census(engine):
    path = compare() / "results-census" / f"{engine}.json"
    return (
        {i: c for i, c in read(path)["cells"].items() if c["status"] == "scored"}
        if path.exists()
        else {}
    )


def cells_for(iso, a):
    """Every (system, mode) the language supports, with the worker configuration."""
    row = read(inputs() / "inventory.json")["languages"][iso]
    lang = row["config"].split("_")[0]
    base = {
        "iso": iso,
        "lang": lang,
        "config": row["config"],
        "treebank": row["treebank"],
        "test": str(inputs() / iso / "test.jsonl"),
        "experiments": str(a.harness),
        "repo": str(compare() / "repo"),
        "cupy": str(a.run / "pylib-cupy"),
    }
    out = []
    bundle = read(a.run / "ref/udlex/manifest.json")["languages"][iso]["bundle"]
    udlex_ref = str(a.run / "ref" / "udlex" / f"{iso}.json")
    out.append(
        ("udlex-rs", dict(base, system="udlex-rs", bundle=bundle, ref=udlex_ref))
    )
    out.append(
        (
            "udlex-py",
            dict(
                base,
                system="udlex-py",
                bundle=bundle,
                ref=udlex_ref,
                python=str(a.run / ".venv-udlex/bin/python"),
            ),
        )
    )
    pythons = {
        "stanza": compare() / ".venv",
        "spacy-sm": compare() / ".venv",
        "simplemma": compare() / ".venv",
        "lemmagen3": compare() / ".venv",
        "spacy-lg": compare() / ".venv-spacy-big",
        "spacy-trf": compare() / ".venv-spacy-big",
        "trankit": compare() / ".venv-trankit-git",
        "udpipe1": a.run / ".venv-udpipe1",
        "corenlp": a.run / ".venv-udlex",  # the client needs the standard library only
    }
    models = {
        "stanza": compare() / "stanza",
        "trankit": compare() / "trankit-cache-git",
        "corenlp": compare() / "corenlp/stanford-corenlp-4.6.0",
    }
    for engine, python in pythons.items():
        scored = census(engine).get(iso)
        if not scored:
            continue
        cfg = dict(
            base,
            system=engine,
            python=str(python / "bin/python"),
            ref=str(compare() / "predictions-census" / engine / f"{iso}.json.gz"),
            model=scored["model"],
            models=str(models.get(engine, "")),
        )
        if engine == "udpipe1":
            cfg["model_file"] = str(
                a.run / "udpipe1-models" / f"{scored['model']}.udpipe"
            )
            if not Path(cfg["model_file"]).is_file():
                continue
        out.append((engine, cfg))
    return [
        (name, mode, cfg)
        for name, cfg in out
        if (not a.systems or name in a.systems)
        for mode in MODES[name]
        if not a.modes or mode in a.modes
    ]


class Cell:
    """Worker processes of one cell; replies are whole lines, one per request."""

    def __init__(self, name, mode, cfg, a, log):
        many = a.cpus16  # "16" modes use every CPU of --cpus16; a smoke test passes two
        self.workers = len(many) if mode == "proc16" else 1
        threads = len(many) if mode in ("thr16", "gpu") else 1
        cpus = (
            [[c] for c in many]
            if mode == "proc16"
            else [[a.cpu1] if mode == "1c" else many]
        )
        self.cpus = sorted({c for cs in cpus for c in cs})
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0" if mode == "gpu" else "")
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[key] = str(threads)
        env.pop("UDLEX_USER_BUNDLE", None)
        if (
            mode == "gpu" and name != "trankit"
        ):  # Trankit's torch 2.0.1 lacks the option
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        cfg = dict(cfg, threads=threads, gpu=mode == "gpu")
        if name in NEURAL and mode in ("thr16", "gpu"):
            cfg["call_sentences"] = pick(a.neural_batch, f"{name}:{mode}")
        if name == "udlex-py":
            cfg["batch"] = pick(a.batch, mode)
        if name == "stanza" and mode == "gpu" and a.stanza_gpu_tokens:
            # 0 keeps Stanza's own POS batch; beside vLLM only ~4 GB are free.
            cfg.setdefault(
                "stanza_kwargs", {"pos_batch_maximum_tokens": a.stanza_gpu_tokens}
            )
        self.cfg = cfg
        if name == "udlex-rs":
            argv = [
                str(a.native),
                cfg["bundle"],
                cfg["iso"],
                cfg["test"],
                cfg["ref"],
                str(threads),
            ]
        else:
            argv = [cfg["python"], "-I", str(HERE / "worker.py"), json.dumps(cfg)]
        self.procs = [
            subprocess.Popen(  # ty: ignore[no-matching-overload]  # cfg values are str here
                ["taskset", "-c", ",".join(map(str, cs)), *argv],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                env=env,
                text=True,
                bufsize=1,
            )
            for cs in cpus[: self.workers]
        ]

    def ask(self, commands, timeout):
        for p, c in zip(self.procs, commands, strict=True):
            p.stdin.write(json.dumps(c) + "\n")  # ty: ignore[unresolved-attribute, no-matching-overload]  # stdin=PIPE, text
            p.stdin.flush()  # ty: ignore[unresolved-attribute]  # stdin=PIPE
        return [self.recv(p, timeout) for p in self.procs]

    def recv(self, p, timeout):
        if not select.select([p.stdout], [], [], timeout)[0]:
            raise TimeoutError(f"no answer in {timeout:.0f} s")
        line = p.stdout.readline()
        if not line:
            raise EOFError(f"worker exited with status {p.wait()}")
        reply = json.loads(line)
        if reply.get("event") == "error":
            raise RuntimeError(reply["reason"])
        return reply

    def kill(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill()
            p.wait()


def pick(values, key):
    """``N`` for every cell, ``key=N`` for one; ``bench.py choose`` writes the keys."""
    sizes = dict(x.split("=") if "=" in x else ("", x) for x in values)
    return int(sizes.get(key, sizes[""]))


def owner_jobs(a):
    """The owner's jobs during which no timing may run (a flag file, a tmux session)."""
    flags = [p for p in a.pause_while if Path(p).exists()]
    sessions = [
        s
        for s in a.pause_tmux
        if subprocess.run(
            ["tmux", "has-session", "-t", s], capture_output=True, check=False
        ).returncode
        == 0
    ]
    return flags + sessions


def wait_quiet(a):
    start = time.time()
    while owner_jobs(a):
        time.sleep(60)
    return round(time.time() - start)


def mean_length(iso):
    with (inputs() / iso / "test.jsonl").open(encoding="utf-8") as fh:
        lengths = [len(json.loads(line)["tokens"]) for line in fh]
    return sum(lengths) / len(lengths)


def run_cell(name, mode, cfg, rnd, a, sampler, logs, memory):
    """Load, warm up and time one cell; the record goes to cells.jsonl whatever happens."""
    iso = cfg["iso"]
    record = {
        "lang": iso,
        "system": name,
        "mode": mode,
        "round": rnd,
        "started": time.time(),
    }
    if mode == "proc16":
        avail = sampler.rows[-1]["mem_avail_kb"] if sampler.rows else 0
        need = len(a.cpus16) * memory.get(name, 2 * 2**20) * 1.2 + 16 * 2**20
        if avail < need:
            return dict(
                record,
                status="skipped",
                reason=f"MemAvailable {avail >> 20} GiB < {need / 2**20:.0f} GiB",
            )
    if mode == "gpu":
        state = gpu_state()
        record["gpu_before"] = state
        if not state or state["mem_total_mb"] - state["mem_used_mb"] < a.gpu_free_mb:
            return dict(record, status="skipped", reason=f"GPU busy: {state}")
    with (logs / f"{iso}-{name}-{mode}-r{rnd}.log").open("w") as log:
        sampler.tag = f"{iso}/{name}/{mode}/r{rnd}"
        cell = None
        try:
            cell = Cell(name, mode, cfg, a, log)
            record["cpus"] = cell.cpus
            record["workers"] = cell.workers
            record["settings"] = {
                k: cell.cfg[k]
                for k in ("batch", "call_sentences", "stanza_kwargs")
                if k in cell.cfg
            }
            procs0, t0 = processes(), time.time()
            record["ready"] = [cell.recv(p, a.load_timeout) for p in cell.procs]
            check_ready(record["ready"])
            # Neural systems with threads or a GPU get calls of --neural-batch sentences.
            floor = cfg.get("call_sentences", 1)
            warm = cell.ask(
                [
                    {
                        "cmd": "warm",
                        "seconds": a.warm,
                        "min_window": a.target,
                        "min_sentences": floor,
                    }
                ]
                * cell.workers,
                a.step_timeout,
            )
            record["warm"] = warm
            rate = statistics.median(w["rate"] for w in warm)
            n = max(floor, math.ceil(a.target * rate / mean_length(iso)))
            windows = []
            for w in range(a.windows):
                for attempt in range(4):
                    seed = zlib.crc32(
                        f"{iso}/{name}/{mode}/{rnd}/{w}/{attempt}".encode()
                    )
                    cell.ask(
                        [
                            {"cmd": "prep", "seed": seed * 64 + i, "sentences": n}
                            for i in range(cell.workers)
                        ],
                        a.step_timeout,
                    )
                    start = time.perf_counter()
                    replies = cell.ask(
                        [{"cmd": "go"}] * cell.workers, a.step_timeout + 20 * a.target
                    )
                    wall = time.perf_counter() - start
                    seconds = wall if cell.workers > 1 else replies[0]["seconds"]
                    window = {
                        "window": w,
                        "sentences": n * cell.workers,
                        "tokens": sum(r["tokens"] for r in replies),
                        "seconds": seconds,
                        "wall": wall,
                        "cpu_s": sum(r["cpu_s"] for r in replies),
                        "checked": sum(r["checked"] for r in replies),
                        "mismatch": sum(r["mismatch"] for r in replies),
                        "bad": [b for r in replies for b in r.get("bad", [])][:3],
                        "bytes_ok": all(r.get("bytes_ok", True) for r in replies),
                        "short": seconds < 1.0,
                    }
                    windows.append(window)
                    # Steer the next window to the target; aim above it after a short one.
                    aim = a.target * (1.2 if window["short"] else 1.0)
                    n = max(floor, math.ceil(n * aim / max(seconds, 1e-6)))
                    if not window["short"]:
                        break
            record["windows"] = windows
            procs1, t1 = processes(), time.time()
            record["exit"] = cell.ask([{"cmd": "exit"}] * cell.workers, a.step_timeout)
            record["foreign_cores"], record["foreign_top"] = foreign(
                procs0, procs1, t1 - t0
            )
            record["status"] = "ok"
            peak = record["exit"][0].get("java", record["exit"][0])["vmhwm_kb"]
            if mode == "1c":
                memory[name] = peak
        except Exception as error:
            record["status"], record["reason"] = (
                "failed",
                f"{type(error).__name__}: {error}"[:400],
            )
        finally:
            if cell:
                cell.kill()
            sampler.tag = "between"
    record["ended"] = time.time()
    record["machine"] = sampler.summary(
        record["started"], record["ended"], record.get("cpus", [])
    )
    if mode == "gpu":
        record["gpu_after"] = gpu_state()
    return record


def check_ready(replies):
    """Raise the reason of the first worker that could not load."""
    for r in replies:
        if r.get("event") != "ready":
            raise RuntimeError(r["reason"])


def least_busy_cpu():
    a = cpu_counters()
    time.sleep(5)
    b = busy(a, cpu_counters())
    return min(SIXTEEN, key=lambda c: b[c] + b[c + 16])


def environment(a):
    def sh(cmd):
        return subprocess.run(
            cmd, shell=True, capture_output=True, check=False, text=True
        ).stdout.strip()

    files = sorted(
        p
        for p in HERE.rglob("*")
        if p.is_file()
        and "target" not in p.parts
        and p.suffix in (".py", ".rs", ".toml", ".sh")
    )
    return {
        "host": sh("uname -srm"),
        "cpu": sh("lscpu | sed -n 's/^Model name: *//p'"),
        "governor": sh(
            f"cat /sys/devices/system/cpu/cpu{a.cpu1}/cpufreq/scaling_governor"
        ),
        "boost": sh("cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null"),
        "thp": sh("cat /sys/kernel/mm/transparent_hugepage/enabled"),
        "cpu1": a.cpu1,
        "cpu1_siblings": sh(
            f"cat /sys/devices/system/cpu/cpu{a.cpu1}/topology/thread_siblings_list"
        ),
        "cpus16": a.cpus16,
        "gpu": sh(
            "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"
        ),
        "java": sh("java -version 2>&1 | head -1"),
        "rapl_readable": os.access(
            "/sys/class/powercap/intel-rapl:0/energy_uj", os.R_OK
        ),
        "perf_event_paranoid": sh("cat /proc/sys/kernel/perf_event_paranoid"),
        "native": str(a.native),
        "native_sha256": sha256(a.native),
        "udlex_wheel": sh(f"ls {a.run}/wheels"),
        "udlex_so_sha256": sh(
            f"sha256sum {a.run}/.venv-udlex/lib/python3.12/site-packages/udlex/_udlex.abi3.so | cut -d' ' -f1"
        ),
        "udlex_source": sh(f"cat {a.run}/src-udlex/COMMIT"),
        "dictionaries": read(a.run / "ref/udlex/manifest.json"),
        "harness_sha256": {str(p.relative_to(HERE)): sha256(p) for p in files},
        "compare_engines_sha256": sha256(a.harness / "compare_engines.py"),
        "loadavg_start": Path("/proc/loadavg").read_text().strip(),
        "note": a.note,
        "owner_jobs_at_start": owner_jobs(a),
        "args": {k: str(v) for k, v in vars(a).items() if k != "func"},
    }


def run(a):
    a.out.mkdir(parents=True, exist_ok=True)
    logs = a.out / "logs"
    logs.mkdir(exist_ok=True)
    if a.cpu1 is None:
        a.cpu1 = least_busy_cpu()
    # A resumed or re-timing run keeps the first environment and adds its own.
    name = "environment.json"
    if (a.out / name).exists():
        name = f"environment-{time.strftime('%Y%m%dT%H%M%S')}.json"
    (a.out / name).write_text(
        json.dumps(environment(a), indent=1, ensure_ascii=False), encoding="utf-8"
    )
    done = set()
    receipts = a.out / "cells.jsonl"
    if receipts.exists():  # resume after a crash: keep finished cells
        done = {
            (r["lang"], r["system"], r["mode"], r["round"])
            for r in map(json.loads, receipts.open())
            if r["status"] != "discarded"
            and not (a.retry_failed and r["status"] == "failed")
        }
    sampler = Sampler(a.out / "load.jsonl")
    sampler.start()
    for iso in a.langs:
        cells = cells_for(iso, a)
        if all(
            (iso, n, m, r) in done for n, m, _ in cells for r in range(1, a.rounds + 1)
        ):
            continue  # everything measured (or reused): skip the idle sample
        memory = {}
        sampler.tag = f"{iso}/before"
        before = processes()
        time.sleep(10)
        idle = {
            "foreign": foreign(before, processes(), 10),
            "machine": sampler.summary(time.time() - 10, time.time(), []),
        }
        for rnd in range(1, a.rounds + 1):
            order = cells[:]
            random.Random(f"{iso}/{rnd}").shuffle(order)
            for k, (name, mode, cfg) in enumerate(order):
                if (iso, name, mode, rnd) in done:
                    continue
                trial = cfg
                for _ in range(3):
                    paused = wait_quiet(a)
                    record = run_cell(name, mode, trial, rnd, a, sampler, logs, memory)
                    record["paused_before_s"] = paused
                    record["order"] = k
                    if rnd == 1 and k == 0:
                        record["session_before"] = idle
                    jobs = owner_jobs(a)
                    oom = (
                        (name, mode) == ("stanza", "gpu")
                        and "stanza_kwargs" not in trial
                        and record["status"] == "failed"
                        and "out of memory" in record.get("reason", "").lower()
                    )
                    if jobs:  # the owner's job started during the cell: time it again
                        record["status"], record["owner_jobs_after"] = "discarded", jobs
                    elif oom:  # Stanza's own POS batch does not fit: the fallback one
                        record["status"] = "discarded"
                        tokens = {"pos_batch_maximum_tokens": a.stanza_gpu_fallback}
                        trial = dict(cfg, stanza_kwargs=tokens)
                    with receipts.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if not jobs and not oom:
                        break
                rates = [
                    w["tokens"] / w["seconds"]
                    for w in record.get("windows", [])
                    if not w["short"]
                ]
                print(
                    f"{time.strftime('%H:%M:%S')} {iso} r{rnd} {name:9s} {mode:6s} {record['status']:7s}"
                    f" {statistics.median(rates) if rates else 0:12.0f} w/s"
                    f" mism={sum(w['mismatch'] for w in record.get('windows', []))}"
                    f" foreign={record.get('foreign_cores')} {record.get('reason', '')}",
                    flush=True,
                )
        sampler.tag = f"{iso}/after"
        after = processes()
        time.sleep(10)
        with (a.out / "sessions.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "lang": iso,
                        "before": idle,
                        "after": {
                            "foreign": foreign(after, processes(), 10),
                            "machine": sampler.summary(
                                time.time() - 10, time.time(), []
                            ),
                        },
                    }
                )
                + "\n"
            )
    sampler.stop.set()
    print("RUN-DONE", flush=True)


# ----------------------------------------------------------------------------- summary


def summarize(a):
    every = [
        json.loads(line) for line in (a.out / "cells.jsonl").open(encoding="utf-8")
    ]
    rows = [r for r in every if r["status"] != "discarded"]  # re-timed ones
    timed = {
        (r["lang"], r["system"], r["mode"], r["round"])
        for r in rows
        if r["status"] == "ok"
    }

    def key(r):
        return (r["lang"], r["system"], r["mode"], r["round"])

    superseded = [r for r in rows if r["status"] == "failed" and key(r) in timed]
    rows = [r for r in rows if not (r["status"] == "failed" and key(r) in timed)]
    env = read(a.out / "environment.json")
    env["discarded"] = [r for r in every if r["status"] == "discarded"] + superseded
    dicts = env["dictionaries"]["languages"]
    groups = {}
    for r in rows:
        groups.setdefault((r["lang"], r["system"], r["mode"]), []).append(r)
    table = {}
    for (iso, name, mode), cells in sorted(groups.items()):
        ok = [c for c in cells if c["status"] == "ok"]
        rounds = []
        for c in ok:
            rates = [w["tokens"] / w["seconds"] for w in c["windows"] if not w["short"]]
            if rates:
                rounds.append(statistics.median(rates))
        entry = {
            "rounds": len(rounds),
            "failed": [c.get("reason") for c in cells if c["status"] != "ok"][:2],
            "checked": sum(w["checked"] for c in ok for w in c["windows"]),
            "mismatch": sum(w["mismatch"] for c in ok for w in c["windows"]),
        }
        if rounds:
            entry |= {
                "wps": statistics.median(rounds),
                "wps_min": min(rounds),
                "wps_max": max(rounds),
                "window_seconds_min": min(
                    w["seconds"] for c in ok for w in c["windows"] if not w["short"]
                ),
                "cpu_util": statistics.median(
                    w["cpu_s"] / w["seconds"]
                    for c in ok
                    for w in c["windows"]
                    if not w["short"]
                ),
                "ttfa_s": statistics.median(
                    statistics.median(r["load_s"] + r["first_s"] for r in c["ready"])
                    for c in ok
                ),
                "import_s": statistics.median(
                    statistics.median(r["import_s"] for r in c["ready"]) for c in ok
                ),
                "peak_rss_mb": statistics.median(
                    sum(x.get("java", x)["vmhwm_kb"] for x in c["exit"]) for c in ok
                )
                / 1024,
                "foreign_cores_median": statistics.median(
                    c.get("foreign_cores", 0) for c in ok
                ),
                "foreign_cores_max": max(c.get("foreign_cores", 0) for c in ok),
                "pinned_busy_pct": statistics.median(
                    (c["machine"].get("pinned_busy_pct") or 0) for c in ok
                ),
                "gpu_util_max": max(
                    (c["machine"].get("gpu_util_max") or 0) for c in ok
                ),
                "model": ok[0]["ready"][0].get("model"),
            }
            cuda = [
                x.get("cuda_reserved_kb")
                for c in ok
                for x in c["exit"]
                if x.get("cuda_reserved_kb")
            ]
            if cuda:
                entry["cuda_reserved_mb"] = statistics.median(cuda) / 1024
            ready = ok[0]["ready"][0]
            if name.startswith("udlex"):
                entry["disk_mb"] = dicts[iso]["bundle_bytes"] / 2**20
                entry["archive_mb"] = dicts[iso].get("archive_bytes", 0) / 2**20
            else:
                entry["disk_mb"] = (
                    ready.get("disk_bytes")
                    or size(*(f for f in ready["files"] if "!" not in f))
                ) / 2**20
        table[f"{iso}/{name}/{mode}"] = entry
    (a.out / "summary.json").write_text(
        json.dumps(table, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    (a.out / "SUMMARY.md").write_text(render(table, env, rows), encoding="utf-8")
    print((a.out / "SUMMARY.md").read_text(encoding="utf-8"))


def fmt(x):
    if x is None:
        return "—"
    return f"{x:,.0f}".replace(",", " ") if x >= 100 else f"{x:.3g}"


def render(table, env, rows):
    langs = list(dict.fromkeys(k.split("/")[0] for k in table))
    systems = list(dict.fromkeys(k.split("/")[1] for k in table))

    def get(iso, s, m):
        return table.get(f"{iso}/{s}/{m}", {})

    def best(iso, name):
        return max(get(iso, name, m).get("wps") or 0 for m in ("thr16", "proc16"))

    out = [
        "# Lemmatizer speed and memory (C9)",
        "",
        (
            f"Machine: {env['cpu']}, {env['host']}; GPU: {env['gpu']}. 1 core: CPU {env['cpu1']} "
            f"(SMT sibling {env['cpu1_siblings']}), 16 cores: CPUs {env['cpus16']}"
            " (0–15: one thread per physical core)."
        ),
        (
            "Numbers are words (UD 2.19 test tokens) per second: median over rounds, min–max of rounds in brackets;"
            " a round is the median of its windows (≥ 1 s, a new random sample of sentences with replacement)."
        ),
        "Load time runs from the start of model construction to the first answer (library import excluded), median over rounds.",
        f"Run note: {env.get('note') or '—'}",
        "",
    ]
    for iso in langs:
        out += [
            f"## {iso}",
            "",
            "| system | 1 core | 16 cores (mode) | GPU | load, s | peak RSS 1 core, MB | RSS 16, MB | disk, MB | answer check |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for s in systems:
            one = get(iso, s, "1c")
            if not one and not any(get(iso, s, m) for m in ("thr16", "proc16", "gpu")):
                continue
            multi = max(
                (
                    (m, get(iso, s, m))
                    for m in ("thr16", "proc16")
                    if get(iso, s, m).get("wps")
                ),
                key=lambda x: x[1]["wps"],
                default=(None, {}),
            )
            gpu = get(iso, s, "gpu")
            checked = sum(get(iso, s, m).get("checked", 0) for m in MODES.get(s, []))
            wrong = sum(get(iso, s, m).get("mismatch", 0) for m in MODES.get(s, []))

            def cell(e):
                if not e.get("wps"):
                    return "not meas." + (
                        f" ({e['failed'][0][:60]})" if e.get("failed") else ""
                    )
                return f"{fmt(e['wps'])} ({fmt(e['wps_min'])}–{fmt(e['wps_max'])})"

            out.append(
                f"| {s} | {cell(one)} | {cell(multi[1]) + (f' {multi[0]}' if multi[0] else '')} | {cell(gpu) if gpu else '—'} "
                f"| {fmt(one.get('ttfa_s'))} | {fmt(one.get('peak_rss_mb'))} | {fmt(multi[1].get('peak_rss_mb'))} "
                f"| {fmt(one.get('disk_mb'))} | {wrong}/{checked} |"
            )
        out.append("")
    out += ["## udlex speed relative to competitors", ""]
    for ours in ("udlex-rs", "udlex-py"):
        out += [
            f"### {ours}",
            "",
            "| competitor | " + " | ".join(langs) + " | geom. mean (n) |",
            "|---|" + "---|" * (len(langs) + 1),
        ]
        for s in systems:
            if s.startswith("udlex"):
                continue
            for label, mine, theirs in (
                ("1 core", "1c", "1c"),
                ("16 cores", None, None),
                ("vs GPU, udlex 16 cores", None, "gpu"),
            ):
                ratios = []
                cells = []
                for iso in langs:
                    a_ = get(iso, ours, mine).get("wps") if mine else best(iso, ours)
                    b_ = get(iso, s, theirs).get("wps") if theirs else best(iso, s)
                    if a_ and b_:
                        ratios.append(a_ / b_)
                        cells.append(
                            f"{a_ / b_:,.0f}×".replace(",", " ")
                            if a_ / b_ >= 10
                            else f"{a_ / b_:.1f}×"
                        )
                    else:
                        cells.append("—")
                if ratios:
                    g = math.exp(statistics.mean(math.log(r) for r in ratios))
                    out.append(
                        f"| {s}, {label} | "
                        + " | ".join(cells)
                        + f" | {g:,.0f}× ({len(ratios)}) |".replace(",", " ")
                    )
        out.append("")
    out += [
        "## Machine load",
        "",
        "Foreign processes: CPU cores used by everything except the harness and its workers during a cell.",
        "",
    ]
    out += [
        "| language | cells | foreign cores, median | max | load1 max | busy share of our CPUs, median % | GPU util max (GPU cells) |",
        "|---|---|---|---|---|---|---|",
    ]
    for iso in langs:
        mine = [r for r in rows if r["lang"] == iso and r["status"] == "ok"]
        if not mine:
            continue
        f_ = [r.get("foreign_cores", 0) for r in mine]
        g_ = [r["machine"].get("gpu_util_max") or 0 for r in mine if r["mode"] == "gpu"]
        out.append(
            f"| {iso} | {len(mine)} | {statistics.median(f_):.2f} | {max(f_):.2f} | {max(r['machine'].get('load1_max', 0) for r in mine):.2f} "
            f"| {statistics.median((r['machine'].get('pinned_busy_pct') or 0) for r in mine):.0f} | {max(g_) if g_ else '—'} |"
        )
    out.append("")
    out += ["## Batch sizes and re-measurements", ""]
    seen = {}
    for r in rows:
        if r["status"] == "ok":
            key = (
                r["system"],
                r["mode"],
                json.dumps(r.get("settings", {}), sort_keys=True),
            )
            seen.setdefault(key, []).append(r["lang"])
    for (system, mode, settings), where in sorted(seen.items()):
        if settings != "{}":
            out.append(
                f"- {system} {mode}: {settings} ({', '.join(sorted(set(where)))})"
            )
    for r in env.get("discarded", []):
        why = r.get("owner_jobs_after") or r.get("reason", "")
        out.append(
            f"- re-measured: {r['lang']} {r['system']} {r['mode']} r{r['round']} — {str(why)[:120]}"
        )
    out.append("")
    failed = [r for r in rows if r["status"] != "ok"]
    if failed:
        out += ["## Not measured", ""]
        seen = set()
        for r in failed:
            key = (r["lang"], r["system"], r["mode"], r.get("reason", "")[:80])
            if key not in seen:
                seen.add(key)
                out.append(
                    f"- {r['lang']} {r['system']} {r['mode']}: {r['status']} — {r.get('reason', '')[:200]}"
                )
        out.append("")
    return "\n".join(out)


def verify(a):
    """The installed udlex answers every test sentence exactly as the references say."""
    import udlex

    so = next(Path(udlex.__file__).parent.glob("_udlex*.so"))
    report = {"udlex_file": udlex.__file__, "so_sha256": sha256(so), "languages": {}}
    for iso, row in read(a.run / "ref/udlex/manifest.json")["languages"].items():
        lexicon = udlex.load(row["bundle"], iso, use_user=False)
        with (inputs() / iso / "test.jsonl").open(encoding="utf-8") as fh:
            tokens = [json.loads(line)["tokens"] for line in fh]
        mine = json.dumps(lexicon.lemmatize_many(tokens), ensure_ascii=False)
        theirs = (a.run / "ref/udlex" / f"{iso}.json").read_text(encoding="utf-8")
        report["languages"][iso] = mine == theirs
    path = a.run / "ref/udlex" / f"verify-{report['so_sha256'][:16]}.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    bad = [iso for iso, same in report["languages"].items() if not same]
    print(
        f"{path}: "
        + (f"DIFFER {' '.join(bad)}" if bad else "identical on every language")
    )
    raise SystemExit(1 if bad else 0)


def choose(a):
    """The batch of every swept cell on eng; prints bench.py run options."""

    def wps(name, key):
        path = a.sweep / name / "summary.json"
        return read(path).get(key, {}).get("wps") if path.exists() else None

    def best(cands):
        # The smallest setting within 5 % of the fastest: past saturation a bigger
        # batch only lengthens the windows.
        cands = [(w, v) for w, v in cands if w]
        top = max((w for w, _ in cands), default=0)
        return min((v for w, v in cands if w >= 0.95 * top), default=None)

    chosen = {}
    for mode in ("1c", "thr16"):
        chosen[f"udlex-py:{mode}"] = best(
            (wps(f"udlex-batch-{n}", f"eng/udlex-py/{mode}"), n)
            for n in (1, 10, 100, 1000, 10000)
        )
    for system in sorted(NEURAL):
        for mode in ("thr16", "gpu"):
            chosen[f"{system}:{mode}"] = best(
                (wps(f"neural-{n}", f"eng/{system}/{mode}"), n)
                # on CPU 1024 sentences make one Trankit window last minutes
                for n in ((16, 64, 256) if mode == "thr16" else (16, 64, 256, 1024))
            )
    chosen["stanza-gpu-tokens"] = best(
        (wps(name, "eng/stanza/gpu"), tokens)
        for name, tokens in (
            ("neural-1024", 2000),
            ("stanza-gpu-5000", 5000),
            ("stanza-gpu-1000", 1000),
        )
    )
    one = chosen["udlex-py:1c"] or 1000
    batch = [str(one), f"proc16={one}"] + (
        [f"thr16={chosen['udlex-py:thr16']}"] if chosen["udlex-py:thr16"] else []
    )
    neural = ["256"] + [
        f"{k}={v}"
        for k, v in chosen.items()
        if ":" in k and not k.startswith("udlex") and v
    ]
    options = ["--batch", *batch, "--neural-batch", *neural]
    if chosen["stanza-gpu-tokens"]:
        options += ["--stanza-gpu-tokens", str(chosen["stanza-gpu-tokens"])]
    a.sweep.mkdir(parents=True, exist_ok=True)
    (a.sweep / "chosen.json").write_text(
        json.dumps({"chosen": chosen, "options": options}, indent=1)
    )
    print(" ".join(options))


def check(a):
    """A dry run must have run udlex everywhere with identical answers and most cells."""
    rows = [json.loads(line) for line in (a.out / "cells.jsonl").open(encoding="utf-8")]
    rows = [
        r for r in rows if r["status"] != "discarded"
    ]  # re-timed after the owner's job
    ok = [r for r in rows if r["status"] == "ok"]
    ours = [r for r in rows if r["system"].startswith("udlex")]
    problems = [
        f"{r['lang']} {r['system']} {r['mode']}: {r['status']} {r.get('reason', '')}"
        for r in ours
        if r["status"] != "ok"
        or any(w["mismatch"] or not w["bytes_ok"] for w in r["windows"])
    ]
    if len(ok) < 0.7 * len(rows):
        problems.append(f"only {len(ok)} of {len(rows)} cells ran")
    print("\n".join(problems) or f"fit: {len(ok)}/{len(rows)} cells ok")
    raise SystemExit(1 if problems else 0)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("prepare")
    s.add_argument("run", type=Path)
    s.add_argument("langs", nargs="*", default=LANGS)
    s.add_argument(
        "--evaluated", action="store_true", help="bundles evaluate.py scored"
    )
    s.set_defaults(func=prepare)
    r = sub.add_parser("run")
    r.add_argument("run", type=Path)
    r.add_argument("out", type=Path)
    r.add_argument("--langs", nargs="+", default=LANGS)
    r.add_argument("--systems", nargs="+")
    r.add_argument("--modes", nargs="+")
    r.add_argument("--rounds", type=int, default=5)
    r.add_argument("--windows", type=int, default=3)
    r.add_argument("--target", type=float, default=2.0, help="seconds a window aims at")
    r.add_argument(
        "--warm", type=float, default=3.0, help="seconds of warm-up per cell"
    )
    r.add_argument(
        "--batch",
        nargs="+",
        default=["1000"],
        help="udlex-py sentences per call: N, mode=N",
    )
    r.add_argument(
        "--cpu1", type=int, help="the single-core CPU; default the least busy"
    )
    r.add_argument("--cpus16", type=int, nargs="+", default=SIXTEEN)
    r.add_argument(
        "--neural-batch",
        nargs="+",
        default=["256"],
        help="sentences per call with threads or GPU: N and/or system:mode=N",
    )
    r.add_argument(
        "--stanza-gpu-tokens",
        type=int,
        default=0,
        help="Stanza POS batch on GPU in tokens; 0 keeps Stanza's own",
    )
    r.add_argument(
        "--stanza-gpu-fallback",
        type=int,
        default=2000,
        help="POS batch when Stanza's own runs out of GPU memory",
    )
    r.add_argument("--gpu-free-mb", type=int, default=3500)
    r.add_argument("--load-timeout", type=float, default=900)
    r.add_argument("--step-timeout", type=float, default=900)
    r.add_argument("--native", type=Path)
    r.add_argument("--note", default="", help="goes into environment.json")
    r.add_argument(
        "--pause-while", nargs="*", default=[], help="wait while these exist"
    )
    r.add_argument("--pause-tmux", nargs="*", default=[], help="wait while these live")
    r.add_argument(
        "--retry-failed", action="store_true", help="time again the failed cells of OUT"
    )
    r.add_argument(
        "--harness",
        type=Path,
        default=HERE.parent,
        help="directory with compare_engines.py",
    )
    r.set_defaults(func=run)
    m = sub.add_parser("summary")
    m.add_argument("out", type=Path)
    m.set_defaults(func=summarize)
    c = sub.add_parser("choose", help="batch options for the final from a sweep")
    c.add_argument("sweep", type=Path)
    c.set_defaults(func=choose)
    v = sub.add_parser("verify", help="installed udlex against the reference answers")
    v.add_argument("run", type=Path)
    v.set_defaults(func=verify)
    k = sub.add_parser("check", help="exit 1 unless a dry run is fit for the final")
    k.add_argument("out", type=Path)
    k.set_defaults(func=check)
    a = p.parse_args()
    if hasattr(a, "run") and isinstance(a.run, Path):
        a.run = a.run.resolve()
    if a.mode == "run" and a.native is None:
        a.native = a.run / "target/release/udlex-speed"
    a.func(a)


if __name__ == "__main__":
    main()
