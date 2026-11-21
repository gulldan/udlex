#!/usr/bin/env python3
"""Machine load beside a run, one JSON line a minute.

usage: load_sampler.py <out.jsonl> <cpu> [interval seconds]

Records the load average, how busy the measuring CPU and its SMT sibling
were over the interval, and the five processes that used the most CPU in it
(with the CPU each last ran on), so a receipt shows what else ran and where.
"""

import json
import os
import sys
import time
from pathlib import Path

out, cpu = Path(sys.argv[1]), int(sys.argv[2])
interval = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
siblings = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text().strip()
tick = os.sysconf("SC_CLK_TCK")


def cpus():
    out = {}
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith("cpu") and line[3].isdigit():
            fields = line.split()
            values = list(map(int, fields[1:]))
            out[int(fields[0][3:])] = (sum(values), values[3] + values[4])
    return out


def procs():
    out = {}
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            stat = Path(entry.path, "stat").read_text()
        except OSError:
            continue
        name = stat[stat.index("(") + 1 : stat.rindex(")")]
        rest = stat[stat.rindex(")") + 2 :].split()
        out[int(entry.name)] = (name, int(rest[11]) + int(rest[12]), int(rest[36]))
    return out


before_cpu, before_proc, start = cpus(), procs(), time.time()
while True:
    time.sleep(interval)
    now_cpu, now_proc, now = cpus(), procs(), time.time()
    busy = {
        c: round(100 * (1 - (now_cpu[c][1] - before_cpu[c][1]) / max(1, now_cpu[c][0] - before_cpu[c][0])), 1)
        for c in now_cpu
    }
    used = []
    for pid, (name, jiffies, last_cpu) in now_proc.items():
        if pid in before_proc:
            share = 100 * (jiffies - before_proc[pid][1]) / tick / (now - start)
            if share >= 1:
                used.append((round(share, 1), name, pid, last_cpu))
    used.sort(reverse=True)
    record = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "loadavg": Path("/proc/loadavg").read_text().split()[:3],
        "cpu": cpu,
        "siblings": siblings,
        "busy_measuring": {s: busy[int(s)] for s in siblings.replace("-", ",").split(",")},
        "busy_all_mean": round(sum(busy.values()) / len(busy), 1),
        "top": [{"cpu_pct": s, "name": n, "pid": p, "last_cpu": c} for s, n, p, c in used[:5]],
    }
    with out.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    before_cpu, before_proc, start = now_cpu, now_proc, now
