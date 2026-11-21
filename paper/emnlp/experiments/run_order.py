"""Run a pinned full-cascade pair for every language with multiple sources and independent gold."""

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from verify_snapshot import read, save, sha


def main(a):
    a.output.mkdir(parents=True, exist_ok=True)
    roster = read(a.inputs / "inventory.json")["languages"]
    stamp = {"inventory": sha(a.inputs / "inventory.json"), "executable": sha(a.binary)}
    receipt = a.output / "run.json"
    if receipt.exists() and read(receipt) != stamp:
        raise ValueError(
            "Different executable or inventory already in this run directory"
        )
    save(receipt, stamp)
    todo = [iso for iso, v in roster.items() if v.get("status") == "paired"]
    todo.sort(key=lambda i: -sum(s["bytes"] for s in roster[i]["sources"]))

    def one(iso):
        for split, s in roster[iso]["splits"].items():
            if sha(a.inputs / iso / (split + ".jsonl")) != s["jsonl_sha256"]:
                raise ValueError(f"Split changed for {iso} {split}")
        out = a.output / (iso + ".json")
        if not out.exists():
            with (a.output / (iso + ".log")).open("w") as log:
                subprocess.run(
                    [a.binary, a.research, a.inputs, iso, out],
                    check=True,
                    stdout=log,
                    stderr=log,
                )
        evidence = read(out)
        if evidence["engine"] != "1.0.0" or evidence["iso"] != iso:
            raise ValueError(f"Invalid result for {iso}")
        for split in ("dev", "test"):
            n = roster[iso]["splits"][split].get("tokens", 0)
            if any(c["tokens"] != n for c in evidence[split]["counts"]):
                raise ValueError(f"Token denominator for {iso} {split} disagrees")
        print("Completed", iso, flush=True)

    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        list(pool.map(one, todo))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ("research", "inputs", "binary", "output"):
        p.add_argument(n, type=Path)
    p.add_argument("--jobs", type=int, default=3)
    main(p.parse_args())
