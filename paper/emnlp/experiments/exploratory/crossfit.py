"""Exploratory: learn the order on training words whose form is absent from the other folds,
then score that order through the full cascade on the official dev and test splits."""
import gzip, hashlib, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

R = Path("/mnt/hdd-4tb/codex-runs/udlex-v1-order-ud219-20261118.splMl6")
SRC = "/mnt/hdd-4tb/lemmas-bench/bench-v280/repo"
BIN = str(R / "cargo-target/dev-fast/paper_order")
ORDER = R / "clean219/inputs/order"
OUT = Path("/mnt/hdd-4tb/codex-runs/udlex-transfer-20261118-full/crossfit")
K = 5
inv = json.loads((ORDER / "inventory.json").read_bytes())["languages"]
home = json.loads(gzip.decompress(Path("/mnt/hdd-4tb/codex-runs/udlex-transfer-20261118-full/study/evidence/ud219/order/pairs.json.gz").read_bytes()))["official"]

def run(root, iso, row, train, dev, test, out):
    d = root / iso
    d.mkdir(parents=True, exist_ok=True)
    for name, src in (("train", train), ("dev", dev), ("test", test)):
        p = d / f"{name}.jsonl"
        if p.exists(): p.unlink()
        if src is None: p.write_bytes(b"")
        else: os.link(src, p)
    (root / "inventory.json").write_text(json.dumps({"languages": {iso: dict(row, status="paired")}}))
    subprocess.run([BIN, SRC, str(root), iso, str(out)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.loads(out.read_bytes())

def one(iso):
    row = inv[iso]
    sents = [json.loads(l) for l in (ORDER / iso / "train.jsonl").open()]
    fold = [int.from_bytes(hashlib.sha256(s["block"].encode()).digest()[:4], "big") % K for s in sents]
    seen = [set() for _ in range(K)]
    for s, k in zip(sents, fold):
        seen[k].update(f.lower() for f, y in zip(s["tokens"], s["lemmas"]) if y is not None)
    work = OUT / "a" / iso
    work.mkdir(parents=True, exist_ok=True)
    kept = total = 0
    with (work / "oov.jsonl").open("w") as fh:
        others = [set().union(*(seen[j] for j in range(K) if j != k)) for k in range(K)]
        for s, k in zip(sents, fold):
            elsewhere = others[k]
            lem = [y if y is not None and f.lower() not in elsewhere else None for f, y in zip(s["tokens"], s["lemmas"])]
            total += sum(y is not None for y in s["lemmas"]); n = sum(y is not None for y in lem); kept += n
            if n: fh.write(json.dumps({"tokens": s["tokens"], "lemmas": lem, "block": s["block"]}, ensure_ascii=False) + "\n")
    a = run(OUT / "a" / iso / "in", iso, row, work / "oov.jsonl", None, None, OUT / "a" / f"{iso}.json")
    order = a["order"][1]
    by = {s["id"]: s for s in row["sources"]}
    row2 = dict(row, sources=[by[i] for i in order])
    b = run(OUT / "b" / iso / "in", iso, row2, ORDER / iso / "train.jsonl", ORDER / iso / "dev.jsonl", ORDER / iso / "test.jsonl", OUT / "b" / f"{iso}.json")
    ref = home[iso][0]
    res = {"iso": iso, "oov_tokens": kept, "train_tokens": total, "declared": ref["order"][0], "learned": ref["order"][1], "crossfit": order}
    for split in ("dev", "test"):
        res[split] = {"tokens": ref[split]["counts"][0]["tokens"], "declared": ref[split]["counts"][0]["correct"],
                      "learned": ref[split]["counts"][1]["correct"], "crossfit": b[split]["counts"][0]["correct"]}
        assert b[split]["counts"][0]["tokens"] == res[split]["tokens"]
    return res

isos = sorted(home, key=lambda i: -sum(s["bytes"] for s in inv[i]["sources"]))
with ThreadPoolExecutor(6) as pool:
    rows = list(pool.map(one, isos))
(OUT / "crossfit.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
print("DONE", len(rows))
