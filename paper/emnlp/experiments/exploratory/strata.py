"""Exploratory: the cascade and Stanza on the same strata of test words."""

import collections
import gzip
import json
import os
from pathlib import Path

COMPARE_DIR = os.environ["UDLEX_COMPARE"]
ORDER_RUN = os.environ["UDLEX_ORDER_RUN"]
TRANSFER_RUN = os.environ["UDLEX_TRANSFER_RUN"]

ORDER = f"{ORDER_RUN}/clean219/inputs/order"
D = f"{TRANSFER_RUN}/anchor-diag3/results"
C = COMPARE_DIR
cov = json.loads(Path(C + "/results/stanza-coverage.json").read_text())
clean = [
    i
    for i, v in cov.items()
    if v["lemma_model"] and 2 * v["covered"] < v["unseen_types"]
]
STR = (
    "seen, one lemma",
    "seen, several lemmas",
    "unseen, corpus lowercase",
    "unseen, in a dictionary",
    "unseen, rule or copy",
)
tot = collections.Counter()
per = {}
for iso in sorted(clean):
    r = json.loads(Path(f"{D}/{iso}.json").read_text())
    ans = {a[0]: a[1:] for a in r["answers"]}
    tm = collections.defaultdict(set)
    with Path(f"{ORDER}/{iso}/train.jsonl").open() as lines:
        for line in lines:
            s = json.loads(line)
            for f, y in zip(s["tokens"], s["lemmas"], strict=True):
                if y is not None:
                    tm[f].add(y)
    st = json.loads(
        gzip.decompress(Path(f"{C}/predictions/stanza/{iso}.json.gz").read_bytes())
    )
    c = collections.Counter()
    with Path(f"{ORDER}/{iso}/test.jsonl").open() as lines:
        for s, sp in zip((json.loads(line) for line in lines), st, strict=True):
            for i, (f, y, z) in enumerate(
                zip(s["tokens"], s["lemmas"], sp, strict=True)
            ):
                if y is None:
                    continue
                a = ans[f]
                text, stage = (a[2], a[3]) if i == 0 else (a[0], a[1])
                if f in tm:
                    k = 0 if len(tm[f]) == 1 else 1
                elif stage in (2, 3, 4):
                    k = 2
                elif stage in (5, 6):
                    k = 3
                else:
                    k = 4
                o = text == y
                t = z == y
                c[(k, "n")] += 1
                c[(k, "ours")] += o
                c[(k, "stanza")] += t
                c[(k, "either")] += o or t
                c[(k, "both")] += o and t
    per[iso] = c
    tot.update(c)
n = sum(tot[(k, "n")] for k in range(5))
print(
    f"{len(per)} languages, {n} words (Stanza models that saw the test or lack a lemmatizer excluded)"
)
print(
    f"{'stratum':26s} {'share':>6s} {'cascade':>8s} {'Stanza':>7s} {'either':>7s}  errors cascade/Stanza"
)
for k, name in enumerate(STR):
    a = tot[(k, "n")]
    print(
        f"{name:26s} {100 * a / n:6.2f} {100 * tot[(k, 'ours')] / a:8.2f} {100 * tot[(k, 'stanza')] / a:7.2f} {100 * tot[(k, 'either')] / a:7.2f}  {a - tot[(k, 'ours')]:6d} / {a - tot[(k, 'stanza')]:6d}"
    )
O_ = sum(tot[(k, "ours")] for k in range(5))
S_ = sum(tot[(k, "stanza")] for k in range(5))
print(
    f"all: cascade {100 * O_ / n:.2f} Stanza {100 * S_ / n:.2f} oracle-of-two {100 * sum(tot[(k, 'either')] for k in range(5)) / n:.2f}"
)


# hybrids: who answers which stratum
def hyb(choice):
    return sum(tot[(k, choice[k])] for k in range(5))


for name, ch in (
    (
        "cascade, Stanza only for rule/copy words",
        ("ours", "ours", "ours", "ours", "stanza"),
    ),
    (
        "cascade on seen + lowercase, Stanza on all unseen",
        ("ours", "ours", "ours", "stanza", "stanza"),
    ),
    ("Stanza on ambiguous + rule/copy", ("ours", "stanza", "ours", "ours", "stanza")),
):
    macro = []
    wins = 0
    for c in per.values():
        nn = sum(c[(k, "n")] for k in range(5))
        h = sum(c[(k, ch[k])] for k in range(5))
        s = sum(c[(k, "stanza")] for k in range(5))
        macro.append(100 * h / nn)
        wins += h >= s
    print(
        f"hybrid [{name}]: pooled {100 * hyb(ch) / n:.2f}, macro {sum(macro) / len(macro):.2f}, >= Stanza in {wins} of {len(per)} languages; share of words sent to Stanza {100 * sum(tot[(k, 'n')] for k in range(5) if ch[k] == 'stanza') / n:.1f}%"
    )
ms = [
    100 * sum(c[(k, "stanza")] for k in range(5)) / sum(c[(k, "n")] for k in range(5))
    for c in per.values()
]
mo = [
    100 * sum(c[(k, "ours")] for k in range(5)) / sum(c[(k, "n")] for k in range(5))
    for c in per.values()
]
print(f"macro: cascade {sum(mo) / len(mo):.2f}, Stanza {sum(ms) / len(ms):.2f}")
Path(C + "/strata.json").write_text(
    json.dumps({i: {f"{k[0]}|{k[1]}": v for k, v in c.items()} for i, c in per.items()})
)
