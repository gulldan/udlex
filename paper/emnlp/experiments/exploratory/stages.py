"""Exploratory: accuracy of the declared cascade by stage, on forms seen and unseen in training."""

import collections
import glob
import json
import os
from pathlib import Path

COMPARE_DIR = os.environ["UDLEX_COMPARE"]
ORDER_RUN = os.environ["UDLEX_ORDER_RUN"]
TRANSFER_RUN = os.environ["UDLEX_TRANSFER_RUN"]

ORDER = f"{ORDER_RUN}/clean219/inputs/order"
D = f"{TRANSFER_RUN}/anchor-diag3/results"
STAGE = {
    2: "corpus",
    3: "corpus",
    4: "corpus",
    5: "external",
    6: "external",
    7: "rule",
    8: "copy",
}
tot = collections.Counter()
per = {}
for path in sorted(glob.glob(D + "/*.json")):  # noqa: PTH207 - glob.glob skips dotfiles
    iso = path.split("/")[-1][:-5]
    if iso == "run":
        continue
    r = json.loads(Path(path).read_text())
    ans = {a[0]: a[1:] for a in r["answers"]}
    train = set()
    with Path(f"{ORDER}/{iso}/train.jsonl").open() as lines:
        for line in lines:
            s = json.loads(line)
            train.update(
                f
                for f, y in zip(s["tokens"], s["lemmas"], strict=True)
                if y is not None
            )
    c = collections.Counter()
    with Path(f"{ORDER}/{iso}/test.jsonl").open() as lines:
        for line in lines:
            s = json.loads(line)
            for i, (f, y) in enumerate(zip(s["tokens"], s["lemmas"], strict=True)):
                if y is None:
                    continue
                a = ans[f]
                text, stage = (a[2], a[3]) if i == 0 else (a[0], a[1])
                k = ("seen" if f in train else "unseen", STAGE.get(stage, str(stage)))
                c[(*k, "n")] += 1
                c[(*k, "ok")] += text == y
    assert (
        sum(v for k, v in c.items() if k[2] == "ok")
        == r["test"]["counts"][0]["correct"]
    ), iso
    per[iso] = c
    tot.update(c)
n = sum(v for k, v in tot.items() if k[2] == "n")
print("pooled over", len(per), "languages,", n, "words")
for seen in ("seen", "unseen"):
    sn = sum(v for k, v in tot.items() if k[0] == seen and k[2] == "n")
    so = sum(v for k, v in tot.items() if k[0] == seen and k[2] == "ok")
    print(f" {seen}: {100 * sn / n:.1f}% of words, accuracy {100 * so / sn:.2f}")
    for st in ("corpus", "external", "rule", "copy"):
        a, b = tot[(seen, st, "n")], tot[(seen, st, "ok")]
        if a:
            print(
                f"    {st:9s} {100 * a / n:5.2f}% of words  accuracy {100 * b / a:.2f}  errors {a - b}"
            )
# macro over languages of unseen accuracy
ua = [
    100
    * sum(v for k, v in c.items() if k[0] == "unseen" and k[2] == "ok")
    / max(1, sum(v for k, v in c.items() if k[0] == "unseen" and k[2] == "n"))
    for c in per.values()
]
sa = [
    100
    * sum(v for k, v in c.items() if k[0] == "seen" and k[2] == "ok")
    / max(1, sum(v for k, v in c.items() if k[0] == "seen" and k[2] == "n"))
    for c in per.values()
]
print(
    f"macro: accuracy on seen {sum(sa) / len(sa):.2f}, on unseen {sum(ua) / len(ua):.2f}"
)
Path(f"{COMPARE_DIR}/stages.json").write_text(
    json.dumps({i: {"|".join(k): v for k, v in c.items()} for i, c in per.items()})
)
