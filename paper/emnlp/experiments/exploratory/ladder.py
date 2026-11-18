"""Exploratory: what a tag-keyed table and a suffix guesser add to the cascade on every language.

usage: ladder.py CENSUS UD PREDICTIONS OUTPUT [ISO...]

Neither addition is neural: a linear window tagger picks the lemma of a form seen with several
lemmas through a (form, tag) table, and suffix decision lists fitted on rare training forms answer
words that reach the rule or copy stage. Tags for the table are gold training tags, not jackknifed.
"""

import collections
import gzip
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier


def conllu(path):
    sent = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) == 10 and p[0].isdecimal():
                sent.append((p[1], p[3]))
            elif not line.strip() and sent:
                yield sent
                sent = []
    if sent:
        yield sent


def feats(words, i):
    w = words[i]
    lw = w.lower()
    f = ["b", "w=" + lw, "cap" if w[:1].isupper() else "low"]
    f += [f"s{k}=" + lw[-k:] for k in range(1, 5) if len(lw) >= k]
    f += [f"p{k}=" + lw[:k] for k in (1, 2) if len(lw) >= k]
    if any(c.isdigit() for c in w):
        f.append("dig")
    if "-" in w:
        f.append("hyp")
    for d, name in ((-1, "l1"), (1, "r1"), (-2, "l2"), (2, "r2")):
        j = i + d
        if 0 <= j < len(words):
            x = words[j].lower()
            f.append(name + "=" + x)
            if abs(d) == 1:
                f.append(name + "s3=" + x[-3:])
        else:
            f.append(name + "=<pad>")
    if i == 0:
        f.append("first")
    return f


def script(form, lemma):
    """Edit script: lower-case or not, characters to cut, suffix to add."""
    lf = form.lower()
    low = lemma != form and lemma[:1].islower() and form[:1].isupper()
    base = lf if low or lemma == lf else form
    k = len(os.path.commonprefix([base, lemma]))
    return (base is lf, len(base) - k, lemma[k:])


def apply(form, s):
    base = form.lower() if s[0] else form
    return None if s[1] > len(base) else base[: len(base) - s[1]] + s[2]


def run(job):
    census, ud, predictions, iso = job
    row = json.loads((census / "inputs/order/inventory.json").read_text())["languages"][
        iso
    ]
    tags = {}
    for split in ("train", "dev", "test"):
        path = ud / row["treebank"] / f"{row['config']}-ud-{split}.conllu"
        if path.exists():
            for s in conllu(path):
                tags.setdefault(tuple(w for w, _ in s), [u for _, u in s])

    def load(split):
        out = []
        with open(census / "inputs/order" / iso / f"{split}.jsonl") as f:
            for line in f:
                s = json.loads(line)
                out.append((s["tokens"], s["lemmas"], tags.get(tuple(s["tokens"]))))
        return out

    train, test = load("train"), load("test")
    if any(us is None for _, _, us in train + test):
        return iso, {"status": "sentences without tags"}

    h = FeatureHasher(n_features=2**20, input_type="string")
    y = np.array([u for _, _, us in train for u in us])
    if len(set(y)) < 2:
        return iso, {"status": "one tag"}
    clf = SGDClassifier(
        loss="hinge",
        alpha=1e-6 if len(y) > 200000 else 1e-5,
        max_iter=12,
        tol=None,
        random_state=0,
        n_jobs=2,
    ).fit(h.transform(feats(w, i) for w, _, _ in train for i in range(len(w))), y)
    guessed = clf.predict(
        h.transform(feats(w, i) for w, _, _ in test for i in range(len(w)))
    )
    gold = np.array([u for _, _, us in test for u in us])

    byf = collections.defaultdict(collections.Counter)
    byp = collections.defaultdict(collections.Counter)
    for w, ls, us in train:
        for f, lemma, u in zip(w, ls, us, strict=True):
            if lemma is not None:
                byf[f][lemma] += 1
                byp[(f, u)][lemma] += 1
    lists = collections.defaultdict(collections.Counter)
    for w, ls, us in train:
        for f, lemma, u in zip(w, ls, us, strict=True):
            # Rare forms resemble the unseen ones the lists will answer.
            if lemma is None or sum(byf[f].values()) > 2:
                continue
            lf = f.lower()
            for k in range(min(7, len(lf)) + 1):
                lists[(lf[len(lf) - k :], u)][script(f, lemma)] += 1

    def guess(f, u):
        lf = f.lower()
        for k in range(min(7, len(lf)), -1, -1):
            for s, _ in lists.get(
                (lf[len(lf) - k :], u), collections.Counter()
            ).most_common(3):
                answer = apply(f, s)
                if answer:
                    return answer
        return f

    result = json.loads((census / "results" / f"{iso}.json").read_text())
    answers = {a[0]: a[1:] for a in result["answers"]}
    theirs = {}
    for engine in ("udpipe2", "stanza"):
        file = predictions / engine / f"{iso}.json.gz"
        if file.exists():
            theirs[engine] = json.loads(gzip.decompress(file.read_bytes()))
    c = collections.Counter()
    j = 0
    for si, (w, ls, _) in enumerate(test):
        for i, (f, lemma) in enumerate(zip(w, ls, strict=True)):
            if lemma is None:
                continue
            u = guessed[j + i]
            a = answers[f]
            text, stage = (a[2], a[3]) if i == 0 else (a[0], a[1])
            tagged = text
            if len(byf.get(f, ())) > 1:
                tagged = (byp.get((f, u)) or byf[f]).most_common(1)[0][0]
            full = guess(f, u) if f not in byf and stage in (7, 8) else tagged
            c["n"] += 1
            c["cascade"] += text == lemma
            c["tag"] += tagged == lemma
            c["tag+guess"] += full == lemma
            for engine, sentences in theirs.items():
                c[engine] += sentences[si][i] == lemma
        j += len(w)
    if c["cascade"] != result["test"]["counts"][0]["correct"]:
        raise ValueError(f"{iso}: answers disagree with the scored run")
    return iso, dict(
        c,
        status="scored",
        tag_accuracy=float((guessed == gold).mean()),
        sources=len(row["sources"]),
        engines=sorted(theirs),
    )


def main():
    census, ud, predictions, output, *isos = sys.argv[1:]
    census, ud, predictions = Path(census), Path(ud), Path(predictions)
    isos = isos or sorted(
        p.stem for p in (census / "results").glob("*.json") if p.stem != "run"
    )
    with ProcessPoolExecutor(5) as pool:
        report = dict(pool.map(run, [(census, ud, predictions, i) for i in isos]))
    Path(output).write_text(json.dumps(report, sort_keys=True, indent=1))
    scored = {i: c for i, c in report.items() if c["status"] == "scored"}
    print(
        len(scored),
        "languages;",
        {i: c["status"] for i, c in report.items() if i not in scored},
    )

    def macro(cells, key):
        return sum(100 * c[key] / c["n"] for c in cells) / len(cells)

    for label, cells in (
        ("all", list(scored.values())),
        ("no dictionary", [c for c in scored.values() if c["sources"] == 0]),
        ("one dictionary", [c for c in scored.values() if c["sources"] == 1]),
        ("two or more", [c for c in scored.values() if c["sources"] >= 2]),
        ("with UDPipe 2", [c for c in scored.values() if "udpipe2" in c["engines"]]),
        ("with Stanza", [c for c in scored.values() if "stanza" in c["engines"]]),
    ):
        if not cells:
            continue
        line = f"{label:15s} langs={len(cells):3d} tagger={macro(cells, 'tag_accuracy') / 100 * 1:.3f}"
        for key in ("cascade", "tag", "tag+guess"):
            line += f" {key}={macro(cells, key):.2f}"
        for engine in ("udpipe2", "stanza"):
            if all(engine in c["engines"] for c in cells):
                line += f" {engine}={macro(cells, engine):.2f}"
                line += f" (ladder not below in {sum(c['tag+guess'] >= c[engine] for c in cells)})"
        print(line)


if __name__ == "__main__":
    main()
