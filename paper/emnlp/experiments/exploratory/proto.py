"""Exploratory prototype: a linear window tagger for seen ambiguous forms and a
suffix decision list for unseen forms, scored on the strata of the official test."""

import sys, json, gzip, collections, os, time
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier

R = "/mnt/hdd-4tb/codex-runs/udlex-v1-order-ud219-20261118.splMl6"
O = R + "/clean219/inputs/order"
D = "/mnt/hdd-4tb/codex-runs/udlex-transfer-20261118-full/anchor-diag3/results"
C = "/mnt/hdd-4tb/codex-runs/udlex-compare-20261119"
inv = json.load(open(O + "/inventory.json"))["languages"]


def conllu(path):
    sent = []
    for l in open(path, encoding="utf-8"):
        p = l.rstrip("\n").split("\t")
        if len(p) == 10 and p[0].isdecimal():
            sent.append((p[1], p[3]))
        elif not l.strip() and sent:
            yield sent
            sent = []
    if sent:
        yield sent


def feats(words, i):
    w = words[i]
    lw = w.lower()
    f = ["b", "w=" + lw]
    for k in range(1, 5):
        if len(lw) >= k:
            f.append(f"s{k}=" + lw[-k:])
    for k in (1, 2):
        if len(lw) >= k:
            f.append(f"p{k}=" + lw[:k])
    f.append("cap" if w[:1].isupper() else "low")
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
    lf = form.lower()
    low = lemma != form and lemma[:1].islower() and form[:1].isupper()
    base = lf if low or lemma == lf else form
    k = len(os.path.commonprefix([base, lemma]))
    return (base is lf, len(base) - k, lemma[k:])


def apply(form, s):
    base = form.lower() if s[0] else form
    if s[1] > len(base):
        return None
    return base[: len(base) - s[1]] + s[2]


def run(iso):
    t0 = time.time()
    row = inv[iso]
    d = f"{R}/inputs/ud-treebanks-v2.19/{row['treebank']}/{row['config']}-ud-"
    tags = {}
    for split in ("train", "dev", "test"):
        if os.path.exists(d + split + ".conllu"):
            for s in conllu(d + split + ".conllu"):
                tags.setdefault(tuple(w for w, _ in s), [u for _, u in s])

    def load(split):
        out = []
        for l in open(f"{O}/{iso}/{split}.jsonl"):
            s = json.loads(l)
            out.append((s["tokens"], s["lemmas"], tags[tuple(s["tokens"])]))
        return out

    train, test = load("train"), load("test")
    # tagger
    h = FeatureHasher(n_features=2**20, input_type="string")
    X = h.transform(feats(w, i) for w, _, _ in train for i in range(len(w)))
    y = np.array([u for _, _, us in train for u in us])
    clf = SGDClassifier(
        loss="hinge",
        alpha=1e-6 if len(y) > 200000 else 1e-5,
        max_iter=12,
        tol=None,
        random_state=0,
        n_jobs=4,
    ).fit(X, y)
    Xt = h.transform(feats(w, i) for w, _, _ in test for i in range(len(w)))
    pt = clf.predict(Xt)
    gold_t = np.array([u for _, _, us in test for u in us])
    tag_acc = float((pt == gold_t).mean())
    # tables
    byf = collections.defaultdict(collections.Counter)
    byp = collections.defaultdict(collections.Counter)
    freq = collections.Counter()
    for w, ls, us in train:
        for f, l, u in zip(w, ls, us):
            if l is not None:
                byf[f][l] += 1
                byp[(f, u)][l] += 1
                freq[f] += 1

    # suffix decision lists from rare training forms (they resemble unseen ones)
    def build(use_pos, rare):
        t = collections.defaultdict(collections.Counter)
        for w, ls, us in train:
            for f, l, u in zip(w, ls, us):
                if l is None or (rare and freq[f] > rare):
                    continue
                s = script(f, l)
                lf = f.lower()
                for k in range(0, min(7, len(lf)) + 1):
                    t[(lf[len(lf) - k :], u if use_pos else None)][s] += 1
        return t

    lists = {
        "all": build(False, 0),
        "rare": build(False, 2),
        "rare+pos": build(True, 2),
    }
    # the same list from dictionary pairs: other lexemes with the same inflection
    dl = collections.defaultdict(collections.Counter)
    SRC = "/mnt/hdd-4tb/lemmas-bench/bench-v280/repo/"
    npairs = 0
    for src in row["sources"]:
        seenf = set()
        for line in open(SRC + src["path"], encoding="utf-8", errors="replace"):
            cells = line.rstrip(chr(10)).split(chr(9))
            if len(cells) < 2:
                continue
            f, l = (cells[1], cells[0]) if src["lemma_first"] else (cells[0], cells[1])
            f = f.strip('"')
            if not f or not l or f in seenf or " " in f:
                continue
            seenf.add(f)
            npairs += 1
            s = script(f, l)
            lf = f.lower()
            for k in range(2, min(7, len(lf)) + 1):
                dl[lf[len(lf) - k :]][s] += 1
            if len(seenf) >= 400000:
                break

    def guess2(f, u):
        lf = f.lower()
        t = lists["rare+pos"]
        for k in range(min(7, len(lf)), -1, -1):
            c = t.get((lf[len(lf) - k :], u))
            d = dl.get(lf[len(lf) - k :]) if k >= 2 else None
            for cand in (c, d) if (c and sum(c.values()) >= 3) else (d, c):
                if cand:
                    for s, _ in cand.most_common(3):
                        a = apply(f, s)
                        if a:
                            return a
        return f

    def guess(t, f, u, use_pos):
        lf = f.lower()
        for k in range(min(7, len(lf)), -1, -1):
            c = t.get((lf[len(lf) - k :], u if use_pos else None))
            if c:
                for s, _ in c.most_common(3):
                    a = apply(f, s)
                    if a:
                        return a
        return f

    ans = {a[0]: a[1:] for a in json.load(open(f"{D}/{iso}.json"))["answers"]}
    st = (
        json.loads(
            gzip.decompress(open(f"{C}/predictions/stanza/{iso}.json.gz", "rb").read())
        )
        if os.path.exists(f"{C}/predictions/stanza/{iso}.json.gz")
        else None
    )
    c = collections.Counter()
    j = 0
    for si, (w, ls, us) in enumerate(test):
        for i, (f, l) in enumerate(zip(w, ls)):
            u = pt[j + i]
            if l is None:
                continue
            a = ans[f]
            text, stage = (a[2], a[3]) if i == 0 else (a[0], a[1])
            z = st[si][i] if st else None
            if f in byf and len(byf[f]) > 1:
                c["amb_n"] += 1
                c["amb_cascade"] += text == l
                c["amb_stanza"] += z == l
                k = byp.get((f, u))
                c["amb_tag"] += (k or byf[f]).most_common(1)[0][0] == l
            elif f not in byf and stage in (7, 8):
                c["oov_n"] += 1
                c["oov_cascade"] += text == l
                c["oov_stanza"] += z == l
                c["oov_dict"] += guess2(f, u) == l
                c["oov_all"] += guess(lists["all"], f, u, False) == l
                c["oov_rare"] += guess(lists["rare"], f, u, False) == l
                c["oov_rare_pos"] += guess(lists["rare+pos"], f, u, True) == l
            elif f not in byf and stage in (5, 6):
                c["dic_n"] += 1
                c["dic_cascade"] += text == l
                c["dic_stanza"] += z == l
                g = guess(lists["rare+pos"], f, u, True)
                c["dic_guess"] += g == l
                c["dic_either"] += (g == l) or (text == l)
        j += len(w)
    return iso, dict(c, tag_acc=tag_acc, seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor

    isos = sys.argv[1:]
    with ProcessPoolExecutor(7) as pool:
        res = dict(pool.map(run, isos))
    json.dump(res, open(C + "/proto2.json", "w"), indent=1)
    tot = collections.Counter()
    for iso, c in res.items():
        tot.update({k: v for k, v in c.items() if k not in ("tag_acc", "seconds")})
        g = lambda a, b: 100 * c.get(a, 0) / max(c.get(b, 0), 1)
        print(
            f"{iso} tag={100 * c['tag_acc']:.1f} | amb n={c.get('amb_n', 0):6d} casc={g('amb_cascade', 'amb_n'):.1f} tag-key={g('amb_tag', 'amb_n'):.1f} stanza={g('amb_stanza', 'amb_n'):.1f} | oov n={c.get('oov_n', 0):5d} casc={g('oov_cascade', 'oov_n'):.1f} all={g('oov_all', 'oov_n'):.1f} rare={g('oov_rare', 'oov_n'):.1f} rare+pos={g('oov_rare_pos', 'oov_n'):.1f} +dict={g('oov_dict', 'oov_n'):.1f} stanza={g('oov_stanza', 'oov_n'):.1f} | dict casc={g('dic_cascade', 'dic_n'):.1f} guess={g('dic_guess', 'dic_n'):.1f} either={g('dic_either', 'dic_n'):.1f} stanza={g('dic_stanza', 'dic_n'):.1f} [{c['seconds']}s]",
            flush=True,
        )
    g = lambda a, b: 100 * tot[a] / max(tot[b], 1)
    print(
        f"POOLED amb: cascade {g('amb_cascade', 'amb_n'):.2f} tag-key {g('amb_tag', 'amb_n'):.2f} stanza {g('amb_stanza', 'amb_n'):.2f} | oov: cascade {g('oov_cascade', 'oov_n'):.2f} all {g('oov_all', 'oov_n'):.2f} rare {g('oov_rare', 'oov_n'):.2f} rare+pos {g('oov_rare_pos', 'oov_n'):.2f} +dict {g('oov_dict', 'oov_n'):.2f} stanza {g('oov_stanza', 'oov_n'):.2f} | dict: cascade {g('dic_cascade', 'dic_n'):.2f} guess {g('dic_guess', 'dic_n'):.2f} either {g('dic_either', 'dic_n'):.2f} stanza {g('dic_stanza', 'dic_n'):.2f}"
    )
