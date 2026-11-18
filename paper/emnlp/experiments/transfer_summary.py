"""Aggregate paired transfer contrasts after checking that each fit is the released one."""

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from order_summary import gate, summarize
from verify_snapshot import read, save, sha

FLOOR = 1000
SHARED = 0.10


def net(runs, arm=1):
    c = sum((Counter(r["test"]["counts"][arm]) for r in runs), Counter())
    return c


def leave_one_out(view):
    rows = {iso: net(v) for iso, v in view.items()}
    gain = {iso: c["wins"] - c["losses"] for iso, c in rows.items()}
    share = {iso: 100 * gain[iso] / c["tokens"] for iso, c in rows.items()}
    n, tokens = len(rows), sum(c["tokens"] for c in rows.values())
    macro = {iso: (sum(share.values()) - share[iso]) / (n - 1) for iso in rows}
    micro = {
        iso: 100 * (sum(gain.values()) - gain[iso]) / (tokens - rows[iso]["tokens"])
        for iso in rows
    }
    return {
        "macro_pp": [min(macro.values()), max(macro.values())],
        "micro_pp": [min(micro.values()), max(micro.values())],
        "lowest_macro_without": min(macro, key=macro.get),
        "lowest_micro_without": min(micro, key=micro.get),
    }


def main(a):
    ledger = read(a.inputs / "ledger.json")
    released = json.loads(gzip.decompress((a.released / "pairs.json.gz").read_bytes()))
    home = released["official"]
    flags = defaultdict(list)
    for key, v in read(a.provenance)["pairs"].items():
        if v["flagged"]:
            name, source = key.split("|")
            flags[name].append(source)
    corpora, runs, stamps = {}, defaultdict(list), set()
    for cfg, row in sorted(ledger["corpora"].items()):
        s = row.get("screen", {})
        iso = row["iso"]
        entry = corpora[cfg] = {
            k: v for k, v in row.items() if k not in ("files", "tier")
        }
        if not s.get("tokens"):
            entry["group"] = "no lemmas" if not row["tier"] else "nothing scorable"
            continue
        scorable = s["sentences_read"] - s.get("no_scorable_word", 0)
        flagged = s.get("exact_overlap", 0) + s.get("near_overlap", 0)
        group = "shared" if flagged >= SHARED * scorable else row["tier"]
        r = read(a.results / cfg / (iso + ".json"))
        stamps.add(read(a.results / cfg / "run.json")["executable"])
        if row["fit"] == "official":
            ref = home[iso][0]
            same = all(
                r[k] == ref[k]
                for k in ("arms", "order", "train_tokens", "train_correct")
            )
            if not same or r["dev"]["counts"] != ref["dev"]["counts"]:
                raise ValueError(f"The fit behind {cfg} is not the released {iso} fit")
        c = r["test"]["counts"]
        entry.update(
            group=group,
            flagged_sources=flags.get(cfg, []),
            order=r["order"][:2],
            accepted=gate(r["dev"]),
            declared_correct=c[0]["correct"],
            external_tokens=r["test"]["external_hits"][0]["tokens"],
            **{k: c[1][k] for k in ("tokens", "wins", "losses", "changed")},
        )
        runs[group].append((cfg, iso, dict(r, corpus=cfg)))

    def cohort(group, keep=lambda cfg: True, floor=FLOOR):
        by = defaultdict(list)
        for cfg, iso, r in runs[group]:
            if keep(cfg):
                by[iso].append(r)
        return {iso: v for iso, v in by.items() if net(v)["tokens"] >= floor}

    gold = cohort("gold")
    views = {
        "primary": gold,
        "without_flagged_sources": cohort("gold", lambda cfg: cfg not in flags),
        "official_fit": {i: v for i, v in gold.items() if i in home},
        "pooled_fit": {i: v for i, v in gold.items() if i not in home},
        "in_domain_same_languages": {i: home[i] for i in gold if i in home},
        "outside_ud": cohort("gold", lambda cfg: not corpora[cfg]["ud"]),
        "no_floor": cohort("gold", floor=1),
        "silver": cohort("silver"),
        "shared_collection": cohort("shared"),
    }
    direction = Counter()
    languages = {}
    for iso, row in read(a.roster or a.released / "languages.json").items():
        if row["sources"] < 2:
            continue
        mine = {g: [r for _, i, r in runs[g] if i == iso] for g in runs}
        entry = languages[iso] = {
            "in_domain_cohort": row["cohort"],
            "anchor": row["treebank"],
            "corpora": {g: [r["corpus"] for r in v] for g, v in mine.items() if v},
            "primary": iso in gold,
        }
        if mine.get("gold"):
            c = net(mine["gold"])
            entry["transfer"] = {
                k: c[k] for k in ("tokens", "wins", "losses", "changed")
            }
            entry["accepted"] = gate(mine["gold"][0]["dev"])
        if iso in home:
            c = net(home[iso])
            entry["in_domain"] = {
                k: c[k] for k in ("tokens", "wins", "losses", "changed")
            }
            if iso in gold:
                t = entry["transfer"]
                signs = [
                    (x["wins"] > x["losses"]) - (x["wins"] < x["losses"])
                    for x in (entry["in_domain"], t)
                ]
                direction[f"{signs[0]:+d} then {signs[1]:+d}"] += 1
    if stamps != {read(a.released / "summary.json")["binary"]["executable"]}:
        raise ValueError("Results do not come from the executable of the released fit")
    summary = {name: summarize(v, True) for name, v in views.items() if v}
    summary.update(
        executable=stamps.pop(),
        floor=FLOOR,
        shared_collection_share=SHARED,
        protocol_sha256=ledger["protocol_sha256"],
        ledger_sha256=sha(a.inputs / "ledger.json"),
        released_pairs_sha256=sha(a.released / "pairs.json.gz"),
        roster={
            "multi_source_languages": len(languages),
            "with_candidate_corpus": len({v["iso"] for v in corpora.values()}),
            "with_gold_tokens": len(cohort("gold", floor=1)),
            "primary": len(gold),
            "scored_gold_corpora": sum(len(v) for v in gold.values()),
            "scored_gold_corpora_outside_ud": sum(
                len(v) for v in views["outside_ud"].values()
            ),
            "languages_outside_ud": len(views["outside_ud"]),
            "corpora": dict(Counter(v["group"] for v in corpora.values())),
        },
        screen=dict(
            sum((Counter(v.get("screen", {})) for v in corpora.values()), Counter())
        ),
        flagged_sources={k: v for k, v in sorted(flags.items())},
        in_domain_then_transfer_sign=dict(direction),
        leave_one_language_out={
            k: leave_one_out(views[k]) for k in ("primary", "official_fit")
        },
        shared_contributors={
            str(flag): summarize(
                cohort(
                    "gold",
                    lambda cfg, flag=flag: corpora[cfg]["shared_contributors"] == flag,
                ),
                True,
            )
            for flag in (True, False)
        },
    )
    save(a.output / "summary.json", summary)
    save(a.output / "corpora.json", corpora)
    save(a.output / "languages.json", languages)
    save(a.output / "ledger.json", ledger)
    groups = {g: defaultdict(list) for g in runs}
    for g, rows in runs.items():
        for _, iso, r in rows:
            groups[g][iso].append(r)
    (a.output / "pairs.json.gz").write_bytes(
        gzip.compress(
            json.dumps(
                groups, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode(),
            mtime=0,
        )
    )
    for name in views:
        if name in summary:
            v = summary[name]
            print(name, v["languages"], v["arms"]["learned"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("inputs", "results", "released", "provenance", "output"):
        p.add_argument(key, type=Path)
    p.add_argument("--roster", type=Path, help="languages.json of the rules-on fit")
    main(p.parse_args())
