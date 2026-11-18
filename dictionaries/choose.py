"""Choose each language's dictionary order and engine options, then build it.

Official split: a source order is fitted on training words whose form the rest
of training does not hold (five folds), and replaces the declared order only
when the full cascade, trained on train alone, is better on dev by an exact
McNemar test at the policy's level and net gain. Options of the engine
(lemma adapter, compound marker) pass the same gate one at a time. The package
is then built with dev for its thresholds and refit; test is never read here.

Two-fold languages keep the declared order (no dev to gate on), are scored
fold by fold, and the package learns from the whole treebank. Languages
without gold lemmas get dictionary-only packages.
"""

from __future__ import annotations

import argparse
import functools
import gzip
import hashlib
import heapq
import itertools
import math
import re
import shutil
import sys
import traceback
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from common import (
    POLICY,
    SOURCES,
    jobs,
    jsonl,
    license_class,
    nfc,
    read_json,
    write_json,
)

import udlex

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

FOLDS = 5
MIN_STRIP_TOKENS = 5000
COMPOUND_MARKERS = "#_+"


def license_of(source: dict, iso: str) -> str | None:
    """SPDX id of one source for one language."""
    return source.get("license_by_language", {}).get(iso, source["license"])


def family_rank(source: dict, iso: str, ud_code: str, order: list[str]) -> int:
    """Position of a source's family in the declared order."""
    family = source["family"]
    if family == "wiktionary-native" and source["id"].split("-", 1)[1] in {
        iso,
        ud_code,
    }:
        family = "wiktionary-own"
    return order.index(family)


def candidates(iso: str, row: dict, work: Path) -> tuple[list[dict], list[dict]]:
    """Included sources in declared order, and excluded ones with the reason.

    Wiktionary editions in other languages are merged into one source, so the
    number of sources, and the order search over them, stays small.
    """
    policy = read_json(POLICY)
    included, excluded = [], []
    for source in read_json(SOURCES)["sources"]:
        manifest = work / "pairs" / source["id"] / "manifest.json"
        if not manifest.is_file() or iso not in read_json(manifest)["languages"]:
            continue
        spdx = license_of(source, iso)
        entry = {
            "id": source["id"],
            "family": source["family"],
            "license": spdx,
            "license_class": license_class(spdx),
            "rank": family_rank(
                source, iso, row["ud_code"], policy["declared_family_order"]
            ),
            "members": [source["id"]],
            "pairs": [str(work / "pairs" / source["id"] / f"{iso}.tsv.gz")],
            "pairs_sha256": [read_json(manifest)["languages"][iso]["sha256"]],
        }
        rule = policy["excluded_sources"].get(source["id"], [])
        if rule == "*" or iso in rule:
            excluded.append(entry | {"reason": "unclear provenance (policy)"})
        elif entry["license_class"] in policy["excluded_license_classes"]:
            excluded.append(entry | {"reason": f"{entry['license_class']} license"})
        else:
            included.append(entry)
    other = [
        s
        for s in included
        if s["family"] == "wiktionary-native"
        and s["rank"] == policy["declared_family_order"].index("wiktionary-native")
    ]
    if len(other) > 1:
        merged = dict(other[0], id="wiktionary-other")
        for key in ("members", "pairs", "pairs_sha256"):
            merged[key] = [v for s in other for v in s[key]]
        if len({s["license"] for s in other}) != 1:
            msg = f"{iso}: Wiktionary editions under different licenses"
            raise ValueError(msg)
        included = [s for s in included if s not in other] + [merged]
    included.sort(key=lambda s: (s["rank"], s["id"]))
    excluded.sort(key=lambda s: (s["rank"], s["id"]))
    return included, excluded


def marks(sentences: list[dict]) -> tuple[set[str], set[str]]:
    """Combining marks seen in training forms and in training lemmas."""
    form_marks: set[str] = set()
    lemma_marks: set[str] = set()
    for sentence in sentences:
        for form, lemma in zip(sentence["tokens"], sentence["lemmas"], strict=True):
            if lemma is None:
                continue
            form_marks.update(
                c
                for c in unicodedata.normalize("NFD", form)
                if unicodedata.category(c) == "Mn"
            )
            lemma_marks.update(
                c
                for c in unicodedata.normalize("NFD", lemma)
                if unicodedata.category(c) == "Mn"
            )
    return form_marks, lemma_marks


def stripper(allowed: set[str] | None) -> Callable[[str], str]:
    """Drop combining marks the treebank never writes (stress, vowel length)."""
    if allowed is None:
        return lambda value: value
    drop = re.compile(
        "["
        + "".join(re.escape(c) for c in nonspacing_marks() if c not in allowed)
        + "]"
    )

    def strip(value: str) -> str:
        if value.isascii():
            return value
        decomposed = unicodedata.normalize("NFD", value)
        return nfc(drop.sub("", decomposed)) if drop.search(decomposed) else value

    return strip


@functools.cache
def nonspacing_marks() -> tuple[str, ...]:
    """Every Unicode nonspacing mark, for the stripping pattern."""
    return tuple(
        c for c in map(chr, range(0x110000)) if unicodedata.category(c) == "Mn"
    )


def lines(path: str) -> Iterator[tuple[str, str, int]]:
    """``(form, lemma, count)`` rows of one extracted table, in its sorted order."""
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            form, lemma, count = line.rstrip("\n").split("\t")
            yield form, lemma, int(count)


def resolve(
    paths: list[str], allowed: tuple[set[str], set[str]] | None
) -> dict[str, str]:
    """One lemma per form: most analyses, then most frequent lemma, then shortest.

    Streams the sorted tables twice instead of holding a counter per form, so a
    source with tens of millions of forms fits in memory.
    shortcut: two original forms that merge after mark stripping are compared
    by their winners' counts, not by summed counts; exact for unstripped forms.
    """
    strip_form, strip_lemma = (stripper(a) for a in (allowed or (None, None)))
    frequency: Counter[str] = Counter()
    for path in paths:
        for _, lemma, count in lines(path):
            frequency[strip_lemma(lemma)] += count
    best: dict[str, tuple[str, int]] = {}
    merged = heapq.merge(*(lines(p) for p in paths), key=lambda row: (row[0], row[1]))
    for form, group in itertools.groupby(merged, key=lambda row: row[0]):
        votes: dict[str, int] = {}
        for _, raw, count in group:
            lemma = sys.intern(strip_lemma(raw))
            votes[lemma] = votes.get(lemma, 0) + count
        winner = max(votes, key=lambda y: (votes[y], frequency[y], -len(y)))
        key = strip_form(form)
        rival = best.get(key)
        rank = (votes[winner], frequency[winner], -len(winner))
        if (
            key
            and winner
            and (not rival or rank > (rival[1], frequency[rival[0]], -len(rival[0])))
        ):
            best[key] = (winner, votes[winner])
    return {form: lemma for form, (lemma, _) in best.items()}


def write_table(table: dict[str, str], path: Path) -> None:
    """A strictly sorted, NFC, form-unique TSV, as ``normalized_unique`` requires."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as out:
        for form in sorted(table):
            out.write(f"{form}\t{table[form]}\n")


def residual_votes(sentences: list[dict]) -> dict[str, Counter[str]]:
    """Gold lemmas of training words whose form the other folds do not hold."""
    fold_of = [
        hashlib.sha256(s["block"].encode()).digest()[0] % FOLDS for s in sentences
    ]
    exact: list[Counter[str]] = [Counter() for _ in range(FOLDS)]
    lower: list[Counter[str]] = [Counter() for _ in range(FOLDS)]
    for k, sentence in zip(fold_of, sentences, strict=True):
        for form in sentence["tokens"]:
            exact[k][form] += 1
            lower[k][form.lower()] += 1
    all_exact = sum(exact, Counter())
    all_lower = sum(lower, Counter())
    votes: dict[str, Counter[str]] = defaultdict(Counter)
    for k, sentence in zip(fold_of, sentences, strict=True):
        for form, lemma in zip(sentence["tokens"], sentence["lemmas"], strict=True):
            unseen = (
                all_exact[form] == exact[k][form]
                and all_lower[form.lower()] == lower[k][form.lower()]
            )
            if lemma is not None and unseen:
                votes[nfc(form)][nfc(lemma)] += 1
    return votes


def subset_gains(
    maps: list[dict[str, str]], gold: dict[str, Counter[str]]
) -> tuple[list[list[int]], list[list[int]]]:
    """Gain over copying and coverage of each source, summed over forms by mask.

    Entry ``[j][mask]`` sums the forms whose covering sources lie within
    ``mask``: what source ``j`` adds when exactly the sources outside ``mask``
    were tried before it.
    """
    count = len(maps)
    end = (1 << count) - 1
    gains = [[0] * (end + 1) for _ in range(count)]
    coverage = [[0] * (end + 1) for _ in range(count)]
    for form, labels in gold.items():
        identity = labels.get(form, 0)
        mask = sum(1 << j for j, m in enumerate(maps) if form in m)
        for j, source in enumerate(maps):
            if form in source:
                gains[j][mask] += labels.get(source[form], 0) - identity
                coverage[j][mask] += sum(labels.values())
    for bit in range(count):
        for mask in range(end + 1):
            if mask & (1 << bit):
                for j in range(count):
                    gains[j][mask] += gains[j][mask ^ (1 << bit)]
                    coverage[j][mask] += coverage[j][mask ^ (1 << bit)]
    return gains, coverage


def learn(
    maps: list[dict[str, str]], gold: dict[str, Counter[str]]
) -> dict[str, list[int]]:
    """Exact best permutation and ordered subset of sources on a copy fallback.

    A port of the study's harness: a subset dynamic program finds the
    permutation with the largest gain, fewest inversions of the declared
    order and lexicographically smallest order at equal gain.
    """
    count = len(maps)
    end = (1 << count) - 1
    gains, coverage = subset_gains(maps, gold)
    best: list[tuple[int, int, list[int]] | None] = [None] * (end + 1)
    best[0] = (0, 0, [])
    for mask in range(end):
        prior = best[mask]
        if prior is None:
            continue
        for j in (j for j in range(count) if not mask & (1 << j)):
            cand = (
                prior[0] + gains[j][end ^ mask],
                prior[1] + sum(v > j for v in prior[2]),
                [*prior[2], j],
            )
            current = best[mask | (1 << j)]
            if current is None or (-cand[0], *cand[1:]) < (-current[0], *current[1:]):
                best[mask | (1 << j)] = cand
    full = best[end][2] if best[end] else []
    subset = min(
        (b for b in best if b is not None), key=lambda b: (-b[0], len(b[2]), b[2])
    )[2]
    greedy, single = greedy_order(gains, coverage, count)
    return {
        "declared": list(range(count)),
        "learned": full,
        "subset": subset,
        "greedy": greedy,
        "single": single,
    }


def greedy_order(
    gains: list[list[int]], coverage: list[list[int]], count: int
) -> tuple[list[int], list[int]]:
    """Greedy ordered subset and the best single source, the study's secondary arms."""
    end = (1 << count) - 1
    greedy: list[int] = []
    single: list[int] = []
    mask = 0
    while mask != end:
        free = end ^ mask
        j = max(
            (j for j in range(count) if not mask & (1 << j)),
            key=lambda j: (gains[j][free], coverage[j][free], -j),
        )
        if not mask:
            single.append(j)
        if gains[j][free] < 0 or coverage[j][free] == 0:
            break
        greedy.append(j)
        mask |= 1 << j
    return greedy, single


def mcnemar(base: list[bool], other: list[bool]) -> tuple[int, int, float]:
    """Discordant counts and the exact two-sided McNemar p-value."""
    wins = sum(o and not b for b, o in zip(base, other, strict=True))
    losses = sum(b and not o for b, o in zip(base, other, strict=True))
    n = wins + losses
    if not n:
        return wins, losses, 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1))
    return wins, losses, min(1.0, 2 * tail / 2**n)


def build(
    dest: Path,
    row: dict,
    train: Path | None,
    dev: Path | None,
    tables: list[tuple[str, Path]],
    **options: object,
) -> udlex.Outcome:
    """One ``build_lexicon`` call into a fresh directory."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    return udlex.build_lexicon(
        str(dest),
        train=str(train) if train else None,
        dev=str(dev) if dev else None,
        test=None,
        paradigms=[(label, str(path), False, True) for label, path in tables],
        name=row["iso639_3"],
        language=row["language"],
        iso639_3=row["iso639_3"],
        **options,
    )


def predict(lexicon: Path, sentences: list[dict]) -> list[bool]:
    """Exact-match correctness of every gold token, gold tokenization."""
    lex = udlex.Lexicon(str(lexicon), use_user=False)
    answers = lex.lemmatize_many([s["tokens"] for s in sentences])
    return [
        nfc(answer) == nfc(gold)
        for sentence, row in zip(sentences, answers, strict=True)
        for answer, gold in zip(row, sentence["lemmas"], strict=True)
        if gold is not None
    ]


def compound_marker(sentences: list[dict]) -> str | None:
    """A boundary character inside at least 1 % of distinct training lemmas."""
    lemmas = {y for s in sentences for y in s["lemmas"] if y and len(y) > 1}
    for marker in COMPOUND_MARKERS:
        inside = sum(marker in y.strip(marker) for y in lemmas)
        if lemmas and inside >= max(20, len(lemmas) // 100):
            return marker
    return None


def prepare_tables(
    sources: list[dict],
    train: list[dict] | None,
    out: Path,
    wanted: set[str] | frozenset[str] = frozenset(),
) -> list[dict[str, str]]:
    """Resolve every source with the training split's mark inventory.

    Full tables go to disk for the engine; only ``wanted`` forms stay in memory.
    """
    tokens = sum(y is not None for s in train or () for y in s["lemmas"])
    # A small treebank need not show every mark of the language's spelling.
    allowed = marks(train) if train and tokens >= MIN_STRIP_TOKENS else None
    maps = []
    for source in sources:
        table = resolve(source["pairs"], allowed)
        write_table(table, out / f"{source['id']}.tsv")
        source["rows"] = len(table)
        maps.append({f: table[f] for f in wanted if f in table})
        del table
    return maps


def gate(base: list[bool], other: list[bool]) -> dict:
    """Paired dev evidence for replacing ``base`` by ``other``."""
    policy = read_json(POLICY)["gate"]
    wins, losses, p = mcnemar(base, other)
    need = max(policy["min_net_gain"], math.ceil(len(base) / policy["tokens_per_gain"]))
    return {
        "wins": wins,
        "losses": losses,
        "net": wins - losses,
        "p": p,
        "required_net": need,
        "accepted": p <= policy["alpha"] and wins - losses >= need,
        "dev_correct": sum(other),
        "dev_tokens": len(base),
    }


def select_official(
    row: dict, sources: list[dict], excluded: list[dict], work: Path
) -> dict:
    """Order and option gate on dev, then the shipped build."""
    iso = row["iso639_3"]
    root = work / "build" / iso
    splits = work / "splits" / iso
    train, dev = jsonl(splits / "train.jsonl"), jsonl(splits / "dev.jsonl")
    votes = residual_votes(train)
    maps = prepare_tables(sources, train, root / "tables", set(votes))
    names = [s["id"] for s in sources]
    policies = learn(maps, votes) if sources else {"declared": []}
    tables = {s["id"]: root / "tables" / f"{s['id']}.tsv" for s in sources}
    trial = root / "trial" / iso

    def score(order: list[str], **extra: object) -> list[bool]:
        tried = [
            (name, tables.get(name) or root / "excluded" / f"{name}.tsv")
            for name in order
        ]
        build(trial, row, splits / "train.jsonl", None, tried, **extra)
        return predict(trial, dev)

    named = {k: [names[j] for j in v] for k, v in policies.items()}
    decision: dict = {"policies": named, "order": named["declared"], "options": {}}
    if any(y is not None for s in dev for y in s["lemmas"]):
        decision |= dev_gate(score, named, train)
        for source in (s for s in excluded if s["license_class"] == "copyleft"):
            prepare_tables([source], train, root / "excluded")
            ranks = {s["id"]: s["rank"] for s in sources}
            order = list(decision["order"])
            order.insert(sum(ranks[n] <= source["rank"] for n in order), source["id"])
            cost = gate(decision["dev_vector"], score(order, **decision["options"]))
            decision["copyleft_cost"][source["id"]] = cost
        del decision["dev_vector"]
        shutil.rmtree(trial.parent, ignore_errors=True)
    outcome = build(
        root / "lexicon" / iso,
        row,
        splits / "train.jsonl",
        splits / "dev.jsonl" if "dev_chosen" in decision else None,
        [(name, tables[name]) for name in decision["order"]],
        **decision["options"],
    )
    return decision | {"outcome": outcome_record(outcome)}


def dev_gate(
    score: Callable[..., list[bool]],
    policies: dict[str, list[str]],
    train: list[dict],
) -> dict:
    """Replace the declared order and default options only on dev evidence."""
    chosen = policies["declared"]
    base = score(chosen)
    tokens = len(base)
    result: dict = {"dev_declared": {"correct": sum(base), "tokens": tokens}}
    trials: dict[str, dict] = {}
    vectors: dict[str, list[bool]] = {}
    for arm in ("learned", "subset"):
        order = policies[arm]
        if order != chosen and all(order != t["order"] for t in trials.values()):
            vectors[arm] = score(order)
            trials[arm] = {"order": order} | gate(base, vectors[arm])
    result["order_gate"] = trials
    accepted = [k for k, v in trials.items() if v["accepted"]]
    if accepted:
        arm = max(accepted, key=lambda k: trials[k]["net"])
        chosen, base = trials[arm]["order"], vectors[arm]
    candidates: dict[str, dict[str, object]] = {
        "lemma_adapter": {"lemma_adapter": True}
    }
    if marker := compound_marker(train):
        candidates["compound_marker"] = {"compound_marker": marker}
    options: dict[str, object] = {}
    result["option_gate"] = {}
    for name, extra in candidates.items():
        evidence = gate(base, score(chosen, **extra))
        result["option_gate"][name] = evidence | {"value": next(iter(extra.values()))}
        if evidence["accepted"]:
            options |= extra
    final = score(chosen, **options) if options else base
    return result | {
        "order": chosen,
        "options": options,
        "dev_chosen": {"correct": sum(final), "tokens": tokens},
        "dev_vector": final,
        "copyleft_cost": {},
    }


def select_twofold(row: dict, sources: list[dict], work: Path) -> dict:
    """Declared order; fold builds for scoring and a whole-treebank package."""
    iso = row["iso639_3"]
    root = work / "build" / iso
    splits = work / "splits" / iso
    decision: dict = {"order": [s["id"] for s in sources], "options": {}, "folds": []}
    for k in (0, 1):
        train_path = splits / f"fold{1 - k}.jsonl"
        prepare_tables(sources, jsonl(train_path), root / f"tables-fold{k}")
        tables = [
            (s["id"], root / f"tables-fold{k}" / f"{s['id']}.tsv") for s in sources
        ]
        outcome = build(root / f"fold{k}" / iso, row, train_path, None, tables)
        decision["folds"].append(outcome_record(outcome))
    prepare_tables(sources, jsonl(splits / "all.jsonl"), root / "tables")
    tables = [(s["id"], root / "tables" / f"{s['id']}.tsv") for s in sources]
    outcome = build(root / "lexicon" / iso, row, splits / "all.jsonl", None, tables)
    return decision | {"outcome": outcome_record(outcome)}


def select_dictionary_only(row: dict, sources: list[dict], work: Path) -> dict:
    """No gold lemmas: the first dictionary trains the rules, the rest are lookups."""
    root = work / "build" / row["iso639_3"]
    prepare_tables(sources, None, root / "tables")
    tables = [(s["id"], root / "tables" / f"{s['id']}.tsv") for s in sources]
    outcome = build(root / "lexicon" / row["iso639_3"], row, None, None, tables)
    return {
        "order": [s["id"] for s in sources],
        "options": {},
        "outcome": outcome_record(outcome),
    }


def outcome_record(outcome: udlex.Outcome) -> dict:
    """The JSON-safe part of a build outcome."""
    return {
        "generation": outcome.generation,
        "source": outcome.source,
        "training_tokens": outcome.training_tokens,
        "thresholds": list(outcome.thresholds),
        "tuned_on_dev": outcome.tuned_on_dev,
        "variant": outcome.variant,
        "selected_on": outcome.selected_on,
        "entries": dict(outcome.entries),
        "observed_inputs": [list(x) for x in outcome.observed_inputs],
    }


def run(iso: str, row: dict, work: Path) -> str:
    """Select and build one language; the decision file marks completion."""
    out = work / "build" / iso / "decision.json"
    if out.is_file():
        return f"{iso} cached"
    try:
        sources, excluded = candidates(iso, row, work)
        if row["mode"] == "official":
            decision = select_official(row, sources, excluded, work)
        elif row["mode"] == "twofold":
            decision = select_twofold(row, sources, work)
        elif sources:
            decision = select_dictionary_only(row, sources, work)
        else:
            return f"{iso} skipped: no lemmas and no dictionary"
        decision |= {
            "iso639_3": iso,
            "mode": row["mode"],
            "sources": sources,
            "excluded": excluded,
        }
        write_json(out, decision)
    except Exception:  # noqa: BLE001 - one language must not stop the others
        return f"{iso} FAILED\n{traceback.format_exc()}"
    return (
        f"{iso} {row['mode']} order={decision['order']} options={decision['options']}"
    )


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("--languages", nargs="*")
    parser.add_argument(
        "--modes", nargs="*", default=["official", "twofold", "no-lemmas"]
    )
    args = parser.parse_args()
    roster = read_json(args.work / "roster.json")
    todo = [
        iso
        for iso, row in roster.items()
        if row["mode"] in args.modes and (not args.languages or iso in args.languages)
    ]
    todo.sort(key=lambda iso: (args.modes.index(roster[iso]["mode"]), iso))
    with ProcessPoolExecutor(max_workers=jobs()) as pool:
        for message in pool.map(
            run, todo, [roster[i] for i in todo], [args.work] * len(todo)
        ):
            print(message, flush=True)


if __name__ == "__main__":
    main()
