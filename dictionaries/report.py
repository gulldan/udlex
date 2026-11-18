"""Render the catalog and the evaluation as one Markdown table per language."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import read_json


def percent(hits: int | None, total: int) -> str:
    """Accuracy cell."""
    return "—" if hits is None or not total else f"{100 * hits / total:.2f}"


def order_cell(entry: dict) -> str:
    """Chosen order, and what the dev gate said about the learned ones."""
    selection = entry["selection"]
    chosen = " > ".join(entry["order"]) or "(none)"
    declared = selection.get("policies", {}).get("declared", entry["order"])
    if "order_gate" not in selection:
        return f"{chosen} (declared, no dev)"
    verdicts = [
        f"{arm} net {v['net']:+d} p={v['p']:.2g}"
        for arm, v in selection["order_gate"].items()
    ]
    origin = "declared" if entry["order"] == declared else "learned"
    return f"{chosen} ({origin}; {'; '.join(verdicts) or 'learned = declared'})"


def excluded_cell(entry: dict) -> str:
    """Left-out sources with their reason and, for copyleft, the dev cost."""
    costs = entry["selection"].get("copyleft_cost", {})
    cells = []
    for item in entry["excluded_sources"]:
        cost = costs.get(item["id"])
        extra = f", dev {cost['net']:+d}/{cost['dev_tokens']}" if cost else ""
        cells.append(f"{item['id']} ({item['reason']}{extra})")
    return "; ".join(cells) or "—"


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    args = parser.parse_args()
    catalog = read_json(args.release / "DICTIONARIES.json")["dictionaries"]
    evaluation = read_json(args.release / "evaluation.json")
    scores = {r["iso639_3"]: r for r in evaluation["languages"]}
    lines = [
        "# Dictionary packages and their accuracy",
        "",
        (
            "Exact lemma match on gold tokens. `test`: official test split;"
            " `2-fold`: two halves of a test-only treebank. 1.0.0 values marked `*`"
            " come from a dictionary that learned from the scored sentences."
        ),
        "",
        "| Group | Languages | New | 1.0.0 | Copy | New better / worse |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, group in evaluation["summary"].items():
        both = group["with_1.0.0"]
        old = both["macro"]
        lines.append(
            f"| {name} | {group['languages']} | {group['macro']['new']:.2f} | "
            + (
                f"{old['old']:.2f} (new {old['new']:.2f} on {both['languages']}) | "
                if old
                else "— | "
            )
            + f"{group['macro']['copy']:.2f} | "
            f"{both['new_better']} / {both['new_worse']} |"
        )
    lines += [
        "",
        (
            "| ISO | Language | Eval | Treebank | Package | Dictionaries |"
            " Excluded | Order | New | 1.0.0 | Copy | Tokens |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for entry in catalog:
        iso = entry["iso639_3"]
        score = scores.get(iso)
        corpus = entry["corpus"]
        treebank = f"{corpus['treebank']} ({corpus['license']})" if corpus else "—"
        sources = (
            ", ".join(f"{s['id']} ({s['license']})" for s in entry["sources"]) or "—"
        )
        mode = {"test": "test", "two-fold": "2-fold"}.get(entry["evaluation"], "none")
        if score:
            tokens = score["tokens"]
            old = percent(score["old"], tokens)
            if score["old"] is not None and score["old_saw_evaluation"]:
                old += "*"
            cells = [percent(score["new"], tokens), old, percent(score["copy"], tokens)]
        else:
            tokens, cells = 0, ["—", "—", "—"]
        lines.append(
            f"| {iso} | {entry['language']} | {mode} | {treebank} | "
            f"{entry['distribution']} | {sources} | {excluded_cell(entry)} | "
            f"{order_cell(entry)} | {' | '.join(cells)} | {tokens or '—'} |"
        )
    (args.release / "EVALUATION.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
