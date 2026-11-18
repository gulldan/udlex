"""Typeset both 2.19 comparisons directly from their own verified result ledgers."""

from pathlib import Path

from render_tables import generate
from verify_snapshot import EVIDENCE, read


def main():
    root = EVIDENCE.parent
    data = EVIDENCE / "ud219"
    catalog = read(data / "catalog/summary.json")
    results = read(data / "order/summary.json")
    generate(data / "catalog", root)
    lines = [
        r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrr@{}}",
        r"\toprule Policy & Macro & Pooled & $\Delta$ macro \\\midrule",
    ]
    labels = [
        ("declared", "Declared order"),
        ("learned", "Learned order"),
        ("dev_gated", "Dev-confirmed order"),
        ("single", "Best single"),
        ("greedy_subset", "Greedy subset"),
        ("exact_subset", "Exact subset"),
    ]
    for key, label in labels:
        v = results["official"]["arms"][key]
        change = "---" if key == "declared" else f"${v['macro_delta_pp']:+.3f}$"
        lines.append(
            f"{label} & {v['macro_accuracy']:.3f} & {v['micro_accuracy']:.3f} & {change}"
            + r" \\"
        )
    lines.append(r"\bottomrule\end{tabular*}")
    (root / "order-table.tex").write_text("\n".join(lines) + "\n")
    a = catalog["all"]
    bank = catalog["run"]["release_test_files"]
    lost = catalog["language_status"]
    nums = dict(
        catLang=a["scored_languages"],
        catN=a["counts"]["tokens"],
        catFiles=catalog["test_files_scored"],
        catMatched=catalog["test_files_in_scope"],
        catAvailable=bank,
        catOmitted=catalog["excluded"]["declared_dictionary_input_file"],
        catNoTest=lost.get("no official test file for code", 0),
        catOverlap=lost.get("all available tests used in dictionary", 0),
        catNoGold=lost.get("no scorable test tokens after exclusions", 0),
        catCorr=a["counts"]["rules_wins"],
        catReg=a["counts"]["rules_losses"],
        catWins=a["full_minus_lookup"]["language_wins_ties_losses"][0],
        catTies=a["full_minus_lookup"]["language_wins_ties_losses"][1],
        catLosses=a["full_minus_lookup"]["language_wins_ties_losses"][2],
        catUd=catalog["UD"]["scored_languages"],
        catExt=catalog["external"]["scored_languages"],
    )
    values = {key: f"{v:,}" for key, v in nums.items()}
    for key, arm in [("catCopy", "copy"), ("catFull", "full"), ("catOff", "lookup")]:
        values[key] = f"{a['macro_accuracy'][arm]:.2f}"
    for key, group in [("cat", "all"), ("ud", "UD"), ("ext", "external")]:
        effect = catalog[group]["full_minus_lookup"]
        for suffix, v in [
            ("Rule", effect["macro_pp"]),
            ("Lo", effect["macro_ci95_pp"][0]),
            ("Hi", effect["macro_ci95_pp"][1]),
            ("Pooled", effect["micro_pp"]),
        ]:
            values[key + suffix] = f"{v:+.2f}"
    (root / "catalog-metrics.tex").write_text(
        "".join(
            r"\newcommand" + "{\\" + key + "}{" + v + "}\n" for key, v in values.items()
        )
    )


if __name__ == "__main__":
    main()
