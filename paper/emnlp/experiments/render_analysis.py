"""Typeset the headroom, the kinds of change and the language-level stress tests."""

from verify_snapshot import EVIDENCE, read

KINDS = [
    ("case", "Case only"),
    ("diacritics", "Diacritics only"),
    ("copy", "Word itself"),
    ("nested", "Nested bases"),
    ("ending", "Different ending"),
    ("other", "Other"),
]


def p_value(p):
    return f"{p:.2f}" if p >= 0.095 else f"{p:.3f}" if p >= 0.001 else f"{p:.4f}"


def main():
    root = EVIDENCE.parent
    data = EVIDENCE / "ud219"
    home = read(data / "changes/anchor.json")["totals"]["official"]
    away = read(data / "changes/transfer.json")["totals"]["official_fit"]
    stress = read(data / "robustness.json")
    order = read(data / "order/summary.json")["official"]
    moved = read(data / "transfer/summary.json")["official_fit"]

    lines = [
        r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrr@{}}",
        r"\toprule Difference & Anchor tests & New corpora \\\midrule",
    ]
    for key, label in KINDS:
        a, b = home["kinds"][key], away["kinds"][key]
        lines.append(
            f"{label} & {a['win']:,} / {a['loss']:,} & {b['win']:,} / {b['loss']:,}"
            + r" \\"
        )
    totals = [
        tuple(sum(t["kinds"][k][o] for k, _ in KINDS) for o in ("win", "loss"))
        for t in (home, away)
    ]
    for (wins, losses), arm in zip(totals, (order, moved), strict=True):
        if (wins, losses) != (
            arm["arms"]["learned"]["wins"],
            arm["arms"]["learned"]["losses"],
        ):
            raise ValueError(
                "The kinds of change do not add up to the reported contrast"
            )
    lines.append(
        r"\midrule All & "
        + " & ".join(f"{w:,} / {l:,}" for w, l in totals)
        + r" \\"
        + "\n"
        + r"\bottomrule\end{tabular*}"
    )
    (root / "kinds-table.tex").write_text("\n".join(lines) + "\n")

    anchor, transfer = stress["anchor_official"], stress["transfer_official_fit"]
    lines = [
        r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}rlrrrrr@{}}",
        r"\toprule & & & \multicolumn{2}{c}{Anchor tests} & \multicolumn{2}{c}{New corpora} \\",
        r"$\alpha$ & Net & Acc. & $\Delta$ & Regr. & $\Delta$ & Regr. \\\midrule",
    ]
    for a, b in zip(
        anchor["confirmation_grid"], transfer["confirmation_grid"], strict=True
    ):
        lines.append(
            f"{a['alpha']:g} & {'yes' if a['net_floor'] else 'no'} & {a['accepted']}"
            f" & ${a['macro_delta_pp']:+.3f}$ & {a['regressions']:,}"
            f" & ${b['macro_delta_pp']:+.3f}$ & {b['regressions']:,}" + r" \\"
        )
    lines.append(r"\bottomrule\end{tabular*}")
    (root / "gate-table.tex").write_text("\n".join(lines) + "\n")

    nums, text = {}, {}
    for key, room, arm in (
        ("an", home["headroom"], order),
        ("at", away["headroom"], moved),
    ):
        gap = room["oracle_correct"] - room["declared_correct"]
        gain = room["learned_correct"] - room["declared_correct"]
        words = arm["arms"]["learned"]["tokens"]
        nums.update(
            {
                key + "Ext": room["external"],
                key + "Disputed": room["disputed"],
                key + "DisputedDeclared": room["declared_correct"],
                key + "Oracle": room["oracle_correct"],
                key + "DisputedLearned": room["learned_correct"],
                key + "Gap": gap,
            }
        )
        text[key + "Realized"] = f"{100 * gain / gap:.1f}"
        text[key + "DisputedShare"] = f"{100 * room['disputed'] / words:.2f}"
        text[key + "GapPooled"] = f"{100 * gap / words:.2f}"
    for key, part in (("an", anchor), ("at", transfer)):
        for name, arm in (("", "learned"), ("Gate", "confirmed")):
            v = part[arm]
            text[key + name + "SignP"] = p_value(v["sign_test_p"])
            text[key + name + "LangLo"] = f"{v['language_bootstrap_ci95_pp'][0]:+.3f}"
            text[key + name + "LangHi"] = f"{v['language_bootstrap_ci95_pp'][1]:+.3f}"
        grid = part["confirmation_grid"]
        nums[key + "GridAcceptMin"] = min(g["accepted"] for g in grid)
        nums[key + "GridAcceptMax"] = max(g["accepted"] for g in grid)
        text[key + "GridMacroMin"] = f"{min(g['macro_delta_pp'] for g in grid):+.3f}"
        text[key + "GridMacroMax"] = f"{max(g['macro_delta_pp'] for g in grid):+.3f}"
        clean = part["without_flagged"]
        text[key + "CleanMacro"] = f"{clean['learned']['macro_delta_pp']:+.3f}"
        text[key + "CleanLo"] = f"{clean['learned']['ci95_pp'][0]:+.3f}"
        text[key + "CleanHi"] = f"{clean['learned']['ci95_pp'][1]:+.3f}"
        text[key + "CleanGate"] = f"{clean['dev_gated']['macro_delta_pp']:+.3f}"
    if (
        anchor["flagged_languages"] != ["grc", "slv"]
        or stress["anchor_cross_validation"]["flagged_languages"]
    ):
        raise ValueError(
            "The manuscript names Ancient Greek and Slovenian as the flagged anchors"
        )
    if nums["anExt"] != order["external_hits_learned"]["tokens"]:
        raise ValueError("External lookup counts disagree between builds")
    net = moved["arms"]["learned"]["wins"] - moved["arms"]["learned"]["losses"]
    nested = away["kinds"]["nested"]
    copied = away["kinds"]["copy"]
    text["atNestedShare"] = f"{100 * (nested['win'] - nested['loss']) / net:.0f}"
    text["atCopyShare"] = f"{100 * (copied['win'] - copied['loss']) / net:.0f}"
    proof = read(data / "transfer/provenance.json")["pairs"][
        "slv:anchor-dev|dictionary"
    ]
    nums.update(anSlvMarked=proof["marked"], anSlvMarkedCovered=proof["marked_covered"])
    for key, table in (("an", home), ("at", away)):
        for kind, label in (
            ("case", "Case"),
            ("nested", "Nested"),
            ("ending", "Ending"),
            ("other", "Other"),
        ):
            nums[key + label + "Corr"] = table["kinds"][kind]["win"]
            nums[key + label + "Reg"] = table["kinds"][kind]["loss"]
    text.update({key: f"{v:,}" for key, v in nums.items()})
    (root / "analysis-metrics.tex").write_text(
        "".join(
            r"\newcommand" + "{\\" + key + "}{" + v + "}\n"
            for key, v in sorted(text.items())
        )
    )


if __name__ == "__main__":
    main()
