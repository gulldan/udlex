"""Typeset the transfer test from its ledgers and check that suffix rules leave the contrast unchanged."""

from verify_snapshot import EVIDENCE, read

ROWS = [
    ("in_domain_same_languages", "Anchor tests"),
    ("official_fit", "New corpora, official fits"),
    ("pooled_fit", "New corpora, pooled fits"),
    ("primary", "New corpora, all"),
    ("without_flagged_sources", "\\quad no flagged source"),
    ("outside_ud", "\\quad outside UD"),
    ("silver", "Automatic lemmas"),
    ("shared_collection", "Shared collection"),
]


def main():
    root = EVIDENCE.parent
    s = read(EVIDENCE / "ud219/transfer/summary.json")
    off = read(EVIDENCE / "ud219/transfer-no-rules/summary.json")
    changes = read(EVIDENCE / "ud219/transfer/changes.json")
    for view, _ in ROWS:
        for arm in ("learned", "dev_gated"):
            a, b = s[view]["arms"][arm], off[view]["arms"][arm]
            if any(a[k] != b[k] for k in ("tokens", "wins", "losses", "changed")):
                raise ValueError(
                    f"The order contrast depends on suffix rules in {view}"
                )
    lines = [
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrr@{}}",
        r"\toprule Evaluation set & Lang. & Words & Declared & Learned $\Delta$ (95\% int.)"
        r" & Confirmed $\Delta$ & Corr. / regr. \\\midrule",
    ]
    for view, label in ROWS:
        v = s[view]
        a, g = v["arms"]["learned"], v["arms"]["dev_gated"]
        lo, hi = a["ci95_pp"]
        lines.append(
            f"{label} & {v['languages']} & {a['tokens']:,} & {v['arms']['declared']['macro_accuracy']:.2f}"
            f" & ${a['macro_delta_pp']:+.3f}$ (${lo:+.3f}$, ${hi:+.3f}$) & ${g['macro_delta_pp']:+.3f}$"
            f" & {a['wins']:,} / {a['losses']:,}" + r" \\"
        )
        if view in ("in_domain_same_languages", "outside_ud"):
            lines.append(r"\midrule")
    lines.append(r"\bottomrule\end{tabular*}")
    (root / "transfer-table.tex").write_text("\n".join(lines) + "\n")

    def arm(view, name="learned"):
        return s[view]["arms"][name]

    nums, text = {}, {}
    for key, view in [
        ("tr", "primary"),
        ("trOff", "official_fit"),
        ("trHome", "in_domain_same_languages"),
        ("trPool", "pooled_fit"),
        ("trOut", "outside_ud"),
        ("trClean", "without_flagged_sources"),
    ]:
        a, g, ext = arm(view), arm(view, "dev_gated"), s[view]["external_hits_learned"]
        nums.update(
            {
                key + "Lang": s[view]["languages"],
                key + "N": a["tokens"],
                key + "Corr": a["wins"],
                key + "Reg": a["losses"],
                key + "Changed": a["changed"],
                key + "Ext": ext["tokens"],
                key + "GateCorr": g["wins"],
                key + "GateReg": g["losses"],
            }
        )
        for i, word in enumerate(("Up", "Tie", "Down")):
            nums[key + word] = a["improved_tied_worse"][i]
            nums[key + "Gate" + word] = g["improved_tied_worse"][i]
        text.update(
            {
                key + "Macro": f"{a['macro_delta_pp']:+.3f}",
                key + "Lo": f"{a['ci95_pp'][0]:+.3f}",
                key + "Hi": f"{a['ci95_pp'][1]:+.3f}",
                key + "Pooled": f"{a['micro_delta_pp']:+.3f}",
                key + "Gate": f"{g['macro_delta_pp']:+.3f}",
                key + "GateLo": f"{g['ci95_pp'][0]:+.3f}",
                key + "GateHi": f"{g['ci95_pp'][1]:+.3f}",
                key + "GatePooled": f"{g['micro_delta_pp']:+.3f}",
                key + "ExtShare": f"{100 * ext['tokens'] / a['tokens']:.1f}",
                key + "ChangedShare": f"{100 * a['changed'] / a['tokens']:.2f}",
                key + "WinShare": f"{100 * a['wins'] / (a['wins'] + a['losses']):.1f}",
            }
        )
    roster, screen, signs = s["roster"], s["screen"], s["in_domain_then_transfer_sign"]
    loo = s["leave_one_language_out"]
    named = [
        loo["official_fit"][k] for k in ("lowest_micro_without", "lowest_macro_without")
    ]
    if named + [loo["primary"]["lowest_macro_without"]] != ["hun", "tur", "tgl"]:
        raise ValueError(
            "The manuscript names Hungarian, Turkish and Tagalog as most influential"
        )
    corpora = read(EVIDENCE / "ud219/transfer/corpora.json")
    team = s["shared_contributors"]
    eee = changes["slv"]["loss"]["top"][0]
    if eee["form"] != "eee":
        raise ValueError(
            "The manuscript names the Slovenian filler as the most frequent regression"
        )
    nums.update(
        trMulti=roster["multi_source_languages"],
        trCandidates=roster["with_candidate_corpus"],
        trWithGold=roster["with_gold_tokens"],
        trGoldCorpora=roster["scored_gold_corpora"],
        trOutsideCorpora=roster["scored_gold_corpora_outside_ud"],
        trOutsideLang=roster["languages_outside_ud"],
        trSentRead=screen["sentences_read"],
        trExact=screen["exact_overlap"],
        trNear=screen["near_overlap"],
        trDocs=screen["documents_removed"],
        trRepeated=screen["repeated_sentence"],
        trSameSign=signs.get("+1 then +1", 0) + signs.get("-1 then -1", 0),
        trOppositeSign=signs.get("+1 then -1", 0) + signs.get("-1 then +1", 0),
        trAccepted=sum(
            v.get("accepted", False)
            for v in read(EVIDENCE / "ud219/transfer/languages.json").values()
            if v["primary"]
        ),
        trSlvEee=eee["tokens"],
        trTeamLang=team["True"]["languages"],
        trOtherLang=team["False"]["languages"],
        trSilverCorpora=roster["corpora"]["silver"],
        trSharedCorpora=roster["corpora"]["shared"],
    )
    for key, iso in [
        ("trHun", "hun"),
        ("trTur", "tur"),
        ("trRon", "ron"),
        ("trTgl", "tgl"),
    ]:
        nums[key + "Corr"], nums[key + "Reg"] = (
            changes[iso][k]["tokens"] for k in ("win", "loss")
        )
        nums[key + "N"] = sum(
            v["tokens"]
            for v in corpora.values()
            if v["iso"] == iso and v["group"] == "gold"
        )
    for key, cfg in [
        ("trSst", "sl_sst"),
        ("trSlMte", "sl_mte1984"),
        ("trParTUT", "it_partut"),
    ]:
        nums[key + "Corr"], nums[key + "Reg"] = (
            corpora[cfg]["wins"],
            corpora[cfg]["losses"],
        )
    languages = read(EVIDENCE / "ud219/transfer/languages.json")
    lost = {
        i
        for i, v in languages.items()
        if v["primary"]
        and "in_domain" in v
        and all(x["wins"] < x["losses"] for x in (v["in_domain"], v["transfer"]))
    }
    if lost != {"ces", "ell", "fin", "nld", "pol"}:
        raise ValueError(
            "The manuscript names the languages that lose in both settings"
        )
    slv = languages["slv"]["transfer"]
    if arm("official_fit", "dev_gated")["improved_tied_worse"][2] != 1 or not (
        languages["slv"]["accepted"] and slv["wins"] < slv["losses"]
    ):
        raise ValueError("The manuscript names Slovenian as the only confirmed loss")
    proof = read(EVIDENCE / "ud219/transfer/provenance.json")["pairs"]
    perseus = proof["grc_perseus|spacy-lookups"]
    nums.update(
        trGrcTypes=proof["grc:anchor-test|spacy-lookups"]["types"],
        trBothSign=nums["trSameSign"] + nums["trOppositeSign"],
        trOutsideCandidates=sum(not v["ud"] for v in corpora.values()),
    )
    text.update(
        trPerseusCover=f"{100 * perseus['covered'] / perseus['types']:.0f}",
        trPerseusAgree=f"{100 * perseus['agree'] / perseus['covered']:.0f}",
        trHunDelta=f"{100 * (nums['trHunCorr'] - nums['trHunReg']) / nums['trHunN']:+.2f}",
    )
    if (
        sum(
            v["iso"] == "hun" and v["group"] == "gold" and not v["ud"]
            for v in corpora.values()
        )
        != 2
    ):
        raise ValueError("The manuscript counts two Hungarian gold corpora outside UD")
    partut = corpora["it_partut"]["screen"]
    nums.update(
        trParTUTRead=partut["sentences_read"], trParTUTExact=partut["exact_overlap"]
    )
    text.update(
        trLooOffPooled=f"{loo['official_fit']['micro_pp'][0]:+.3f}",
        trLooMacro=f"{loo['primary']['macro_pp'][0]:+.3f}",
        trLooOffMacroLo=f"{loo['official_fit']['macro_pp'][0]:+.3f}",
        trTeamMacro=f"{team['True']['arms']['learned']['macro_delta_pp']:+.3f}",
        trOtherMacro=f"{team['False']['arms']['learned']['macro_delta_pp']:+.3f}",
    )
    text.update({key: f"{v:,}" for key, v in nums.items()})
    (root / "transfer-metrics.tex").write_text(
        "".join(
            r"\newcommand" + "{\\" + key + "}{" + v + "}\n"
            for key, v in sorted(text.items())
        )
    )


if __name__ == "__main__":
    main()
