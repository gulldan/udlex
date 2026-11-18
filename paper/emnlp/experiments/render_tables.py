"""Generate LaTeX results tables from the complete language and treebank ledgers."""
from pathlib import Path

from verify_snapshot import EVIDENCE, read

# Some package labels repeat the ISO code. These are the corresponding UD names.
NAMES = {
    "aze": "Azerbaijani", "bel": "Belarusian", "bre": "Breton", "bul": "Bulgarian",
    "cat": "Catalan", "chu": "Old Church Slavonic", "fao": "Faroese", "gle": "Irish",
    "hun": "Hungarian", "hye": "Armenian", "isl": "Icelandic", "ita": "Italian",
    "jpn": "Japanese", "kaz": "Kazakh", "lav": "Latvian", "lit": "Lithuanian",
    "spa": "Spanish", "sqi": "Albanian", "swe": "Swedish", "uzb": "Uzbek",
    "xcl": "Classical Armenian",
}

def latex(s):
    return "".join({"&": r"\&", "_": r"\_", "%": r"\%", "#": r"\#", "$": r"\$"}.get(c, c) for c in s)


def generate(evidence=EVIDENCE, root=None):
    root = root or EVIDENCE.parent
    snap = read(evidence / "snapshot.json")["languages"]
    rows = read(evidence / "languages.json")
    result = read(evidence / "summary.json")
    nl = r" \\"
    output = [
        r"\begingroup\small",
        r"\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{0.95}",
        r"\newcommand{\cathead}{\toprule Language & Sets & Words & Copy & Off & Full \\\midrule}",
        r"\tablefirsthead{\cathead}\tablehead{\cathead}",
        r"\tabletail{\bottomrule}\tablelasttail{\bottomrule}",
        r"\begin{supertabular*}{\columnwidth}{@{\extracolsep{\fill}}p{.31\columnwidth}rrrrr@{}}",
    ]
    reasons = {"all available tests used in dictionary": "Input file",
               "no official test file for code": "No test",
               "no scorable test tokens after exclusions": "No gold"}
    names = {iso: NAMES[iso] if row["name"] == iso else row["name"] for iso, row in snap.items()}
    for iso in sorted(rows, key=lambda k:(names[k].casefold(), k)):
        r, name = rows[iso], latex(names[iso])
        c = r["counts"]
        sets = f"{r['scored_files']}/{r['test_files']}"
        label = r"\raggedright " + name + r" {\scriptsize(" + latex(iso) + ")}"
        if c["tokens"]:
            values = [f"{100*c[a]/c['tokens']:.1f}" for a in ("copy", "lookup", "full")]
        else:
            values = [r"\multicolumn{3}{c}{" + reasons[r["status"]] + "}"]
        output.append(" & ".join([label, sets, f"{c['tokens']:,}", *values]) + nl)
    output.extend([r"\end{supertabular*}", r"\endgroup"])
    (root / "catalogue.tex").write_text("\n".join(output) + "\n")

    output = [r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}llrrr@{}}\toprule",
              "Resource & Mean & Copy & Rules off & Full" + nl + r"\midrule"]
    for cohort, name in [("all", "All"), ("UD","UD-derived"), ("external","External")]:
        for metric, label in [("macro_accuracy","Macro"), ("micro_accuracy","Micro")]:
            m = result[cohort][metric]
            output.append(" & ".join([name if label=="Macro" else "", label,
                                     *(f"{m[a]:.2f}" for a in ("copy","lookup","full"))]) + nl)
        if cohort != "external":
            output.append(r"\addlinespace[2pt]")
    output.append(r"\bottomrule\end{tabular*}")
    (root / "accuracy.tex").write_text("\n".join(output) + "\n")


if __name__ == "__main__":
    generate()
