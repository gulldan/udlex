"""Attach a measurement-only entry point to a scratch export of the canonical engine."""

import argparse
import shutil
from pathlib import Path


SOLO = """    let mut solo = Vec::new();
    for map in &maps {
        tables.external = eligible
            .iter()
            .filter_map(|f| map.get(f).map(|v| (f.clone(), v.clone())))
            .collect();
        let view = View::new(&tables, Thresholds::default(), Variant::Full);
        let answers: Vec<[Prediction; 2]> = forms
            .keys()
            .map(|form| {
                std::array::from_fn(|i| {
                    let a = probe::lemma_with_info(&view, form, i != 0, Options::default());
                    Prediction {
                        text: a.lemma.into_owned(),
                        stage: a.source,
                    }
                })
            })
            .collect();
        solo.push(answers);
    }
    let room = json!({"dev": headroom(&dev, &forms, &pred, &solo),
                      "test": headroom(&test, &forms, &pred, &solo)});
"""


def expose(path, old, new):
    body = path.read_text()
    if new in body:
        return
    if body.count(old) != 1:
        raise ValueError(f"Unexpected source in {path}")
    path.write_text(body.replace(old, new))


def prepare(root, disable_rules=False, examples=None):
    src = root / "rust/src"
    if (root / ".git").exists() or not src.is_dir():
        raise ValueError("Use a separate git archive export of the pinned 1.0.0 source")
    expose(
        src / "builder.rs",
        "mod compounds;",
        "pub(crate) mod paper_order;\nmod compounds;",
    )
    expose(
        src / "lib.rs",
        "pub mod format;",
        '#[cfg(feature="train")]\npub use builder::paper_order::run;\npub mod format;',
    )
    for name in ("stream_normalized_unique", "stream_resolved_preferred"):
        expose(src / "builder/sources.rs", f"\nfn {name}(", f"\npub(super) fn {name}(")
    shutil.copyfile(
        Path(__file__).with_name("order_native.rs"), src / "builder/paper_order.rs"
    )
    if disable_rules:
        expose(
            src / "builder/paper_order.rs",
            "i != 0, Options::default()",
            "i != 0, Options { use_rules: false, ..Options::default() }",
        )
    if examples:
        # Diagnostic build: list every changed prediction instead of five per
        # outcome, and count where sources that hold the same key disagree.
        harness = src / "builder/paper_order.rs"
        expose(harness, "n_examples[which] < 5", f"n_examples[which] < {examples}")
        expose(
            harness,
            "    let result = json!({",
            SOLO + '    let result = json!({"headroom":room,',
        )
        with harness.open("a") as out:
            out.write(Path(__file__).with_name("order_diagnostic.rs").read_text())
    (src / "bin").mkdir(exist_ok=True)
    (src / "bin/paper_order.rs").write_text(
        "fn main() -> Result<(), Box<dyn std::error::Error>> { udlex_rs::run() }\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("export", type=Path)
    p.add_argument("--disable-rules", action="store_true")
    p.add_argument("--examples", type=int)
    args = p.parse_args()
    prepare(args.export, args.disable_rules, args.examples)
