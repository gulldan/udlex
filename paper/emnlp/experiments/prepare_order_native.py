"""Attach a measurement-only entry point to a scratch export of the canonical engine."""

import argparse
import shutil
from pathlib import Path


def expose(path, old, new):
    body = path.read_text()
    if new in body:
        return
    if body.count(old) != 1:
        raise ValueError(f"Unexpected source in {path}")
    path.write_text(body.replace(old, new))


def prepare(root, disable_rules=False):
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
    (src / "bin").mkdir(exist_ok=True)
    (src / "bin/paper_order.rs").write_text(
        "fn main() -> Result<(), Box<dyn std::error::Error>> { udlex_rs::run() }\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("export", type=Path)
    p.add_argument("--disable-rules", action="store_true")
    args = p.parse_args()
    prepare(args.export, args.disable_rules)
