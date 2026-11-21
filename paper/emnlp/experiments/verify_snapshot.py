"""Verify the canonical engine and every dictionary before an evaluation."""

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

MAIN = "7afe2fe8f3ea6ea56fff835bbc61d8af49da731f"
TAG = "2270bc1cb1c7144a85cf581318ca669417608fe4"
CATALOG = "57be9c16077c1a81ccb1f04d5fdac101364133f08f90e70069925e0cfea4804b"
ENGINE = "udlex-1.0.0-cp311-abi3-manylinux_2_34_x86_64.whl"
EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    tmp.replace(path)


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def verify(repo, assets, build, report):
    import udlex
    import udlex._udlex

    sources = ["rust", "src", "pyproject.toml", "uv.lock", "rust-toolchain.toml"]
    if git(repo, "rev-parse", "v1.0.0^{}") != TAG:
        raise ValueError("Unexpected v1.0.0 tag")
    git(repo, "merge-base", "--is-ancestor", MAIN, "HEAD")
    git(repo, "diff", "--exit-code", TAG, "HEAD", "--", *sources)
    git(repo, "diff", "--exit-code", "HEAD", "--", *sources)
    if udlex.__version__ != "1.0.0" or sha(assets / "DICTIONARIES.json") != CATALOG:
        raise ValueError("Unexpected engine or dictionary catalog")
    data = read(assets / "DICTIONARIES.json")
    locks = {a["name"]: a for a in read(EVIDENCE / "asset-digests.json")}
    rows = {r["iso639_3"]: r for r in data["dictionaries"]}
    if (
        len(rows) != 172
        or set(rows) != set(data["language_codes"])
        or data["release"] != "v1.0.0"
    ):
        raise ValueError("Incomplete dictionary catalog")
    wanted = {"DICTIONARIES.json", ENGINE, *(r["wheel"]["name"] for r in rows.values())}
    if len(locks) != 355 or not wanted.issubset(locks):
        raise ValueError("Asset roster differs from the upstream draft")
    for name, pin in locks.items():
        path = assets / name
        if path.stat().st_size != pin["size"] or sha(path) != pin["sha256"]:
            raise ValueError(f"Asset digest mismatch for {name}")
    with zipfile.ZipFile(assets / ENGINE) as wheel:
        for entry in wheel.namelist():
            if entry.startswith("udlex/") and entry.endswith((".py", ".so")):
                installed = Path(udlex.__file__).parent / entry.removeprefix("udlex/")
                if sha(installed) != hashlib.sha256(wheel.read(entry)).hexdigest():
                    raise ValueError(
                        f"Installed runtime differs from the wheel at {entry}"
                    )
    inventory = {}
    for iso, row in sorted(rows.items()):
        wheel = row["wheel"]
        pin = locks[wheel["name"]]
        if (
            row["engine_commit"] != TAG
            or row["dictionary_version"] != "1.0.0"
            or any(wheel[k] != pin[k] for k in ("name", "size", "sha256"))
        ):
            raise ValueError(f"Dictionary identity changed for {iso}")
        folder = build / iso
        prefix = f"udlex_bundle/{iso}/"
        with zipfile.ZipFile(assets / wheel["name"]) as archive:
            for entry in archive.namelist():
                if entry.startswith(prefix) and not entry.endswith("/"):
                    relative = Path(entry.removeprefix(prefix))
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError(f"Invalid wheel member {entry}")
                    target = folder / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(entry) as src, target.open("wb") as dest:
                        shutil.copyfileobj(src, dest)
        gen = (folder / "current").read_text().strip()
        meta = read(folder / ".generations" / gen / "meta.json")
        if (
            gen != row["compiled_generation"]
            or gen != meta["generation"]
            or meta["iso639_3"] != iso
        ):
            raise ValueError(f"Unexpected compiled generation for {iso}")
        lex = udlex.Lexicon(str(folder), use_user=False)
        examples = row["examples"]
        verified = sum(
            lex.lemmatize([word])[0] == lemma for word, lemma in examples.items()
        )
        ud_repo = (
            row["source_home"].split("/tree/")[0].split("/")[-1]
            if "/UniversalDependencies/UD_" in row["source_home"]
            else None
        )
        inventory[iso] = {
            "name": row["language"],
            "source": row["source"],
            "source_url": row["download"],
            "source_split": row.get("source_split"),
            "ud_repository": ud_repo,
            "variant": meta["variant"],
            "generation": gen,
            "wheel_sha256": wheel["sha256"],
            "license": row["license"],
            "examples": len(examples),
            "examples_matched": verified,
        }
    save(
        report,
        {
            "upstream": "gulldan/udlex",
            "base_commit": MAIN,
            "dictionary_engine_commit": TAG,
            "upstream_release_status_at_snapshot": "draft",
            "version": udlex.__version__,
            "engine_wheel_sha256": locks[ENGINE]["sha256"],
            "catalog_sha256": CATALOG,
            "rust_tree": git(repo, "rev-parse", "HEAD:rust"),
            "python_tree": git(repo, "rev-parse", "HEAD:src"),
            "verified_assets": len(locks),
            "evaluation_assets": len(wanted),
            "languages": inventory,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("repo", "assets", "bundle", "report"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    verify(args.repo, args.assets, args.bundle, args.report)
