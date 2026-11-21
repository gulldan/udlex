"""Pack a transferable benchmark, excluding results from previous runs."""

# ruff: noqa: INP001
import argparse
import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
NAME = "opensearch-bench-3.10.0-224"
FILES = (
    "README.md",
    "LICENSE",
    "Dockerfile",
    "compose.yaml",
    "candidate/.gitignore",
    ".dockerignore",
    ".gitignore",
    "benchmark.lock.json",
    "build-inputs.sha256",
    "pack.py",
    "results/.gitkeep",
)
DIRECTORIES = ("runner", "reference", "build-inputs")


def sha256(path: Path) -> str:
    """Hash one input without loading the whole file into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def content() -> list[Path]:
    """Collect the benchmark files and verify all pinned inputs."""
    files = [BASE / s for s in FILES]
    for name in DIRECTORIES:
        files.extend(
            p
            for p in (BASE / name).rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and not p.name.startswith(".")
        )
    source = BASE / "inputs"
    lock = json.loads((BASE / "benchmark.lock.json").read_bytes())
    expected = {
        source / "DICTIONARIES.json": lock["dictionary_catalog_sha256"],
        source / "benchmark-inputs.tar.zst": lock["benchmark_inputs_sha256"],
    }
    expected.update(
        {
            BASE / "reference" / name: digest
            for name, digest in lock["reference_sha256"].items()
        }
    )
    catalog = json.loads((source / "DICTIONARIES.json").read_bytes())
    if len(catalog["dictionaries"]) != lock["dictionary_count"]:
        msg = "Dictionary inventory differs from lock"
        raise ValueError(msg)
    for item in catalog["dictionaries"]:
        archive = item["archive"]
        expected[source / "dictionaries" / archive["name"]] = archive["sha256"]
    for line in (BASE / "build-inputs.sha256").read_text().splitlines():
        checksum, name = line.split("  ", 1)
        expected[BASE / "build-inputs" / name] = checksum
    if {p for p in files if p.parent == BASE / "build-inputs"} != {
        p for p in expected if p.parent == BASE / "build-inputs"
    }:
        msg = "Build input inventory differs from lock"
        raise ValueError(msg)
    for path, checksum in expected.items():
        if sha256(path) != checksum:
            msg = f"Unexpected content in {path}"
            raise ValueError(msg)
    return sorted(set(files) | set(expected))


def create(output: Path, plugin: Path | None = None) -> None:
    """Write a portable tar and its checksum without replacing an earlier run."""
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    files = {path.relative_to(BASE).as_posix(): path for path in content()}
    if plugin is not None:
        version = json.loads((BASE / "benchmark.lock.json").read_bytes())["opensearch"]
        files[f"candidate/analysis-udlex-{version}.zip"] = plugin
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=output.name + ".", delete=False
    ) as temp:
        partial = Path(temp.name)
    try:
        with tarfile.open(partial, "w", format=tarfile.USTAR_FORMAT) as archive:
            for name, path in sorted(files.items()):
                entry = tarfile.TarInfo(f"{NAME}/{name}")
                entry.size = path.stat().st_size
                entry.mode = 0o644
                with path.open("rb") as stream:
                    archive.addfile(entry, stream)
        partial.chmod(0o644)
        checksum = sha256(partial)
        os.link(partial, output)
    finally:
        partial.unlink(missing_ok=True)
    output.with_name(output.name + ".sha256").write_text(f"{checksum}  {output.name}\n")
    print(f"{checksum}  {output.name}  ({len(files)} files)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help=f"new archive path, e.g. {NAME}.tar")
    parser.add_argument("--plugin", type=Path, help="ship and measure this plugin ZIP")
    args = parser.parse_args()
    create(args.output, args.plugin)
