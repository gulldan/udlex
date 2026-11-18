"""Compare packaged Python and Rust sources directly with the pinned Git tree."""
import argparse
import tarfile
import tomllib
import zipfile
from pathlib import Path

from verify_snapshot import ENGINE, MAIN, TAG, git, save, sha


def verify(repo, assets, out):
    sdist = assets / "udlex-1.0.0.tar.gz"
    tracked = git(repo, "ls-files", "src/udlex", "rust/src", "rust/Cargo.lock", "pyproject.toml").splitlines()
    with tarfile.open(sdist, "r:gz") as archive:
        root = "udlex-1.0.0/"
        for name in tracked:
            if archive.extractfile(root + name).read() != (repo / name).read_bytes():
                raise ValueError(f"Source distribution differs from Git at {name}")
        packaged = tomllib.loads(archive.extractfile(root + "rust/Cargo.toml").read().decode())
        canonical = tomllib.loads((repo / "rust/Cargo.toml").read_text())
        if canonical["package"]["readme"] != "../README.md" or packaged["package"]["readme"] != "README.md":
            raise ValueError("Unexpected README packaging path")
        canonical["package"]["readme"] = "README.md"
        if canonical["workspace"].pop("members") != ["capi"] or canonical["workspace"].pop("default-members") != ["."]:
            raise ValueError("Unexpected native workspace packaging change")
        if canonical != packaged:
            raise ValueError("Unknown Cargo manifest difference between Git and the source distribution")
    with zipfile.ZipFile(assets / ENGINE) as wheel:
        modules = [n for n in wheel.namelist() if n.startswith("udlex/") and n.endswith(".py")]
        for name in modules:
            if wheel.read(name) != (repo / "src" / name).read_bytes():
                raise ValueError(f"Wheel module differs from Git at {name}")
    save(out, {"base_commit": MAIN, "tag_commit": TAG, "source_distribution_sha256": sha(sdist),
               "wheel_sha256": sha(assets / ENGINE), "sdist_files_equal_to_git": len(tracked),
               "python_modules_equal_to_git": len(modules), "cargo_manifest_packaging_changes":
               ["README path inside sdist", "standalone core crate omits C API workspace membership"]})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "assets", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    verify(a.repo, a.assets, a.output)
