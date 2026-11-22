"""Validate release refs and wheel metadata for GitHub Actions."""

# ruff: noqa: INP001

import argparse
import gzip
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

VERSION = r"[0-9]+\.[0-9]+\.[0-9]+"
DICTIONARY_PROJECT = re.compile(r"udlex-dict-[a-z]{3}(?:-nc)?")


def git(*args: str) -> str:
    """Read Git output without invoking a shell."""
    return subprocess.check_output(["git", *args], text=True).strip()  # noqa: S603, S607


def release_commit(tag: str) -> str:
    """Resolve a stable release tag reachable from main and check its version."""
    if not re.fullmatch(f"v{VERSION}", tag):
        sys.exit("expected a stable vX.Y.Z release tag")
    commit = git("rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}")
    git("merge-base", "--is-ancestor", commit, "origin/main")
    crate = tomllib.loads(git("show", f"{commit}:rust/Cargo.toml"))
    version = crate["package"]["version"]
    if tag != f"v{version}":
        sys.exit("release tag must match the crate version")
    return commit


def prepare() -> None:
    """Pass checked source identity and build pins to the build matrix."""
    commit = git("rev-parse", "HEAD")
    if (
        os.environ["GITHUB_EVENT_NAME"] == "push"
        and release_commit(os.environ["GITHUB_REF_NAME"]) != commit
    ):
        sys.exit("release tag does not match the checkout")
    crate = tomllib.loads(Path("rust/Cargo.toml").read_text(encoding="utf-8"))
    toolchain = tomllib.loads(Path("rust-toolchain.toml").read_text(encoding="utf-8"))
    version, rust = crate["package"]["version"], toolchain["toolchain"]["channel"]
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    [maturin] = project["build-system"]["requires"]
    pin = re.fullmatch(f"maturin==({VERSION})", maturin)
    if pin is None or maturin not in project["dependency-groups"]["dev"]:
        sys.exit("the build backend and dev tools must pin the same maturin version")
    if any(not re.fullmatch(VERSION, value) for value in (version, rust)):
        sys.exit("expected stable crate and Rust versions")
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        output.write(
            f"commit={commit}\nversion={version}\nrust={rust}\nmaturin=v{pin[1]}\n"
        )


def dictionary_wheel(tag: str, project: str) -> str:
    """Select one dictionary wheel from the verified release manifest."""
    if not DICTIONARY_PROJECT.fullmatch(project):
        sys.exit("use an exact udlex-dict project name")
    manifest = json.loads(Path("DICTIONARIES.json").read_text(encoding="utf-8"))
    if manifest["release"] != tag:
        sys.exit("dictionary manifest does not match the release tag")
    version = tag.removeprefix("v")
    wheel = f"{project.replace('-', '_')}-{version}-py3-none-any.whl"
    approved = [
        entry["wheel"]["name"]
        for entry in manifest["dictionaries"]
        if entry["distribution"] == project
    ]
    if approved != [wheel]:
        sys.exit(f"expected one matching dictionary wheel in the release for {project}")
    return wheel


def check_wheel() -> None:
    """Check the built wheel's authoritative ABI tags and native extension."""
    platform = os.environ["WHEEL_PLATFORM"]
    suffix = ".pyd" if platform.startswith("win") else ".so"
    python = os.environ["BUILD_PYTHON"]
    abis = {
        "3.11": ("cp311", "abi3"),
        "3.14t": ("cp314", "cp314t"),
        "3.15t": ("cp315", "abi3.abi3t"),
    }
    interpreter, abi = abis[python]
    version = os.environ["EXPECTED_VERSION"]
    expected = f"udlex-{version}-{interpreter}-{abi}-{platform}.whl"
    wheels = list(Path("dist").glob("*.whl"))
    if len(wheels) != 1 or wheels[0].name != expected:
        sys.exit(f"expected {expected}, found {wheels}")
    expected_tags = {f"{interpreter}-{item}-{platform}" for item in abi.split(".")}
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = wheel.namelist()
        wheel_metadata = f"udlex-{version}.dist-info/WHEEL"
        if names.count(wheel_metadata) != 1:
            sys.exit("expected one WHEEL metadata file")
        metadata = wheel.read(wheel_metadata).decode("utf-8")
        tags = {line[5:] for line in metadata.splitlines() if line.startswith("Tag: ")}
        extensions = [name for name in names if name.endswith(suffix)]
        if tags != expected_tags or len(extensions) != 1:
            sys.exit(f"incompatible wheel tags or native extension in {expected}")
        extension = wheel.read(extensions[0])
        build_paths = (b"/Users/", b"/home/runner/", b"/root/.cargo/", b"runneradmin")
        if any(path in extension for path in build_paths):
            sys.exit("wheel extension exposes build paths")
        if any(
            b"file://" in wheel.read(name)
            for name in names
            if name.startswith(f"udlex-{version}.dist-info/sboms/")
        ):
            sys.exit("wheel SBOM exposes local file paths")
    if python == "3.15t" and not re.fullmatch(
        r"udlex/_udlex\.abi3t(?:-[A-Za-z0-9_-]+)?\.so", extensions[0]
    ):
        sys.exit("free-threaded wheel lacks an abi3t extension")


GNU_LIBS = "-lgcc_s -lutil -lrt -lpthread -lm -ldl -lc"
APPLE_LIBS = "-liconv -lSystem -lc -lm"
# Platform: summary, shared suffix, static flags from --print=native-static-libs.
NATIVE = {
    "linux-x86_64": ("Linux x86-64, glibc 2.34 or later", "so", GNU_LIBS),
    "linux-aarch64": ("Linux aarch64, glibc 2.34 or later", "so", GNU_LIBS),
    "macos-arm64": ("macOS arm64 (11.0 or later)", "dylib", APPLE_LIBS),
    "macos-x86_64": ("macOS x86_64 (10.12 or later)", "dylib", APPLE_LIBS),
}


def pack_native(platform: str, built: Path) -> None:
    """Pack the C library of one platform as a byte-reproducible archive."""
    summary, shared, static = NATIVE[platform]
    library = f"libudlex_capi.{shared}"
    version = os.environ["EXPECTED_VERSION"]
    name = f"udlex-native-{version}-{platform}"
    linking = (
        f"{summary}. Include include/udlex.h.\n"
        "Shared link: -L. -ludlex_capi\n"
        f"Static link: libudlex_capi.a {static}\n"
    ).encode()
    members = [
        ("LICENSE", Path("LICENSE").read_bytes(), 0o644),
        ("LINKING.txt", linking, 0o644),
        ("include/udlex.h", Path("rust/capi/include/udlex.h").read_bytes(), 0o644),
        ("libudlex_capi.a", (built / "libudlex_capi.a").read_bytes(), 0o644),
        (library, (built / library).read_bytes(), 0o755),
    ]
    # Never dist/: locally it may hold dictionary builds.
    Path("dist-native").mkdir(exist_ok=True)
    with (
        Path("dist-native", f"{name}.tar.gz").open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as packed,
        tarfile.open(fileobj=packed, mode="w", format=tarfile.USTAR_FORMAT) as tar,
    ):
        for directory in (name, f"{name}/include"):
            entry = tarfile.TarInfo(directory)
            entry.type, entry.mode = tarfile.DIRTYPE, 0o755
            tar.addfile(entry)
        for path, data, mode in sorted(members):
            entry = tarfile.TarInfo(f"{name}/{path}")
            entry.size, entry.mode = len(data), mode
            tar.addfile(entry, io.BytesIO(data))


def main() -> None:
    """Run one release check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "step", choices=("prepare", "tag", "dictionary-wheel", "wheel", "native")
    )
    parser.add_argument("tag", nargs="?")
    parser.add_argument("built", nargs="?", type=Path)
    args = parser.parse_args()
    if args.step == "prepare":
        prepare()
    elif args.step == "tag":
        if args.tag is None:
            parser.error("tag step requires a release tag")
        print(release_commit(args.tag))
    elif args.step == "dictionary-wheel":
        if args.tag is None:
            parser.error("dictionary-wheel step requires a release tag")
        print(dictionary_wheel(args.tag, os.environ["RELEASE_PROJECT"]))
    elif args.step == "native":
        if args.tag not in NATIVE or args.built is None:
            parser.error("native step requires a platform and the build directory")
        pack_native(args.tag, args.built)
    else:
        check_wheel()


if __name__ == "__main__":
    main()
