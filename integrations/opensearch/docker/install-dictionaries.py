"""Install the complete dictionary set under the OpenSearch config directory."""

# ruff: noqa: INP001
import argparse
import errno
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import TypedDict

CATALOG_SHA256 = "3a05785bafbb4c049e773b7be8a3fc2683d0d0b9d8095a35d18918b5f33e24df"
MARKER = ".udlex-catalog-sha256"
KIT = Path(__file__).resolve().parent


def checksum(path: Path) -> str:
    """Calculate the SHA-256 of a catalog or package."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _Table(TypedDict):
    files: list[str]


class _Archive(TypedDict):
    name: str
    sha256: str


class _Dictionary(TypedDict):
    iso639_3: str
    compiled_generation: str
    tables: list[_Table]
    archive: _Archive


def _verify_dictionary(bundle: Path, item: _Dictionary) -> None:
    iso = item["iso639_3"]
    generation = item["compiled_generation"]
    language = bundle / iso
    if (language / "current").read_text().strip() != generation:
        msg = f"Dictionary generation mismatch for {iso}"
        raise ValueError(msg)
    directory = language / ".generations" / generation
    meta = json.loads((directory / "meta.json").read_bytes())
    files = meta["files"]
    expected = {name for part in item["tables"] for name in part["files"]}
    if (
        meta["generation"] != generation
        or meta["iso639_3"] != iso
        or expected != {"meta.json", *files}
    ):
        msg = f"Dictionary metadata mismatch for {iso}"
        raise ValueError(msg)
    for name, record in files.items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != record["size"]:
            msg = f"Missing or incomplete dictionary file for {iso}: {name}"
            raise ValueError(msg)


def _already_installed(path: Path, entries: list[_Dictionary]) -> bool:
    marker = path / MARKER
    if not marker.is_file() or marker.read_text().strip() != CATALOG_SHA256:
        return False
    for item in entries:
        _verify_dictionary(path, item)
    return True


def _extract_dictionary(archive: Path, bundle: Path, iso: str) -> None:
    with tarfile.open(archive, "r|gz") as package:
        for member in package:
            parts = PurePosixPath(member.name).parts
            if (
                not member.isfile()
                or len(parts) < 3
                or parts[0] != "bundle"
                or parts[1] not in {iso, ".manifests"}
                or ".." in parts
            ):
                msg = f"Unexpected member {member.name} in {archive.name}"
                raise ValueError(msg)
            target = bundle.joinpath(*parts[1:])
            target.parent.mkdir(parents=True, exist_ok=True)
            src = package.extractfile(member)
            if src is None:
                msg = f"Missing member data for {member.name} in {archive.name}"
                raise ValueError(msg)
            with src, target.open("xb") as dest:
                shutil.copyfileobj(src, dest, length=1024 * 1024)


def install(destination: Path) -> None:
    """Verify and publish the complete dictionary set in one directory rename."""
    catalog = KIT / "DICTIONARIES.json"
    if checksum(catalog) != CATALOG_SHA256:
        msg = "Unexpected dictionary catalog SHA-256"
        raise ValueError(msg)
    entries: list[_Dictionary] = json.loads(catalog.read_bytes())["dictionaries"]
    if len(entries) != 224 or len({item["iso639_3"] for item in entries}) != 224:
        msg = "Expected 224 distinct languages"
        raise ValueError(msg)
    if _already_installed(destination, entries):
        print("This 224-language dictionary set is already installed at", destination)
        return
    if destination.exists():
        msg = f"Choose an unused directory for this catalog: {destination}"
        raise FileExistsError(msg)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".udlex-install-", dir=destination.parent
    ) as tmp:
        bundle = Path(tmp) / "bundle"
        bundle.mkdir(mode=0o755)
        for item in entries:
            iso = item["iso639_3"]
            record = item["archive"]
            archive = KIT / "dictionaries" / record["name"]
            if checksum(archive) != record["sha256"]:
                msg = f"Dictionary checksum mismatch for {iso}"
                raise ValueError(msg)
            _extract_dictionary(archive, bundle, iso)
            _verify_dictionary(bundle, item)
        (bundle / MARKER).write_text(CATALOG_SHA256 + "\n")
        try:
            bundle.rename(destination)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            if not _already_installed(destination, entries):
                raise
    print("Installed and verified all 224 languages at", destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    os.umask(0o022)
    install(parser.parse_args().destination)
