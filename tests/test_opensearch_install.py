import hashlib
import importlib.util
import io
import json
import string
import tarfile
from itertools import islice, product
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def installer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    script = (
        Path(__file__).resolve().parents[1]
        / "integrations/opensearch/docker/install-dictionaries.py"
    )
    spec = importlib.util.spec_from_file_location("install_dictionaries", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    packages = tmp_path / "dictionaries"
    packages.mkdir()
    entries = []
    for letters in islice(product(string.ascii_lowercase, repeat=3), 224):
        iso = "".join(letters)
        package = packages / f"{iso}.tar.gz"
        meta = {
            "iso639_3": iso,
            "generation": "g1",
            "files": {"table": {"size": 3}},
        }
        members = {
            "current": b"g1\n",
            ".generations/g1/meta.json": json.dumps(meta).encode(),
            ".generations/g1/table": b"abc",
        }
        with tarfile.open(package, "w:gz") as tar:
            for name, content in members.items():
                member = tarfile.TarInfo(f"bundle/{iso}/{name}")
                member.size = len(content)
                tar.addfile(member, io.BytesIO(content))
        entries.append(
            {
                "iso639_3": iso,
                "compiled_generation": "g1",
                "tables": [{"files": ["table", "meta.json"]}],
                "archive": {"name": package.name, "sha256": module.checksum(package)},
            }
        )
    catalog = json.dumps({"dictionaries": entries}).encode()
    (tmp_path / "DICTIONARIES.json").write_bytes(catalog)
    monkeypatch.setattr(module, "KIT", tmp_path)
    monkeypatch.setattr(module, "CATALOG_SHA256", hashlib.sha256(catalog).hexdigest())
    return module


def test_installer_rechecks_published_tables(
    installer: ModuleType, tmp_path: Path
) -> None:
    destination = tmp_path / "installed"
    installer.install(destination)
    (tmp_path / "dictionaries").rename(tmp_path / "archives-offline")
    installer.install(destination)
    (destination / "aaa/.generations/g1/table").write_bytes(b"a")
    with pytest.raises(ValueError, match="incomplete dictionary file for aaa"):
        installer.install(destination)


def test_invalid_archive_leaves_no_partial_install(
    installer: ModuleType, tmp_path: Path
) -> None:
    destination = tmp_path / "installed"
    (tmp_path / "dictionaries/aaa.tar.gz").write_bytes(b"incomplete")
    with pytest.raises(ValueError, match="checksum mismatch for aaa"):
        installer.install(destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".udlex-install-*"))


def test_new_install_opens_through_both_names(
    installer: ModuleType, tmp_path: Path
) -> None:
    destination = tmp_path / "config/udlex"
    installer.install(destination)
    legacy = destination.with_name(installer.LEGACY_DIRECTORY)
    assert legacy.is_symlink()
    assert legacy.samefile(destination)
    installer.install(destination)


def test_upgrade_reuses_previous_install(installer: ModuleType, tmp_path: Path) -> None:
    destination = tmp_path / "config/udlex"
    legacy = destination.with_name(installer.LEGACY_DIRECTORY)
    installer.install(legacy)
    (tmp_path / "dictionaries").rename(tmp_path / "archives-offline")
    installer.install(destination)
    assert destination.is_symlink()
    assert destination.samefile(legacy)
