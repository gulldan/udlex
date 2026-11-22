"""An upgraded reader can recover bundles left by the old build pipeline."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import pytest

import udlex
from udlex import catalog
from udlex._io import (
    BUILD_JOURNAL,
    mark_build_transaction_committed,
    prepare_build_transaction,
    recover_bundle_transactions,
)

if TYPE_CHECKING:
    from pathlib import Path


def _manifest(generation: str | None) -> bytes:
    rows = (
        [{"iso639_3": "eng", "config": "en_test", "generation": generation}]
        if generation is not None
        else []
    )
    return json.dumps(
        {"format_version": udlex.FORMAT_VERSION, "lexicons": rows}
    ).encode()


def _interrupted_build(root: Path, *, replacement: str | None = "replacement") -> None:
    old = "original"
    config = root / "en_test"
    for generation in (old, replacement):
        if generation is None:
            continue
        leaf = config / ".generations" / generation
        leaf.mkdir(parents=True)
        (leaf / "meta.json").write_text(
            json.dumps({"generation": generation}), encoding="utf-8"
        )
    payload = prepare_build_transaction(
        root,
        old_manifest=_manifest(old),
        new_manifest=_manifest(replacement),
        operations=[("en_test", old, replacement)],
    )
    (root / "manifest.json").write_bytes(_manifest(replacement))
    if replacement is not None:
        (config / "current").write_text(replacement + "\n", encoding="utf-8")
    else:
        mark_build_transaction_committed(root, payload)


def test_catalogue_read_recovers_build_journal_before_native_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bundle"
    _interrupted_build(root)

    def native_languages(directory: str) -> tuple[()]:
        assert directory == str(root)
        assert not (root / BUILD_JOURNAL).exists()
        assert (root / "manifest.json").read_bytes() == _manifest("original")
        assert (root / "en_test" / "current").read_text(
            encoding="utf-8"
        ) == "original\n"
        return ()

    monkeypatch.setattr(catalog, "languages", native_languages)
    assert catalog.read(root) == ()


def test_recovery_rejects_ambiguous_external_pointer_mutation(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _interrupted_build(root)
    current = root / "en_test" / "current"
    current.write_text("external\n", encoding="utf-8")

    with pytest.raises(OSError, match="ambiguous build transaction mutation"):
        recover_bundle_transactions(root)

    assert (root / BUILD_JOURNAL).is_file()
    assert current.read_text(encoding="utf-8") == "external\n"
    assert (root / "manifest.json").read_bytes() == _manifest("replacement")


@pytest.mark.xfail(
    sys.platform == "win32",
    reason="Windows cannot rename a lexicon directory while its writer lock is held",
    strict=True,
)
def test_recovery_finishes_a_committed_config_retirement(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _interrupted_build(root, replacement=None)
    assert (root / "en_test").is_dir()

    assert recover_bundle_transactions(root) == "committed"

    assert not (root / BUILD_JOURNAL).exists()
    assert not (root / "en_test").exists()
    assert (root / "manifest.json").read_bytes() == _manifest(None)
