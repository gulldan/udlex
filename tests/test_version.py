"""The release version lives only in the crate; the format version is separate."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

import udlex
from udlex.bundle import (
    FORMAT_RANGE,
    FORMAT_VERSION,
    MIN_FORMAT_VERSION,
    format_accepted,
    format_refusal,
)

ROOT = Path(__file__).resolve().parent.parent


def _crate_version() -> str:
    """The crate's ``[package]`` version (a regex would also hit dependencies)."""
    manifest = tomllib.loads((ROOT / "rust" / "Cargo.toml").read_text())
    return manifest["package"]["version"]


def test_the_package_reports_the_crate_version() -> None:
    assert udlex.__version__ == _crate_version()


def test_version_is_a_release_number() -> None:
    """Guard the shape, since it is what an index sorts releases by."""
    assert re.fullmatch(r"\d+\.\d+\.\d+", udlex.__version__), udlex.__version__


def test_pyproject_declares_no_version_of_its_own() -> None:
    """The declaration that keeps the crate the only place to edit."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "version" not in pyproject["project"], (
        "pyproject.toml pins a literal version again; it should stay dynamic "
        "so that the crate remains the only place a release number is written"
    )
    assert "version" in pyproject["project"].get("dynamic", ())


@pytest.mark.parametrize(
    ("declared", "readable"),
    [
        (MIN_FORMAT_VERSION - 1, False),
        (MIN_FORMAT_VERSION, True),
        (FORMAT_VERSION, True),
        (FORMAT_VERSION + 1, False),
    ],
)
def test_the_reader_accepts_a_range_and_stops_exactly_at_its_bounds(
    declared: int, *, readable: bool
) -> None:
    """The four numbers a one-version-wide rule cannot tell apart."""
    assert format_accepted(declared) is readable


@pytest.mark.parametrize("declared", [None, "4", 4.0, True, -1, 1 << 32])
def test_a_document_with_no_usable_format_number_is_not_read(declared: object) -> None:
    """The rule answers about `u32`s; everything else is simply not a version."""
    assert format_accepted(declared) is False
    assert "unsupported bundle format" in format_refusal("doc", declared)


@pytest.mark.parametrize(
    ("declared", "action", "wrong"),
    [
        (
            MIN_FORMAT_VERSION - 1,
            "rebuild the bundle with udlex.build_lexicon",
            "upgrade udlex",
        ),
        (
            FORMAT_VERSION + 1,
            "upgrade udlex",
            "rebuild the bundle with udlex.build_lexicon",
        ),
    ],
)
def test_a_refusal_names_both_versions_and_the_side_that_must_move(
    declared: int, action: str, wrong: str
) -> None:
    message = format_refusal("bundle/xxx", declared)
    assert str(declared) in message
    assert FORMAT_RANGE in message
    assert action in message
    assert wrong not in message
