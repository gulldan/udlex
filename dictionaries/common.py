"""Helpers shared by the dictionary build stages."""

from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SOURCES = HERE / "sources.json"
POLICY = HERE / "policy.json"
LANGUAGES = HERE / "languages.json"

LICENSE_CLASSES = {
    "open": (
        "CC0-1.0",
        "LicenseRef-C-UDA-1.0",
        "LicenseRef-Public-Domain",
        "MIT",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "Apache-2.0",
        "CC-BY-3.0",
        "CC-BY-4.0",
        "CC-BY-SA-2.0",
        "CC-BY-SA-2.5",
        "CC-BY-SA-3.0",
        "CC-BY-SA-4.0",
    ),
    "nc": (
        "CC-BY-NC-4.0",
        "CC-BY-NC-SA-2.5",
        "CC-BY-NC-SA-3.0",
        "CC-BY-NC-SA-4.0",
    ),
    "copyleft": (
        "GPL-2.0-only",
        "GPL-2.0-or-later",
        "GPL-3.0-only",
        "GPL-3.0-or-later",
        "AGPL-3.0-only",
        "AGPL-3.0-or-later",
        "LGPL-2.1-only",
        "LGPL-2.1-or-later",
        "LGPL-3.0-only",
        "LGPL-3.0-or-later",
        "LicenseRef-LGPL-LR",
        "LicenseRef-GPL",
    ),
}
# Strictness of licenses that may share one package; a later entry wins.
STRICTNESS = (
    "CC0-1.0",
    "LicenseRef-C-UDA-1.0",
    "LicenseRef-Public-Domain",
    "MIT",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "Apache-2.0",
    "CC-BY-3.0",
    "CC-BY-4.0",
    "CC-BY-SA-2.0",
    "CC-BY-SA-2.5",
    "CC-BY-SA-3.0",
    "CC-BY-SA-4.0",
    "CC-BY-NC-4.0",
    "CC-BY-NC-SA-2.5",
    "CC-BY-NC-SA-3.0",
    "CC-BY-NC-SA-4.0",
)


def license_class(spdx: str | None) -> str:
    """Return ``open``, ``nc``, ``copyleft`` or ``unclear`` for one SPDX id."""
    for name, members in LICENSE_CLASSES.items():
        if spdx in members:
            return name
    return "unclear"


def sha256(path: Path) -> str:
    """Hex SHA-256 of a file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest_bytes(data: bytes) -> str:
    """Hex SHA-256 of bytes."""
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: object) -> bytes:
    """Canonical, diff-friendly JSON encoding."""
    text = json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False)
    return (text + "\n").encode()


def read_json(path: Path) -> Any:  # noqa: ANN401 - JSON documents are untyped
    """Read one JSON document."""
    return json.loads(path.read_bytes())


def write_json(path: Path, value: object) -> None:
    """Write JSON atomically, so an interrupted stage never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(json_bytes(value))
    partial.replace(path)


def nfc(value: str) -> str:
    """NFC normalization, as the engine applies it."""
    return unicodedata.normalize("NFC", value)


def lexical(value: str) -> bool:
    """Whether a string can be a dictionary key or value: one token, no controls.

    Format characters such as ZWNJ stay: Persian and Indic words contain them.
    """
    return bool(
        value
        and value != "_"
        and not any(ch.isspace() or unicodedata.category(ch) == "Cc" for ch in value)
    )


def jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL split."""
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a JSONL split atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    partial.replace(path)


def jobs(default: int = 4) -> int:
    """Worker count; the shared server asks for a bounded share."""
    return int(os.environ.get("UDLEX_DICT_JOBS", default))
