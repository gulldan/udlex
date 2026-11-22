"""Shared test fixtures.

``write_lexicon`` writes tables through the extension and ``meta.json``
literally, so tests choose their provenance instead of deriving it from corpora.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

import udlex

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _write_lexicon(  # noqa: PLR0913 - каждый аргумент — поле записываемой меты
    destination: Path,
    *,
    name: str,
    iso639_3: str,
    language: str,
    forms: dict[str, str],
    accuracy: float | None = None,
    baseline: float | None = None,
    beats_baseline: bool | None = None,
    span_case: str | None = None,
    span_multiword: str | None = None,
    external: dict[str, str] | None = None,
    token_case: str | None = None,
) -> None:
    """Write one small lexicon: real tables, literal metadata.

    Args:
        destination: Lexicon directory to create and fill.
        name: The lexicon identifier, recorded and used as the directory name.
        iso639_3: ISO 639-3 code.
        language: Human-readable language name.
        forms: ``form -> lemma`` pairs to learn the tables from.
        accuracy: Exact-match accuracy to record, when the test needs one.
        baseline: Copy-the-form accuracy to record alongside it.
        beats_baseline: Whether the lexicon is recorded as worth using.
        span_case: Span case convention to stamp, when the test needs one.
        span_multiword: Multiword convention to stamp alongside it.
        external: ``form -> lemma`` pairs for the external table.
        token_case: Token case convention to stamp, when the test needs one.
    """
    tables = udlex.build(
        [(form, lemma, False) for form, lemma in forms.items()],
        list((external or {}).items()),
    )
    entries = dict(tables.write_sources(str(destination), "full", (0.0, 0.0)))
    meta = {
        "name": name,
        "language": language,
        "iso639_3": iso639_3,
        "source": f"{name}.train",
        "training_tokens": len(forms),
        "variant": "full",
        "selected_on": None,
        "max_suffix": 6,
        "format_version": udlex.FORMAT_VERSION,
        "entity_encoding": udlex.ENTITY_ENCODING,
        "entries": entries,
        "traffic": {},
        "gold_tokens": len(forms) if accuracy is not None else None,
        "accuracy": accuracy,
        "baseline": baseline,
        "beats_baseline": beats_baseline,
    }
    if span_case is not None:
        meta["span_case"] = span_case
    if span_multiword is not None:
        meta["span_multiword"] = span_multiword
    if token_case is not None:
        meta["token_case"] = token_case
    (destination / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    # Seal the hand-written source fixture through the real generation compiler.
    udlex.compile_bundle(destination)


@pytest.fixture
def write_lexicon() -> Callable[..., None]:
    """The lexicon writer above, handed to tests as a fixture."""
    return _write_lexicon
