"""A shared lexicon gives the same answers in every thread."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import pytest

import udlex

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.fixture
def shared_lexicon(tmp_path: Path, write_lexicon: Callable[..., None]) -> udlex.Lexicon:
    """One compiled lexicon, to be queried from many threads at once."""
    write_lexicon(
        tmp_path / "xx_one",
        name="xx_one",
        iso639_3="xxa",
        language="Firstish",
        forms={"коты": "кот", "котами": "кот", "мыла": "мыть"},
    )
    udlex.compile_bundle(str(tmp_path / "xx_one"))
    return udlex.load(str(tmp_path), "xxa")


def test_one_lexicon_answers_every_thread_identically(
    shared_lexicon: udlex.Lexicon,
) -> None:
    tokens = ["Мама", "мыла", "котами", "коты", "неизвестное"] * 200
    expected = shared_lexicon.lemmatize(tokens)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(shared_lexicon.lemmatize, tokens) for _ in range(16)]
        results = [future.result() for future in futures]

    assert all(result == expected for result in results)


def test_rich_batches_answer_every_thread_identically(
    shared_lexicon: udlex.Lexicon,
) -> None:
    tokens = ["Мама", "мыла", "котами", "коты", "неизвестное"] * 200

    def answer() -> list[tuple[str, udlex.Source]]:
        return [
            (info.lemma, info.source)
            for info in shared_lexicon.lemmatize(tokens, info=True)
        ]

    expected = answer()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _index: answer(), range(16)))

    assert all(result == expected for result in results)
