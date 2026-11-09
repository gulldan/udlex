"""Public API, token casing, context keys and diagnostics."""

from __future__ import annotations

import json
import sys
import warnings
from typing import TYPE_CHECKING

import pytest

import udlex

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _bundle(tmp_path: Path, write_lexicon: Callable[..., None]) -> Path:
    bundle = tmp_path / "bundle"
    write_lexicon(
        bundle / "ru_fixture",
        name="ru_fixture",
        iso639_3="rus",
        language="Russian",
        forms={"котами": "кот", "кошками": "кошка", "собаками": "собака"},
        external={"ладоге": "ладога"},
    )
    return bundle


def test_answers_reuse_string_objects(
    tmp_path: Path, write_lexicon: Callable[..., None]
) -> None:
    """An unchanged token keeps its object; a repeated table lemma shares one."""
    lexicon = udlex.load(_bundle(tmp_path, write_lexicon), "rus")
    tokens = ["котами", "неизвестное", "кошками", "котами"]
    first = lexicon.lemmatize(tokens)
    second = lexicon.lemmatize(tokens)
    assert first == second == ["кот", "неизвестное", "кошка", "кот"]
    assert first[1] is tokens[1]
    # Free-threaded, a shared lemma string would serialise threads on its lock.
    if getattr(sys, "_is_gil_enabled", lambda: True)():
        assert first[0] is first[3] is second[0] is lexicon.lemma("котами")
    assert lexicon.lemmatize(["Котами"], sentence_initial=[True]) == ["кот"]
    assert lexicon.lemma_span("ладоге") == "ладога"


def test_deprecated_names_warn_once_and_match_the_new_spelling(
    tmp_path: Path, write_lexicon: Callable[..., None]
) -> None:
    lexicon = udlex.load(_bundle(tmp_path, write_lexicon), "rus")
    pairs = [
        (
            lambda: lexicon.lemma_with_info("котами"),  # ty: ignore[deprecated]
            lambda: lexicon.lemma("котами", info=True),
        ),
        (
            lambda: lexicon.lemmatize_with_info(["котами"]),  # ty: ignore[deprecated]
            lambda: lexicon.lemmatize(["котами"], info=True),
        ),
        (
            lambda: lexicon.lemma_span_with_info(["котами"]),  # ty: ignore[deprecated]
            lambda: lexicon.lemma_span(["котами"], info=True),
        ),
        (
            lambda: lexicon.lemma_span_text_with_info("котами"),  # ty: ignore[deprecated]
            lambda: lexicon.lemma_span("котами", info=True),
        ),
        (
            lambda: lexicon.lemma_span_text("котами"),  # ty: ignore[deprecated]
            lambda: lexicon.lemma_span("котами"),
        ),
    ]
    for old, new in pairs:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert repr(old()) == repr(new())
        # The first call of a process may warn; the deprecated name must never
        # warn twice, because it sits in per-token loops.
        assert all(issubclass(w.category, DeprecationWarning) for w in caught)
        with warnings.catch_warnings(record=True) as again:
            warnings.simplefilter("always")
            old()
        assert again == []


def test_beginner_mistakes_get_actionable_errors(
    tmp_path: Path,
    write_lexicon: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path, write_lexicon)
    lexicon = udlex.load(bundle, "rus")

    with pytest.raises(TypeError, match="list of tokens, not a str"):
        lexicon.lemmatize("котами собаками")
    with pytest.raises(TypeError, match="apply to a token list"):
        lexicon.lemma_span("котами собаками", 1)
    assert lexicon.lemma_span("котами") == "кот"

    with pytest.raises(FileNotFoundError, match="arguments look swapped"):
        udlex.load("rus", bundle)  # ty: ignore[invalid-argument-type]
    with pytest.raises(OSError, match=r"no lexicon for 'ru' \(did you mean 'rus'\?\)"):
        udlex.load(bundle, "ru")

    monkeypatch.setenv("UDLEX_BUNDLE", str(bundle))
    assert udlex.load("rus").lemma("котами") == "кот"
    with pytest.raises(
        ValueError, match=r"did you mean 'rus'.*pass a bundle directory"
    ):
        udlex.load("Russian")
    with pytest.raises(ValueError, match="is a path, not a language"):
        udlex.load(str(bundle / "ru_fixture"))

    monkeypatch.setenv("UDLEX_USER_BUNDLE", str(tmp_path / "missing-user"))
    with pytest.raises(OSError, match="UDLEX_USER_BUNDLE="):
        udlex.load("rus")


def test_without_external_disables_its_lowercase_retry_too(
    tmp_path: Path, write_lexicon: Callable[..., None]
) -> None:
    bundle = _bundle(tmp_path, write_lexicon)
    assert udlex.load(bundle, "rus").lemma("ЛАДОГЕ", info=True).source is (
        udlex.Source.EXTERNAL_LOWER
    )
    without = udlex.load(bundle, "rus", use_external=False)
    assert without.lemma("ЛАДОГЕ", info=True).source is not udlex.Source.EXTERNAL_LOWER


def test_lemmatize_many_equals_one_call_per_sentence(
    tmp_path: Path, write_lexicon: Callable[..., None]
) -> None:
    lexicon = udlex.load(_bundle(tmp_path, write_lexicon), "rus")
    sentences = [["Котами", "кошками"], [], ["ЛАДОГЕ", "собаками", "котами"]]
    assert lexicon.lemmatize_many(sentences) == [
        lexicon.lemmatize(sentence) for sentence in sentences
    ]
    assert [
        [repr(row) for row in rows]
        for rows in lexicon.lemmatize_many(sentences, info=True)
    ] == [
        [repr(row) for row in lexicon.lemmatize(sentence, info=True)]
        for sentence in sentences
    ]
    with pytest.raises(TypeError, match="list of tokens, not a str"):
        lexicon.lemmatize_many(["котами кошками"])
    span = ["в", "Ладоге", "и", "котами"]
    assert lexicon.lemma_span(span, 1, 2) == lexicon.lemma_span(["Ладоге"])
    with pytest.raises(ValueError, match="out of bounds"):
        lexicon.lemma_span(span, 3, 9)


def test_stamped_surface_recases_tokens_and_the_override_wins(
    tmp_path: Path, write_lexicon: Callable[..., None]
) -> None:
    lexicon = tmp_path / "uk_test"
    write_lexicon(
        lexicon,
        name="uk_test",
        iso639_3="ukr",
        language="Ukrainian",
        forms={"у": "у"},
        external={"рівненщини": "рівненщина"},
        token_case="surface",  # noqa: S106 - не пароль
    )
    tokens = ["у", "Рівненщини"]
    stamped = udlex.Lexicon(str(lexicon))
    assert stamped.lemmatize(tokens) == ["у", "Рівненщина"]
    # В начале предложения заглавная — не имя по позиции.
    assert stamped.lemmatize(tokens[1:]) == ["рівненщина"]
    for override, expected in (("keep", "рівненщина"), ("surface", "Рівненщина")):
        lemmas = udlex.Lexicon(str(lexicon), token_case=override).lemmatize(tokens)
        assert lemmas == ["у", expected]
    with pytest.raises(ValueError, match="token_case"):
        udlex.Lexicon(str(lexicon), token_case="title")  # noqa: S106 - не пароль


def test_keys_answer_only_the_calls_that_pass_context(tmp_path: Path) -> None:
    verb = {
        "tokens": ["Они", "стали", "ждать"],
        "lemmas": ["они", "стать", "ждать"],
        "upos": ["PRON", "VERB", "VERB"],
    }
    noun = {
        "tokens": ["Из", "стали"],
        "lemmas": ["из", "сталь"],
        "upos": ["ADP", "NOUN"],
    }
    train = tmp_path / "train.jsonl"
    train.write_text(
        "".join(json.dumps(row) + "\n" for row in [verb, verb, verb, noun, noun]),
        encoding="utf-8",
    )
    destination = tmp_path / "ru_context"
    udlex.build_lexicon(
        str(destination),
        train=str(train),
        dev=None,
        test=None,
        paradigms=[],
        name="ru_context",
        language="Russian",
        iso639_3="rus",
        context_keys=True,
    )
    lexicon = udlex.Lexicon(str(destination))
    tokens = ["Из", "стали"]
    assert lexicon.lemmatize(tokens) == ["из", "стать"]
    assert lexicon.lemmatize(tokens, upos=["ADP", "NOUN"]) == ["из", "сталь"]
    assert lexicon.lemmatize(tokens, upos=["ADP", ""]) == ["из", "стать"]
    assert lexicon.lemmatize(tokens, context=True) == ["из", "сталь"]
    infos = lexicon.lemmatize(tokens, upos=["ADP", "NOUN"], info=True)
    assert [info.source for info in infos] == [udlex.Source.EXACT, udlex.Source.CONTEXT]
    # Граница предложения из маски — край и для соседей.
    sentence = ["Они", "стали", "ждать"]
    assert lexicon.lemmatize(sentence, context=True)[1] == "стать"
    split = lexicon.lemmatize(
        sentence, context=True, sentence_initial=[True, False, True]
    )
    assert split[1] == "сталь"
    with pytest.raises(ValueError, match="upos has 1 items, expected 2"):
        lexicon.lemmatize(tokens, upos=["NOUN"])
