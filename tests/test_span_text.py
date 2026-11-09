"""Спановый вход для готового NER-упоминания, пришедшего одной строкой.

Владелец отдаёт упоминание строкой с пробелом внутри («Ленинградские
области»), а не списком токенов. Проверяется, что такой вход — тот же
спановый контракт, что и `lemma_span` по токенам, а не узкий профиль
`entity=True`, и что зазоры между токенами не схлопываются по дороге.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

import udlex
from udlex import _udlex

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

# Четырёх наблюдений хватает, чтобы суффиксное правило было выучено, и все
# четыре — с заглавной: правила разложены по регистровым корзинам, а проверяем
# мы их на упоминании с заглавной.
NAMES = {
    "Пушкина": "Пушкин",
    "Гагарина": "Гагарин",
    "Ленина": "Ленин",
    "Есенина": "Есенин",
}
ENTITIES = [
    ("Ленинградские области", "Ленинградская область"),
    ("нижний новгород", "Нижний Новгород"),
]


def _russian_lexicon(
    destination: Path,
    *,
    use_entities_lower: bool | None = None,
    stamped_lower: bool = False,
) -> udlex.Lexicon:
    """Маленький русский лексикон с многословной entity-таблицей."""
    training = [(form, lemma, False) for form, lemma in NAMES.items()]
    training.append(("областями", "область", False))
    tables = udlex.build(training, [("Ладоге", "Ладога")], ENTITIES)
    entries = dict(tables.write_sources(str(destination), "full", (0.0, 0.0)))
    (destination / "meta.json").write_text(
        json.dumps(
            {
                "name": destination.name,
                "language": "Russian",
                "iso639_3": "rus",
                "source": "test",
                "training_tokens": len(training),
                "variant": "full",
                "selected_on": None,
                "max_suffix": 6,
                "format_version": udlex.FORMAT_VERSION,
                "entity_encoding": udlex.ENTITY_ENCODING,
                "entries": entries,
                "traffic": {},
                "gold_tokens": None,
                "accuracy": None,
                "baseline": None,
                "beats_baseline": None,
                **({"entity_lower": True} if stamped_lower else {}),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    udlex.compile_bundle(destination)
    return udlex.Lexicon(str(destination), use_entities_lower=use_entities_lower)


def test_a_ready_mention_as_one_string_takes_the_whole_span_path(
    tmp_path: Path,
) -> None:
    lexicon = _russian_lexicon(tmp_path / "ru_text")

    # Ровно тот случай, который называет владелец: готовый NER одной строкой.
    assert lexicon.lemma_span("Ленинградские области") == "Ленинградская область"
    answer = lexicon.lemma_span("Ленинградские области", info=True)
    assert (
        answer.lemma,
        answer.source,
        answer.source.is_known(),
        answer.lemma != "Ленинградские области",
    ) == (
        "Ленинградская область",
        udlex.Source.ENTITY,
        True,
        True,
    )

    # Узкий профиль на той же строке умеет только точный entity-поиск: здесь
    # он совпадает, но всё, что ниже, ему недоступно — см. следующий тест.
    assert (
        lexicon.lemma("Ленинградские области", entity=True) == "Ленинградская область"
    )


def test_the_string_entry_and_the_token_entry_answer_identically(
    tmp_path: Path,
) -> None:
    lexicon = _russian_lexicon(tmp_path / "ru_same")

    for surface, tokens, gaps in [
        ("Ленинградские области", ["Ленинградские", "области"], [" "]),
        ("Пушкина", ["Пушкина"], []),
        ("Уварова", ["Уварова"], []),
        ("Ленинградские  области", ["Ленинградские", "области"], ["  "]),
    ]:
        by_text = lexicon.lemma_span(surface, info=True)
        by_tokens = lexicon.lemma_span(tokens, separators=gaps or None, info=True)
        assert (by_text.lemma, by_text.source) == (by_tokens.lemma, by_tokens.source), (
            surface
        )


def test_gaps_are_preserved_rather_than_normalised(tmp_path: Path) -> None:
    lexicon = _russian_lexicon(tmp_path / "ru_gaps")

    # Двойной пробел — другое упоминание: ключ с одним пробелом не спрашивают,
    # и поверхность возвращается побайтово. Схлопывание засчитало бы ответ на
    # строку, которой в тексте не было.
    widened = lexicon.lemma_span("Ленинградские  области", info=True)
    assert (
        widened.lemma,
        widened.source,
        widened.lemma != "Ленинградские  области",
    ) == (
        "Ленинградские  области",
        udlex.Source.IDENTITY,
        False,
    )
    for surface in ["Совет\tЕвропы", "Нижний Новгород", "Рады — Европы"]:
        assert lexicon.lemma_span(surface) == surface

    # Нетронутый ответ — тот же объект строки, а не её копия.
    surface = "Совет Европы"
    assert lexicon.lemma_span(surface) is surface


def test_one_word_string_behaves_exactly_like_a_one_word_span(
    tmp_path: Path,
) -> None:
    lexicon = _russian_lexicon(tmp_path / "ru_single")

    # Обычная цепочка токена целиком: exact, external и — на незнакомом слове,
    # а их около четверти — суффиксное правило.
    known = lexicon.lemma_span("Пушкина", info=True)
    assert (known.lemma, known.source) == ("Пушкин", udlex.Source.EXACT)
    external = lexicon.lemma_span("Ладоге", info=True)
    assert (external.lemma, external.source) == ("Ладога", udlex.Source.EXTERNAL)
    guessed = lexicon.lemma_span("Уварова", info=True)
    assert (guessed.lemma, guessed.source, guessed.source.is_known()) == (
        "Уваров",
        udlex.Source.RULE,
        False,
    )

    # Узкий entity-профиль на тех же словах не умеет ничего из этого.
    for surface in ["Пушкина", "Ладоге", "Уварова"]:
        assert lexicon.lemma(surface, entity=True) == surface


def test_the_locale_lowercase_entity_stage_needs_the_same_opt_in(
    tmp_path: Path,
) -> None:
    strict = _russian_lexicon(tmp_path / "ru_strict")
    relaxed = _russian_lexicon(tmp_path / "ru_relaxed", use_entities_lower=True)

    # Таблица знает ключ в нижнем регистре; ответ у обоих совпадает по тексту,
    # но у strict он остался поверхностью, а не найденной записью.
    assert (
        strict.lemma_span("Нижний Новгород", info=True).source is udlex.Source.IDENTITY
    )
    relaxed_answer = relaxed.lemma_span("Нижний Новгород", info=True)
    assert (
        relaxed_answer.lemma,
        relaxed_answer.source,
        relaxed_answer.source.is_known(),
    ) == (
        "Нижний Новгород",
        udlex.Source.ENTITY_LOWER,
        True,
    )


def test_a_policy_stamped_lowercase_retry_is_the_default(tmp_path: Path) -> None:
    # Измеренная политика открыла ретрай — он включён без опций, но явный
    # выбор вызывающего по-прежнему побеждает.
    stamped = _russian_lexicon(tmp_path / "ru_stamped", stamped_lower=True)
    answer = stamped.lemma_span("Нижний Новгород", info=True)
    assert answer.source is udlex.Source.ENTITY_LOWER
    # Политика мерила ретрай на спанах; узкий entity-профиль токена его не
    # получает (на SlavicNER ru он там проигрывает).
    narrow = stamped.lemma("Нижний Новгород", entity=True, info=True)
    assert narrow.source is udlex.Source.IDENTITY
    closed = _russian_lexicon(
        tmp_path / "ru_closed", stamped_lower=True, use_entities_lower=False
    )
    assert (
        closed.lemma_span("Нижний Новгород", info=True).source is udlex.Source.IDENTITY
    )


def test_degenerate_surfaces_raise_instead_of_being_trimmed(tmp_path: Path) -> None:
    lexicon = _russian_lexicon(tmp_path / "ru_degenerate")

    with pytest.raises(ValueError, match="span surface is empty"):
        lexicon.lemma_span("")
    with pytest.raises(ValueError, match="whitespace only"):
        lexicon.lemma_span("   ")
    for surface in [" Ленинградские области", "Ленинградские области ", " Пушкина"]:
        with pytest.raises(ValueError, match="leading or trailing whitespace"):
            lexicon.lemma_span(surface)
        with pytest.raises(ValueError, match="leading or trailing whitespace"):
            lexicon.lemma_span(surface, info=True)


def test_protection_returns_the_surface_without_any_lookup(tmp_path: Path) -> None:
    lexicon = _russian_lexicon(tmp_path / "ru_protected")

    guarded = lexicon.lemma_span("Ленинградские области", protected=True, info=True)
    assert (
        guarded.lemma,
        guarded.source,
        guarded.lemma != "Ленинградские области",
    ) == (
        "Ленинградские области",
        udlex.Source.PROTECTED,
        False,
    )
    assert lexicon.lemma_span("Пушкина", protected=True) == "Пушкина"


SEPARATOR = _udlex.PHRASE_SEPARATOR


KEY = f"го{SEPARATOR}да"


PATTERN = f"его{SEPARATOR}ий{SEPARATOR}а{SEPARATOR}"


def test_a_loaded_table_answers_only_uncovered_multiword_spans(
    tmp_path: Path, write_lexicon: Callable[..., None]
) -> None:
    destination = tmp_path / "bundle" / "xx_one"
    write_lexicon(
        destination,
        name="xx_one",
        iso639_3="xxa",
        language="Firstish",
        forms={"Города": "Города"},
    )
    reference = udlex.load(tmp_path / "bundle", "xxa", phrase_patterns=[])
    candidate = udlex.load(tmp_path / "bundle", "xxa", phrase_patterns=[(KEY, PATTERN)])
    span = ["Дальнего", "Города"]

    assert reference.lemma_span(span) == "Дальнего Города"
    assert candidate.lemma_span(span) == "Дальний Город"
    assert candidate.lemma_span(span, info=True).source is udlex.Source.PHRASE
    # Single-word spans are the token chain's, byte for byte.
    for token in span:
        assert candidate.lemma_span([token]) == reference.lemma_span([token])
    # A duplicate key is a broken table, not a silent last-wins.
    with pytest.raises(OSError, match="duplicated"):
        udlex.load(
            tmp_path / "bundle",
            "xxa",
            phrase_patterns=[(KEY, PATTERN), (KEY, PATTERN)],
        )


def test_the_phrase_branch_answers_before_the_stamped_composition(
    tmp_path: Path, write_lexicon: Callable[..., None]
) -> None:
    """Stamped ``compose`` shapes the reference; the phrase table answers first."""
    write_lexicon(
        tmp_path / "bundle" / "xx_one",
        name="xx_one",
        iso639_3="xxa",
        language="Firstish",
        forms={"Дальнего": "Дальний", "Города": "Городок"},
        span_multiword="compose",
    )
    reference = udlex.load(tmp_path / "bundle", "xxa", phrase_patterns=[])
    candidate = udlex.load(tmp_path / "bundle", "xxa", phrase_patterns=[(KEY, PATTERN)])
    span = ["Дальнего", "Города"]

    # No override kwargs: the stamped convention composes per-word lemmas.
    assert reference.lemma_span(span) == "Дальний Городок"
    assert reference.lemma_span(span, info=True).source is udlex.Source.COMPOSED
    # The table answers before composition ever runs.
    assert candidate.lemma_span(span) == "Дальний Город"
    assert candidate.lemma_span(span, info=True).source is udlex.Source.PHRASE
