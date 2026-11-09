"""Canonical language identities for treebank-scoped compatibility.

Overrides are keyed by treebank, never by a bare code (PADT is Modern Standard
Arabic, not the macrolanguage). Request aliases never become bundle rows,
source-table names, or benchmark cells.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection
    from typing import TypeGuard

P1_REQUESTED_LANGUAGES = (
    "eng",
    "rus",
    "arb",
    "deu",
    "fra",
    "ita",
    "jpn",
    "kor",
    "nld",
    "spa",
    "cmn",
    "tuk",
    "ukr",
    "pol",
    "lav",
    "tur",
    "fin",
    "heb",
    "por",
    "hin",
    "uzb",
    "pes",
    "fas",
    "kaz",
)
"""Ordered P1 selectors promised by the V1 research cohort."""

TREEBANK_CANONICAL_ISO: dict[str, str] = {
    "ar_padt": "arb",
    "ar_pud": "arb",
    "fa_perdt": "pes",
    "fa_seraji": "pes",
    "tk_tud": "tuk",
}
"""Exact treebanks whose language is narrower than their upstream ISO key."""

LANGUAGE_REQUEST_ALIASES: dict[str, str] = {
    "ara": "arb",
    "fas": "pes",
}
"""Explicit compatibility selectors; values are canonical artifact codes."""

CANONICAL_P1_LANGUAGES = tuple(
    dict.fromkeys(
        LANGUAGE_REQUEST_ALIASES.get(selector, selector)
        for selector in P1_REQUESTED_LANGUAGES
    )
)
"""Ordered, de-duplicated artifact identities for P1 scheduling/validation."""

EXTERNAL_RESOURCE_ISO: dict[str, str] = {
    "arb": "ara",
    "pes": "fas",
}
"""Broad upstream codes of language-neutral resources; used only at that boundary."""


def canonical_treebank_iso(config: str, upstream_iso639_3: str) -> str:
    """Return the corpus-specific ISO identity for one exact treebank.

    Args:
        config: Stable UD treebank configuration name.
        upstream_iso639_3: The broader code declared by upstream metadata.

    Returns:
        A treebank-specific code only for an evidenced exact configuration;
        otherwise the upstream code unchanged.
    """
    return TREEBANK_CANONICAL_ISO.get(config, upstream_iso639_3)


def resolve_language_request(value: str) -> str:
    """Resolve one compatibility selector without inventing a second model."""
    return LANGUAGE_REQUEST_ALIASES.get(value, value)


def external_resource_iso(canonical: str) -> str:
    """Return the broad code expected by a generic external resource."""
    return EXTERNAL_RESOURCE_ISO.get(canonical, canonical)


def source_language_candidates(canonical: str) -> tuple[str, ...]:
    """Return canonical-first physical table names for a source lookup."""
    resource = external_resource_iso(canonical)
    return (canonical,) if resource == canonical else (canonical, resource)


def source_table_scope(languages: Collection[str]) -> dict[str, tuple[str, ...]]:
    """Map each physical table name a build may open to the languages wanting it.

    Tables are named by resource code, so the requested languages are kept to
    name the one that pulled a refused receipt in. An empty request selects
    nothing; callers spell "everything" as ``None``.
    """
    scope: dict[str, list[str]] = {}
    for language in sorted(set(languages)):
        for candidate in source_language_candidates(language):
            scope.setdefault(candidate, []).append(language)
    return {name: tuple(rows) for name, rows in sorted(scope.items())}


def valid_iso(value: object) -> TypeGuard[str]:
    """Whether ``value`` is a canonical, lowercase ASCII ISO 639-3 spelling."""
    return (
        isinstance(value, str)
        and len(value) == 3
        and value.isascii()
        and value.isalpha()
        and value.islower()
    )
