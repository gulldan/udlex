"""Transactional manual overrides stored outside trained dictionary bundles."""

from __future__ import annotations

import getpass
import os
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

from udlex._udlex import (
    _user_dictionary_entries as native_entries,
)
from udlex._udlex import (
    _user_dictionary_history as native_history,
)
from udlex._udlex import (
    _user_dictionary_import as native_import,
)
from udlex._udlex import (
    _user_dictionary_init as native_init,
)
from udlex._udlex import (
    _user_dictionary_remove as native_remove,
)
from udlex._udlex import (
    _user_dictionary_rollback as native_rollback,
)
from udlex._udlex import (
    _user_dictionary_set as native_set,
)
from udlex._udlex import (
    _user_dictionary_verify as native_verify,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

MatchMode: TypeAlias = Literal["exact", "lower"]
Pathish: TypeAlias = str | PathLike[str]


@dataclass(frozen=True, slots=True)
class UserEntry:
    """One canonical ``form -> lemma`` override."""

    form: str
    lemma: str
    match: MatchMode


@dataclass(frozen=True, slots=True)
class UserDictionaryUpdate:
    """Outcome of one idempotent immutable-generation transaction."""

    changed: bool
    generation: str | None
    entries: int


@dataclass(frozen=True, slots=True)
class UserBundleVerification:
    """Counts from authenticating a complete user-bundle root."""

    languages: int
    generations: int
    active_entries: int


def _actor(value: str | None) -> str:
    """Choose a stable local audit identity when the caller omits one."""
    return value if value is not None else getpass.getuser()


def _update(raw: tuple[bool, str | None, int]) -> UserDictionaryUpdate:
    return UserDictionaryUpdate(*raw)


class UserDictionary:
    """CRUD facade for one ISO language in one independent user bundle.

    Every mutating method locks the language, authenticates the current full
    snapshot, builds and self-opens a staged immutable generation, and only
    then atomically replaces ``current``. Existing loaded lexicons keep their
    old snapshot until explicitly reloaded.
    """

    __slots__ = ("bundle", "iso639_3")

    def __init__(self, bundle: Pathish, iso639_3: str) -> None:
        """Select one independent root and ISO language without writing it."""
        self.bundle = Path(os.fspath(bundle))
        self.iso639_3 = iso639_3

    def init(
        self,
        *,
        actor: str | None = None,
        reason: str,
    ) -> UserDictionaryUpdate:
        """Create one authenticated empty snapshot, idempotently."""
        return _update(
            native_init(
                os.fspath(self.bundle),
                self.iso639_3,
                _actor(actor),
                reason,
            )
        )

    def set(  # noqa: PLR0913 - explicit audit fields are the public contract.
        self,
        form: str,
        lemma: str,
        *,
        match: MatchMode = "exact",
        replace: bool = False,
        actor: str | None = None,
        reason: str,
    ) -> UserDictionaryUpdate:
        """Add an absent key or explicitly replace its value."""
        return _update(
            native_set(
                os.fspath(self.bundle),
                self.iso639_3,
                form,
                lemma,
                match,
                replace,
                _actor(actor),
                reason,
            )
        )

    def remove(
        self,
        form: str,
        *,
        match: MatchMode = "exact",
        actor: str | None = None,
        reason: str,
    ) -> UserDictionaryUpdate:
        """Remove one override; a missing key is a no-op."""
        return _update(
            native_remove(
                os.fspath(self.bundle),
                self.iso639_3,
                form,
                match,
                _actor(actor),
                reason,
            )
        )

    def import_file(
        self,
        path: Pathish,
        *,
        replace: bool = False,
        actor: str | None = None,
        reason: str,
    ) -> UserDictionaryUpdate:
        """Merge a strict headerless ``form<TAB>lemma<TAB>mode`` snapshot."""
        return _update(
            native_import(
                os.fspath(self.bundle),
                self.iso639_3,
                os.fspath(path),
                replace,
                _actor(actor),
                reason,
            )
        )

    def entries(self) -> tuple[UserEntry, ...]:
        """List the active canonical snapshot in stable key order."""
        return tuple(
            UserEntry(form, lemma, match)
            for form, lemma, match in native_entries(
                os.fspath(self.bundle), self.iso639_3
            )
        )

    def history(self) -> tuple[Mapping[str, object], ...]:
        """Return authenticated receipts with ancestors before their descendants."""
        return tuple(native_history(os.fspath(self.bundle), self.iso639_3))

    def rollback(
        self,
        generation: str,
        *,
        actor: str | None = None,
        reason: str,
    ) -> UserDictionaryUpdate:
        """Copy an old snapshot into a new auditable generation."""
        return _update(
            native_rollback(
                os.fspath(self.bundle),
                self.iso639_3,
                generation,
                _actor(actor),
                reason,
            )
        )


def verify_user_bundle(bundle: Pathish) -> UserBundleVerification:
    """Authenticate every active and historical generation in ``bundle``."""
    return UserBundleVerification(*native_verify(os.fspath(bundle)))


__all__ = [
    "MatchMode",
    "UserBundleVerification",
    "UserDictionary",
    "UserDictionaryUpdate",
    "UserEntry",
    "verify_user_bundle",
]
