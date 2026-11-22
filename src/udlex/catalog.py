"""Read what a bundle offers, and find one language in it by ISO 639-3 code.

The extension reads the catalogue; this module turns names into lexicon
directories. A published manifest fails closed; only an unmanifested scratch
directory keeps best-effort discovery.
"""

from __future__ import annotations

import stat
from pathlib import Path
from typing import TYPE_CHECKING

from udlex._io import (
    _is_reparse_point,
    read_bounded_regular,
    recover_bundle_transactions,
    strict_json_loads,
)
from udlex._udlex import languages
from udlex.language_codes import LANGUAGE_REQUEST_ALIASES
from udlex.language_codes import valid_iso as _valid_iso

if TYPE_CHECKING:
    from collections.abc import Iterable

    from udlex._udlex import Language

__all__ = ["is_manifested", "locate", "read", "resolve", "suggest"]

MAX_MANIFEST_BYTES = 16 * 1024 * 1024


def read(directory: Path) -> tuple[Language, ...]:
    """List the languages a bundle can lemmatise, ordered by ISO 639-3 code.

    Args:
        directory: A bundle directory.

    Returns:
        One entry per language. A manifest-backed V1 bundle requires every
        declared row to be valid; an unmanifested scratch directory leaves an
        unreadable subdirectory out so the remaining local lexicons can still
        be inspected. An absent bundle is an empty catalogue.
    """
    if not directory.is_dir():
        return ()
    recover_bundle_transactions(directory)
    listed = tuple(languages(str(directory)))
    seen: dict[str, str] = {}
    for language in listed:
        if (previous := seen.get(language.iso639_3)) and previous != language.lexicon:
            message = (
                f"{directory}: duplicate ISO 639-3 code "
                f"{language.iso639_3!r} in {previous!r} and "
                f"{language.lexicon!r}"
            )
            raise ValueError(message)
        seen[language.iso639_3] = language.lexicon
    return listed


def locate(name: str, directory: Path) -> Path | None:
    """Find the lexicon directory a name refers to.

    Args:
        name: An ISO 639-3 code such as ``rus``, a treebank name such as
            ``ru_taiga``, or a path to a lexicon directory.
        directory: The bundle to search.

    Returns:
        The declared lexicon directory, an explicit external/scratch path, or
        ``None`` when a manifest-backed bundle declares no such language.
    """
    recover_bundle_transactions(directory)
    candidate = Path(name)
    normal_component = name == candidate.name and name not in {"", ".", ".."}
    explicit = not normal_component and candidate.is_dir()
    if explicit and not _inside(candidate, directory):
        return candidate  # independent of the selected bundle

    manifested = is_manifested(directory)
    listed = read(directory)
    manifested = manifested or is_manifested(directory)
    if not manifested and normal_component and (directory / name).is_dir():
        # Scratch bundle: direct config lookup keeps the loader's diagnostics.
        return directory / name
    if found := _locate_in(name, candidate, directory, listed, explicit=explicit):
        return found

    # Only a scratch directory accepts an undeclared path.
    if not manifested and candidate.is_dir():
        return candidate
    return None


def _locate_in(
    name: str,
    candidate: Path,
    directory: Path,
    listed: tuple[Language, ...],
    *,
    explicit: bool,
) -> Path | None:
    """Find a name/path only among authenticated catalogue rows."""
    if explicit:
        resolved_candidate = candidate.resolve()
        for language in listed:
            declared = directory / language.lexicon
            if declared.resolve() == resolved_candidate:
                return declared
    for language in listed:
        if name in {language.iso639_3, language.lexicon}:
            return directory / language.lexicon
    resolved = _resolve_from(name, directory, listed)
    for language in listed:
        if language.iso639_3 == resolved:
            return directory / language.lexicon
    return None


def resolve(name: str, directory: Path) -> str:
    """Resolve a compatibility alias; a real ISO/config in the bundle always wins."""
    listed = read(directory)
    return _resolve_from(name, directory, listed)


def suggest(name: str, listed: Iterable[Language]) -> str | None:
    """Name the one listed language a mistyped request most plausibly meant.

    Matches case-insensitively on the ISO 639-3 code, the config, the English
    name and the config's ISO 639-1 prefix (``ru`` of ``ru_taiga``); only a
    unique match is offered, so a hint never guesses between two languages.
    """
    key = name.casefold()
    matches = {
        language.iso639_3
        for language in listed
        if key
        in {
            language.iso639_3,
            language.lexicon,
            language.name.casefold(),
            language.lexicon.partition("_")[0],
        }
    }
    return matches.pop() if len(matches) == 1 else None


def _resolve_from(name: str, directory: Path, listed: tuple[Language, ...]) -> str:
    """Resolve ``name`` against one already authenticated catalogue."""
    if any(name in {language.iso639_3, language.lexicon} for language in listed):
        return name
    aliases = dict(LANGUAGE_REQUEST_ALIASES)
    if is_manifested(directory):
        for manifest in _manifest_paths(directory):
            payload = strict_json_loads(
                read_bounded_regular(
                    manifest,
                    MAX_MANIFEST_BYTES,
                    description="bundle manifest",
                ),
                manifest,
            )
            if not isinstance(payload, dict):
                msg = f"{manifest}: manifest must be an object"
                raise TypeError(msg)
            raw_aliases = payload.get("language_aliases", {})
            if not isinstance(raw_aliases, dict):
                msg = f"{manifest}: language_aliases must be an object"
                raise TypeError(msg)
            for alias, target in raw_aliases.items():
                if not _valid_iso(alias) or not _valid_iso(target) or alias == target:
                    msg = f"{manifest}: invalid language alias {alias!r}->{target!r}"
                    raise ValueError(msg)
                previous = aliases.get(alias)
                if previous is not None and previous != target:
                    msg = (
                        f"{manifest}: conflicting language alias "
                        f"{alias!r}->{target!r}; already maps to {previous!r}"
                    )
                    raise ValueError(msg)
                aliases[alias] = target
    target = aliases.get(name)
    if target is None:
        return name
    return target if any(language.iso639_3 == target for language in listed) else name


def is_manifested(directory: Path) -> bool:
    """Whether ``directory`` has any manifest boundary, even a malformed one.

    Chooses strict published-bundle semantics; :func:`read` validates it.
    """
    for path in (directory / "manifest.json", directory / ".manifests"):
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        return True
    return False


def _manifest_paths(directory: Path) -> tuple[Path, ...]:  # noqa: C901
    """Return the root manifest or the strict set of installed-wheel manifests."""
    root = directory / "manifest.json"
    wheel_root = directory / ".manifests"
    try:
        root.lstat()
    except FileNotFoundError:
        has_root = False
    else:
        has_root = True
    try:
        wheel_meta = wheel_root.lstat()
    except FileNotFoundError:
        wheel_meta = None
    if has_root and wheel_meta is not None:
        msg = f"{directory}: root manifest and wheel manifests cannot coexist"
        raise ValueError(msg)
    if has_root:
        return (root,)
    if wheel_meta is None:
        return ()
    reparse = _is_reparse_point(wheel_meta)
    if (
        not stat.S_ISDIR(wheel_meta.st_mode)
        or stat.S_ISLNK(wheel_meta.st_mode)
        or reparse
    ):
        msg = f"{wheel_root}: wheel manifest container must be a real directory"
        raise ValueError(msg)
    paths: list[Path] = []
    for path in sorted(wheel_root.iterdir(), key=lambda item: item.name):
        metadata = path.lstat()
        name = path.name
        expected_name = (
            name.startswith("udlex_dict_")
            and name.endswith(".json")
            and _valid_iso(name.removeprefix("udlex_dict_").removesuffix(".json"))
        )
        if (
            not expected_name
            or not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
        ):
            msg = f"{path}: unexpected wheel manifest entry"
            raise ValueError(msg)
        paths.append(path)
    if not paths:
        msg = f"{wheel_root}: wheel manifest container is empty"
        raise ValueError(msg)
    return tuple(paths)


def _inside(candidate: Path, directory: Path) -> bool:
    """Whether an existing candidate resolves inside the selected bundle."""
    return candidate.resolve().is_relative_to(directory.resolve())
