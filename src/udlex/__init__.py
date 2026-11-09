"""Dictionary lemmatization: Rust lookup and training, Python discovery and CLI."""

import os
from importlib.machinery import PathFinder
from os import PathLike
from pathlib import Path
from typing import TypeAlias, TypedDict, Unpack

from udlex._provenance import Source
from udlex._udlex import (
    ENTITY_ENCODING,
    FORMAT_RANGE,
    FORMAT_VERSION,
    MIN_FORMAT_VERSION,
    NORMALIZATION,
    PHRASE_CODEC,
    SOURCES,
    USER_DICTIONARY_SCHEMA,
    Compiled,
    Language,
    LemmaInfo,
    Lexicon,
    Outcome,
    Tables,
    __version__,
    build,
    build_lexicon,
    choose,
    score,
    tune,
)
from udlex._udlex import (
    compile_bundle as _compile_bundle,
)
from udlex._udlex import (
    languages as _languages,
)
from udlex._udlex import (
    load as _load,
)
from udlex._udlex import (
    prune_bundle as _prune_bundle,
)
from udlex.catalog import is_manifested as _is_manifested
from udlex.catalog import read as _read_catalog
from udlex.catalog import resolve as _resolve_language
from udlex.catalog import suggest as _suggest
from udlex.user_dictionary import (
    UserBundleVerification,
    UserDictionary,
    UserDictionaryUpdate,
    UserEntry,
    verify_user_bundle,
)

__all__ = [
    "ENTITY_ENCODING",
    "FORMAT_RANGE",
    "FORMAT_VERSION",
    "MIN_FORMAT_VERSION",
    "NORMALIZATION",
    "PHRASE_CODEC",
    "SOURCES",
    "USER_BUNDLE_VARIABLE",
    "USER_DICTIONARY_SCHEMA",
    "Compiled",
    "Language",
    "LemmaInfo",
    "Lexicon",
    "Outcome",
    "Probes",
    "Source",
    "Tables",
    "UserBundleVerification",
    "UserDictionary",
    "UserDictionaryUpdate",
    "UserEntry",
    "__version__",
    "build",
    "build_lexicon",
    "bundle_path",
    "bundle_paths",
    "choose",
    "compile_bundle",
    "languages",
    "load",
    "prune_bundle",
    "score",
    "supported_languages",
    "tune",
    "verify_user_bundle",
]

DEFAULT_BUNDLE = Path("bundle")
"""Local fallback when neither environment nor dictionary wheels select data."""

BUNDLE_VARIABLE = "UDLEX_BUNDLE"
USER_BUNDLE_VARIABLE = "UDLEX_USER_BUNDLE"
StrPath: TypeAlias = str | PathLike[str]


def bundle_paths() -> tuple[Path, ...]:
    """Return every installed bundle location in lookup order.

    :envvar:`UDLEX_BUNDLE` wins; otherwise every portion of the ``udlex_bundle``
    namespace; otherwise ``./bundle``.
    """
    if configured := os.environ.get(BUNDLE_VARIABLE):
        return (Path(configured),)
    # Read only PEP 420 namespace locations, without importing package code.
    spec = PathFinder.find_spec("udlex_bundle")
    if spec is None or spec.loader is not None or spec.origin is not None:
        return (DEFAULT_BUNDLE,)
    paths = spec.submodule_search_locations or ()
    return tuple(dict.fromkeys(Path(path) for path in paths)) or (DEFAULT_BUNDLE,)


def _installed_catalog() -> tuple[tuple[Language, Path], ...]:
    """Index every installed namespace portion without choosing a winner."""
    found: list[tuple[Language, Path]] = []
    by_iso: dict[str, tuple[Language, Path]] = {}
    by_config: dict[str, tuple[Language, Path]] = {}
    for directory in bundle_paths():
        for language in _read_catalog(directory):
            if previous := by_iso.get(language.iso639_3):
                other, origin = previous
                message = (
                    f"duplicate ISO 639-3 code {language.iso639_3!r} in "
                    f"{origin / other.lexicon} and {directory / language.lexicon}"
                )
                raise ValueError(message)
            if previous := by_config.get(language.lexicon):
                other, origin = previous
                message = (
                    f"duplicate config {language.lexicon!r} in "
                    f"{origin / other.lexicon} ({other.iso639_3}) and "
                    f"{directory / language.lexicon} ({language.iso639_3})"
                )
                raise ValueError(message)
            entry = (language, directory)
            found.append(entry)
            by_iso[language.iso639_3] = entry
            by_config[language.lexicon] = entry
    return tuple(sorted(found, key=lambda entry: entry[0].iso639_3))


def _installed_entry(name: str) -> tuple[Language, Path]:
    """Resolve one installed ISO/config/alias after validating all portions."""
    installed = _installed_catalog()
    direct = [
        entry for entry in installed if name in {entry[0].iso639_3, entry[0].lexicon}
    ]
    if direct:
        return direct[0]

    aliases: list[tuple[Language, Path]] = []
    for directory in dict.fromkeys(origin for _, origin in installed):
        resolved = _resolve_language(name, directory)
        aliases.extend(
            entry
            for entry in installed
            if entry[1] == directory and entry[0].iso639_3 == resolved
        )
    if len(aliases) == 1:
        return aliases[0]
    if aliases:
        choices = ", ".join(
            f"{language.iso639_3} in {origin}" for language, origin in aliases
        )
        message = f"ambiguous installed language alias {name!r}: {choices}"
        raise ValueError(message)
    raise ValueError(_not_installed(name, [language for language, _ in installed]))


def _not_installed(name: str, listed: list[Language]) -> str:
    """Explain a miss: what was meant, where it was looked for, how to get it."""
    hint = _suggest(name, listed)
    meant = f" (did you mean {hint!r}?)" if hint and hint != name else ""
    available = ", ".join(language.iso639_3 for language in listed) or "none"
    searched = ", ".join(os.fspath(path.absolute()) for path in bundle_paths())
    configured = os.environ.get(BUNDLE_VARIABLE)
    origin = (
        f"{BUNDLE_VARIABLE}={configured}"
        if configured
        else f"{BUNDLE_VARIABLE} is unset"
    )
    return (
        f"no installed lexicon for {name!r}{meant}. Available: {available}. "
        f"Searched {searched} ({origin}). Please pass a bundle directory or set "
        f"{BUNDLE_VARIABLE}."
    )


def bundle_path(language: str | None = None) -> Path:
    """Return the one installed directory that can answer a request.

    The namespace may span several directories (PEP 420); without a language a
    path is returned only when there is exactly one.

    Args:
        language: Installed ISO code, config, or compatibility alias. Omit only
            when the namespace has one physical directory.

    Returns:
        The unique directory containing ``language``.

    Raises:
        ValueError: If a split namespace has no language selector, contains a
            duplicate ISO/config, or cannot resolve the requested language.
    """
    if language is not None:
        return _installed_entry(language)[1]
    paths = bundle_paths()
    if len(paths) == 1:
        return paths[0]
    locations = ", ".join(os.fspath(path) for path in paths)
    message = (
        "installed dictionary namespace spans multiple directories; "
        f"select a language with bundle_path(language): {locations}"
    )
    raise ValueError(message)


def languages(directory: StrPath) -> list[Language]:
    """List one bundle, accepting every standard path-like object."""
    return _languages(os.fspath(directory))


class Probes(TypedDict, total=False):
    """Override the shipped lookup profile for ablations and acceptance runs."""

    use_exact: bool
    use_lower: bool
    use_exact_lower: bool
    use_external_lower: bool
    use_entities: bool
    use_user: bool
    protected: bool
    span_case: str | None
    span_multiword: str | None
    phrase_patterns: list[tuple[str, str]] | None
    token_case: str | None


def load(  # noqa: PLR0913
    bundle_or_language: StrPath,
    language: str | None = None,
    /,
    *,
    user_bundle: StrPath | None = None,
    use_rules: bool = True,
    use_external: bool = True,
    use_entities_lower: bool | None = None,
    **advanced: Unpack[Probes],
) -> Lexicon:
    """Open an installed language, or one from an explicit bundle directory.

    ``load("rus")`` uses :func:`bundle_paths` and rejects duplicate ISO/config
    identities. ``load(directory, "rus")`` opens one caller-selected bundle.

    Args:
        bundle_or_language: An ISO 639-3 code such as ``"rus"`` (or a treebank
            name such as ``"ru_taiga"``) when ``language`` is omitted;
            otherwise the bundle directory.
        language: The language inside an explicit bundle directory.
        user_bundle: Manual-override root; defaults to
            :envvar:`UDLEX_USER_BUNDLE` when set.
        use_rules: Enable suffix guesses for unknown words.
        use_external: Enable exact and lowercase external-dictionary lookup.
        use_entities_lower: Retry entity lookup with language-aware lowercase.
            ``None`` uses the span policy (enabled for ``ces`` and ``rus``)
            and disables retry for entity tokens; a bool sets both profiles.
        **advanced: :class:`Probes` for ablations and acceptance runs.

    Returns:
        The loaded lexicon; load it once per process and share it.

    Raises:
        ValueError: If no installed dictionary has the language; the message
            says what was searched and how to supply a bundle.
        FileNotFoundError: If an explicit bundle directory does not exist.
        OSError: If the bundle has no such language or it cannot be opened.
    """
    if language is None:
        requested = os.fspath(bundle_or_language)
        if os.sep in requested or (os.altsep and os.altsep in requested):
            message = (
                f"{requested!r} is a path, not a language: call "
                f"udlex.load({requested!r}, '<iso639-3>') to open a bundle directory"
            )
            raise ValueError(message)
        installed, directory_path = _installed_entry(requested)
        resolved_language = installed.iso639_3
    else:
        directory_path = Path(bundle_or_language)
        if not directory_path.is_dir():
            swapped = Path(language).is_dir()
            message = f"bundle directory {directory_path.absolute()} does not exist" + (
                f"; the arguments look swapped — call "
                f"udlex.load({language!r}, {os.fspath(bundle_or_language)!r})"
                if swapped
                else ""
            )
            raise FileNotFoundError(message)
        resolved_language = _resolve_language(language, directory_path)
    resolved_user_bundle = user_bundle
    if resolved_user_bundle is None and advanced.get("use_user", True):
        resolved_user_bundle = os.environ.get(USER_BUNDLE_VARIABLE)
        if resolved_user_bundle is not None and not Path(resolved_user_bundle).is_dir():
            message = (
                f"{USER_BUNDLE_VARIABLE}={resolved_user_bundle}: no such directory; "
                f"unset it or create the user bundle with udlex.UserDictionary"
            )
            raise FileNotFoundError(message)
    if not use_external:
        advanced.setdefault("use_external_lower", False)
    try:
        return _load(
            os.fspath(directory_path),
            resolved_language,
            use_rules=use_rules,
            use_external=use_external,
            use_entities_lower=use_entities_lower,
            user_bundle=(
                None
                if resolved_user_bundle is None
                else os.fspath(resolved_user_bundle)
            ),
            **advanced,
        )
    except OSError as error:
        if language is not None:
            listed = list(_read_catalog(directory_path))
            if not any(
                resolved_language in {entry.iso639_3, entry.lexicon} for entry in listed
            ):
                hint = _suggest(language, listed)
                meant = f" (did you mean {hint!r}?)" if hint else ""
                available = ", ".join(entry.iso639_3 for entry in listed) or "none"
                message = (
                    f"{directory_path.absolute()} has no lexicon for "
                    f"{language!r}{meant}; available: {available}"
                )
                raise OSError(message) from error
        raise


def compile_bundle(directory: StrPath) -> Compiled:
    """Compile a path-like lexicon directory."""
    return _compile_bundle(os.fspath(directory))


def prune_bundle(directory: StrPath) -> bool:
    """Atomically republish a compiled lexicon without builder source TSVs.

    Refuses a lexicon of a manifest-backed bundle: the new generation would
    no longer match the manifest, and the whole bundle would stop loading.

    Args:
        directory: A lexicon directory, not a bundle directory.

    Returns:
        Whether anything was pruned. ``False`` means the lexicon already
        carried no builder sources.

    Raises:
        ValueError: If the lexicon belongs to a manifest-backed bundle.
        OSError: If the directory cannot be read or republished.
    """
    path = Path(os.fspath(directory))
    if _is_manifested(path.parent):
        msg = (
            f"{path}: refusing to prune a lexicon of a generation-bound published "
            "bundle; repackage the bundle so manifest generation rows stay atomic"
        )
        raise ValueError(msg)
    return _prune_bundle(os.fspath(directory))


def supported_languages(bundle: StrPath | None = None) -> tuple[Language, ...]:
    """List the languages a bundle can lemmatise, ordered by ISO 639-3 code.

    Reads metadata without opening lookup tables. Without a directory, merges
    installed namespace portions and rejects duplicate ISO codes or configs.

    Args:
        bundle: One explicit bundle directory.
            ``None`` means all installed namespace portions (or the configured
            :envvar:`UDLEX_BUNDLE`/local fallback).

    Returns:
        One entry per language. Manifest-backed V1 bundles fail closed on any
        declared row; an unmanifested scratch directory retains best-effort
        discovery. A directory that is not a bundle gives an empty catalogue.
    """
    if bundle is not None:
        return _read_catalog(Path(bundle))
    return tuple(language for language, _ in _installed_catalog())
