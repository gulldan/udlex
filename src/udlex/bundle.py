"""The portable on-disk lexicon format, read from Python.

The crate writes and owns the format (``rust/src/builder.rs``); this module
reads ``meta.json`` (:func:`read_meta`), reports whether builder sources remain
(:func:`has_sources`) and asks the crate's compatibility rule
(:func:`format_accepted`, :func:`format_refusal`).

A built lexicon publishes immutable generations under
``<bundle>/<lexicon>/.generations/<id>/`` and atomically switches ``current``.
Builder sources of a generation:

===============  ============================================================
``strings.txt``  One string per line; the line number is its identifier.
``exact.tsv``    ``form <TAB> string id``, sorted by form.
``lower.tsv``    ``lowercased form <TAB> string id``, sorted by form.
``external.tsv`` Same shape, merged from offline dictionaries and analysers.
``entities.tsv`` Isolated named-entity surface forms and safe inflections.
``phrases.tsv``  ``span key <TAB> component-wise suffix rewrite``, sorted.
``rules.tsv``    ``bucket + suffix <TAB> packed rule``, sorted by key.
``meta.json``    Provenance and table sizes.
===============  ============================================================

``udlex.compile_bundle`` encodes each lookup table in its smallest codec
(``.da``, ``.map``, ``.inline``, or ``.hybrid`` for ``exact``/``lower``),
records the choice under ``codecs`` and writes ``strings.idx`` (``u32``
offsets for memory-mapping ``strings.txt``). ``phrases.inline`` and the rule
files have fixed encodings. Entity exact/lower keys share one table under
disjoint namespaces (``entity_encoding = "namespaced-v1"``).

A build writes one format number, :data:`FORMAT_VERSION`; a reader accepts
:data:`FORMAT_RANGE`. Neither is the core's or a dictionary wheel's version.

A rule is packed into one unsigned 64-bit integer:

======  ====================================================================
56..63  characters to cut from the end of the lowercased form
48..55  case action, or 255 for the stop marker
0..47   identifier of the string to append
======  ====================================================================

Confidence is not stored: a rule below its bucket's threshold is written as a
stop marker, not deleted, so the reader gives up instead of falling back to a
vaguer rule.
"""

from __future__ import annotations

import math
import stat
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from udlex._io import (
    _is_reparse_point,
    is_sha256_hex,
    read_bounded_regular,
    strict_json_loads,
    valid_generation_name,
)

# Format constants and the compatibility rule come from the crate, never copied.
from udlex._udlex import (
    ENTITY_ENCODING,
    FORMAT_RANGE,
    FORMAT_VERSION,
    MIN_CORE_RELEASE,
    MIN_FORMAT_VERSION,
    NORMALIZATION,
    PHRASE_CODEC,
    SOURCES,
    format_remedy,
    format_supported,
)
from udlex.language_codes import valid_iso

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path
    from types import UnionType

__all__ = [
    "ENTITY_ENCODING",
    "FORMAT_RANGE",
    "FORMAT_VERSION",
    "LOOKUP_TABLES",
    "MIN_CORE_RELEASE",
    "MIN_FORMAT_VERSION",
    "PHRASE_CODEC",
    "SOURCES",
    "BundleFormatError",
    "BundleMetadataError",
    "LexiconMeta",
    "active_directory",
    "format_accepted",
    "format_refusal",
    "has_sources",
    "lookup_rows",
    "read_meta",
]

#: Widest value the crate's format rule can be asked about.
_U32_MAX = 0xFFFF_FFFF


def _declared_format(declared: object) -> int | None:
    """The declared version as a number the crate's rule can be asked about."""
    if isinstance(declared, bool) or not isinstance(declared, int):
        return None
    return declared if 0 <= declared <= _U32_MAX else None


def format_accepted(declared: object) -> bool:
    """Whether this build reads a document declaring ``declared``.

    Asks the crate's rule; anything that is not a format number is refused.
    """
    version = _declared_format(declared)
    return version is not None and format_supported(version)


def format_refusal(source: object, declared: object) -> str:
    """One refusal text naming the declared format, the readable range and the fix.

    A document without a usable version is treated as too old.
    """
    version = _declared_format(declared)
    remedy = format_remedy(version if version is not None else 0)
    return (
        f"{source}: unsupported bundle format {declared!r}, "
        f"this udlex reads format {FORMAT_RANGE}; {remedy}"
    )


#: Tables whose rows are form-to-lemma pairs (not ``rules`` or ``strings``).
LOOKUP_TABLES = ("exact", "lower", "external", "entities")


def lookup_rows(entries: Mapping[str, int]) -> int:
    """How many form-to-lemma rows a lexicon's lookup tables hold.

    The one definition of "entries" every command reports.
    """
    return sum(entries.get(name, 0) for name in LOOKUP_TABLES)


class BundleMetadataError(ValueError):
    """A bundle metadata document is malformed or internally inconsistent."""


class BundleFormatError(BundleMetadataError):
    """A bundle was written in an incompatible on-disk format."""


@dataclass(frozen=True, slots=True)
class LexiconMeta:
    """Provenance of a bundled lexicon.

    Attributes:
        name: The lexicon identifier, normally a UD treebank code.
        language: Human-readable language name.
        iso639_3: ISO 639-3 code of the language.
        source: What the lexicon was learned from.
        training_tokens: How many annotated tokens went into it.
        variant: Which parts of the learned lexicon this bundle ships, one of
            the acceptance lattice: ``full``, ``lookup``
            (no suffix rules), ``rules`` (no external vocabulary), ``corpus``
            (neither), or ``identity`` (no common-word stages, because nothing
            beat copying the surface form). An isolated entity table may remain
            available to explicitly entity-marked calls.
        selected_on: ``dev`` when a development split chose the variant,
            ``train-cv`` when deterministic form-disjoint training holdout did,
            or ``None`` when no selection split was available. Held-out test
            data is score-only and is never valid active lexicon metadata.
        max_suffix: Longest suffix indexed, mirrored here so a consumer need
            not hard-code it.
        format_version: Version of this on-disk layout.
        entity_encoding: Exact encoding contract for the shared exact/lower
            entity table.
        entries: Number of rows per table.
        traffic: Share of running-text tokens that reaches the ``lower`` and
            ``external`` probes, measured on text the tables were not built
            from. The compiler weighs an encoding's byte savings against it;
            ``exact`` is the first probe, so its share is 1 by construction
            and is not recorded.
        gold_tokens: How many gold tokens the lexicon was scored on, when a
            test split existed.
        accuracy: Exact-match accuracy on those tokens, in ``[0, 1]``.
        baseline: Accuracy of copying the surface form on the same tokens.
        beats_baseline: Whether the lexicon is at least as good as copying the
            surface form. ``False`` marks a lexicon measurably worse than
            copying — in practice one built from external paradigms whose lemma
            convention disagrees with the corpus; a consumer should prefer the
            surface form over such a lexicon.
    """

    name: str
    language: str
    iso639_3: str
    source: str
    training_tokens: int
    variant: str = "full"
    selected_on: str | None = None
    max_suffix: int = 6
    format_version: int = FORMAT_VERSION
    entity_encoding: str = ENTITY_ENCODING
    entries: Mapping[str, int] = field(default_factory=dict)
    traffic: Mapping[str, float] = field(default_factory=dict)
    gold_tokens: int | None = None
    accuracy: float | None = None
    baseline: float | None = None
    beats_baseline: bool | None = None


def active_directory(source: Path) -> Path:
    """Resolve a lexicon root to the immutable generation it currently names."""
    return _resolve_active(source)[0]


def _real_directory(path: Path, description: str) -> None:
    """Require a concrete directory rather than a followed symlink."""
    value = path.lstat()
    reparse = _is_reparse_point(value)
    if not stat.S_ISDIR(value.st_mode) or stat.S_ISLNK(value.st_mode) or reparse:
        msg = f"{path}: {description} must be a real directory"
        raise BundleMetadataError(msg)


def _resolve_active(source: Path) -> tuple[Path, str | None]:
    """Resolve ``current`` with the same fail-closed rules as the Rust reader."""
    _real_directory(source, "lexicon root")
    current = source / "current"
    try:
        payload = read_bounded_regular(
            current,
            129,
            description="generation pointer",
        )
    except FileNotFoundError:
        return source, None
    except OSError as error:
        msg = f"{current}: current must be a regular text file no larger than 129 bytes"
        raise BundleMetadataError(msg) from error
    try:
        raw = payload.decode("utf-8")
    except UnicodeDecodeError:
        msg = f"{current}: generation identifier must be UTF-8"
        raise BundleMetadataError(msg) from None
    generation = raw.rstrip("\r\n")
    if not valid_generation_name(generation) or raw.strip() != generation:
        msg = f"{current}: invalid generation identifier {raw!r}"
        raise BundleMetadataError(msg)
    generations = source / ".generations"
    candidate = source / ".generations" / generation
    _real_directory(generations, "generations path")
    _real_directory(candidate, "generation path")
    return candidate, generation


def has_sources(source: Path) -> bool:
    """Whether a lexicon directory still holds its builder sources.

    ``False`` for every installed (pruned) lexicon.
    """
    active = active_directory(source)
    return all((active / name).exists() for name in SOURCES)


def read_meta(source: Path) -> LexiconMeta:  # noqa: C901, PLR0912, PLR0915
    """Load just the metadata of a bundled lexicon, without its tables.

    Args:
        source: A lexicon directory.

    Returns:
        The metadata.

    Raises:
        ValueError: If the bundle declares a format outside the range this
            build reads, or its metadata violates the format.
    """
    active, current_generation = _resolve_active(source)
    path = active / "meta.json"
    try:
        metadata = read_bounded_regular(
            path,
            4 * 1024 * 1024,
            description="bundle metadata",
        )
        loaded = strict_json_loads(metadata, path)
    except (OSError, ValueError) as error:
        raise BundleMetadataError(str(error)) from None
    if not isinstance(loaded, dict):
        msg = f"{path}: metadata must be a JSON object"
        raise BundleMetadataError(msg)
    payload = cast("dict[str, object]", loaded)
    if not format_accepted(payload.get("format_version")):
        raise BundleFormatError(format_refusal(source, payload.get("format_version")))
    if payload.get("entity_encoding") != ENTITY_ENCODING:
        msg = (
            f"{path}: field 'entity_encoding' must be {ENTITY_ENCODING!r}; "
            "legacy format-3 entity tables are ambiguous and must be rebuilt"
        )
        raise BundleMetadataError(msg)
    required_strings = ("name", "language", "iso639_3", "source", "variant")
    variant_value = payload.get("variant", "full")
    for field_name in required_strings:
        value = variant_value if field_name == "variant" else payload.get(field_name)
        if not isinstance(value, str) or not value:
            msg = f"{path}: field {field_name!r} must be a non-empty string"
            raise BundleMetadataError(msg)
    iso = cast("str", payload["iso639_3"])
    if not valid_iso(iso):
        msg = f"{path}: field 'iso639_3' must contain three lowercase ASCII letters"
        raise BundleMetadataError(msg)
    if payload.get("normalization") != NORMALIZATION:
        msg = f"{path}: field 'normalization' must be {NORMALIZATION!r}"
        raise BundleMetadataError(msg)
    generation = payload.get("generation")
    if not valid_generation_name(generation):
        msg = f"{path}: field 'generation' is invalid"
        raise BundleMetadataError(msg)
    if current_generation is not None and generation != current_generation:
        msg = f"{path}: metadata generation does not match current"
        raise BundleMetadataError(msg)
    variant_value = cast("str", variant_value)
    if variant_value not in {"identity", "corpus", "rules", "lookup", "full"}:
        msg = f"{path}: unknown variant {variant_value!r}"
        raise BundleMetadataError(msg)
    required_integers = ("training_tokens", "max_suffix")
    for field_name in required_integers:
        value = payload.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            msg = f"{path}: field {field_name!r} must be a non-negative integer"
            raise BundleMetadataError(msg)
        if value > 2**64 - 1:
            msg = f"{path}: field {field_name!r} exceeds u64"
            raise BundleMetadataError(msg)
    max_suffix = cast("int", payload["max_suffix"])
    if max_suffix > 6:
        msg = f"{path}: max_suffix exceeds supported limit 6"
        raise BundleMetadataError(msg)
    entries = _entry_mapping(path, payload.get("entries"))
    if lookup_rows(entries) > 2**64 - 1:
        msg = f"{path}: aggregate lookup entry count exceeds u64"
        raise BundleMetadataError(msg)
    traffic = _traffic_mapping(path, payload.get("traffic", {}))
    files = payload.get("files", {})
    codecs = _codec_mapping(
        path, payload.get("codecs", {}), entries, compiled=bool(files)
    )
    _file_mapping(path, files, entries, codecs)
    counts = lambda name: entries.get(name, 0)  # noqa: E731
    variant = variant_value
    variant_valid = {
        "identity": all(
            counts(name) == 0 for name in ("exact", "lower", "external", "rules")
        ),
        "corpus": counts("external") == 0 and counts("rules") == 0,
        "rules": counts("external") == 0,
        "lookup": counts("rules") == 0,
        "full": True,
    }[variant]
    if not variant_valid:
        msg = f"{path}: entries contradict variant {variant!r}"
        raise BundleMetadataError(msg)
    selected_on = payload.get("selected_on")
    if selected_on not in {None, "dev", "train-cv"}:
        msg = f"{path}: field 'selected_on' must be 'dev', 'train-cv', or null"
        raise BundleMetadataError(msg)

    def nullable_number(name: str, kinds: type | UnionType) -> float | None:
        value = payload.get(name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, kinds)
        ):
            msg = f"{path}: field {name!r} must be a number or null"
            raise BundleMetadataError(msg)
        return cast("float | None", value)

    gold_tokens = cast("int | None", nullable_number("gold_tokens", int))
    if gold_tokens is not None and not 0 < gold_tokens <= 2**64 - 1:
        msg = f"{path}: field 'gold_tokens' must be a positive u64"
        raise BundleMetadataError(msg)
    shares: dict[str, float | None] = {}
    for name in ("accuracy", "baseline"):
        value = nullable_number(name, int | float)
        if value is not None and (
            not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
        ):
            msg = f"{path}: field {name!r} must be between 0 and 1"
            raise BundleMetadataError(msg)
        shares[name] = None if value is None else float(value)
    present = [
        payload.get(name) is not None
        for name in ("gold_tokens", "accuracy", "baseline")
    ]
    if len(set(present)) != 1:
        msg = f"{path}: gold_tokens, accuracy and baseline must be present together"
        raise BundleMetadataError(msg)
    beats_baseline = payload.get("beats_baseline")
    if beats_baseline is not None and not isinstance(beats_baseline, bool):
        msg = f"{path}: field 'beats_baseline' must be a boolean or null"
        raise BundleMetadataError(msg)
    measurements_present = present[0]
    if (beats_baseline is not None) != measurements_present:
        msg = (
            f"{path}: beats_baseline must be present exactly with held-out measurements"
        )
        raise BundleMetadataError(msg)
    if measurements_present:
        accuracy = cast("float", shares["accuracy"])
        baseline = cast("float", shares["baseline"])
        if cast("bool", beats_baseline) != (accuracy >= baseline):
            msg = f"{path}: beats_baseline contradicts accuracy and baseline"
            raise BundleMetadataError(msg)
    return LexiconMeta(
        name=cast("str", payload["name"]),
        language=cast("str", payload["language"]),
        iso639_3=iso,
        source=cast("str", payload["source"]),
        training_tokens=cast("int", payload["training_tokens"]),
        variant=variant,
        selected_on=cast("str | None", selected_on),
        max_suffix=max_suffix,
        format_version=cast("int", payload["format_version"]),
        entity_encoding=cast("str", payload["entity_encoding"]),
        entries={key: int(value) for key, value in entries.items()},
        traffic={key: float(value) for key, value in traffic.items()},
        gold_tokens=gold_tokens,
        accuracy=shares["accuracy"],
        baseline=shares["baseline"],
        beats_baseline=beats_baseline,
    )


def _entry_mapping(path: Path, value: object) -> dict[str, int]:
    """Validate the exact table-count schema owned by the Rust reader."""
    if not isinstance(value, dict):
        msg = f"{path}: field 'entries' must be an object"
        raise BundleMetadataError(msg)
    expected = {
        "entities",
        "exact",
        "external",
        "lower",
        "phrases",
        "rules",
        "strings",
    }
    if set(value) != expected:
        msg = f"{path}: entries must contain exactly {sorted(expected)}"
        raise BundleMetadataError(msg)
    result: dict[str, int] = {}
    for key, number in value.items():
        if (
            not isinstance(key, str)
            or isinstance(number, bool)
            or not isinstance(number, int)
            or not 0 <= number <= 2**64 - 1
        ):
            msg = f"{path}: entries.{key} must be a non-negative u64"
            raise BundleMetadataError(msg)
        result[key] = number
    return result


def _traffic_mapping(path: Path, value: object) -> dict[str, float]:
    """Validate finite held-out traffic shares."""
    if not isinstance(value, dict) or not set(value).issubset({"lower", "external"}):
        msg = f"{path}: traffic may contain only lower/external"
        raise BundleMetadataError(msg)
    result: dict[str, float] = {}
    for key, number in value.items():
        if (
            isinstance(number, bool)
            or not isinstance(number, int | float)
            or not math.isfinite(float(number))
            or not 0.0 <= float(number) <= 1.0
        ):
            msg = f"{path}: invalid traffic entry {key!r}"
            raise BundleMetadataError(msg)
        result[str(key)] = float(number)
    return result


def _codec_mapping(
    path: Path,
    value: object,
    entries: Mapping[str, int],
    *,
    compiled: bool,
) -> dict[str, str]:
    """Validate table codec names and their count agreement."""
    if not isinstance(value, dict):
        msg = f"{path}: codecs must be an object"
        raise BundleMetadataError(msg)
    codecs: dict[str, str] = {}
    for name, codec in value.items():
        if name not in {"exact", "lower", "external", "entities"} or codec not in {
            "da",
            "map",
            "inline",
            "hybrid",
        }:
            msg = f"{path}: invalid codec {name!r}: {codec!r}"
            raise BundleMetadataError(msg)
        codecs[str(name)] = str(codec)
    if compiled and any(
        (entries[name] > 0) != (name in codecs)
        for name in ("exact", "lower", "external", "entities")
    ):
        msg = f"{path}: codec presence disagrees with table counts"
        raise BundleMetadataError(msg)
    return codecs


def _runtime_files(entries: Mapping[str, int], codecs: Mapping[str, str]) -> set[str]:
    """Name the runtime files these table counts imply, the reader's rule."""
    expected = {"strings.idx", "strings.txt"}
    for name in ("exact", "lower", "external", "entities"):
        if entries[name] > 0:
            expected.add(f"{name}.{codecs[name]}")
    # `phrases` and `rules` have one fixed encoding each.
    if entries["phrases"] > 0:
        expected.add(f"phrases.{PHRASE_CODEC}")
    if entries["rules"] > 0:
        expected.update({"rules.da", "rules.values"})
    return expected


def _file_mapping(
    path: Path,
    value: object,
    entries: Mapping[str, int],
    codecs: Mapping[str, str],
) -> None:
    """Validate the immutable runtime-file integrity manifest schema."""
    if not isinstance(value, dict):
        msg = f"{path}: files must be an object"
        raise BundleMetadataError(msg)
    if not value:
        return
    if set(value) != _runtime_files(entries, codecs):
        msg = f"{path}: declared runtime files differ from table metadata"
        raise BundleMetadataError(msg)
    for name, declaration in value.items():
        if not isinstance(declaration, dict) or set(declaration) != {"size", "sha256"}:
            msg = f"{path}: files.{name} must contain exactly size/sha256"
            raise BundleMetadataError(msg)
        size = declaration.get("size")
        digest = declaration.get("sha256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or not 0 <= size <= 2**64 - 1
            or not is_sha256_hex(digest)
        ):
            msg = f"{path}: files.{name} has invalid integrity metadata"
            raise BundleMetadataError(msg)
