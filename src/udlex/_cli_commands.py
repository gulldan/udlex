"""Command handlers for :mod:`udlex.cli`; no parsing, network only via uv or pip."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from udlex import BUNDLE_VARIABLE, _udlex, bundle, catalog
from udlex import supported_languages as _supported_languages
from udlex.bundle import lookup_rows
from udlex.language_codes import resolve_language_request, valid_iso

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Iterator

DICTIONARY_VERSION = "1.0.0"
"""Release of the per-language dictionary wheels that ``udlex install`` fetches."""

DICTIONARIES_VARIABLE = "UDLEX_DICTIONARIES"
"""Base URL or local directory holding the wheels, for mirrors and tests."""

RELEASE_URL = "https://github.com/gulldan/udlex/releases/download/v{version}"

NONCOMMERCIAL_DICTIONARIES = frozenset(
    {
        "arb",
        "bul",
        "ces",
        "che",
        "chu",
        "ctn",
        "dar",
        "ell",
        "eng",
        "est",
        "eus",
        "frm",
        "fro",
        "glg",
        "got",
        "grc",
        "gun",
        "hbo",
        "heb",
        "hin",
        "hun",
        "ita",
        "kbc",
        "lat",
        "lzh",
        "naq",
        "nld",
        "oci",
        "pol",
        "qpm",
        "qtd",
        "ruc",
        "sjo",
        "srd",
        "tam",
        "ukr",
        "urd",
        "xcl",
        "yrl",
    }
)
"""Languages whose wheel carries ``_nc``; pinned to the catalog by a test."""


def _missing_bundle(directory: Path) -> SystemExit:
    """Say where the bundle was looked for and how to point elsewhere."""
    return SystemExit(
        f"udlex: no bundle at {directory}\n"
        f"       set {BUNDLE_VARIABLE} or pass --bundle"
    )


def _resolve(name: str, directory: Path) -> Path:
    """Find an ISO code, treebank name, or explicit lexicon path."""
    found = catalog.locate(name, directory)
    if found is not None:
        return found
    if not directory.is_dir():
        raise _missing_bundle(directory)
    message = (
        f"udlex: no lexicon for {name!r} in {directory}\n"
        f"       run 'udlex languages' to see what there is"
    )
    raise SystemExit(message)


def _languages(args: argparse.Namespace) -> int:
    """List the available lexicons, optionally filtered."""
    if args.bundle is not None and not args.bundle.is_dir():
        raise _missing_bundle(args.bundle)
    listed = (
        _supported_languages() if args.bundle is None else catalog.read(args.bundle)
    )
    location = args.bundle if args.bundle is not None else "installed namespace"
    pattern = (args.pattern or "").casefold()
    shown = 0
    print(
        f"{'code':5} {'language':26} {'accuracy':>9} {'copy':>7} {'entries':>10}  mode"
    )
    for language in listed:
        haystack = f"{language.iso639_3} {language.name} {language.lexicon}".casefold()
        if pattern and pattern not in haystack:
            continue
        # Without a test score, show the selection sample's, starred.
        measured, copied, mark = language.accuracy, language.baseline, ""
        if measured is None and language.selection_accuracy is not None:
            measured, copied = language.selection_accuracy, language.selection_baseline
            mark = "*"
        accuracy = "—" if measured is None else f"{100 * measured:.1f}%{mark}"
        baseline = "—" if copied is None else f"{100 * copied:.1f}%{mark}"
        print(
            f"{language.iso639_3:5} {language.name[:26]:26} {accuracy:>9} "
            f"{baseline:>7} {language.entries:>10}  {language.variant}"
        )
        shown += 1
    print(f"\n{shown} lexicons in {location}", file=sys.stderr)
    print(
        "* on the sample that selected the variant (dev or train-cv)", file=sys.stderr
    )
    return 0


def dictionary_wheel(name: str) -> str:
    """Name the dictionary wheel for an ISO 639-3 code or request alias."""
    iso = resolve_language_request(name.casefold().removesuffix("-nc"))
    if not valid_iso(iso):
        message = f"udlex: {name!r} is not an ISO 639-3 code (e.g. eng, rus, fin)"
        raise SystemExit(message)
    suffix = "_nc" if iso in NONCOMMERCIAL_DICTIONARIES else ""
    return f"udlex_dict_{iso}{suffix}-{DICTIONARY_VERSION}-py3-none-any.whl"


def _installer() -> list[str]:
    """Return an install command bound to this interpreter: uv, else pip."""
    if uv := shutil.which("uv"):
        return [uv, "pip", "install", "--python", sys.executable]
    if importlib.util.find_spec("pip") is not None:
        return [sys.executable, "-m", "pip", "install"]
    message = (
        "udlex: neither uv nor pip is available for this interpreter\n"
        "       install uv (https://docs.astral.sh/uv/) or pip"
    )
    raise SystemExit(message)


def _in_uv_project() -> bool:
    """Tell whether the working directory belongs to a locked uv project."""
    here = Path.cwd()
    return any((folder / "uv.lock").is_file() for folder in (here, *here.parents))


def _as_url(target: str) -> str:
    """Spell a local wheel as the file URL a uv requirement needs."""
    return target if "://" in target else Path(target).resolve().as_uri()


def _install(args: argparse.Namespace) -> int:
    """Install dictionary wheels into the current interpreter."""
    source = os.environ.get(DICTIONARIES_VARIABLE) or RELEASE_URL.format(
        version=DICTIONARY_VERSION
    )
    wheels = [dictionary_wheel(name) for name in args.languages]
    if "://" in source:
        targets = [f"{source.rstrip('/')}/{wheel}" for wheel in wheels]
    else:
        targets = [str(Path(source) / wheel) for wheel in wheels]
    if args.dry_run:
        print("\n".join(targets))
        return 0
    command = [*_installer(), *targets]
    code = subprocess.run(command, check=False).returncode  # noqa: S603 - argv, no shell
    if code == 0 and _in_uv_project():
        # uv sync removes packages the lock does not list.
        adds = " ".join(
            f'"{wheel.split("-", 1)[0].replace("_", "-")} @ {_as_url(target)}"'
            for wheel, target in zip(wheels, targets, strict=True)
        )
        print(
            f"udlex: to keep it across 'uv sync', run: uv add {adds}", file=sys.stderr
        )
    return code


def _words(args: argparse.Namespace) -> Iterator[str]:
    """Yield tokens from arguments or standard input."""
    if args.words:
        yield from args.words
    else:
        for line in sys.stdin:
            token = line.strip()
            if token:
                yield token


def _open_lexicon(
    directory: Path,
    *,
    use_rules: bool,
    user_bundle: Path | None = None,
    use_user: bool = False,
) -> _udlex.Lexicon:
    """Open the compiled reader, with a diagnostic for source-only fixtures."""
    try:
        return _udlex.Lexicon(
            str(directory),
            use_rules=use_rules,
            user_bundle=None if user_bundle is None else str(user_bundle),
            use_user=use_user,
        )
    except OSError:
        if not bundle.has_sources(directory):
            raise
        message = (
            f"udlex: {directory} is built but not compiled\n"
            f"       run 'udlex compile {directory.name}' first"
        )
        raise SystemExit(message) from None


def _lemma(args: argparse.Namespace) -> int:
    """Lemmatise words given on the command line or piped in."""
    directory = _resolve(args.language, args.bundle)
    meta = bundle.read_meta(directory)
    lexicon = _open_lexicon(
        directory,
        use_rules=not args.lookup_only,
        user_bundle=args.user_bundle,
        use_user=args.use_user,
    )
    if not args.quiet:
        print(
            f"# {meta.name} ({meta.language}), variant {meta.variant}, "
            f"source {meta.source}",
            file=sys.stderr,
        )
    for index, word in enumerate(_words(args)):
        info = lexicon.lemma(
            word,
            sentence_initial=index == 0,
            protected=args.protected,
            entity=args.entity,
            info=True,
        )
        if args.json:
            print(
                json.dumps(
                    {"token": word, "lemma": info.lemma, "source": info.source.value},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        elif args.with_info:
            print(f"{info.lemma}\t{info.source.value}")
        else:
            print(info.lemma)
    return 0


def _compile(args: argparse.Namespace) -> int:  # noqa: C901, PLR0912, PLR0915
    """Compile source tables into the tries consumed by the reader."""
    if not args.bundle.is_dir():
        raise _missing_bundle(args.bundle)
    # Read the inventory even for an explicit request: a corrupt row must fail.
    listed = catalog.read(args.bundle)
    manifested = catalog.is_manifested(args.bundle)
    if manifested and args.prune:
        msg = (
            "udlex: refusing to prune a generation-bound published bundle; "
            "rebuild/package it so manifest generation rows stay atomic"
        )
        raise SystemExit(msg)
    candidates: list[tuple[Path, bundle.LexiconMeta]] = []
    if args.languages:
        selected: list[Path] = []
        seen: set[Path] = set()
        missing: list[str] = []
        for requested in args.languages:
            path = catalog.locate(requested, args.bundle)
            if path is None:
                missing.append(requested)
                continue
            identity = path.resolve()
            if identity not in seen:
                seen.add(identity)
                selected.append(path)
        if missing:
            msg = (
                f"udlex: no such lexicon in {args.bundle}: {', '.join(sorted(missing))}"
            )
            raise SystemExit(msg)
        paths = selected
    elif manifested:
        paths = [args.bundle / language.lexicon for language in listed]
    else:
        # An unmanifested scratch directory stays repairable by a direct scan.
        paths = sorted(path for path in args.bundle.iterdir() if path.is_dir())

    for path in paths:
        try:
            meta = bundle.read_meta(path)
        except FileNotFoundError:
            continue
        if manifested:
            try:
                _udlex.Lexicon(str(path))
            except OSError:
                if bundle.has_sources(path):
                    msg = (
                        "udlex: refusing to mutate an uncompiled generation-bound "
                        f"lexicon: {path}; rebuild the published bundle atomically"
                    )
                    raise SystemExit(msg) from None
                raise
        candidates.append((path, meta))
    if not candidates:
        msg = f"udlex: no lexicons to compile in {args.bundle}"
        raise SystemExit(msg)

    total = 0
    for path, meta in candidates:
        iso = meta.iso639_3
        compiled = _udlex.compile_bundle(str(path))
        total += compiled.tables
        source_directory = bundle.active_directory(path)
        source_sizes = [
            (source_directory / source_name).stat().st_size
            for source_name in bundle.SOURCES
            if (source_directory / source_name).is_file()
        ]
        pruned = args.prune and _udlex.prune_bundle(str(path))
        if not compiled.from_sources and not pruned:
            print(f"  {iso:5} {path.name:24} already compiled", flush=True)
            continue
        freed = sum(source_sizes) if pruned else 0
        suffix = f", freed {freed / 1e6:.1f} MB" if freed else ""
        print(f"  {iso:5} {path.name:24} {compiled.tables} tables{suffix}", flush=True)
    print(f"\n{total} tables compiled in {args.bundle}")
    return 0


def _verify(args: argparse.Namespace) -> int:
    """Check every bundled lexicon loads and answers consistently."""
    if not args.bundle.is_dir():
        raise _missing_bundle(args.bundle)
    # A bad published declaration aborts instead of reading as a missing language.
    listed = catalog.read(args.bundle)
    if catalog.is_manifested(args.bundle):
        directories = [args.bundle / language.lexicon for language in listed]
    else:
        # Scratch bundles report every visible directory, even unreadable ones.
        directories = sorted(path for path in args.bundle.iterdir() if path.is_dir())
    failures = 0
    for directory in directories:
        try:
            meta = bundle.read_meta(directory)
        except (OSError, ValueError) as error:
            print(f"  FAIL {directory.name}: {error}")
            failures += 1
            continue
        try:
            probe_lemma = _open_lexicon(directory, use_rules=True).lemma
            probe_lemma("x", sentence_initial=False)
            _verify_sources(directory, probe_lemma)
        except SystemExit:
            print(f"  FAIL {directory.name}: built but not compiled")
            failures += 1
            continue
        except (OSError, ValueError) as error:
            print(f"  FAIL {directory.name}: {error}")
            failures += 1
            continue
        size = f"{lookup_rows(meta.entries):>9} entries"
        print(f"  ok   {meta.name:24} {size}  {meta.variant}")
    if not directories:
        print(f"\nno lexicons in {args.bundle}")
        return 1
    print(f"\n{len(directories)} checked, {failures} failure(s)")
    return 1 if failures else 0


def _verify_sources(directory: Path, lemma: Callable[..., str]) -> None:
    """Compare every retained exact source row with the compiled reader."""
    active = bundle.active_directory(directory)
    present = [name for name in bundle.SOURCES if (active / name).exists()]
    if present and len(present) != len(bundle.SOURCES):
        missing = sorted(set(bundle.SOURCES) - set(present))
        msg = f"{directory}: incomplete sources; missing {', '.join(missing)}"
        raise ValueError(msg)
    if not present:
        return
    strings = (active / "strings.txt").read_text(encoding="utf-8").splitlines()
    exact = active / "exact.tsv"
    with exact.open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.removesuffix("\n").removesuffix("\r")
            parts = line.split("\t")
            if len(parts) != 2 or not parts[0]:
                msg = f"{exact}:{line_number}: expected form<TAB>string-id"
                raise ValueError(msg)
            try:
                string_id = int(parts[1])
            except ValueError:
                msg = f"{exact}:{line_number}: invalid string id {parts[1]!r}"
                raise ValueError(msg) from None
            if not 0 <= string_id < len(strings):
                msg = f"{exact}:{line_number}: invalid string id {parts[1]!r}"
                raise ValueError(msg)
            expected = strings[string_id]
            actual = lemma(parts[0], sentence_initial=False)
            if actual != expected:
                msg = (
                    f"{exact}:{line_number}: compiled lookup returned {actual!r}, "
                    f"expected {expected!r}"
                )
                raise ValueError(msg)
