"""Read, verify, and compile local lexicons and manage manual overrides."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from udlex import BUNDLE_VARIABLE, DEFAULT_BUNDLE
from udlex import bundle_path as resolved_bundle_path
from udlex import bundle_paths as installed_bundle_paths
from udlex._cli_commands import (
    DICTIONARIES_VARIABLE,
    _compile,
    _install,
    _languages,
    _lemma,
    _verify,
)
from udlex._cli_user_dictionary import user_dictionary_command

__all__ = ["main"]


def _add_user_dictionary_commands(commands: argparse._SubParsersAction) -> None:
    """Add the manual-override command group without bloating the root parser."""
    user_dictionary = commands.add_parser(
        "user-dict", help="manage transactional manual lemma overrides"
    )
    user_commands = user_dictionary.add_subparsers(
        dest="user_dictionary_command", required=True
    )

    def json_flag(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--json", action="store_true", help="machine-readable output"
        )

    def audit(command: argparse.ArgumentParser) -> None:
        command.add_argument("--actor", help="audit identity (default: local user)")
        command.add_argument("--reason", required=True, help="non-empty audit reason")
        json_flag(command)

    init = user_commands.add_parser(
        "init", help="create an authenticated empty language snapshot"
    )
    init.add_argument("language", help="ISO 639-3 code")
    audit(init)

    set_command = user_commands.add_parser("set", help="add or replace one override")
    set_command.add_argument("language", help="ISO 639-3 code")
    set_command.add_argument("form")
    set_command.add_argument("lemma")
    set_command.add_argument("--match", choices=("exact", "lower"), default="exact")
    set_command.add_argument("--replace", action="store_true")
    audit(set_command)

    remove = user_commands.add_parser("remove", help="remove one override")
    remove.add_argument("language", help="ISO 639-3 code")
    remove.add_argument("form")
    remove.add_argument("--match", choices=("exact", "lower"), default="exact")
    audit(remove)

    import_command = user_commands.add_parser("import", help="merge a strict TSV file")
    import_command.add_argument("language", help="ISO 639-3 code")
    import_command.add_argument("path", help="TSV path, or - for stdin")
    import_command.add_argument("--replace", action="store_true")
    audit(import_command)

    for name, help_text in (
        ("list", "list the active canonical snapshot"),
        ("history", "list authenticated immutable-generation receipts"),
    ):
        command = user_commands.add_parser(name, help=help_text)
        command.add_argument("language", help="ISO 639-3 code")
        json_flag(command)

    rollback = user_commands.add_parser(
        "rollback", help="copy an old snapshot into a new generation"
    )
    rollback.add_argument("language", help="ISO 639-3 code")
    rollback.add_argument("generation")
    audit(rollback)

    status = user_commands.add_parser(
        "status", help="compare active overrides with the current base"
    )
    status.add_argument("language", help="ISO 639-3 code")
    json_flag(status)

    json_flag(
        user_commands.add_parser(
            "verify", help="authenticate every language and generation"
        )
    )
    user_dictionary.set_defaults(handler=user_dictionary_command)


def _parser() -> argparse.ArgumentParser:
    """Construct the complete command parser with focused handler callbacks."""
    parser = argparse.ArgumentParser(prog="udlex", description=__doc__)
    try:
        default_bundle = resolved_bundle_path()
    except ValueError:
        # A namespace spanning several roots is resolved per command after parsing.
        default_bundle = None
    default_description = default_bundle or "installed namespace by language"
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help=(
            f"bundle directory (default: {default_description}, or set "
            f"{BUNDLE_VARIABLE})"
        ),
    )
    configured_user_bundle = os.environ.get("UDLEX_USER_BUNDLE")
    parser.add_argument(
        "--user-bundle",
        type=Path,
        default=(
            None if configured_user_bundle is None else Path(configured_user_bundle)
        ),
        help=(
            "independent manual-override bundle (default: UDLEX_USER_BUNDLE; "
            "never stored in --bundle)"
        ),
    )
    parser.add_argument(
        "--no-user",
        action="store_false",
        dest="use_user",
        help="load and validate an explicit user bundle but disable its lookup layer",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    languages = commands.add_parser("languages", help="list available lexicons")
    languages.add_argument("pattern", nargs="?", help="filter by code or name")
    languages.set_defaults(handler=_languages)

    lemma = commands.add_parser("lemma", help="lemmatise words")
    lemma.add_argument("language", help="ISO 639-3 code, treebank name, or path")
    lemma.add_argument(
        "words",
        nargs="*",
        help="tokens; omit to read from stdin. Put options before the language "
        "or after the tokens, not between them",
    )
    lemma.add_argument(
        "--lookup-only",
        action="store_true",
        help="never guess from suffixes; the right setting for NER input",
    )
    lemma.add_argument(
        "--entity",
        action="store_true",
        help="use only the isolated exact entity table, then preserve the token",
    )
    lemma.add_argument(
        "--protected",
        action="store_true",
        help="preserve every input token without consulting any table",
    )
    lemma.add_argument("-q", "--quiet", action="store_true", help="no header")
    lemma_output = lemma.add_mutually_exclusive_group()
    lemma_output.add_argument(
        "--with-info",
        action="store_true",
        help="print lemma and provenance source as tab-separated fields",
    )
    lemma_output.add_argument(
        "--json",
        action="store_true",
        help="print one JSON object with lemma and provenance per token",
    )
    lemma.set_defaults(handler=_lemma)

    _add_user_dictionary_commands(commands)

    install = commands.add_parser(
        "install", help="install dictionary wheels from the release"
    )
    install.add_argument("languages", nargs="+", help="ISO 639-3 codes")
    install.add_argument(
        "--dry-run",
        action="store_true",
        help=f"print the wheel locations only (source: {DICTIONARIES_VARIABLE})",
    )
    install.set_defaults(handler=_install)

    compile_command = commands.add_parser("compile", help="build the lookup tries")
    compile_command.add_argument(
        "languages", nargs="*", help="ISO 639-3 codes or treebanks; omit for all"
    )
    compile_command.add_argument(
        "--prune",
        action="store_true",
        help="delete the TSV sources afterwards; they are only the build input",
    )
    compile_command.set_defaults(handler=_compile)

    verify = commands.add_parser("verify", help="check a bundle loads")
    verify.set_defaults(handler=_verify)
    return parser


def _resolve_default_bundle(args: argparse.Namespace) -> None:
    """Fill in ``--bundle`` when it was not given, per command."""
    if args.bundle is not None or args.command in {"languages", "install"}:
        return
    if args.command == "user-dict":
        if args.user_dictionary_command == "status":
            args.bundle = resolved_bundle_path(args.language)
        return
    if args.command == "lemma":
        candidate = Path(args.language)
        if args.language != candidate.name and candidate.is_dir():
            resolved = candidate.resolve()
            args.bundle = next(
                (
                    portion
                    for portion in installed_bundle_paths()
                    if resolved.is_relative_to(portion.resolve())
                ),
                DEFAULT_BUNDLE,
            )
        else:
            args.bundle = resolved_bundle_path(args.language)
        return
    try:
        args.bundle = resolved_bundle_path()
    except ValueError:
        message = (
            "installed dictionaries span multiple directories; "
            "pass --bundle for commands that modify or verify a directory"
        )
        raise ValueError(message) from None


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, dispatch one command, and present expected errors."""
    args = _parser().parse_args(argv)
    try:
        _resolve_default_bundle(args)
        return args.handler(args)
    except (OSError, TypeError, ValueError) as error:
        message = f"udlex: {error}"
        raise SystemExit(message) from None


if __name__ == "__main__":
    raise SystemExit(main())
