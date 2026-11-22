"""Operational CLI for transactional user-dictionary overlays."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import udlex
from udlex.user_dictionary import UserDictionary, UserDictionaryUpdate

if TYPE_CHECKING:
    import argparse
    from collections.abc import Mapping


def _required_root(args: argparse.Namespace) -> Path:
    root = args.user_bundle
    if root is None:
        msg = (
            "user-dict requires --user-bundle (or UDLEX_USER_BUNDLE); "
            "the user dictionary is never stored inside the base bundle"
        )
        raise ValueError(msg)
    return Path(root)


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _receipt_for_update(
    dictionary: UserDictionary, update: UserDictionaryUpdate
) -> Mapping[str, object] | None:
    if not update.changed or update.generation is None:
        return None
    return next(
        (
            receipt
            for receipt in reversed(dictionary.history())
            if receipt.get("generation") == update.generation
        ),
        None,
    )


def _emit_update(
    dictionary: UserDictionary,
    update: UserDictionaryUpdate,
    *,
    as_json: bool,
) -> None:
    receipt = _receipt_for_update(dictionary, update)
    payload: dict[str, object] = asdict(update)
    payload["iso639_3"] = dictionary.iso639_3
    if receipt is not None:
        payload["changes"] = receipt.get("changes", [])
    if as_json:
        _print_json(payload)
        return
    state = "published" if update.changed else "unchanged"
    generation = update.generation or "none"
    print(f"{state}: {dictionary.iso639_3} {generation} ({update.entries} entries)")
    changes = payload.get("changes")
    if not isinstance(changes, list):
        return
    for change in changes:
        if not isinstance(change, dict):
            continue
        # Shows what the writer committed, including locale lowercasing.
        value = change.get("new")
        rendered = "<removed>" if value is None else str(value)
        print(f"{change.get('key')}\t{rendered}\t{change.get('match')}")


def _import_stdin(
    dictionary: UserDictionary, args: argparse.Namespace
) -> UserDictionaryUpdate:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="udlex-user-import-", suffix=".tsv", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            shutil.copyfileobj(sys.stdin.buffer, stream, length=1024 * 1024)
            stream.flush()
        return dictionary.import_file(
            temporary_path,
            replace=args.replace,
            actor=args.actor,
            reason=args.reason,
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _set(args: argparse.Namespace, dictionary: UserDictionary) -> UserDictionaryUpdate:
    return dictionary.set(
        args.form,
        args.lemma,
        match=args.match,
        replace=args.replace,
        actor=args.actor,
        reason=args.reason,
    )


def _init(args: argparse.Namespace, dictionary: UserDictionary) -> UserDictionaryUpdate:
    return dictionary.init(actor=args.actor, reason=args.reason)


def _remove(
    args: argparse.Namespace, dictionary: UserDictionary
) -> UserDictionaryUpdate:
    return dictionary.remove(
        args.form, match=args.match, actor=args.actor, reason=args.reason
    )


def _import(
    args: argparse.Namespace, dictionary: UserDictionary
) -> UserDictionaryUpdate:
    if args.path == "-":
        return _import_stdin(dictionary, args)
    return dictionary.import_file(
        args.path, replace=args.replace, actor=args.actor, reason=args.reason
    )


def _rollback(
    args: argparse.Namespace, dictionary: UserDictionary
) -> UserDictionaryUpdate:
    return dictionary.rollback(args.generation, actor=args.actor, reason=args.reason)


def _list(args: argparse.Namespace, dictionary: UserDictionary) -> int:
    rows = [asdict(entry) for entry in dictionary.entries()]
    if args.json:
        _print_json(
            {"iso639_3": dictionary.iso639_3, "entries": rows, "count": len(rows)}
        )
    else:
        for row in rows:
            print(f"{row['form']}\t{row['lemma']}\t{row['match']}")
        print(f"{len(rows)} entries", file=sys.stderr)
    return 0


def _history(args: argparse.Namespace, dictionary: UserDictionary) -> int:
    rows = list(dictionary.history())
    if args.json:
        _print_json({"iso639_3": dictionary.iso639_3, "history": rows})
    else:
        for row in rows:
            previous = row.get("previous_generation") or "none"
            rollback = row.get("rollback_from")
            suffix = f" rollback={rollback}" if rollback else ""
            print(
                f"{row['generation']}\t{row['operation']}\t"
                f"actor={row['actor']}\tprevious={previous}{suffix}\t{row['reason']}"
            )
    return 0


def _status(args: argparse.Namespace, dictionary: UserDictionary) -> int:
    # cli._resolve_default_bundle always resolves --bundle for status.
    base = udlex.load(args.bundle, dictionary.iso639_3, use_user=False)
    rows: list[dict[str, Any]] = []
    counts = {
        "new_coverage": 0,
        "changed_answer": 0,
        "identity_guard": 0,
        "redundant": 0,
    }
    for entry in dictionary.entries():
        base_info = base.lemma(entry.form, info=True)
        if not base.knows(entry.form):
            category = "new_coverage"
        elif base_info.lemma == entry.lemma:
            category = "redundant"
        elif entry.lemma == entry.form:
            category = "identity_guard"
        else:
            category = "changed_answer"
        counts[category] += 1
        rows.append(
            {
                **asdict(entry),
                "category": category,
                "base_lemma": base_info.lemma,
                "base_source": base_info.source.value,
            }
        )
    payload = {
        "iso639_3": dictionary.iso639_3,
        "base_bundle": str(args.bundle),
        "base_lexicon": base.name,
        "base_generation": base.generation,
        "base_format_version": base.format_version,
        "user_bundle": str(dictionary.bundle),
        "counts": counts,
        "entries": rows,
    }
    if args.json:
        _print_json(payload)
    else:
        for row in rows:
            print(
                f"{row['category']}\t{row['form']}\t{row['lemma']}\t"
                f"{row['match']}\tbase={row['base_lemma']} ({row['base_source']})"
            )
        print(" ".join(f"{name}={count}" for name, count in counts.items()))
    return 0


def _verify(args: argparse.Namespace) -> int:
    root = _required_root(args)
    verified = udlex.verify_user_bundle(root)
    payload = asdict(verified)
    payload["user_bundle"] = str(root)
    if args.json:
        _print_json(payload)
    else:
        print(
            f"ok: {root} ({verified.languages} languages, "
            f"{verified.generations} generations, "
            f"{verified.active_entries} active entries)"
        )
    return 0


_UPDATES = {
    "init": _init,
    "set": _set,
    "remove": _remove,
    "import": _import,
    "rollback": _rollback,
}
_READS = {"list": _list, "history": _history, "status": _status}


def user_dictionary_command(args: argparse.Namespace) -> int:
    """Dispatch one fully local user-dictionary operation."""
    command = args.user_dictionary_command
    if command == "verify":
        return _verify(args)
    dictionary = UserDictionary(_required_root(args), args.language)
    if command in _READS:
        return _READS[command](args, dictionary)
    update = _UPDATES[command](args, dictionary)
    _emit_update(dictionary, update, as_json=args.json)
    return 0


__all__ = ["user_dictionary_command"]
