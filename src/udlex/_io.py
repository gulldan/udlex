"""Shared filesystem primitives: bounded reads, locks and bundle transactions."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import unicodedata
import uuid
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, TypeGuard, cast

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

_LOCK_NAME = ".udlex.lock"
INSTALL_JOURNAL = ".udlex-install-journal.json"
BUILD_JOURNAL = ".udlex-build-journal.json"
_MAX_BUILD_JOURNAL_BYTES = 1024 * 1024
_MAX_BUNDLE_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_GENERATION_META_BYTES = 4 * 1024 * 1024
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_CONFIG_NAME = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$", re.ASCII)
_WHEEL_MANIFEST_TARGET = re.compile(
    r"^\.manifests/udlex_dict_[a-z]{3}\.json$", re.ASCII
)
#: ``v1`` journals left by an older udlex must still be recovered.
INSTALL_JOURNAL_SCHEMAS = ("udlex-install/v1", "udlex-install/v2")
_GENERATION_NAME = re.compile(r"^[A-Za-z0-9_-]{1,128}$", re.ASCII)
_SHA256 = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_TRANSACTION_ID = re.compile(r"^[0-9a-f]{32}$", re.ASCII)


def has_control_character(value: str) -> bool:
    """Whether the text carries a Unicode Cc character (stable across versions)."""
    return any(unicodedata.category(character) == "Cc" for character in value)


def is_sha256_hex(value: object) -> TypeGuard[str]:
    """Whether a value is a lowercase hex SHA-256 digest."""
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object while rejecting recursively duplicated keys."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            msg = f"duplicate JSON key {key!r}"
            raise ValueError(msg)
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    """Reject the non-standard NaN/Infinity spellings accepted by json.loads."""
    msg = f"non-finite JSON number {value}"
    raise ValueError(msg)


def _finite_float(value: str) -> float:
    """Reject a well-formed number such as ``1e400`` that decodes to infinity."""
    parsed = float(value)
    if not math.isfinite(parsed):
        msg = f"non-finite JSON number {value}"
        raise ValueError(msg)
    return parsed


def strict_json_loads(payload: str | bytes, source: object) -> object:
    """Decode unambiguous RFC 8259 JSON for control and integrity metadata."""
    try:
        return json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        msg = f"{source}: invalid JSON: {error}"
        raise ValueError(msg) from None


def _is_reparse_point(value: os.stat_result) -> bool:
    """Whether a Windows stat result names a junction/symlink-like object."""
    return bool(getattr(value, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT)


def _validate_directory(directory: Path) -> os.stat_result:
    """Create and then validate a real directory without following a link."""
    if directory.exists() and directory.is_symlink():
        msg = f"refusing symlink bundle directory: {directory}"
        raise OSError(msg)
    directory.mkdir(parents=True, exist_ok=True)
    value = os.lstat(directory)
    if not stat.S_ISDIR(value.st_mode) or _is_reparse_point(value):
        msg = f"refusing non-regular bundle directory: {directory}"
        raise OSError(msg)
    return value


def _same_object(left: os.stat_result, right: os.stat_result) -> bool:
    """Compare two stat results without trusting a pathname after opening."""
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


@contextmanager
def _stable_descriptor(
    path: Path, max_bytes: int, description: str
) -> Generator[int, None, None]:
    """Open exactly the regular file inspected, and prove it held still.

    Closes the lstat-then-open swap window: the descriptor is compared with the
    path before and after the caller's reads, refusing links and in-place edits.
    """
    if max_bytes < 0:
        msg = "max_bytes must be non-negative"
        raise ValueError(msg)
    before = os.lstat(path)
    if (
        not stat.S_ISREG(before.st_mode)
        or _is_reparse_point(before)
        or before.st_size > max_bytes
    ):
        msg = f"refusing unsafe or oversized {description}: {path}"
        raise OSError(msg)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        after_open = os.lstat(path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _is_reparse_point(opened)
            or not _same_object(before, opened)
            or not _same_object(opened, after_open)
            or opened.st_size > max_bytes
        ):
            msg = f"refusing changed or unsafe {description}: {path}"
            raise OSError(msg)
        yield descriptor
        final = os.fstat(descriptor)
        after_read = os.lstat(path)
        if (
            not _same_object(opened, final)
            or not _same_object(final, after_read)
            or opened.st_size != final.st_size
            or opened.st_mtime_ns != final.st_mtime_ns
            or opened.st_ctime_ns != final.st_ctime_ns
        ):
            msg = f"refusing changed {description}: {path}"
            raise OSError(msg)
    finally:
        os.close(descriptor)


def read_bounded_regular(
    path: Path,
    max_bytes: int,
    *,
    description: str = "file",
) -> bytes:
    """Read one bounded regular file through a verified no-follow descriptor."""
    payload = bytearray()
    with _stable_descriptor(path, max_bytes, description) as descriptor:
        while chunk := os.read(
            descriptor, min(1024 * 1024, max_bytes - len(payload) + 1)
        ):
            payload.extend(chunk)
            if len(payload) > max_bytes:
                msg = f"refusing oversized {description}: {path}"
                raise OSError(msg)
    return bytes(payload)


def sha256_bounded_regular(
    path: Path,
    max_bytes: int,
    *,
    description: str = "file",
) -> tuple[str, int]:
    """Hash one bounded regular file through a stable no-follow descriptor."""
    digest = hashlib.sha256()
    size = 0
    with _stable_descriptor(path, max_bytes, description) as descriptor:
        while chunk := os.read(descriptor, 1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                msg = f"refusing oversized {description}: {path}"
                raise OSError(msg)
            digest.update(chunk)
    return digest.hexdigest(), size


def _validate_lock(
    path: Path, descriptor: int, before: os.stat_result | None, what: str
) -> None:
    """Prove the opened lock is the regular path entry we inspected."""
    opened = os.fstat(descriptor)
    after = os.lstat(path)
    if (
        not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(after.st_mode)
        or _is_reparse_point(opened)
        or _is_reparse_point(after)
        or not _same_object(opened, after)
        or (before is not None and not _same_object(before, opened))
    ):
        msg = f"refusing unsafe {what}: {path}"
        raise OSError(msg)


@contextmanager
def _exclusive_lock(
    directory: Path, name: str, what: str, owner: str
) -> Generator[None, None, None]:
    """Hold one kernel lock on ``directory / name``; see :func:`bundle_lock`."""
    directory_before = _validate_directory(directory)
    path = directory / name
    try:
        before = os.lstat(path)
    except FileNotFoundError:
        before = None
    else:
        if not stat.S_ISREG(before.st_mode) or _is_reparse_point(before):
            msg = f"refusing unsafe {what}: {path}"
            raise OSError(msg)
    flags = os.O_RDWR | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        directory_after = os.lstat(directory)
        if (
            not _same_object(directory_before, directory_after)
            or not stat.S_ISDIR(directory_after.st_mode)
            or _is_reparse_point(directory_after)
        ):
            msg = f"{owner} changed while locking: {directory}"
            raise OSError(msg)  # noqa: TRY301 - descriptor must close below
        _validate_lock(path, descriptor, before, what)
    except BaseException:
        os.close(descriptor)
        raise
    with os.fdopen(descriptor, "a+b") as stream:
        if os.name == "nt":
            import msvcrt  # noqa: PLC0415 - absent on POSIX

            if path.stat().st_size == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl  # noqa: PLC0415 - absent on Windows

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def bundle_lock(directory: Path) -> Generator[None, None, None]:
    """Hold the inter-process writer lock for ``directory``.

    A kernel lock, so a killed writer cannot leave the bundle locked; readers
    never take it.
    """
    with _exclusive_lock(directory, _LOCK_NAME, "bundle lock", "bundle directory"):
        yield


@contextmanager
def generation_writer_lock(directory: Path) -> Generator[None, None, None]:
    """Share the native builder's per-lexicon ``.write.lock`` protocol."""
    with _exclusive_lock(
        directory, ".write.lock", "generation writer lock", "lexicon directory"
    ):
        yield


@contextmanager
def generation_writer_locks(
    directory: Path, configs: Iterable[str]
) -> Generator[None, None, None]:
    """Hold native writer locks for canonical configs in deterministic order."""
    ordered = sorted(set(configs))
    if any(not valid_config_name(config) for config in ordered):
        msg = "generation lock set contains an invalid config"
        raise ValueError(msg)
    with ExitStack() as stack:
        for config in ordered:
            stack.enter_context(generation_writer_lock(directory / config))
        yield


def valid_generation_name(value: object) -> TypeGuard[str]:
    """Whether ``value`` is a native immutable-generation identifier."""
    return isinstance(value, str) and _GENERATION_NAME.fullmatch(value) is not None


def read_current_generation(root: Path) -> str | None:
    """Read one native ``current`` pointer without following a symlink."""
    try:
        root_meta = os.lstat(root)
    except FileNotFoundError:
        return None
    if (
        not stat.S_ISDIR(root_meta.st_mode)
        or _is_reparse_point(root_meta)
        or root.is_symlink()
    ):
        msg = f"{root}: lexicon root must be a real directory"
        raise ValueError(msg)
    current = root / "current"
    try:
        payload = read_bounded_regular(
            current,
            129,
            description="lexicon generation pointer",
        )
    except FileNotFoundError:
        return None
    try:
        decoded = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        msg = f"{current}: generation pointer is not UTF-8"
        raise ValueError(msg) from None
    generation = decoded.rstrip("\r\n")
    if not valid_generation_name(generation) or decoded.strip() != generation:
        msg = f"{current}: invalid generation pointer"
        raise ValueError(msg)
    return generation


def validate_sealed_generation(root: Path, generation: str) -> None:
    """Require an immutable leaf and its metadata to name the same identity."""
    if not valid_generation_name(generation):
        msg = f"{root}: invalid sealed generation {generation!r}"
        raise ValueError(msg)
    generations = root / ".generations"
    leaf = generations / generation
    for path in (generations, leaf):
        try:
            metadata = os.lstat(path)
        except FileNotFoundError:
            msg = f"{path}: sealed generation is missing"
            raise ValueError(msg) from None
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or _is_reparse_point(metadata)
            or path.is_symlink()
        ):
            msg = f"{path}: sealed generation path must be a real directory"
            raise ValueError(msg)
    meta_path = leaf / "meta.json"
    loaded = strict_json_loads(
        read_bounded_regular(
            meta_path,
            _MAX_GENERATION_META_BYTES,
            description="sealed generation metadata",
        ),
        meta_path,
    )
    if not isinstance(loaded, dict) or loaded.get("generation") != generation:
        msg = f"{meta_path}: metadata generation does not match leaf"
        raise ValueError(msg)


def switch_current_locked(
    root: Path,
    *,
    expected: str | None,
    replacement: str | None,
) -> None:
    """CAS one active pointer while its native writer lock is already held."""
    observed = read_current_generation(root)
    if observed != expected:
        msg = f"{root}: current generation changed: {observed!r} != {expected!r}"
        raise ValueError(msg)
    if replacement == observed:
        return
    current = root / "current"
    if replacement is None:
        current.unlink(missing_ok=True)
        _sync_directory(root)
        return
    validate_sealed_generation(root, replacement)
    atomic_write_text(current, replacement + "\n")


def atomic_write_text(path: Path, text: str) -> None:
    """Replace a UTF-8 text file without exposing a partial write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
        _sync_directory(path.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _file_present(path: Path) -> bool:
    """Return pathname-entry presence without following a symlink."""
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _sha256_bytes(payload: bytes) -> str:
    """Return the lowercase SHA-256 identifier used by build journals."""
    return hashlib.sha256(payload).hexdigest()


def _manifest_facts(payload: bytes) -> dict[str, object]:
    """Describe exact manifest bytes without embedding them in the journal."""
    return {"size": len(payload), "sha256": _sha256_bytes(payload)}


def _render_build_journal(payload: dict[str, object]) -> str:
    """Serialize the durable build transaction control record."""
    return json.dumps(payload, ensure_ascii=True, indent=1, sort_keys=True) + "\n"


def prepare_build_transaction(
    directory: Path,
    *,
    old_manifest: bytes | None,
    new_manifest: bytes,
    operations: Iterable[tuple[str, str | None, str | None]],
) -> dict[str, object]:
    """Durably record a prepared bundle-generation transaction.

    The caller holds the bundle lock and every affected generation writer lock.
    The exact old manifest lives in a bounded backup; the journal authenticates
    that backup, the replacement manifest, and every pointer transition.
    """
    journal = directory / BUILD_JOURNAL
    if _file_present(journal):
        msg = f"unfinished build transaction already exists: {journal}"
        raise OSError(msg)
    transaction = uuid.uuid4().hex
    backup_name = f".udlex-build-manifest-backup-{transaction}"
    backup = directory / backup_name
    if _file_present(backup):
        msg = f"build manifest backup already exists: {backup}"
        raise OSError(msg)
    old_payload = b"" if old_manifest is None else old_manifest
    try:
        old_text = old_payload.decode("utf-8", errors="strict")
        new_manifest.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        msg = "bundle transaction manifests must be UTF-8"
        raise ValueError(msg) from None
    rows = [
        {"config": config, "old": old, "new": new} for config, old, new in operations
    ]
    payload: dict[str, object] = {
        "schema": "udlex-build/v1",
        "phase": "prepared",
        "transaction": transaction,
        "manifest_existed": old_manifest is not None,
        "manifest_backup": backup_name,
        "old_manifest": _manifest_facts(old_payload),
        "new_manifest": _manifest_facts(new_manifest),
        "operations": rows,
    }
    atomic_write_text(backup, old_text)
    try:
        atomic_write_text(journal, _render_build_journal(payload))
    except BaseException:
        backup.unlink(missing_ok=True)
        _sync_directory(directory)
        raise
    return payload


def mark_build_transaction_committed(
    directory: Path, payload: dict[str, object]
) -> dict[str, object]:
    """Durably mark a prepared build transaction as fully committed."""
    committed = {**payload, "phase": "committed"}
    atomic_write_text(
        directory / BUILD_JOURNAL,
        _render_build_journal(committed),
    )
    return committed


def _build_retirement_paths(
    directory: Path, payload: dict[str, object]
) -> dict[str, Path]:
    """Return hidden retirement targets for operations removing a config."""
    transaction = payload.get("transaction")
    operations = payload.get("operations")
    if not isinstance(transaction, str) or not isinstance(operations, list):
        msg = "invalid build transaction retirement metadata"
        raise TypeError(msg)
    retired: dict[str, Path] = {}
    for raw in operations:
        if not isinstance(raw, dict):
            msg = "invalid build transaction retirement operation"
            raise TypeError(msg)
        config = raw.get("config")
        if raw.get("new") is None:
            if not valid_config_name(config):
                msg = "invalid build transaction retirement config"
                raise ValueError(msg)
            retired[config] = directory / (
                f".udlex-build-retired-{transaction}-{config}"
            )
    return retired


def retire_build_transaction_roots(directory: Path, payload: dict[str, object]) -> None:
    """Hide configs removed by a committed manifest before clearing its journal."""
    for config, retired in _build_retirement_paths(directory, payload).items():
        root = directory / config
        root_present = _file_present(root)
        retired_present = _file_present(retired)
        if root_present and retired_present:
            msg = f"ambiguous build retirement paths for {config}"
            raise OSError(msg)
        if not root_present:
            continue
        root.replace(retired)
        _sync_directory(directory)


def finish_build_transaction(directory: Path, payload: dict[str, object]) -> None:
    """Remove a completed journal before its now-unneeded manifest backup."""
    journal = directory / BUILD_JOURNAL
    backup_name = payload.get("manifest_backup")
    if not isinstance(backup_name, str):
        msg = "build transaction has no manifest backup"
        raise TypeError(msg)
    journal.unlink(missing_ok=True)
    _sync_directory(directory)
    (directory / backup_name).unlink(missing_ok=True)
    for retired in _build_retirement_paths(directory, payload).values():
        _remove_path(retired)
    _sync_directory(directory)


def _journal_facts(value: object, field: str) -> tuple[int, str]:
    """Validate one ``{size, sha256}`` journal object."""
    if not isinstance(value, dict):
        msg = f"invalid build transaction {field} facts"
        raise OSError(msg)
    size = value.get("size")
    sha256 = value.get("sha256")
    if (
        not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
        or size > _MAX_BUNDLE_MANIFEST_BYTES
        or not is_sha256_hex(sha256)
    ):
        msg = f"invalid build transaction {field} facts"
        raise OSError(msg)
    return size, sha256


def _journal_generation(value: object, field: str) -> str | None:
    """Validate an optional generation recorded in one journal operation."""
    if value is None or valid_generation_name(value):
        return value
    msg = f"invalid build transaction {field} generation"
    raise OSError(msg)


def _load_build_transaction(
    directory: Path,
) -> tuple[dict[str, object], bytes, list[tuple[str, str | None, str | None]]]:
    """Read and authenticate a build journal and its exact old-manifest backup."""
    journal = directory / BUILD_JOURNAL
    loaded = strict_json_loads(
        read_bounded_regular(
            journal,
            _MAX_BUILD_JOURNAL_BYTES,
            description="build transaction journal",
        ),
        journal,
    )
    if (
        not isinstance(loaded, dict)
        or loaded.get("schema") != "udlex-build/v1"
        or loaded.get("phase") not in {"prepared", "committed"}
        or not isinstance(loaded.get("manifest_existed"), bool)
        or not isinstance(loaded.get("operations"), list)
    ):
        msg = f"invalid build transaction journal: {journal}"
        raise OSError(msg)
    payload = cast("dict[str, object]", loaded)
    transaction = payload.get("transaction")
    if not isinstance(transaction, str) or not _TRANSACTION_ID.fullmatch(transaction):
        msg = f"invalid build transaction identifier in {journal}"
        raise OSError(msg)
    backup_name = payload.get("manifest_backup")
    expected_backup = f".udlex-build-manifest-backup-{transaction}"
    if backup_name != expected_backup:
        msg = f"invalid build manifest backup in {journal}"
        raise OSError(msg)
    old_size, old_sha256 = _journal_facts(payload.get("old_manifest"), "old manifest")
    _journal_facts(payload.get("new_manifest"), "new manifest")
    backup_path = directory / expected_backup
    old_payload = read_bounded_regular(
        backup_path,
        _MAX_BUNDLE_MANIFEST_BYTES,
        description="build manifest backup",
    )
    if len(old_payload) != old_size or _sha256_bytes(old_payload) != old_sha256:
        msg = f"build manifest backup does not match {journal}"
        raise OSError(msg)
    if not payload["manifest_existed"] and old_payload:
        msg = f"absent old manifest has a non-empty backup in {journal}"
        raise OSError(msg)
    raw_operations = cast("list[object]", payload["operations"])
    operations: list[tuple[str, str | None, str | None]] = []
    seen: set[str] = set()
    for raw in raw_operations:
        if not isinstance(raw, dict):
            msg = f"invalid build transaction operation in {journal}"
            raise OSError(msg)
        config = raw.get("config")
        if not valid_config_name(config) or config in seen:
            msg = f"invalid or duplicate build transaction config in {journal}"
            raise OSError(msg)
        seen.add(config)
        operations.append(
            (
                config,
                _journal_generation(raw.get("old"), "old"),
                _journal_generation(raw.get("new"), "new"),
            )
        )
    return payload, old_payload, operations


def _read_optional_manifest(path: Path) -> bytes | None:
    """Read exact bounded manifest bytes, distinguishing absence from empty."""
    try:
        return read_bounded_regular(
            path,
            _MAX_BUNDLE_MANIFEST_BYTES,
            description="bundle manifest",
        )
    except FileNotFoundError:
        return None


def _matches_facts(payload: bytes | None, facts: object) -> bool:
    """Whether optional bytes match journal size and digest facts."""
    if payload is None:
        return False
    size, sha256 = _journal_facts(facts, "manifest")
    return len(payload) == size and _sha256_bytes(payload) == sha256


def _restore_old_build_state(
    directory: Path,
    payload: dict[str, object],
    old_manifest: bytes,
    operations: list[tuple[str, str | None, str | None]],
) -> None:
    """Idempotently restore the prepared transaction's old manifest/pointers."""
    manifest_path = directory / "manifest.json"
    if payload["manifest_existed"]:
        try:
            old_text = old_manifest.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            msg = "build manifest backup is not UTF-8"
            raise OSError(msg) from None
        if _read_optional_manifest(manifest_path) != old_manifest:
            atomic_write_text(manifest_path, old_text)
    elif _file_present(manifest_path):
        manifest_path.unlink()
        _sync_directory(directory)
    for config, old, new in reversed(operations):
        root = directory / config
        observed = read_current_generation(root)
        if observed != old:
            switch_current_locked(root, expected=new, replacement=old)


def recover_build_transaction_locked(  # noqa: C901
    directory: Path,
    *,
    generation_locks_held: bool = False,
) -> str | None:
    """Recover one durable build journal while the bundle lock is held.

    A prepared transaction always rolls back. A committed transaction is only
    cleaned up after proving that its replacement manifest and every active
    generation are still exact. Any third-party mutation is ambiguous and
    fails closed without changing another byte.
    """
    journal = directory / BUILD_JOURNAL
    if not _file_present(journal):
        return None
    payload, old_manifest, operations = _load_build_transaction(directory)
    manifest_path = directory / "manifest.json"
    current_manifest = _read_optional_manifest(manifest_path)
    old_matches = (
        current_manifest == old_manifest
        if payload["manifest_existed"]
        else current_manifest is None
    )
    new_matches = _matches_facts(current_manifest, payload.get("new_manifest"))
    retirement_paths = _build_retirement_paths(directory, payload)
    completed_retirements: set[str] = set()
    for config, retired in retirement_paths.items():
        root_present = _file_present(directory / config)
        retired_present = _file_present(retired)
        if root_present and retired_present:
            msg = f"ambiguous build retirement paths for {config}"
            raise OSError(msg)
        if retired_present:
            if payload["phase"] != "committed":
                msg = f"prepared build transaction unexpectedly retired {config}"
                raise OSError(msg)
            completed_retirements.add(config)
    lock_configs = (
        config
        for config, _old, _new in operations
        if config not in completed_retirements
    )
    locks = (
        nullcontext()
        if generation_locks_held
        else generation_writer_locks(directory, lock_configs)
    )
    with locks:
        observed = {
            config: (
                None
                if config in completed_retirements
                else read_current_generation(directory / config)
            )
            for config, _old, _new in operations
        }
        for config, old, new in operations:
            if observed[config] not in {old, new}:
                msg = (
                    f"ambiguous build transaction mutation for {config}: "
                    f"{observed[config]!r} is neither {old!r} nor {new!r}"
                )
                raise OSError(msg)
        if payload["phase"] == "prepared":
            if not old_matches and not new_matches:
                msg = "ambiguous manifest mutation during prepared build transaction"
                raise OSError(msg)
            _restore_old_build_state(directory, payload, old_manifest, operations)
            recovered = "rolled-back"
        else:
            if not new_matches:
                msg = "committed build transaction manifest is not its replacement"
                raise OSError(msg)
            mismatches = [
                config for config, _old, new in operations if observed[config] != new
            ]
            if mismatches:
                names = ", ".join(mismatches)
                msg = f"committed build transaction pointers changed: {names}"
                raise OSError(msg)
            retire_build_transaction_roots(directory, payload)
            recovered = "committed"
    finish_build_transaction(directory, payload)
    return recovered


def recover_bundle_transactions_locked(directory: Path) -> str | None:
    """Recover the sole permitted bundle transaction under its writer lock."""
    build_present = _file_present(directory / BUILD_JOURNAL)
    install_present = _file_present(directory / INSTALL_JOURNAL)
    if build_present and install_present:
        msg = f"conflicting build and install transactions in {directory}"
        raise OSError(msg)
    if build_present:
        return recover_build_transaction_locked(directory)
    if install_present:
        recover_install_transaction_locked(directory)
        return "install-recovered"
    return None


def recover_bundle_transactions(directory: Path) -> str | None:
    """Recover an interrupted build/install before observing a bundle."""
    if not (
        _file_present(directory / BUILD_JOURNAL)
        or _file_present(directory / INSTALL_JOURNAL)
    ):
        return None
    with bundle_lock(directory):
        return recover_bundle_transactions_locked(directory)


def _remove_path(path: Path) -> None:
    """Remove one transaction-owned path without following a symlink."""
    try:
        value = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISDIR(value.st_mode) and not stat.S_ISLNK(value.st_mode):
        shutil.rmtree(path)
    else:
        path.unlink()


def _transaction_name(value: object, prefix: str | None = None) -> str:
    """Validate one journal path as a single, optionally prefixed component."""
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or (prefix is not None and not value.startswith(prefix))
    ):
        msg = f"unsafe install transaction path {value!r}"
        raise OSError(msg)
    return value


def valid_config_name(value: object) -> TypeGuard[str]:
    """Whether a bundle config is a non-reserved canonical path component."""
    return isinstance(value, str) and _CONFIG_NAME.fullmatch(value) is not None


def install_target_backup(name: object, transaction: str) -> str:
    """Return the flat backup component that stands in for one journal target.

    A target is a top-level lexicon directory or a ``.manifests/`` wheel
    manifest; the backup name says which.
    """
    if valid_config_name(name):
        return f".udlex-config-backup-{transaction}-{name}"
    if isinstance(name, str) and _WHEEL_MANIFEST_TARGET.fullmatch(name):
        return f".udlex-wheel-backup-{transaction}-{name.split('/', 1)[1]}"
    msg = f"unsafe install transaction target {name!r}"
    raise OSError(msg)


def recover_install_transaction_locked(  # noqa: C901, PLR0912, PLR0915
    directory: Path,
) -> None:
    """Recover/finish an install while the caller holds :func:`bundle_lock`."""
    journal = directory / INSTALL_JOURNAL
    try:
        journal_payload = read_bounded_regular(
            journal,
            1024 * 1024,
            description="install transaction journal",
        )
    except FileNotFoundError:
        return
    loaded = strict_json_loads(journal_payload, journal)
    if (
        not isinstance(loaded, dict)
        or loaded.get("schema") not in INSTALL_JOURNAL_SCHEMAS
        or loaded.get("phase") not in {"prepared", "committed"}
        or not isinstance(loaded.get("manifest_existed"), bool)
        or not isinstance(loaded.get("targets"), list)
    ):
        msg = f"invalid install transaction journal: {journal}"
        raise OSError(msg)
    payload = cast("dict[str, object]", loaded)
    manifest_backup = _transaction_name(
        payload.get("manifest_backup"), ".udlex-manifest-backup-"
    )
    transaction = manifest_backup.removeprefix(".udlex-manifest-backup-")
    if not _TRANSACTION_ID.fullmatch(transaction):
        msg = f"invalid install transaction identifier in {journal}"
        raise OSError(msg)
    manifest = directory / "manifest.json"
    manifest_backup_path = directory / manifest_backup
    rows: list[tuple[Path, Path, bool]] = []
    seen_targets: set[str] = set()
    target_rows = cast("list[object]", payload["targets"])
    for row in target_rows:
        if not isinstance(row, dict) or not isinstance(row.get("existed"), bool):
            msg = f"invalid install transaction target in {journal}"
            raise OSError(msg)
        target_row = cast("dict[str, object]", row)
        target_name = target_row.get("name")
        expected_backup = install_target_backup(target_name, transaction)
        if target_name in seen_targets:
            msg = f"invalid or duplicate install transaction target in {journal}"
            raise OSError(msg)
        seen_targets.add(cast("str", target_name))
        target = directory / cast("str", target_name)
        if target.parent != directory:
            # A valid leaf name must not let recovery follow a linked container.
            parent = target.parent.lstat()
            if not stat.S_ISDIR(parent.st_mode) or _is_reparse_point(parent):
                msg = f"unsafe install transaction parent: {target.parent}"
                raise OSError(msg)
        if _transaction_name(target_row.get("backup")) != expected_backup:
            msg = f"install transaction backup does not match its target in {journal}"
            raise OSError(msg)
        backup_name = expected_backup
        backup = directory / backup_name
        rows.append((target, backup, bool(target_row["existed"])))
    if payload["phase"] == "prepared":
        for target, backup, existed in rows:
            if backup.exists() or backup.is_symlink():
                _remove_path(target)
                backup.replace(target)
            elif not existed:
                _remove_path(target)
        if bool(payload["manifest_existed"]):
            if not manifest_backup_path.exists():
                msg = (
                    "install recovery manifest backup is missing: "
                    f"{manifest_backup_path}"
                )
                raise OSError(msg)
            backup_payload = read_bounded_regular(
                manifest_backup_path,
                _MAX_BUNDLE_MANIFEST_BYTES,
                description="install recovery manifest backup",
            )
            try:
                backup_text = backup_payload.decode("utf-8")
            except UnicodeDecodeError:
                msg = f"install recovery manifest is not UTF-8: {manifest_backup_path}"
                raise OSError(msg) from None
            atomic_write_text(manifest, backup_text)
        else:
            _remove_path(manifest)
    for _, backup, _ in rows:
        _remove_path(backup)
    _remove_path(manifest_backup_path)
    journal.unlink()
    _sync_directory(directory)


def _sync_directory(directory: Path) -> None:
    """Persist a rename where the platform exposes directory ``fsync``."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
