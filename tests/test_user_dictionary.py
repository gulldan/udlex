"""End-to-end contract for independent transactional user dictionaries."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import udlex
from udlex.cli import main

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "rust" / "tests" / "data" / "user_dictionary"


@pytest.fixture
def layered_base(tmp_path: Path, write_lexicon: Callable[..., None]) -> Path:
    """A real compiled base with known, identity and unknown answers."""
    root = tmp_path / "base"
    write_lexicon(
        root / "xx_layered",
        name="xx_layered",
        iso639_3="eng",
        language="Layered test",
        forms={
            "DASHBOARDS": "base-dashboard",
            "guard": "base-guard",
            "cats": "cat",
            "same": "same",
        },
        accuracy=0.8,
        baseline=0.7,
        beats_baseline=True,
        span_case="verbatim",
        span_multiword="compose",
    )
    return root


def _editor(root: Path) -> udlex.UserDictionary:
    return udlex.UserDictionary(root, "eng")


def _generation(root: Path, iso: str = "eng") -> str:
    return (root / iso / "current").read_text(encoding="utf-8").strip()


def test_init_creates_one_authenticated_empty_generation(
    layered_base: Path, tmp_path: Path
) -> None:
    user = tmp_path / "user"
    editor = _editor(user)
    created = editor.init(reason="create empty overlay")
    assert created.changed
    assert created.entries == 0
    assert editor.entries() == ()
    assert editor.history()[0]["operation"] == "init"
    repeated = editor.init(reason="idempotent")
    assert not repeated.changed
    assert repeated.generation == created.generation
    assert len(editor.history()) == 1
    loaded = udlex.load(layered_base, "eng", user_bundle=user)
    assert not loaded.has_user_dictionary
    assert loaded.user_entries == 0
    assert loaded.user_generation == created.generation
    assert loaded.accuracy == loaded.base_accuracy == 0.8
    assert loaded.lemma("cats") == "cat"


def test_python_analyst_flow_and_every_answer_shape(
    layered_base: Path, tmp_path: Path
) -> None:
    user = tmp_path / "user"
    editor = _editor(user)
    lower = editor.set(
        "DASHBOARDS", "manual-dashboard", match="lower", reason="product terminology"
    )
    editor.set("guard", "guard", reason="identity guard")
    editor.set("newword", "newlemma", reason="new coverage")
    assert lower.changed

    base = udlex.load(layered_base, "eng", use_user=False)
    lexicon = udlex.load(layered_base, "eng", user_bundle=user)
    assert lexicon.lemma("DASHBOARDS") == "manual-dashboard"
    assert lexicon.lemma("DASHBOARDS", info=True).source is udlex.Source.USER_LOWER
    identity = lexicon.lemma("guard", info=True)
    assert (identity.lemma, identity.source) == ("guard", udlex.Source.USER)
    assert lexicon.knows("guard")

    # The lower namespace shadows even an exact base row; a miss is the base answer.
    assert base.lemma("DASHBOARDS", info=True).source is udlex.Source.EXACT
    for token in ("cats", "same", "unknown"):
        layered_info = lexicon.lemma(token, info=True)
        base_info = base.lemma(token, info=True)
        assert (layered_info.lemma, layered_info.source) == (
            base_info.lemma,
            base_info.source,
        )

    rich = lexicon.lemmatize(["DASHBOARDS", "guard", "cats"], info=True)
    assert [(row.lemma, row.source) for row in rich] == [
        ("manual-dashboard", udlex.Source.USER_LOWER),
        ("guard", udlex.Source.USER),
        ("cat", udlex.Source.EXACT),
    ]
    assert lexicon.lemmatize(["DASHBOARDS", "guard", "cats"]) == [
        row.lemma for row in rich
    ]
    single = lexicon.lemma_span(["guard"], info=True)
    assert (single.lemma, single.source) == ("guard", udlex.Source.USER)
    text = lexicon.lemma_span("DASHBOARDS cats", info=True)
    assert (text.lemma, text.source) == ("manual-dashboard cat", udlex.Source.COMPOSED)
    assert (
        lexicon.lemma("guard", protected=True, info=True).source
        is udlex.Source.PROTECTED
    )
    entity = lexicon.lemma("guard", entity=True, info=True)
    assert (entity.lemma, entity.source) == ("guard", udlex.Source.IDENTITY)

    assert lexicon.has_user_dictionary
    assert lexicon.user_entries == 3
    assert lexicon.user_generation == _generation(user)
    assert (lexicon.accuracy, lexicon.baseline, lexicon.is_useful) == (None,) * 3
    assert (
        lexicon.base_accuracy,
        lexicon.base_baseline,
        lexicon.base_is_useful,
    ) == (0.8, 0.7, True)

    # Loaded objects retain their mmap snapshot; only a new load sees removal.
    old = lexicon
    editor.remove("DASHBOARDS", match="lower", reason="base repaired")
    fresh = udlex.load(layered_base, "eng", user_bundle=user)
    assert old.lemma("DASHBOARDS", info=True).source is udlex.Source.USER_LOWER
    assert fresh.lemma("DASHBOARDS", info=True).source is udlex.Source.EXACT


def test_load_resolution_explicit_env_absent_and_opt_out(
    layered_base: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_root = tmp_path / "env-user"
    explicit_root = tmp_path / "explicit-user"
    _editor(env_root).set("cats", "env", reason="env")
    _editor(explicit_root).set("cats", "explicit", reason="explicit")

    monkeypatch.setenv(udlex.USER_BUNDLE_VARIABLE, os.fspath(env_root))
    assert udlex.load(layered_base, "eng").lemma("cats") == "env"
    explicit = udlex.load(layered_base, "eng", user_bundle=explicit_root)
    assert explicit.lemma("cats", info=True).lemma == "explicit"
    disabled = udlex.load(
        layered_base, "eng", user_bundle=explicit_root, use_user=False
    )
    assert disabled.lemma("cats", info=True).lemma == "cat"
    assert disabled.has_user_dictionary

    monkeypatch.delenv(udlex.USER_BUNDLE_VARIABLE)
    absent = udlex.load(layered_base, "eng")
    assert not absent.has_user_dictionary
    assert absent.user_generation is None
    assert absent.accuracy == absent.base_accuracy == 0.8

    direct = udlex.Lexicon(
        os.fspath(layered_base / "xx_layered"), user_bundle=os.fspath(explicit_root)
    )
    assert direct.lemma("cats", info=True).source is udlex.Source.USER

    with pytest.raises(OSError, match=r"missing|No such file|user bundle"):
        udlex.load(layered_base, "eng", user_bundle=tmp_path / "missing")

    # A valid explicit root without this ISO is base-only, not a missing root.
    other = tmp_path / "other-language"
    udlex.UserDictionary(other, "rus").set("котами", "кот", reason="Russian")
    no_english = udlex.load(layered_base, "eng", user_bundle=other)
    assert not no_english.has_user_dictionary
    assert no_english.lemma("cats") == "cat"


def test_base_and_user_bundle_schemas_are_not_interchangeable(
    layered_base: Path, tmp_path: Path
) -> None:
    user = tmp_path / "user"
    _editor(user).set("cats", "manual", reason="schema boundary")
    with pytest.raises(OSError, match=r"user-bundle\.json"):
        udlex.verify_user_bundle(layered_base)
    with pytest.raises((OSError, ValueError), match=r"lexicon|metadata|manifest"):
        udlex.load(user, "eng", use_user=False)


def test_shared_import_fixtures_conflicts_audit_and_rollback(tmp_path: Path) -> None:
    user = tmp_path / "user"
    editor = udlex.UserDictionary(user, "rus")
    accepted = editor.import_file(FIXTURES / "valid.tsv", reason="shared fixture")
    assert accepted.changed
    assert editor.entries() == (
        udlex.UserEntry("ACME", "ACME", "exact"),
        udlex.UserEntry("дашборды", "дашборд", "lower"),
    )
    active = accepted.generation
    history_size = len(editor.history())

    for path in sorted(FIXTURES.glob("*.tsv")):
        if path.name == "valid.tsv":
            continue
        with pytest.raises(OSError, match=path.name):
            editor.import_file(path, reason=f"reject {path.name}")
        assert _generation(user, "rus") == active
        assert len(editor.history()) == history_size

    noop = editor.set("ACME", "ACME", reason="idempotent")
    assert not noop.changed
    assert noop.generation == active
    assert len(editor.history()) == history_size
    with pytest.raises(OSError, match="replace=true"):
        editor.set("ACME", "Other", reason="conflict")
    assert _generation(user, "rus") == active
    replaced = editor.set("ACME", "Other", replace=True, reason="approved")
    assert replaced.changed
    with pytest.raises(OSError, match="actor and reason"):
        editor.set("field", "value", reason="")
    with pytest.raises(OSError, match="actor and reason"):
        editor.set("field", "value", actor=" ", reason="audit")
    with pytest.raises(OSError, match="iso639_3"):
        udlex.UserDictionary(user, "RU").set("x", "x", reason="invalid ISO")

    rolled = editor.rollback(active or "", reason="undo replacement")
    assert rolled.changed
    assert rolled.generation != active
    assert editor.entries()[0] == udlex.UserEntry("ACME", "ACME", "exact")
    last = editor.history()[-1]
    assert last["operation"] == "rollback"
    assert last["rollback_from"] == active
    assert udlex.verify_user_bundle(user).active_entries == 2


@pytest.mark.parametrize(
    ("timestamps", "previous_overrides", "expected_order"),
    [
        ((100, 100, 100), {}, ("z-first", "y-second", "x-rollback")),
        ((300, 200, 100), {}, ("z-first", "y-second", "x-rollback")),
        (
            (100, 100, 100),
            {"z-first": "missing"},
            ("z-first", "y-second", "x-rollback"),
        ),
        (
            (100, 100, 100),
            {"x-rollback": "z-first"},
            ("z-first", "x-rollback", "y-second"),
        ),
        ((100, 100, 100), {"z-first": "x-rollback"}, None),
    ],
    ids=["same-millisecond", "clock-backward", "missing-parent", "branch", "cycle"],
)
def test_history_uses_generation_ancestry(
    tmp_path: Path,
    timestamps: tuple[int, int, int],
    previous_overrides: dict[str, str],
    expected_order: tuple[str, str, str] | None,
) -> None:
    user = tmp_path / "user"
    editor = _editor(user)
    first = editor.set("cats", "cat", reason="first").generation
    second = editor.set("dogs", "dog", reason="second").generation
    third = editor.rollback(first or "", reason="undo second").generation
    generation_names = ("z-first", "y-second", "x-rollback")
    renamed = dict(zip((first, second, third), generation_names, strict=True))
    generations = user / "eng" / ".generations"
    for old, new, timestamp in zip(
        (first, second, third), generation_names, timestamps, strict=True
    ):
        assert old is not None
        directory = generations / new
        (generations / old).rename(directory)
        receipt_path = directory / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["generation"] = new
        receipt["timestamp_unix_ms"] = timestamp
        for key in ("previous_generation", "rollback_from"):
            if receipt[key] is not None:
                receipt[key] = renamed[receipt[key]]
        if new in previous_overrides:
            receipt["previous_generation"] = previous_overrides[new]
        encoded = json.dumps(receipt).encode("utf-8")
        receipt_path.write_bytes(encoded)
        meta_path = directory / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["generation"] = new
        meta["files"]["receipt.json"] = {
            "size": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
    (user / "eng" / "current").write_text(generation_names[-1] + "\n", encoding="utf-8")

    if expected_order is None:
        with pytest.raises(OSError, match="cyclic user dictionary history"):
            editor.history()
        return

    history = editor.history()
    assert tuple(receipt["generation"] for receipt in history) == expected_order
    assert {
        receipt["generation"]: receipt["timestamp_unix_ms"] for receipt in history
    } == dict(zip(generation_names, timestamps, strict=True))
    assert udlex.verify_user_bundle(user).active_entries == 1


def test_concurrent_python_writers_do_not_lose_updates(tmp_path: Path) -> None:
    user = tmp_path / "user"
    _editor(user).set("seed", "seed", reason="seed")

    def write(row: tuple[str, str]) -> None:
        _editor(user).set(*row, reason="concurrent")

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(write, [("first", "one"), ("second", "two")]))
    assert {(entry.form, entry.lemma) for entry in _editor(user).entries()} == {
        ("seed", "seed"),
        ("first", "one"),
        ("second", "two"),
    }


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_corruption_truncation_path_swap_and_symlink_fail_closed(
    layered_base: Path, tmp_path: Path
) -> None:
    def seeded(name: str) -> Path:
        root = tmp_path / name
        _editor(root).set("cats", "manual", reason="seed")
        return root

    tampered = seeded("tampered")
    leaf = tampered / "eng" / ".generations" / _generation(tampered)
    (leaf / "entries.tsv").write_text("cats\tchanged\texact\n", encoding="utf-8")
    with pytest.raises(OSError, match=r"entries\.tsv"):
        udlex.load(layered_base, "eng", user_bundle=tampered)

    truncated = seeded("truncated")
    leaf = truncated / "eng" / ".generations" / _generation(truncated)
    (leaf / "strings.idx").write_bytes(b"")
    with pytest.raises(OSError, match=r"strings\.idx"):
        udlex.verify_user_bundle(truncated)

    swapped = seeded("swapped")
    original = _generation(swapped)
    original_path = swapped / "eng" / ".generations" / original
    wrong = "u00000000000000000000000000000000"
    original_path.rename(original_path.parent / wrong)
    (swapped / "eng" / "current").write_text(wrong + "\n", encoding="utf-8")
    with pytest.raises(OSError, match="generation"):
        udlex.load(layered_base, "eng", user_bundle=swapped)

    linked = seeded("linked")
    leaf = linked / "eng" / ".generations" / _generation(linked)
    entries = leaf / "entries.tsv"
    outside = tmp_path / "outside.tsv"
    shutil.copyfile(entries, outside)
    entries.unlink()
    entries.symlink_to(outside)
    with pytest.raises(OSError, match=r"entries\.tsv"):
        udlex.load(layered_base, "eng", user_bundle=linked)


def test_cli_complete_flow_json_status_stdin_and_negative_atomicity(
    layered_base: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    user = tmp_path / "user"
    prefix = ["--bundle", os.fspath(layered_base), "--user-bundle", os.fspath(user)]

    def cli(command: str) -> int:
        return main([*prefix, *command.split()])

    assert cli("user-dict init eng --reason init --json") == 0
    assert json.loads(capsys.readouterr().out)["entries"] == 0
    assert cli("user-dict set eng DASHBOARDS manual --match lower --reason test") == 0
    assert "dashboards\tmanual\tlower" in capsys.readouterr().out
    current = _generation(user)

    with pytest.raises(SystemExit, match="replace=true"):
        cli("user-dict set eng dashboards conflict --match lower --reason reject")
    assert _generation(user) == current

    stdin = io.TextIOWrapper(io.BytesIO(b"stdin\timported\texact\n"), encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", stdin)
    assert cli("user-dict import eng - --reason stdin") == 0
    capsys.readouterr()

    assert cli("lemma -q --with-info eng DASHBOARDS") == 0
    assert capsys.readouterr().out == "manual\tuser_lower\n"

    for command in ("list eng", "history eng", "status eng", "verify"):
        assert cli(f"user-dict {command} --json") == 0
        assert isinstance(json.loads(capsys.readouterr().out), dict)

    assert (
        cli("user-dict remove eng DASHBOARDS --match lower --reason fixed --json") == 0
    )
    assert json.loads(capsys.readouterr().out)["changed"] is True
    assert udlex.load(layered_base, "eng", user_bundle=user).lemma("DASHBOARDS") == (
        "base-dashboard"
    )


def test_cli_requires_separate_user_root_and_env_is_opt_in(
    layered_base: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit, match="requires --user-bundle"):
        main(["user-dict", "list", "eng"])

    monkeypatch.setenv(udlex.USER_BUNDLE_VARIABLE, os.fspath(tmp_path / "env-user"))
    assert (
        main(["user-dict", "set", "eng", "cats", "environment", "--reason", "env"]) == 0
    )
    capsys.readouterr()
    lemma = ["lemma", "-q", "--with-info", "eng", "cats"]
    assert main(["--bundle", os.fspath(layered_base), *lemma]) == 0
    assert capsys.readouterr().out == "environment\tuser\n"
