"""Local CLI resolution, diagnostics and bundle integrity."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import udlex
from udlex import _cli_commands, bundle
from udlex.cli import main

if TYPE_CHECKING:
    from collections.abc import Callable


def _declare_bundle(directory: Path, *lexicons: tuple[str, str]) -> None:
    """Write a generation-bound V1 inventory for CLI boundary tests."""
    rows = []
    for iso, config in lexicons:
        generation = (
            (directory / config / "current").read_text(encoding="utf-8").strip()
        )
        rows.append({"iso639_3": iso, "config": config, "generation": generation})
    (directory / "manifest.json").write_text(
        json.dumps({"format_version": udlex.FORMAT_VERSION, "lexicons": rows}) + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def small_bundle(tmp_path: Path, write_lexicon: Callable[..., None]) -> Path:
    """A two-language bundle, enough to exercise resolution and listing."""
    for name, iso, language, forms in (
        ("xx_one", "xxa", "Firstish", {"коты": "кот"}),
        ("xx_two", "xxb", "Secondish", {"cats": "cat"}),
    ):
        write_lexicon(
            tmp_path / name,
            name=name,
            iso639_3=iso,
            language=language,
            forms=forms,
        )
    return tmp_path


@pytest.fixture
def compiled_bundle(small_bundle: Path) -> Path:
    """The same bundle, compiled: readers only open the compiled tries."""
    for directory in sorted(p for p in small_bundle.iterdir() if p.is_dir()):
        udlex.compile_bundle(str(directory))
    return small_bundle


def test_a_language_is_named_by_its_iso_code(
    compiled_bundle: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--bundle", str(compiled_bundle), "lemma", "-q", "xxa", "коты"]) == 0
    assert capsys.readouterr().out == "кот\n"


def test_a_language_is_also_named_by_its_treebank(
    compiled_bundle: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(["--bundle", str(compiled_bundle), "lemma", "-q", "xx_one", "коты"]) == 0
    )
    assert capsys.readouterr().out == "кот\n"


def test_a_language_is_also_named_by_path(
    compiled_bundle: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = str(compiled_bundle / "xx_one")
    assert main(["lemma", "-q", directory, "коты"]) == 0
    assert capsys.readouterr().out == "кот\n"


def test_an_unknown_language_says_what_to_do_next(small_bundle: Path) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["--bundle", str(small_bundle), "lemma", "zzz", "коты"])
    assert "udlex languages" in str(raised.value)


def test_listing_filters_on_code_and_name(
    small_bundle: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--bundle", str(small_bundle), "languages", "second"]) == 0
    output = capsys.readouterr().out
    assert "xxb" in output
    assert "xxa" not in output


def test_verify_accepts_a_healthy_bundle(
    compiled_bundle: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--bundle", str(compiled_bundle), "verify"]) == 0
    assert "0 failure(s)" in capsys.readouterr().out


def test_verify_does_not_pass_an_empty_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Nothing checked is not the same as nothing wrong."""
    assert main(["--bundle", str(tmp_path), "verify"]) == 1
    assert "no lexicons" in capsys.readouterr().out


def test_verify_uses_manifest_inventory_and_ignores_undeclared_stale_roots(
    compiled_bundle: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stale = compiled_bundle / "zz_stale"
    stale.mkdir()
    (stale / "meta.json").write_text("{ broken", encoding="utf-8")
    _declare_bundle(compiled_bundle, ("xxa", "xx_one"))

    assert main(["--bundle", str(compiled_bundle), "verify"]) == 0

    output = capsys.readouterr().out
    assert "1 checked, 0 failure(s)" in output
    assert "zz_stale" not in output


@pytest.mark.parametrize("command", ["verify", "compile"])
def test_cli_fails_closed_when_a_manifested_lexicon_is_missing(
    compiled_bundle: Path, command: str
) -> None:
    _declare_bundle(compiled_bundle, ("xxa", "xx_one"))
    manifest = json.loads(
        (compiled_bundle / "manifest.json").read_text(encoding="utf-8")
    )
    manifest["lexicons"].append(
        {"iso639_3": "zzb", "config": "zz_missing", "generation": "missing"}
    )
    (compiled_bundle / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    with pytest.raises(SystemExit, match="zz_missing"):
        main(["--bundle", str(compiled_bundle), command])


@pytest.mark.parametrize("command", ["verify", "compile"])
def test_cli_fails_closed_on_a_corrupt_bundle_manifest(
    compiled_bundle: Path, command: str
) -> None:
    (compiled_bundle / "manifest.json").write_text("{ broken", encoding="utf-8")

    with pytest.raises(SystemExit, match=r"manifest|JSON"):
        main(["--bundle", str(compiled_bundle), command])


def test_compile_uses_manifest_inventory_and_rejects_an_explicit_stale_root(
    small_bundle: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stale = small_bundle / "zz_stale"
    stale.mkdir()
    (stale / "meta.json").write_text("{ broken", encoding="utf-8")
    _declare_bundle(small_bundle, ("xxa", "xx_one"))

    assert main(["--bundle", str(small_bundle), "compile"]) == 0
    assert "zz_stale" not in capsys.readouterr().out
    with pytest.raises(SystemExit, match="no such lexicon"):
        main(["--bundle", str(small_bundle), "compile", "zz_stale"])
    with pytest.raises(SystemExit, match="generation-bound published bundle"):
        main(["--bundle", str(small_bundle), "compile", "--prune"])


@pytest.mark.parametrize(
    ("declared", "action", "wrong"),
    [
        (
            bundle.MIN_FORMAT_VERSION - 1,
            "rebuild the bundle with udlex.build_lexicon",
            "upgrade udlex",
        ),
        (
            bundle.FORMAT_VERSION + 1,
            "upgrade udlex",
            "rebuild the bundle with udlex.build_lexicon",
        ),
    ],
)
def test_a_bundle_outside_the_range_says_which_side_is_stale(
    tmp_path: Path, declared: int, action: str, wrong: str
) -> None:
    """Name the stale side, and never suggest "compile", which deletes tables."""
    lexicon = tmp_path / "xx_one"
    lexicon.mkdir()
    (lexicon / "meta.json").write_text(
        json.dumps(
            {
                "name": "xx_one",
                "language": "Firstish",
                "iso639_3": "xxa",
                "source": "xx_one.train",
                "training_tokens": 1,
                "max_suffix": 6,
                "format_version": declared,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as raised:
        main(["--bundle", str(tmp_path), "lemma", "-q", "xx_one", "коты"])
    message = str(raised.value)
    assert "unsupported bundle format" in message
    assert str(declared) in message
    assert bundle.FORMAT_RANGE in message
    assert action in message
    assert wrong not in message
    assert "compile" not in message


def test_a_built_lexicon_is_already_compiled(small_bundle: Path) -> None:
    compiled = udlex.compile_bundle(str(small_bundle / "xx_one"))
    assert not compiled.from_sources
    assert compiled.tables == 0
    assert udlex.Lexicon(str(small_bundle / "xx_one")).lemma("коты") == "кот"


def test_compiling_a_pruned_lexicon_leaves_it_alone(small_bundle: Path) -> None:
    """A missing source must never read as an empty table and wipe the tries."""
    lexicon = small_bundle / "xx_one"
    assert udlex.prune_bundle(lexicon)
    before = {
        str(path.relative_to(lexicon)): path.read_bytes()
        for path in sorted(lexicon.rglob("*"))
        if path.is_file()
    }

    compiled = udlex.compile_bundle(str(lexicon))

    assert not compiled.from_sources
    assert compiled.tables == 0
    assert {
        str(path.relative_to(lexicon)): path.read_bytes()
        for path in sorted(lexicon.rglob("*"))
        if path.is_file()
    } == before
    assert udlex.Lexicon(str(lexicon)).lemma("коты") == "кот"


def test_compiling_half_a_lexicon_refuses(small_bundle: Path) -> None:
    """Sources partly there means partly deleted, and guessing which is worse."""
    lexicon = small_bundle / "xx_one"
    (bundle.active_directory(lexicon) / "external.tsv").unlink()
    with pytest.raises(OSError, match="incomplete sources"):
        udlex.prune_bundle(lexicon)


def test_the_command_line_reports_an_already_compiled_lexicon(
    small_bundle: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--bundle", str(small_bundle), "compile", "--prune"]) == 0
    capsys.readouterr()
    assert main(["--bundle", str(small_bundle), "compile"]) == 0
    assert "already compiled" in capsys.readouterr().out


def test_install_names_every_released_wheel() -> None:
    """Wheel names, the ``_nc`` set and SHA256SUMS agree with the catalog."""
    directory = Path(__file__).parents[1] / "dictionaries" / "catalog"
    catalog = json.loads((directory / "DICTIONARIES.json").read_text(encoding="utf-8"))
    assert catalog["dictionary_version"] == _cli_commands.DICTIONARY_VERSION
    noncommercial = set(catalog["noncommercial_languages"])
    assert noncommercial == _cli_commands.NONCOMMERCIAL_DICTIONARIES
    sums = (directory / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    listed = {name: digest for digest, name in (line.split("  ", 1) for line in sums)}
    assert "SHA256SUMS" not in listed
    for entry in catalog["dictionaries"]:
        wheel = _cli_commands.dictionary_wheel(entry["iso639_3"])
        assert wheel == entry["wheel"]["name"]
        assert listed[f"packages/{wheel}"] == entry["wheel"]["sha256"]


def test_install_dry_run_prints_release_urls(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("UDLEX_DICTIONARIES", raising=False)
    assert main(["install", "--dry-run", "fin", "ENG", "ara", "eng-nc"]) == 0
    base = "https://github.com/gulldan/udlex/releases/download/v1.0.0/"
    assert capsys.readouterr().out.split() == [
        base + "udlex_dict_fin-1.0.0-py3-none-any.whl",
        base + "udlex_dict_eng_nc-1.0.0-py3-none-any.whl",
        base + "udlex_dict_arb_nc-1.0.0-py3-none-any.whl",
        base + "udlex_dict_eng_nc-1.0.0-py3-none-any.whl",
    ]


def test_install_rejects_a_name_that_is_not_a_code() -> None:
    with pytest.raises(SystemExit, match="not an ISO 639-3 code"):
        main(["install", "english"])


def _record_runs(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    runs: list[list[str]] = []

    def run(command: list[str], *, check: bool) -> object:
        assert not check
        runs.append(command)
        return type("Done", (), {"returncode": 0})()

    monkeypatch.setattr(_cli_commands.subprocess, "run", run)
    return runs


def test_install_prefers_uv_bound_to_this_interpreter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runs = _record_runs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UDLEX_DICTIONARIES", "https://mirror.test/d/")
    monkeypatch.setattr("shutil.which", lambda _name: "/bin/uv")
    assert main(["install", "fin"]) == 0
    wheel = "https://mirror.test/d/udlex_dict_fin-1.0.0-py3-none-any.whl"
    assert runs == [["/bin/uv", "pip", "install", "--python", sys.executable, wheel]]


def test_install_falls_back_to_pip_and_a_local_mirror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runs = _record_runs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UDLEX_DICTIONARIES", str(tmp_path))
    monkeypatch.setattr("shutil.which", lambda _name: None)
    monkeypatch.setattr("importlib.util.find_spec", lambda _name: object())
    assert main(["install", "fin"]) == 0
    wheel = str(tmp_path / "udlex_dict_fin-1.0.0-py3-none-any.whl")
    assert runs == [[sys.executable, "-m", "pip", "install", wheel]]


def test_install_in_a_uv_project_names_the_uv_add_that_keeps_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _record_runs(monkeypatch)
    (tmp_path / "uv.lock").write_text("", encoding="utf-8")
    (tmp_path / "src").mkdir()
    monkeypatch.chdir(tmp_path / "src")
    monkeypatch.setenv("UDLEX_DICTIONARIES", "https://mirror.test/d")
    monkeypatch.setattr("shutil.which", lambda _name: "/bin/uv")
    assert main(["install", "eng"]) == 0
    wheel = "https://mirror.test/d/udlex_dict_eng_nc-1.0.0-py3-none-any.whl"
    assert f'uv add "udlex-dict-eng-nc @ {wheel}"' in capsys.readouterr().err
    monkeypatch.setenv("UDLEX_DICTIONARIES", str(tmp_path))
    assert main(["install", "eng"]) == 0
    local = (tmp_path / "udlex_dict_eng_nc-1.0.0-py3-none-any.whl").resolve()
    assert f'"udlex-dict-eng-nc @ {local.as_uri()}"' in capsys.readouterr().err


def test_install_explains_when_no_installer_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = _record_runs(monkeypatch)
    monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
    monkeypatch.setattr("shutil.which", lambda _name: None)
    with pytest.raises(SystemExit, match="neither uv nor pip"):
        main(["install", "fin"])
    assert runs == []
