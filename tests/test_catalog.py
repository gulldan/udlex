"""The catalogue answers without loading lexicons and survives a broken one."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import udlex
from udlex import bundle, catalog

if TYPE_CHECKING:
    from collections.abc import Callable


def _installed_namespace(monkeypatch: pytest.MonkeyPatch, *portions: Path) -> None:
    """Expose real PEP 420 namespace portions without importing them."""
    monkeypatch.delenv("UDLEX_BUNDLE", raising=False)
    roots = [str(portion.parent) for portion in portions]
    other_paths = [entry for entry in sys.path if entry not in roots]
    monkeypatch.setattr(sys, "path", roots + other_paths)


def _portion(
    root: Path,
    write_lexicon: Callable[..., None],
    config: str,
    iso: str,
    forms: dict[str, str],
) -> Path:
    """Write one lexicon into a namespace portion and return the portion."""
    namespace = root / "udlex_bundle"
    write_lexicon(
        namespace / config, name=config, iso639_3=iso, language=iso, forms=forms
    )
    return namespace


@pytest.fixture
def bundle_of_two(tmp_path: Path, write_lexicon: Callable[..., None]) -> Path:
    """Two languages, written out of order so ordering is actually tested."""
    for name, iso, language, forms, accuracy in (
        ("zz_two", "zzb", "Secondish", {"cats": "cat"}, None),
        ("aa_one", "aaa", "Firstish", {"коты": "кот", "кота": "кот"}, 0.9),
    ):
        write_lexicon(
            tmp_path / name,
            name=name,
            iso639_3=iso,
            language=language,
            forms=forms,
            accuracy=accuracy,
            baseline=0.5 if accuracy is not None else None,
            beats_baseline=True if accuracy is not None else None,
        )
    return tmp_path


def test_entries_are_ordered_and_describe_the_language(bundle_of_two: Path) -> None:
    first, second = udlex.supported_languages(bundle_of_two)
    assert (first.iso639_3, second.iso639_3) == ("aaa", "zzb")
    assert first.name == "Firstish"
    assert first.lexicon == "aa_one"
    assert first.entries == 2
    assert first.accuracy == pytest.approx(0.9)
    assert first.gain == pytest.approx(0.4)
    assert first.is_useful
    # An unmeasured language has no gain.
    assert (second.accuracy, second.gain, second.is_useful) == (None, None, None)


def test_a_directory_that_is_not_a_bundle_is_an_empty_catalogue(
    tmp_path: Path,
) -> None:
    assert udlex.supported_languages(tmp_path / "nowhere") == ()
    assert udlex.supported_languages(tmp_path) == ()


def test_one_unreadable_lexicon_does_not_hide_the_others(bundle_of_two: Path) -> None:
    (bundle.active_directory(bundle_of_two / "zz_two") / "meta.json").write_text(
        "{ broken", encoding="utf-8"
    )
    listed = udlex.supported_languages(bundle_of_two)
    assert [language.iso639_3 for language in listed] == ["aaa"]


def test_locating_accepts_a_code_a_treebank_or_a_path(bundle_of_two: Path) -> None:
    assert catalog.locate("aaa", bundle_of_two) == bundle_of_two / "aa_one"
    assert catalog.locate("aa_one", bundle_of_two) == bundle_of_two / "aa_one"
    assert catalog.locate(str(bundle_of_two / "aa_one"), bundle_of_two) == Path(
        str(bundle_of_two / "aa_one")
    )
    assert catalog.locate("zzz", bundle_of_two) is None


def test_manifest_catalogue_never_locates_an_undeclared_stale_directory(
    tmp_path: Path, write_lexicon: Callable[..., None]
) -> None:
    bundle_root = tmp_path / "bundle"
    declared = bundle_root / "aa_one"
    stale = bundle_root / "zz_stale"
    external = tmp_path / "xx_external"
    for path, name, iso in (
        (declared, "aa_one", "aaa"),
        (stale, "zz_stale", "zzb"),
        (external, "xx_external", "xxa"),
    ):
        write_lexicon(
            path,
            name=name,
            iso639_3=iso,
            language=name,
            forms={"cats": "cat"},
        )
    (bundle_root / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": udlex.FORMAT_VERSION,
                "lexicons": [{"iso639_3": "aaa", "config": "aa_one"}],
            }
        ),
        encoding="utf-8",
    )

    assert catalog.locate("aaa", bundle_root) == declared
    assert catalog.locate("aa_one", bundle_root) == declared
    assert catalog.locate("zz_stale", bundle_root) is None
    assert catalog.locate(str(stale), bundle_root) is None
    assert catalog.locate(str(external), bundle_root) == external


def test_the_wrapper_serves_the_same_catalogue_as_the_extension(
    bundle_of_two: Path,
) -> None:
    wrapped = udlex.supported_languages(bundle_of_two)
    direct = udlex.languages(str(bundle_of_two))

    assert all(isinstance(language, udlex.Language) for language in wrapped)
    assert [language.lexicon for language in wrapped] == [
        language.lexicon for language in direct
    ]


def test_split_namespace_resolves_each_language_without_first_path_shadowing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_lexicon: Callable[..., None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = _portion(
        tmp_path / "first", write_lexicon, "aa_one", "aaa", {"cats": "cat"}
    )
    second = _portion(
        tmp_path / "second", write_lexicon, "zz_two", "zzb", {"dogs": "dog"}
    )
    _installed_namespace(monkeypatch, first, second)

    assert udlex.bundle_paths() == (first, second)
    assert [item.iso639_3 for item in udlex.supported_languages()] == ["aaa", "zzb"]
    with pytest.raises(ValueError, match="multiple directories"):
        udlex.bundle_path()
    assert udlex.bundle_path("aaa") == first
    assert udlex.bundle_path("zzb") == second
    assert udlex.load("aaa").lemma("cats") == "cat"
    assert udlex.load("zzb").lemma("dogs") == "dog"

    _installed_namespace(monkeypatch, second, first)
    assert [item.iso639_3 for item in udlex.supported_languages()] == ["aaa", "zzb"]
    assert udlex.bundle_path("aaa") == first
    assert udlex.bundle_path("zzb") == second

    from udlex.cli import main  # noqa: PLC0415

    assert main(["lemma", "-q", "zzb", "dogs"]) == 0
    assert capsys.readouterr().out == "dog\n"
    assert main(["lemma", "-q", str(first / "aa_one"), "cats"]) == 0
    assert capsys.readouterr().out == "cat\n"
    assert main(["languages"]) == 0
    captured = capsys.readouterr()
    assert {line.split()[0] for line in captured.out.splitlines()[1:]} >= {
        "aaa",
        "zzb",
    }
    assert "installed namespace" in captured.err


def test_split_namespace_resolves_fas_alias_in_the_pes_portion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_lexicon: Callable[..., None],
) -> None:
    english = _portion(
        tmp_path / "english", write_lexicon, "en_test", "eng", {"cats": "cat"}
    )
    persian = _portion(
        tmp_path / "persian", write_lexicon, "fa_test", "pes", {"کتاب‌ها": "کتاب"}
    )
    _installed_namespace(monkeypatch, english, persian)

    assert udlex.bundle_path("fas") == persian
    assert udlex.load("fas").lemma("کتاب‌ها") == "کتاب"


@pytest.mark.parametrize(
    "case",
    [
        ("aaa", "shared", "aaa", "shared"),
        ("aaa", "aa_left", "aaa", "zz_right"),
        ("aaa", "shared", "zzb", "shared"),
    ],
)
def test_split_namespace_rejects_every_duplicate_iso_or_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_lexicon: Callable[..., None],
    case: tuple[str, str, str, str],
) -> None:
    left_iso, left_config, right_iso, right_config = case
    first = _portion(
        tmp_path / "first", write_lexicon, left_config, left_iso, {"cats": "cat"}
    )
    second = _portion(
        tmp_path / "second", write_lexicon, right_config, right_iso, {"dogs": "dog"}
    )

    for portions in ((first, second), (second, first)):
        _installed_namespace(monkeypatch, *portions)
        with pytest.raises(ValueError, match=r"duplicate (ISO 639-3 code|config)"):
            udlex.supported_languages()


def test_split_namespace_propagates_manifest_integrity_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_lexicon: Callable[..., None],
) -> None:
    """A corrupt published portion is not reclassified as unsupported."""
    declared = tmp_path / "declared" / "udlex_bundle"
    declared.mkdir(parents=True)
    (declared / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": udlex.FORMAT_VERSION,
                "lexicons": [{"iso639_3": "aaa", "config": "aa_missing"}],
            }
        ),
        encoding="utf-8",
    )
    healthy = _portion(
        tmp_path / "healthy", write_lexicon, "zz_two", "zzb", {"dogs": "dog"}
    )
    _installed_namespace(monkeypatch, healthy, declared)

    with pytest.raises(OSError, match="aa_missing"):
        udlex.supported_languages()


def test_installed_wheel_manifest_binds_the_active_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_lexicon: Callable[..., None],
) -> None:
    portion = _portion(
        tmp_path / "wheel", write_lexicon, "en_test", "eng", {"cats": "cat"}
    )
    manifests = portion / ".manifests"
    manifests.mkdir()
    manifest = manifests / "udlex_dict_eng.json"
    row = {
        "config": "en_test",
        "iso639_3": "eng",
        "generation": (portion / "en_test" / "current")
        .read_text(encoding="utf-8")
        .strip(),
    }
    document = {"format_version": udlex.FORMAT_VERSION, "lexicons": [row]}
    manifest.write_text(json.dumps(document), encoding="utf-8")
    _installed_namespace(monkeypatch, portion)

    assert udlex.load("eng").lemma("cats") == "cat"

    row["generation"] = "stale"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(OSError, match="generation"):
        udlex.load("eng")


def test_bundle_discovery_does_not_execute_package_initializers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "udlex_bundle"
    package.mkdir()
    (package / "__init__.py").write_text(
        'raise AssertionError("dictionary discovery executed package code")\n',
        encoding="utf-8",
    )
    monkeypatch.delitem(sys.modules, "udlex_bundle", raising=False)
    _installed_namespace(monkeypatch, package)

    assert udlex.bundle_paths() == (udlex.DEFAULT_BUNDLE,)
