"""Checks of the dictionary build's selection, resolution, splitting and licensing."""

from __future__ import annotations

import gzip
import itertools
import random
from collections import Counter
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

choose = pytest.importorskip("choose")
extract = pytest.importorskip("extract")
package = pytest.importorskip("package")
splits = pytest.importorskip("splits")


def objective(
    maps: list[dict[str, str]], gold: dict[str, Counter[str]], order: list[int]
) -> int:
    """Correct training tokens when the first listed source holding a form answers."""
    total = 0
    for form, labels in gold.items():
        answer = next((maps[j][form] for j in order if form in maps[j]), form)
        total += labels.get(answer, 0)
    return total


def test_learned_order_is_the_exact_optimum() -> None:
    rng = random.Random(7)  # noqa: S311 - reproducible test data
    for _ in range(40):
        forms = [f"f{i}" for i in range(12)]
        lemmas = ["a", "b", "c"]
        maps = [
            {f: rng.choice(lemmas) for f in forms if rng.random() < 0.6}
            for _ in range(4)
        ]
        gold = {
            f: Counter({rng.choice([*lemmas, f]): rng.randint(1, 3)}) for f in forms
        }
        policies = choose.learn(maps, gold)
        best = max(
            objective(maps, gold, list(p)) for p in itertools.permutations(range(4))
        )
        assert objective(maps, gold, policies["learned"]) == best
        assert objective(maps, gold, policies["subset"]) >= best
        assert sorted(policies["learned"]) == list(range(4))


def test_mcnemar_exact_two_sided() -> None:
    assert choose.mcnemar([False] * 6, [True] * 6) == (6, 0, 2 / 64)
    assert choose.mcnemar([True, False], [True, False])[2] == 1.0


def test_resolution_prefers_count_then_frequency_then_length(tmp_path: Path) -> None:
    path = tmp_path / "t.tsv.gz"
    rows = [
        ("a", "x", 2),
        ("a", "y", 1),
        ("b", "long", 1),
        ("b", "s", 1),
        ("c", "s", 1),
    ]
    with gzip.open(path, "wt", encoding="utf-8") as out:
        out.writelines(f"{f}\t{y}\t{n}\n" for f, y, n in sorted(rows))
    assert choose.resolve([str(path)], None) == {"a": "x", "b": "s", "c": "s"}


def test_marks_absent_from_training_are_stripped() -> None:
    strip = choose.stripper({"̆"})
    assert strip("ру́сский") == "русский"
    assert strip("й") == "й"


def test_residual_votes_only_count_forms_unseen_elsewhere() -> None:
    sentences = [
        {"tokens": ["x", "y"], "lemmas": ["x", "y"], "block": f"d:{n}"}
        for n in range(10)
    ] + [{"tokens": ["rare"], "lemmas": ["r"], "block": "d:rare"}]
    assert set(choose.residual_votes(sentences)) == {"rare"}


def test_package_license_follows_the_strictest_component() -> None:
    assert package.package_license(["CC-BY-SA-4.0", "BSD-2-Clause"]) == (
        "CC-BY-SA-4.0",
        False,
    )
    assert package.package_license(["CC-BY-4.0", "CC-BY-SA-3.0"]) == (
        "CC-BY-SA-3.0",
        False,
    )
    assert package.package_license(["CC-BY-NC-SA-3.0", "CC-BY-SA-4.0"]) == (
        "CC-BY-NC-SA-4.0",
        True,
    )


def test_disagreeing_readme_and_license_count_as_noncommercial(tmp_path: Path) -> None:
    bank = tmp_path / "UD_X-Y"
    bank.mkdir()
    (bank / "README.md").write_text("License: CC BY-SA 4.0\n")
    (bank / "LICENSE.txt").write_text(
        "http://creativecommons.org/licenses/by-nc-sa/4.0/\n"
    )
    found = splits.treebank_license(bank, "xxx")
    assert found["noncommercial"]
    assert found["requires_license_review"]
    assert found["license"] == "CC-BY-NC-SA-4.0"


def test_official_splits_drop_test_sentences_seen_in_train(tmp_path: Path) -> None:
    bank = tmp_path / "UD_X-Y"
    bank.mkdir()
    sentence = "1\tDogs\tdog\tNOUN\t_\t_\t0\troot\t_\t_\n\n"
    other = "1\tCats\tcat\tNOUN\t_\t_\t0\troot\t_\t_\n\n"
    (bank / "xx_y-ud-train.conllu").write_text(
        "# newdoc id = d1\n# sent_id = s1\n" + sentence
    )
    (bank / "xx_y-ud-dev.conllu").write_text(
        "# newdoc id = d2\n# sent_id = s2\n" + other
    )
    (bank / "xx_y-ud-test.conllu").write_text(
        "# newdoc id = d1\n# sent_id = s3\n"
        + other
        + "# newdoc id = d3\n# sent_id = s4\n"
        + sentence
    )
    result = splits.write_official("xx_y", bank, tmp_path / "out")
    assert result["test"].get("tokens", 0) == 0
    assert result["test"]["dropped_shared"] == 2


def test_table_parser_skips_the_copyright_block(tmp_path: Path) -> None:
    path = tmp_path / "t.tab.gz"
    with gzip.open(path, "wt", encoding="utf-8") as out:
        out.write("#!DICT-ID x\n#<COPYRIGHT>\nCopyright notice\n#</COPYRIGHT>\n")
        out.write("kota\tkot:Sm2\tsubst\n")
    spec = {
        "container": "gz",
        "delimiter": "\t",
        "form": 0,
        "lemma": 1,
        "comment": "#",
        "block": ["#<COPYRIGHT>", "#</COPYRIGHT>"],
        "lemma_sub": [["(?<=.):.*$", ""]],
    }
    notices: list[str] = []
    assert list(extract.table(path, spec, notices)) == [("kota", "kot")]
    assert notices == ["Copyright notice"]
