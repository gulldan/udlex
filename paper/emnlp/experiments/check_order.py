"""Check the fitted policy against exhaustive search and the installed 1.0.0 wheel."""

import argparse
import itertools
import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import udlex

from verify_snapshot import save, sha


def main(binary):
    if udlex.__version__ != "1.0.0":
        raise ValueError("Use the canonical engine wheel")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        folder = root / "eng"
        folder.mkdir()
        words = [
            {
                "cats": "cat",
                "dogs": "hound",
                "birds": "bird",
                "foals": "pony",
                "rains": "rain",
            },
            {
                "cats": "kitten",
                "dogs": "dog",
                "birds": "bird",
                "foals": "foal",
                "rains": "raining",
            },
            {
                "cats": "cat",
                "dogs": "dog",
                "birds": "birdie",
                "foals": "pony",
                "rains": "rainy",
            },
        ]
        names = [f"source-{j}" for j in range(len(words))]
        inputs = []
        for j, mapping in enumerate(words):
            file = root / (names[j] + ".tsv")
            file.write_text(
                "".join(x + "\t" + y + "\n" for x, y in sorted(mapping.items()))
            )
            inputs.append(
                dict(
                    id=names[j],
                    path=file.name,
                    sha256=sha(file),
                    lemma_first=False,
                    unique=True,
                )
            )
        save(
            root / "inventory.json",
            {"languages": {"eng": {"status": "paired", "sources": inputs}}},
        )
        train = (
            [("cats", "cat")] * 2
            + [("dogs", "dog")] * 5
            + [("birds", "bird")] * 3
            + [("runs", "run")]
        )
        test = [
            ("foals", "foal"),
            ("rains", "rain"),
            ("dogs", "dog"),
            ("fawns", "fawn"),
            ("cats", "cat"),
            ("missing", "missing"),
        ]
        for split, data in [("train", train), ("dev", []), ("test", test)]:
            (folder / (split + ".jsonl")).write_text(
                "".join(
                    json.dumps({"tokens": [x], "lemmas": [y], "block": "s:" + str(i)})
                    + "\n"
                    for i, (x, y) in enumerate(data)
                )
            )
        out = root / "result.json"
        subprocess.run([binary, root, root, "eng", out], check=True)
        got = json.loads(out.read_bytes())

        def answer(order, x):
            return next((words[i][x] for i in order if x in words[i]), x)

        def ranking(order):
            score = sum(answer(order, x) == y for x, y in train)
            inversions = sum(a > b for a, b in itertools.combinations(order, 2))
            return -score, inversions, order

        fixed = tuple(range(3))
        best = min(itertools.permutations(fixed), key=ranking)
        if best == fixed or got["order"][1] != [names[j] for j in best]:
            raise AssertionError("Wrong winning permutation")
        all_orders = [
            perm for n in range(4) for perm in itertools.permutations(fixed, n)
        ]
        if got["train_correct"][1] != -ranking(best)[0] or got["train_correct"][
            4
        ] != max(-ranking(o)[0] for o in all_orders):
            raise AssertionError(
                "Native training optimum disagrees with exhaustive enumeration"
            )
        results = []
        for i, order in enumerate([fixed, best]):
            dst = root / f"wheel-{i}" / "eng"
            dst.mkdir(parents=True)
            ext = {x: answer(order, x) for m in words for x in m}
            tables = udlex.build(
                [(x, y, True) for x, y in train], list(ext.items()), []
            )
            entries = dict(tables.write_sources(str(dst), "full", (0.0, 0.0)))
            save(
                dst / "meta.json",
                dict(
                    name="eng",
                    iso639_3="eng",
                    language="English",
                    format_version=5,
                    source="enumerated fixture",
                    training_tokens=len(train),
                    max_suffix=6,
                    variant="full",
                    entity_encoding="namespaced-v1",
                    entries=entries,
                    token_case="keep",
                    normalization="NFC+locale-v1",
                    traffic={},
                ),
            )
            udlex.compile_bundle(str(dst))
            lex = udlex.Lexicon(str(dst), use_user=False)
            results.append([lex.lemmatize([x])[0] for x, _ in test])
        c = Counter(tokens=len(test), correct=0, wins=0, losses=0, changed=0)
        for (_, y), a, b in zip(test, *results, strict=True):
            c.update(
                correct=int(b == y),
                wins=int(b == y and a != y),
                losses=int(b != y and a == y),
                changed=int(a != b),
            )
        if c != got["test"]["counts"][1] or not c["changed"]:
            raise AssertionError((c, got["test"]["counts"][1]))
        print(
            "Exhaustive training optimum and full-cascade wheel predictions match",
            dict(c),
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("binary", type=Path)
    main(p.parse_args().binary)
