# udlex system demonstration paper

Draft for the NAACL 2027 System Demonstrations track. [PDF](../../output/pdf/udlex-demo.pdf)

## Rebuild the paper

From the repository root:

```sh
uv run --frozen python -I paper/naacl/scripts/make_tables.py
tectonic -X compile paper/naacl/udlex-demo.tex --outdir output/pdf
```

`make_tables.py` writes `generated/numbers.tex`. Every number in the text and both
tables comes from there; nothing is typed by hand. It reads only committed files:

| Input | What it holds |
| --- | --- |
| `evidence/speed/` | Speed and memory receipts (C9): `cells.jsonl.gz` has one record per timed cell, `summary.json` the medians, `environment.json` the machine and the hashes of engine, wheel and harness |
| `evidence/release/` | The dictionary release: `evaluation.json` (accuracy of every package), `DICTIONARIES.json.gz` (sources, licenses, input hashes, archive sizes) |
| `../emnlp/evidence/ud219/census/` | Other lemmatizers scored on the same test words, and the split-drift report used to drop test leakage |

## Where the evidence comes from

The speed receipts were produced by the harness in
[`../emnlp/experiments/speed`](../emnlp/experiments/speed): `quiet_run.sh` drives
`all_languages.sh` (CPU modes) and a GPU phase, `bench.py` times each cell in a fresh
pinned process and checks every answer against the system's own accuracy run, and
`native/` is the Rust worker for udlex. The CPU cells of the other systems were reused
from an earlier session on the same machine with the same harness;
[`evidence/speed/REUSED.md`](evidence/speed/REUSED.md) says which, and each reused cell
carries `reused_from`.

`scripts/collect_evidence.py` copied the receipts and the release manifest into this
directory. It keeps every measured value and drops what describes the machine's other
users: names of foreign processes (their CPU share stays) and absolute paths. The
SHA-256 of each original file is in [`evidence/sources.json`](evidence/sources.json).
Running it again on the same originals gives byte-identical files.

## Comparison rules

- Accuracy is exact lemma match on gold tokens of the UD 2.19 test split.
- A system is compared with udlex only on languages where both were scored on the same
  test words. Rows of the accuracy table therefore cover different languages and are not
  comparable with each other.
- A language is dropped for a system when its model was trained on more than 1 % of our
  test sentences (split-drift report), when Stanza has no lemmatizer for it, or when
  Stanza's lemma dictionary knows at least half of the test word types our train lacks.
- Speed-ups are geometric means of per-language ratios on the languages both systems
  were timed on; rates are medians over languages.
