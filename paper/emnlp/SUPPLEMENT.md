# Supplementary material

Code, protocols and count-level evidence for the paper "Which Dictionary Should a Multilingual Lemmatizer Trust?". No treebank text is redistributed: the evidence holds counts, file digests and short word-level examples.

## Contents

- `experiments/` — protocols (`*-protocol.md`, `transfer-amendments.md`), the measurement harness for the engine (`order_native.rs`, `order_diagnostic.rs`), input preparation, runners, summaries and the scripts that typeset every table and number of the paper.
- `experiments/transfer-sources.json` — URL, size and SHA-256 of each corpus outside UD.
- `evidence/ud219/order/` — the source-order experiment on UD 2.19: inventory of sources and splits, per-language ledger, paired counts per bootstrap block, summary.
- `evidence/ud219/order-no-rules/`, `evidence/ud219/factorial.json` — the same fits with suffix rules disabled and the two-by-two check.
- `evidence/ud219/catalog/` — the single-resource packages with rules on and off.
- `evidence/ud219/transfer/`, `evidence/ud219/transfer-no-rules/` — the transfer test: ledger of candidate corpora and the overlap screen, per-corpus and per-language results, paired counts, the provenance screen of lexical sources.
- `evidence/ud219/changes/` — every changed prediction aggregated by type, the classes of change, the headroom counts and two fixed samples prepared for annotators (not annotated).
- `evidence/ud219/robustness.json` — sign tests, language-resampled intervals, the contrast without flagged languages, variants of the confirmation rule.
- `evidence/order/` and the remaining files in `evidence/` — the earlier UD 2.18 run and the audit of the original packages.

## Checking the reported numbers without data

The tables and in-text numbers are generated from the evidence. With Python 3.12 and NumPy 2.5.1 on Linux x86-64, the platform of the experiment:

```sh
cd experiments
python check_factorial.py ../evidence/ud219/order ../evidence/ud219/order-no-rules /tmp/factorial.json
python render_transfer.py     # writes transfer-table.tex and transfer-metrics.tex next to evidence/
python render_analysis.py     # writes kinds-table.tex, gate-table.tex and analysis-metrics.tex
```

`check_factorial.py` recomputes each cohort summary from the per-block counts and compares it with the shipped summary. Counts must match on any platform. The bootstrap intervals (`ci95_pp`, and the resampled intervals of `order_robustness.py`) match byte for byte only on Linux x86-64: NumPy's multinomial sampler gives other draws for the same seed on macOS arm64, where the intervals differ in the third decimal; the script then says so instead of failing on the counts. The render scripts stop if a language or number named in the text no longer follows from the evidence.

## Rerunning

Rerunning needs the engine source at the pinned revision, the official UD 2.19 archive (SHA-256 in `evidence/ud219/order/inventory.json`), the lexical source tables listed there with their digests, and for the transfer test the corpora in `transfer-sources.json`. The order of commands is: `prepare_order_native.py` and a cargo build of `paper_order`; `order_inputs.py`, `order_cv.py`, `run_order.py`, `order_summary.py`; then `transfer_external.py`, `transfer_inputs.py`, `transfer_provenance.py`, `run_order.py` once per transfer corpus, `transfer_summary.py`; and for the diagnostic analyses `prepare_order_native.py --examples`, `changes.py` and `order_robustness.py`. Every script states its arguments with `--help`. Scripts that read a working directory of the original run take its path from the environment: `UDLEX_COMPARE` (engine comparison), `UDLEX_ORDER_RUN` (order fits and UD inputs), `UDLEX_TRANSFER_RUN` (transfer fits) and `UDLEX_BENCH_REPO` (the source registry). Scripts that read third-party corpora are meant to run with `python -I`.

A repeat in empty directories with rebuilt executables and fresh downloads reproduced the source inventory digest, the paired counts of the order experiment and all 167 transfer result files byte for byte. The executable digest itself depends on the build path.
