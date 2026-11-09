# G1: T1.1 vs. frequency baseline

First completed research checkpoint on `dictorpus-data` v2026.11 using the published text-level split. T1.1 was selected on `dev` from the preapproved grid `C = 0.1, 0.3, 1, 3, 7, 10, 13, 30, 42, 100`; the selection criterion was the highest observed Top-1, with MRR and Top-3 reported as secondary metrics. The selected value was `C=10`.

Uncertainty intervals use 1,000 bootstrap resamples of whole `(language, text_id)` groups with replacement and percentile 95% intervals (seed `20261109`). The `test` split was evaluated once for T1.1 and once for the train-frequency baseline, after the C selection was recorded.

| Test metric | Frequency baseline | T1.1, C=10 | Paired difference, T1.1 − baseline (95% CI) |
| --- | ---: | ---: | ---: |
| Top-1 accuracy | 0.6874 | 0.8480 | +0.1606 (+0.1330, +0.1963) |
| MRR | 0.8352 | 0.9204 | +0.0852 (+0.0705, +0.1042) |
| Top-3 accuracy | 0.9906 | 0.9976 | +0.0071 (+0.0034, +0.0114) |

The paired bootstrap interval for the Top-1 difference is above zero. Per-language and approved diagnostic slices are in `slices.csv`; the most frequent T1.1 Top-1 grammar-set confusions are in `top-confusions.csv`. The selected C grid summary and selection record are also included.

## Files

- `test-comparison.json` — overall test results and paired bootstrap intervals.
- `dev-grid-summary.json` — ten-C dev results and adjacent-C paired bootstrap intervals.
- `dev-grid.csv` — compact table of dev metrics, confidence intervals, and adjacent-C Top-1 differences.
- `selection.json` — records the dev-only C selection made before test evaluation.
- `slices.csv` — Top-1, MRR, Top-3 and text-bootstrap Top-1 intervals for language, training familiarity, candidate count, and NOM/ACC/GEN ambiguity slices.
- `top-confusions.csv` — the 20 most frequent gold-to-T1.1-Top-1 gramset pairs among T1.1 Top-1 errors.
- `t1.1-test.csv.zst` and `frequency-baseline-test.csv.zst` — complete validated test predictions.

This checkpoint evaluates the strict benchmark with one expert-selected gold analysis per word occurrence. It does not include ablations, the frequency-augmented T1.2 model, manual error classification, or a test of published inflection rules. The C grid was selected on dev and its intervals are pointwise; they do not adjust for searching ten settings. The held-out test result is the final estimate for this checkpoint and must not be used to tune follow-up models.
