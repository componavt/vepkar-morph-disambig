# T1.1: feature-based candidate ranking

Binary logistic regression with morphological, suffix, and
immediate-context features, without train-frequency features.

- Data tag: v2026.09
- C: 0.1
- Evaluation split: dev

This is a development-set result, not a final test evaluation.

| Metric | Value |
| --- | ---: |
| Word occurrences | 2826 |
| Candidate rows | 6961 |
| Top-1 accuracy | 0.8132 |
| MRR | 0.9031 |
| Top-3 accuracy | 0.9972 |

- `metrics.json` — evaluation metrics and experiment settings in the
  vepkar-metrics JSON format.
- `dev.csv.zst` — the run's dev predictions, compressed with zstandard.

The trained model is not included in this directory.

See the [T1 feature-based model instructions](../../../../README.md#t1-feature-based-model) for the full
experimental workflow.
