# Frequency baseline

Source data: `dictorpus-data`, version `v2026.09`.

`test.csv.zst` contains predictions for final evaluation.
`dev.csv.zst` is used to diagnose the frequency baseline.

## Method

`score` is the number of expert selections of the pair `(wordform_id, gramset)`
in the training split `train`. If the pair was never selected, its frequency
is set to 0 (a zero-frequency, unseen pair).

Candidates are ordered:
1. By descending `score`.
2. For equal frequencies, by ascending `wordform_id`.
3. For equal frequencies and `wordform_id`, by lexicographic `gramset` order.

## Test results

- Word occurrences: 2,801.
- Candidate rows: 6,950.

| Metric | Value |
|---|---:|
| Top-1 accuracy | 0.7319 |
| MRR | 0.8584 |
| Top-3 accuracy | 0.9921 |
