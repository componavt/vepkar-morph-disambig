# Frequency baseline

Source data: `dictorpus-data`, version `v2026.09`.

`test.csv.zst` contains predictions for final evaluation.
`dev.csv.zst` is used to diagnose the frequency baseline.

`dev-diagnostics.csv.zst` is a diagnostic table derived from `dev.csv.zst`
and the strict benchmark for version `v2026.09`.
It contains 2,826 rows: 2,005 correct Top-1 predictions and 821 errors.
Top-1 accuracy: 0.7095.

The main finding of the summary is that the baseline's weakness is not
merely “difficult words”, but primarily zero train frequency of the
expert-selected analysis and frequency ties. For expert candidates with
positive frequency in `train`, Top-1 accuracy on `dev` is about 90%;
for those with zero frequency, it is about 22%.

[How to generate diagnostics](../../../../docs/evaluation.md#frequency-baseline-diagnostics).

## Method

`score` is the number of expert selections of the pair `(wordform_id, gramset)`
in the training split `train`. If the pair was never selected in `train`,
its frequency is set to 0 (a zero-frequency, unseen pair).

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
