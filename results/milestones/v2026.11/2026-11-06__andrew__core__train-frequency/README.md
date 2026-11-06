🌐 [Читать на русском](README.ru.md)

# Frequency baseline

Source data: `dictorpus-data`, version `v2026.11`.
Split: `data/derived/splits_v2026.11.csv`.

`dev.csv.zst` contains the `dev` predictions. The method has not been run on `test` for this version.

## Method

`score` is the number of expert selections of the pair `(wordform_id, gramset)` in `train`.
A pair with no such selections receives `score = 0`.

## Dev result

- Word occurrences: 2,613.
- Candidate rows: 6,519.

| Metric | Value |
|---|---:|
| Top-1 accuracy | 0.6801 |
| MRR | 0.8310 |
| Top-3 accuracy | 0.9946 |

These results cannot be compared directly with `v2026.09`: the words and the split differ.
