# Frequency baseline

Results from the run on 1 October 2026.

`score` is the number of expert selections of the pair
`(wordform_id, gramset)` among strict `train` instances.
Frequencies are shared across the four subcorpora;
a pair with no such selections receives `score = 0`.

The training split contains 22,390 expert analyses and
7,314 distinct `(wordform_id, gramset)` pairs.

| Split | Word occurrences | Candidate rows | Top-1 accuracy | MRR | Top-3 accuracy |
|---|---:|---:|---:|---:|---:|
| dev | 2,826 | 6,961 | 0.7095 | 0.8454 | 0.9890 |
| test | 2,801 | 6,950 | 0.7319 | 0.8584 | 0.9921 |

Both prediction files passed validation when created and were checked
separately with `validate-predictions`.

Ties are resolved deterministically: the candidate with the smaller
`wordform_id` is ranked higher; if `wordform_id` is also equal, the
lexicographically smaller `gramset` is ranked higher.

The `dev` and `test` predictions and the `dev` diagnostic table
are stored in the
[baseline bundle](../results/milestones/v2026.09/2026-10-01__andrew__core__train-frequency/README.md).

[Reproduction workflow and metric definitions](evaluation.md).
