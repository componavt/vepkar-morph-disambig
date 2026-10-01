🌐 [Read in Russian](evaluation.ru.md)

# Evaluation

This document describes how to create, validate, and evaluate a
predictions file for the `dev` or `test` split.

## Before you start

Run the commands from the repository root with the `.venv` environment
activated. The source data must be downloaded and the split file must
be created; see the [README](../README.md).

The current dataset uses `data/derived/splits_v2026.09.csv`.
For a description of the split, see
[data/derived/README.md](../data/derived/README.md).

## Complete workflow

Example for the `test` split:

```bash
python src/cli.py frequency-baseline \
  --split test \
  --output /tmp/vepkar-frequency-test-v2.csv

python src/cli.py validate-predictions \
  --predictions /tmp/vepkar-frequency-test-v2.csv \
  --split test

python src/cli.py evaluate-predictions \
  --predictions /tmp/vepkar-frequency-test-v2.csv \
  --split test
```

For `dev`, replace `--split test` with `--split dev` and use a
different output file, for example
`/tmp/vepkar-frequency-dev-v2.csv`.

`frequency-baseline` ranks all candidates of each word occurrence by
the frequency of expert analyses in `train`, writes a CSV file, and
validates it. Expert answers from `dev` and `test` are not used for
training.

An existing output file is not overwritten. To run the command again,
choose a new path or delete the previous file if it is no longer needed.
Files in `/tmp/` are intended for temporary storage.

`validate-predictions` checks prediction completeness and validity.
`evaluate-predictions` performs the same check before computing metrics.

Validation and evaluation do not modify the predictions file and do not
save new results.

[CSV format and prediction requirements](../results/milestones/prediction-format.md).

## Metrics

Every evaluated word occurrence has the same weight. The correct
candidate is identified by the exact `(wordform_id, gramset)` pair
selected by the expert.

The evaluation ranks existing candidates. The correct candidate is the
one selected by the expert; no new candidates are created.

- `Top-1 accuracy` — the proportion of word occurrences for which the
  correct candidate is ranked first.
- `MRR` — the mean reciprocal rank of the correct candidate: rank 1
  contributes 1, rank 2 contributes 1/2, and rank 3 contributes 1/3.
- `Top-3 accuracy` — the proportion of word occurrences for which the
  correct candidate appears among the first three positions. If there
  are fewer than three candidates, all available positions are counted.

The metrics are displayed as proportions from 0 to 1 with four decimal
places. Only the displayed values are rounded; the underlying
calculations remain unrounded.

## Comparing methods

A valid comparison requires the same source-data version, split file,
evaluated word occurrences, candidate sets, metrics, and weighting.

Use `dev` to select settings and variants of a method. Use `test` for
the final evaluation, not for selecting settings. Report the results
for the two splits separately.

[Frequency baseline results](frequency_baseline.md).
