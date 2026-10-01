🌐 [Read in Russian](evaluation.ru.md)

# Quality evaluation

This document describes building, validating, and evaluating a prediction file
for `dev` or `test`.

## Before running

Run the commands from the repository root with the `.venv` environment
activated. The source data must be fetched and the split file prepared;
the commands are given in [README](../README.md).

For the current dataset, `data/derived/splits_v2026.09.csv` is used.
The split is described in
[data/derived/README.md](../data/derived/README.md).

## Full run procedure

Example for `test`:

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

For `dev`, replace `--split test` with `--split dev` and use a different
output file, for example `/tmp/vepkar-frequency-dev-v2.csv`.

`frequency-baseline` ranks all candidates of each word occurrence by the
frequency of expert analyses in `train`, writes a CSV, and validates it.
The expert answers for `dev` and `test` are not used for training.

An existing output file is not overwritten. To rerun, choose a new path or
delete the previous file if it is no longer needed. Files in `/tmp/` are
intended for temporary storage.

`validate-predictions` separately checks the completeness and validity of
predictions. `evaluate-predictions` repeats this check before computing
metrics, so a separate validation call is useful for diagnostics but not
required.

Validation and evaluation do not modify the prediction file and do not save
new results.

[CSV format and prediction requirements](../results/milestones/prediction-format.md).

## Metrics

Every evaluated word occurrence has equal weight. The correct candidate is
determined by the exact `(wordform_id, gramset)` pair chosen by the expert.

- `Top-1 accuracy` — the fraction of word occurrences for which the correct
  candidate is ranked first.
- `MRR` — the mean reciprocal rank of the correct candidate: first place
  gives 1, second — 1/2, third — 1/3.
- `Top-3 accuracy` — the fraction of word occurrences for which the correct
  candidate is among the first three positions. If there are fewer than three
  candidates, all available positions are considered.

Metrics are printed as fractions from 0 to 1 with four decimal places.
Only the output is rounded, not the underlying computations.

## Comparing methods

A correct comparison requires one version of the source data, one split file,
the same evaluated word occurrences and candidate sets, the same metrics and
weighting method.

Settings and method variants are selected on `dev`. The `test` set is used
for the final evaluation, not for choosing settings. The results of the two
splits are reported separately.

[Frequency baseline results](frequency_baseline.md).
