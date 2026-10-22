🌐 [Read in Russian](evaluation.ru.md)

# Evaluation

This document describes how to run the frequency baseline on `dev`
or `test`, the evaluation metrics, and frequency-baseline diagnostics
on `dev`. For the T1 workflow, see the
[README](../README.md#t1-feature-based-model).

## Before you start

Run the commands from the repository root with the `.venv` environment
activated. Obtain the source data by following the
[instructions in the README](../README.md#work-with-data).

For evaluation, use the published split
`data/derived/splits_v2026.09.csv`.
Its structure and generation are described in the
[split section](../data/derived/README.md#traindevtest-split).

## Frequency baseline workflow

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

`validate-predictions` and `evaluate-predictions` do not modify
the predictions file or save new results.
The T1 command `evaluate-t1-run` saves metrics to `metrics.json`.

[CSV format and prediction requirements](../results/milestones/prediction-format.md).

## Metrics

Every evaluated word occurrence has the same weight. The correct
candidate is identified by the exact `(wordform_id, gramset)` pair
selected by the expert.

The metrics evaluate the ranking of existing candidates.
The correct candidate is the one selected by the expert;
no new candidates are created.

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

## Frequency baseline diagnostics

After generating predictions for `dev`, run:

```bash
python src/cli.py diagnose-frequency-baseline \
  --predictions /tmp/vepkar-frequency-dev-v2.csv \
  --output /tmp/vepkar-frequency-dev-diagnostics.csv
```

The command works only with `dev`. It validates the predictions
and confirms that candidate identities, ranks, and scores match
the frequency baseline for the same data and split.

The input must be an uncompressed predictions CSV. Decompress a
`.csv.zst` archive beforehand. The output directory must exist;
an existing file or symbolic link is not replaced.

The diagnostic CSV contains one row per evaluated word occurrence,
including correct predictions:

```csv
language,word_id,word,candidate_count,gold_wordform_id,gold_gramset,top1_wordform_id,top1_gramset,gold_rank,gold_train_frequency,top1_train_frequency
```

`word` is copied from the normalized corpus field.
`gold_*` describes the expert-selected candidate; `top1_*` describes
the top-ranked candidate. `gold_rank = 1` means a correct Top-1 prediction.
Candidates are compared by the exact `(wordform_id, gramset)` pair.

Frequencies count expert selections of this pair in `train`,
not occurrences of the word's spelling in the corpus.

After saving the CSV, the command prints overall statistics,
the distribution of the expert candidate's rank, zero-frequency
and tie statistics, and tables by subcorpus and candidate count.
Ratios with a zero denominator are displayed as `N/A`.

The diagnostic table is not a predictions file:
it contains expert information and is intended for error analysis.

## Comparing methods

A valid comparison requires the same source-data version, split file,
evaluated word occurrences, candidate sets, metrics, and weighting.

Use `dev` to select settings and variants of a method. Use `test` for
the final evaluation, not for selecting settings. Report the results
for the two splits separately.

[Frequency baseline results](frequency_baseline.md).
