# Saved results

This directory stores results manually accepted as reference points
or significant results of method comparisons.

Results are grouped by source-data version.
Each result is stored in a separate directory:

```text
<tag>/<run_id>/
├── README.md
├── test.csv.zst
└── dev.csv.zst
```

`README.md` briefly describes the result: the source-data version,
word-occurrence and candidate-row counts, and metrics for `test`.

`test.csv.zst` is the accepted final evaluation result.
`dev.csv.zst` is saved when needed for diagnostics and is optional.

Result directory names follow this pattern:

```text
YYYY-MM-DD__author__task__method
```

Allowed `task` values: `core`, `t1-features`, `t2-context`, `t3-transfer`.

Predictions are validated and evaluated before acceptance.
Compressed CSV files must be decompressed before validation.
Creating a file and successfully validating it do not automatically
make the result accepted.

Previously accepted predictions are not overwritten.
A newly accepted method variant is stored in a separate directory.

## CSV compression

Example command for maximum compression (`csv` -> `zstd`):

```bash
zstd --ultra -22 file.csv
```

[Prediction format](prediction-format.md).
[Evaluation procedure](../../docs/evaluation.md).
