# Comparing T1.1 runs

This directory stores selected per-occurrence comparisons of completed results,
grouped by source-data version.

## Running a comparison

Run the command from the repository root:

```bash
.venv/bin/python src/cli.py compare-t1-runs \
  --run-dirs \
  results/milestones/v2026.09/t1.1-C0.1-2026-10-29 \
  results/milestones/v2026.09/t1.1-C1-2026-10-27 \
  results/milestones/v2026.09/t1.1-C10-2026-10-29
```

Provide at least two distinct directories. Each must contain `metrics.json`
and exactly one prediction file: `dev.csv` or `dev.csv.zst`.
Working runs and exported results can be mixed.

Validation and expert answers use the local corpus and the published split
`data/derived/splits_<tag>.csv`. The corpus location is read from
`t1.local.toml` if configured; otherwise, `data/dictorpus-data` is used.

The command validates the predictions and compares every pair:

```text
Three runs:          Resulting comparisons:

C=0.1                C=0.1 ↔ C=1
C=1         --->     C=0.1 ↔ C=10
C=10                 C=1   ↔ C=10
```

The terminal report lists the sources, pairwise changes, and leaders for
each metric. For four or more sources, leaders appear in a separate table,
unless one run is the sole leader on all three metrics.

## Direction A → B

Within each pair, B is selected by prioritizing Top-1, then MRR, then Top-3,
using unrounded values. If all three metrics tie, the first input becomes A
and the second becomes B.

```text
A ─── compare predictions ───> B

Corrected: A was wrong; B selected the expert analysis.
Regressed: A selected the expert analysis; B was wrong.

Deltas = metric B − metric A.
```

B may perform worse than A on another metric or an individual language variety.
Not all deltas are necessarily positive.

## Output files

One CSV is created for each pair:

```text
work/comparisons/<tag>/<method A>-C<value A>__vs__<method B>-C<value B>.csv
```

For example:

```text
t1.1-C0.1__vs__t1.1-C10.csv
```

Method and C come from run metadata. Dates and directory suffixes are not
included in the comparison filename. The terminal report identifies the
specific A/B paths.

Output paths are listed under `Outputs`. Existing files are not overwritten.
If different pairs produce the same short filename, the command reports
the collision before writing any files.

## CSV contents

Each row represents one dev word occurrence. All occurrences are included,
not only those whose predictions changed. The correct analysis is identified
by the full `(wordform_id, gramset)` pair.

| Fields | Meaning |
|---|---|
| `language`, `word_id`, `word` | Language code, word identifier, and word form |
| `candidate_count` | Number of candidates |
| `gold_wordform_id`, `gold_gramset` | Expert analysis |
| `a_top1_wordform_id`, `a_top1_gramset` | Top-ranked candidate in A |
| `b_top1_wordform_id`, `b_top1_gramset` | Top-ranked candidate in B |
| `a_gold_rank`, `b_gold_rank` | Rank of the expert analysis in each ranking |
| `top1_changed` | `1` if the top-ranked candidate changed, otherwise `0` |
| `outcome` | Change category |
| `delta_rr` | Change in reciprocal gold rank: `1/b_gold_rank − 1/a_gold_rank` |

Values of `outcome`:

- `both_correct` — both predictions are correct;
- `corrected` — A's error is corrected in B;
- `regressed` — A's correct prediction becomes incorrect in B;
- `both_wrong_same` — both predictions are incorrect and identical;
- `both_wrong_changed` — both predictions are incorrect but differ.

The mean `delta_rr` over all rows equals ΔMRR.

## Saved comparisons

```text
work/comparisons/<tag>/*.csv
              |
              | selected files are copied and compressed manually
              v
results/comparisons/<tag>/*.csv.zst
```

Working CSV files remain local. The comparison command does not move them
into `results/` or add files to Git.

[Comparisons for v2026.11](v2026.11/README.md).

[Comparisons for v2026.09](v2026.09/README.md).
