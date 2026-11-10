🌐 [Read in Russian](README.ru.md)

# vepkar-morph-disambig

This project studies contextual ranking of candidate morphological analyses in the VepKar corpus: selecting a wordform and a set of grammatical features from the proposed alternatives.

The repository covers three research directions:

- `src/t1_features/` — feature-based models;
- `src/t2_context/` — contextual models;
- `src/t3_transfer/` — transfer learning between language varieties;

Shared algorithms are in `src/core/`, and command handlers are in
`src/commands/`. The command-line entry point is `src/cli.py`.

## Set up the environment

Python 3.11 or newer and Git are required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas zstandard scikit-learn scipy joblib pytest
python -m pytest -q
```

## Work with data

Fetch the source data from [dictorpus-data](https://github.com/componavt/dictorpus-data/):

```bash
python src/cli.py fetch-data v2026.11
```

Inspect the corpus:

```bash
python src/cli.py inspect-data krl
```

For experiments with `v2026.11`, use the published split file
`data/derived/splits_v2026.11.csv`. There is no need to regenerate
the split before running a model. The code reads only this release's format.

> **Unannotated words.** A word with a candidate whose `gramset_id = 0` is treated
> as unannotated and is excluded from the evaluation set as a whole, even if it has
> other candidates. `gramset_id = 7` is handled the same way: this number is absent
> from the gramset reference table. The `gramset` field of such candidates is empty.

Language codes: `vep` — the Veps language; `krl`, `olo`, and `lud` —
the Karelian Proper, Livvi, and Ludic dialects of the Karelian language,
respectively.

[Evaluation set, manual review queue, and data split](data/derived/README.md).

## Evaluation

A frequency baseline is implemented: candidates are ranked by the frequency
of expert analyses in the training split. Quality is evaluated separately
on `dev` and `test`.

Diagnostics for the frequency baseline are available on `dev`:
comparison with expert answers, a per-occurrence CSV table,
and a compact numerical summary.

[Evaluation workflow and metrics](docs/evaluation.md).
[Frequency baseline results](docs/frequency_baseline.md).

## T1: feature-based model

T1.1 ranks candidates using logistic regression. Features include
the candidate's grammatical features, word suffixes, and immediately
adjacent words. Analysis frequencies are not used as features.
A single shared model is trained on the combined `train` split of
the four language varieties. T1 prediction supports `dev` and `test`;
`evaluate-t1-run` publishes metrics for `dev`.

Each prepared run records its feature configuration. `--suffix-max-length K`
adds all surface suffixes from length 1 through `K` (default: 4); `K=0`
disables surface suffix features for an ablation.
The later T1 context ablations use a fixed `K=5`, selected as a working setting
through a `dev` comparison rather than established as a global optimum. The CLI
default remains 4.
`--include-affix` adds dictionary `affix` values joined by
`wordform_id + gramset_id`; multiple linked values are retained. Empty,
unknown (`\N`), special (`#`), and missing dictionary links have distinct
feature states. Older version-4 run passports remain readable and use the
default suffix length 4 without dictionary affix features.

Run the commands from the repository root with the `.venv` environment
activated, after obtaining the source data.

### Preparing and running an experiment

Prepare a new run with `C=1.0`:

```bash
python src/cli.py prepare-t1-run --C 1.0
```

Copy the path from the `Run directory:` line into `RUN_DIR`:

```bash
RUN_DIR="path from the Run directory: line"
python src/cli.py train-t1 --run-dir "$RUN_DIR"
python src/cli.py predict-t1 --run-dir "$RUN_DIR"
python src/cli.py evaluate-t1-run --run-dir "$RUN_DIR"
```

The run directory is populated in sequence with `run.json` containing
the experiment settings, `model.bundle` containing the model,
`dev.csv` containing predictions, and `metrics.json` containing metrics.
Existing results are not overwritten.

For a final held-out evaluation, predict `test` once after selecting the
settings on `dev`:

```bash
python src/cli.py predict-t1 --run-dir "$RUN_DIR" --split test
python src/cli.py validate-predictions --predictions "$RUN_DIR/test.csv" --split test
python src/cli.py evaluate-predictions --predictions "$RUN_DIR/test.csv" --split test
```

`evaluate-predictions` also accepts `--data-dir` and `--split-file` when the
default paths are not in use. It prints test metrics without writing over
the run's dev `metrics.json`.

No configuration is needed when using the default paths. To use other
paths, create the optional `t1.local.toml`; see `t1.local.example.toml`
for an example. After preparation, the commands read the run settings
from `run.json`.

For a different C, [prepare a new run](#preparing-and-running-an-experiment)
with the desired `--C` value. Copy the path after `Run directory:`
in the output of `prepare-t1-run` and assign it to `RUN_DIR`.
Then run training, prediction, and evaluation for that run.

Select C on `dev`; do not use `test` to select settings.

### Running with a single command

From the repository root, run:

```bash
bash scripts/t1-run-tools.sh 0.1
```

The helper creates a new run with the specified C and sequentially performs
preparation, training, prediction, and evaluation. If no argument is supplied,
C=1.0 is used.

Activating `.venv` is not required: the helper uses the project's interpreter,
`.venv/bin/python`. Path settings are read from `t1.local.toml`, if present.

Command output remains in the terminal; no logs are saved.
The helper stops on error. To continue manually, use the
[individual commands](#preparing-and-running-an-experiment)
with the interrupted run's directory.

Result export is performed separately.

### Exporting a result

To export a completed run, copy the desired run's path from the
`Run directory:` line and assign it to `RUN_DIR`:

```bash
RUN_DIR="path from the Run directory: line"
python src/cli.py export-t1-milestone --run-dir "$RUN_DIR"
```

The command creates a separate result package:

```text
work/runs/<run>/
        |
        v
results/milestones/<tag>/<run>/
    README.md       — run description and metrics;
    metrics.json    — a copy of the metrics and experiment settings;
    dev.csv.zst     — compressed dev predictions.
```

Here, `<run>` is the run directory name, and `<tag>` is the source-data
version recorded in `metrics.json`, for example `v2026.11`.

The package path is printed in the `Milestone directory:` line.
The source run remains unchanged.
An existing result directory is not overwritten.

### Comparing runs

`compare-t1-runs` compares existing dev predictions from two or more runs,
reports changes in quality, and saves per-occurrence comparisons for every pair.

[Comparison workflow, CSV fields, and saved results](results/comparisons/README.md).
