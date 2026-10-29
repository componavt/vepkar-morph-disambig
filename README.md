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
python src/cli.py fetch-data v2026.09
```

Inspect the corpus:

```bash
python src/cli.py inspect-data krl
```

For experiments with `v2026.09`, use the published split file
`data/derived/splits_v2026.09.csv`. There is no need to regenerate
the split before running a model.

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
the four language varieties; prediction and evaluation currently
support only `dev`.

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
version recorded in `metrics.json`, for example `v2026.09`.

The package path is printed in the `Milestone directory:` line.
The source run remains unchanged.
An existing result directory is not overwritten.
