🌐 [Read in Russian](README.ru.md)

# vepkar-morph-disambig

This project studies contextual ranking of candidate morphological analyses in the VepKar corpus: selecting a wordform and a set of grammatical features from the proposed alternatives.

The repository covers three research directions:

- `src/t1_features/` — feature-based models;
- `src/t2_context/` — context-based models;
- `src/t3_transfer/` — transfer learning between language varieties.

Shared code is in `src/core/`.

## Set up the environment

Python 3.11 or newer and Git are required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pandas zstandard pytest
python -m pytest -q
```

## Work with data

Fetch the source data from [dictorpus-data](https://github.com/componavt/dictorpus-data/):

```bash
python src/cli.py fetch-data v2026.09
```

Inspect the corpus, build a set of instances, and create a split:

```bash
python src/cli.py inspect-data krl
python src/cli.py build-instances
python src/cli.py make-splits
```

Language codes: `vep` — Vepsian, `krl` — Karelian Proper,
`olo` — Livvi Karelian, `lud` — Ludic Karelian.

The generated split is described in
[data/derived/README.md](data/derived/README.md).

## Quality evaluation

A frequency baseline is implemented: candidates are ranked by the frequency
of expert analyses in the training split. Quality is evaluated separately
on `dev` and `test`.

[How to run and compute metrics](docs/evaluation.md).
[Frequency baseline results](docs/frequency_baseline.md).
