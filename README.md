🌐 [Читать на русском](README.ru.md)

# vepkar-morph-disambig

This repository investigates ranking candidate morphological analyses of words
in VepKar context: given one or more possible analyses produced for a word
form, the goal is to rank them using corpus evidence.

The repository contains the owner's implementations of all three research
directions:

- `src/core/` — code shared by the three directions.
- `src/t1_features/` — feature-based morphological candidate ranking.
- `src/t2_context/` — context-based methods.
- `src/t3_transfer/` — transfer experiments.

## Status

This is the initial scaffold: the command-line entry point and the directory
layout. Data fetching, validation, models, and evaluation are not implemented
yet.

## Usage

```bash
python3 src/cli.py --help
python3 src/cli.py --version
python3 src/cli.py
```

## Tests

```bash
python3 -m pytest -q tests/test_cli.py
```

## Data

Corpus data will later be obtained separately from `dictorpus-data` and is
not committed here; `data/dictorpus-data/` is Git-ignored. Derived shared
inputs will live under `data/derived/`.