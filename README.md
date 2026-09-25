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

The CLI scaffold and the tagged source-data fetch command are implemented.
CSV reading, validation, models, and evaluation are not implemented yet.

## Usage

Git and Python 3.11 or newer are required. Fetch all source exports once with
a specific published tag of `componavt/dictorpus-data`:

```bash
python3 src/cli.py fetch-data v2026.09
```

The tag above is an example; use a tag actually published by the data repository.
The command creates `data/dictorpus-data/`, an ignored local Git checkout. A
repeat request for the same tag leaves a clean checkout unchanged; an existing
checkout with another tag or local modifications is never replaced. Future
experiments will read this local checkout without repeating the tag argument.

```bash
python3 src/cli.py --help
python3 src/cli.py --version
```

## Tests

```bash
python3 -m pytest -q
```

## Data

Corpus data come separately from `dictorpus-data` and are not committed here.
Derived shared inputs will live under `data/derived/`.
