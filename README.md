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

The CLI scaffold, the tagged source-data fetch command, typed corpus loading
with schema/join validation (`inspect-data`), and reproducible
morphological-ranking-instance construction (`build-instances`) are
implemented. Models and evaluation are not implemented yet.

## Usage

Git and Python 3.11 or newer are required. Fetch all source exports once with
a specific published tag of `componavt/dictorpus-data`:

```bash
python3 src/cli.py fetch-data v2026.09
```

In 2026, `v2026.09` is the published recommended tag for the first local
checkout. The command creates `data/dictorpus-data/`, an ignored local Git
checkout. A repeat request for the same tag leaves a clean checkout unchanged;
an existing checkout with another tag or local modifications is never replaced.
Future experiments will read this local checkout without repeating the tag
argument.

```bash
python3 src/cli.py --help
python3 src/cli.py --version
python3 src/cli.py inspect-data krl
python3 src/cli.py build-instances
```

`inspect-data` reads the local tagged corpus checkout under
`data/dictorpus-data/corpus/`, parses the four corpus tables into typed
DataFrames, and reports row counts, schema and identifier checks, duplicate and
join integrity, candidate-group counts, and `relevance=2` expert-selection
diagnostics. Pipe-separated `corpus_id` and `genre_id` source fields are parsed
into tuple-valued `corpus_ids` and `genre_ids`. The command prints the complete
available diagnostics before it returns a nonzero status for broken links or
duplicate keys. Zero `word_number` values and empty `gramset` values are
preserved and reported as findings rather than removed. For each reported
finding, at most three contextual examples are shown with the original sentence
XML when the link is unique; when the sentence XML cannot be retrieved, the
reason is stated explicitly. No per-row issue file is generated. The command
writes no derived data and never modifies `dictorpus-data`. An optional
`--data-dir` overrides the checkout location (used by the offline tests)
and writes no derived data.
Models, splits, benchmarks, and evaluation are not implemented yet.

`build-instances` reads each of the four language varieties one at a time from
the same local checkout, constructs one ranking problem per corpus word
occurrence, and prints a sequential eligibility funnel:

```text
0. all source words
1. words with >=2 candidate rows and exactly one relevance=2 row
2. ... with >=2 distinct (wordform_id, gramset) analyses
3. ... with no repeated candidate identity
4. ... whose selected analysis has a nonempty gramset
5. ... whose sentence links to one unambiguous existing text
6. ... whose entire sentence has only unique, positive word_number values
```

Each transition prints how many instances were retained and removed. Overlap
diagnostics among step-1 words and an empty-unselected-gramset policy
(primary pool versus a conservative alternative pool) are also reported. The
command then writes the single developer-review file
`data/derived/quality/zero_word_number_sentences_<tag>.csv` with the exact
schema `language,sentence_id`, one row per distinct sentence containing a
`word_number == 0`, sorted deterministically and derived from all source
`words` rows. Zero positions and repeated positive positions exclude whole
sentences from instances; empty `gramset` on an unselected candidate is
retained in the primary pool. The output filename's tag is read from the
local Git checkout, and the checked-out source data are never modified. An
optional `--output-dir` overrides the derived-output location (used by the
offline tests).

## Tests

```bash
python3 -m pytest -q
```

## Data

Corpus data come separately from `dictorpus-data` and are not committed here.
Derived shared inputs will live under `data/derived/`.
