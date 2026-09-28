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
with schema/join validation (`inspect-data`), reproducible
morphological-ranking-instance construction (`build-instances`), and a
deterministic whole-text `train/dev/test` split (`make-splits`) are
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
python3 src/cli.py make-splits
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
`--data-dir` overrides the checkout location (used by the offline tests).
`inspect-data` remains the separate detailed source-diagnostics command; it
does not apply the benchmark eligibility funnel.

`build-instances` reads each of the four language varieties one at a time from
the same local checkout, constructs one ranking problem per corpus word
occurrence, and reports one compact summary table (with one short progress line
per variety on stderr):

```text
0. all source words
1. words with >=2 candidate rows and exactly one relevance=2 row
2. ... with >=2 distinct (wordform_id, gramset) analyses
3. ... with no repeated candidate identity
4. ... whose selected analysis has a nonempty gramset
5. ... whose sentence links to one unambiguous existing text
6. ... whose entire sentence has only unique, positive word_number values
7. ... where every candidate has a nonempty gramset
```

A word occurrence with any empty `gramset` candidate is excluded from the
primary benchmark as a whole (step 7); an empty gramset on an unselected
candidate that had passed step 4 is reported as a step-7 loss, never counted
twice. Source `candidate_analyses_*.csv.zst` files and `inspect-data` still
preserve and report empty gramsets. For tag `v2026.09` the measured reference
is about 28,400 step-6 instances and a new primary pool of about 28,017; the
command prints the actually computed numbers. The command also writes the
single developer-review file
`data/derived/quality/zero_word_number_sentences_<tag>.csv` with the exact
schema `language,sentence_id`, one row per distinct sentence containing a
`word_number == 0`, sorted deterministically and derived from all source
`words` rows. The output filename's tag is read from the local Git checkout,
and the checked-out source data are never modified. Optional `--data-dir` and
`--output-dir` override the checkout and derived-output locations (used by the
offline tests).

`make-splits` builds the same step-7 primary instances without repeating the
`build-instances` report and assigns every whole text of each language variety
to exactly one of `train` / `dev` / `test`. The assignment minimizes deviation
from a candidate-row target of 80% train, 10% dev, and 10% test per variety,
using a documented fixed seed (`20260928`), a fixed number of seeded restarts,
and deterministic tie-breaking; 80/10/10 is a target, not an exact guarantee.
A single shared CSV `data/derived/splits_<tag>.csv` with the exact schema
`language,text_id,split` is produced, unique and sorted deterministically by
language and numeric `text_id`. If that file already exists with identical
content it is left untouched; if it differs, the command fails rather than
overwriting the shared test assignment. An existing identical file is
preserved; a different one is rejected. The command prints two compact tables
(the benchmark composition and the split distribution with actual candidate
percentages) after all four varieties have been processed successfully. Both
`build-instances` and `make-splits` keep progress to four flushed stderr lines
per command.

## Tests

```bash
python3 -m pytest -q
```

## Data

Corpus data come separately from `dictorpus-data` and are not committed here.
Derived shared inputs will live under `data/derived/`.
