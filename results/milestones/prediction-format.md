🌐 [Читать на русском](prediction-format.ru.md)

# Prediction format

This document defines the prediction-file contract for models evaluated in the `vepkar-morph-disambig` benchmark.

A prediction file ranks the already existing candidate morphological analyses of corpus word occurrences. It does not add candidates, alter source annotations, or contain gold labels.

## Scope

The benchmark task is morphological candidate ranking:

```text
word occurrence in a sentence
        +
its existing candidate analyses
        ↓
rank the candidates
        ↓
select wordform_id + gramset
```

A candidate is identified by:

```text
(word_id, wordform_id, gramset)
```

The source-data field `relevance` is not included in prediction files. It is used only during benchmark construction to identify the expert-selected gold candidate (`relevance = 2`).

## One file, one split

One prediction file belongs to exactly one split:

```text
train
 dev
 test
```

The split is not stored in every CSV row. It is supplied to the validator:

```bash
python src/cli.py validate-predictions \
  --predictions /some/path/predictions.csv \
  --split test
```

The validator obtains the evaluated word occurrences from the shared `splits_<tag>.csv` file and the strict benchmark instances.

Temporary model outputs may be stored anywhere. A selected official result is stored in `results/milestones/`.

## CSV schema

A prediction file must have exactly these five columns in this order:

```csv
word_id,wordform_id,gramset,rank,score
```

| Column | Type | Meaning |
|---|---|---|
| `word_id` | Positive integer | Identifier of the evaluated corpus word occurrence |
| `wordform_id` | Positive integer | Identifier of the candidate dictionary wordform |
| `gramset` | Nonempty string | Candidate grammatical feature set |
| `rank` | Positive integer | Candidate position; `1` is the best-ranked candidate |
| `score` | Finite number | Model-specific candidate score; a higher score is better |

Example:

```csv
word_id,wordform_id,gramset,rank,score
501,9001,SG+NOM,1,125
501,9001,SG+ACC,2,17
501,9002,PL+GEN,3,3
502,7712,SG+NOM,1,42
502,7712,SG+GEN,2,8
```

This is one file containing predictions for multiple word occurrences. It is not one file per `word_id`.

## Completeness rules

For every evaluated `word_id`, the prediction file must include every candidate in the strict benchmark exactly once.

```text
Strict benchmark, word_id=501:
  A = (9001, SG+NOM)
  B = (9001, SG+ACC)
  C = (9002, PL+GEN)

Prediction file for word_id=501:
  A, B, C
```

The following are invalid:

```text
- Only the top-ranked candidate is present.
- A benchmark candidate is missing.
- An extra candidate not present in the benchmark is added.
- The same (word_id, wordform_id, gramset) occurs more than once.
- A word occurrence outside the declared split is included.
```

A file must not contain `relevance`, a gold label, or a new candidate invented by the model.

## Rank and score

For a word occurrence with `N` candidates, ranks must be exactly:

```text
1, 2, 3, ..., N
```

Invalid ranks include:

```text
1, 1, 3    duplicate rank
1, 3, 4    missing rank 2
0, 1, 2    rank does not start at 1
```

Scores must be finite numbers and must agree with ranks:

```text
rank=1 score >= rank=2 score >= rank=3 score >= ...
```

Equal scores are allowed, but ranks must still be unique and complete. The model must resolve a tie deterministically.

Scores are meaningful only inside one model run. Do not compare raw scores between different models:

```text
frequency baseline score = 125
neural model score       = 0.91

125 > 0.91 does not mean that the baseline is better.
```

Compare models using ranks and evaluation metrics, not their raw scores.

The first train-frequency baseline uses this deterministic tie-break rule:

```text
1. Smaller wordform_id ranks higher.
2. If wordform_id is equal, lexicographically smaller gramset ranks higher.
```

Other models may use their own internal scoring methods, but their final CSV must still provide a complete, unambiguous ranking.

## Validation

`validate-predictions` checks that a file is compatible with the selected benchmark split. It validates:

```text
- Exact header and column order.
- Required types and nonempty gramset values.
- Finite numeric scores.
- Candidate membership in the strict benchmark.
- Completeness of each word's candidate set.
- Absence of duplicate candidate tuples.
- Membership of every word_id in the declared split.
- Exact ranks 1..N for every word_id.
- Nonincreasing scores as rank worsens.
```

Validation answers:

```text
"Is this file structurally complete and compatible with the benchmark?"
```

It does not answer:

```text
"Is this model accurate?"
```

Accuracy, MRR, Top-k measures, and model comparison are calculated separately.

## Milestone files

A milestone is a prediction file selected manually as a meaningful, reproducible result. Creation and validation do not automatically make a file a milestone.

The accepted `run_id` pattern is:

```text
YYYY-MM-DD__author__task__method
```

Allowed task values are:

```text
core
t1-features
t2-context
t3-transfer
```

A saved milestone filename is:

```text
<run_id>__predictions.csv
```

Example:

```text
2026-09-29__andrew__core__train-frequency__predictions.csv
```

Milestone prediction files normally represent the final evaluation split (`test`). Before saving a milestone, validate it explicitly:

```bash
python src/cli.py validate-predictions \
  --predictions results/milestones/2026-09-29__andrew__core__train-frequency__predictions.csv \
  --split test
```

Use the same shared split and this format when comparing the work of different authors and models.
