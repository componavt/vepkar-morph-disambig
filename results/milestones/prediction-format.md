🌐 [Read in Russian](prediction-format.ru.md)

# Prediction format

This document defines the prediction-file contract for models evaluated
in the `vepkar-morph-disambig` benchmark.

A prediction file ranks existing candidate morphological analyses of corpus
word occurrences. It does not add candidates, alter source annotations,
or contain gold labels.

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

A candidate is identified by the triple:

```text
(word_id, wordform_id, gramset)
```

The source field `relevance` is not included in prediction files.
It is used only when constructing the benchmark to identify the
expert-selected gold candidate (`relevance = 2`).

## One file, one split

One prediction file contains a complete ranking for one evaluation split:
`dev` or `test`. Splits must not be mixed in one file.

The split is not repeated in each CSV row. It is passed to the validator
as an argument:

```bash
python src/cli.py validate-predictions \
  --predictions /some/path/predictions.csv \
  --split test
```

The validator obtains the evaluated word occurrences from the shared
`splits_<tag>.csv` file and the strict benchmark instances.

Temporary model outputs may be stored anywhere. A selected official result
is stored in `results/milestones/`.

## CSV schema

A prediction file must contain exactly five columns in this order:

```csv
word_id,wordform_id,gramset,rank,score
```

| Column | Type | Meaning |
|---|---|---|
| `word_id` | Positive integer | Identifier of the evaluated corpus word occurrence |
| `wordform_id` | Positive integer | Identifier of the candidate dictionary wordform |
| `gramset` | Nonempty string | Candidate grammatical feature set |
| `rank` | Positive integer | Candidate position; `1` is the best candidate |
| `score` | Finite number | Model-specific candidate score; higher is better |

Example:

```csv
word_id,wordform_id,gramset,rank,score
501,9001,SG+NOM,1,125
501,9001,SG+ACC,2,17
501,9002,PL+GEN,3,3
502,7712,SG+NOM,1,42
502,7712,SG+GEN,2,8
```

This is one file containing predictions for many word occurrences,
not a separate file for each `word_id`.

## Completeness rules

For each evaluated `word_id`, the prediction file must contain every
strict benchmark candidate exactly once.

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
- A candidate not present in the benchmark is added.
- The same triple (word_id, wordform_id, gramset) occurs more than once.
- A word occurrence outside the declared split is included.
```

The file must not contain `relevance`, a gold label, or a new candidate
invented by the model.

## Rank and score

For a word occurrence with `N` candidates, ranks must be exactly:

```text
1, 2, 3, ..., N
```

Invalid ranks:

```text
1, 1, 3    duplicate rank
1, 3, 4    missing rank=2
0, 1, 2    ranks do not start at 1
```

`score` must be finite and consistent with rank:

```text
rank=1 score >= rank=2 score >= rank=3 score >= ...
```

Equal scores are allowed, but ranks must still be unique and complete.
The model must resolve ties deterministically.

Scores of different methods are not directly comparable.
Method quality is compared using ranking metrics.

Other models may use their own internal scores, but their final CSV must
still define a complete, unambiguous ranking.

## Validation

`validate-predictions` checks compatibility with the selected benchmark split:

```text
- Exact header and column order.
- Required types and nonempty gramset values.
- Finite numeric scores.
- Candidate membership in the strict benchmark.
- Complete candidate sets for every word_id.
- No duplicate candidate tuples.
- Membership of every word_id in the declared split.
- Exact ranks 1..N for every word_id.
- Scores do not increase as rank increases.
```

Validation answers:

```text
"Is this file structurally complete and compatible with the benchmark?"
```

It does not answer:

```text
"Is this model accurate?"
```

The `evaluate-predictions` command validates the file and computes
`Top-1 accuracy`, `MRR`, and `Top-3 accuracy`.
[Evaluation procedure](../../docs/evaluation.md).

## Saving results

A validated prediction file may be manually accepted as a significant result.
Its contents must conform to the format described above.

[Result storage and documentation rules](README.md).
