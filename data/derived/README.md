🌐 [Read in Russian](README.ru.md)

# Derived data

This directory contains data derived from the version of
[dictorpus-data](https://github.com/componavt/dictorpus-data/)
identified in each filename. The source exports are not stored here.

## Strict evaluation set

The set includes word occurrences with at least two distinct candidates
`(wordform_id, gramset)` and exactly one expert selection
(`relevance = 2`). Duplicate candidate records are not allowed;
every candidate must have a nonempty `gramset`.

The word occurrence, its sentence, and its text must be uniquely
identified by the source identifiers. The entire sentence is excluded
if any of its words has `word_number = 0` or if positive word positions
are repeated.

If a candidate is unsuitable, the entire word occurrence is excluded
rather than reducing its list of alternatives. The source data
are not modified.

## Building the evaluation set and manual review

The following command builds the strict set of word occurrences
using the criteria above and writes a queue of sentences containing
a word with `word_number = 0` for manual review:

```bash
python src/cli.py build-instances
```

Run the command from the repository root after obtaining the source data.
[Manual review queue](quality/README.md).

## Train/dev/test split

For experiments with version `v2026.09` of `dictorpus-data`,
use the published `splits_v2026.09.csv`. There is no need to
regenerate it before training or evaluation.

To generate a split, run:

```bash
python src/cli.py make-splits
```

The source-data version is included in the filename: `splits_<tag>.csv`.

The file has three columns:

| Column | Meaning |
|---|---|
| `language` | `vep`, `krl`, `olo`, or `lud` |
| `text_id` | Identifier of a text in the corresponding corpus export |
| `split` | `train`, `dev`, or `test` |

Each row assigns an **entire text** to one split. All eligible word occurrences
in that text inherit its assignment through the `(language, text_id)` pair.

The target for each language variety is 80% / 10% / 10% by the number of
candidate rows. The proportions by number of texts or word occurrences may
differ because a text cannot be split across parts.

- `train` — train a model;
- `dev` — select its settings and variant;
- `test` — evaluate it after selection.

Use the same split file when comparing methods.

The command reports `created` when it creates the file. If an existing
file has the same content, it reports `unchanged` and does not overwrite
it. An existing file with different content causes the command to fail.
