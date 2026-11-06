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

> **Unannotated words.** A word with a candidate whose `gramset_id = 0` is treated
> as unannotated and is excluded from the evaluation set as a whole, even if it has
> other candidates. `gramset_id = 7` is handled the same way: this number is absent
> from the gramset reference table. The `gramset` field of such candidates is empty.

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

For experiments with version `v2026.11` of `dictorpus-data`,
use the published `splits_v2026.11.csv`. There is no need to
regenerate it before training or evaluation.

To generate a split, run:

```bash
python src/cli.py make-splits
```

The source-data version is included in the filename: `splits_<tag>.csv`.

The command makes 200 random attempts for each language. Among attempts whose
total deviation from the 80/10/10 shares is at most 1%, it selects the one with
the larger of the smaller `Neff` values of the `dev` and `test` parts:

```text
Neff = (Σ wᵢ)² / Σ wᵢ²,   wᵢ — the candidate rows of text i
```

`Neff` is the effective number of texts: the smaller it is, the more a part depends
on a few large texts. If no attempt qualifies, the smallest deviation wins. Model
results are not used in this choice.

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
