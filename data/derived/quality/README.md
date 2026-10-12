🌐 [Read in Russian](README.ru.md)

# Manual review queue

Files named `zero_word_number_sentences_<tag>.csv` list sentences for manual review.
Each sentence is included because at least one of its word records has
`word_number=0`.

CSV columns:

```csv
language,sentence_id
```

Each row identifies a sentence; the CSV does not contain its text or XML.

Illustrative example: the record for the word “cat” has a zero position,
`word_number=0`.

Rows of the `words` table:

```csv
word_id,sentence_id,word_number,word
101,10,1,the
102,10,0,cat
103,10,3,grins
```

XML of the same sentence:

```xml
<s id="1"><w id="41">The</w> <w id="42">cat</w> <w id="43">grins</w>.</s>
```

The sentence with `sentence_id=10` is included in the queue because one
of its word records has `word_number=0`. Its `word_id` and the `id`
attribute of the `<w>` element are both positive.

`word_number` is the word position in the corpus table, whereas the `id`
attribute of `<w>` is its XML identifier. Queue membership is determined
by a zero `word_number`, not by a zero `id` attribute in XML.

This is an illustrative example, not an actual corpus sentence.
To determine the cause of the zero position, compare the word records
with the original sentence XML; queue membership alone does not
establish the cause.

These lists are intended for manual review, not for training or evaluating
models.
