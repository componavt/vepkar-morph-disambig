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

Illustrative example of a sentence XML with a `<w id="0">` element (zero ID):

```xml
<s id="1"><w id="1">Этот</w> <w id="0">кот</w> <w id="2">спит</w>.</s>
```

This is a simplified illustration, not an actual corpus sentence. During review,
a word record must be matched against the sentence XML; queue membership alone
does not explain the cause of `word_number=0`.

These lists are intended for manual review, not for training or evaluating
models.
