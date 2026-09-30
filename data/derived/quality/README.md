# Manual review queue

Files named `zero_word_number_sentences_<tag>.csv` list sentences for manual review.
Each sentence is included because at least one of its word records has
`word_number=0`.

CSV columns:

```csv
language,sentence_id
```

Each row identifies a sentence; the CSV does not contain its text or XML.

Illustrative `sentence_xml` with one word marked by a zero ID:

```xml
<s id="1"><w id="1">The</w> <w id="0">cat</w> <w id="2">sleeps</w>.</s>
```

This is a simplified example, not an actual corpus sentence. Review the affected
word record together with the sentence XML; queue membership alone does not
establish the cause of `word_number=0`.

These files are review queues, not training data for the frequency baseline.
