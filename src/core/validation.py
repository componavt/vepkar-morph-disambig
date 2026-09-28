"""Cross-table integrity inspection and candidate diagnostics for a corpus.

All checks that can be completed on the four readable tables are performed and
returned as counts inside :class:`CorpusInspection`; duplicates and broken
foreign-key joins are reported instead of raising.  Zero word positions and
empty gramsets are preserved and reported as findings, not treated as fatal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd

from core.data import CorpusTables
from core.instances import GOLD_RELEVANCE


class CorpusError(Exception):
    """A corpus export could not be inspected completely."""


@dataclass(frozen=True)
class Example:
    """One display-ready contextual example for a reported finding class."""

    detail: str
    sentence_xml: str


@dataclass(frozen=True)
class CorpusInspection:
    """All requested row, duplicate, orphan, and diagnostic counts."""

    lang: str
    row_texts: int
    row_sentences: int
    row_words: int
    row_candidates: int
    zero_word_number: int
    zero_word_number_sentences: int
    empty_gramset: int
    empty_gramset_relevance_0: int
    empty_gramset_relevance_1: int
    empty_gramset_relevance_2: int
    duplicate_text_id: int
    duplicate_sentence_id: int
    duplicate_word_id: int
    duplicate_candidate_tuple: int
    orphan_sentences_without_text: int
    orphan_words_without_sentence: int
    orphan_candidates_without_word: int
    words_with_zero_candidates: int
    words_with_one_candidate: int
    words_with_multiple_candidates: int
    words_with_zero_selected: int
    words_with_one_selected: int
    words_with_multiple_selected: int
    examples: dict[str, tuple[Example, ...]]

    @property
    def has_broken_integrity(self) -> bool:
        """True when duplicate keys or broken parent links are present."""
        return any(
            (
                self.duplicate_text_id,
                self.duplicate_sentence_id,
                self.duplicate_word_id,
                self.duplicate_candidate_tuple,
                self.orphan_sentences_without_text,
                self.orphan_words_without_sentence,
                self.orphan_candidates_without_word,
            )
        )


def _sentence_xml_lookup(sentences: pd.DataFrame) -> dict[int, tuple[int, str]]:
    """Map sentence_id to (occurrence count, first original sentence_xml)."""
    lookup: dict[int, tuple[int, str]] = {}
    for sentence_id, xml in zip(sentences["sentence_id"], sentences["sentence_xml"]):
        sentence_id = int(sentence_id)
        if sentence_id in lookup:
            count, first = lookup[sentence_id]
            lookup[sentence_id] = (count + 1, first)
        else:
            lookup[sentence_id] = (1, xml)
    return lookup


def _word_sentence_lookup(words: pd.DataFrame) -> dict[int, tuple[int, int]]:
    """Map word_id to (occurrence count, first sentence_id)."""
    lookup: dict[int, tuple[int, int]] = {}
    for word_id, sentence_id in zip(words["word_id"], words["sentence_id"]):
        word_id = int(word_id)
        sentence_id = int(sentence_id)
        if word_id in lookup:
            count, first = lookup[word_id]
            lookup[word_id] = (count + 1, first)
        else:
            lookup[word_id] = (1, sentence_id)
    return lookup


def _word_text_lookup(words: pd.DataFrame) -> dict[int, tuple[int, str]]:
    """Map word_id to (occurrence count, first word text)."""
    lookup: dict[int, tuple[int, str]] = {}
    for word_id, word in zip(words["word_id"], words["word"]):
        word_id = int(word_id)
        if word_id in lookup:
            count, first = lookup[word_id]
            lookup[word_id] = (count + 1, first)
        else:
            lookup[word_id] = (1, word)
    return lookup


def _word_text_for_id(
    word_id: int, word_lookup: dict[int, tuple[int, str]]
) -> str:
    """Return the word text or an explicit unavailability marker."""
    entry = word_lookup.get(word_id)
    if entry is None:
        return "<unavailable>"
    count, word = entry
    if count > 1:
        return "<unavailable>"
    return word


def _sentence_xml_for_id(
    sentence_id: int, lookup: dict[int, tuple[int, str]]
) -> str:
    """Return original sentence XML or an explicit unavailability reason."""
    entry = lookup.get(sentence_id)
    if entry is None:
        return f"unavailable — sentence_id={sentence_id} absent from sentences"
    count, xml = entry
    if count > 1:
        return f"unavailable — sentence_id={sentence_id} is ambiguous"
    return xml


def _sentence_xml_for_candidate(
    word_id: int,
    word_lookup: dict[int, tuple[int, int]],
    sentence_lookup: dict[int, tuple[int, str]],
) -> str:
    entry = word_lookup.get(word_id)
    if entry is None:
        return f"unavailable — word_id={word_id} absent from words"
    count, sentence_id = entry
    if count > 1:
        return f"unavailable — word_id={word_id} is ambiguous"
    return _sentence_xml_for_id(sentence_id, sentence_lookup)


def _examples(
    detail: Iterator[str], sentence_xml: Iterator[str]
) -> tuple[Example, ...]:
    return tuple(
        Example(detail=fragment, sentence_xml=xml)
        for fragment, xml in zip(detail, sentence_xml, strict=True)
    )


def _candidate_histogram(
    candidates: pd.DataFrame, word_ids: set[int]
) -> tuple[int, int, int]:
    per_word = candidates["word_id"].value_counts()
    zero = sum(1 for word_id in word_ids if word_id not in per_word.index)
    one = int((per_word == 1).sum())
    multiple = int((per_word >= 2).sum())
    return zero, one, multiple


def _selection_histogram(
    candidates: pd.DataFrame, word_ids: set[int]
) -> tuple[int, int, int]:
    selected = candidates.loc[candidates["relevance"] == 2, "word_id"].value_counts()
    zero = sum(1 for word_id in word_ids if word_id not in selected.index)
    one = int((selected == 1).sum())
    multiple = int((selected >= 2).sum())
    return zero, one, multiple


def _word_examples(
    rows: pd.DataFrame, sentence_lookup: dict[int, tuple[int, str]]
) -> tuple[Example, ...]:
    return _examples(
        (
            f"word_id={int(r.word_id)}; sentence_id={int(r.sentence_id)}; "
            f"word={r.word}"
            for r in rows.itertuples()
        ),
        (
            _sentence_xml_for_id(int(r.sentence_id), sentence_lookup)
            for r in rows.itertuples()
        ),
    )


def _pick_empty_gramset_example(rows: pd.DataFrame) -> pd.DataFrame:
    """Select one informative empty-gramset row: gold if present, else first."""
    gold = rows.loc[rows["relevance"] == GOLD_RELEVANCE]
    return gold.head(1) if len(gold) else rows.head(1)


def _candidate_examples(
    rows: pd.DataFrame,
    word_lookup: dict[int, tuple[int, int]],
    sentence_lookup: dict[int, tuple[int, str]],
    word_text_lookup: dict[int, tuple[int, str]],
) -> tuple[Example, ...]:
    return _examples(
        (
            f"word_id={int(r.word_id)}; wordform_id={int(r.wordform_id)}; "
            f"word={_word_text_for_id(int(r.word_id), word_text_lookup)}; "
            f"relevance={int(r.relevance)}"
            for r in rows.itertuples()
        ),
        (
            _sentence_xml_for_candidate(
                int(r.word_id), word_lookup, sentence_lookup
            )
            for r in rows.itertuples()
        ),
    )


def inspect_corpus(tables: CorpusTables, lang: str) -> CorpusInspection:
    """Inspect all four tables and return every finding as counts."""
    texts, sentences, words, candidates = (
        tables.texts,
        tables.sentences,
        tables.words,
        tables.candidates,
    )
    sentence_lookup = _sentence_xml_lookup(sentences)
    word_lookup = _word_sentence_lookup(words)
    word_text_lookup = _word_text_lookup(words)

    zero_mask = words["word_number"] == 0
    zero_word_number = int(zero_mask.sum())
    zero_word_number_sentences = int(words.loc[zero_mask, "sentence_id"].nunique())

    empty_gramset_mask = candidates["gramset"] == ""
    empty_gramset = int(empty_gramset_mask.sum())
    empty_rel = candidates.loc[empty_gramset_mask, "relevance"]
    empty_gramset_relevance_0 = int((empty_rel == 0).sum())
    empty_gramset_relevance_1 = int((empty_rel == 1).sum())
    empty_gramset_relevance_2 = int((empty_rel == 2).sum())

    duplicate_text_id = int(texts.duplicated(subset="text_id", keep="first").sum())
    duplicate_sentence_id = int(
        sentences.duplicated(subset="sentence_id", keep="first").sum()
    )
    duplicate_word_id = int(words.duplicated(subset="word_id", keep="first").sum())
    duplicate_candidate_tuple = int(
        candidates.duplicated(
            subset=("word_id", "wordform_id", "gramset"), keep="first"
        ).sum()
    )

    orphan_sentences_mask = ~sentences["text_id"].isin(texts["text_id"])
    orphan_sentences_without_text = int(orphan_sentences_mask.sum())
    orphan_words_mask = ~words["sentence_id"].isin(sentences["sentence_id"])
    orphan_words_without_sentence = int(orphan_words_mask.sum())
    orphan_candidates_mask = ~candidates["word_id"].isin(words["word_id"])
    orphan_candidates_without_word = int(orphan_candidates_mask.sum())

    examples: dict[str, tuple[Example, ...]] = {}

    if zero_word_number:
        examples["word_number=0"] = _word_examples(
            words.loc[zero_mask].head(1), sentence_lookup
        )

    if empty_gramset:
        examples["empty gramset"] = _candidate_examples(
            _pick_empty_gramset_example(candidates.loc[empty_gramset_mask]),
            word_lookup,
            sentence_lookup,
            word_text_lookup,
        )

    if duplicate_text_id:
        dup = texts.duplicated(subset="text_id", keep="first")
        rows = texts.loc[dup].head(1)
        examples["duplicate text_id"] = _examples(
            (f"text_id={int(r.text_id)}" for r in rows.itertuples()),
            (
                f"unavailable — text_id={int(r.text_id)} is a duplicate key"
                for r in rows.itertuples()
            ),
        )

    if duplicate_sentence_id:
        dup = sentences.duplicated(subset="sentence_id", keep="first")
        rows = sentences.loc[dup].head(1)
        examples["duplicate sentence_id"] = _examples(
            (f"sentence_id={int(r.sentence_id)}" for r in rows.itertuples()),
            (
                f"unavailable — sentence_id={int(r.sentence_id)} is a duplicate key"
                for r in rows.itertuples()
            ),
        )

    if duplicate_word_id:
        dup = words.duplicated(subset="word_id", keep="first")
        rows = words.loc[dup].head(1)
        examples["duplicate word_id"] = _examples(
            (f"word_id={int(r.word_id)}" for r in rows.itertuples()),
            (
                f"unavailable — word_id={int(r.word_id)} is a duplicate key"
                for r in rows.itertuples()
            ),
        )

    if duplicate_candidate_tuple:
        dup = candidates.duplicated(
            subset=("word_id", "wordform_id", "gramset"), keep="first"
        )
        rows = candidates.loc[dup].head(1)
        examples["duplicate candidate tuple"] = _examples(
            (
                f"word_id={int(r.word_id)}; wordform_id={int(r.wordform_id)}; "
                f"gramset={r.gramset!r}"
                for r in rows.itertuples()
            ),
            (
                "unavailable — candidate tuple "
                f"({int(r.word_id)}, {int(r.wordform_id)}, {r.gramset!r}) "
                "is a duplicate key"
                for r in rows.itertuples()
            ),
        )

    if orphan_sentences_without_text:
        rows = sentences.loc[orphan_sentences_mask].head(1)
        examples["sentences without text"] = _examples(
            (
                f"sentence_id={int(r.sentence_id)}; text_id={int(r.text_id)}"
                for r in rows.itertuples()
            ),
            (
                _sentence_xml_for_id(int(r.sentence_id), sentence_lookup)
                for r in rows.itertuples()
            ),
        )

    if orphan_words_without_sentence:
        examples["words without sentence"] = _word_examples(
            words.loc[orphan_words_mask].head(1), sentence_lookup
        )

    if orphan_candidates_without_word:
        rows = candidates.loc[orphan_candidates_mask].head(1)
        examples["candidates without word"] = _candidate_examples(
            rows, word_lookup, sentence_lookup, word_text_lookup
        )

    word_ids = set(words["word_id"])
    zero_cand, one_cand, multi_cand = _candidate_histogram(candidates, word_ids)
    zero_sel, one_sel, multi_sel = _selection_histogram(candidates, word_ids)

    return CorpusInspection(
        lang=lang,
        row_texts=len(texts),
        row_sentences=len(sentences),
        row_words=len(words),
        row_candidates=len(candidates),
        zero_word_number=zero_word_number,
        zero_word_number_sentences=zero_word_number_sentences,
        empty_gramset=empty_gramset,
        empty_gramset_relevance_0=empty_gramset_relevance_0,
        empty_gramset_relevance_1=empty_gramset_relevance_1,
        empty_gramset_relevance_2=empty_gramset_relevance_2,
        duplicate_text_id=duplicate_text_id,
        duplicate_sentence_id=duplicate_sentence_id,
        duplicate_word_id=duplicate_word_id,
        duplicate_candidate_tuple=duplicate_candidate_tuple,
        orphan_sentences_without_text=orphan_sentences_without_text,
        orphan_words_without_sentence=orphan_words_without_sentence,
        orphan_candidates_without_word=orphan_candidates_without_word,
        words_with_zero_candidates=zero_cand,
        words_with_one_candidate=one_cand,
        words_with_multiple_candidates=multi_cand,
        words_with_zero_selected=zero_sel,
        words_with_one_selected=one_sel,
        words_with_multiple_selected=multi_sel,
        examples=examples,
    )