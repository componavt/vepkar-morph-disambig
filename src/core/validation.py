"""Cross-table integrity validation and candidate diagnostics for a corpus.

Fatal problems (duplicates and broken foreign-key joins) raise a compact
aggregated :class:`CorpusError`.  Candidate-group and expert-selection counts
are diagnostics only: they are reported but never treated as errors in this
stage.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.data import CorpusTables


class CorpusError(Exception):
    """Fatal integrity errors found in a corpus export."""


@dataclass(frozen=True)
class CorpusInspection:
    """All requested row, duplicate, orphan, and diagnostic counts."""

    lang: str
    row_texts: int
    row_sentences: int
    row_words: int
    row_candidates: int
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


def _example_string(values: pd.Series) -> str:
    return ", ".join(str(v) for v in values.dropna().unique()[:5].tolist())


def _collect_problems(tables: CorpusTables, lang: str) -> list[str]:
    problems: list[str] = []
    texts, sentences, words, candidates = (
        tables.texts,
        tables.sentences,
        tables.words,
        tables.candidates,
    )

    for label, frame, keys in (
        ("text_id", texts, "text_id"),
        ("sentence_id", sentences, "sentence_id"),
        ("word_id", words, "word_id"),
        ("candidate tuple", candidates, ["word_id", "wordform_id", "gramset"]),
    ):
        mask = frame.duplicated(subset=keys, keep="first")
        count = int(mask.sum())
        if count:
            key_column = keys[0] if isinstance(keys, list) else keys
            examples = _example_string(frame.loc[mask, key_column])
            problems.append(
                f"{lang}: duplicate {label}: {count} duplicate row(s); example key: "
                f"{examples}"
            )

    orphan_sentences = ~sentences["text_id"].isin(texts["text_id"])
    count = int(orphan_sentences.sum())
    if count:
        problems.append(
            f"{lang}: sentences without text: {count} row(s); example text_id: "
            f"{_example_string(sentences.loc[orphan_sentences, 'text_id'])}"
        )

    orphan_words = ~words["sentence_id"].isin(sentences["sentence_id"])
    count = int(orphan_words.sum())
    if count:
        problems.append(
            f"{lang}: words without sentence: {count} row(s); example sentence_id: "
            f"{_example_string(words.loc[orphan_words, 'sentence_id'])}"
        )

    orphan_candidates = ~candidates["word_id"].isin(words["word_id"])
    count = int(orphan_candidates.sum())
    if count:
        problems.append(
            f"{lang}: candidates without word: {count} row(s); example word_id: "
            f"{_example_string(candidates.loc[orphan_candidates, 'word_id'])}"
        )

    return problems


def _candidate_histogram(candidates: pd.DataFrame, word_ids: set[int]) -> tuple[int, int, int]:
    per_word = candidates["word_id"].value_counts()
    zero = sum(1 for word_id in word_ids if word_id not in per_word.index)
    one = int((per_word == 1).sum())
    multiple = int((per_word >= 2).sum())
    return zero, one, multiple


def _selection_histogram(candidates: pd.DataFrame, word_ids: set[int]) -> tuple[int, int, int]:
    selected = candidates.loc[candidates["relevance"] == 2, "word_id"].value_counts()
    zero = sum(1 for word_id in word_ids if word_id not in selected.index)
    one = int((selected == 1).sum())
    multiple = int((selected >= 2).sum())
    return zero, one, multiple


def inspect_corpus(tables: CorpusTables, lang: str) -> CorpusInspection:
    """Validate joins and duplicates; raise :class:`CorpusError` on any fatal error."""
    problems = _collect_problems(tables, lang)
    if problems:
        detail = "\n".join(problems)
        raise CorpusError(
            f"Corpus integrity validation failed for {lang}:\n{detail}"
        )

    word_ids = set(tables.words["word_id"])
    zero_cand, one_cand, multi_cand = _candidate_histogram(tables.candidates, word_ids)
    zero_sel, one_sel, multi_sel = _selection_histogram(tables.candidates, word_ids)

    return CorpusInspection(
        lang=lang,
        row_texts=len(tables.texts),
        row_sentences=len(tables.sentences),
        row_words=len(tables.words),
        row_candidates=len(tables.candidates),
        duplicate_text_id=0,
        duplicate_sentence_id=0,
        duplicate_word_id=0,
        duplicate_candidate_tuple=0,
        orphan_sentences_without_text=0,
        orphan_words_without_sentence=0,
        orphan_candidates_without_word=0,
        words_with_zero_candidates=zero_cand,
        words_with_one_candidate=one_cand,
        words_with_multiple_candidates=multi_cand,
        words_with_zero_selected=zero_sel,
        words_with_one_selected=one_sel,
        words_with_multiple_selected=multi_sel,
    )