"""Construction of reproducible morphological-ranking benchmark instances.

Each instance is one occurrence of a corpus word (its ``word_id``) together
with the distinct candidate morphological analyses and exactly one
expert-selected ``relevance == 2`` analysis.  The sequential eligibility
funnel, overlap diagnostics, and the developer sentence-review pairs are
computed from the typed corpus tables without modifying the source checkout.
"""

from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from core.data import CorpusTables

GOLD_RELEVANCE = 2

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "derived" / "quality"
)

REVIEW_CSV_PREFIX = "zero_word_number_sentences"
REVIEW_CSV_HEADER = ("language", "sentence_id")

_FUNNEL_NAMES = (
    "all source words",
    ">=2 candidate rows, exactly one relevance=2",
    ">=2 distinct analyses",
    "no repeated candidate identity",
    "selected analysis has nonempty gramset",
    "sentence and text uniquely identified",
    "sentence positions unique and positive",
)


class CorpusTagError(Exception):
    """The source-data checkout tag could not be determined unambiguously."""


@dataclass(frozen=True)
class Candidate:
    """One distinct morphological analysis of a word."""

    wordform_id: int
    gramset: str


@dataclass(frozen=True)
class Instance:
    """One usable benchmark instance for a single word occurrence."""

    language: str
    word_id: int
    sentence_id: int
    text_id: int
    word: str
    word_number: int
    sentence_xml: str
    candidates: tuple[Candidate, ...]
    gold_analysis: Candidate


@dataclass(frozen=True)
class FunnelStep:
    """One stage of the sequential word-instance eligibility funnel."""

    index: int
    name: str
    retained: int
    removed: int


@dataclass(frozen=True)
class LanguageInstances:
    """All funnel, overlap, and instance outputs for one language variety."""

    language: str
    funnel: tuple[FunnelStep, ...]
    instances: tuple[Instance, ...]
    zero_position_sentences: frozenset[int]
    repeated_position_sentences: frozenset[int]
    step1_words_with_duplicate_identity: int
    step1_words_with_selected_empty_gramset: int
    step1_words_with_unselected_empty_gramset: int
    step1_words_with_unavailable_sentence: int
    step1_words_with_zero_position_sentence: int
    step1_words_with_repeated_position_sentence: int
    primary_words_with_unselected_empty_gramset: int

    @property
    def alternative_primary_pool_size(self) -> int:
        """Primary pool size if entire words with unselected empty gramsets
        were excluded.  Diagnostic only; the primary pool is not changed."""
        return len(self.instances) - self.primary_words_with_unselected_empty_gramset


def determine_data_tag(data_dir: Path) -> str:
    """Return the single Git tag pointing at HEAD of the source checkout."""
    try:
        result = subprocess.run(
            ["git", "-C", str(data_dir), "tag", "--points-at", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise CorpusTagError("Git is not installed or is not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or str(exc)
        raise CorpusTagError(
            f"Could not read the dictorpus-data checkout tag from {data_dir}: {detail}"
        ) from exc
    tags = sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})
    if not tags:
        raise CorpusTagError(
            f"No Git tag points at HEAD in {data_dir}; cannot name derived outputs"
        )
    if len(tags) > 1:
        raise CorpusTagError(
            f"Multiple Git tags point at HEAD in {data_dir}: {', '.join(tags)}"
        )
    return tags[0]


def build_language_instances(language: str, tables: CorpusTables) -> LanguageInstances:
    """Build instances and all funnel diagnostics for one language variety."""
    words, sentences, texts, candidates = (
        tables.words,
        tables.sentences,
        tables.texts,
        tables.candidates,
    )

    word_occurrence: dict[int, int] = {}
    word_sentence: dict[int, int] = {}
    word_number: dict[int, int] = {}
    word_text_token: dict[int, str] = {}
    for word_id, sentence_id, number, token in zip(
        words["word_id"], words["sentence_id"], words["word_number"], words["word"]
    ):
        word_id, sentence_id, number = int(word_id), int(sentence_id), int(number)
        word_occurrence[word_id] = word_occurrence.get(word_id, 0) + 1
        word_sentence.setdefault(word_id, sentence_id)
        word_number.setdefault(word_id, number)
        word_text_token.setdefault(word_id, token)
    ambiguous_word_ids = {
        word_id for word_id, count in word_occurrence.items() if count > 1
    }

    sentence_count: dict[int, int] = {}
    sentence_text: dict[int, int] = {}
    sentence_xml: dict[int, str] = {}
    for sentence_id, text_id, xml in zip(
        sentences["sentence_id"], sentences["text_id"], sentences["sentence_xml"]
    ):
        sentence_id, text_id = int(sentence_id), int(text_id)
        sentence_count[sentence_id] = sentence_count.get(sentence_id, 0) + 1
        sentence_text.setdefault(sentence_id, text_id)
        sentence_xml.setdefault(sentence_id, xml)
    text_count: dict[int, int] = {}
    for text_id in texts["text_id"]:
        text_id = int(text_id)
        text_count[text_id] = text_count.get(text_id, 0) + 1

    def context_ok(word_id: int) -> bool:
        if word_id in ambiguous_word_ids:
            return False
        sentence_id = word_sentence[word_id]
        if sentence_count.get(sentence_id, 0) != 1:
            return False
        if text_count.get(sentence_text[sentence_id], 0) != 1:
            return False
        return True

    zero_position_sentences = frozenset(
        int(sentence_id)
        for sentence_id, number in zip(words["sentence_id"], words["word_number"])
        if int(number) == 0
    )

    repeated_position_sentences: set[int] = set()
    if len(words):
        for sentence_id, group in words.groupby("sentence_id"):
            positive = [int(number) for number in group["word_number"] if int(number) > 0]
            if len(positive) != len(set(positive)):
                repeated_position_sentences.add(int(sentence_id))
    repeated_position_sentences = frozenset(repeated_position_sentences)

    rows_per_word = candidates["word_id"].value_counts()
    selected = candidates[candidates["relevance"] == GOLD_RELEVANCE]
    selected_per_word = selected["word_id"].value_counts()
    identity_counts = candidates.groupby(
        ["word_id", "wordform_id", "gramset"]
    ).size()
    distinct_per_word = identity_counts.groupby(level="word_id").size()
    dup_mask = candidates.duplicated(
        subset=["word_id", "wordform_id", "gramset"], keep=False
    )
    duplicate_identity_words = set(
        int(word_id) for word_id in candidates.loc[dup_mask, "word_id"].unique()
    )
    selected_empty_words = set(
        int(word_id)
        for word_id in selected.loc[selected["gramset"] == "", "word_id"].unique()
    )
    unselected_empty_counts = candidates.loc[
        (candidates["relevance"] != GOLD_RELEVANCE) & (candidates["gramset"] == ""),
        "word_id",
    ].value_counts()

    step0 = set(word_occurrence)
    step1 = {
        word_id
        for word_id in step0
        if rows_per_word.get(word_id, 0) >= 2
        and selected_per_word.get(word_id, 0) == 1
    }
    step2 = {
        word_id for word_id in step1 if distinct_per_word.get(word_id, 0) >= 2
    }
    step3 = {word_id for word_id in step2 if word_id not in duplicate_identity_words}
    step4 = {word_id for word_id in step3 if word_id not in selected_empty_words}
    step5 = {word_id for word_id in step4 if context_ok(word_id)}
    invalid_sentence_ids = zero_position_sentences | repeated_position_sentences
    step6 = {
        word_id
        for word_id in step5
        if word_sentence[word_id] not in invalid_sentence_ids
    }

    funnel: list[FunnelStep] = [FunnelStep(0, _FUNNEL_NAMES[0], len(step0), 0)]
    funnel_sets = [step0, step1, step2, step3, step4, step5, step6]
    for index in range(1, len(funnel_sets)):
        retained = len(funnel_sets[index])
        removed = len(funnel_sets[index - 1]) - retained
        funnel.append(FunnelStep(index, _FUNNEL_NAMES[index], retained, removed))

    step1_words_with_duplicate_identity = sum(
        1 for word_id in step1 if word_id in duplicate_identity_words
    )
    step1_words_with_selected_empty_gramset = sum(
        1 for word_id in step1 if word_id in selected_empty_words
    )
    step1_words_with_unselected_empty_gramset = sum(
        1 for word_id in step1 if unselected_empty_counts.get(word_id, 0) >= 1
    )
    step1_words_with_unavailable_sentence = sum(
        1 for word_id in step1 if not context_ok(word_id)
    )
    step1_words_with_zero_position_sentence = sum(
        1 for word_id in step1 if word_sentence[word_id] in zero_position_sentences
    )
    step1_words_with_repeated_position_sentence = sum(
        1
        for word_id in step1
        if word_sentence[word_id] in repeated_position_sentences
    )
    primary_words_with_unselected_empty_gramset = sum(
        1 for word_id in step6 if unselected_empty_counts.get(word_id, 0) >= 1
    )

    final_rows = candidates[candidates["word_id"].isin(step6)]
    instances: list[Instance] = []
    for word_id in sorted(step6):
        word_rows = final_rows[final_rows["word_id"] == word_id]
        identities = sorted(
            {
                (int(r.wordform_id), r.gramset)
                for r in word_rows.itertuples()
            },
            key=lambda pair: (pair[0], pair[1]),
        )
        gold = next(
            Candidate(int(r.wordform_id), r.gramset)
            for r in word_rows.itertuples()
            if int(r.relevance) == GOLD_RELEVANCE
        )
        sentence_id = word_sentence[word_id]
        instances.append(
            Instance(
                language=language,
                word_id=word_id,
                sentence_id=sentence_id,
                text_id=sentence_text[sentence_id],
                word=word_text_token[word_id],
                word_number=word_number[word_id],
                sentence_xml=sentence_xml[sentence_id],
                candidates=tuple(
                    Candidate(wordform_id, gramset)
                    for wordform_id, gramset in identities
                ),
                gold_analysis=gold,
            )
        )

    return LanguageInstances(
        language=language,
        funnel=tuple(funnel),
        instances=tuple(instances),
        zero_position_sentences=zero_position_sentences,
        repeated_position_sentences=repeated_position_sentences,
        step1_words_with_duplicate_identity=step1_words_with_duplicate_identity,
        step1_words_with_selected_empty_gramset=step1_words_with_selected_empty_gramset,
        step1_words_with_unselected_empty_gramset=step1_words_with_unselected_empty_gramset,
        step1_words_with_unavailable_sentence=step1_words_with_unavailable_sentence,
        step1_words_with_zero_position_sentence=step1_words_with_zero_position_sentence,
        step1_words_with_repeated_position_sentence=step1_words_with_repeated_position_sentence,
        primary_words_with_unselected_empty_gramset=primary_words_with_unselected_empty_gramset,
    )


def review_csv_path(output_dir: Path, tag: str) -> Path:
    """Return the version-specific developer review CSV path."""
    return output_dir / f"{REVIEW_CSV_PREFIX}_{tag}.csv"


def build_review_rows(
    results: Iterable[LanguageInstances],
) -> list[tuple[str, int]]:
    """Return sorted unique ``(language, sentence_id)`` rows for the CSV."""
    pairs: set[tuple[str, int]] = set()
    for result in results:
        pairs.update((result.language, int(sentence_id)) for sentence_id in result.zero_position_sentences)
    return sorted(pairs, key=lambda pair: (pair[0], pair[1]))


def write_review_csv(path: Path, rows: Iterable[tuple[str, int]]) -> int:
    """Write the developer review CSV; return the number of data rows."""
    unique = sorted(
        {(lang, int(sentence_id)) for lang, sentence_id in rows},
        key=lambda pair: (pair[0], pair[1]),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(REVIEW_CSV_HEADER)
        writer.writerows(unique)
    return len(unique)