"""Offline validation of one temporary prediction CSV against a benchmark split.

A prediction file ranks the strict benchmark candidates of the word
occurrences belonging to exactly one split.  This module validates the file
structure, candidate membership and completeness, rank completeness, and the
rank/score agreement without loading the corpus, running Git, reading the
shared split file, or writing anything.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from core.instances import Instance

PREDICTION_HEADER = ("word_id", "wordform_id", "gramset", "rank", "score")

_MAX_ERROR_EXAMPLES = 3


class PredictionFileReadError(RuntimeError):
    """The prediction CSV could not be opened or read."""


class PredictionCsvParseError(RuntimeError):
    """The prediction CSV is syntactically malformed."""


class BenchmarkIntegrityError(RuntimeError):
    """The strict benchmark instances are internally inconsistent."""

_POSITIVE_INTEGER = re.compile(r"[1-9][0-9]*")


def _record_error(
    errors: dict[str, list[str]],
    error_counts: dict[str, int],
    category: str,
    example: str,
) -> None:
    """Append one example to a category, keeping at most three examples."""
    error_counts[category] = error_counts.get(category, 0) + 1
    examples = errors.setdefault(category, [])
    if len(examples) < _MAX_ERROR_EXAMPLES:
        examples.append(example)


def _parse_positive_int(value: str) -> int | None:
    """Parse a strictly positive decimal integer or return None."""
    if _POSITIVE_INTEGER.fullmatch(value) is None:
        return None
    return int(value)


def _parse_score(value: str) -> float | None:
    """Parse a finite number or return None (blank, nonnumeric, NaN, +/-inf)."""
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


@dataclass(frozen=True)
class _PredictionRow:
    """One well-formed prediction row."""

    word_id: int
    wordform_id: int
    gramset: str
    rank: int
    score: float


@dataclass(frozen=True)
class PredictionValidation:
    """Structured result of validating one prediction file for one split."""

    is_valid: bool
    split: str
    expected_word_count: int
    expected_candidate_count: int
    predicted_word_count: int
    predicted_candidate_count: int
    errors: dict[str, tuple[str, ...]]
    error_counts: dict[str, int]


def validate_predictions(
    predictions_path: Path,
    split: str,
    instances: Iterable[Instance],
    split_rows: Iterable[tuple[str, int, str]],
) -> PredictionValidation:
    """Validate one prediction CSV against the strict instances of ``split``.

    ``split_rows`` are ``(language, text_id, split)`` rows in the shape of
    :func:`core.splits.build_split_rows`.  The expected candidates for
    ``split`` are the candidates of every instance whose ``(language,
    text_id)`` is assigned to ``split``.
    """
    instance_list = tuple(instances)
    word_ids = [instance.word_id for instance in instance_list]
    if len(word_ids) != len(set(word_ids)):
        raise BenchmarkIntegrityError(
            "Strict benchmark has duplicate word_id values across instances; "
            "predictions.csv cannot identify candidates unambiguously."
        )
    split_row_tuple = tuple(split_rows)
    split_by_text = {
        (language, int(text_id)): part
        for language, text_id, part in split_row_tuple
    }
    split_texts = {
        key for key, part in split_by_text.items() if part == split
    }

    expected_candidates: dict[int, tuple[tuple[int, str], ...]] = {}
    split_of_word: dict[int, str | None] = {}
    for instance in instance_list:
        key = (instance.language, int(instance.text_id))
        part = split_by_text.get(key)
        split_of_word[instance.word_id] = part
        if key in split_texts:
            expected_candidates[instance.word_id] = tuple(
                (candidate.wordform_id, candidate.gramset)
                for candidate in instance.candidates
            )
    expected_word_ids = set(expected_candidates)
    expected_candidate_count = sum(
        len(candidates) for candidates in expected_candidates.values()
    )

    errors: dict[str, list[str]] = {}
    error_counts: dict[str, int] = {}
    rows: list[_PredictionRow] = []
    rows_by_word: dict[int, list[_PredictionRow]] = {}
    seen_identities: set[tuple[int, int, str]] = set()

    try:
        with open(predictions_path, encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh, strict=True)
            header = next(reader, None)
            if tuple(header) != PREDICTION_HEADER:
                _record_error(
                    errors,
                    error_counts,
                    "header",
                    f"expected {','.join(PREDICTION_HEADER)}; got {header!r}",
                )
            else:
                for raw in reader:
                    if len(raw) != len(PREDICTION_HEADER):
                        _record_error(
                            errors,
                            error_counts,
                            "row width",
                            f"expected {len(PREDICTION_HEADER)} columns; "
                            f"got {len(raw)}: {raw!r}",
                        )
                        continue
                    word_id_text, wordform_id_text, gramset, rank_text, score_text = raw
                    word_id = _parse_positive_int(word_id_text)
                    wordform_id = _parse_positive_int(wordform_id_text)
                    rank = _parse_positive_int(rank_text)
                    score = _parse_score(score_text)
                    if word_id is None:
                        _record_error(
                            errors,
                            error_counts,
                            "invalid word_id",
                            f"word_id={word_id_text!r}",
                        )
                    if wordform_id is None:
                        _record_error(
                            errors,
                            error_counts,
                            "invalid wordform_id",
                            f"wordform_id={wordform_id_text!r}",
                        )
                    if gramset == "":
                        _record_error(
                            errors,
                            error_counts,
                            "empty gramset",
                            f"word_id={word_id_text!r}; gramset={gramset!r}",
                        )
                    elif gramset.strip() == "":
                        _record_error(
                            errors,
                            error_counts,
                            "empty gramset",
                            f"word_id={word_id_text!r}; gramset={gramset!r}",
                        )
                    if rank is None:
                        _record_error(
                            errors,
                            error_counts,
                            "invalid rank",
                            f"rank={rank_text!r}",
                        )
                    if score is None:
                        _record_error(
                            errors,
                            error_counts,
                            "invalid score",
                            f"word_id={word_id_text!r}; score={score_text!r}",
                        )
                    if (
                        word_id is None
                        or wordform_id is None
                        or gramset.strip() == ""
                        or rank is None
                        or score is None
                    ):
                        continue
                    identity = (word_id, wordform_id, gramset)
                    if identity in seen_identities:
                        _record_error(
                            errors,
                            error_counts,
                            "duplicate candidate",
                            f"{identity!r}",
                        )
                    else:
                        seen_identities.add(identity)
                    row = _PredictionRow(word_id, wordform_id, gramset, rank, score)
                    rows.append(row)
                    rows_by_word.setdefault(word_id, []).append(row)
    except csv.Error as exc:
        raise PredictionCsvParseError(
            f"cannot parse predictions CSV: {predictions_path}: {exc}"
        ) from exc
    except OSError as exc:
        raise PredictionFileReadError(
            f"cannot read predictions file: {predictions_path}: {exc}"
        ) from exc

    predicted_word_ids = set(rows_by_word)
    predicted_candidate_count = len(rows)

    for word_id in sorted(expected_word_ids):
        expected_tuples = expected_candidates[word_id]
        word_rows = rows_by_word.get(word_id, [])
        predicted_tuples = {(row.wordform_id, row.gramset) for row in word_rows}
        expected_set = set(expected_tuples)
        for missing in sorted(expected_set - predicted_tuples):
            _record_error(
                errors,
                error_counts,
                "missing candidate",
                f"word_id={word_id}; candidate={missing!r}",
            )
        for unexpected in sorted(predicted_tuples - expected_set):
            _record_error(
                errors,
                error_counts,
                "unexpected candidate",
                f"word_id={word_id}; candidate={unexpected!r}",
            )
        ranks = [row.rank for row in word_rows]
        expected_ranks = set(range(1, len(expected_tuples) + 1))
        if set(ranks) != expected_ranks or len(ranks) != len(set(ranks)):
            _record_error(
                errors,
                error_counts,
                "rank set",
                f"word_id={word_id}; ranks={sorted(ranks)}; "
                f"expected {sorted(expected_ranks)}",
            )
        ordered = sorted(word_rows, key=lambda row: row.rank)
        for first, second in zip(ordered, ordered[1:]):
            if first.score < second.score:
                _record_error(
                    errors,
                    error_counts,
                    "score order",
                    f"word_id={word_id}; rank={first.rank} score={first.score} "
                    f"< rank={second.rank} score={second.score}",
                )

    missing_words = expected_word_ids - predicted_word_ids
    for word_id in sorted(missing_words):
        _record_error(errors, error_counts, "missing word", f"word_id={word_id}")

    for word_id in sorted(predicted_word_ids - expected_word_ids):
        if word_id in split_of_word:
            _record_error(
                errors,
                error_counts,
                "wrong split",
                f"word_id={word_id} belongs to split {split_of_word[word_id]!r}",
            )
        else:
            _record_error(errors, error_counts, "unknown word", f"word_id={word_id}")

    return PredictionValidation(
        is_valid=not errors,
        split=split,
        expected_word_count=len(expected_word_ids),
        expected_candidate_count=expected_candidate_count,
        predicted_word_count=len(predicted_word_ids),
        predicted_candidate_count=predicted_candidate_count,
        errors={category: tuple(examples) for category, examples in errors.items()},
        error_counts=dict(error_counts),
    )