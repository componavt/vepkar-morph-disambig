"""Read and verify one frequency-baseline predictions CSV against an expected baseline.

The caller has already validated the file with :func:`core.predictions.
validate_predictions` for the dev split and has the complete baseline produced
by :func:`core.frequency.rank_by_train_frequency` for the same data and split.
This module only verifies that the file is identical to that baseline, using
exact decimal arithmetic so that no score is ever rounded or compared through
float.  It never touches the corpus, Git, or the filesystem beyond reading the
one CSV file.
"""

from __future__ import annotations

import csv
import decimal
from pathlib import Path
from typing import Iterable

from core.frequency import FrequencyPrediction
from core.predictions import (
    PREDICTION_HEADER,
    PredictionCsvParseError,
    PredictionFileReadError,
)

_Identity = tuple[int, int, str]


class FrequencyBaselineMismatchError(ValueError):
    """The predictions CSV differs from the expected frequency baseline."""


def load_verified_frequency_predictions(
    path: Path,
    expected: Iterable[FrequencyPrediction],
) -> tuple[FrequencyPrediction, ...]:
    """Read a predictions CSV and confirm it matches the expected baseline.

    ``expected`` must be the complete baseline for the same data and split,
    with unique identities and nonnegative integer scores; it is consumed once.
    Returns a tuple of the input rows in input CSV order, with scores as ints.
    """
    expected_rows: dict[_Identity, FrequencyPrediction] = {
        (row.word_id, row.wordform_id, row.gramset): row for row in expected
    }

    rows: list[FrequencyPrediction] = []
    seen_identities: set[_Identity] = set()
    try:
        with open(path, encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh, strict=True)
            header = next(reader, None)
            if header is None or tuple(header) != PREDICTION_HEADER:
                raise PredictionCsvParseError(
                    f"expected header {','.join(PREDICTION_HEADER)}; "
                    f"got {header!r}"
                )
            for raw in reader:
                if len(raw) != len(PREDICTION_HEADER):
                    raise PredictionCsvParseError(
                        f"expected {len(PREDICTION_HEADER)} columns; "
                        f"got {len(raw)}: {raw!r}"
                    )
                word_id_text, wordform_id_text, gramset, rank_text, score_text = raw
                try:
                    word_id = int(word_id_text)
                    wordform_id = int(wordform_id_text)
                    rank = int(rank_text)
                except ValueError as exc:
                    raise PredictionCsvParseError(
                        f"unparseable integer field in {path}: {exc}"
                    ) from exc
                try:
                    score = decimal.Decimal(score_text)
                except (decimal.InvalidOperation, ValueError) as exc:
                    raise PredictionCsvParseError(
                        f"unparseable score {score_text!r} in {path}: {exc}"
                    ) from exc

                identity = (word_id, wordform_id, gramset)
                if identity in seen_identities:
                    raise FrequencyBaselineMismatchError(
                        f"duplicate input identity {identity!r}"
                    )
                seen_identities.add(identity)
                expected_row = expected_rows.get(identity)
                if expected_row is None:
                    raise FrequencyBaselineMismatchError(
                        f"unexpected identity {identity!r}"
                    )
                if not score.is_finite():
                    raise FrequencyBaselineMismatchError(
                        f"identity {identity!r}: score {score_text!r} is not "
                        f"finite; expected {expected_row.score}"
                    )
                if score < 0:
                    raise FrequencyBaselineMismatchError(
                        f"identity {identity!r}: score {score_text!r} is "
                        f"negative; expected {expected_row.score}"
                    )
                if score != score.to_integral_value():
                    raise FrequencyBaselineMismatchError(
                        f"identity {identity!r}: score {score_text!r} is not "
                        f"an integer; expected {expected_row.score}"
                    )
                if score != expected_row.score:
                    raise FrequencyBaselineMismatchError(
                        f"identity {identity!r}: score {score_text!r} does not "
                        f"match expected {expected_row.score}"
                    )
                if rank != expected_row.rank:
                    raise FrequencyBaselineMismatchError(
                        f"identity {identity!r}: rank {rank_text!r} does not "
                        f"match expected {expected_row.rank}"
                    )
                rows.append(
                    FrequencyPrediction(
                        word_id=word_id,
                        wordform_id=wordform_id,
                        gramset=gramset,
                        rank=rank,
                        score=int(score),
                    )
                )
    except csv.Error as exc:
        raise PredictionCsvParseError(
            f"cannot parse predictions CSV: {path}: {exc}"
        ) from exc
    except OSError as exc:
        raise PredictionFileReadError(
            f"cannot read predictions file: {path}: {exc}"
        ) from exc

    missing = sorted(set(expected_rows) - seen_identities)
    if missing:
        key = missing[0]
        expected_row = expected_rows[key]
        raise FrequencyBaselineMismatchError(
            f"missing expected identity {key!r}: expected rank "
            f"{expected_row.rank}, score {expected_row.score}"
        )
    return tuple(rows)