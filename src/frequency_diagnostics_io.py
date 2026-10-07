"""Read, verify, and write frequency-baseline diagnostic CSVs.

The caller has already validated the file with :func:`core.predictions.
validate_predictions` for the dev split and has the complete baseline produced
by :func:`core.frequency.rank_by_train_frequency` for the same data and split.
This module verifies that the file is identical to that baseline, using exact
decimal arithmetic so that no score is ever rounded or compared through float,
and writes diagnostic rows safely without overwriting an existing file.  It
never touches the corpus or Git.
"""

from __future__ import annotations

import csv
import decimal
import os
import tempfile
from dataclasses import astuple, fields
from pathlib import Path
from typing import Iterable

from core.diagnostics import FrequencyDiagnostic
from core.frequency import FrequencyPrediction
from core.predictions import (
    PREDICTION_HEADER,
    PredictionCsvParseError,
    PredictionFileReadError,
)

_DIAGNOSTICS_HEADER = tuple(field.name for field in fields(FrequencyDiagnostic))

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


def write_frequency_diagnostics(
    path: Path, rows: Iterable[FrequencyDiagnostic]
) -> int:
    """Write diagnostic rows unchanged to ``path`` and return the row count.

    The destination must not already exist (files, directories, and symlinks
    including broken ones are all rejected) and its parent directory must
    already exist; nothing is created besides the CSV itself.  The rows are
    written to a sibling temporary file that is then hard-linked into place, so
    an existing destination is never overwritten or replaced.
    """
    if path.is_symlink() or path.exists():
        raise FileExistsError(f"output path already exists: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"output path is a directory: {path}")
    parent = path.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {parent}")
    descriptor, temp_name = tempfile.mkstemp(
        prefix=".vepkar-diagnostics-", suffix=".tmp", dir=str(parent)
    )
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        with temp_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(_DIAGNOSTICS_HEADER)
            count = 0
            for row in rows:
                writer.writerow(astuple(row))
                count += 1
        os.link(temp_path, path)
        os.unlink(temp_path)
        return count
    except BaseException:
        try:
            temp_path.unlink()
        except OSError:
            pass
        raise