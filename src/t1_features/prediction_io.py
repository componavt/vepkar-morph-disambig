"""Write already-ranked split predictions to a temporary CSV, validate them
against the supplied benchmark, and publish them without overwriting an
existing destination.

The caller supplies already-scored, ranked :class:`PredictionRow` objects and
an existing full :class:`BenchmarkContext`.  This module does not select
splits, score, rank, sort, reconstruct candidates, or load data.  The rows are
written to a temporary sibling file, validated for the requested split, and
published with a hard link so an existing destination is never overwritten.
"""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Iterable, Literal

from benchmark_context import BenchmarkContext
from core.predictions import (
    PREDICTION_HEADER,
    BenchmarkIntegrityError,
    PredictionCsvParseError,
    PredictionFileReadError,
    PredictionRow,
    PredictionValidation,
    validate_predictions,
)

_TEMP_PREFIX = ".vepkar-t1-predictions-"
_TEMP_SUFFIX = ".tmp"


class T1PredictionWriteError(ValueError):
    """A split's predictions CSV could not be written, validated, or published."""

    def __init__(
        self,
        message: str,
        *,
        validation: PredictionValidation | None = None,
    ):
        super().__init__(message)
        self.validation = validation


class T1PredictionPublishedCleanupError(T1PredictionWriteError):
    """The predictions CSV was published, but temporary cleanup failed."""

    def __init__(
        self,
        output_path: Path,
        temp_path: Path,
        *,
        split: Literal["dev", "test"] = "dev",
    ):
        self.output_path = output_path
        self.temp_path = temp_path
        super().__init__(
            f"{split} predictions published at {output_path} but removal of "
            f"temporary {temp_path} failed"
        )


def _expected_word_count(benchmark: BenchmarkContext, split: str) -> int:
    split_texts = {
        (language, int(text_id))
        for language, text_id, part in benchmark.split_rows
        if part == split
    }
    return sum(
        1
        for instance in benchmark.instances
        if (instance.language, int(instance.text_id)) in split_texts
    )


def write_split_predictions(
    rows: Iterable[PredictionRow],
    output_path: Path,
    *,
    benchmark: BenchmarkContext,
    split: Literal["dev", "test"],
) -> PredictionValidation:
    """Write ``rows`` to ``output_path`` without overwriting anything.

    The destination must not already exist (files, directories, and symlinks
    including broken ones are all rejected) and its parent directory must
    already exist.  The rows are written to a temporary sibling file that is
    then validated for ``split`` and hard-linked into place, so an
    existing destination is never overwritten or replaced.  A successful hard
    link publishes the output; a later failure to remove the temporary file
    raises :class:`T1PredictionPublishedCleanupError` and never touches the
    published output.
    """
    if output_path.is_symlink() or output_path.exists():
        raise T1PredictionWriteError(
            f"cannot write {split} predictions: destination already exists: {output_path}"
        )
    parent = output_path.parent
    if not parent.is_dir():
        raise T1PredictionWriteError(
            f"cannot write {split} predictions: parent directory does not exist: {parent}"
        )
    if _expected_word_count(benchmark, split) == 0:
        raise T1PredictionWriteError(
            f"cannot write {split} predictions: benchmark has no {split} occurrences: "
            f"{benchmark.tag}"
        )
    try:
        descriptor, temp_name = tempfile.mkstemp(
            prefix=_TEMP_PREFIX, suffix=_TEMP_SUFFIX, dir=str(parent)
        )
    except OSError as exc:
        raise T1PredictionWriteError(
            f"cannot write {split} predictions {output_path}: create temporary file: {exc}"
        ) from exc
    temp_path = Path(temp_name)
    try:
        try:
            stream = os.fdopen(descriptor, "w", encoding="utf-8", newline="")
        except OSError as exc:
            try:
                os.close(descriptor)
            except OSError as close_exc:
                wrapped = T1PredictionWriteError(
                    f"cannot write {split} predictions {output_path}: open temporary "
                    f"stream {temp_path}: {exc}"
                )
                wrapped.add_note(
                    f"failed to close temporary {split} predictions descriptor: {close_exc}"
                )
                raise wrapped from exc
            raise T1PredictionWriteError(
                f"cannot write {split} predictions {output_path}: open temporary "
                f"stream {temp_path}: {exc}"
            ) from exc
        try:
            writer = csv.writer(stream)
            writer.writerow(PREDICTION_HEADER)
            for row in rows:
                writer.writerow(
                    [row.word_id, row.wordform_id, row.gramset, row.rank, row.score]
                )
        except (OSError, csv.Error) as exc:
            try:
                stream.close()
            except OSError as close_exc:
                wrapped = T1PredictionWriteError(
                    f"cannot write {split} predictions {output_path}: write temporary "
                    f"stream {temp_path}: {exc}"
                )
                wrapped.add_note(
                    f"failed to close temporary {split} predictions stream {temp_path}: "
                    f"{close_exc}"
                )
                raise wrapped from exc
            raise T1PredictionWriteError(
                f"cannot write {split} predictions {output_path}: write temporary "
                f"stream {temp_path}: {exc}"
            ) from exc
        except BaseException as primary:
            try:
                stream.close()
            except OSError as close_exc:
                primary.add_note(
                    f"failed to close temporary {split} predictions stream {temp_path}: "
                    f"{close_exc}"
                )
            raise
        try:
            stream.close()
        except OSError as exc:
            raise T1PredictionWriteError(
                f"cannot write {split} predictions {output_path}: close temporary "
                f"stream {temp_path}: {exc}"
            ) from exc
        try:
            validation = validate_predictions(
                predictions_path=temp_path,
                split=split,
                instances=benchmark.instances,
                split_rows=benchmark.split_rows,
            )
        except (
            PredictionFileReadError,
            PredictionCsvParseError,
            BenchmarkIntegrityError,
        ) as exc:
            raise T1PredictionWriteError(
                f"cannot write {split} predictions {output_path}: validate temporary "
                f"CSV {temp_path}: {exc}"
            ) from exc
        if not validation.is_valid:
            raise T1PredictionWriteError(
                f"cannot write {split} predictions {output_path}: temporary CSV "
                f"{temp_path} failed {split} validation",
                validation=validation,
            )
        try:
            os.link(temp_path, output_path)
        except OSError as exc:
            raise T1PredictionWriteError(
                f"cannot write {split} predictions {output_path}: publish {temp_path}: {exc}"
            ) from exc
    except BaseException as exc:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as cleanup_exc:
            exc.add_note(
                f"failed to clean up temporary {split} predictions {temp_path}: "
                f"{cleanup_exc}"
            )
        raise
    try:
        temp_path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise T1PredictionPublishedCleanupError(
            output_path,
            temp_path,
            split=split,
        ) from exc
    return validation


def write_dev_predictions(
    rows: Iterable[PredictionRow],
    output_path: Path,
    *,
    benchmark: BenchmarkContext,
) -> PredictionValidation:
    """Backward-compatible helper for dev predictions."""
    return write_split_predictions(
        rows,
        output_path,
        benchmark=benchmark,
        split="dev",
    )
