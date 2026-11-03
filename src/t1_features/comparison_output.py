"""Choose the automatic destination and safely publish a T1 ranking comparison
as a comparative CSV.

The caller supplies the loaded comparison from
:func:`t1_features.comparison_inputs.load_t1_comparison`; this module never
recomputes roles, metrics, deltas, outcomes, or language summaries and never
reads the source run directories.  The writer uses the same narrow
temporary-sibling/no-overwrite lifecycle as the T1 metrics and prediction
writers: an existing destination (file, directory, or symlink, including a
broken one) is refused, the rows are written to a temporary sibling file, and
the file is published with a hard link so nothing is ever overwritten.
"""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Iterable

from core.ranking_comparison import OccurrenceComparison
from t1_features.comparison_inputs import T1LoadedComparison

COMPARISON_HEADER = [
    "language",
    "word_id",
    "word",
    "candidate_count",
    "gold_wordform_id",
    "gold_gramset",
    "a_top1_wordform_id",
    "a_top1_gramset",
    "b_top1_wordform_id",
    "b_top1_gramset",
    "a_gold_rank",
    "b_gold_rank",
    "top1_changed",
    "outcome",
    "delta_rr",
]

_TEMP_PREFIX = ".vepkar-t1-comparison-"
_TEMP_SUFFIX = ".tmp"


class T1ComparisonWriteError(ValueError):
    """A comparative CSV could not be written or published."""


class T1ComparisonPublishedCleanupError(T1ComparisonWriteError):
    """The comparative CSV was published, but removing its temporary name failed."""

    def __init__(self, output_path: Path, temp_path: Path):
        self.output_path = output_path
        self.temp_path = temp_path
        super().__init__(
            f"comparison CSV published at {output_path} but removal of "
            f"temporary {temp_path} failed"
        )


def _c_label(value: int | float) -> str:
    """Format one C value without rounding, dropping only a real trailing '.0'."""
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _short_label(source) -> str:
    return f"{source.method}-C{_c_label(source.C)}"


def default_comparison_path(
    loaded: T1LoadedComparison,
    *,
    repo_root: Path,
) -> Path:
    """Return the automatic comparative CSV path for a loaded comparison.

    The path is derived from the globally assigned input-order source metadata
    (``source_1``/``source_2``), never from comparison roles or directory
    names, so reversing the two input directories reverses the path labels
    and the core's winner orientation is irrelevant.
    """
    tag = loaded.source_1.data_tag
    return (
        repo_root
        / "work"
        / "comparisons"
        / tag
        / f"{_short_label(loaded.source_1)}__vs__{_short_label(loaded.source_2)}.csv"
    )


def write_comparison_csv(
    rows: Iterable[OccurrenceComparison],
    output_path: Path,
) -> None:
    """Publish ``rows`` to ``output_path`` without overwriting anything.

    The destination must not already exist (files, directories, and symlinks
    including broken ones are all rejected) and its parent directory must
    already exist.  The rows are written to a temporary sibling file that is
    then hard-linked into place, so an existing destination is never
    overwritten or replaced.  A successful hard link publishes the output; a
    later failure to remove the temporary file raises
    :class:`T1ComparisonPublishedCleanupError` and never touches the published
    output.
    """
    if output_path.is_symlink() or output_path.exists():
        raise T1ComparisonWriteError(
            f"cannot write comparison CSV: destination already exists: {output_path}"
        )
    parent = output_path.parent
    if not parent.is_dir():
        raise T1ComparisonWriteError(
            f"cannot write comparison CSV: parent directory does not exist: {parent}"
        )
    try:
        descriptor, temp_name = tempfile.mkstemp(
            prefix=_TEMP_PREFIX, suffix=_TEMP_SUFFIX, dir=str(parent)
        )
    except OSError as exc:
        raise T1ComparisonWriteError(
            f"cannot write comparison CSV {output_path}: create temporary file: {exc}"
        ) from exc
    temp_path = Path(temp_name)
    try:
        try:
            stream = os.fdopen(descriptor, "w", encoding="utf-8", newline="")
        except OSError as exc:
            try:
                os.close(descriptor)
            except OSError as close_exc:
                wrapped = T1ComparisonWriteError(
                    f"cannot write comparison CSV {output_path}: open temporary "
                    f"stream {temp_path}: {exc}"
                )
                wrapped.add_note(
                    f"failed to close temporary comparison descriptor: {close_exc}"
                )
                raise wrapped from exc
            raise T1ComparisonWriteError(
                f"cannot write comparison CSV {output_path}: open temporary "
                f"stream {temp_path}: {exc}"
            ) from exc
        try:
            writer = csv.writer(stream)
            writer.writerow(COMPARISON_HEADER)
            for row in rows:
                writer.writerow(
                    [
                        row.language,
                        row.word_id,
                        row.word,
                        row.candidate_count,
                        row.gold_wordform_id,
                        row.gold_gramset,
                        row.a_top1_wordform_id,
                        row.a_top1_gramset,
                        row.b_top1_wordform_id,
                        row.b_top1_gramset,
                        row.a_gold_rank,
                        row.b_gold_rank,
                        1 if row.top1_changed else 0,
                        row.outcome,
                        row.delta_rr,
                    ]
                )
        except (OSError, csv.Error) as exc:
            try:
                stream.close()
            except OSError as close_exc:
                wrapped = T1ComparisonWriteError(
                    f"cannot write comparison CSV {output_path}: write temporary "
                    f"stream {temp_path}: {exc}"
                )
                wrapped.add_note(
                    f"failed to close temporary comparison stream {temp_path}: "
                    f"{close_exc}"
                )
                raise wrapped from exc
            raise T1ComparisonWriteError(
                f"cannot write comparison CSV {output_path}: write temporary "
                f"stream {temp_path}: {exc}"
            ) from exc
        except BaseException as primary:
            try:
                stream.close()
            except OSError as close_exc:
                primary.add_note(
                    f"failed to close temporary comparison stream {temp_path}: "
                    f"{close_exc}"
                )
            raise
        try:
            stream.close()
        except OSError as exc:
            raise T1ComparisonWriteError(
                f"cannot write comparison CSV {output_path}: close temporary "
                f"stream {temp_path}: {exc}"
            ) from exc
        try:
            os.link(temp_path, output_path)
        except OSError as exc:
            raise T1ComparisonWriteError(
                f"cannot write comparison CSV {output_path}: publish {temp_path}: {exc}"
            ) from exc
    except BaseException as exc:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as cleanup_exc:
            exc.add_note(
                f"failed to clean up temporary comparison CSV {temp_path}: "
                f"{cleanup_exc}"
            )
        raise
    try:
        temp_path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise T1ComparisonPublishedCleanupError(output_path, temp_path) from exc
