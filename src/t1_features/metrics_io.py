"""Build and safely publish the T1 metrics JSON for one finished dev run.

The builder fingerprints the raw bytes of ``<run_dir>/dev.csv`` and records
the run-passport data identities together with the supplied
:class:`core.metrics.RankingMetrics` for that run's complete dev predictions.
It never computes metrics, evaluates predictions, or reads anything except
that one file.  The writer serializes the payload built by
:func:`build_t1_metrics` to a temporary sibling file and publishes it with a
hard link so an existing destination is never overwritten.  This module is
self-contained and uses no generic publication framework.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from core.metrics import RankingMetrics
from t1_features.run_context import T1RunContext

_FORMAT_NAME = "vepkar-metrics"
_FORMAT_VERSION = 2
_METHOD = "t1.1"
_TARGET_SPLIT = "dev"
_TEMP_PREFIX = ".vepkar-t1-metrics-"
_TEMP_SUFFIX = ".tmp"
_CHUNK_BYTES = 65536


class T1MetricsError(ValueError):
    """The T1 metrics JSON could not be built, serialized, or published."""


class T1MetricsPublishedCleanupError(T1MetricsError):
    """The T1 metrics JSON was published, but removing its temporary name failed."""

    def __init__(self, output_path: Path, temp_path: Path):
        self.output_path = output_path
        self.temp_path = temp_path
        super().__init__(
            f"T1 metrics published at {output_path} but removal of temporary "
            f"{temp_path} failed"
        )


def _dev_predictions_sha256(run_dir: Path) -> str:
    path = run_dir / "dev.csv"
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as stream:
            while True:
                block = stream.read(_CHUNK_BYTES)
                if not block:
                    break
                digest.update(block)
    except OSError as exc:
        raise T1MetricsError(
            f"cannot read T1 dev predictions {path}: {exc}"
        ) from exc
    return digest.hexdigest()


def build_t1_metrics(
    context: T1RunContext,
    *,
    metrics: RankingMetrics,
    candidate_count: int,
) -> dict:
    """Construct the complete T1 metrics payload for one finished dev run.

    ``context`` must come from :func:`load_t1_run_context` and ``metrics``
    must have been computed for that run's complete dev predictions; the
    caller remains responsible for evaluation.  The only input read is
    ``<context.run_dir>/dev.csv``, in binary chunks, to fingerprint its raw
    uncompressed bytes.
    """
    try:
        manifest = context.manifest
        payload = {
            "format": _FORMAT_NAME,
            "format_version": _FORMAT_VERSION,
            "method": _METHOD,
            "parameters": {"C": manifest["parameters"]["C"]},
            "split": _TARGET_SPLIT,
            "word_count": metrics.word_count,
            "candidate_count": candidate_count,
            "top1_correct": metrics.top1_correct,
            "top3_correct": metrics.top3_correct,
            "metrics": {
                "top1_accuracy": metrics.top1_accuracy,
                "mrr": metrics.mrr,
                "top3_accuracy": metrics.top3_accuracy,
            },
            "data": {
                "tag": manifest["data"]["tag"],
                "commit": manifest["data"]["commit"],
                "split_sha256": manifest["split"]["sha256"],
            },
            "predictions_sha256": "0" * 64,
        }
    except (KeyError, TypeError, AttributeError) as exc:
        raise T1MetricsError(
            f"cannot build T1 metrics: incomplete run context: {exc}"
        ) from exc
    payload["predictions_sha256"] = _dev_predictions_sha256(context.run_dir)
    return payload


def write_t1_metrics(payload: dict, output_path: Path) -> None:
    """Publish the T1 metrics JSON to ``output_path`` without overwriting.

    The destination must not already exist (files, directories, and symlinks
    including broken ones are all rejected) and its parent directory must
    already exist; nothing else is created.  The payload is serialized before
    any file is created, then written to a temporary sibling file that is
    hard-linked into place.  A successful hard link publishes the output; a
    later failure to remove the temporary file raises
    :class:`T1MetricsPublishedCleanupError` and never touches the published
    output.
    """
    try:
        text = (
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n"
        )
    except (TypeError, ValueError) as exc:
        raise T1MetricsError(
            f"cannot write T1 metrics {output_path}: serialize payload: {exc}"
        ) from exc
    if output_path.is_symlink() or output_path.exists():
        raise T1MetricsError(
            f"cannot write T1 metrics: destination already exists: {output_path}"
        )
    parent = output_path.parent
    if not parent.is_dir():
        raise T1MetricsError(
            f"cannot write T1 metrics: parent directory does not exist: {parent}"
        )
    try:
        descriptor, temp_name = tempfile.mkstemp(
            prefix=_TEMP_PREFIX, suffix=_TEMP_SUFFIX, dir=str(parent)
        )
    except OSError as exc:
        raise T1MetricsError(
            f"cannot write T1 metrics {output_path}: create temporary file: {exc}"
        ) from exc
    temp_path = Path(temp_name)
    try:
        try:
            stream = os.fdopen(descriptor, "w", encoding="utf-8")
        except OSError as exc:
            try:
                os.close(descriptor)
            except OSError as close_exc:
                wrapped = T1MetricsError(
                    f"cannot write T1 metrics {output_path}: open temporary "
                    f"stream {temp_path}: {exc}"
                )
                wrapped.add_note(
                    f"failed to close temporary T1 metrics descriptor: {close_exc}"
                )
                raise wrapped from exc
            raise T1MetricsError(
                f"cannot write T1 metrics {output_path}: open temporary "
                f"stream {temp_path}: {exc}"
            ) from exc
        try:
            stream.write(text)
        except OSError as exc:
            try:
                stream.close()
            except OSError as close_exc:
                wrapped = T1MetricsError(
                    f"cannot write T1 metrics {output_path}: write temporary "
                    f"stream {temp_path}: {exc}"
                )
                wrapped.add_note(
                    f"failed to close temporary T1 metrics stream {temp_path}: "
                    f"{close_exc}"
                )
                raise wrapped from exc
            raise T1MetricsError(
                f"cannot write T1 metrics {output_path}: write temporary "
                f"stream {temp_path}: {exc}"
            ) from exc
        except BaseException as primary:
            try:
                stream.close()
            except OSError as close_exc:
                primary.add_note(
                    f"failed to close temporary T1 metrics stream {temp_path}: "
                    f"{close_exc}"
                )
            raise
        try:
            stream.close()
        except OSError as exc:
            raise T1MetricsError(
                f"cannot write T1 metrics {output_path}: close temporary "
                f"stream {temp_path}: {exc}"
            ) from exc
        try:
            os.link(temp_path, output_path)
        except OSError as exc:
            raise T1MetricsError(
                f"cannot write T1 metrics {output_path}: publish {temp_path}: {exc}"
            ) from exc
    except BaseException as exc:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as cleanup_exc:
            exc.add_note(
                f"failed to clean up temporary T1 metrics {temp_path}: "
                f"{cleanup_exc}"
            )
        raise
    try:
        temp_path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise T1MetricsPublishedCleanupError(output_path, temp_path) from exc