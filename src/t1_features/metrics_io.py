"""Build and safely publish the T1 metrics JSON for one finished dev run.

The builder fingerprints the raw bytes of ``<run_dir>/dev.csv`` and records
the run-passport identities together with the supplied
:class:`core.metrics.RankingMetrics` for that run's complete dev predictions.
It never computes metrics, evaluates predictions, or reads anything except
that one file.  The writer validates the complete payload, serializes it to a
temporary sibling file, and publishes it with a hard link so an existing
destination is never overwritten.  This module is self-contained and uses no
generic publication framework.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path

from core.metrics import RankingMetrics
from t1_features.run_context import T1RunContext

_FORMAT_NAME = "vepkar-metrics"
_FORMAT_VERSION = 1
_METHOD = "t1.1"
_TARGET_SPLIT = "dev"
_TEMP_PREFIX = ".vepkar-t1-metrics-"
_TEMP_SUFFIX = ".tmp"
_CHUNK_BYTES = 65536
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_TOP_LEVEL_KEYS = (
    "format",
    "format_version",
    "method",
    "parameters",
    "split",
    "word_count",
    "candidate_count",
    "top1_correct",
    "top3_correct",
    "metrics",
    "data",
    "code",
    "predictions_sha256",
)
_PARAMETER_KEYS = ("C",)
_METRICS_KEYS = ("top1_accuracy", "mrr", "top3_accuracy")
_DATA_KEYS = ("tag", "commit", "split_sha256")
_CODE_KEYS = ("prepared_commit", "evaluation_commit")


class T1MetricsError(ValueError):
    """The T1 metrics JSON could not be built, validated, or published."""


class T1MetricsPublishedCleanupError(T1MetricsError):
    """The T1 metrics JSON was published, but removing its temporary name failed."""

    def __init__(self, output_path: Path, temp_path: Path):
        self.output_path = output_path
        self.temp_path = temp_path
        super().__init__(
            f"T1 metrics published at {output_path} but removal of temporary "
            f"{temp_path} failed"
        )


def _require_mapping(value, location: str) -> dict:
    if not isinstance(value, dict):
        raise T1MetricsError(
            f"invalid T1 metrics payload: {location} must be a dictionary, "
            f"got {value!r}"
        )
    return value


def _require_exact_keys(mapping: dict, expected, location: str) -> None:
    missing = [key for key in expected if key not in mapping]
    extra = [key for key in mapping if key not in expected]
    if not missing and not extra:
        return
    problems = []
    if missing:
        problems.append(f"missing keys {missing}")
    if extra:
        problems.append(f"unexpected keys {extra}")
    raise T1MetricsError(
        f"invalid T1 metrics payload: {location}: " + "; ".join(problems)
    )


def _require_integer(value, location: str) -> int:
    if type(value) is not int:
        raise T1MetricsError(
            f"invalid T1 metrics payload: {location} must be an integer, "
            f"got {value!r}"
        )
    return value


def _require_finite_number(value, location: str) -> None:
    if type(value) is bool or not isinstance(value, (int, float)):
        raise T1MetricsError(
            f"invalid T1 metrics payload: {location} must be a finite "
            f"number, got {value!r}"
        )
    try:
        finite = math.isfinite(value)
    except OverflowError as exc:
        raise T1MetricsError(
            f"invalid T1 metrics payload: {location} overflows the finite "
            f"representation, got an oversized integer"
        ) from exc
    if not finite:
        raise T1MetricsError(
            f"invalid T1 metrics payload: {location} must be finite, "
            f"got {value!r}"
        )


def _require_positive_finite_number(value, location: str) -> None:
    _require_finite_number(value, location)
    if value <= 0:
        raise T1MetricsError(
            f"invalid T1 metrics payload: {location} must be positive, "
            f"got {value!r}"
        )


def _require_nonempty_string(value, location: str) -> None:
    if not isinstance(value, str) or not value:
        raise T1MetricsError(
            f"invalid T1 metrics payload: {location} must be a nonempty "
            f"string, got {value!r}"
        )


def _require_sha256(value, location: str) -> None:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise T1MetricsError(
            f"invalid T1 metrics payload: {location} must be a 64-character "
            f"hexadecimal SHA-256, got {value!r}"
        )


def _require_consistent_accuracy(
    value, numerator: int, denominator: int, location: str
) -> None:
    try:
        expected = numerator / denominator
    except OverflowError as exc:
        raise T1MetricsError(
            f"invalid T1 metrics payload: {location} cannot be compared "
            f"against oversized counts"
        ) from exc
    if not math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise T1MetricsError(
            f"invalid T1 metrics payload: {location} {value!r} is "
            f"inconsistent with the recorded counts (expected {expected!r})"
        )


def _validate_payload(payload: object) -> None:
    """Require the exact T1 metrics schema; used by builder and writer."""
    if not isinstance(payload, dict):
        raise T1MetricsError(
            f"invalid T1 metrics payload: expected a dictionary, "
            f"got {type(payload).__name__}"
        )
    _require_exact_keys(payload, _TOP_LEVEL_KEYS, "manifest")
    if payload["format"] != _FORMAT_NAME:
        raise T1MetricsError(
            f"invalid T1 metrics payload: format must be {_FORMAT_NAME!r}, "
            f"got {payload['format']!r}"
        )
    if (
        type(payload["format_version"]) is not int
        or payload["format_version"] != _FORMAT_VERSION
    ):
        raise T1MetricsError(
            f"invalid T1 metrics payload: format_version must be the integer "
            f"{_FORMAT_VERSION}, got {payload['format_version']!r}"
        )
    if payload["method"] != _METHOD:
        raise T1MetricsError(
            f"invalid T1 metrics payload: method must be {_METHOD!r}, "
            f"got {payload['method']!r}"
        )
    if payload["split"] != _TARGET_SPLIT:
        raise T1MetricsError(
            f"invalid T1 metrics payload: split must be {_TARGET_SPLIT!r}, "
            f"got {payload['split']!r}"
        )

    parameters = _require_mapping(payload["parameters"], "parameters")
    _require_exact_keys(parameters, _PARAMETER_KEYS, "parameters")
    _require_positive_finite_number(parameters["C"], "parameters.C")

    word_count = _require_integer(payload["word_count"], "word_count")
    if word_count < 1:
        raise T1MetricsError(
            f"invalid T1 metrics payload: word_count must be a positive "
            f"integer, got {payload['word_count']!r}"
        )
    candidate_count = _require_integer(
        payload["candidate_count"], "candidate_count"
    )
    if candidate_count < word_count:
        raise T1MetricsError(
            f"invalid T1 metrics payload: candidate_count "
            f"{candidate_count!r} must be at least word_count "
            f"{word_count!r}"
        )
    top1_correct = _require_integer(payload["top1_correct"], "top1_correct")
    top3_correct = _require_integer(payload["top3_correct"], "top3_correct")
    if not (0 <= top1_correct <= top3_correct <= word_count):
        raise T1MetricsError(
            f"invalid T1 metrics payload: counts must satisfy "
            f"0 <= top1_correct <= top3_correct <= word_count, got "
            f"top1_correct={top1_correct!r}, top3_correct={top3_correct!r}, "
            f"word_count={word_count!r}"
        )

    metrics = _require_mapping(payload["metrics"], "metrics")
    _require_exact_keys(metrics, _METRICS_KEYS, "metrics")
    top1_accuracy = metrics["top1_accuracy"]
    mrr = metrics["mrr"]
    top3_accuracy = metrics["top3_accuracy"]
    _require_finite_number(top1_accuracy, "metrics.top1_accuracy")
    _require_finite_number(mrr, "metrics.mrr")
    _require_finite_number(top3_accuracy, "metrics.top3_accuracy")
    if not (0 <= top1_accuracy <= 1):
        raise T1MetricsError(
            f"invalid T1 metrics payload: metrics.top1_accuracy must be in "
            f"[0, 1], got {top1_accuracy!r}"
        )
    if not (0 <= top3_accuracy <= 1):
        raise T1MetricsError(
            f"invalid T1 metrics payload: metrics.top3_accuracy must be in "
            f"[0, 1], got {top3_accuracy!r}"
        )
    if not (0 < mrr <= 1):
        raise T1MetricsError(
            f"invalid T1 metrics payload: metrics.mrr must be in (0, 1], "
            f"got {mrr!r}"
        )
    _require_consistent_accuracy(
        top1_accuracy, top1_correct, word_count, "metrics.top1_accuracy"
    )
    _require_consistent_accuracy(
        top3_accuracy, top3_correct, word_count, "metrics.top3_accuracy"
    )

    data = _require_mapping(payload["data"], "data")
    _require_exact_keys(data, _DATA_KEYS, "data")
    _require_nonempty_string(data["tag"], "data.tag")
    _require_nonempty_string(data["commit"], "data.commit")
    _require_sha256(data["split_sha256"], "data.split_sha256")

    code = _require_mapping(payload["code"], "code")
    _require_exact_keys(code, _CODE_KEYS, "code")
    _require_nonempty_string(code["prepared_commit"], "code.prepared_commit")
    _require_nonempty_string(code["evaluation_commit"], "code.evaluation_commit")

    _require_sha256(payload["predictions_sha256"], "predictions_sha256")


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

    The ``context`` must come from :func:`load_t1_run_context` and
    ``metrics`` must have been computed for that run's complete dev
    predictions; the caller remains responsible for validation and
    evaluation.  The only input read is ``<context.run_dir>/dev.csv``, in
    binary chunks, to fingerprint its raw uncompressed bytes.
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
            "code": {
                "prepared_commit": manifest["code"]["commit"],
                "evaluation_commit": context.current_code_commit,
            },
            "predictions_sha256": "0" * 64,
        }
    except (KeyError, TypeError, AttributeError) as exc:
        raise T1MetricsError(
            f"cannot build T1 metrics: incomplete run context: {exc}"
        ) from exc
    _validate_payload(payload)
    payload["predictions_sha256"] = _dev_predictions_sha256(context.run_dir)
    return payload


def write_t1_metrics(payload: dict, output_path: Path) -> None:
    """Publish the T1 metrics JSON to ``output_path`` without overwriting.

    The destination must not already exist (files, directories, and symlinks
    including broken ones are all rejected) and its parent directory must
    already exist; nothing else is created.  The payload is validated and
    serialized before any file is created, then written to a temporary
    sibling file that is hard-linked into place.  A successful hard link
    publishes the output; a later failure to remove the temporary file raises
    :class:`T1MetricsPublishedCleanupError` and never touches the published
    output.
    """
    _validate_payload(payload)
    if output_path.is_symlink() or output_path.exists():
        raise T1MetricsError(
            f"cannot write T1 metrics: destination already exists: {output_path}"
        )
    parent = output_path.parent
    if not parent.is_dir():
        raise T1MetricsError(
            f"cannot write T1 metrics: parent directory does not exist: {parent}"
        )
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
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