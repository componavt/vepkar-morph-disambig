"""File-input layer connecting two completed T1.1 result packages to the
ranking-comparison core.

Given two finished dev-run directories, this layer reads each package's
``metrics.json`` and its single prediction file (``dev.csv`` or
``dev.csv.zst``), checks that the two packages agree on their recorded data
identity, loads one common published benchmark from the local data checkout,
validates and loads both prediction inputs, and returns the core's
:class:`core.ranking_comparison.RankingComparison` result together with the
original source metadata in input order.  It performs no fingerprint
verification, computes no metrics, and writes no output.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import zstandard

from benchmark_context import _preflight_benchmark, load_benchmark_context
from core.data import DataError
from core.instances import CorpusTagError
from core.metrics import MetricsInputError
from core.predictions import (
    BenchmarkIntegrityError,
    PredictionCsvParseError,
    PredictionFileReadError,
    PredictionValidation,
    validate_predictions,
)
from core.ranking_comparison import RankingComparison, compare_rankings
from core.splits import SplitError, split_csv_path
from t1_features.config import T1ConfigError, load_t1_config

from commands.evaluate_predictions import load_ranked_predictions

_FORMAT_NAME = "vepkar-metrics"
_FORMAT_VERSION = 2
_METHOD = "t1.1"
_TARGET_SPLIT = "dev"


class T1ComparisonInputError(ValueError):
    """A T1 comparison input package or the shared benchmark could not be used.

    Carries the invalid :class:`core.predictions.PredictionValidation` object
    through ``validation`` when a prediction file fails validation.
    """

    def __init__(self, message: str, *, validation: PredictionValidation | None = None):
        super().__init__(message)
        self.validation = validation


@dataclass(frozen=True)
class T1ComparisonSource:
    """Metadata of one completed T1.1 dev package and its original prediction path."""

    run_dir: Path
    predictions_path: Path
    method: str
    C: int | float
    data_tag: str
    data_commit: str
    split_sha256: str


@dataclass(frozen=True)
class T1LoadedComparison:
    """Two compared T1 sources in input order and the core's comparison result."""

    source_1: T1ComparisonSource
    source_2: T1ComparisonSource
    comparison: RankingComparison


def _check_tag_safe(tag: str, metrics_path: Path) -> None:
    if "/" in tag or "\\" in tag or "\x00" in tag or tag in (".", ".."):
        raise T1ComparisonInputError(
            f"invalid T1 metrics {metrics_path}: data.tag {tag!r} is not a "
            "safe filename component"
        )


def _check_payload(payload: object, metrics_path: Path) -> tuple[int | float, str, str, str]:
    if not isinstance(payload, dict):
        raise T1ComparisonInputError(
            f"invalid T1 metrics {metrics_path}: payload is not a JSON object"
        )
    if payload.get("format") != _FORMAT_NAME:
        raise T1ComparisonInputError(
            f"unsupported T1 metrics format in {metrics_path}: "
            f"{payload.get('format')!r}"
        )
    format_version = payload.get("format_version")
    if not (
        isinstance(format_version, int)
        and not isinstance(format_version, bool)
        and format_version == _FORMAT_VERSION
    ):
        raise T1ComparisonInputError(
            f"unsupported T1 metrics format_version in {metrics_path}: "
            f"{format_version!r}"
        )
    if payload.get("method") != _METHOD:
        raise T1ComparisonInputError(
            f"unsupported T1 metrics method in {metrics_path}: "
            f"{payload.get('method')!r}"
        )
    if payload.get("split") != _TARGET_SPLIT:
        raise T1ComparisonInputError(
            f"unsupported T1 metrics split in {metrics_path}: "
            f"{payload.get('split')!r}"
        )
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise T1ComparisonInputError(
            f"invalid T1 metrics {metrics_path}: parameters is not an object"
        )
    C = parameters.get("C")
    if not (isinstance(C, (int, float)) and not isinstance(C, bool)):
        raise T1ComparisonInputError(
            f"invalid T1 metrics {metrics_path}: parameters.C is not a number: {C!r}"
        )
    data = payload.get("data")
    if not isinstance(data, dict):
        raise T1ComparisonInputError(
            f"invalid T1 metrics {metrics_path}: data is not an object"
        )
    tag = data.get("tag")
    commit = data.get("commit")
    split_sha256 = data.get("split_sha256")
    if not (isinstance(tag, str) and tag):
        raise T1ComparisonInputError(
            f"invalid T1 metrics {metrics_path}: data.tag must be a nonempty string"
        )
    if not (isinstance(commit, str) and commit):
        raise T1ComparisonInputError(
            f"invalid T1 metrics {metrics_path}: data.commit must be a nonempty string"
        )
    if not (isinstance(split_sha256, str) and split_sha256):
        raise T1ComparisonInputError(
            f"invalid T1 metrics {metrics_path}: data.split_sha256 must be a "
            "nonempty string"
        )
    _check_tag_safe(tag, metrics_path)
    return C, tag, commit, split_sha256


def _load_source(run_dir: Path) -> T1ComparisonSource:
    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise T1ComparisonInputError(
            f"T1 comparison input directory does not exist: {run_dir}"
        )
    metrics_path = run_dir / "metrics.json"
    try:
        with metrics_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError:
        raise T1ComparisonInputError(
            f"T1 comparison input is missing metrics.json: {metrics_path}"
        ) from None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise T1ComparisonInputError(
            f"cannot read T1 metrics {metrics_path}: {exc}"
        ) from exc

    predictions_csv = run_dir / "dev.csv"
    predictions_zst = run_dir / "dev.csv.zst"
    csv_exists = predictions_csv.is_file()
    zst_exists = predictions_zst.is_file()
    if csv_exists and zst_exists:
        raise T1ComparisonInputError(
            f"T1 comparison input {run_dir} has both dev.csv and dev.csv.zst; "
            "ambiguous prediction source"
        )
    if not csv_exists and not zst_exists:
        raise T1ComparisonInputError(
            f"T1 comparison input {run_dir} has neither dev.csv nor dev.csv.zst"
        )
    predictions_path = predictions_csv if csv_exists else predictions_zst

    C, tag, commit, split_sha256 = _check_payload(payload, metrics_path)
    return T1ComparisonSource(
        run_dir=run_dir,
        predictions_path=predictions_path,
        method=_METHOD,
        C=C,
        data_tag=tag,
        data_commit=commit,
        split_sha256=split_sha256,
    )


def _check_metadata_compatible(
    source_1: T1ComparisonSource, source_2: T1ComparisonSource
) -> None:
    if (
        source_1.data_tag != source_2.data_tag
        or source_1.data_commit != source_2.data_commit
        or source_1.split_sha256 != source_2.split_sha256
    ):
        raise T1ComparisonInputError(
            "T1 comparison inputs have mismatched recorded data identities: "
            f"tag {source_1.data_tag!r} vs {source_2.data_tag!r}, commit "
            f"{source_1.data_commit!r} vs {source_2.data_commit!r}, "
            f"split_sha256 {source_1.split_sha256!r} vs "
            f"{source_2.split_sha256!r}"
        )


def _prepare_readable(
    predictions_path: Path, temp_dir: Path, index: int
) -> Path:
    """Return a readable CSV path, decompressing ``dev.csv.zst`` into ``temp_dir``."""
    if predictions_path.suffix != ".zst":
        return predictions_path
    temp_csv = temp_dir / f"dev_input{index}.csv"
    try:
        decompressor = zstandard.ZstdDecompressor()
        with predictions_path.open("rb") as src, temp_csv.open("wb") as dst:
            decompressor.copy_stream(src, dst)
    except (OSError, zstandard.ZstdError) as exc:
        raise T1ComparisonInputError(
            f"cannot decompress T1 predictions {predictions_path}: {exc}"
        ) from exc
    return temp_csv


def _validate_and_load(
    readable_path: Path,
    source: T1ComparisonSource,
    benchmark,
) -> list:
    try:
        validation = validate_predictions(
            readable_path,
            _TARGET_SPLIT,
            benchmark.instances,
            benchmark.split_rows,
        )
    except (
        PredictionFileReadError,
        PredictionCsvParseError,
        UnicodeDecodeError,
        BenchmarkIntegrityError,
    ) as exc:
        raise T1ComparisonInputError(
            f"cannot validate T1 predictions {source.predictions_path}: {exc}"
        ) from exc
    if not validation.is_valid:
        raise T1ComparisonInputError(
            f"invalid T1 predictions {source.predictions_path}",
            validation=validation,
        )
    try:
        return load_ranked_predictions(readable_path)
    except (PredictionFileReadError, PredictionCsvParseError, UnicodeDecodeError) as exc:
        raise T1ComparisonInputError(
            f"cannot load T1 predictions {source.predictions_path}: {exc}"
        ) from exc


def load_t1_comparison(
    run_dir_1: Path,
    run_dir_2: Path,
    *,
    repo_root: Path,
) -> T1LoadedComparison:
    """Compare two completed T1.1 dev packages on one common published benchmark."""
    source_1 = _load_source(run_dir_1)
    source_2 = _load_source(run_dir_2)
    _check_metadata_compatible(source_1, source_2)

    common_tag = source_1.data_tag
    published_split_path: Path | None = None
    try:
        config = load_t1_config(repo_root)
        data_dir, local_tag = _preflight_benchmark(config.data_dir)
        if local_tag != common_tag:
            raise T1ComparisonInputError(
                f"local data checkout tag {local_tag!r} does not match the "
                f"recorded comparison tag {common_tag!r}"
            )
        published_split_path = split_csv_path(
            config.repo_root / "data" / "derived", common_tag
        )
        benchmark = load_benchmark_context(
            data_dir, common_tag, published_split_path
        )
    except T1ComparisonInputError:
        raise
    except (T1ConfigError, DataError, CorpusTagError, SplitError, UnicodeDecodeError) as exc:
        split_detail = (
            f" (split {published_split_path})"
            if published_split_path is not None
            else ""
        )
        raise T1ComparisonInputError(
            f"cannot prepare the common benchmark for tag {common_tag!r}"
            f"{split_detail}: {exc}"
        ) from exc

    try:
        with tempfile.TemporaryDirectory(prefix="vepkar-t1-compare-") as temp:
            temp_dir = Path(temp)
            readable_1 = _prepare_readable(source_1.predictions_path, temp_dir, 1)
            readable_2 = _prepare_readable(source_2.predictions_path, temp_dir, 2)
            rows_1 = _validate_and_load(readable_1, source_1, benchmark)
            rows_2 = _validate_and_load(readable_2, source_2, benchmark)

            dev_texts = {
                (language, text_id)
                for language, text_id, split in benchmark.split_rows
                if split == _TARGET_SPLIT
            }
            dev_instances = [
                instance
                for instance in benchmark.instances
                if (instance.language, instance.text_id) in dev_texts
            ]

            try:
                comparison = compare_rankings(dev_instances, rows_1, rows_2)
            except MetricsInputError as exc:
                raise T1ComparisonInputError(
                    f"comparison of T1 predictions {source_1.predictions_path} and "
                    f"{source_2.predictions_path} failed: {exc}"
                ) from exc
            result = T1LoadedComparison(
                source_1=source_1,
                source_2=source_2,
                comparison=comparison,
            )
    except OSError as exc:
        raise T1ComparisonInputError(
            f"T1 comparison temporary workspace failed for "
            f"{source_1.predictions_path} and {source_2.predictions_path}: {exc}"
        ) from exc
    return result