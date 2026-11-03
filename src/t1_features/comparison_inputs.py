"""File-input layer connecting completed T1.1 result packages to the
ranking-comparison core.

Given two or more finished dev-run directories, this layer reads each
package's ``metrics.json`` and its single prediction file (``dev.csv`` or
``dev.csv.zst``), checks that all packages agree on their recorded data
identity, loads one common published benchmark from the local data checkout,
validates and loads each prediction input once, and returns every unordered
comparison pair together with the original source metadata in input order.
It performs no fingerprint verification, computes no metrics, and writes no
output.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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


@dataclass(frozen=True)
class T1LoadedComparisonBatch:
    """Every input source in input order and every unordered comparison pair."""

    sources: tuple[T1ComparisonSource, ...]
    comparisons: tuple[T1LoadedComparison, ...]


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


def _resolve_run_dirs(run_dirs: Iterable[Path]) -> tuple[Path, ...]:
    """Resolve each input path and reject repeats or too few distinct inputs."""
    materialized = tuple(Path(path) for path in run_dirs)
    if len(materialized) < 2:
        raise T1ComparisonInputError(
            "at least two distinct T1 comparison input directories are required"
        )
    resolved = tuple(path.expanduser().resolve() for path in materialized)
    seen: set[Path] = set()
    for path in resolved:
        if path in seen:
            raise T1ComparisonInputError(
                f"duplicate T1 comparison input directory: {path}"
            )
        seen.add(path)
    return resolved


def load_t1_comparisons(
    run_dirs: Iterable[Path],
    *,
    repo_root: Path,
) -> T1LoadedComparisonBatch:
    """Compare every pair of N completed T1.1 dev packages on one benchmark.

    All inputs must record the same data identity, which is established before
    the expensive benchmark corpus is loaded.  The benchmark, the configuration
    preflight, and each package's preparation/validation/loading happen exactly
    once; only the core's metric computations run once per unordered pair.
    """
    resolved = _resolve_run_dirs(run_dirs)
    sources = tuple(_load_source(path) for path in resolved)
    for source in sources[1:]:
        _check_metadata_compatible(sources[0], source)

    common_tag = sources[0].data_tag
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

    prediction_paths = tuple(source.predictions_path for source in sources)
    application_error: T1ComparisonInputError | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="vepkar-t1-compare-") as temp:
            temp_dir = Path(temp)
            try:
                readable = [
                    _prepare_readable(source.predictions_path, temp_dir, index)
                    for index, source in enumerate(sources, start=1)
                ]
                rows = [
                    _validate_and_load(readable_path, source, benchmark)
                    for readable_path, source in zip(readable, sources)
                ]

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

                comparisons: list[T1LoadedComparison] = []
                for i in range(len(sources)):
                    for j in range(i + 1, len(sources)):
                        try:
                            comparison = compare_rankings(
                                dev_instances, rows[i], rows[j]
                            )
                        except MetricsInputError as exc:
                            raise T1ComparisonInputError(
                                f"comparison of T1 predictions "
                                f"{sources[i].predictions_path} and "
                                f"{sources[j].predictions_path} failed: {exc}"
                            ) from exc
                        comparisons.append(
                            T1LoadedComparison(
                                source_1=sources[i],
                                source_2=sources[j],
                                comparison=comparison,
                            )
                        )
                result = T1LoadedComparisonBatch(
                    sources=sources,
                    comparisons=tuple(comparisons),
                )
            except T1ComparisonInputError as exc:
                application_error = exc
                raise
    except OSError as exc:
        if application_error is not None:
            application_error.add_note(
                f"temporary workspace cleanup failed for "
                f"{len(prediction_paths)} T1 comparison inputs: "
                f"{', '.join(str(path) for path in prediction_paths)}: {exc}"
            )
            raise application_error
        raise T1ComparisonInputError(
            f"T1 comparison temporary workspace failed for "
            f"{', '.join(str(path) for path in prediction_paths)}: {exc}"
        ) from exc
    return result


def load_t1_comparison(
    run_dir_1: Path,
    run_dir_2: Path,
    *,
    repo_root: Path,
) -> T1LoadedComparison:
    """Compare two completed T1.1 dev packages on one common published benchmark."""
    batch = load_t1_comparisons((run_dir_1, run_dir_2), repo_root=repo_root)
    return batch.comparisons[0]