"""Evaluate one validated predictions CSV against a benchmark split."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from benchmark_context import _read_split_rows
from core.data import (
    SUPPORTED_LANGUAGES,
    DataError,
    read_corpus_tables,
    require_local_corpus,
    resolve_data_dir,
)
from core.frequency import FrequencyPrediction
from core.instances import (
    CorpusTagError,
    Instance,
    build_language_instances,
    determine_data_tag,
)
from core.metrics import (
    MetricsInputError,
    RankingMetrics,
    compute_ranking_metrics,
)
from core.predictions import (
    PREDICTION_HEADER,
    BenchmarkIntegrityError,
    PredictionCsvParseError,
    PredictionFileReadError,
    validate_predictions,
)
from core.splits import (
    DEFAULT_SPLIT_OUTPUT_DIR,
    SplitError,
    split_csv_path,
)

from commands.prediction_reports import _print_validation_failure


def load_ranked_predictions(path: Path) -> list[FrequencyPrediction]:
    """Read an already validated predictions CSV into ranked prediction rows."""
    rows: list[FrequencyPrediction] = []

    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh, strict=True)

            header = next(reader, None)
            if tuple(header or ()) != PREDICTION_HEADER:
                raise PredictionCsvParseError(
                    f"cannot parse predictions CSV {path}: unexpected header "
                    f"{header!r}",
                )

            for raw in reader:
                if len(raw) != 5:
                    raise PredictionCsvParseError(
                        f"cannot parse predictions CSV {path}: "
                        f"invalid row {raw!r}",
                    )

                try:
                    word_id = int(raw[0])
                    wordform_id = int(raw[1])
                    rank = int(raw[3])
                    score = float(raw[4])
                except ValueError as exc:
                    raise PredictionCsvParseError(
                        f"cannot parse predictions CSV {path}: "
                        f"invalid row {raw!r}",
                    ) from exc

                rows.append(
                    FrequencyPrediction(
                        word_id=word_id,
                        wordform_id=wordform_id,
                        gramset=raw[2],
                        rank=rank,
                        score=score,
                    )
                )
    except csv.Error as exc:
        raise PredictionCsvParseError(
            f"cannot parse predictions CSV {path}: {exc}",
        ) from exc
    except OSError as exc:
        raise PredictionFileReadError(
            f"cannot read predictions file {path}",
        ) from exc

    return rows


def print_evaluation_report(
    *,
    split: str,
    predictions_path: Path,
    word_count: int,
    candidate_count: int,
    metrics: RankingMetrics,
) -> None:
    print("Prediction evaluation: OK")
    print(f"Predictions: {predictions_path}")
    print(f"Split: {split}")
    print(
        f"Evaluated: {word_count} word instances, "
        f"{candidate_count} candidate rows"
    )
    print()
    print("Metric             Value")
    print(f"Top-1 accuracy     {metrics.top1_accuracy:.4f}")
    print(f"MRR                {metrics.mrr:.4f}")
    print(f"Top-3 accuracy     {metrics.top3_accuracy:.4f}")


def run_evaluate_predictions(args: argparse.Namespace) -> int:
    data_dir = resolve_data_dir(args.data_dir)

    try:
        require_local_corpus(data_dir)
        tag = determine_data_tag(data_dir)
    except (CorpusTagError, DataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    instances: list[Instance] = []

    for language in SUPPORTED_LANGUAGES:
        try:
            tables = read_corpus_tables(language, data_dir)
            result = build_language_instances(language, tables)
        except DataError as exc:
            print(f"{language}: FAILED: {exc}", file=sys.stderr, flush=True)
            return 1

        instances.extend(result.instances)

    split_path = (
        args.split_file
        if args.split_file is not None
        else split_csv_path(DEFAULT_SPLIT_OUTPUT_DIR, tag)
    )

    try:
        split_rows = _read_split_rows(split_path)
    except SplitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        validation = validate_predictions(
            predictions_path=args.predictions,
            split=args.split,
            instances=instances,
            split_rows=split_rows,
        )
    except PredictionFileReadError as exc:
        print("error: cannot read predictions file", file=sys.stderr)
        print(args.predictions, file=sys.stderr)
        if exc.__cause__ is not None:
            print(exc.__cause__, file=sys.stderr)
        return 1
    except PredictionCsvParseError as exc:
        print("error: cannot parse predictions CSV", file=sys.stderr)
        print(args.predictions, file=sys.stderr)
        if exc.__cause__ is not None:
            print(exc.__cause__, file=sys.stderr)
        return 1
    except BenchmarkIntegrityError as exc:
        print(
            "error: strict benchmark is internally inconsistent",
            file=sys.stderr,
        )
        print(exc, file=sys.stderr)
        return 1

    if not validation.is_valid:
        _print_validation_failure(validation)
        return 1

    prediction_rows = load_ranked_predictions(args.predictions)

    split_texts = {
        (language, text_id)
        for language, text_id, split in split_rows
        if split == args.split
    }
    evaluation_instances = [
        instance
        for instance in instances
        if (instance.language, instance.text_id) in split_texts
    ]

    try:
        metrics = compute_ranking_metrics(
            evaluation_instances,
            prediction_rows,
        )
    except MetricsInputError as exc:
        print("error: cannot compute ranking metrics", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 1

    print_evaluation_report(
        split=args.split,
        predictions_path=args.predictions,
        word_count=validation.expected_word_count,
        candidate_count=validation.expected_candidate_count,
        metrics=metrics,
    )
    return 0