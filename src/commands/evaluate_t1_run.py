"""Evaluate the saved dev predictions of one T1 run and publish its metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmark_context import load_benchmark_context
from commands.evaluate_predictions import (
    load_ranked_predictions,
    print_evaluation_report,
)
from commands.prediction_reports import _print_validation_failure
from core.metrics import MetricsInputError, compute_ranking_metrics
from core.predictions import (
    BenchmarkIntegrityError,
    PredictionCsvParseError,
    PredictionFileReadError,
    validate_predictions,
)
from core.splits import SplitError
from t1_features.metrics_io import (
    T1MetricsError,
    T1MetricsPublishedCleanupError,
    build_t1_metrics,
    write_t1_metrics,
)
from t1_features.run_context import T1RunContextError, load_t1_run_context


def _run_evaluate_t1_run(args: argparse.Namespace) -> int:
    try:
        context = load_t1_run_context(args.run_dir)
    except T1RunContextError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    predictions_path = context.run_dir / "dev.csv"
    metrics_path = context.run_dir / "metrics.json"
    if metrics_path.is_dir():
        print("error: metrics output path is a directory:", file=sys.stderr)
        print(f"  {metrics_path}", file=sys.stderr)
        return 1
    if metrics_path.exists() or metrics_path.is_symlink():
        print("error: metrics already exists:", file=sys.stderr)
        print(f"  {metrics_path}", file=sys.stderr)
        return 1

    data_dir = Path(context.manifest["data"]["root"])
    tag = context.manifest["data"]["tag"]
    split_file = Path(context.manifest["split"]["path"])
    try:
        benchmark = load_benchmark_context(data_dir, tag, split_file)
    except SplitError as exc:
        print("error: cannot load the strict benchmark:", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        return 1
    except UnicodeDecodeError as exc:
        print("error: cannot decode benchmark tables as UTF-8:", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        return 1
    except BenchmarkIntegrityError as exc:
        print(
            "error: strict benchmark is internally inconsistent",
            file=sys.stderr,
        )
        print(exc, file=sys.stderr)
        return 1

    try:
        validation = validate_predictions(
            predictions_path=predictions_path,
            split="dev",
            instances=benchmark.instances,
            split_rows=benchmark.split_rows,
        )
    except PredictionFileReadError as exc:
        print("error: cannot read predictions file", file=sys.stderr)
        print(predictions_path, file=sys.stderr)
        if exc.__cause__ is not None:
            print(exc.__cause__, file=sys.stderr)
        return 1
    except PredictionCsvParseError as exc:
        print("error: cannot parse predictions CSV", file=sys.stderr)
        print(predictions_path, file=sys.stderr)
        if exc.__cause__ is not None:
            print(exc.__cause__, file=sys.stderr)
        return 1
    except UnicodeDecodeError as exc:
        print("error: cannot decode predictions file as UTF-8:", file=sys.stderr)
        print(f"  {predictions_path}", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
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

    try:
        prediction_rows = load_ranked_predictions(predictions_path)
    except (PredictionFileReadError, PredictionCsvParseError) as exc:
        print("error: cannot load predictions for evaluation:", file=sys.stderr)
        print(f"  {predictions_path}", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        return 1
    except UnicodeDecodeError as exc:
        print("error: cannot decode predictions file as UTF-8:", file=sys.stderr)
        print(f"  {predictions_path}", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        return 1

    split_texts = {
        (language, text_id)
        for language, text_id, split in benchmark.split_rows
        if split == "dev"
    }
    evaluation_instances = [
        instance
        for instance in benchmark.instances
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

    try:
        payload = build_t1_metrics(
            context,
            metrics=metrics,
            candidate_count=validation.expected_candidate_count,
        )
        write_t1_metrics(payload, metrics_path)
    except T1MetricsPublishedCleanupError as exc:
        print(
            "error: metrics.json was published, but temporary cleanup failed:",
            file=sys.stderr,
        )
        print(f"  destination: {exc.output_path}", file=sys.stderr)
        print(f"  temporary: {exc.temp_path}", file=sys.stderr)
        if exc.__cause__ is not None:
            print(f"  {exc.__cause__}", file=sys.stderr)
        return 1
    except T1MetricsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        for note in getattr(exc, "__notes__", ()):
            print(f"  {note}", file=sys.stderr)
        return 1

    print_evaluation_report(
        split="dev",
        predictions_path=predictions_path,
        word_count=validation.expected_word_count,
        candidate_count=validation.expected_candidate_count,
        metrics=metrics,
    )
    print(f"Run directory: {context.run_dir}")
    print(f"Metrics: {metrics_path}")
    return 0
