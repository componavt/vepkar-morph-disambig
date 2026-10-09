"""Validate one temporary prediction CSV against a benchmark split."""

from __future__ import annotations

import argparse
import sys

from benchmark_context import _preflight_benchmark, load_benchmark_context
from core.data import DataError
from core.instances import CorpusTagError
from core.predictions import (
    BenchmarkIntegrityError,
    PredictionCsvParseError,
    PredictionFileReadError,
    validate_predictions,
)
from core.splits import SplitError

from commands.prediction_reports import _print_validation_failure


def _run_validate_predictions(args: argparse.Namespace) -> int:
    try:
        data_dir, tag = _preflight_benchmark(args.data_dir)
    except (CorpusTagError, DataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        with args.predictions.open("rb"):
            pass
    except OSError as exc:
        print("error: cannot read predictions file:", file=sys.stderr)
        print(f"  {args.predictions}", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        return 1
    try:
        context = load_benchmark_context(data_dir, tag, args.split_file)
    except DataError:
        return 1
    except SplitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        result = validate_predictions(
            predictions_path=args.predictions,
            split=args.split,
            instances=context.instances,
            split_rows=context.split_rows,
        )
    except PredictionFileReadError as exc:
        print("error: cannot read predictions file:", file=sys.stderr)
        print(f"  {args.predictions}", file=sys.stderr)
        if exc.__cause__ is not None:
            print(f"  {exc.__cause__}", file=sys.stderr)
        return 1
    except PredictionCsvParseError as exc:
        print("error: cannot parse predictions CSV:", file=sys.stderr)
        print(f"  {args.predictions}", file=sys.stderr)
        if exc.__cause__ is not None:
            print(f"  {exc.__cause__}", file=sys.stderr)
        return 1
    except BenchmarkIntegrityError as exc:
        print("error: strict benchmark is internally inconsistent:", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        return 1
    if result.is_valid:
        print("Predictions validation: OK")
        print(f"Split: {args.split}")
        print(
            f"Validated: {result.expected_word_count:,} word instances, "
            f"{result.expected_candidate_count:,} candidate rows"
        )
        return 0
    _print_validation_failure(result)
    return 1