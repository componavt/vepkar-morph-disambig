"""Generate a train-frequency baseline predictions CSV for dev or test."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

from benchmark_context import _preflight_benchmark, load_benchmark_context
from core.data import DataError
from core.frequency import (
    TargetSplitError,
    build_train_frequency,
    rank_by_train_frequency,
)
from core.instances import CorpusTagError
from core.predictions import (
    PREDICTION_HEADER,
    BenchmarkIntegrityError,
    PredictionCsvParseError,
    PredictionFileReadError,
    validate_predictions,
)
from core.splits import SplitError

from commands.prediction_reports import _print_validation_failure


def _run_frequency_baseline(args: argparse.Namespace) -> int:
    output_path = Path(args.output)
    try:
        data_dir, tag = _preflight_benchmark(args.data_dir)
    except (CorpusTagError, DataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if output_path.is_dir():
        print("error: output path is a directory:", file=sys.stderr)
        print(f"  {output_path}", file=sys.stderr)
        return 1
    if output_path.exists() or output_path.is_symlink():
        print("error: output file already exists:", file=sys.stderr)
        print(f"  {output_path}", file=sys.stderr)
        return 1
    if not output_path.parent.is_dir():
        print("error: output directory does not exist:", file=sys.stderr)
        print(f"  {output_path.parent}", file=sys.stderr)
        return 1
    try:
        context = load_benchmark_context(data_dir, tag, args.split_file)
    except DataError:
        return 1
    except SplitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        rows = rank_by_train_frequency(
            context.instances, context.split_rows, args.split
        )
    except TargetSplitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    frequency = build_train_frequency(context.instances, context.split_rows)
    train_occurrences = sum(frequency.values())
    distinct_keys = len(frequency)
    word_count = len({row.word_id for row in rows})
    candidate_count = len(rows)

    descriptor, temp_name = tempfile.mkstemp(
        prefix=".vepkar-frequency-", suffix=".tmp", dir=str(output_path.parent)
    )
    os.close(descriptor)
    temp_path = Path(temp_name)
    published = False
    try:
        with temp_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(PREDICTION_HEADER)
            for row in rows:
                writer.writerow(
                    [row.word_id, row.wordform_id, row.gramset, row.rank, row.score]
                )
        try:
            validation = validate_predictions(
                predictions_path=temp_path,
                split=args.split,
                instances=context.instances,
                split_rows=context.split_rows,
            )
        except PredictionFileReadError as exc:
            print(
                "error: frequency baseline could not read its temporary "
                "predictions file:",
                file=sys.stderr,
            )
            print(f"  {temp_path}", file=sys.stderr)
            if exc.__cause__ is not None:
                print(f"  {exc.__cause__}", file=sys.stderr)
            return 1
        except PredictionCsvParseError as exc:
            print(
                "error: frequency baseline could not parse its temporary "
                "predictions CSV:",
                file=sys.stderr,
            )
            print(f"  {temp_path}", file=sys.stderr)
            if exc.__cause__ is not None:
                print(f"  {exc.__cause__}", file=sys.stderr)
            return 1
        except BenchmarkIntegrityError as exc:
            print(
                "error: strict benchmark is internally inconsistent:",
                file=sys.stderr,
            )
            print(f"  {exc}", file=sys.stderr)
            return 1
        if not validation.is_valid:
            print(
                "error: frequency baseline generated an invalid predictions CSV",
                file=sys.stderr,
            )
            _print_validation_failure(validation)
            return 1
        os.replace(temp_path, output_path)
        published = True
    finally:
        if not published and temp_path.exists():
            temp_path.unlink()
    print("Frequency baseline: OK")
    print(f"Training gold occurrences: {train_occurrences:,}")
    print(f"Distinct train candidate keys: {distinct_keys:,}")
    print(f"Split: {args.split}")
    print(
        f"Generated: {word_count:,} word instances, "
        f"{candidate_count:,} candidate rows"
    )
    print(f"Output: {output_path}")
    print("Validation: OK")
    return 0