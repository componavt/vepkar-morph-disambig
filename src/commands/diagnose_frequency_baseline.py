"""Frequency-baseline diagnostic command handler."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmark_context import _preflight_benchmark, load_benchmark_context
from core.data import DataError
from core.diagnostics import DiagnosticMappingError, build_frequency_diagnostics
from core.frequency import rank_by_train_frequency
from core.instances import CorpusTagError
from core.predictions import (
    BenchmarkIntegrityError,
    PredictionCsvParseError,
    PredictionFileReadError,
    validate_predictions,
)
from core.splits import SplitError
from frequency_diagnostics_io import (
    DiagnosticCleanupError,
    FrequencyBaselineMismatchError,
    load_verified_frequency_predictions,
    write_frequency_diagnostics,
)
from frequency_diagnostics_summary import format_frequency_diagnostics_summary

from commands.prediction_reports import _print_validation_failure


def _run_diagnose_frequency_baseline(args: argparse.Namespace) -> int:
    try:
        data_dir, tag = _preflight_benchmark(args.data_dir)
    except (CorpusTagError, DataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for role, path in (("predictions", args.predictions), ("output", args.output)):
        if path.suffix.lower() == ".zst":
            print(
                f"error: {role} path must be an uncompressed CSV, not .zst "
                "(decompress it beforehand):",
                file=sys.stderr,
            )
            print(f"  {path}", file=sys.stderr)
            return 1
    output_path = Path(args.output)
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
        validation = validate_predictions(
            predictions_path=args.predictions,
            split="dev",
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
    except UnicodeDecodeError:
        print("error: cannot decode predictions file as UTF-8:", file=sys.stderr)
        print(f"  {args.predictions}", file=sys.stderr)
        return 1
    if not validation.is_valid:
        _print_validation_failure(validation)
        return 1
    try:
        expected = rank_by_train_frequency(
            context.instances, context.split_rows, "dev"
        )
        verified = load_verified_frequency_predictions(args.predictions, expected)
    except FrequencyBaselineMismatchError as exc:
        print(
            "error: predictions do not match the expected frequency baseline:",
            file=sys.stderr,
        )
        print(f"  {exc}", file=sys.stderr)
        return 1
    except (PredictionFileReadError, PredictionCsvParseError) as exc:
        print("error: cannot verify predictions against the baseline:", file=sys.stderr)
        print(f"  {args.predictions}", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        return 1
    except UnicodeDecodeError:
        print("error: cannot decode predictions file as UTF-8:", file=sys.stderr)
        print(f"  {args.predictions}", file=sys.stderr)
        return 1
    dev_texts = {
        (language, text_id)
        for language, text_id, part in context.split_rows
        if part == "dev"
    }
    dev_instances = tuple(
        inst
        for inst in context.instances
        if (inst.language, inst.text_id) in dev_texts
    )
    try:
        rows = build_frequency_diagnostics(dev_instances, verified)
    except BenchmarkIntegrityError as exc:
        print("error: strict benchmark is internally inconsistent:", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        return 1
    except DiagnosticMappingError as exc:
        print("error: cannot map predictions to dev instances:", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        return 1
    summary_lines = format_frequency_diagnostics_summary(rows)
    try:
        count = write_frequency_diagnostics(args.output, rows)
    except DiagnosticCleanupError as exc:
        print(
            "error: CSV published, but temporary cleanup failed:",
            file=sys.stderr,
        )
        print(f"  output: {exc.output_path}", file=sys.stderr)
        print(f"  temporary: {exc.temp_path}", file=sys.stderr)
        if exc.__cause__ is not None:
            print(f"  {exc.__cause__}", file=sys.stderr)
        return 1
    except OSError as exc:
        print("error: cannot write diagnostics CSV:", file=sys.stderr)
        print(f"  {args.output}", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        return 1
    print("Frequency diagnostics: OK")
    print("Split: dev")
    print(f"Diagnostic rows: {count}")
    print(f"Output: {args.output}")
    for line in summary_lines:
        print(line)
    return 0