"""Compare two completed T1.1 dev runs and publish a comparative CSV.

The handler delegates all input work to the T1 comparison input layer, which
owns benchmark loading, decompression, validation, and the ranking-comparison
core, then publishes the core's rows through the comparison output writer and
prints a concise terminal report.  Reporting stays local to this module.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from commands.prediction_reports import _print_validation_failure
from t1_features.comparison_inputs import (
    T1ComparisonInputError,
    load_t1_comparison,
)
from t1_features.comparison_output import (
    T1ComparisonPublishedCleanupError,
    T1ComparisonWriteError,
    default_comparison_path,
    write_comparison_csv,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_SELECTION_TEXT = {
    "top1": "B selected by Top-1",
    "mrr": "B selected by MRR",
    "top3": "B selected by Top-3",
    "tie": "Tie on all three metrics; input order preserved",
}

_METRIC_ROWS = (
    ("Top-1 accuracy", "top1_accuracy", "delta_top1"),
    ("MRR", "mrr", "delta_mrr"),
    ("Top-3 accuracy", "top3_accuracy", "delta_top3"),
)

_EVENT_FIELDS = (
    "both_correct",
    "corrected",
    "regressed",
    "both_wrong_same",
    "both_wrong_changed",
    "top1_changed",
    "gold_rank_improved",
    "gold_rank_worsened",
    "gold_rank_unchanged",
)

_LANGUAGE_FIELDS = ("word_count", "corrected", "regressed")


def _c_label(value: int | float) -> str:
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _delta(value: float | None) -> str:
    return "N/A" if value is None else f"{value:+.4f}"


def _print_comparison_report(loaded, output_path: Path) -> None:
    comparison = loaded.comparison
    a_source = loaded.source_1 if comparison.a_input_index == 1 else loaded.source_2
    b_source = loaded.source_2 if comparison.b_input_index == 2 else loaded.source_1
    overall = comparison.overall

    print("T1 comparison: OK")
    print(f"A: {a_source.run_dir} ({a_source.method}, C={_c_label(a_source.C)})")
    print(f"B: {b_source.run_dir} ({b_source.method}, C={_c_label(b_source.C)})")
    print(_SELECTION_TEXT[comparison.selection_reason])
    print(
        f"Evaluated: {overall.word_count:,} word instances, "
        f"{overall.candidate_count:,} candidate rows"
    )
    print()
    print("Metric          A       B       B-A")
    for label, metric_field, delta_field in _METRIC_ROWS:
        print(
            f"{label:<14} "
            f"{_metric(getattr(overall.metrics_a, metric_field)):<7} "
            f"{_metric(getattr(overall.metrics_b, metric_field)):<7} "
            f"{_delta(getattr(overall, delta_field))}"
        )
    print()
    print("Events")
    for field in _EVENT_FIELDS:
        print(f"{field}: {getattr(overall, field)}")
    print()
    print(
        "Language  Words  Corrected  Regressed  Delta top-1  Delta MRR  "
        "Delta top-3"
    )
    for language, summary in comparison.by_language.items():
        print(
            f"{language:<8} "
            f"{getattr(summary, _LANGUAGE_FIELDS[0]):<6} "
            f"{getattr(summary, _LANGUAGE_FIELDS[1]):<10} "
            f"{getattr(summary, _LANGUAGE_FIELDS[2]):<10} "
            f"{_delta(summary.delta_top1):<12} "
            f"{_delta(summary.delta_mrr):<11} "
            f"{_delta(summary.delta_top3)}"
        )
    print()
    print(f"Comparison CSV: {output_path}")


def _run_compare_t1_runs(args: argparse.Namespace) -> int:
    try:
        loaded = load_t1_comparison(
            args.run_dirs[0],
            args.run_dirs[1],
            repo_root=_PROJECT_ROOT,
        )
    except T1ComparisonInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        for note in getattr(exc, "__notes__", ()):
            print(f"  {note}", file=sys.stderr)
        if exc.validation is not None:
            _print_validation_failure(exc.validation)
        return 1

    automatic = args.output is None
    if automatic:
        output_path = default_comparison_path(loaded, repo_root=_PROJECT_ROOT)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print(
                "error: cannot create comparison output directory:",
                file=sys.stderr,
            )
            print(f"  {output_path.parent}", file=sys.stderr)
            print(f"  {exc}", file=sys.stderr)
            return 1
    else:
        requested = Path(os.path.expanduser(str(args.output)))
        if not requested.is_absolute():
            requested = Path.cwd() / requested
        output_path = requested

    try:
        write_comparison_csv(loaded.comparison.rows, output_path)
    except T1ComparisonPublishedCleanupError as exc:
        print(
            "error: comparison CSV was published, but temporary cleanup failed:",
            file=sys.stderr,
        )
        print(f"  destination: {exc.output_path}", file=sys.stderr)
        print(f"  temporary: {exc.temp_path}", file=sys.stderr)
        if exc.__cause__ is not None:
            print(f"  {exc.__cause__}", file=sys.stderr)
        return 1
    except T1ComparisonWriteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        for note in getattr(exc, "__notes__", ()):
            print(f"  {note}", file=sys.stderr)
        if automatic and "already exists" in str(exc):
            print(
                "hint: the automatic comparison name is reused by repeated "
                "experiments; use --output to choose a different destination",
                file=sys.stderr,
            )
        return 1

    _print_comparison_report(loaded, output_path)
    return 0
