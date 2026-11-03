"""Compare two or more completed T1.1 dev runs and publish comparative CSVs.

The handler delegates all input work to the T1 comparison input layer, which
owns benchmark loading, decompression, validation, and the ranking-comparison
core, preflights every automatic destination before creating any output,
publishes each pair's rows through the comparison output writer, and prints a
compact terminal report.  Reporting stays local to this module.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from commands.prediction_reports import _print_validation_failure
from t1_features.comparison_inputs import (
    T1ComparisonInputError,
    T1LoadedComparisonBatch,
    load_t1_comparisons,
)
from t1_features.comparison_output import (
    T1ComparisonPublishedCleanupError,
    T1ComparisonWriteError,
    default_comparison_path,
    write_comparison_csv,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_REASON_TEXT = {
    "top1": "Top-1",
    "mrr": "MRR",
    "top3": "Top-3",
    "tie": "tie",
}

_METRIC_HEADERS = ("Top-1", "MRR", "Top-3")
_METRIC_FIELDS = ("top1_accuracy", "mrr", "top3_accuracy")

_MIN_INPUTS = 2


def _c_label(value: int | float) -> str:
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _delta(value: float | None) -> str:
    return "N/A" if value is None else f"{value:+.4f}"


def _pair_indices(
    batch: T1LoadedComparisonBatch, pair
) -> tuple[int, int]:
    """Return the 1-based input indices of a pair's two sources."""
    index_1 = batch.sources.index(pair.source_1) + 1
    index_2 = batch.sources.index(pair.source_2) + 1
    return index_1, index_2


def _pair_label(batch: T1LoadedComparisonBatch, pair) -> str:
    index_1, index_2 = _pair_indices(batch, pair)
    return f"{index_1} vs {index_2}"


def _metrics_by_source(
    batch: T1LoadedComparisonBatch,
) -> list:
    """Extract each source's unrounded metrics from its first applicable pair."""
    by_source: dict = {}
    for pair in batch.comparisons:
        for local_index, metrics in (
            (pair.comparison.a_input_index, pair.comparison.overall.metrics_a),
            (pair.comparison.b_input_index, pair.comparison.overall.metrics_b),
        ):
            source = pair.source_1 if local_index == 1 else pair.source_2
            if id(source) not in by_source:
                by_source[id(source)] = metrics
    return [by_source[id(source)] for source in batch.sources]


def _best_indices(metrics_by_source: list, field: str) -> list[int]:
    values = [getattr(metrics, field) for metrics in metrics_by_source]
    best = max(values)
    return [i + 1 for i, value in enumerate(values) if value == best]


def _print_success_report(batch: T1LoadedComparisonBatch, paths: list[Path]) -> None:
    sources = batch.sources
    pair_count = len(batch.comparisons)
    print(
        f"T1 comparison: OK ({len(sources)} sources, "
        f"{pair_count} pair{'s' if pair_count != 1 else ''})"
    )
    print()
    print("Sources")
    for i, source in enumerate(sources, start=1):
        print(
            f"  {i}  {source.run_dir}  {source.method}  "
            f"C={_c_label(source.C)}"
        )
    print()
    print("Pairs")
    print(
        "Pair     A\u2192B   Reason  Corrected  Regressed  "
        "\u0394Top-1  \u0394MRR  \u0394Top-3"
    )
    for pair in batch.comparisons:
        index_1, index_2 = _pair_indices(batch, pair)
        a_index = index_1 if pair.comparison.a_input_index == 1 else index_2
        b_index = index_1 if pair.comparison.b_input_index == 1 else index_2
        overall = pair.comparison.overall
        print(
            f"{index_1} vs {index_2:<5} {a_index}\u2192{b_index:<4} "
            f"{_REASON_TEXT[pair.comparison.selection_reason]:<5} "
            f"{overall.corrected:<10} {overall.regressed:<9} "
            f"{_delta(overall.delta_top1):<8} {_delta(overall.delta_mrr):<7} "
            f"{_delta(overall.delta_top3)}"
        )
    print()
    _print_leader_report(batch)
    print()
    print("Outputs")
    for pair, path in zip(batch.comparisons, paths):
        print(f"  {_pair_label(batch, pair)}  {path}")


def _print_leader_report(batch: T1LoadedComparisonBatch) -> None:
    sources = batch.sources
    metrics_by_source = _metrics_by_source(batch)
    leaders = {
        header: _best_indices(metrics_by_source, field)
        for header, field in zip(_METRIC_HEADERS, _METRIC_FIELDS)
    }
    if all(len(indices) == 1 for indices in leaders.values()) and len(
        {indices[0] for indices in leaders.values()}
    ) == 1:
        index = leaders[_METRIC_HEADERS[0]][0]
        print(
            f"Best run: source {index} (C={_c_label(sources[index - 1].C)}) "
            "leads all three metrics"
        )
        return
    if len(sources) >= 4:
        print("Best runs")
        print("Metric  Best sources  C values  Value")
        for header, field in zip(_METRIC_HEADERS, _METRIC_FIELDS):
            indices = leaders[header]
            values = [getattr(metrics_by_source[i - 1], field) for i in indices]
            print(
                f"{header:<7} "
                f"{','.join(str(i) for i in indices):<14} "
                f"{','.join('C=' + _c_label(sources[i - 1].C) for i in indices):<16} "
                f"{values[0]:.4f}"
            )
        return
    for header, field in zip(_METRIC_HEADERS, _METRIC_FIELDS):
        indices = leaders[header]
        value = getattr(metrics_by_source[indices[0] - 1], field)
        names = ", ".join(str(i) for i in indices)
        c_values = ", ".join("C=" + _c_label(sources[i - 1].C) for i in indices)
        print(f"Best {header}: {names} ({c_values}) {value:.4f}")


def _print_input_error(exc: T1ComparisonInputError) -> None:
    print(f"error: {exc}", file=sys.stderr)
    for note in getattr(exc, "__notes__", ()):
        print(f"  {note}", file=sys.stderr)
    if exc.validation is not None:
        _print_validation_failure(exc.validation)


def _print_publication_failure(
    batch: T1LoadedComparisonBatch,
    saved: list[tuple],
    failed_pair,
    failed_path: Path,
    *,
    write_error: T1ComparisonWriteError | None = None,
    cleanup_error: T1ComparisonPublishedCleanupError | None = None,
) -> None:
    print("error: comparison publication failed", file=sys.stderr)
    print(
        f"  completed: {len(saved)} of {len(batch.comparisons)} comparisons",
        file=sys.stderr,
    )
    print(
        f"  failed pair: {_pair_label(batch, failed_pair)} -> {failed_path}",
        file=sys.stderr,
    )
    if saved:
        print("  already saved:", file=sys.stderr)
        for _, path in saved:
            print(f"    {path}", file=sys.stderr)
    if cleanup_error is not None:
        print(
            "  note: the failed pair's CSV was published, but temporary "
            "cleanup failed",
            file=sys.stderr,
        )
        print(f"    retained CSV: {cleanup_error.output_path}", file=sys.stderr)
        print(f"    temporary: {cleanup_error.temp_path}", file=sys.stderr)
    error = write_error if write_error is not None else cleanup_error
    if error is not None:
        print(f"  {error}", file=sys.stderr)
        if error.__cause__ is not None:
            print(f"  {error.__cause__}", file=sys.stderr)
        for note in getattr(error, "__notes__", ()):
            print(f"  {note}", file=sys.stderr)


def _run_compare_t1_runs(args: argparse.Namespace) -> int:
    if len(args.run_dirs) < _MIN_INPUTS:
        print(
            f"error: compare-t1-runs requires at least {_MIN_INPUTS} run "
            "directories",
            file=sys.stderr,
        )
        return 1

    try:
        batch = load_t1_comparisons(args.run_dirs, repo_root=_PROJECT_ROOT)
    except T1ComparisonInputError as exc:
        _print_input_error(exc)
        return 1

    destinations = [
        default_comparison_path(pair, repo_root=_PROJECT_ROOT)
        for pair in batch.comparisons
    ]
    owners: dict[Path, object] = {}
    for pair, path in zip(batch.comparisons, destinations):
        previous = owners.get(path)
        if previous is not None:
            print(
                "error: comparison destination collision: pairs "
                f"{_pair_label(batch, previous)} and {_pair_label(batch, pair)} "
                f"both map to {path}",
                file=sys.stderr,
            )
            return 1
        owners[path] = pair

    for path in destinations:
        if path.is_symlink() or path.exists():
            print(
                "error: comparison outputs are not overwritten; destination "
                f"already exists: {path}",
                file=sys.stderr,
            )
            return 1

    saved: list[tuple] = []
    for pair, path in zip(batch.comparisons, destinations):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print("error: cannot create comparison output directory:", file=sys.stderr)
            print(f"  {path.parent}", file=sys.stderr)
            print(f"  {exc}", file=sys.stderr)
            _print_publication_failure(batch, saved, pair, path)
            return 1
        try:
            write_comparison_csv(pair.comparison.rows, path)
        except T1ComparisonPublishedCleanupError as exc:
            _print_publication_failure(
                batch, saved, pair, path, cleanup_error=exc
            )
            return 1
        except T1ComparisonWriteError as exc:
            _print_publication_failure(
                batch, saved, pair, path, write_error=exc
            )
            return 1
        saved.append((pair, path))

    _print_success_report(batch, destinations)
    return 0
