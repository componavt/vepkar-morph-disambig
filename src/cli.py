"""Command-line interface for vepkar-morph-disambig."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from commands.benchmark_reports import (
    _language_counts,
    _print_build_summary,
    _print_benchmark_table,
    _split_stats,
    _print_split_table,
    _print_split_report,
)
from commands.build_instances import _run_build_instances
from commands.diagnose_frequency_baseline import _run_diagnose_frequency_baseline
from commands.evaluate_predictions import (
    run_evaluate_predictions,
    load_ranked_predictions,
    print_evaluation_report,
)
from commands.evaluate_t1_run import _run_evaluate_t1_run
from commands.export_t1_milestone import _run_export_t1_milestone
from commands.fetch_data import _run_fetch_data
from commands.frequency_baseline import _run_frequency_baseline
from commands.inspect_data import (
    _run_inspect_data,
    _print_report,
    _print_examples,
    _print_class_examples,
)
from commands.make_splits import _run_make_splits
from commands.predict_t1 import _run_predict_t1
from commands.prepare_t1_run import _run_prepare_t1_run
from commands.train_t1 import _run_train_t1
from commands.validate_predictions import _run_validate_predictions

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = _PROJECT_ROOT / "pyproject.toml"


def _version() -> str:
    with _PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)
    return data["project"]["version"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vepkar-morph-disambig",
        description="Rank candidate morphological analyses of words in VepKar context.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    commands = parser.add_subparsers(dest="command")
    fetch = commands.add_parser(
        "fetch-data", help="Clone a tagged dictorpus-data release locally"
    )
    fetch.add_argument("tag", help="Required Git tag of dictorpus-data")
    inspect = commands.add_parser(
        "inspect-data", help="Inspect the typed local corpus tables"
    )
    inspect.add_argument("language", help="One of: vep, krl, olo, lud")
    inspect.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Root of a local dictorpus-data checkout (default: data/dictorpus-data)",
    )
    build = commands.add_parser(
        "build-instances",
        help="Build benchmark instances and the developer sentence-review CSV",
    )
    build.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Root of a local dictorpus-data checkout (default: data/dictorpus-data)",
    )
    build.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for derived outputs (default: data/derived/quality)",
    )
    splits = commands.add_parser(
        "make-splits",
        help="Assign whole texts to train/dev/test by candidate-row weights",
    )
    splits.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Root of a local dictorpus-data checkout (default: data/dictorpus-data)",
    )
    splits.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the derived split CSV (default: data/derived)",
    )
    validate = commands.add_parser(
        "validate-predictions",
        help="Validate one temporary prediction CSV against a benchmark split",
    )
    validate.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to the temporary predictions CSV",
    )
    validate.add_argument(
        "--split",
        choices=("dev", "test"),
        required=True,
        help="Benchmark split the predictions file belongs to",
    )
    validate.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Root of a local dictorpus-data checkout (default: data/dictorpus-data)",
    )
    validate.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="Override the published splits CSV (default: data/derived/splits_<tag>.csv)",
    )
    baseline = commands.add_parser(
        "frequency-baseline",
        help="Generate a train-frequency baseline predictions CSV for dev or test",
    )
    baseline.add_argument(
        "--split",
        choices=("dev", "test"),
        required=True,
        help="Benchmark split the baseline ranks",
    )
    baseline.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the generated predictions CSV",
    )
    baseline.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Root of a local dictorpus-data checkout (default: data/dictorpus-data)",
    )
    baseline.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="Override the published splits CSV (default: data/derived/splits_<tag>.csv)",
    )
    evaluate = commands.add_parser(
        "evaluate-predictions",
        help="Evaluate one validated predictions CSV against a benchmark split",
    )
    evaluate.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to the predictions CSV to evaluate",
    )
    evaluate.add_argument(
        "--split",
        choices=("dev", "test"),
        required=True,
        help="Benchmark split the predictions file belongs to",
    )
    evaluate.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Root of a local dictorpus-data checkout (default: data/dictorpus-data)",
    )
    evaluate.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="Override the published splits CSV (default: data/derived/splits_<tag>.csv)",
    )
    diagnose = commands.add_parser(
        "diagnose-frequency-baseline",
        help=(
            "Diagnose a frequency-baseline dev predictions CSV "
            "(uncompressed CSV input and output)"
        ),
    )
    diagnose.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to the uncompressed dev predictions CSV",
    )
    diagnose.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the uncompressed diagnostics CSV output",
    )
    diagnose.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Root of a local dictorpus-data checkout (default: data/dictorpus-data)",
    )
    diagnose.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="Override the published splits CSV (default: data/derived/splits_<tag>.csv)",
    )
    train_t1 = commands.add_parser(
        "train-t1",
        help="Train the T1 shared model on the real train split and save one bundle",
        allow_abbrev=False,
    )
    train_t1.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Existing prepared T1 run directory",
    )
    predict_t1 = commands.add_parser(
        "predict-t1",
        help="Predict dev candidate rankings with a saved T1 model bundle",
        allow_abbrev=False,
    )
    predict_t1.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Existing prepared T1 run directory",
    )
    evaluate_t1_run = commands.add_parser(
        "evaluate-t1-run",
        help="Evaluate the saved dev predictions of a T1 run and publish metrics",
        allow_abbrev=False,
    )
    evaluate_t1_run.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Existing prepared T1 run directory",
    )
    prepare_t1 = commands.add_parser(
        "prepare-t1-run",
        help="Record provenance and intended settings of a T1 dev experiment",
        allow_abbrev=False,
    )
    prepare_t1.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Directory for the prepared T1 run "
            "(default: unique new directory under runs_dir)"
        ),
    )
    prepare_t1.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Logistic regression inverse regularization strength (default: 1.0)",
    )
    prepare_t1.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Root of a local dictorpus-data checkout (default: data/dictorpus-data)",
    )
    prepare_t1.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="Override the published splits CSV (default: published split for the data tag)",
    )
    export_t1_milestone = commands.add_parser(
        "export-t1-milestone",
        help="Export one completed T1.1 dev result into a compact milestone directory",
        allow_abbrev=False,
    )
    export_t1_milestone.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Existing finished T1 run directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "fetch-data":
        return _run_fetch_data(args, parser)
    if args.command == "inspect-data":
        return _run_inspect_data(args, parser)
    if args.command == "build-instances":
        return _run_build_instances(args, parser)
    if args.command == "make-splits":
        return _run_make_splits(args)
    if args.command == "validate-predictions":
        return _run_validate_predictions(args)
    if args.command == "frequency-baseline":
        return _run_frequency_baseline(args)
    if args.command == "diagnose-frequency-baseline":
        return _run_diagnose_frequency_baseline(args)
    if args.command == "evaluate-predictions":
        return run_evaluate_predictions(args)
    if args.command == "train-t1":
        return _run_train_t1(args)
    if args.command == "predict-t1":
        return _run_predict_t1(args)
    if args.command == "evaluate-t1-run":
        return _run_evaluate_t1_run(args)
    if args.command == "prepare-t1-run":
        return _run_prepare_t1_run(args)
    if args.command == "export-t1-milestone":
        return _run_export_t1_milestone(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())