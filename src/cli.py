"""Command-line interface for vepkar-morph-disambig."""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

from core.data import (
    SUPPORTED_LANGUAGES,
    DataError,
    read_corpus_tables,
    require_local_corpus,
    resolve_data_dir,
)
from core.instances import (
    DEFAULT_OUTPUT_DIR,
    CorpusTagError,
    LanguageInstances,
    build_language_instances,
    build_review_rows,
    determine_data_tag,
    review_csv_path,
    write_review_csv,
)
from core.splits import (
    DEFAULT_SPLIT_OUTPUT_DIR,
    SplitError,
    TextWeight,
    assign_texts,
    build_split_rows,
    compute_text_weights,
    split_csv_bytes,
    split_csv_path,
    split_overlap_counts,
    validate_split_assignments,
    write_split_csv,
)

from commands.diagnose_frequency_baseline import _run_diagnose_frequency_baseline
from commands.evaluate_predictions import (
    run_evaluate_predictions,
    load_ranked_predictions,
    print_evaluation_report,
)
from commands.fetch_data import _run_fetch_data
from commands.frequency_baseline import _run_frequency_baseline
from commands.inspect_data import (
    _run_inspect_data,
    _print_report,
    _print_examples,
    _print_class_examples,
)
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
    return parser


def _language_counts(
    result: LanguageInstances,
) -> tuple[int, int, int]:
    """Return (primary instances, distinct texts, candidate rows)."""
    primary = len(result.instances)
    text_ids = sorted({int(inst.text_id) for inst in result.instances})
    candidate_rows = sum(len(inst.candidates) for inst in result.instances)
    return primary, len(text_ids), candidate_rows


def _print_build_summary(tag: str, results: dict[str, LanguageInstances], path: Path, written: int) -> None:
    print(f"BENCHMARK — dictorpus-data {tag}")
    print()
    print(
        f"{'Lang':<6}{'Before strict':>14}{'Empty-candidate loss':>21}"
        f"{'Primary instances':>18}{'Texts':>7}{'Candidate rows':>15}"
    )
    totals = {"before": 0, "loss": 0, "primary": 0, "texts": 0, "rows": 0}
    for lang in SUPPORTED_LANGUAGES:
        result = results[lang]
        before = result.funnel[6].retained
        loss = result.funnel[7].removed
        primary, texts, rows = _language_counts(result)
        totals["before"] += before
        totals["loss"] += loss
        totals["primary"] += primary
        totals["texts"] += texts
        totals["rows"] += rows
        print(
            f"{lang:<6}{before:>14}{loss:>21}{primary:>18}{texts:>7}{rows:>15}"
        )
    print(
        f"{'ALL':<6}{totals['before']:>14}{totals['loss']:>21}"
        f"{totals['primary']:>18}{totals['texts']:>7}{totals['rows']:>15}"
    )
    print()
    print(
        "Before strict = eligible words before the final gramset check; "
        "Empty-candidate loss = words removed because a candidate has no gramset."
    )
    print(
        "Primary instances = words left for the benchmark; "
        "Candidate rows = the analyses available for those words."
    )
    print()
    print(f"Review CSV: {path} ({written} rows)")


def _print_benchmark_table(tag: str, results: dict[str, LanguageInstances]) -> None:
    print(f"BENCHMARK — dictorpus-data {tag}")
    print()
    print(
        f"{'Lang':<6}{'Primary instances':>18}{'Texts':>7}{'Candidate rows':>15}"
    )
    totals = {"primary": 0, "texts": 0, "rows": 0}
    for lang in SUPPORTED_LANGUAGES:
        primary, texts, rows = _language_counts(results[lang])
        totals["primary"] += primary
        totals["texts"] += texts
        totals["rows"] += rows
        print(f"{lang:<6}{primary:>18}{texts:>7}{rows:>15}")
    print(
        f"{'ALL':<6}{totals['primary']:>18}{totals['texts']:>7}{totals['rows']:>15}"
    )


def _split_stats(
    assignments: dict[str, dict[str, tuple[TextWeight, ...]]],
) -> dict[str, dict[str, tuple[int, int, int]]]:
    """Return per language and part ``(texts, instances, candidates)``."""
    stats: dict[str, dict[str, tuple[int, int, int]]] = {}
    for lang in SUPPORTED_LANGUAGES:
        stats[lang] = {}
        for part in ("train", "dev", "test"):
            weights = assignments[lang][part]
            texts = len(weights)
            instances = sum(weight.instance_count for weight in weights)
            candidates = sum(weight.candidate_row_count for weight in weights)
            stats[lang][part] = (texts, instances, candidates)
    return stats


def _print_split_table(assignments) -> None:
    stats = _split_stats(assignments)
    totals = {"train": [0, 0, 0], "dev": [0, 0, 0], "test": [0, 0, 0]}
    for lang in SUPPORTED_LANGUAGES:
        for part in ("train", "dev", "test"):
            for index in range(3):
                totals[part][index] += stats[lang][part][index]
    print("SPLIT — target by candidate rows: train 80% / dev 10% / test 10%")
    print()
    print(
        f"{'Lang':<6}{'Part':<7}{'Texts':>6}{'Instances':>10}"
        f"{'Candidates':>11}{'Candidate %':>13}"
    )
    for lang in SUPPORTED_LANGUAGES:
        total_rows = sum(stats[lang][part][2] for part in ("train", "dev", "test"))
        for index, part in enumerate(("train", "dev", "test")):
            texts, instances, candidates = stats[lang][part]
            percent = 100.0 * candidates / total_rows if total_rows else 0.0
            name = lang if index == 0 else ""
            print(
                f"{name:<6}{part:<7}{texts:>6}{instances:>10}{candidates:>11}"
                f"{percent:>11.1f}%"
            )
    all_rows = sum(totals[part][2] for part in ("train", "dev", "test"))
    for index, part in enumerate(("train", "dev", "test")):
        texts, instances, candidates = totals[part]
        percent = 100.0 * candidates / all_rows if all_rows else 0.0
        name = "ALL" if index == 0 else ""
        print(
            f"{name:<6}{part:<7}{texts:>6}{instances:>10}{candidates:>11}"
            f"{percent:>11.1f}%"
        )


def _print_split_report(tag, results, assignments, path: Path, written: bool) -> None:
    _print_benchmark_table(tag, results)
    print()
    _print_split_table(assignments)
    print()
    text_overlap, instance_overlap = split_overlap_counts(assignments)
    print(
        f"Checks: text overlap = {text_overlap}; instance overlap = {instance_overlap}"
    )
    status = "created" if written else "unchanged"
    print(f"Split: {path} ({status})")


def _run_make_splits(args: argparse.Namespace) -> int:
    data_dir = resolve_data_dir(args.data_dir)
    output_dir = (
        args.output_dir if args.output_dir is not None else DEFAULT_SPLIT_OUTPUT_DIR
    )
    try:
        require_local_corpus(data_dir)
        tag = determine_data_tag(data_dir)
        results: dict[str, LanguageInstances] = {}
        all_instances = []
        for lang in SUPPORTED_LANGUAGES:
            try:
                tables = read_corpus_tables(lang, data_dir)
                result = build_language_instances(lang, tables)
            except DataError as exc:
                print(f"  {lang}  FAILED: {exc}", file=sys.stderr, flush=True)
                return 1
            results[lang] = result
            all_instances.extend(result.instances)
            print(f"  {lang}  OK", file=sys.stderr, flush=True)
        weights = compute_text_weights(all_instances)
        assignments: dict[str, dict[str, tuple[TextWeight, ...]]] = {}
        for lang in SUPPORTED_LANGUAGES:
            lang_weights = tuple(
                weight for weight in weights if weight.language == lang
            )
            assignments[lang] = assign_texts(lang_weights)
        validate_split_assignments(assignments, weights)
        rows = build_split_rows(assignments)
        content = split_csv_bytes(rows)
        path = split_csv_path(output_dir, tag)
        written = write_split_csv(path, content)
    except (CorpusTagError, DataError, SplitError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _print_split_report(tag, results, assignments, path, written)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "fetch-data":
        return _run_fetch_data(args, parser)
    if args.command == "inspect-data":
        return _run_inspect_data(args, parser)
    if args.command == "build-instances":
        data_dir = resolve_data_dir(args.data_dir)
        output_dir = args.output_dir if args.output_dir is not None else DEFAULT_OUTPUT_DIR
        try:
            require_local_corpus(data_dir)
            tag = determine_data_tag(data_dir)
        except (CorpusTagError, DataError) as exc:
            parser.exit(1, f"error: {exc}\n")
        results = {}
        for lang in SUPPORTED_LANGUAGES:
            try:
                tables = read_corpus_tables(lang, data_dir)
                results[lang] = build_language_instances(lang, tables)
            except DataError as exc:
                print(f"  {lang}  FAILED: {exc}", file=sys.stderr, flush=True)
                return 1
            print(f"  {lang}  OK", file=sys.stderr, flush=True)
        try:
            rows = build_review_rows(results.values())
            path = review_csv_path(output_dir, tag)
            written = write_review_csv(path, rows)
        except CorpusTagError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        _print_build_summary(tag, results, path, written)
        return 0
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
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())