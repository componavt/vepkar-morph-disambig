"""Command-line interface for vepkar-morph-disambig."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import tomllib
from pathlib import Path

from core.data import (
    DEFAULT_DATA_DIR,
    SUPPORTED_LANGUAGES,
    DataError,
    read_corpus_tables,
    require_local_corpus,
    resolve_data_dir,
)
from core.fetch import FetchError, fetch_data
from core.frequency import (
    FrequencyPrediction,
    TargetSplitError,
    build_train_frequency,
    rank_by_train_frequency,
)
from core.metrics import (
    MetricsInputError,
    RankingMetrics,
    compute_ranking_metrics,
)
from core.instances import (
    DEFAULT_OUTPUT_DIR,
    CorpusTagError,
    Instance,
    LanguageInstances,
    build_language_instances,
    build_review_rows,
    determine_data_tag,
    review_csv_path,
    write_review_csv,
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
from core.validation import CorpusError, inspect_corpus

from benchmark_context import (
    _preflight_benchmark,
    _read_split_rows,
    load_benchmark_context,
)

from commands.diagnose_frequency_baseline import _run_diagnose_frequency_baseline
from commands.prediction_reports import _print_validation_failure

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


def _print_examples(examples: tuple) -> None:
    for example in examples:
        print(f"  {example.detail}")
        xml = example.sentence_xml.replace("\r", " ").replace("\n", " ")
        print(f"  sentence_xml: {xml}")


def _print_class_examples(examples: tuple) -> None:
    _print_examples(examples)
    if examples:
        print()


def _print_report(inspection, corpus_dir: Path) -> None:
    print(f"Language: {inspection.lang}")
    print(f"Source: {corpus_dir}/")
    print()
    print("TABLES")
    rows = {
        "texts": inspection.row_texts,
        "sentences": inspection.row_sentences,
        "words": inspection.row_words,
        "candidate_analyses": inspection.row_candidates,
    }
    width = max(len(name) for name in rows)
    for name, count in rows.items():
        print(f"{name.ljust(width)}  rows: {count}")
    print()
    examples = inspection.examples
    print("WARNINGS")
    print(
        f"word_number=0: {inspection.zero_word_number} words in "
        f"{inspection.zero_word_number_sentences} sentences"
    )
    _print_class_examples(examples.get("word_number=0", ()))
    print(
        f"{'empty gramset'.ljust(width)}  candidate rows: {inspection.empty_gramset}"
    )
    if inspection.empty_gramset:
        print(
            f"{'  relevance'.ljust(width)}  "
            f"relevance=0: {inspection.empty_gramset_relevance_0}; "
            f"relevance=1: {inspection.empty_gramset_relevance_1}; "
            f"relevance=2: {inspection.empty_gramset_relevance_2}"
        )
    _print_class_examples(examples.get("empty gramset", ()))
    print("DUPLICATES")
    print(f"{'text_id'.ljust(width)}: {inspection.duplicate_text_id}")
    _print_class_examples(examples.get("duplicate text_id", ()))
    print(f"{'sentence_id'.ljust(width)}: {inspection.duplicate_sentence_id}")
    _print_class_examples(examples.get("duplicate sentence_id", ()))
    print(f"{'word_id'.ljust(width)}: {inspection.duplicate_word_id}")
    _print_class_examples(examples.get("duplicate word_id", ()))
    print(f"{'candidate tuple'.ljust(width)}: {inspection.duplicate_candidate_tuple}")
    _print_class_examples(examples.get("duplicate candidate tuple", ()))
    print("ORPHANS")
    print(
        f"{'sentences without text'.ljust(width)}: "
        f"{inspection.orphan_sentences_without_text}"
    )
    _print_class_examples(examples.get("sentences without text", ()))
    print(
        f"{'words without sentence'.ljust(width)}: "
        f"{inspection.orphan_words_without_sentence}"
    )
    _print_class_examples(examples.get("words without sentence", ()))
    print(
        f"{'candidates without word'.ljust(width)}: "
        f"{inspection.orphan_candidates_without_word}"
    )
    _print_class_examples(examples.get("candidates without word", ()))
    print("CANDIDATE GROUPS")
    print(f"{'words with 0 candidates'.ljust(width)}: {inspection.words_with_zero_candidates}")
    print(f"{'words with 1 candidate'.ljust(width)}: {inspection.words_with_one_candidate}")
    print(f"{'words with 2+ candidates'.ljust(width)}: {inspection.words_with_multiple_candidates}")
    print()
    print("EXPERT-SELECTION STATUS")
    print(f"{'words with 0 relevance=2'.ljust(width)}: {inspection.words_with_zero_selected}")
    print(f"{'words with 1 relevance=2'.ljust(width)}: {inspection.words_with_one_selected}")
    print(f"{'words with 2+ relevance=2'.ljust(width)}: {inspection.words_with_multiple_selected}")
    print()
    result = (
        "integrity issues found"
        if inspection.has_broken_integrity
        else "no integrity issues found"
    )
    print(f"Result: {result}; source data unchanged")


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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "fetch-data":
        try:
            created = fetch_data(args.tag)
        except FetchError as exc:
            parser.exit(1, f"error: {exc}\n")
        state = "Downloaded" if created else "Already present"
        print(f"{state}: dictorpus-data ({args.tag}) in data/dictorpus-data/")
        return 0
    if args.command == "inspect-data":
        data_dir = args.data_dir if args.data_dir is not None else DEFAULT_DATA_DIR
        try:
            require_local_corpus(data_dir)
            tables = read_corpus_tables(args.language, data_dir)
            inspection = inspect_corpus(tables, args.language)
        except (DataError, CorpusError) as exc:
            parser.exit(1, f"error: {exc}\n")
        _print_report(inspection, data_dir / "corpus")
        return 1 if inspection.has_broken_integrity else 0
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