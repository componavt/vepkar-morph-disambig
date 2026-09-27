"""Command-line interface for vepkar-morph-disambig."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from core.data import (
    DEFAULT_DATA_DIR,
    SUPPORTED_LANGUAGES,
    DataError,
    read_corpus_tables,
    resolve_data_dir,
)
from core.fetch import FetchError, fetch_data
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
from core.validation import CorpusError, inspect_corpus

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
    return parser


def _print_examples(examples: tuple) -> None:
    for example in examples:
        print(f"  {example.detail}")
        xml = example.sentence_xml.replace("\r", " ").replace("\n", " ")
        print(f"  sentence_xml: {xml}")


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
    print("WARNINGS")
    print(
        f"{'word_number=0'.ljust(width)}  words: {inspection.zero_word_number}"
    )
    _print_examples(inspection.examples.get("word_number=0", ()))
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
    _print_examples(inspection.examples.get("empty gramset", ()))
    print()
    print("DUPLICATES")
    print(f"{'text_id'.ljust(width)}: {inspection.duplicate_text_id}")
    _print_examples(inspection.examples.get("duplicate text_id", ()))
    print(f"{'sentence_id'.ljust(width)}: {inspection.duplicate_sentence_id}")
    _print_examples(inspection.examples.get("duplicate sentence_id", ()))
    print(f"{'word_id'.ljust(width)}: {inspection.duplicate_word_id}")
    _print_examples(inspection.examples.get("duplicate word_id", ()))
    print(f"{'candidate tuple'.ljust(width)}: {inspection.duplicate_candidate_tuple}")
    _print_examples(inspection.examples.get("duplicate candidate tuple", ()))
    print()
    print("ORPHANS")
    print(
        f"{'sentences without text'.ljust(width)}: "
        f"{inspection.orphan_sentences_without_text}"
    )
    _print_examples(inspection.examples.get("sentences without text", ()))
    print(
        f"{'words without sentence'.ljust(width)}: "
        f"{inspection.orphan_words_without_sentence}"
    )
    _print_examples(inspection.examples.get("words without sentence", ()))
    print(
        f"{'candidates without word'.ljust(width)}: "
        f"{inspection.orphan_candidates_without_word}"
    )
    _print_examples(inspection.examples.get("candidates without word", ()))
    print()
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


def _print_language_report(result: LanguageInstances) -> None:
    print()
    print(f"=== {result.language} ===")
    print("FUNNEL (word instances)")
    for step in result.funnel:
        print(f"  step {step.index}  {step.name}")
        print(f"      retained: {step.retained}")
        if step.index > 0:
            print(f"      removed at this transition: {step.removed}")
    print("OVERLAP DIAGNOSTICS (among step-1 words)")
    print(
        "  duplicate candidate identities:             "
        f"{result.step1_words_with_duplicate_identity}"
    )
    print(
        "  selected analysis with empty gramset:       "
        f"{result.step1_words_with_selected_empty_gramset}"
    )
    print(
        "  unselected analysis with empty gramset:     "
        f"{result.step1_words_with_unselected_empty_gramset}"
    )
    print(
        "  unavailable sentence or text:               "
        f"{result.step1_words_with_unavailable_sentence}"
    )
    print(
        "  zero-position sentence:                     "
        f"{result.step1_words_with_zero_position_sentence}"
    )
    print(
        "  repeated positive-position sentence:        "
        f"{result.step1_words_with_repeated_position_sentence}"
    )
    print(f"ZERO-POSITION SENTENCES: {len(result.zero_position_sentences)}")
    print("EMPTY UNSELECTED GRAMSET POLICY")
    print(
        "  primary instances with >=1 unselected empty gramset: "
        f"{result.primary_words_with_unselected_empty_gramset}"
    )
    print(f"  primary pool size: {len(result.instances)}")
    print(f"  alternative pool size: {result.alternative_primary_pool_size}")


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
            tag = determine_data_tag(data_dir)
            print(f"Source tag: {tag}")
            results = {}
            total_primary = 0
            for lang in SUPPORTED_LANGUAGES:
                tables = read_corpus_tables(lang, data_dir)
                result = build_language_instances(lang, tables)
                results[lang] = result
                total_primary += len(result.instances)
                _print_language_report(result)
            rows = build_review_rows(results.values())
            path = review_csv_path(output_dir, tag)
            written = write_review_csv(path, rows)
        except (DataError, CorpusTagError) as exc:
            parser.exit(1, f"error: {exc}\n")
        print()
        print(f"Total primary instances (step 6): {total_primary}")
        print(f"Review CSV: {path} ({written} rows)")
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())