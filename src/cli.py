"""Command-line interface for vepkar-morph-disambig."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from core.data import DEFAULT_DATA_DIR, DataError, read_corpus_tables
from core.fetch import FetchError, fetch_data
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
    return parser


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
    print("DUPLICATES")
    print(f"{'text_id'.ljust(width)}: {inspection.duplicate_text_id}")
    print(f"{'sentence_id'.ljust(width)}: {inspection.duplicate_sentence_id}")
    print(f"{'word_id'.ljust(width)}: {inspection.duplicate_word_id}")
    print(f"{'candidate tuple'.ljust(width)}: {inspection.duplicate_candidate_tuple}")
    print()
    print("ORPHANS")
    print(f"{'sentences without text'.ljust(width)}: {inspection.orphan_sentences_without_text}")
    print(f"{'words without sentence'.ljust(width)}: {inspection.orphan_words_without_sentence}")
    print(f"{'candidates without word'.ljust(width)}: {inspection.orphan_candidates_without_word}")
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
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())