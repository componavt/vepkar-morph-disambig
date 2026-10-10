"""Inspect the typed local corpus tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from core.data import (
    DEFAULT_DATA_DIR,
    DataError,
    read_corpus_tables,
    require_local_corpus,
)
from core.validation import CorpusError, inspect_corpus


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


def _run_inspect_data(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    data_dir = args.data_dir if args.data_dir is not None else DEFAULT_DATA_DIR
    try:
        require_local_corpus(data_dir)
        tables = read_corpus_tables(args.language, data_dir)
        inspection = inspect_corpus(tables, args.language)
    except (DataError, CorpusError) as exc:
        parser.exit(1, f"error: {exc}\n")
    _print_report(inspection, data_dir / "corpus")
    return 1 if inspection.has_broken_integrity else 0