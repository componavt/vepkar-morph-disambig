"""Assign whole texts to train/dev/test by candidate-row weights."""

from __future__ import annotations

import argparse
import sys

from core.data import (
    SUPPORTED_LANGUAGES,
    DataError,
    read_corpus_tables,
    require_local_corpus,
    resolve_data_dir,
)
from core.instances import (
    CorpusTagError,
    LanguageInstances,
    build_language_instances,
    determine_data_tag,
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
    validate_split_assignments,
    write_split_csv,
)

from commands.benchmark_reports import _print_split_report


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