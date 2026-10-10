"""Build benchmark instances and the developer sentence-review CSV."""

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
    DEFAULT_OUTPUT_DIR,
    CorpusTagError,
    build_language_instances,
    build_review_rows,
    determine_data_tag,
    review_csv_path,
    write_review_csv,
)

from commands.benchmark_reports import _print_build_summary


def _run_build_instances(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
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