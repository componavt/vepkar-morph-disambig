"""Shared benchmark preparation: strict instances and split rows for one tag."""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

from core.data import (
    DataError,
    SUPPORTED_LANGUAGES,
    read_corpus_tables,
    require_local_corpus,
    resolve_data_dir,
)
from core.instances import (
    Instance,
    build_language_instances,
    determine_data_tag,
)
from core.splits import (
    DEFAULT_SPLIT_OUTPUT_DIR,
    SPLIT_CSV_HEADER,
    SplitError,
    split_csv_path,
)


def _read_split_rows(path: Path) -> tuple[tuple[str, int, str], ...]:
    """Read ``(language, text_id, split)`` rows from a published splits CSV."""
    rows: list[tuple[str, int, str]] = []
    try:
        with open(path, encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh, strict=True)
            header = next(reader, None)
            if tuple(header or ()) != SPLIT_CSV_HEADER:
                raise SplitError(
                    f"cannot parse split CSV: {path}: unexpected header {header!r}"
                )
            for raw in reader:
                if len(raw) != 3:
                    raise SplitError(
                        f"cannot parse split CSV: {path}: invalid row {raw!r}"
                    )
                language, text_id, split = raw
                try:
                    parsed_text_id = int(text_id)
                except ValueError as exc:
                    raise SplitError(
                        f"cannot parse split CSV: {path}: invalid text_id in row {raw!r}"
                    ) from exc
                rows.append((language, parsed_text_id, split))
    except csv.Error as exc:
        raise SplitError(f"cannot parse split CSV: {path}: {exc}") from exc
    except OSError as exc:
        raise SplitError(f"cannot read split file: {path}: {exc}") from exc
    return tuple(rows)


@dataclass(frozen=True)
class BenchmarkContext:
    """Read-only strict instances and split rows for one benchmark tag."""

    tag: str
    split_path: Path
    instances: tuple[Instance, ...]
    split_rows: tuple[tuple[str, int, str], ...]


def _preflight_benchmark(
    data_dir_override: Path | None,
) -> tuple[Path, str]:
    """Resolve checkout, require it locally, and determine its tag."""
    data_dir = resolve_data_dir(data_dir_override)
    require_local_corpus(data_dir)
    tag = determine_data_tag(data_dir)
    return data_dir, tag


def load_benchmark_context(
    data_dir: Path,
    tag: str,
    split_file_override: Path | None = None,
) -> BenchmarkContext:
    """Load strict instances and split rows without writing anything."""
    instances: list[Instance] = []
    for lang in SUPPORTED_LANGUAGES:
        try:
            tables = read_corpus_tables(lang, data_dir)
            result = build_language_instances(lang, tables)
        except DataError as exc:
            print(f"  {lang}  FAILED: {exc}", file=sys.stderr, flush=True)
            raise
        instances.extend(result.instances)
    split_path = (
        split_file_override
        if split_file_override is not None
        else split_csv_path(DEFAULT_SPLIT_OUTPUT_DIR, tag)
    )
    split_rows = _read_split_rows(split_path)
    return BenchmarkContext(
        tag=tag,
        split_path=split_path,
        instances=tuple(instances),
        split_rows=split_rows,
    )
