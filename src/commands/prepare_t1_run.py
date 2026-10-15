"""Prepare provenance metadata for a forthcoming T1 dev experiment."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from benchmark_context import _preflight_benchmark
from core.data import DataError
from core.instances import CorpusTagError
from t1_features.run_manifest import (
    RunManifestError,
    collect_run_manifest,
    write_run_manifest,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_DIRTY_WARNING = (
    "Recorded local changes; commit IDs alone do not identify\n"
    "the exact executed code/data. Review changes before the experiment."
)


def _run_prepare_t1_run(args: argparse.Namespace) -> int:
    C = args.C
    if not math.isfinite(C) or C <= 0:
        print("error: C must be finite and positive:", file=sys.stderr)
        print(f"  {C!r}", file=sys.stderr)
        return 1
    output_path = Path(args.output)
    if output_path.is_dir():
        print("error: output path is a directory:", file=sys.stderr)
        print(f"  {output_path}", file=sys.stderr)
        return 1
    if output_path.exists() or output_path.is_symlink():
        print("error: run passport already exists:", file=sys.stderr)
        print(f"  {output_path}", file=sys.stderr)
        return 1
    if not output_path.parent.is_dir():
        print("error: output directory does not exist:", file=sys.stderr)
        print(f"  {output_path.parent}", file=sys.stderr)
        return 1
    try:
        data_dir, tag = _preflight_benchmark(args.data_dir)
    except (CorpusTagError, DataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        manifest = collect_run_manifest(
            code_dir=_PROJECT_ROOT,
            data_dir=data_dir,
            data_tag=tag,
            split_file=Path(args.split_file),
            C=C,
        )
    except RunManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        for note in getattr(exc, "__notes__", ()):
            print(f"  {note}", file=sys.stderr)
        return 1
    if manifest["code"]["dirty"] or manifest["data"]["dirty"]:
        print(_DIRTY_WARNING, file=sys.stderr)
    try:
        write_run_manifest(manifest, output_path)
    except RunManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        for note in getattr(exc, "__notes__", ()):
            print(f"  {note}", file=sys.stderr)
        return 1
    print("T1 run metadata prepared")
    print(f"Run passport: {output_path}")
    print(f"Code commit: {manifest['code']['commit']}")
    print(f"Data tag: {manifest['data']['tag']}")
    print(f"Data commit: {manifest['data']['commit']}")
    print(f"Split SHA-256: {manifest['split']['sha256']}")
    print(f"C: {C}")
    print("Target split: dev")
    return 0