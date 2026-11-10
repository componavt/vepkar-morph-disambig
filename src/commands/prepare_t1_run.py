"""Prepare provenance metadata for a forthcoming T1 dev experiment."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from t1_features.config import T1ConfigError, load_t1_config
from t1_features.run_manifest import RunManifestError, load_run_manifest
from t1_features.workspace import T1WorkspaceError, create_t1_run

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run_prepare_t1_run(args: argparse.Namespace) -> int:
    C = args.C
    try:
        valid = not isinstance(C, bool) and math.isfinite(C) and C > 0
    except OverflowError:
        valid = False
    if not valid:
        print("error: C must be finite and positive:", file=sys.stderr)
        print(f"  {C!r}", file=sys.stderr)
        return 1
    suffix_max_length = getattr(args, "suffix_max_length", 4)
    if type(suffix_max_length) is not int or suffix_max_length < 0:
        print("error: suffix maximum length must be a nonnegative integer:", file=sys.stderr)
        print(f"  {suffix_max_length!r}", file=sys.stderr)
        return 1
    include_affix = getattr(args, "include_affix", False)
    try:
        config = load_t1_config(
            _PROJECT_ROOT,
            data_dir=args.data_dir,
            split_file=args.split_file,
        )
    except T1ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not config.config_found:
        print(
            "Local configuration not found: "
            f"{config.config_path}; using defaults and explicit overrides.",
            file=sys.stderr,
        )
    try:
        created = create_t1_run(
            config,
            C=C,
            suffix_max_length=suffix_max_length,
            include_affix=include_affix,
            run_dir=args.run_dir,
        )
    except T1WorkspaceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        for note in getattr(exc, "__notes__", ()):
            print(f"  {note}", file=sys.stderr)
        return 1
    try:
        manifest = load_run_manifest(created / "run.json")
    except RunManifestError as exc:
        print(
            "error: run directory was created but its passport could not "
            f"be read for reporting: {created}",
            file=sys.stderr,
        )
        print(f"  {exc}", file=sys.stderr)
        return 1
    print("T1 run prepared")
    print(f"Run directory: {created}")
    print(f"Data tag: {manifest['data']['tag']}")
    print(f"Split file: {manifest['split']['path']}")
    print(f"C: {manifest['parameters']['C']}")
    print(f"Target split: {manifest['parameters']['target_split']}")
    print(f"Suffix maximum length: {manifest['parameters']['suffix_max_length']}")
    print(f"Dictionary affix feature: {manifest['parameters']['include_affix']}")
    return 0
