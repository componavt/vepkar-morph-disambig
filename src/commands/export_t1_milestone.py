"""Export one finished T1.1 dev run into a compact milestone directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from t1_features.milestone_export import (
    T1MilestoneExportError,
    export_t1_milestone,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_milestones_dir() -> Path:
    return _PROJECT_ROOT / "results" / "milestones"


def _run_export_t1_milestone(args: argparse.Namespace) -> int:
    resolved_run_dir = Path(args.run_dir).expanduser().resolve()
    try:
        destination = export_t1_milestone(
            resolved_run_dir,
            milestones_dir=_default_milestones_dir(),
        )
    except T1MilestoneExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        for note in getattr(exc, "__notes__", ()):
            print(f"  {note}", file=sys.stderr)
        return 1
    print("T1 milestone export: OK")
    print(f"Run directory: {resolved_run_dir}")
    print(f"Milestone directory: {destination}")
    return 0