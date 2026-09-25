"""Command-line interface for vepkar-morph-disambig."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = _PROJECT_ROOT / "pyproject.toml"


def _version() -> str:
    with _PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)
    return data["project"]["version"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vepkar-morph-disambig",
        description=(
            "Ranking candidate morphological analyses of words in VepKar "
            "context. Future subcommands will handle corpus data fetching, "
            "validation, experiments, and comparison."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {_version()}"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())