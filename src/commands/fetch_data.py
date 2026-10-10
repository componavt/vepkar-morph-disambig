"""Clone a tagged dictorpus-data release locally."""

from __future__ import annotations

import argparse

from core.fetch import FetchError, fetch_data


def _run_fetch_data(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    try:
        created = fetch_data(args.tag)
    except FetchError as exc:
        parser.exit(1, f"error: {exc}\n")
    state = "Downloaded" if created else "Already present"
    print(f"{state}: dictorpus-data ({args.tag}) in data/dictorpus-data/")
    return 0