"""Fetch a tagged source-data checkout; never modify an existing checkout."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

SOURCE_URL = "https://github.com/componavt/dictorpus-data.git"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "dictorpus-data"


class FetchError(Exception):
    """A source-data checkout could not be safely obtained."""


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True
        ).stdout.strip()
    except FileNotFoundError as exc:
        raise FetchError("Git is not installed or is not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise FetchError(f"Git failed: {detail}") from exc


def _verify_checkout(path: Path, tag: str, repository_url: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise FetchError(f"Destination is not a regular directory: {path}")
    location = str(path)
    origin = _git("-C", location, "remote", "get-url", "origin")
    if origin != repository_url:
        raise FetchError(f"Destination has a different Git origin: {origin}")
    head = _git("-C", location, "rev-parse", "HEAD")
    tagged = _git("-C", location, "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}")
    if head != tagged:
        raise FetchError(f"Destination is not checked out at tag {tag}")
    if _git("-C", location, "status", "--porcelain"):
        raise FetchError(f"Destination has local changes: {path}")


def fetch_data(
    tag: str, *, repository_url: str = SOURCE_URL, destination: Path | None = None
) -> bool:
    """Fetch a tagged checkout. Return True if created, False if already present."""
    if not tag or tag.startswith("-"):
        raise FetchError("Provide a valid Git tag")
    _git("check-ref-format", f"refs/tags/{tag}")
    target = Path(destination) if destination is not None else DATA_DIR
    if target.exists() or target.is_symlink():
        _verify_checkout(target, tag, repository_url)
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dictorpus-data-", dir=target.parent) as temp:
        checkout = Path(temp) / "checkout"
        _git(
            "clone", "--depth", "1", "--branch", tag,
            "--", repository_url, str(checkout),
        )
        _verify_checkout(checkout, tag, repository_url)
        if target.exists() or target.is_symlink():
            raise FetchError(f"Destination appeared during download: {target}")
        checkout.rename(target)
    return True
