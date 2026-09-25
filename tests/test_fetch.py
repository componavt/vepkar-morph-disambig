"""Offline tests for tagged source-data checkout."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from core.fetch import FetchError, fetch_data  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="Git unavailable")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def source_repo(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    git("init", "-q", str(source))
    (source / "README.md").write_text("source export\n", encoding="utf-8")
    git("-C", str(source), "add", "README.md")
    git(
        "-C", str(source), "-c", "user.name=Test", "-c", "user.email=test@example.org",
        "commit", "-qm", "Initial export",
    )
    git("-C", str(source), "tag", "v2026.09")
    return source


def test_fetch_tag_and_repeat_without_changes(source_repo: Path, tmp_path: Path):
    target = tmp_path / "data" / "dictorpus-data"
    url = source_repo.as_uri()
    assert fetch_data("v2026.09", repository_url=url, destination=target) is True
    assert (target / "README.md").read_text(encoding="utf-8") == "source export\n"
    assert git("-C", str(target), "rev-parse", "HEAD") == git(
        "-C", str(target), "rev-parse", "refs/tags/v2026.09^{}"
    )
    assert fetch_data("v2026.09", repository_url=url, destination=target) is False


def test_missing_tag_does_not_leave_checkout(source_repo: Path, tmp_path: Path):
    target = tmp_path / "data" / "dictorpus-data"
    with pytest.raises(FetchError):
        fetch_data("v2099.01", repository_url=source_repo.as_uri(), destination=target)
    assert not target.exists()


def test_different_tag_does_not_replace_existing_checkout(source_repo: Path, tmp_path: Path):
    target = tmp_path / "dictorpus-data"
    url = source_repo.as_uri()
    fetch_data("v2026.09", repository_url=url, destination=target)
    git("-C", str(source_repo), "tag", "v2027.02")
    with pytest.raises(FetchError):
        fetch_data("v2027.02", repository_url=url, destination=target)
    assert git("-C", str(target), "rev-parse", "HEAD") == git(
        "-C", str(target), "rev-parse", "refs/tags/v2026.09^{}"
    )


def test_existing_non_checkout_is_preserved(source_repo: Path, tmp_path: Path):
    target = tmp_path / "dictorpus-data"
    target.mkdir()
    sentinel = target / "keep.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    with pytest.raises(FetchError):
        fetch_data("v2026.09", repository_url=source_repo.as_uri(), destination=target)
    assert sentinel.read_text(encoding="utf-8") == "unchanged"


def test_dirty_checkout_is_rejected(source_repo: Path, tmp_path: Path):
    target = tmp_path / "dictorpus-data"
    url = source_repo.as_uri()
    fetch_data("v2026.09", repository_url=url, destination=target)
    (target / "README.md").write_text("locally modified\n", encoding="utf-8")
    with pytest.raises(FetchError, match="local changes"):
        fetch_data("v2026.09", repository_url=url, destination=target)


def test_cli_requires_tag():
    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / "cli.py"), "fetch-data"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode != 0
    assert "tag" in result.stderr.lower()
