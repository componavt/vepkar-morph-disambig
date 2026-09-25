import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLI = ROOT / "src" / "cli.py"
SRC = ROOT / "src"

SOURCE_DIRS = ("core", "t1_features", "t2_context", "t3_transfer")


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def declared_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)["project"]["version"]


def test_help_exits_zero():
    result = run_cli("--help")
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()
    assert "morphological" in result.stdout


def test_version_matches_pyproject():
    result = run_cli("--version")
    assert result.returncode == 0
    assert declared_version() in result.stdout


def test_no_arguments_prints_help():
    result = run_cli()
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()


def test_unknown_argument_fails():
    result = run_cli("--bogus")
    assert result.returncode != 0
    assert "bogus" in result.stderr


def test_source_directories_exist_without_package_dir():
    for name in SOURCE_DIRS:
        assert (SRC / name).is_dir()
    assert not (SRC / "vepkar_morph_disambig").exists()