import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

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


def _assert_missing_checkout(result: subprocess.CompletedProcess, missing: Path) -> None:
    assert result.returncode == 1
    assert "dictorpus-data" in result.stderr
    assert str(missing) in result.stderr
    assert "Traceback" not in result.stderr
    assert "  vep  OK" not in result.stderr
    assert "BENCHMARK" not in result.stdout
    assert "SPLIT" not in result.stdout


@pytest.mark.parametrize(
    "args",
    [
        ("inspect-data", "krl"),
        ("build-instances",),
        ("make-splits",),
    ],
)
def test_missing_custom_data_dir_fails_all_three(tmp_path, args):
    missing = tmp_path / "missing-dictorpus-data"
    result = run_cli(*args, "--data-dir", str(missing))
    _assert_missing_checkout(result, missing)
    assert "fetch-data" not in result.stderr


@pytest.mark.parametrize(
    "args",
    [
        ("inspect-data", "krl"),
        ("build-instances",),
        ("make-splits",),
    ],
)
def test_missing_default_checkout_offers_fetch_command(args):
    result = run_cli(*args)
    assert result.returncode == 1
    assert "dictorpus-data" in result.stderr
    assert "fetch-data v2026.09" in result.stderr
    assert "Traceback" not in result.stderr
    assert "  vep  OK" not in result.stderr
    assert "BENCHMARK" not in result.stdout
    assert "SPLIT" not in result.stdout


def test_incomplete_checkout_reports_missing_csv(tmp_path):
    checkout = tmp_path / "checkout"
    (checkout / "corpus").mkdir(parents=True)
    result = run_cli("inspect-data", "krl", "--data-dir", str(checkout))
    assert result.returncode == 1
    assert "texts_krl.csv.zst" in result.stderr
    assert "no corpus" not in result.stderr
    assert "Traceback" not in result.stderr