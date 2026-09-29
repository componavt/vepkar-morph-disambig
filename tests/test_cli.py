import csv
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CLI = ROOT / "src" / "cli.py"
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "tests"))

from test_splits import GIT_AVAILABLE, _tagged_checkout  # noqa: E402
from core.data import SUPPORTED_LANGUAGES, read_corpus_tables  # noqa: E402
from core.instances import build_language_instances  # noqa: E402

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
def test_missing_default_checkout_offers_fetch_command(tmp_path, monkeypatch, capsys, args):
    monkeypatch.syspath_prepend(str(SRC))
    import cli
    import core.data as data

    missing = tmp_path / "missing-default-checkout"

    monkeypatch.setattr(data, "DEFAULT_DATA_DIR", missing)
    monkeypatch.setattr(cli, "DEFAULT_DATA_DIR", missing)

    try:
        status = cli.main(list(args))
    except SystemExit as exc:
        status = exc.code

    captured = capsys.readouterr()
    assert status == 1
    assert "dictorpus-data" in captured.err
    assert str(missing) in captured.err
    assert "fetch-data v2026.09" in captured.err
    assert "Traceback" not in captured.err
    assert "  vep  OK" not in captured.err
    assert "BENCHMARK" not in captured.out
    assert "SPLIT" not in captured.out


def test_incomplete_checkout_reports_missing_csv(tmp_path):
    checkout = tmp_path / "checkout"
    (checkout / "corpus").mkdir(parents=True)
    result = run_cli("inspect-data", "krl", "--data-dir", str(checkout))
    assert result.returncode == 1
    assert "texts_krl.csv.zst" in result.stderr
    assert "no corpus" not in result.stderr
    assert "Traceback" not in result.stderr


def _strict_instances(data_dir: Path):
    instances = []
    for lang in SUPPORTED_LANGUAGES:
        tables = read_corpus_tables(lang, data_dir)
        instances.extend(build_language_instances(lang, tables).instances)
    return instances


def _write_split_fixture(path: Path, instances, part: str) -> None:
    keys = sorted({(inst.language, int(inst.text_id)) for inst in instances})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["language", "text_id", "split"])
        writer.writerows((lang, text_id, part) for lang, text_id in keys)


def _write_valid_predictions(path: Path, instances) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["word_id", "wordform_id", "gramset", "rank", "score"])
        for instance in sorted(
            instances, key=lambda inst: (inst.language, inst.word_id)
        ):
            for index, candidate in enumerate(instance.candidates, start=1):
                writer.writerow(
                    [
                        instance.word_id,
                        candidate.wordform_id,
                        candidate.gramset,
                        index,
                        len(instance.candidates) - index + 1,
                    ]
                )


def _expected_counts(instances) -> tuple[int, int]:
    return len(instances), sum(len(inst.candidates) for inst in instances)


def _run_validate(tmp_path: Path, part: str) -> subprocess.CompletedProcess:
    checkout = _tagged_checkout(tmp_path)
    instances = _strict_instances(checkout)
    split_file = tmp_path / f"splits_{part}.csv"
    _write_split_fixture(split_file, instances, part)
    predictions = tmp_path / f"predictions_{part}.csv"
    _write_valid_predictions(predictions, instances)
    result = run_cli(
        "validate-predictions",
        "--predictions", str(predictions),
        "--split", part,
        "--data-dir", str(checkout),
        "--split-file", str(split_file),
    )
    return result, instances


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_validate_predictions_dev_ok(tmp_path):
    result, instances = _run_validate(tmp_path, "dev")
    word_count, candidate_count = _expected_counts(instances)
    assert result.returncode == 0, result.stderr
    assert "Predictions validation: OK" in result.stdout
    assert "Split: dev" in result.stdout
    assert (
        f"Validated: {word_count} word instances, {candidate_count} candidate rows"
        in result.stdout
    )
    assert "Traceback" not in result.stderr


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_validate_predictions_test_ok(tmp_path):
    result, instances = _run_validate(tmp_path, "test")
    word_count, candidate_count = _expected_counts(instances)
    assert result.returncode == 0, result.stderr
    assert "Predictions validation: OK" in result.stdout
    assert "Split: test" in result.stdout
    assert (
        f"Validated: {word_count} word instances, {candidate_count} candidate rows"
        in result.stdout
    )
    assert "Traceback" not in result.stderr


def test_validate_predictions_rejects_train(tmp_path):
    result = run_cli(
        "validate-predictions",
        "--predictions", str(tmp_path / "predictions.csv"),
        "--split", "train",
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr
    assert "train" in result.stderr


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_validate_predictions_creates_no_derived_output(tmp_path):
    def snapshot() -> set[tuple[str, str]]:
        files: set[tuple[str, str]] = set()
        for root in (ROOT / "results", ROOT / "data" / "derived"):
            if root.is_dir():
                for path in root.rglob("*"):
                    if path.is_file():
                        files.add((str(root), str(path.relative_to(root))))
        return files

    checkout = _tagged_checkout(tmp_path)
    instances = _strict_instances(checkout)
    split_file = tmp_path / "splits_dev.csv"
    _write_split_fixture(split_file, instances, "dev")
    predictions = tmp_path / "predictions_dev.csv"
    _write_valid_predictions(predictions, instances)
    before = snapshot()
    result = run_cli(
        "validate-predictions",
        "--predictions", str(predictions),
        "--split", "dev",
        "--data-dir", str(checkout),
        "--split-file", str(split_file),
    )
    assert result.returncode == 0, result.stderr
    assert snapshot() == before


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_validate_predictions_missing_predictions_file(tmp_path):
    checkout = _tagged_checkout(tmp_path)
    instances = _strict_instances(checkout)
    split_file = tmp_path / "splits_dev.csv"
    _write_split_fixture(split_file, instances, "dev")
    result = run_cli(
        "validate-predictions",
        "--predictions", str(tmp_path / "missing.csv"),
        "--split", "dev",
        "--data-dir", str(checkout),
        "--split-file", str(split_file),
    )
    assert result.returncode == 1
    assert "error:" in result.stderr
    assert "cannot read predictions file" in result.stderr
    assert "Traceback" not in result.stderr