import argparse
import csv
import subprocess
import sys
import tomllib
from collections import Counter
from dataclasses import astuple
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
CLI = ROOT / "src" / "cli.py"
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "tests"))

from test_splits import GIT_AVAILABLE, _tagged_checkout  # noqa: E402
from core.data import SUPPORTED_LANGUAGES, read_corpus_tables  # noqa: E402
from core.diagnostics import build_frequency_diagnostics  # noqa: E402
from core.frequency import (  # noqa: E402
    FrequencyPrediction,
    build_train_frequency,
    rank_by_train_frequency,
)
from core.instances import Candidate, Instance, build_language_instances  # noqa: E402
from core.predictions import validate_predictions  # noqa: E402
from frequency_diagnostics_io import load_verified_frequency_predictions  # noqa: E402

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


def _valid_prediction_rows(instances) -> list[list]:
    rows = []
    for instance in sorted(
        instances, key=lambda inst: (inst.language, inst.word_id)
    ):
        for index, candidate in enumerate(instance.candidates, start=1):
            rows.append(
                [
                    instance.word_id,
                    candidate.wordform_id,
                    candidate.gramset,
                    index,
                    len(instance.candidates) - index + 1,
                ]
            )
    return rows


def _write_valid_predictions(path: Path, instances) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["word_id", "wordform_id", "gramset", "rank", "score"])
        writer.writerows(_valid_prediction_rows(instances))


def _write_predictions(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["word_id", "wordform_id", "gramset", "rank", "score"])
        writer.writerows(rows)


def _run_validate_file(tmp_path: Path, part: str, build_rows):
    checkout = _tagged_checkout(tmp_path)
    instances = _strict_instances(checkout)
    built = build_rows(instances)
    if isinstance(built, tuple):
        rows, split_rows = built
    else:
        rows, split_rows = built, None
    split_file = tmp_path / f"splits_{part}.csv"
    if split_rows is None:
        _write_split_fixture(split_file, instances, part)
    else:
        with split_file.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["language", "text_id", "split"])
            writer.writerows(split_rows)
    predictions = tmp_path / f"predictions_{part}.csv"
    _write_predictions(predictions, rows)
    result = run_cli(
        "validate-predictions",
        "--predictions", str(predictions),
        "--split", part,
        "--data-dir", str(checkout),
        "--split-file", str(split_file),
    )
    return result, instances


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
    missing = tmp_path / "missing.csv"
    result = run_cli(
        "validate-predictions",
        "--predictions", str(missing),
        "--split", "dev",
        "--data-dir", str(checkout),
        "--split-file", str(split_file),
    )
    assert result.returncode == 1
    assert "cannot read predictions file" in result.stderr
    assert str(missing) in result.stderr
    assert "Predictions validation: OK" not in result.stdout
    assert "Predictions validation: FAILED" not in result.stdout
    assert "Traceback" not in result.stderr
    assert "Traceback" not in result.stdout


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_validate_predictions_directory_path(tmp_path):
    checkout = _tagged_checkout(tmp_path)
    instances = _strict_instances(checkout)
    split_file = tmp_path / "splits_dev.csv"
    _write_split_fixture(split_file, instances, "dev")
    directory = tmp_path / "predictions_dir"
    directory.mkdir()
    result = run_cli(
        "validate-predictions",
        "--predictions", str(directory),
        "--split", "dev",
        "--data-dir", str(checkout),
        "--split-file", str(split_file),
    )
    assert result.returncode == 1
    assert "cannot read predictions file" in result.stderr
    assert str(directory) in result.stderr
    assert "Errno" in result.stderr
    assert "Predictions validation" not in result.stdout
    assert "Traceback" not in result.stderr


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_validate_predictions_unreadable_bypasses_benchmark_loading(
    tmp_path, monkeypatch, capsys, kind
):
    import cli as cli_pkg

    checkout = _tagged_checkout(tmp_path)
    if kind == "directory":
        predictions = tmp_path / "predictions_dir"
        predictions.mkdir()
    else:
        predictions = tmp_path / "missing.csv"

    def forbidden_load(*args, **kwargs):
        pytest.fail("Benchmark loading must not occur for an unreadable prediction path")

    monkeypatch.setattr(cli_pkg, "load_benchmark_context", forbidden_load)
    status = cli_pkg._run_validate_predictions(
        argparse.Namespace(
            predictions=predictions,
            split="dev",
            data_dir=checkout,
            split_file=tmp_path / "splits_dev.csv",
        )
    )
    captured = capsys.readouterr()
    assert status == 1
    assert "cannot read predictions file" in captured.err
    assert str(predictions) in captured.err
    assert "OK" not in captured.out
    assert "FAILED" not in captured.out


class _RecordingPath(Path):
    opened = False

    def open(self, *args, **kwargs):
        _RecordingPath.opened = True
        return super().open(*args, **kwargs)


def test_validate_predictions_missing_checkout_priority_skips_open(
    tmp_path, monkeypatch, capsys
):
    import cli as cli_pkg

    missing_checkout = tmp_path / "missing-dictorpus-data"
    _RecordingPath.opened = False
    predictions = _RecordingPath(tmp_path / "missing-predictions.csv")

    def forbidden_load(*args, **kwargs):
        pytest.fail("Benchmark loading must not occur")

    monkeypatch.setattr(cli_pkg, "load_benchmark_context", forbidden_load)
    status = cli_pkg._run_validate_predictions(
        argparse.Namespace(
            predictions=predictions,
            split="dev",
            data_dir=missing_checkout,
            split_file=tmp_path / "missing-splits.csv",
        )
    )
    captured = capsys.readouterr()
    assert status == 1
    assert "dictorpus-data" in captured.err
    assert str(missing_checkout) in captured.err
    assert "cannot read predictions file" not in captured.err
    assert _RecordingPath.opened is False
    assert "Predictions validation" not in captured.out


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_validate_predictions_malformed_csv(tmp_path):
    checkout = _tagged_checkout(tmp_path)
    instances = _strict_instances(checkout)
    split_file = tmp_path / "splits_dev.csv"
    _write_split_fixture(split_file, instances, "dev")
    predictions = tmp_path / "malformed.csv"
    predictions.write_text(
        "word_id,wordform_id,gramset,rank,score\n"
        '501,9001,"SG+NOM,1,125\n',
        encoding="utf-8",
    )
    result = run_cli(
        "validate-predictions",
        "--predictions", str(predictions),
        "--split", "dev",
        "--data-dir", str(checkout),
        "--split-file", str(split_file),
    )
    assert result.returncode == 1
    assert "cannot parse predictions CSV" in result.stderr
    assert str(predictions) in result.stderr
    assert "OK" not in result.stdout
    assert "FAILED" not in result.stdout
    assert "Traceback" not in result.stderr
    assert "Traceback" not in result.stdout


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_validate_predictions_invalid_file_reports_failed(tmp_path):
    checkout = _tagged_checkout(tmp_path)
    instances = _strict_instances(checkout)
    split_file = tmp_path / "splits_dev.csv"
    _write_split_fixture(split_file, instances, "dev")
    predictions = tmp_path / "predictions_invalid.csv"
    _write_valid_predictions(predictions, instances)
    lines = predictions.read_text(encoding="utf-8").splitlines()
    predictions.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    word_count, candidate_count = _expected_counts(instances)
    result = run_cli(
        "validate-predictions",
        "--predictions", str(predictions),
        "--split", "dev",
        "--data-dir", str(checkout),
        "--split-file", str(split_file),
    )
    assert result.returncode == 1
    assert "Predictions validation: FAILED" in result.stdout
    assert "Split: dev" in result.stdout
    assert (
        f"Expected: {word_count} word instances, {candidate_count} candidate rows"
        in result.stdout
    )
    assert "Found:" in result.stdout
    assert "Traceback" not in result.stderr


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_validate_predictions_missing_split_file(tmp_path):
    checkout = _tagged_checkout(tmp_path)
    missing_split = tmp_path / "missing-splits.csv"
    predictions = tmp_path / "predictions.csv"
    predictions.write_text(
        "word_id,wordform_id,gramset,rank,score\n", encoding="utf-8"
    )
    result = run_cli(
        "validate-predictions",
        "--predictions", str(predictions),
        "--split", "dev",
        "--data-dir", str(checkout),
        "--split-file", str(missing_split),
    )
    assert result.returncode == 1
    assert "cannot read split file" in result.stderr
    assert str(missing_split) in result.stderr
    assert "OK" not in result.stdout
    assert "FAILED" not in result.stdout
    assert "Traceback" not in result.stderr
    assert "Traceback" not in result.stdout


def test_validate_predictions_missing_checkout_checked_first(tmp_path):
    missing = tmp_path / "missing-dictorpus-data"
    result = run_cli(
        "validate-predictions",
        "--predictions", str(tmp_path / "missing-predictions.csv"),
        "--split", "dev",
        "--data-dir", str(missing),
        "--split-file", str(tmp_path / "missing-splits.csv"),
    )
    assert result.returncode == 1
    assert "dictorpus-data" in result.stderr
    assert str(missing) in result.stderr
    assert "cannot read predictions file" not in result.stderr
    assert "split file" not in result.stderr
    assert "Predictions validation" not in result.stdout
    assert "Traceback" not in result.stderr
    assert "Traceback" not in result.stdout


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_validate_predictions_benchmark_integrity_failure(tmp_path, monkeypatch, capsys):
    import cli as cli_pkg
    from core.instances import Candidate, Instance

    checkout = _tagged_checkout(tmp_path)
    split_file = tmp_path / "splits_dev.csv"
    split_file.write_text("language,text_id,split\nkrl,1,dev\n", encoding="utf-8")
    predictions = tmp_path / "predictions.csv"
    predictions.write_text(
        "word_id,wordform_id,gramset,rank,score\n", encoding="utf-8"
    )

    def stub_tables(lang, data_dir):
        return object()

    duplicate = Instance(
        language="krl",
        word_id=7,
        sentence_id=1,
        text_id=1,
        word="dup",
        word_number=1,
        sentence_xml="<s/>",
        candidates=(Candidate(1, "A"), Candidate(2, "B")),
        gold_analysis=Candidate(1, "A"),
    )

    class StubResult:
        def __init__(self, instances):
            self.instances = instances

    monkeypatch.setattr(cli_pkg, "read_corpus_tables", stub_tables)
    monkeypatch.setattr(
        cli_pkg, "build_language_instances", lambda lang, tables: StubResult((duplicate,))
    )
    status = cli_pkg._run_validate_predictions(
        argparse.Namespace(
            predictions=predictions,
            split="dev",
            data_dir=checkout,
            split_file=split_file,
        )
    )
    captured = capsys.readouterr()
    assert status == 1
    assert "strict benchmark" in captured.err
    assert "duplicate word_id" in captured.err
    assert "Predictions validation" not in captured.out
    assert "Traceback" not in captured.err


def _first_two(instances):
    ordered = sorted(instances, key=lambda inst: (inst.language, inst.word_id))
    return ordered[0], ordered[1]


def _multi_category_rows(instances):
    rows = _valid_prediction_rows(instances)
    first, second = _first_two(instances)
    rows.append([first.word_id, 9999999, "X+YZ", 3, "nan"])
    removed = next(
        row
        for row in rows
        if row[0] == second.word_id
        and row[1] == second.candidates[-1].wordform_id
        and row[2] == second.candidates[-1].gramset
    )
    rows.remove(removed)
    rows.append([999999, 55555, "N+PL+NOM", 1, 1.0])
    return rows


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_validate_predictions_failed_report_multi_category(tmp_path):
    result, instances = _run_validate_file(tmp_path, "dev", _multi_category_rows)
    assert result.returncode == 1
    assert "Predictions validation: FAILED" in result.stdout
    assert "Predictions validation: OK" not in result.stdout
    assert "Split: dev" in result.stdout
    word_count, candidate_count = _expected_counts(instances)
    assert (
        f"Expected: {word_count} word instances, {candidate_count} candidate rows"
        in result.stdout
    )
    assert "Found:" in result.stdout
    for heading in (
        "invalid score: 1 violation",
        "missing candidate: 1 violation",
        "rank set: 1 violation",
        "unknown word: 1 violation",
    ):
        assert heading in result.stdout
    first, second = _first_two(instances)
    assert f"word_id='{first.word_id}'; score='nan'" in result.stdout
    assert (
        f"word_id={second.word_id}; "
        f"candidate=({second.candidates[1].wordform_id}, "
        f"'{second.candidates[1].gramset}')"
    ) in result.stdout
    assert f"word_id={second.word_id}; ranks=[1]; expected [1, 2]" in result.stdout
    assert "word_id=999999" in result.stdout
    assert "Traceback" not in result.stderr


def _four_invalid_score_rows(instances):
    rows = _valid_prediction_rows(instances)
    first, _ = _first_two(instances)
    for _ in range(4):
        rows.append([first.word_id, 9999999, "X+YZ", 1, "nan"])
    return rows


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_validate_predictions_failed_report_full_count_not_truncated(tmp_path):
    result, _ = _run_validate_file(tmp_path, "dev", _four_invalid_score_rows)
    assert result.returncode == 1
    heading = "invalid score: 4 violations"
    assert heading in result.stdout
    lines = result.stdout.splitlines()
    start = lines.index(heading)
    examples = []
    for line in lines[start + 1 :]:
        if line.startswith("  "):
            examples.append(line)
        else:
            break
    assert len(examples) == 3
    assert len(examples) <= 3


def _single_invalid_score_rows(instances):
    rows = _valid_prediction_rows(instances)
    first, _ = _first_two(instances)
    rows.append([first.word_id, 9999999, "X+YZ", 1, "nan"])
    return rows


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_validate_predictions_failed_report_singular_grammar(tmp_path):
    result, _ = _run_validate_file(tmp_path, "dev", _single_invalid_score_rows)
    assert result.returncode == 1
    assert "Predictions validation: FAILED" in result.stdout
    assert "invalid score: 1 violation" in result.stdout
    assert "invalid score: 1 violations" not in result.stdout
    assert "missing word" not in result.stdout


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_validate_predictions_failed_report_category_order(tmp_path):
    def build_rows(instances):
        krl = [inst for inst in instances if inst.language == "krl"]
        by_text: dict[int, list] = {}
        for inst in krl:
            by_text.setdefault(int(inst.text_id), []).append(inst)
        moved_text = sorted(by_text)[2]
        moved = by_text[moved_text]
        keys = {(inst.language, int(inst.text_id)) for inst in instances}
        split_rows = [
            (
                lang,
                text_id,
                "test" if (lang == "krl" and text_id == moved_text) else "dev",
            )
            for lang, text_id in sorted(keys)
        ]
        dev_instances = [
            inst
            for inst in instances
            if not (inst.language == "krl" and int(inst.text_id) == moved_text)
        ]
        rows = _valid_prediction_rows(dev_instances)
        for inst in sorted(moved, key=lambda inst: inst.word_id):
            for index, candidate in enumerate(inst.candidates, start=1):
                rows.append(
                    [
                        inst.word_id,
                        candidate.wordform_id,
                        candidate.gramset,
                        index,
                        len(inst.candidates) - index + 1,
                    ]
                )
        rows.append([999999, 55555, "N+PL+NOM", 1, 1.0])
        return rows, split_rows

    result, _ = _run_validate_file(tmp_path, "dev", build_rows)
    assert result.returncode == 1
    assert "wrong split" in result.stdout
    assert "unknown word" in result.stdout
    assert result.stdout.index("wrong split") < result.stdout.index("unknown word")
    assert "Traceback" not in result.stderr


def _part_of(text_id: int) -> str:
    index = text_id % 100
    if index <= 3:
        return "train"
    if index <= 5:
        return "dev"
    return "test"


def _write_split_parts(path: Path, instances) -> None:
    keys = sorted({(inst.language, int(inst.text_id)) for inst in instances})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["language", "text_id", "split"])
        writer.writerows((lang, text_id, _part_of(text_id)) for lang, text_id in keys)


def _read_split_rows(path: Path) -> tuple[tuple[str, int, str], ...]:
    rows = []
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, strict=True)
        next(reader)
        for language, text_id, part in reader:
            rows.append((language, int(text_id), part))
    return tuple(rows)


def _split_instances(instances, split_rows: tuple[tuple[str, int, str], ...], part: str):
    split_by_text = {
        (language, text_id): split for language, text_id, split in split_rows
    }
    return [
        inst
        for inst in instances
        if split_by_text.get((inst.language, int(inst.text_id))) == part
    ]


def _run_frequency_baseline(tmp_path: Path, part: str):
    checkout = _tagged_checkout(tmp_path)
    instances = _strict_instances(checkout)
    split_file = tmp_path / f"splits_{part}.csv"
    _write_split_parts(split_file, instances)
    output = tmp_path / f"frequency-{part}.csv"
    result = run_cli(
        "frequency-baseline",
        "--split", part,
        "--output", str(output),
        "--data-dir", str(checkout),
        "--split-file", str(split_file),
    )
    return result, output, instances, split_file, checkout


def _read_generated(path: Path) -> tuple[list[str], list[list]]:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, strict=True)
        header = next(reader)
        rows = list(reader)
    return header, rows


def _repo_snapshot() -> set[tuple[str, str]]:
    files: set[tuple[str, str]] = set()
    for root in (ROOT / "results", ROOT / "data" / "derived"):
        if root.is_dir():
            for path in root.rglob("*"):
                if path.is_file():
                    files.add((str(path.relative_to(root)), path.stat().st_size))
    return files


def _checkout_snapshot(checkout: Path) -> set[tuple[str, str]]:
    return {
        (str(path.relative_to(checkout)), path.stat().st_size)
        for path in checkout.rglob("*")
        if path.is_file()
    }


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_frequency_baseline_dev_ok(tmp_path):
    result, output, instances, split_file, _ = _run_frequency_baseline(tmp_path, "dev")
    assert result.returncode == 0, result.stderr
    split_rows = _read_split_rows(split_file)
    expected = rank_by_train_frequency(instances, split_rows, "dev")
    target_instances = _split_instances(instances, split_rows, "dev")
    frequency = build_train_frequency(instances, split_rows)
    for line in (
        "Frequency baseline: OK",
        "Training gold occurrences:",
        "Distinct train candidate keys:",
        "Split: dev",
        "Generated:",
        "Validation: OK",
    ):
        assert line in result.stdout
    assert str(output) in result.stdout
    assert (
        f"Training gold occurrences: {sum(frequency.values())}" in result.stdout
    )
    assert f"Distinct train candidate keys: {len(frequency)}" in result.stdout
    assert (
        f"Generated: {len({inst.word_id for inst in target_instances})} "
        f"word instances, {len(expected)} candidate rows"
    ) in result.stdout
    assert "Traceback" not in result.stderr
    assert output.is_file()

    header, raw_rows = _read_generated(output)
    assert header == ["word_id", "wordform_id", "gramset", "rank", "score"]
    assert raw_rows
    assert all(len(row) == 5 for row in raw_rows)
    rows = [(int(r[0]), int(r[1]), r[2], int(r[3]), int(r[4])) for r in raw_rows]
    target_word_ids = {inst.word_id for inst in target_instances}
    assert {row[0] for row in rows} == target_word_ids
    assert rows == [
        (row.word_id, row.wordform_id, row.gramset, row.rank, row.score)
        for row in expected
    ]
    validation = validate_predictions(output, "dev", instances, split_rows)
    assert validation.is_valid


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_frequency_baseline_test_ok(tmp_path):
    result, output, instances, split_file, _ = _run_frequency_baseline(tmp_path, "test")
    assert result.returncode == 0, result.stderr
    split_rows = _read_split_rows(split_file)
    expected = rank_by_train_frequency(instances, split_rows, "test")
    target_instances = _split_instances(instances, split_rows, "test")
    assert "Split: test" in result.stdout
    header, raw_rows = _read_generated(output)
    assert header == ["word_id", "wordform_id", "gramset", "rank", "score"]
    rows = [(int(r[0]), int(r[1]), r[2], int(r[3]), int(r[4])) for r in raw_rows]
    assert {row[0] for row in rows} == {inst.word_id for inst in target_instances}
    assert rows == [
        (row.word_id, row.wordform_id, row.gramset, row.rank, row.score)
        for row in expected
    ]
    validation = validate_predictions(output, "test", instances, split_rows)
    assert validation.is_valid


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_frequency_baseline_creates_no_repo_outputs(tmp_path):
    checkout = _tagged_checkout(tmp_path)
    instances = _strict_instances(checkout)
    split_file = tmp_path / "splits_dev.csv"
    _write_split_parts(split_file, instances)
    output = tmp_path / "frequency-dev.csv"
    before_files = {path.name for path in tmp_path.iterdir()}
    repo_before = _repo_snapshot()
    checkout_before = _checkout_snapshot(checkout)
    result = run_cli(
        "frequency-baseline",
        "--split", "dev",
        "--output", str(output),
        "--data-dir", str(checkout),
        "--split-file", str(split_file),
    )
    assert result.returncode == 0, result.stderr
    after_files = {path.name for path in tmp_path.iterdir()}
    assert after_files == before_files | {output.name}
    assert _repo_snapshot() == repo_before
    assert _checkout_snapshot(checkout) == checkout_before


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_frequency_baseline_existing_output_protected(tmp_path):
    checkout = _tagged_checkout(tmp_path)
    instances = _strict_instances(checkout)
    split_file = tmp_path / "splits_dev.csv"
    _write_split_parts(split_file, instances)
    output = tmp_path / "frequency-dev.csv"
    original = "word_id,wordform_id,gramset,rank,score\n501,9001,SG+NOM,1,1\n"
    output.write_text(original, encoding="utf-8")
    repo_before = _repo_snapshot()
    result = run_cli(
        "frequency-baseline",
        "--split", "dev",
        "--output", str(output),
        "--data-dir", str(checkout),
        "--split-file", str(split_file),
    )
    assert result.returncode == 1
    assert str(output) in result.stderr
    assert "already exists" in result.stderr
    assert output.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob(".vepkar-frequency-*.tmp"))
    assert _repo_snapshot() == repo_before
    assert "Frequency baseline: OK" not in result.stdout
    assert "Traceback" not in result.stderr


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_frequency_baseline_missing_output_dir(tmp_path):
    checkout = _tagged_checkout(tmp_path)
    instances = _strict_instances(checkout)
    split_file = tmp_path / "splits_dev.csv"
    _write_split_parts(split_file, instances)
    missing_parent = tmp_path / "no-such-dir"
    output = missing_parent / "out.csv"
    result = run_cli(
        "frequency-baseline",
        "--split", "dev",
        "--output", str(output),
        "--data-dir", str(checkout),
        "--split-file", str(split_file),
    )
    assert result.returncode == 1
    assert "output directory does not exist" in result.stderr
    assert str(missing_parent) in result.stderr
    assert not missing_parent.exists()
    assert not list(tmp_path.glob(".vepkar-frequency-*.tmp"))
    assert "Traceback" not in result.stderr


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_frequency_baseline_output_is_directory(tmp_path):
    checkout = _tagged_checkout(tmp_path)
    instances = _strict_instances(checkout)
    split_file = tmp_path / "splits_dev.csv"
    _write_split_parts(split_file, instances)
    output = tmp_path / "outdir"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    result = run_cli(
        "frequency-baseline",
        "--split", "dev",
        "--output", str(output),
        "--data-dir", str(checkout),
        "--split-file", str(split_file),
    )
    assert result.returncode == 1
    assert "error: output path is a directory:" in result.stderr
    assert f"  {output}" in result.stderr
    assert output.is_dir()
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before
    assert not list(tmp_path.glob(".vepkar-frequency-*.tmp"))
    assert "Frequency baseline: OK" not in result.stdout
    assert "Traceback" not in result.stderr


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_frequency_baseline_broken_symlink_protected(tmp_path):
    checkout = _tagged_checkout(tmp_path)
    instances = _strict_instances(checkout)
    split_file = tmp_path / "splits_dev.csv"
    _write_split_parts(split_file, instances)
    output = tmp_path / "frequency.csv"
    missing_target = tmp_path / "missing-target.csv"

    try:
        output.symlink_to(missing_target)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is unavailable in this environment")

    result = run_cli(
        "frequency-baseline",
        "--split", "dev",
        "--output", str(output),
        "--data-dir", str(checkout),
        "--split-file", str(split_file),
    )
    assert result.returncode == 1
    assert str(output) in result.stderr
    assert "already exists" in result.stderr
    assert output.is_symlink()
    assert output.readlink() == missing_target
    assert not missing_target.exists()
    assert not list(tmp_path.glob(".vepkar-frequency-*.tmp"))
    assert "Frequency baseline: OK" not in result.stdout
    assert "Traceback" not in result.stderr


def test_frequency_baseline_missing_checkout_before_existing_output(tmp_path):
    missing = tmp_path / "missing-dictorpus-data"
    output = tmp_path / "out.csv"
    original = "word_id,wordform_id,gramset,rank,score\n501,9001,SG+NOM,1,1\n"
    output.write_text(original, encoding="utf-8")
    result = run_cli(
        "frequency-baseline",
        "--split", "dev",
        "--output", str(output),
        "--data-dir", str(missing),
    )
    assert result.returncode == 1
    assert "dictorpus-data" in result.stderr
    assert str(missing) in result.stderr
    assert "already exists" not in result.stderr
    assert output.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob(".vepkar-frequency-*.tmp"))
    assert "Traceback" not in result.stderr


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_frequency_baseline_existing_output_rejected_before_loading(
    tmp_path, monkeypatch, capsys
):
    import cli as cli_pkg

    checkout = _tagged_checkout(tmp_path)
    split_file = tmp_path / "splits_dev.csv"
    split_file.write_text("language,text_id,split\n", encoding="utf-8")
    output = tmp_path / "frequency-dev.csv"
    output.write_text("original", encoding="utf-8")

    def fail_tables(lang, data_dir):
        raise AssertionError("read_corpus_tables must not be called")

    resolved = []
    real_tag = cli_pkg.determine_data_tag
    monkeypatch.setattr(cli_pkg, "read_corpus_tables", fail_tables)
    monkeypatch.setattr(
        cli_pkg,
        "determine_data_tag",
        lambda data_dir: resolved.append(True) or real_tag(data_dir),
    )
    status = cli_pkg._run_frequency_baseline(
        argparse.Namespace(
            split="dev",
            output=output,
            data_dir=checkout,
            split_file=split_file,
        )
    )
    captured = capsys.readouterr()
    assert status == 1
    assert resolved == [True]
    assert "already exists" in captured.err
    assert str(output) in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_report_counts_use_thousands_separators(tmp_path, monkeypatch, capsys):
    import cli as cli_pkg

    checkout = _tagged_checkout(tmp_path)
    split_file = tmp_path / "splits_dev.csv"
    split_file.write_text("language,text_id,split\n", encoding="utf-8")

    class EmptyInstances:
        instances = ()

    monkeypatch.setattr(
        cli_pkg, "read_corpus_tables", lambda lang, data_dir: object()
    )
    monkeypatch.setattr(
        cli_pkg,
        "build_language_instances",
        lambda lang, tables: EmptyInstances(),
    )
    monkeypatch.setattr(cli_pkg, "_read_split_rows", lambda path: ())

    predictions = tmp_path / "predictions.csv"
    predictions.write_text(
        "word_id,wordform_id,gramset,rank,score\n", encoding="utf-8"
    )

    valid = SimpleNamespace(
        is_valid=True,
        split="dev",
        expected_word_count=2801,
        expected_candidate_count=6950,
        predicted_word_count=2800,
        predicted_candidate_count=6949,
        error_counts={},
        errors={},
    )
    monkeypatch.setattr(cli_pkg, "validate_predictions", lambda **kwargs: valid)

    status = cli_pkg._run_validate_predictions(
        argparse.Namespace(
            predictions=predictions,
            split="dev",
            data_dir=checkout,
            split_file=split_file,
        )
    )
    captured = capsys.readouterr()
    assert status == 0, captured.err
    assert "Validated: 2,801 word instances, 6,950 candidate rows" in captured.out

    failed = SimpleNamespace(
        split="dev",
        expected_word_count=2801,
        expected_candidate_count=6950,
        predicted_word_count=2800,
        predicted_candidate_count=6949,
        error_counts={"missing candidate": 6950, "invalid score": 1},
        errors={
            "missing candidate": ("word_id=1; candidate=(2, 'X')",),
            "invalid score": ("word_id=1; score='nan'",),
        },
    )
    cli_pkg._print_validation_failure(failed)
    captured = capsys.readouterr()
    assert "Expected: 2,801 word instances, 6,950 candidate rows" in captured.out
    assert "Found: 2,800 word instances, 6,949 prediction rows" in captured.out
    assert "missing candidate: 6,950 violations" in captured.out
    assert "invalid score: 1 violation" in captured.out
    assert "invalid score: 1 violations" not in captured.out

    rows = tuple(
        FrequencyPrediction(
            word_id=(index % 2826) + 1,
            wordform_id=index + 1,
            gramset="G",
            rank=1,
            score=0,
        )
        for index in range(6961)
    )
    frequency = Counter({(index, "G"): 1 for index in range(7313)})
    frequency[(99999, "G")] += 22390 - 7313
    monkeypatch.setattr(
        cli_pkg,
        "rank_by_train_frequency",
        lambda instances, split_rows, split: rows,
    )
    monkeypatch.setattr(
        cli_pkg,
        "build_train_frequency",
        lambda instances, split_rows: frequency,
    )

    output = tmp_path / "frequency-out.csv"
    status = cli_pkg._run_frequency_baseline(
        argparse.Namespace(
            split="dev",
            output=output,
            data_dir=checkout,
            split_file=split_file,
        )
    )
    captured = capsys.readouterr()
    assert status == 0, captured.err
    assert "Training gold occurrences: 22,390" in captured.out
    assert "Distinct train candidate keys: 7,314" in captured.out
    assert "Generated: 2,826 word instances, 6,961 candidate rows" in captured.out
    assert "Traceback" not in captured.err


def test_frequency_baseline_rejects_train(tmp_path):
    result = run_cli(
        "frequency-baseline",
        "--split", "train",
        "--output", str(tmp_path / "out.csv"),
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr
    assert "train" in result.stderr


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_frequency_baseline_missing_checkout_checked_first(tmp_path):
    missing = tmp_path / "missing-dictorpus-data"
    output = tmp_path / "out.csv"
    result = run_cli(
        "frequency-baseline",
        "--split", "dev",
        "--output", str(output),
        "--data-dir", str(missing),
    )
    assert result.returncode == 1
    assert "dictorpus-data" in result.stderr
    assert str(missing) in result.stderr
    assert "output file already exists" not in result.stderr
    assert not output.exists()
    assert "Traceback" not in result.stderr


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_frequency_baseline_missing_split_file(tmp_path):
    checkout = _tagged_checkout(tmp_path)
    missing_split = tmp_path / "missing-splits.csv"
    output = tmp_path / "out.csv"
    result = run_cli(
        "frequency-baseline",
        "--split", "dev",
        "--output", str(output),
        "--data-dir", str(checkout),
        "--split-file", str(missing_split),
    )
    assert result.returncode == 1
    assert "cannot read split file" in result.stderr
    assert str(missing_split) in result.stderr
    assert not output.exists()
    assert "Traceback" not in result.stderr


def _run_evaluate(tmp_path: Path, part: str, rows=None) -> subprocess.CompletedProcess:
    checkout = _tagged_checkout(tmp_path)
    instances = _strict_instances(checkout)
    split_file = tmp_path / f"splits_{part}.csv"
    _write_split_fixture(split_file, instances, part)
    predictions = tmp_path / f"predictions_{part}.csv"
    if rows is None:
        _write_valid_predictions(predictions, instances)
    else:
        _write_predictions(predictions, rows(instances))
    result = run_cli(
        "evaluate-predictions",
        "--predictions", str(predictions),
        "--split", part,
        "--data-dir", str(checkout),
        "--split-file", str(split_file),
    )
    return result, instances, predictions


def _assert_evaluation_report(result, instances, part, predictions) -> None:
    word_count, candidate_count = _expected_counts(instances)
    assert result.returncode == 0, result.stderr
    assert "Prediction evaluation: OK" in result.stdout
    assert f"Predictions: {predictions}" in result.stdout
    assert f"Split: {part}" in result.stdout
    assert (
        f"Evaluated: {word_count} word instances, "
        f"{candidate_count} candidate rows"
    ) in result.stdout
    assert "Top-1 accuracy" in result.stdout
    assert "MRR" in result.stdout
    assert "Top-3 accuracy" in result.stdout
    assert "Traceback" not in result.stderr


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_evaluate_predictions_dev_ok(tmp_path):
    result, instances, predictions = _run_evaluate(tmp_path, "dev")
    _assert_evaluation_report(result, instances, "dev", predictions)


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_evaluate_predictions_test_ok(tmp_path):
    result, instances, predictions = _run_evaluate(tmp_path, "test")
    _assert_evaluation_report(result, instances, "test", predictions)


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_evaluate_predictions_invalid_file_fails(tmp_path):
    result, instances, predictions = _run_evaluate(tmp_path, "dev")
    lines = predictions.read_text(encoding="utf-8").splitlines()
    predictions.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    result = run_cli(
        "evaluate-predictions",
        "--predictions", str(predictions),
        "--split", "dev",
        "--data-dir", str(tmp_path / "checkout"),
        "--split-file", str(tmp_path / "splits_dev.csv"),
    )
    assert result.returncode == 1
    assert "Predictions validation: FAILED" in result.stdout
    assert "Prediction evaluation: OK" not in result.stdout
    assert "Top-1 accuracy" not in result.stdout
    assert "Traceback" not in result.stderr


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_evaluate_predictions_missing_predictions_file(tmp_path):
    checkout = _tagged_checkout(tmp_path)
    instances = _strict_instances(checkout)
    split_file = tmp_path / "splits_dev.csv"
    _write_split_fixture(split_file, instances, "dev")
    missing = tmp_path / "missing.csv"
    result = run_cli(
        "evaluate-predictions",
        "--predictions", str(missing),
        "--split", "dev",
        "--data-dir", str(checkout),
        "--split-file", str(split_file),
    )
    assert result.returncode == 1
    assert "cannot read predictions file" in result.stderr
    assert str(missing) in result.stderr
    assert "Prediction evaluation: OK" not in result.stdout
    assert "Traceback" not in result.stderr


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_evaluate_predictions_malformed_csv(tmp_path):
    checkout = _tagged_checkout(tmp_path)
    instances = _strict_instances(checkout)
    split_file = tmp_path / "splits_dev.csv"
    _write_split_fixture(split_file, instances, "dev")
    predictions = tmp_path / "malformed.csv"
    predictions.write_text(
        "word_id,wordform_id,gramset,rank,score\n"
        '501,9001,"SG+NOM,1,125\n',
        encoding="utf-8",
    )
    result = run_cli(
        "evaluate-predictions",
        "--predictions", str(predictions),
        "--split", "dev",
        "--data-dir", str(checkout),
        "--split-file", str(split_file),
    )
    assert result.returncode == 1
    assert "cannot parse predictions CSV" in result.stderr
    assert str(predictions) in result.stderr
    assert "Prediction evaluation: OK" not in result.stdout
    assert "Top-1 accuracy" not in result.stdout
    assert "Traceback" not in result.stderr


def test_evaluate_predictions_missing_checkout_checked_first(tmp_path):
    missing = tmp_path / "missing-dictorpus-data"
    result = run_cli(
        "evaluate-predictions",
        "--predictions", str(tmp_path / "missing-predictions.csv"),
        "--split", "dev",
        "--data-dir", str(missing),
        "--split-file", str(tmp_path / "missing-splits.csv"),
    )
    assert result.returncode == 1
    assert "dictorpus-data" in result.stderr
    assert str(missing) in result.stderr
    assert "cannot read predictions file" not in result.stderr
    assert "cannot read split file" not in result.stderr
    assert "Prediction evaluation" not in result.stdout
    assert "Traceback" not in result.stderr


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_evaluate_predictions_missing_split_file(tmp_path):
    checkout = _tagged_checkout(tmp_path)
    missing_split = tmp_path / "missing-splits.csv"
    predictions = tmp_path / "predictions.csv"
    predictions.write_text(
        "word_id,wordform_id,gramset,rank,score\n", encoding="utf-8"
    )
    result = run_cli(
        "evaluate-predictions",
        "--predictions", str(predictions),
        "--split", "dev",
        "--data-dir", str(checkout),
        "--split-file", str(missing_split),
    )
    assert result.returncode == 1
    assert "cannot read split file" in result.stderr
    assert str(missing_split) in result.stderr
    assert "Prediction evaluation: OK" not in result.stdout
    assert "Traceback" not in result.stderr


def test_evaluate_predictions_rejects_train(tmp_path):
    result = run_cli(
        "evaluate-predictions",
        "--predictions", str(tmp_path / "predictions.csv"),
        "--split", "train",
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr
    assert "train" in result.stderr


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_evaluate_predictions_writes_no_files(tmp_path):
    checkout = _tagged_checkout(tmp_path)
    instances = _strict_instances(checkout)
    split_file = tmp_path / "splits_dev.csv"
    _write_split_fixture(split_file, instances, "dev")
    predictions = tmp_path / "predictions_dev.csv"
    _write_valid_predictions(predictions, instances)
    repo_before = _repo_snapshot()
    checkout_before = _checkout_snapshot(checkout)
    result = run_cli(
        "evaluate-predictions",
        "--predictions", str(predictions),
        "--split", "dev",
        "--data-dir", str(checkout),
        "--split-file", str(split_file),
    )
    assert result.returncode == 0, result.stderr
    assert _repo_snapshot() == repo_before
    assert _checkout_snapshot(checkout) == checkout_before


def _reversed_prediction_rows(instances) -> list[list]:
    rows = []
    for instance in sorted(
        instances, key=lambda inst: (inst.language, inst.word_id)
    ):
        candidates = list(instance.candidates)
        for rank, candidate in enumerate(reversed(candidates), start=1):
            rows.append(
                [
                    instance.word_id,
                    candidate.wordform_id,
                    candidate.gramset,
                    rank,
                    len(candidates) - rank + 1,
                ]
            )
    return rows


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_evaluate_predictions_metrics_match_hand_calculated(tmp_path):
    result, instances, _ = _run_evaluate(
        tmp_path, "dev", _reversed_prediction_rows
    )
    word_count, candidate_count = _expected_counts(instances)
    assert result.returncode == 0, result.stderr
    assert (
        f"Evaluated: {word_count} word instances, "
        f"{candidate_count} candidate rows"
    ) in result.stdout
    assert "Top-1 accuracy     0.0000" in result.stdout
    assert "MRR                0.5000" in result.stdout
    assert "Top-3 accuracy     1.0000" in result.stdout
    assert "Traceback" not in result.stderr


def _text_keys(instances) -> list[tuple[str, int]]:
    return sorted({(inst.language, int(inst.text_id)) for inst in instances})


def _write_split_parts_fixture(path: Path, instances, part_by_text) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["language", "text_id", "split"])
        for language, text_id in _text_keys(instances):
            writer.writerow((language, text_id, part_by_text[(language, text_id)]))


def _instances_in_part(instances, part_by_text, part) -> list:
    return [
        inst
        for inst in instances
        if part_by_text[(inst.language, int(inst.text_id))] == part
    ]


def _evaluate_partial_split(tmp_path, part: str):
    checkout = _tagged_checkout(tmp_path)
    instances = _strict_instances(checkout)
    keys = _text_keys(instances)
    split_key = keys[0]
    part_by_text = {
        key: (part if key == split_key else "train") for key in keys
    }
    split_file = tmp_path / f"splits_{part}_partial.csv"
    _write_split_parts_fixture(split_file, instances, part_by_text)
    target = _instances_in_part(instances, part_by_text, part)
    predictions = tmp_path / f"predictions_{part}_partial.csv"
    _write_valid_predictions(predictions, target)
    result = run_cli(
        "evaluate-predictions",
        "--predictions", str(predictions),
        "--split", part,
        "--data-dir", str(checkout),
        "--split-file", str(split_file),
    )
    return result, target


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_evaluate_predictions_uses_only_selected_split_words(tmp_path):
    result, target = _evaluate_partial_split(tmp_path, "dev")
    word_count = len(target)
    candidate_count = sum(len(inst.candidates) for inst in target)
    assert result.returncode == 0, result.stderr
    assert "Prediction evaluation: OK" in result.stdout
    assert "Split: dev" in result.stdout
    assert (
        f"Evaluated: {word_count} word instances, "
        f"{candidate_count} candidate rows"
    ) in result.stdout
    assert "Top-1 accuracy" in result.stdout
    assert "MRR" in result.stdout
    assert "Top-3 accuracy" in result.stdout
    assert "Traceback" not in result.stderr


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_evaluate_predictions_dev_test_isolated(tmp_path):
    checkout = _tagged_checkout(tmp_path)
    instances = _strict_instances(checkout)
    part_by_text = {
        key: ("dev" if index % 2 == 0 else "test")
        for index, key in enumerate(_text_keys(instances))
    }
    split_file = tmp_path / "splits_both.csv"
    _write_split_parts_fixture(split_file, instances, part_by_text)
    dev_instances = _instances_in_part(instances, part_by_text, "dev")
    test_instances = _instances_in_part(instances, part_by_text, "test")
    assert dev_instances
    assert test_instances
    dev_predictions = tmp_path / "predictions_dev.csv"
    _write_valid_predictions(dev_predictions, dev_instances)
    test_predictions = tmp_path / "predictions_test.csv"
    _write_valid_predictions(test_predictions, test_instances)

    dev_result = run_cli(
        "evaluate-predictions",
        "--predictions", str(dev_predictions),
        "--split", "dev",
        "--data-dir", str(checkout),
        "--split-file", str(split_file),
    )
    assert dev_result.returncode == 0, dev_result.stderr
    assert "Split: dev" in dev_result.stdout
    assert (
        f"Evaluated: {len(dev_instances)} word instances, "
        f"{sum(len(inst.candidates) for inst in dev_instances)} candidate rows"
    ) in dev_result.stdout
    assert "Traceback" not in dev_result.stderr

    test_result = run_cli(
        "evaluate-predictions",
        "--predictions", str(test_predictions),
        "--split", "test",
        "--data-dir", str(checkout),
        "--split-file", str(split_file),
    )
    assert test_result.returncode == 0, test_result.stderr
    assert "Split: test" in test_result.stdout
    assert (
        f"Evaluated: {len(test_instances)} word instances, "
        f"{sum(len(inst.candidates) for inst in test_instances)} candidate rows"
    ) in test_result.stdout
    assert "Traceback" not in test_result.stderr


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_evaluate_predictions_missing_candidate_still_fails_validation(tmp_path):
    checkout = _tagged_checkout(tmp_path)
    instances = _strict_instances(checkout)
    keys = _text_keys(instances)
    part_by_text = {key: ("test" if key == keys[0] else "train") for key in keys}
    split_file = tmp_path / "splits_strict.csv"
    _write_split_parts_fixture(split_file, instances, part_by_text)
    target = _instances_in_part(instances, part_by_text, "test")
    partial = [
        [inst.word_id, inst.candidates[0].wordform_id, inst.candidates[0].gramset, 1, 1]
        for inst in sorted(target, key=lambda inst: inst.word_id)
    ]
    predictions = tmp_path / "predictions_partial.csv"
    _write_predictions(predictions, partial)
    result = run_cli(
        "evaluate-predictions",
        "--predictions", str(predictions),
        "--split", "test",
        "--data-dir", str(checkout),
        "--split-file", str(split_file),
    )
    assert result.returncode == 1
    assert "Predictions validation: FAILED" in result.stdout
    assert "missing candidate" in result.stdout
    assert "Prediction evaluation: OK" not in result.stdout
    assert "Traceback" not in result.stderr


DIAGNOSTICS_HEADER = [
    "language",
    "word_id",
    "word",
    "candidate_count",
    "gold_wordform_id",
    "gold_gramset",
    "top1_wordform_id",
    "top1_gramset",
    "gold_rank",
    "gold_train_frequency",
    "top1_train_frequency",
]


def _write_frequency_baseline_predictions(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["word_id", "wordform_id", "gramset", "rank", "score"])
        writer.writerows(rows)


def _baseline_csv_rows(instances, split_rows, split="dev") -> list[list]:
    expected = rank_by_train_frequency(instances, split_rows, split)
    return [
        [row.word_id, row.wordform_id, row.gramset, row.rank, row.score]
        for row in expected
    ]


def _run_diagnose(tmp_path: Path, build_rows=None):
    checkout = _tagged_checkout(tmp_path)
    instances = _strict_instances(checkout)
    split_file = tmp_path / "splits_dev.csv"
    _write_split_parts(split_file, instances)
    split_rows = _read_split_rows(split_file)
    dev_instances = _split_instances(instances, split_rows, "dev")
    baseline_rows = _baseline_csv_rows(instances, split_rows, "dev")
    rows = (
        baseline_rows
        if build_rows is None
        else build_rows(baseline_rows, dev_instances)
    )
    predictions = tmp_path / "predictions-dev.csv"
    _write_frequency_baseline_predictions(predictions, rows)
    output = tmp_path / "diagnostics-dev.csv"
    result = run_cli(
        "diagnose-frequency-baseline",
        "--predictions", str(predictions),
        "--output", str(output),
        "--data-dir", str(checkout),
        "--split-file", str(split_file),
    )
    return result, output, instances, split_rows, dev_instances, baseline_rows


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_diagnose_frequency_baseline_dev_ok(tmp_path):
    result, output, instances, split_rows, dev_instances, baseline_rows = (
        _run_diagnose(tmp_path)
    )
    assert result.returncode == 0, result.stderr
    assert "Frequency diagnostics: OK" in result.stdout
    assert "Split: dev" in result.stdout
    assert f"Diagnostic rows: {len(dev_instances)}" in result.stdout
    assert f"Output: {output}" in result.stdout
    assert "Traceback" not in result.stderr

    header, raw_rows = _read_generated(output)
    assert header == DIAGNOSTICS_HEADER
    assert len(raw_rows) == len(dev_instances)
    keys = [(row[0], int(row[1])) for row in raw_rows]
    assert keys == sorted(keys)
    assert {int(row[1]) for row in raw_rows} == {
        inst.word_id for inst in dev_instances
    }

    expected = rank_by_train_frequency(instances, split_rows, "dev")
    verified = load_verified_frequency_predictions(output.parent / "predictions-dev.csv", expected)
    expected_rows = build_frequency_diagnostics(dev_instances, verified)
    assert raw_rows == [list(map(str, astuple(row))) for row in expected_rows]


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_diagnose_reordered_predictions_accepted(tmp_path):
    result, output, _, _, dev_instances, _ = _run_diagnose(
        tmp_path, lambda baseline, dev: list(reversed(baseline))
    )
    assert result.returncode == 0, result.stderr
    assert f"Diagnostic rows: {len(dev_instances)}" in result.stdout
    assert output.is_file()
    assert "Traceback" not in result.stderr


def _non_baseline_valid_rows(dev_instances):
    rows = _valid_prediction_rows(dev_instances)
    first = sorted(
        dev_instances, key=lambda inst: (inst.language, inst.word_id)
    )[0]
    first_rows = [row for row in rows if row[0] == first.word_id]
    first_rows[0], first_rows[1] = first_rows[1], first_rows[0]
    return rows


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_diagnose_structurally_valid_non_baseline_rejected(tmp_path):
    result, output, _, _, dev_instances, baseline_rows = _run_diagnose(
        tmp_path, lambda baseline, dev: _non_baseline_valid_rows(dev)
    )
    assert [tuple(row) for row in _non_baseline_valid_rows(dev_instances)] != [
        tuple(row) for row in baseline_rows
    ]
    assert result.returncode == 1
    assert "do not match the expected frequency baseline" in result.stderr
    assert not output.exists()
    assert "Frequency diagnostics: OK" not in result.stdout
    assert not list(tmp_path.glob(".vepkar-diagnostics-*.tmp"))
    assert "Traceback" not in result.stderr


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_diagnose_invalid_predictions_validation_failure(tmp_path):
    result, output, _, _, _, _ = _run_diagnose(
        tmp_path, lambda baseline, dev: baseline[:-1]
    )
    assert result.returncode == 1
    assert "Predictions validation: FAILED" in result.stdout
    assert "Split: dev" in result.stdout
    assert not output.exists()
    assert "Frequency diagnostics: OK" not in result.stdout
    assert not list(tmp_path.glob(".vepkar-diagnostics-*.tmp"))
    assert "Traceback" not in result.stderr


def test_diagnose_help_states_uncompressed_and_dev_only(tmp_path):
    result = run_cli("diagnose-frequency-baseline", "--help")
    assert result.returncode == 0
    assert "uncompressed" in result.stdout
    assert "--predictions" in result.stdout
    assert "--output" in result.stdout
    assert "--data-dir" in result.stdout
    assert "--split-file" in result.stdout
    assert "--split {dev,test}" not in result.stdout
    assert "diagnose-frequency-baseline" in result.stdout


def test_diagnose_missing_checkout_checked_first(tmp_path):
    missing = tmp_path / "missing-dictorpus-data"
    output = tmp_path / "out.csv"
    result = run_cli(
        "diagnose-frequency-baseline",
        "--predictions", str(tmp_path / "missing-predictions.csv"),
        "--output", str(output),
        "--data-dir", str(missing),
    )
    assert result.returncode == 1
    assert "dictorpus-data" in result.stderr
    assert str(missing) in result.stderr
    assert "cannot read predictions file" not in result.stderr
    assert not output.exists()
    assert "Frequency diagnostics" not in result.stdout
    assert "Traceback" not in result.stderr


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_diagnose_rejects_zst_input(tmp_path):
    checkout = _tagged_checkout(tmp_path)
    zst = tmp_path / "predictions.csv.zst"
    zst.write_text("word_id,wordform_id,gramset,rank,score\n", encoding="utf-8")
    output = tmp_path / "diagnostics.csv"
    result = run_cli(
        "diagnose-frequency-baseline",
        "--predictions", str(zst),
        "--output", str(output),
        "--data-dir", str(checkout),
    )
    assert result.returncode == 1
    assert "uncompressed" in result.stderr
    assert str(zst) in result.stderr
    assert not output.exists()
    assert "Frequency diagnostics: OK" not in result.stdout
    assert "Traceback" not in result.stderr


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_diagnose_rejects_zst_output(tmp_path):
    checkout = _tagged_checkout(tmp_path)
    predictions = tmp_path / "predictions-dev.csv"
    predictions.write_text(
        "word_id,wordform_id,gramset,rank,score\n", encoding="utf-8"
    )
    output = tmp_path / "diagnostics.csv.zst"
    result = run_cli(
        "diagnose-frequency-baseline",
        "--predictions", str(predictions),
        "--output", str(output),
        "--data-dir", str(checkout),
    )
    assert result.returncode == 1
    assert "uncompressed" in result.stderr
    assert str(output) in result.stderr
    assert not output.exists()
    assert "Frequency diagnostics: OK" not in result.stdout
    assert "Traceback" not in result.stderr


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_diagnose_existing_output_protected(tmp_path):
    checkout = _tagged_checkout(tmp_path)
    instances = _strict_instances(checkout)
    split_file = tmp_path / "splits_dev.csv"
    _write_split_parts(split_file, instances)
    output = tmp_path / "diagnostics-dev.csv"
    original = "language,word_id,word\n"
    output.write_text(original, encoding="utf-8")
    predictions = tmp_path / "predictions-dev.csv"
    predictions.write_text(
        "word_id,wordform_id,gramset,rank,score\n", encoding="utf-8"
    )
    result = run_cli(
        "diagnose-frequency-baseline",
        "--predictions", str(predictions),
        "--output", str(output),
        "--data-dir", str(checkout),
        "--split-file", str(split_file),
    )
    assert result.returncode == 1
    assert "already exists" in result.stderr
    assert str(output) in result.stderr
    assert output.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob(".vepkar-diagnostics-*.tmp"))
    assert "Frequency diagnostics: OK" not in result.stdout
    assert "Traceback" not in result.stderr


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_diagnose_output_directory_rejected(tmp_path):
    checkout = _tagged_checkout(tmp_path)
    instances = _strict_instances(checkout)
    split_file = tmp_path / "splits_dev.csv"
    _write_split_parts(split_file, instances)
    output = tmp_path / "outdir"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")
    predictions = tmp_path / "predictions-dev.csv"
    predictions.write_text(
        "word_id,wordform_id,gramset,rank,score\n", encoding="utf-8"
    )
    result = run_cli(
        "diagnose-frequency-baseline",
        "--predictions", str(predictions),
        "--output", str(output),
        "--data-dir", str(checkout),
        "--split-file", str(split_file),
    )
    assert result.returncode == 1
    assert "output path is a directory" in result.stderr
    assert f"  {output}" in result.stderr
    assert output.is_dir()
    assert (output / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert not list(tmp_path.glob(".vepkar-diagnostics-*.tmp"))
    assert "Frequency diagnostics: OK" not in result.stdout
    assert "Traceback" not in result.stderr


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_diagnose_broken_symlink_output_protected(tmp_path):
    checkout = _tagged_checkout(tmp_path)
    instances = _strict_instances(checkout)
    split_file = tmp_path / "splits_dev.csv"
    _write_split_parts(split_file, instances)
    output = tmp_path / "diagnostics.csv"
    missing_target = tmp_path / "missing-target.csv"
    try:
        output.symlink_to(missing_target)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is unavailable in this environment")
    predictions = tmp_path / "predictions-dev.csv"
    predictions.write_text(
        "word_id,wordform_id,gramset,rank,score\n", encoding="utf-8"
    )
    result = run_cli(
        "diagnose-frequency-baseline",
        "--predictions", str(predictions),
        "--output", str(output),
        "--data-dir", str(checkout),
        "--split-file", str(split_file),
    )
    assert result.returncode == 1
    assert "already exists" in result.stderr
    assert output.is_symlink()
    assert output.readlink() == missing_target
    assert not missing_target.exists()
    assert not list(tmp_path.glob(".vepkar-diagnostics-*.tmp"))
    assert "Frequency diagnostics: OK" not in result.stdout
    assert "Traceback" not in result.stderr


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_diagnose_missing_parent_rejected_without_creation(tmp_path):
    checkout = _tagged_checkout(tmp_path)
    instances = _strict_instances(checkout)
    split_file = tmp_path / "splits_dev.csv"
    _write_split_parts(split_file, instances)
    missing_parent = tmp_path / "no-such-dir"
    output = missing_parent / "out.csv"
    predictions = tmp_path / "predictions-dev.csv"
    predictions.write_text(
        "word_id,wordform_id,gramset,rank,score\n", encoding="utf-8"
    )
    result = run_cli(
        "diagnose-frequency-baseline",
        "--predictions", str(predictions),
        "--output", str(output),
        "--data-dir", str(checkout),
        "--split-file", str(split_file),
    )
    assert result.returncode == 1
    assert "output directory does not exist" in result.stderr
    assert str(missing_parent) in result.stderr
    assert not missing_parent.exists()
    assert not list(tmp_path.glob(".vepkar-diagnostics-*.tmp"))
    assert "Frequency diagnostics: OK" not in result.stdout
    assert "Traceback" not in result.stderr


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_diagnose_preserves_unicode_commas_quotes(tmp_path, monkeypatch, capsys):
    import cli as cli_pkg

    checkout = _tagged_checkout(tmp_path)
    instances = (
        Instance(
            language="krl",
            word_id=1,
            sentence_id=1,
            text_id=1,
            word='Šuuruš’a, "päiv"',
            word_number=1,
            sentence_xml="<s/>",
            candidates=(Candidate(10, "N+SG,NOM"), Candidate(11, "N+SG+GEN")),
            gold_analysis=Candidate(10, "N+SG,NOM"),
        ),
        Instance(
            language="vep",
            word_id=2,
            sentence_id=1,
            text_id=2,
            word="kala, 'kal'",
            word_number=1,
            sentence_xml="<s/>",
            candidates=(Candidate(20, "A"), Candidate(21, "B")),
            gold_analysis=Candidate(20, "A"),
        ),
    )
    split_rows = (("krl", 1, "dev"), ("vep", 2, "dev"))
    baseline = rank_by_train_frequency(instances, split_rows, "dev")
    predictions = tmp_path / "predictions-uni.csv"
    _write_frequency_baseline_predictions(
        predictions,
        [
            [row.word_id, row.wordform_id, row.gramset, row.rank, row.score]
            for row in baseline
        ],
    )
    output = tmp_path / "diagnostics-uni.csv"

    def stub_context(data_dir, tag, split_file=None):
        return cli_pkg.BenchmarkContext(
            tag=tag,
            split_path=split_file or predictions,
            instances=instances,
            split_rows=split_rows,
        )

    monkeypatch.setattr(cli_pkg, "load_benchmark_context", stub_context)
    status = cli_pkg._run_diagnose_frequency_baseline(
        argparse.Namespace(
            predictions=predictions,
            output=output,
            data_dir=checkout,
            split_file=tmp_path / "splits_dev.csv",
        )
    )
    captured = capsys.readouterr()
    assert status == 0, captured.err
    assert "Frequency diagnostics: OK" in captured.out
    assert "Diagnostic rows: 2" in captured.out
    header, raw = _read_generated(output)
    assert header == DIAGNOSTICS_HEADER
    assert len(raw) == 2
    by_word = {int(row[1]): row for row in raw}
    assert by_word[1][2] == 'Šuuruš’a, "päiv"'
    assert by_word[1][5] == "N+SG,NOM"
    assert by_word[1][7] == "N+SG,NOM"
    assert by_word[2][2] == "kala, 'kal'"
    assert "Traceback" not in captured.err