"""Offline tests for the predict-t1 CLI command."""

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import joblib

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import benchmark_context  # noqa: E402
import cli  # noqa: E402
import commands.predict_t1 as predict_pkg  # noqa: E402
from core.data import SUPPORTED_LANGUAGES, CorpusTables  # noqa: E402
from core.predictions import (  # noqa: E402
    PREDICTION_HEADER,
    PredictionValidation,
    validate_predictions,
)
from t1_features.features import CandidateExample, build_candidate_examples  # noqa: E402
from t1_features.model import score_candidate_examples, train_candidate_model  # noqa: E402
from t1_features.model_io import load_model_bundle, save_model_bundle  # noqa: E402
from t1_features.prediction_io import T1PredictionWriteError  # noqa: E402


def _make_texts(text_ids):
    return pd.DataFrame(
        {
            "text_id": pd.array(text_ids, dtype="Int64"),
            "corpus_ids": [tuple() for _ in text_ids],
            "dialect_code": pd.array([None] * len(text_ids), dtype="string"),
            "genre_ids": [tuple() for _ in text_ids],
            "year_recorded": pd.array([pd.NA] * len(text_ids), dtype="Int64"),
        }
    )


def _make_sentences(rows):
    return pd.DataFrame(
        {
            "sentence_id": pd.array([r[0] for r in rows], dtype="Int64"),
            "text_id": pd.array([r[1] for r in rows], dtype="Int64"),
            "sentence_xml": pd.array([r[2] for r in rows], dtype="string"),
            "sentence_ru": pd.array([r[3] for r in rows], dtype="string"),
        }
    )


def _make_words(rows):
    return pd.DataFrame(
        {
            "word_id": pd.array([r[0] for r in rows], dtype="Int64"),
            "sentence_id": pd.array([r[1] for r in rows], dtype="Int64"),
            "word_number": pd.array([r[2] for r in rows], dtype="Int64"),
            "word": pd.array([r[3] for r in rows], dtype="string"),
        }
    )


def _make_candidates(rows):
    return pd.DataFrame(
        {
            "word_id": pd.array([r[0] for r in rows], dtype="Int64"),
            "wordform_id": pd.array([r[1] for r in rows], dtype="Int64"),
            "gramset": pd.array([r[2] for r in rows], dtype="string"),
            "relevance": pd.array([r[3] for r in rows], dtype="Int64"),
        }
    )


def _empty_corpus():
    return CorpusTables(
        texts=_make_texts([]),
        sentences=_make_sentences([]),
        words=_make_words([]),
        candidates=_make_candidates([]),
    )


def _krl_corpus():
    return CorpusTables(
        texts=_make_texts([1, 2, 3]),
        sentences=_make_sentences(
            [
                (10, 1, '<s id="10"/>', ""),
                (20, 2, '<s id="20"/>', ""),
                (30, 3, '<s id="30"/>', ""),
            ]
        ),
        words=_make_words(
            [
                (101, 10, 1, "left"),
                (102, 10, 2, "trainkala"),
                (103, 10, 3, "right"),
                (201, 20, 1, "left"),
                (205, 20, 2, "devkala"),
                (206, 20, 3, "right"),
                (301, 30, 1, "left"),
                (308, 30, 2, "testkala"),
                (309, 30, 3, "right"),
            ]
        ),
        candidates=_make_candidates(
            [
                (102, 1020, "SG", 2),
                (102, 1021, "PL", 1),
                (205, 2050, "DEV+ONLY", 2),
                (205, 2051, "DEV+ALT", 1),
                (308, 3080, "TEST+ONLY", 2),
                (308, 3081, "TEST+ALT", 1),
            ]
        ),
    )


def _install_tables(monkeypatch):
    tables = {lang: _empty_corpus() for lang in SUPPORTED_LANGUAGES}
    tables["krl"] = _krl_corpus()
    monkeypatch.setattr(
        benchmark_context,
        "read_corpus_tables",
        lambda lang, data_dir: tables[lang],
    )
    return tables


def _stub_preflight(monkeypatch, tag="fixture-tag"):
    monkeypatch.setattr(
        predict_pkg,
        "_preflight_benchmark",
        lambda data_dir: (data_dir, tag),
    )


def _forbid_preflight(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("must be rejected before corpus access")

    monkeypatch.setattr(predict_pkg, "_preflight_benchmark", forbidden)


def _forbid_model_load(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("model bundle must not be loaded")

    monkeypatch.setattr(predict_pkg, "load_model_bundle", forbidden)


def _write_split_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["language", "text_id", "split"])
        writer.writerows(rows)


def _split_rows():
    return [("krl", 1, "train"), ("krl", 2, "dev"), ("krl", 3, "test")]


def _train_examples():
    return (
        CandidateExample(
            language="krl",
            word_id=1,
            wordform_id=10,
            gramset="SG",
            word="a",
            word_length=1,
            left_word="<BOS>",
            right_word="b",
            features={"gramset:SG": 1.0, "lang:krl": 1.0},
            label=1,
            sample_weight=0.5,
        ),
        CandidateExample(
            language="krl",
            word_id=1,
            wordform_id=11,
            gramset="PL",
            word="a",
            word_length=1,
            left_word="<BOS>",
            right_word="b",
            features={"gramset:PL": 1.0, "lang:krl": 1.0},
            label=0,
            sample_weight=0.5,
        ),
    )


def _make_bundle(tmp_path):
    bundle_dir = tmp_path / "models"
    bundle_dir.mkdir()
    path = bundle_dir / "model.bundle"
    model = train_candidate_model(_train_examples())
    save_model_bundle(model, path)
    return path


def _success_env(tmp_path):
    data_dir = tmp_path / "checkout"
    data_dir.mkdir()
    split_file = tmp_path / "splits.csv"
    _write_split_csv(split_file, _split_rows())
    bundle = _make_bundle(tmp_path)
    output = tmp_path / "dev-predictions.csv"
    return data_dir, split_file, bundle, output


def _read_csv(path):
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, strict=True)
        header = next(reader)
        rows = list(reader)
    return header, rows


class TestArgumentParsing:
    def test_help_lists_arguments(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["predict-t1", "--help"])
        captured = capsys.readouterr()
        assert exc.value.code == 0
        assert "--model" in captured.out
        assert "--split" in captured.out
        assert "--output" in captured.out
        assert "--data-dir" in captured.out
        assert "--split-file" in captured.out

    def test_requires_model_output_and_split(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["predict-t1"])
        captured = capsys.readouterr()
        assert exc.value.code == 2
        assert "required" in captured.err
        assert "--model" in captured.err
        assert "--output" in captured.err
        assert "--split" in captured.err

    @pytest.mark.parametrize("split", ["train", "test"])
    def test_rejects_non_dev_splits(self, tmp_path, capsys, split):
        with pytest.raises(SystemExit) as exc:
            cli.main(
                [
                    "predict-t1",
                    "--model", str(tmp_path / "m.bundle"),
                    "--split", split,
                    "--output", str(tmp_path / "out.csv"),
                ]
            )
        captured = capsys.readouterr()
        assert exc.value.code == 2
        assert "invalid choice" in captured.err
        assert split in captured.err

    def test_rejects_unsupported_force_flag(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(
                [
                    "predict-t1",
                    "--model", str(tmp_path / "m.bundle"),
                    "--split", "dev",
                    "--output", str(tmp_path / "out.csv"),
                    "--force",
                ]
            )
        captured = capsys.readouterr()
        assert exc.value.code == 2
        assert "unrecognized arguments" in captured.err
        assert "--force" in captured.err


class TestDestinationProtection:
    def test_existing_output_protected_before_loading(
        self, tmp_path, monkeypatch, capsys
    ):
        _forbid_preflight(monkeypatch)
        _forbid_model_load(monkeypatch)
        output = tmp_path / "dev-predictions.csv"
        output.write_text("original", encoding="utf-8")
        status = cli.main(
            [
                "predict-t1",
                "--model", str(tmp_path / "m.bundle"),
                "--split", "dev",
                "--output", str(output),
            ]
        )
        captured = capsys.readouterr()
        assert status == 1
        assert "already exists" in captured.err
        assert str(output) in captured.err
        assert output.read_text(encoding="utf-8") == "original"
        assert not list(tmp_path.glob(".vepkar-t1-predictions-*.tmp"))
        assert "T1 prediction: OK" not in captured.out

    def test_missing_parent_rejected_without_creation(
        self, tmp_path, monkeypatch, capsys
    ):
        _forbid_preflight(monkeypatch)
        _forbid_model_load(monkeypatch)
        missing_parent = tmp_path / "no-such-dir"
        output = missing_parent / "out.csv"
        status = cli.main(
            [
                "predict-t1",
                "--model", str(tmp_path / "m.bundle"),
                "--split", "dev",
                "--output", str(output),
            ]
        )
        captured = capsys.readouterr()
        assert status == 1
        assert "output directory does not exist" in captured.err
        assert str(missing_parent) in captured.err
        assert not missing_parent.exists()
        assert not list(tmp_path.glob(".vepkar-t1-predictions-*.tmp"))
        assert "T1 prediction: OK" not in captured.out


class TestModelBundleFailures:
    def test_missing_model_bundle_reported_without_corpus_access(
        self, tmp_path, monkeypatch, capsys
    ):
        _forbid_preflight(monkeypatch)
        missing = tmp_path / "missing.bundle"
        output = tmp_path / "out.csv"
        status = cli.main(
            [
                "predict-t1",
                "--model", str(missing),
                "--split", "dev",
                "--output", str(output),
            ]
        )
        captured = capsys.readouterr()
        assert status == 1
        assert "cannot load T1 model bundle" in captured.err
        assert str(missing) in captured.err
        assert not output.exists()
        assert "T1 prediction: OK" not in captured.out

    def test_incompatible_model_bundle_reported_without_corpus_access(
        self, tmp_path, monkeypatch, capsys
    ):
        _forbid_preflight(monkeypatch)
        bundle = tmp_path / "bad.bundle"
        joblib.dump({"wrong": "payload"}, bundle)
        output = tmp_path / "out.csv"
        status = cli.main(
            [
                "predict-t1",
                "--model", str(bundle),
                "--split", "dev",
                "--output", str(output),
            ]
        )
        captured = capsys.readouterr()
        assert status == 1
        assert "cannot load T1 model bundle" in captured.err
        assert not output.exists()
        assert "T1 prediction: OK" not in captured.out


class TestEmptyDevSelection:
    def test_reported_without_invoking_writer(self, tmp_path, monkeypatch, capsys):
        _install_tables(monkeypatch)
        _stub_preflight(monkeypatch)
        data_dir = tmp_path / "checkout"
        data_dir.mkdir()
        split_file = tmp_path / "splits.csv"
        _write_split_csv(
            split_file,
            [("krl", 1, "train"), ("krl", 2, "train"), ("krl", 3, "test")],
        )
        bundle = _make_bundle(tmp_path)
        output = tmp_path / "out.csv"

        def forbidden_write(*args, **kwargs):
            pytest.fail("writer must not be invoked for an empty dev selection")

        monkeypatch.setattr(predict_pkg, "write_dev_predictions", forbidden_write)
        status = cli.main(
            [
                "predict-t1",
                "--model", str(bundle),
                "--split", "dev",
                "--output", str(output),
                "--data-dir", str(data_dir),
                "--split-file", str(split_file),
            ]
        )
        captured = capsys.readouterr()
        assert status == 1
        assert "no dev" in captured.err
        assert not output.exists()
        assert "T1 prediction: OK" not in captured.out


class TestWriterFailure:
    def test_invalid_validation_details_printed_and_no_success(
        self, tmp_path, monkeypatch, capsys
    ):
        _install_tables(monkeypatch)
        _stub_preflight(monkeypatch)
        data_dir, split_file, bundle, output = _success_env(tmp_path)
        validation = PredictionValidation(
            is_valid=False,
            split="dev",
            expected_word_count=1,
            expected_candidate_count=2,
            predicted_word_count=1,
            predicted_candidate_count=1,
            errors={
                "missing candidate": ("word_id=205; candidate=(2051, 'DEV+ALT')",)
            },
            error_counts={"missing candidate": 1},
        )

        def failing_write(rows, output_path, *, benchmark):
            raise T1PredictionWriteError(
                "cannot write dev predictions: temporary CSV failed dev validation",
                validation=validation,
            )

        monkeypatch.setattr(predict_pkg, "write_dev_predictions", failing_write)
        status = cli.main(
            [
                "predict-t1",
                "--model", str(bundle),
                "--split", "dev",
                "--output", str(output),
                "--data-dir", str(data_dir),
                "--split-file", str(split_file),
            ]
        )
        captured = capsys.readouterr()
        assert status == 1
        assert "cannot write dev predictions" in captured.err
        assert "Predictions validation: FAILED" in captured.out
        assert "Expected: 1 word instances, 2 candidate rows" in captured.out
        assert "Found: 1 word instances, 1 prediction rows" in captured.out
        assert "missing candidate: 1 violation" in captured.out
        assert "T1 prediction: OK" not in captured.out
        assert not output.exists()


class TestSuccessfulPrediction:
    def test_real_offline_pipeline_via_cli_main(self, tmp_path, monkeypatch, capsys):
        tables = _install_tables(monkeypatch)
        _stub_preflight(monkeypatch)
        data_dir, split_file, bundle, output = _success_env(tmp_path)
        words_before = tables["krl"].words.copy(deep=True)
        split_before = split_file.read_bytes()
        status = cli.main(
            [
                "predict-t1",
                "--model", str(bundle),
                "--split", "dev",
                "--output", str(output),
                "--data-dir", str(data_dir),
                "--split-file", str(split_file),
            ]
        )
        captured = capsys.readouterr()
        assert status == 0, captured.err
        assert captured.err == ""
        lines = captured.out.splitlines()
        assert lines[0] == "T1 prediction: OK"
        assert f"Model: {bundle}" in captured.out
        assert "Data tag: fixture-tag" in captured.out
        assert f"Split file: {split_file}" in captured.out
        assert "Split: dev" in captured.out
        assert "Predicted word occurrences: 1" in captured.out
        assert "Predicted candidate rows: 2" in captured.out
        assert f"Output: {output}" in captured.out
        assert "Validation: OK" in captured.out

        header, raw_rows = _read_csv(output)
        assert header == list(PREDICTION_HEADER)
        assert len(raw_rows) == 2
        rows = [
            (int(r[0]), int(r[1]), r[2], int(r[3]), float(r[4]))
            for r in raw_rows
        ]
        assert {row[0] for row in rows} == {205}
        assert {(row[1], row[2]) for row in rows} == {
            (2050, "DEV+ONLY"),
            (2051, "DEV+ALT"),
        }
        assert sorted(row[3] for row in rows) == [1, 2]
        assert all(row[0] not in (102, 308) for row in rows)

        loaded = load_model_bundle(bundle)
        inputs = predict_pkg.load_t1_inputs(
            data_dir,
            "fixture-tag",
            split="dev",
            split_file_override=split_file,
        )
        dev_examples = build_candidate_examples(
            inputs.instances, inputs.context_words, mode="predict"
        )
        real_scores = score_candidate_examples(loaded, dev_examples)
        real_by_identity = {
            (example.wordform_id, example.gramset): score
            for example, score in zip(dev_examples, real_scores)
        }
        for row in rows:
            assert row[4] == pytest.approx(real_by_identity[(row[1], row[2])])

        validation = validate_predictions(
            output,
            "dev",
            inputs.benchmark.instances,
            inputs.benchmark.split_rows,
        )
        assert validation.is_valid
        assert validation.expected_word_count == 1
        assert validation.expected_candidate_count == 2

        assert not list(tmp_path.glob(".vepkar-t1-predictions-*.tmp"))
        assert set(p.name for p in tmp_path.iterdir()) == {
            "checkout",
            "dev-predictions.csv",
            "models",
            "splits.csv",
        }
        pd.testing.assert_frame_equal(tables["krl"].words, words_before)
        assert split_file.read_bytes() == split_before
