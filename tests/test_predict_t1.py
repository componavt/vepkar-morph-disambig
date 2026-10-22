"""Offline tests for the predict-t1 CLI command."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import joblib
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import benchmark_context  # noqa: E402
import cli  # noqa: E402
import commands.predict_t1 as predict_pkg  # noqa: E402
from core.data import DataError, SUPPORTED_LANGUAGES, CorpusTables  # noqa: E402
from core.predictions import (  # noqa: E402
    PREDICTION_HEADER,
    BenchmarkIntegrityError,
    PredictionValidation,
    validate_predictions,
)
from core.splits import SplitError  # noqa: E402
from t1_features.features import (  # noqa: E402
    CandidateExample,
    FeatureConstructionError,
    build_candidate_examples,
)
from t1_features.model import score_candidate_examples, train_candidate_model  # noqa: E402
from t1_features.model_io import load_model_bundle, save_model_bundle  # noqa: E402
from t1_features.prediction_io import T1PredictionWriteError  # noqa: E402
from t1_features.run_context import T1RunContext, T1RunContextError  # noqa: E402


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


def _forbid_corpus(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("corpus tables must not be loaded")

    monkeypatch.setattr(benchmark_context, "read_corpus_tables", forbidden)


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
    positive_features = {
        '["gramset","DEV+ONLY"]': 1.0,
        '["language","krl"]': 1.0,
    }
    negative_features = {
        '["gramset","DEV+ALT"]': 1.0,
        '["language","krl"]': 1.0,
    }
    return (
        CandidateExample(
            language="krl",
            word_id=1,
            wordform_id=10,
            gramset="DEV+ONLY",
            word="a",
            word_length=1,
            left_word="<BOS>",
            right_word="b",
            features=positive_features,
            label=1,
            sample_weight=0.5,
        ),
        CandidateExample(
            language="krl",
            word_id=1,
            wordform_id=11,
            gramset="DEV+ALT",
            word="a",
            word_length=1,
            left_word="<BOS>",
            right_word="b",
            features=negative_features,
            label=0,
            sample_weight=0.5,
        ),
    )


def _write_passport(run_dir, *, data_root, tag, split_path):
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "vepkar-t1-run",
        "format_version": 4,
        "stage": "prepared",
        "data": {
            "root": str(data_root),
            "tag": tag,
            "commit": "d" * 40,
        },
        "split": {"path": str(split_path), "sha256": "0" * 64},
        "parameters": {"C": 1.0, "target_split": "dev"},
    }
    (run_dir / "run.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _stub_run_context(monkeypatch, run_dir, manifest):
    recorded = {"calls": 0}

    def fake_load(run_dir_arg):
        recorded["calls"] += 1
        recorded["run_dir"] = run_dir_arg
        return T1RunContext(
            run_dir=run_dir.resolve(),
            manifest=manifest,
        )

    monkeypatch.setattr(predict_pkg, "load_t1_run_context", fake_load)
    return recorded


def _make_bundle(run_dir):
    model = train_candidate_model(_train_examples())
    path = run_dir / "model.bundle"
    save_model_bundle(model, path)
    return path


def _success_env(tmp_path):
    data_dir = tmp_path / "checkout"
    data_dir.mkdir()
    split_file = tmp_path / "splits.csv"
    _write_split_csv(split_file, _split_rows())
    run_dir = tmp_path / "runs" / "run-a"
    manifest = _write_passport(
        run_dir,
        data_root=data_dir,
        tag="fixture-tag",
        split_path=split_file,
    )
    bundle = _make_bundle(run_dir)
    return data_dir, split_file, run_dir, manifest, bundle


def _read_csv(path):
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, strict=True)
        header = next(reader)
        rows = list(reader)
    return header, rows


def _recursive_entries(root):
    return {path.relative_to(root).as_posix() for path in root.rglob("*")}


def _raise_stub(exc_type, message):
    def raise_exc(*args, **kwargs):
        raise exc_type(message)

    return raise_exc


def _forbid_stage_call(monkeypatch, name):
    def forbidden(*args, **kwargs):
        pytest.fail(f"stage must not be called: {name}")

    monkeypatch.setattr(predict_pkg, name, forbidden)


class TestArgumentParsing:
    def test_help_lists_arguments(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["predict-t1", "--help"])
        captured = capsys.readouterr()
        assert exc.value.code == 0
        assert "--run-dir" in captured.out
        assert "--strict-code" not in captured.out
        for removed in ("--model", "--split", "--output", "--data-dir", "--split-file"):
            assert removed not in captured.out

    def test_requires_run_dir(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["predict-t1"])
        captured = capsys.readouterr()
        assert exc.value.code == 2
        assert "required" in captured.err
        assert "--run-dir" in captured.err

    def test_rejects_strict_code(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["predict-t1", "--run-dir", str(tmp_path), "--strict-code"])
        captured = capsys.readouterr()
        assert exc.value.code == 2
        assert "unrecognized arguments" in captured.err
        assert "--strict-code" in captured.err

    @pytest.mark.parametrize(
        "removed",
        [
            ["--model", "m.bundle"],
            ["--split", "dev"],
            ["--output", "out.csv"],
            ["--data-dir", "checkout"],
            ["--split-file", "splits.csv"],
            ["--force", "x"],
        ],
    )
    def test_rejects_removed_and_unsupported_options(self, tmp_path, capsys, removed):
        with pytest.raises(SystemExit) as exc:
            cli.main(["predict-t1", "--run-dir", str(tmp_path), *removed])
        captured = capsys.readouterr()
        assert exc.value.code == 2
        assert "unrecognized arguments" in captured.err

    @pytest.mark.parametrize("abbreviation", ["--run", "--str"])
    def test_rejects_abbreviated_options(self, tmp_path, capsys, abbreviation):
        with pytest.raises(SystemExit) as exc:
            cli.main(["predict-t1", "--run-dir", str(tmp_path), abbreviation])
        captured = capsys.readouterr()
        assert exc.value.code == 2
        assert "unrecognized arguments" in captured.err


class TestContextFailure:
    def test_context_error_reported_without_pipeline_access(
        self, tmp_path, monkeypatch, capsys
    ):
        _forbid_model_load(monkeypatch)
        _forbid_corpus(monkeypatch)
        _forbid_stage_call(monkeypatch, "load_t1_inputs")
        _forbid_stage_call(monkeypatch, "write_dev_predictions")
        run_dir = tmp_path / "runs" / "run-a"

        def fail_load(run_dir_arg):
            raise T1RunContextError(
                f"cannot load T1 run passport {run_dir_arg / 'run.json'}: missing"
            )

        monkeypatch.setattr(predict_pkg, "load_t1_run_context", fail_load)
        status = cli.main(["predict-t1", "--run-dir", str(run_dir)])
        captured = capsys.readouterr()
        assert status == 1
        assert "cannot load T1 run passport" in captured.err
        assert str(run_dir / "run.json") in captured.err
        assert not (run_dir / "dev.csv").exists()
        assert "T1 prediction: OK" not in captured.out


class TestDestinationProtection:
    def test_existing_dev_csv_protected_before_loading(
        self, tmp_path, monkeypatch, capsys
    ):
        run_dir = tmp_path / "runs" / "run-a"
        manifest = _write_passport(
            run_dir,
            data_root=tmp_path / "checkout",
            tag="fixture-tag",
            split_path=tmp_path / "splits.csv",
        )
        _stub_run_context(monkeypatch, run_dir, manifest)
        _forbid_model_load(monkeypatch)
        _forbid_corpus(monkeypatch)
        output = run_dir / "dev.csv"
        output.write_text("original", encoding="utf-8")
        status = cli.main(["predict-t1", "--run-dir", str(run_dir)])
        captured = capsys.readouterr()
        assert status == 1
        assert "already exists" in captured.err
        assert str(output) in captured.err
        assert output.read_text(encoding="utf-8") == "original"
        assert not list(run_dir.glob(".vepkar-t1-predictions-*.tmp"))
        assert "T1 prediction: OK" not in captured.out

    def test_dev_csv_directory_rejected_before_loading(
        self, tmp_path, monkeypatch, capsys
    ):
        run_dir = tmp_path / "runs" / "run-a"
        manifest = _write_passport(
            run_dir,
            data_root=tmp_path / "checkout",
            tag="fixture-tag",
            split_path=tmp_path / "splits.csv",
        )
        _stub_run_context(monkeypatch, run_dir, manifest)
        _forbid_model_load(monkeypatch)
        _forbid_corpus(monkeypatch)
        output = run_dir / "dev.csv"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("original", encoding="utf-8")
        entries_before = _recursive_entries(tmp_path)
        status = cli.main(["predict-t1", "--run-dir", str(run_dir)])
        captured = capsys.readouterr()
        assert status == 1
        assert "directory" in captured.err
        assert str(output) in captured.err
        assert marker.read_text(encoding="utf-8") == "original"
        assert _recursive_entries(tmp_path) == entries_before
        assert not list(run_dir.glob(".vepkar-t1-predictions-*.tmp"))
        assert "T1 prediction: OK" not in captured.out

    def test_dev_csv_dangling_symlink_rejected_before_loading(
        self, tmp_path, monkeypatch, capsys
    ):
        run_dir = tmp_path / "runs" / "run-a"
        manifest = _write_passport(
            run_dir,
            data_root=tmp_path / "checkout",
            tag="fixture-tag",
            split_path=tmp_path / "splits.csv",
        )
        _stub_run_context(monkeypatch, run_dir, manifest)
        _forbid_model_load(monkeypatch)
        _forbid_corpus(monkeypatch)
        target = tmp_path / "absent.csv"
        output = run_dir / "dev.csv"
        try:
            output.symlink_to(target)
        except (OSError, NotImplementedError):
            pytest.skip("Symlink creation is unavailable in this environment")
        entries_before = _recursive_entries(tmp_path)
        status = cli.main(["predict-t1", "--run-dir", str(run_dir)])
        captured = capsys.readouterr()
        assert status == 1
        assert "already exists" in captured.err
        assert str(output) in captured.err
        assert output.is_symlink()
        assert output.readlink() == target
        assert _recursive_entries(tmp_path) == entries_before
        assert not list(run_dir.glob(".vepkar-t1-predictions-*.tmp"))
        assert "T1 prediction: OK" not in captured.out


class TestModelBundleFailures:
    def test_missing_model_bundle_reported_without_corpus_access(
        self, tmp_path, monkeypatch, capsys
    ):
        run_dir = tmp_path / "runs" / "run-a"
        manifest = _write_passport(
            run_dir,
            data_root=tmp_path / "checkout",
            tag="fixture-tag",
            split_path=tmp_path / "splits.csv",
        )
        _stub_run_context(monkeypatch, run_dir, manifest)
        _forbid_corpus(monkeypatch)
        bundle = run_dir / "model.bundle"
        status = cli.main(["predict-t1", "--run-dir", str(run_dir)])
        captured = capsys.readouterr()
        assert status == 1
        assert "cannot load T1 model bundle" in captured.err
        assert str(bundle) in captured.err
        assert not (run_dir / "dev.csv").exists()
        assert "T1 prediction: OK" not in captured.out

    def test_incompatible_model_bundle_reported_without_corpus_access(
        self, tmp_path, monkeypatch, capsys
    ):
        run_dir = tmp_path / "runs" / "run-a"
        manifest = _write_passport(
            run_dir,
            data_root=tmp_path / "checkout",
            tag="fixture-tag",
            split_path=tmp_path / "splits.csv",
        )
        _stub_run_context(monkeypatch, run_dir, manifest)
        _forbid_corpus(monkeypatch)
        bundle = run_dir / "model.bundle"
        joblib.dump({"wrong": "payload"}, bundle)
        status = cli.main(["predict-t1", "--run-dir", str(run_dir)])
        captured = capsys.readouterr()
        assert status == 1
        assert "cannot load T1 model bundle" in captured.err
        assert not (run_dir / "dev.csv").exists()
        assert "T1 prediction: OK" not in captured.out


class TestEmptyDevSelection:
    def test_reported_without_invoking_writer(self, tmp_path, monkeypatch, capsys):
        _install_tables(monkeypatch)
        data_dir = tmp_path / "checkout"
        data_dir.mkdir()
        split_file = tmp_path / "splits.csv"
        _write_split_csv(
            split_file,
            [("krl", 1, "train"), ("krl", 2, "train"), ("krl", 3, "test")],
        )
        run_dir = tmp_path / "runs" / "run-a"
        manifest = _write_passport(
            run_dir,
            data_root=data_dir,
            tag="fixture-tag",
            split_path=split_file,
        )
        _stub_run_context(monkeypatch, run_dir, manifest)
        _make_bundle(run_dir)

        def forbidden_write(*args, **kwargs):
            pytest.fail("writer must not be invoked for an empty dev selection")

        monkeypatch.setattr(predict_pkg, "write_dev_predictions", forbidden_write)
        status = cli.main(["predict-t1", "--run-dir", str(run_dir)])
        captured = capsys.readouterr()
        assert status == 1
        assert "no dev" in captured.err
        assert not (run_dir / "dev.csv").exists()
        assert "T1 prediction: OK" not in captured.out


class TestWriterFailure:
    def test_invalid_validation_details_printed_and_no_success(
        self, tmp_path, monkeypatch, capsys
    ):
        _install_tables(monkeypatch)
        data_dir, split_file, run_dir, manifest, bundle = _success_env(tmp_path)
        _stub_run_context(monkeypatch, run_dir, manifest)
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
        output = run_dir / "dev.csv"

        def failing_write(rows, output_path, *, benchmark):
            raise T1PredictionWriteError(
                "cannot write dev predictions: temporary CSV failed dev validation",
                validation=validation,
            )

        monkeypatch.setattr(predict_pkg, "write_dev_predictions", failing_write)
        status = cli.main(["predict-t1", "--run-dir", str(run_dir)])
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
        data_dir, split_file, run_dir, manifest, bundle = _success_env(tmp_path)
        recorded = _stub_run_context(monkeypatch, run_dir, manifest)

        passport_before = (run_dir / "run.json").read_bytes()
        split_before = split_file.read_bytes()
        bundle_before = bundle.read_bytes()
        tables_before = {
            lang: {
                table_name: getattr(tables[lang], table_name).copy(deep=True)
                for table_name in ("texts", "sentences", "words", "candidates")
            }
            for lang in SUPPORTED_LANGUAGES
        }
        entries_before = _recursive_entries(tmp_path)

        captured_model = {}

        def forbid_fit(*args, **kwargs):
            raise AssertionError("prediction must never retrain or refit the model")

        monkeypatch.setattr(LogisticRegression, "fit", forbid_fit)
        monkeypatch.setattr(DictVectorizer, "fit", forbid_fit)
        monkeypatch.setattr(DictVectorizer, "fit_transform", forbid_fit)

        original_load = predict_pkg.load_model_bundle

        def guarded_load_model(path):
            model = original_load(path)
            captured_model["model"] = model
            captured_model["snapshot"] = {
                "vocabulary": dict(model.vectorizer.vocabulary_),
                "feature_names": model.vectorizer.get_feature_names_out().copy(),
                "classes": model.classifier.classes_.copy(),
                "coef": model.classifier.coef_.copy(),
                "intercept": model.classifier.intercept_.copy(),
            }
            return model

        monkeypatch.setattr(predict_pkg, "load_model_bundle", guarded_load_model)

        status = cli.main(
            ["predict-t1", "--run-dir", str(run_dir)]
        )
        captured = capsys.readouterr()
        output = run_dir / "dev.csv"
        assert status == 0, captured.err
        assert captured.err == ""
        lines = captured.out.splitlines()
        assert lines[0] == "T1 prediction: OK"
        assert f"Run directory: {run_dir.resolve()}" in captured.out
        assert f"Model: {bundle}" in captured.out
        assert "Data tag: fixture-tag" in captured.out
        assert f"Split file: {split_file}" in captured.out
        assert "Split: dev" in captured.out
        assert "Predicted word occurrences: 1" in captured.out
        assert "Predicted candidate rows: 2" in captured.out
        assert f"Output: {output}" in captured.out
        assert "Validation: OK" in captured.out
        assert recorded["calls"] == 1
        assert recorded["run_dir"] == run_dir

        header, raw_rows = _read_csv(output)
        assert header == list(PREDICTION_HEADER)
        assert len(raw_rows) == 2
        assert [(int(r[1]), r[2], int(r[3])) for r in raw_rows] == [
            (2050, "DEV+ONLY", 1),
            (2051, "DEV+ALT", 2),
        ]
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
        assert rows[0][4] != rows[1][4]
        assert rows[0][4] > rows[1][4]

        model = captured_model["model"]
        snapshot = captured_model["snapshot"]
        assert dict(model.vectorizer.vocabulary_) == snapshot["vocabulary"]
        np.testing.assert_array_equal(
            model.vectorizer.get_feature_names_out(), snapshot["feature_names"]
        )
        np.testing.assert_array_equal(model.classifier.classes_, snapshot["classes"])
        np.testing.assert_array_equal(model.classifier.coef_, snapshot["coef"])
        np.testing.assert_array_equal(
            model.classifier.intercept_, snapshot["intercept"]
        )

        assert bundle.read_bytes() == bundle_before
        assert split_file.read_bytes() == split_before
        assert (run_dir / "run.json").read_bytes() == passport_before
        for lang in SUPPORTED_LANGUAGES:
            for table_name in ("texts", "sentences", "words", "candidates"):
                pd.testing.assert_frame_equal(
                    getattr(tables[lang], table_name),
                    tables_before[lang][table_name],
                )
        assert _recursive_entries(tmp_path) == entries_before | {
            "runs/run-a/dev.csv"
        }
        assert not list(run_dir.glob(".vepkar-t1-predictions-*.tmp"))

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


class TestStageFailures:
    @pytest.mark.parametrize(
        "case",
        [
            {
                "stage": "input",
                "target": "load_t1_inputs",
                "exc_type": SplitError,
                "exc_message": "Unsupported split label 'bogus' for ('krl', 1)",
                "diag": ("error: Unsupported split label 'bogus'",),
                "forbid_later": (
                    "build_candidate_examples",
                    "score_candidate_examples",
                    "rank_candidate_scores",
                    "write_dev_predictions",
                ),
            },
            {
                "stage": "input",
                "target": "load_t1_inputs",
                "exc_type": BenchmarkIntegrityError,
                "exc_message": "Strict benchmark has duplicate word_id values",
                "diag": (
                    "error: strict benchmark is internally inconsistent:",
                    "duplicate word_id",
                ),
                "forbid_later": (
                    "build_candidate_examples",
                    "score_candidate_examples",
                    "rank_candidate_scores",
                    "write_dev_predictions",
                ),
            },
            {
                "stage": "load-data",
                "exc_type": DataError,
                "exc_message": "Expected corpus export not found: vep texts table",
                "diag": ("FAILED", "Expected corpus export not found"),
                "forbid_later": (
                    "build_candidate_examples",
                    "score_candidate_examples",
                    "rank_candidate_scores",
                    "write_dev_predictions",
                ),
                "install_tables": False,
            },
            {
                "stage": "construction",
                "target": "build_candidate_examples",
                "exc_type": FeatureConstructionError,
                "exc_message": "target word missing from sentence context",
                "diag": (
                    "error: T1 feature construction failed:",
                    "target word missing",
                ),
                "forbid_later": (
                    "score_candidate_examples",
                    "rank_candidate_scores",
                    "write_dev_predictions",
                ),
            },
            {
                "stage": "scoring",
                "target": "score_candidate_examples",
                "exc_type": NotFittedError,
                "exc_message": "This LogisticRegression instance is not fitted yet",
                "diag": ("error: T1 model is not fitted:", "not fitted"),
                "forbid_later": ("rank_candidate_scores", "write_dev_predictions"),
            },
            {
                "stage": "scoring",
                "target": "score_candidate_examples",
                "exc_type": ValueError,
                "exc_message": "candidate scores must be finite",
                "diag": (
                    "error: T1 scoring failed:",
                    "candidate scores must be finite",
                ),
                "forbid_later": ("rank_candidate_scores", "write_dev_predictions"),
            },
            {
                "stage": "ranking",
                "target": "rank_candidate_scores",
                "exc_type": BenchmarkIntegrityError,
                "exc_message": "duplicate candidate identity",
                "diag": (
                    "error: strict benchmark is internally inconsistent:",
                    "duplicate candidate identity",
                ),
                "forbid_later": ("write_dev_predictions",),
            },
            {
                "stage": "ranking",
                "target": "rank_candidate_scores",
                "exc_type": ValueError,
                "exc_message": "score must be finite",
                "diag": ("error: T1 ranking failed:", "score must be finite"),
                "forbid_later": ("write_dev_predictions",),
            },
            {
                "stage": "writer",
                "exc_type": T1PredictionWriteError,
                "exc_message": "cannot write dev predictions: publish failed: test failure",
                "note": (
                    "failed to clean up temporary dev predictions "
                    ".vepkar-t1-predictions-1.tmp: Permission denied"
                ),
                "diag": (
                    "error: cannot write dev predictions",
                    "failed to clean up temporary",
                ),
                "forbid_later": (),
            },
        ],
        ids=lambda case: f"{case['stage']}-{case['exc_type'].__name__}",
    )
    def test_stage_failure_reported(self, tmp_path, monkeypatch, capsys, case):
        if case.get("install_tables", True):
            _install_tables(monkeypatch)
        data_dir, split_file, run_dir, manifest, bundle = _success_env(tmp_path)
        _stub_run_context(monkeypatch, run_dir, manifest)

        for name in case["forbid_later"]:
            _forbid_stage_call(monkeypatch, name)

        if case["stage"] == "load-data":

            def raise_data_error(*args, **kwargs):
                raise DataError(case["exc_message"])

            monkeypatch.setattr(
                benchmark_context, "read_corpus_tables", raise_data_error
            )
        elif case["stage"] == "writer":

            def failing_writer(rows, output_path, *, benchmark):
                error = T1PredictionWriteError(case["exc_message"])
                error.add_note(case["note"])
                raise error

            monkeypatch.setattr(predict_pkg, "write_dev_predictions", failing_writer)
        else:
            monkeypatch.setattr(
                predict_pkg,
                case["target"],
                _raise_stub(case["exc_type"], case["exc_message"]),
            )

        status = cli.main(["predict-t1", "--run-dir", str(run_dir)])
        captured = capsys.readouterr()
        assert status == 1
        for fragment in case["diag"]:
            assert fragment in captured.err
        if case["stage"] == "load-data":
            assert captured.err.count(case["exc_message"]) == 1
        assert "Traceback" not in captured.err
        assert "T1 prediction: OK" not in captured.out
        assert not (run_dir / "dev.csv").exists()
        assert not list(run_dir.glob(".vepkar-t1-predictions-*.tmp"))


class TestPublishedCleanupFailure:
    def test_published_csv_reported_when_cleanup_fails(
        self, tmp_path, monkeypatch, capsys
    ):
        _install_tables(monkeypatch)
        data_dir, split_file, run_dir, manifest, bundle = _success_env(tmp_path)
        _stub_run_context(monkeypatch, run_dir, manifest)
        output = run_dir / "dev.csv"
        bundle_before = bundle.read_bytes()

        original_unlink = Path.unlink

        def guarded_unlink(path, *args, **kwargs):
            if (
                path.name.startswith(".vepkar-t1-predictions-")
                and path.name.endswith(".tmp")
            ):
                raise PermissionError(
                    f"permission denied: remove temporary predictions file {path}"
                )
            return original_unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", guarded_unlink)

        status = cli.main(["predict-t1", "--run-dir", str(run_dir)])
        captured = capsys.readouterr()
        temp_files = list(run_dir.glob(".vepkar-t1-predictions-*.tmp"))

        assert status == 1
        assert "were published" in captured.err
        assert "temporary cleanup failed" in captured.err
        assert str(output) in captured.err
        assert len(temp_files) == 1
        assert str(temp_files[0]) in captured.err
        assert "permission denied" in captured.err
        assert "T1 prediction: OK" not in captured.out
        assert output.exists()

        header, raw_rows = _read_csv(output)
        assert header == list(PREDICTION_HEADER)
        assert len(raw_rows) == 2

        inputs = predict_pkg.load_t1_inputs(
            data_dir,
            "fixture-tag",
            split="dev",
            split_file_override=split_file,
        )
        validation = validate_predictions(
            output,
            "dev",
            inputs.benchmark.instances,
            inputs.benchmark.split_rows,
        )
        assert validation.is_valid
        assert validation.expected_word_count == 1
        assert validation.expected_candidate_count == 2
        assert validation.predicted_candidate_count == 2

        assert bundle.read_bytes() == bundle_before

        original_unlink(temp_files[0])
