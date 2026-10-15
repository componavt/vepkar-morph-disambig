"""Offline tests for the train-t1 CLI command."""

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import benchmark_context  # noqa: E402
import cli  # noqa: E402
import commands.train_t1 as train_pkg  # noqa: E402
from core.data import SUPPORTED_LANGUAGES, CorpusTables, DataError  # noqa: E402
from core.instances import CorpusTagError  # noqa: E402
from core.predictions import BenchmarkIntegrityError  # noqa: E402
from core.splits import SplitError  # noqa: E402
from t1_features.features import FeatureConstructionError  # noqa: E402
from t1_features.model import T1Model  # noqa: E402
from t1_features.model_io import ModelBundleError, load_model_bundle  # noqa: E402


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
        train_pkg,
        "_preflight_benchmark",
        lambda data_dir: (data_dir, tag),
    )


def _forbid_preflight(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("must be rejected before corpus access")

    monkeypatch.setattr(train_pkg, "_preflight_benchmark", forbidden)


def _write_split_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["language", "text_id", "split"])
        writer.writerows(rows)


def _success_env(tmp_path):
    data_dir = tmp_path / "checkout"
    data_dir.mkdir()
    split_file = tmp_path / "splits.csv"
    _write_split_csv(
        split_file,
        [("krl", 1, "train"), ("krl", 2, "dev"), ("krl", 3, "test")],
    )
    models = tmp_path / "models"
    models.mkdir()
    output = models / "model.bundle"
    return data_dir, split_file, models, output


def _stub_inputs(monkeypatch):
    benchmark = SimpleNamespace(split_path=Path("splits.csv"))
    monkeypatch.setattr(
        train_pkg,
        "load_t1_inputs",
        lambda data_dir, tag, *, split, split_file_override=None: SimpleNamespace(
            benchmark=benchmark,
            split=split,
            instances=("instance",),
            context_words=("word",),
        ),
    )


def _stub_build_and_train(monkeypatch, model=object()):
    monkeypatch.setattr(
        train_pkg,
        "build_candidate_examples",
        lambda instances, context_words, *, mode: ("example",),
    )
    monkeypatch.setattr(
        train_pkg,
        "train_candidate_model",
        lambda examples, *, C: model,
    )


@pytest.mark.parametrize("c_arg,expected_c", [(None, 1.0), ("2.5", 2.5)])
def test_train_t1_success_via_cli_main(
    tmp_path, monkeypatch, capsys, c_arg, expected_c
):
    tables = _install_tables(monkeypatch)
    _stub_preflight(monkeypatch)
    data_dir, split_file, models, output = _success_env(tmp_path)
    words_before = tables["krl"].words.copy(deep=True)
    split_before = split_file.read_bytes()
    args = [
        "train-t1",
        "--output", str(output),
        "--data-dir", str(data_dir),
        "--split-file", str(split_file),
    ]
    if c_arg is not None:
        args += ["--C", c_arg]
    status = cli.main(args)
    captured = capsys.readouterr()
    assert status == 0, captured.err
    assert captured.err == ""
    assert captured.out.splitlines()[0] == "T1 training: OK"
    assert "Data tag: fixture-tag" in captured.out
    assert f"Split file: {split_file}" in captured.out
    assert "Split: train" in captured.out
    assert "Training word occurrences: 1" in captured.out
    assert "Training candidate examples: 2" in captured.out
    assert f"Model: {output}" in captured.out
    model = load_model_bundle(output)
    assert isinstance(model, T1Model)
    assert model.classifier.C == pytest.approx(expected_c)
    vocabulary = set(model.vectorizer.vocabulary_)
    assert f"Feature columns: {len(vocabulary)}" in captured.out
    for untrain_feature in (
        '["gramset","DEV+ONLY"]',
        '["element","DEV"]',
        '["gramset","TEST+ONLY"]',
        '["element","TEST"]',
    ):
        assert untrain_feature not in vocabulary
    assert sorted(p.name for p in models.iterdir()) == ["model.bundle"]
    assert not list(models.glob(".vepkar-t1-model-*.tmp"))
    assert set(p.name for p in tmp_path.iterdir()) == {
        "checkout",
        "splits.csv",
        "models",
    }
    pd.testing.assert_frame_equal(tables["krl"].words, words_before)
    assert split_file.read_bytes() == split_before


def test_train_t1_help_lists_arguments(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["train-t1", "--help"])
    captured = capsys.readouterr()
    assert exc.value.code == 0
    assert "--output" in captured.out
    assert "--C" in captured.out
    assert "--data-dir" in captured.out
    assert "--split-file" in captured.out
    assert "--split {" not in captured.out


def test_train_t1_requires_output(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["train-t1"])
    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert "required" in captured.err
    assert "--output" in captured.err


@pytest.mark.parametrize("flag", ["--split", "--force"])
def test_train_t1_rejects_unsupported_flags(tmp_path, capsys, flag):
    with pytest.raises(SystemExit) as exc:
        cli.main(
            ["train-t1", "--output", str(tmp_path / "m.bundle"), flag, "x"]
        )
    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert "unrecognized arguments" in captured.err


@pytest.mark.parametrize("value", ["0", "-1", "-inf", "inf", "nan"])
def test_train_t1_invalid_c_rejected_before_loading(
    tmp_path, monkeypatch, capsys, value
):
    _forbid_preflight(monkeypatch)
    output = tmp_path / "model.bundle"
    status = cli.main(
        ["train-t1", "--output", str(output), f"--C={value}"]
    )
    captured = capsys.readouterr()
    assert status == 1
    assert "finite" in captured.err and "positive" in captured.err
    assert not output.exists()
    assert not list(tmp_path.glob(".vepkar-t1-model-*.tmp"))
    assert "T1 training: OK" not in captured.out


def test_train_t1_existing_output_protected(tmp_path, monkeypatch, capsys):
    _forbid_preflight(monkeypatch)
    output = tmp_path / "model.bundle"
    output.write_text("original", encoding="utf-8")
    status = cli.main(["train-t1", "--output", str(output)])
    captured = capsys.readouterr()
    assert status == 1
    assert "already exists" in captured.err
    assert str(output) in captured.err
    assert output.read_text(encoding="utf-8") == "original"
    assert not list(tmp_path.glob(".vepkar-t1-model-*.tmp"))
    assert "T1 training: OK" not in captured.out


def test_train_t1_output_directory_rejected(tmp_path, monkeypatch, capsys):
    _forbid_preflight(monkeypatch)
    output = tmp_path / "outdir"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")
    status = cli.main(["train-t1", "--output", str(output)])
    captured = capsys.readouterr()
    assert status == 1
    assert "output path is a directory" in captured.err
    assert str(output) in captured.err
    assert output.is_dir()
    assert (output / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert "T1 training: OK" not in captured.out


def test_train_t1_dangling_symlink_protected(tmp_path, monkeypatch, capsys):
    _forbid_preflight(monkeypatch)
    output = tmp_path / "model.bundle"
    missing = tmp_path / "missing-target.bundle"
    try:
        output.symlink_to(missing)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is unavailable in this environment")
    status = cli.main(["train-t1", "--output", str(output)])
    captured = capsys.readouterr()
    assert status == 1
    assert "already exists" in captured.err
    assert str(output) in captured.err
    assert output.is_symlink()
    assert output.readlink() == missing
    assert not missing.exists()
    assert "T1 training: OK" not in captured.out


def test_train_t1_missing_parent_rejected_without_creation(
    tmp_path, monkeypatch, capsys
):
    _forbid_preflight(monkeypatch)
    missing_parent = tmp_path / "no-such-dir"
    output = missing_parent / "model.bundle"
    status = cli.main(["train-t1", "--output", str(output)])
    captured = capsys.readouterr()
    assert status == 1
    assert "output directory does not exist" in captured.err
    assert str(missing_parent) in captured.err
    assert not missing_parent.exists()
    assert "T1 training: OK" not in captured.out


def test_train_t1_missing_checkout_reported(tmp_path, monkeypatch, capsys):
    missing = tmp_path / "missing-dictorpus-data"

    def fail_preflight(data_dir):
        raise DataError(f"Local dictorpus-data checkout is missing: {missing}")

    monkeypatch.setattr(train_pkg, "_preflight_benchmark", fail_preflight)
    output = tmp_path / "model.bundle"
    status = cli.main(
        ["train-t1", "--output", str(output), "--data-dir", str(missing)]
    )
    captured = capsys.readouterr()
    assert status == 1
    assert "dictorpus-data" in captured.err
    assert str(missing) in captured.err
    assert not output.exists()
    assert "T1 training: OK" not in captured.out


def test_train_t1_tag_failure_reported(tmp_path, monkeypatch, capsys):
    def fail_preflight(data_dir):
        raise CorpusTagError("No Git tag points at HEAD")

    monkeypatch.setattr(train_pkg, "_preflight_benchmark", fail_preflight)
    output = tmp_path / "model.bundle"
    status = cli.main(["train-t1", "--output", str(output)])
    captured = capsys.readouterr()
    assert status == 1
    assert "No Git tag" in captured.err
    assert not output.exists()
    assert "T1 training: OK" not in captured.out


def test_train_t1_data_error_during_loading_not_duplicated(
    tmp_path, monkeypatch, capsys
):
    _stub_preflight(monkeypatch)

    def fail_tables(lang, data_dir):
        raise DataError(f"corpus tables unavailable for {lang}")

    monkeypatch.setattr(benchmark_context, "read_corpus_tables", fail_tables)
    output = tmp_path / "model.bundle"
    status = cli.main(["train-t1", "--output", str(output)])
    captured = capsys.readouterr()
    assert status == 1
    assert "FAILED" in captured.err
    assert "corpus tables unavailable" in captured.err
    assert not output.exists()
    assert "T1 training: OK" not in captured.out


def test_train_t1_loading_split_error_reported(tmp_path, monkeypatch, capsys):
    _stub_preflight(monkeypatch)

    def fail_load(data_dir, tag, *, split, split_file_override=None):
        raise SplitError("cannot read split file: nope")

    monkeypatch.setattr(train_pkg, "load_t1_inputs", fail_load)
    output = tmp_path / "model.bundle"
    status = cli.main(["train-t1", "--output", str(output)])
    captured = capsys.readouterr()
    assert status == 1
    assert "cannot read split file" in captured.err
    assert not output.exists()
    assert "T1 training: OK" not in captured.out


def test_train_t1_loading_integrity_error_reported(tmp_path, monkeypatch, capsys):
    _stub_preflight(monkeypatch)

    def fail_load(data_dir, tag, *, split, split_file_override=None):
        raise BenchmarkIntegrityError("duplicate word_id values across instances")

    monkeypatch.setattr(train_pkg, "load_t1_inputs", fail_load)
    output = tmp_path / "model.bundle"
    status = cli.main(["train-t1", "--output", str(output)])
    captured = capsys.readouterr()
    assert status == 1
    assert "strict benchmark is internally inconsistent" in captured.err
    assert "duplicate word_id" in captured.err
    assert not output.exists()
    assert "T1 training: OK" not in captured.out


def test_train_t1_construction_error_reported(tmp_path, monkeypatch, capsys):
    _stub_preflight(monkeypatch)
    _stub_inputs(monkeypatch)

    def fail_build(instances, context_words, *, mode):
        raise FeatureConstructionError("position mismatch")

    monkeypatch.setattr(train_pkg, "build_candidate_examples", fail_build)
    output = tmp_path / "model.bundle"
    status = cli.main(["train-t1", "--output", str(output)])
    captured = capsys.readouterr()
    assert status == 1
    assert "T1 feature construction failed" in captured.err
    assert "position mismatch" in captured.err
    assert not output.exists()
    assert "T1 training: OK" not in captured.out


def test_train_t1_training_value_error_reported(tmp_path, monkeypatch, capsys):
    _stub_preflight(monkeypatch)
    _stub_inputs(monkeypatch)

    def fail_train(examples, *, C):
        raise ValueError("train data must contain both classes")

    monkeypatch.setattr(
        train_pkg,
        "build_candidate_examples",
        lambda instances, context_words, *, mode: ("example",),
    )
    monkeypatch.setattr(train_pkg, "train_candidate_model", fail_train)
    output = tmp_path / "model.bundle"
    status = cli.main(["train-t1", "--output", str(output)])
    captured = capsys.readouterr()
    assert status == 1
    assert "T1 training failed" in captured.err
    assert "both classes" in captured.err
    assert not output.exists()
    assert "T1 training: OK" not in captured.out


def test_train_t1_convergence_warning_reported(tmp_path, monkeypatch, capsys):
    _stub_preflight(monkeypatch)
    _stub_inputs(monkeypatch)

    def fail_train(examples, *, C):
        raise ConvergenceWarning("lbfgs failed to converge")

    monkeypatch.setattr(
        train_pkg,
        "build_candidate_examples",
        lambda instances, context_words, *, mode: ("example",),
    )
    monkeypatch.setattr(train_pkg, "train_candidate_model", fail_train)
    output = tmp_path / "model.bundle"
    status = cli.main(["train-t1", "--output", str(output)])
    captured = capsys.readouterr()
    assert status == 1
    assert "did not converge" in captured.err
    assert "lbfgs failed to converge" in captured.err
    assert not output.exists()
    assert "T1 training: OK" not in captured.out


def test_train_t1_save_error_reported(tmp_path, monkeypatch, capsys):
    _stub_preflight(monkeypatch)
    _stub_inputs(monkeypatch)
    _stub_build_and_train(monkeypatch)

    def fail_save(model, path):
        raise ModelBundleError("destination already exists")

    monkeypatch.setattr(train_pkg, "save_model_bundle", fail_save)
    output = tmp_path / "model.bundle"
    status = cli.main(["train-t1", "--output", str(output)])
    captured = capsys.readouterr()
    assert status == 1
    assert "cannot save T1 model bundle" in captured.err
    assert "destination already exists" in captured.err
    assert not output.exists()
    assert "T1 training: OK" not in captured.out


def test_train_t1_published_cleanup_failure_distinguished(
    tmp_path, monkeypatch, capsys
):
    _install_tables(monkeypatch)
    _stub_preflight(monkeypatch)
    data_dir, split_file, models, output = _success_env(tmp_path)
    real_save = train_pkg.save_model_bundle
    retained = {"path": None}

    def failing_cleanup_save(model, path):
        real_save(model, path)
        temp_path = path.parent / ".vepkar-t1-model-retained.tmp"
        temp_path.write_bytes(b"retained")
        retained["path"] = temp_path
        raise train_pkg.ModelBundlePublishedCleanupError(
            path, temp_path
        ) from PermissionError("cleanup denied")

    monkeypatch.setattr(train_pkg, "save_model_bundle", failing_cleanup_save)
    status = cli.main(
        [
            "train-t1",
            "--output", str(output),
            "--data-dir", str(data_dir),
            "--split-file", str(split_file),
        ]
    )
    captured = capsys.readouterr()
    temp_path = retained["path"]
    assert temp_path is not None
    assert status == 1
    assert "published" in captured.err
    assert "temporary cleanup failed" in captured.err
    assert str(output) in captured.err
    assert str(temp_path) in captured.err
    assert "cleanup denied" in captured.err
    assert "cannot save T1 model bundle" not in captured.err
    assert "T1 training: OK" not in captured.out
    model = load_model_bundle(output)
    assert isinstance(model, T1Model)
    assert temp_path.exists()
    assert sorted(p.name for p in models.iterdir()) == [
        ".vepkar-t1-model-retained.tmp",
        "model.bundle",
    ]
    temp_path.unlink()
    assert sorted(p.name for p in models.iterdir()) == ["model.bundle"]