"""Offline tests for the train-t1 CLI command."""

import csv
import json
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
from core.predictions import BenchmarkIntegrityError  # noqa: E402
from core.splits import SplitError  # noqa: E402
from t1_features.features import FeatureConstructionError  # noqa: E402
from t1_features.model import T1Model  # noqa: E402
from t1_features.model_io import ModelBundleError, load_model_bundle  # noqa: E402
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


def _write_split_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["language", "text_id", "split"])
        writer.writerows(rows)


def _write_passport(run_dir, *, data_root, tag, split_path, C=1.0):
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "vepkar-t1-run",
        "format_version": 3,
        "stage": "prepared",
        "code": {"root": "/code", "commit": "c" * 40},
        "data": {
            "root": str(data_root),
            "tag": tag,
            "commit": "d" * 40,
        },
        "split": {"path": str(split_path), "sha256": "0" * 64},
        "parameters": {"C": C, "target_split": "dev"},
    }
    (run_dir / "run.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _stub_run_context(monkeypatch, run_dir, manifest):
    recorded = {}

    def fake_load(run_dir_arg, *, strict_code=False):
        recorded["run_dir"] = run_dir_arg
        recorded["strict_code"] = strict_code
        return T1RunContext(
            run_dir=run_dir.resolve(),
            manifest=manifest,
            current_code_commit=manifest["code"]["commit"],
        )

    monkeypatch.setattr(train_pkg, "load_t1_run_context", fake_load)
    return recorded


def _success_env(tmp_path, C=1.0):
    data_dir = tmp_path / "checkout"
    data_dir.mkdir()
    split_file = tmp_path / "splits.csv"
    _write_split_csv(
        split_file,
        [("krl", 1, "train"), ("krl", 2, "dev"), ("krl", 3, "test")],
    )
    run_dir = tmp_path / "runs" / "run-a"
    manifest = _write_passport(
        run_dir,
        data_root=data_dir,
        tag="fixture-tag",
        split_path=split_file,
        C=C,
    )
    return data_dir, split_file, run_dir, manifest


def _context_env(tmp_path, monkeypatch):
    run_dir = tmp_path / "run-a"
    manifest = _write_passport(
        run_dir,
        data_root=tmp_path / "checkout",
        tag="fixture-tag",
        split_path=tmp_path / "splits.csv",
    )
    recorded = _stub_run_context(monkeypatch, run_dir, manifest)
    return run_dir, manifest, recorded


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


def _forbid_pipeline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("training pipeline must not run")

    monkeypatch.setattr(train_pkg, "load_t1_inputs", forbidden)
    monkeypatch.setattr(train_pkg, "build_candidate_examples", forbidden)
    monkeypatch.setattr(train_pkg, "train_candidate_model", forbidden)
    monkeypatch.setattr(train_pkg, "save_model_bundle", forbidden)


def _forbid_corpus(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("corpus tables must not be loaded")

    monkeypatch.setattr(benchmark_context, "read_corpus_tables", forbidden)


@pytest.mark.parametrize("passport_c", [1.0, 0.1])
def test_train_t1_success_via_cli_main(
    tmp_path, monkeypatch, capsys, passport_c
):
    tables = _install_tables(monkeypatch)
    data_dir, split_file, run_dir, manifest = _success_env(tmp_path, C=passport_c)
    recorded = _stub_run_context(monkeypatch, run_dir, manifest)
    words_before = tables["krl"].words.copy(deep=True)
    split_before = split_file.read_bytes()
    passport_before = (run_dir / "run.json").read_bytes()
    status = cli.main(
        ["train-t1", "--run-dir", str(run_dir), "--strict-code"]
    )
    captured = capsys.readouterr()
    assert status == 0, captured.err
    assert captured.err == ""
    assert recorded["run_dir"] == run_dir
    assert recorded["strict_code"] is True
    assert captured.out.splitlines()[0] == "T1 training: OK"
    assert f"Run directory: {run_dir.resolve()}" in captured.out
    assert "Data tag: fixture-tag" in captured.out
    assert f"Split file: {split_file}" in captured.out
    assert "Split: train" in captured.out
    assert "Training word occurrences: 1" in captured.out
    assert "Training candidate examples: 2" in captured.out
    assert f"C: {passport_c}" in captured.out
    output = run_dir / "model.bundle"
    assert f"Model: {output}" in captured.out
    model = load_model_bundle(output)
    assert isinstance(model, T1Model)
    assert model.classifier.C == pytest.approx(passport_c)
    vocabulary = set(model.vectorizer.vocabulary_)
    assert f"Feature columns: {len(vocabulary)}" in captured.out
    for untrain_feature in (
        '["gramset","DEV+ONLY"]',
        '["element","DEV"]',
        '["gramset","TEST+ONLY"]',
        '["element","TEST"]',
    ):
        assert untrain_feature not in vocabulary
    assert sorted(p.name for p in run_dir.iterdir()) == ["model.bundle", "run.json"]
    assert not list(run_dir.glob(".vepkar-t1-model-*.tmp"))
    pd.testing.assert_frame_equal(tables["krl"].words, words_before)
    assert split_file.read_bytes() == split_before
    assert (run_dir / "run.json").read_bytes() == passport_before


def test_train_t1_strict_code_defaults_to_false(
    tmp_path, monkeypatch, capsys
):
    _install_tables(monkeypatch)
    data_dir, split_file, run_dir, manifest = _success_env(tmp_path)
    recorded = _stub_run_context(monkeypatch, run_dir, manifest)
    status = cli.main(["train-t1", "--run-dir", str(run_dir)])
    captured = capsys.readouterr()
    assert status == 0, captured.err
    assert recorded["strict_code"] is False


def test_train_t1_help_lists_arguments(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["train-t1", "--help"])
    captured = capsys.readouterr()
    assert exc.value.code == 0
    assert "--run-dir" in captured.out
    assert "--strict-code" in captured.out
    for removed in ("--output", "--C", "--data-dir", "--split-file"):
        assert removed not in captured.out


def test_train_t1_requires_run_dir(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["train-t1"])
    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert "required" in captured.err
    assert "--run-dir" in captured.err


@pytest.mark.parametrize(
    "removed",
    [
        ["--output", "model.bundle"],
        ["--C", "2.5"],
        ["--data-dir", "checkout"],
        ["--split-file", "splits.csv"],
        ["--force", "x"],
        ["--split", "train"],
    ],
)
def test_train_t1_rejects_removed_and_unsupported_options(
    tmp_path, capsys, removed
):
    with pytest.raises(SystemExit) as exc:
        cli.main(["train-t1", "--run-dir", str(tmp_path), *removed])
    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert "unrecognized arguments" in captured.err


def test_train_t1_context_error_reported(tmp_path, monkeypatch, capsys):
    _forbid_pipeline(monkeypatch)

    def fail_load(run_dir_arg, *, strict_code=False):
        raise T1RunContextError(
            f"cannot load T1 run passport {run_dir_arg / 'run.json'}: missing"
        )

    monkeypatch.setattr(train_pkg, "load_t1_run_context", fail_load)
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    status = cli.main(["train-t1", "--run-dir", str(run_dir)])
    captured = capsys.readouterr()
    assert status == 1
    assert "cannot load T1 run passport" in captured.err
    assert str(run_dir / "run.json") in captured.err
    assert not (run_dir / "model.bundle").exists()
    assert "T1 training: OK" not in captured.out


def test_train_t1_strict_code_refusal_reported(tmp_path, monkeypatch, capsys):
    _forbid_pipeline(monkeypatch)
    recorded = {}

    def fail_load(run_dir_arg, *, strict_code=False):
        recorded["strict_code"] = strict_code
        raise T1RunContextError(
            f"T1 run {run_dir_arg.resolve()}: strict-code refusal: recorded "
            f"code commit does not match current code commit"
        )

    monkeypatch.setattr(train_pkg, "load_t1_run_context", fail_load)
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    status = cli.main(
        ["train-t1", "--run-dir", str(run_dir), "--strict-code"]
    )
    captured = capsys.readouterr()
    assert recorded["strict_code"] is True
    assert status == 1
    assert "strict-code refusal" in captured.err
    assert str(run_dir.resolve()) in captured.err
    assert not (run_dir / "model.bundle").exists()
    assert "T1 training: OK" not in captured.out


def test_train_t1_existing_model_protected(tmp_path, monkeypatch, capsys):
    _forbid_corpus(monkeypatch)
    run_dir, manifest, _ = _context_env(tmp_path, monkeypatch)
    output = run_dir / "model.bundle"
    output.write_text("original", encoding="utf-8")
    status = cli.main(["train-t1", "--run-dir", str(run_dir)])
    captured = capsys.readouterr()
    assert status == 1
    assert "already exists" in captured.err
    assert str(output) in captured.err
    assert output.read_text(encoding="utf-8") == "original"
    assert not list(run_dir.glob(".vepkar-t1-model-*.tmp"))
    assert "T1 training: OK" not in captured.out


def test_train_t1_model_directory_rejected(tmp_path, monkeypatch, capsys):
    _forbid_corpus(monkeypatch)
    run_dir, manifest, _ = _context_env(tmp_path, monkeypatch)
    output = run_dir / "model.bundle"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")
    status = cli.main(["train-t1", "--run-dir", str(run_dir)])
    captured = capsys.readouterr()
    assert status == 1
    assert "directory" in captured.err
    assert str(output) in captured.err
    assert output.is_dir()
    assert (output / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert "T1 training: OK" not in captured.out


def test_train_t1_dangling_symlink_protected(tmp_path, monkeypatch, capsys):
    _forbid_corpus(monkeypatch)
    run_dir, manifest, _ = _context_env(tmp_path, monkeypatch)
    output = run_dir / "model.bundle"
    missing = tmp_path / "missing-target.bundle"
    try:
        output.symlink_to(missing)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is unavailable in this environment")
    status = cli.main(["train-t1", "--run-dir", str(run_dir)])
    captured = capsys.readouterr()
    assert status == 1
    assert "already exists" in captured.err
    assert str(output) in captured.err
    assert output.is_symlink()
    assert output.readlink() == missing
    assert not missing.exists()
    assert "T1 training: OK" not in captured.out


def test_train_t1_data_error_during_loading_not_duplicated(
    tmp_path, monkeypatch, capsys
):
    run_dir, manifest, _ = _context_env(tmp_path, monkeypatch)

    def fail_tables(lang, data_dir):
        raise DataError(f"corpus tables unavailable for {lang}")

    monkeypatch.setattr(benchmark_context, "read_corpus_tables", fail_tables)
    status = cli.main(["train-t1", "--run-dir", str(run_dir)])
    captured = capsys.readouterr()
    assert status == 1
    assert "FAILED" in captured.err
    assert "corpus tables unavailable" in captured.err
    assert not (run_dir / "model.bundle").exists()
    assert "T1 training: OK" not in captured.out


def test_train_t1_loading_split_error_reported(tmp_path, monkeypatch, capsys):
    run_dir, manifest, _ = _context_env(tmp_path, monkeypatch)

    def fail_load(data_dir, tag, *, split, split_file_override=None):
        raise SplitError("cannot read split file: nope")

    monkeypatch.setattr(train_pkg, "load_t1_inputs", fail_load)
    status = cli.main(["train-t1", "--run-dir", str(run_dir)])
    captured = capsys.readouterr()
    assert status == 1
    assert "cannot read split file" in captured.err
    assert not (run_dir / "model.bundle").exists()
    assert "T1 training: OK" not in captured.out


def test_train_t1_loading_integrity_error_reported(
    tmp_path, monkeypatch, capsys
):
    run_dir, manifest, _ = _context_env(tmp_path, monkeypatch)

    def fail_load(data_dir, tag, *, split, split_file_override=None):
        raise BenchmarkIntegrityError("duplicate word_id values across instances")

    monkeypatch.setattr(train_pkg, "load_t1_inputs", fail_load)
    status = cli.main(["train-t1", "--run-dir", str(run_dir)])
    captured = capsys.readouterr()
    assert status == 1
    assert "strict benchmark is internally inconsistent" in captured.err
    assert "duplicate word_id" in captured.err
    assert not (run_dir / "model.bundle").exists()
    assert "T1 training: OK" not in captured.out


def test_train_t1_construction_error_reported(tmp_path, monkeypatch, capsys):
    run_dir, manifest, _ = _context_env(tmp_path, monkeypatch)
    _stub_inputs(monkeypatch)

    def fail_build(instances, context_words, *, mode):
        raise FeatureConstructionError("position mismatch")

    monkeypatch.setattr(train_pkg, "build_candidate_examples", fail_build)
    status = cli.main(["train-t1", "--run-dir", str(run_dir)])
    captured = capsys.readouterr()
    assert status == 1
    assert "T1 feature construction failed" in captured.err
    assert "position mismatch" in captured.err
    assert not (run_dir / "model.bundle").exists()
    assert "T1 training: OK" not in captured.out


def test_train_t1_training_value_error_reported(tmp_path, monkeypatch, capsys):
    run_dir, manifest, _ = _context_env(tmp_path, monkeypatch)
    _stub_inputs(monkeypatch)

    def fail_train(examples, *, C):
        raise ValueError("train data must contain both classes")

    monkeypatch.setattr(
        train_pkg,
        "build_candidate_examples",
        lambda instances, context_words, *, mode: ("example",),
    )
    monkeypatch.setattr(train_pkg, "train_candidate_model", fail_train)
    status = cli.main(["train-t1", "--run-dir", str(run_dir)])
    captured = capsys.readouterr()
    assert status == 1
    assert "T1 training failed" in captured.err
    assert "both classes" in captured.err
    assert not (run_dir / "model.bundle").exists()
    assert "T1 training: OK" not in captured.out


def test_train_t1_convergence_warning_reported(tmp_path, monkeypatch, capsys):
    run_dir, manifest, _ = _context_env(tmp_path, monkeypatch)
    _stub_inputs(monkeypatch)

    def fail_train(examples, *, C):
        raise ConvergenceWarning("lbfgs failed to converge")

    monkeypatch.setattr(
        train_pkg,
        "build_candidate_examples",
        lambda instances, context_words, *, mode: ("example",),
    )
    monkeypatch.setattr(train_pkg, "train_candidate_model", fail_train)
    status = cli.main(["train-t1", "--run-dir", str(run_dir)])
    captured = capsys.readouterr()
    assert status == 1
    assert "did not converge" in captured.err
    assert "lbfgs failed to converge" in captured.err
    assert not (run_dir / "model.bundle").exists()
    assert "T1 training: OK" not in captured.out


def test_train_t1_save_error_reported(tmp_path, monkeypatch, capsys):
    run_dir, manifest, _ = _context_env(tmp_path, monkeypatch)
    _stub_inputs(monkeypatch)
    _stub_build_and_train(monkeypatch)

    def fail_save(model, path):
        raise ModelBundleError("destination already exists")

    monkeypatch.setattr(train_pkg, "save_model_bundle", fail_save)
    status = cli.main(["train-t1", "--run-dir", str(run_dir)])
    captured = capsys.readouterr()
    assert status == 1
    assert "cannot save T1 model bundle" in captured.err
    assert "destination already exists" in captured.err
    assert not (run_dir / "model.bundle").exists()
    assert "T1 training: OK" not in captured.out


def test_train_t1_published_cleanup_failure_distinguished(
    tmp_path, monkeypatch, capsys
):
    _install_tables(monkeypatch)
    data_dir, split_file, run_dir, manifest = _success_env(tmp_path)
    _stub_run_context(monkeypatch, run_dir, manifest)
    output = run_dir / "model.bundle"
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
    status = cli.main(["train-t1", "--run-dir", str(run_dir)])
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
    assert sorted(p.name for p in run_dir.iterdir()) == [
        ".vepkar-t1-model-retained.tmp",
        "model.bundle",
        "run.json",
    ]
    temp_path.unlink()
    assert sorted(p.name for p in run_dir.iterdir()) == ["model.bundle", "run.json"]
