"""Offline tests for the evaluate-t1-run CLI command."""

import csv
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import benchmark_context  # noqa: E402
import cli  # noqa: E402
import commands.evaluate_t1_run as evaluate_pkg  # noqa: E402
from core.data import SUPPORTED_LANGUAGES, CorpusTables, DataError  # noqa: E402
from core.metrics import MetricsInputError  # noqa: E402
from core.predictions import PredictionCsvParseError  # noqa: E402
from core.splits import SplitError  # noqa: E402
from t1_features.metrics_io import (  # noqa: E402
    T1MetricsError,
    T1MetricsPublishedCleanupError,
    write_t1_metrics,
)
from t1_features.run_context import T1RunContext, T1RunContextError  # noqa: E402

_TAG = "fixture-tag"
_COMMIT = "d" * 40
_SPLIT_SHA256 = "0" * 64


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


def _krl_corpus():
    texts = _make_texts([1, 2, 3])
    sentences = _make_sentences(
        [
            (10, 1, '<s id="10"/>', ""),
            (20, 2, '<s id="20"/>', ""),
            (21, 2, '<s id="21"/>', ""),
            (22, 2, '<s id="22"/>', ""),
            (30, 3, '<s id="30"/>', ""),
        ]
    )
    words = _make_words(
        [
            (101, 10, 1, "trainkala"),
            (201, 20, 1, "devkala"),
            (202, 21, 1, "devkala-alt"),
            (204, 22, 1, "devkala-fourth"),
            (301, 30, 1, "testkala"),
        ]
    )
    candidates = _make_candidates(
        [
            (101, 1010, "SG", 2),
            (101, 1011, "PL", 1),
            (201, 2010, "D1", 2),
            (201, 2011, "D1+ALT", 1),
            (202, 2020, "D2", 1),
            (202, 2021, "D2+ALT", 2),
            (204, 2040, "D4", 1),
            (204, 2041, "D4+ALT", 1),
            (204, 2042, "D4+THIRD", 1),
            (204, 2043, "D4+FOURTH", 2),
            (301, 3010, "T1", 2),
            (301, 3011, "T1+ALT", 1),
        ]
    )
    return CorpusTables(
        texts=texts,
        sentences=sentences,
        words=words,
        candidates=candidates,
    )


def _empty_corpus():
    return CorpusTables(
        texts=_make_texts([]),
        sentences=_make_sentences([]),
        words=_make_words([]),
        candidates=_make_candidates([]),
    )


def _dev_csv_bytes():
    rows = [
        (201, 2010, "D1", 1, 2.0),
        (201, 2011, "D1+ALT", 2, 1.0),
        (202, 2020, "D2", 1, 3.0),
        (202, 2021, "D2+ALT", 2, 2.0),
        (204, 2040, "D4", 1, 4.0),
        (204, 2041, "D4+ALT", 2, 3.0),
        (204, 2042, "D4+THIRD", 3, 2.0),
        (204, 2043, "D4+FOURTH", 4, 1.0),
    ]
    lines = ["word_id,wordform_id,gramset,rank,score"]
    lines.extend(
        f"{word_id},{wordform_id},{gramset},{rank},{score}"
        for word_id, wordform_id, gramset, rank, score in rows
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


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


def _write_split_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["language", "text_id", "split"])
        writer.writerows(rows)


def _written_dev_csv(run_dir):
    data = _dev_csv_bytes()
    (run_dir / "dev.csv").write_bytes(data)
    return data


def _write_passport(run_dir, *, split_path):
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "vepkar-t1-run",
        "format_version": 4,
        "stage": "prepared",
        "data": {
            "root": str(run_dir.parent / "checkout"),
            "tag": _TAG,
            "commit": _COMMIT,
        },
        "split": {"path": str(split_path), "sha256": _SPLIT_SHA256},
        "parameters": {"C": 0.1, "target_split": "dev"},
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

    monkeypatch.setattr(evaluate_pkg, "load_t1_run_context", fake_load)
    return recorded


def _success_env(tmp_path):
    split_file = tmp_path / "splits.csv"
    _write_split_csv(
        split_file,
        [("krl", 1, "train"), ("krl", 2, "dev"), ("krl", 3, "test")],
    )
    run_dir = tmp_path / "runs" / "run-a"
    manifest = _write_passport(run_dir, split_path=split_file)
    return split_file, run_dir, manifest


def _forbid_stage_call(monkeypatch, name):
    def forbidden(*args, **kwargs):
        pytest.fail(f"stage must not be called: {name}")

    monkeypatch.setattr(evaluate_pkg, name, forbidden)


class TestArgumentParsing:
    def test_help_lists_arguments(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["evaluate-t1-run", "--help"])
        captured = capsys.readouterr()
        assert exc.value.code == 0
        assert "--run-dir" in captured.out
        for removed in (
            "--predictions",
            "--split",
            "--output",
            "--data-dir",
            "--split-file",
            "--C",
            "--strict-code",
            "--force",
            "--resume",
            "--overwrite",
        ):
            assert removed not in captured.out

    def test_requires_run_dir(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["evaluate-t1-run"])
        captured = capsys.readouterr()
        assert exc.value.code == 2
        assert "required" in captured.err
        assert "--run-dir" in captured.err

    @pytest.mark.parametrize(
        "removed",
        [
            ["--split", "dev"],
            ["--force", "x"],
        ],
    )
    def test_rejects_removed_and_unsupported_options(self, tmp_path, capsys, removed):
        with pytest.raises(SystemExit) as exc:
            cli.main(["evaluate-t1-run", "--run-dir", str(tmp_path), *removed])
        captured = capsys.readouterr()
        assert exc.value.code == 2
        assert "unrecognized arguments" in captured.err

    @pytest.mark.parametrize("abbreviation", ["--run"])
    def test_rejects_abbreviated_options(self, tmp_path, capsys, abbreviation):
        with pytest.raises(SystemExit) as exc:
            cli.main(["evaluate-t1-run", "--run-dir", str(tmp_path), abbreviation])
        captured = capsys.readouterr()
        assert exc.value.code == 2
        assert "unrecognized arguments" in captured.err


class TestContextFailure:
    def test_context_error_reported_without_pipeline_access(
        self, tmp_path, monkeypatch, capsys
    ):
        _forbid_corpus(monkeypatch)
        _forbid_stage_call(monkeypatch, "load_benchmark_context")
        _forbid_stage_call(monkeypatch, "write_t1_metrics")
        run_dir = tmp_path / "runs" / "run-a"

        def fail_load(run_dir_arg):
            raise T1RunContextError(
                f"cannot load T1 run passport {run_dir_arg / 'run.json'}: missing"
            )

        monkeypatch.setattr(evaluate_pkg, "load_t1_run_context", fail_load)
        status = cli.main(["evaluate-t1-run", "--run-dir", str(run_dir)])
        captured = capsys.readouterr()
        assert status == 1
        assert "cannot load T1 run passport" in captured.err
        assert str(run_dir / "run.json") in captured.err
        assert not (run_dir / "metrics.json").exists()
        assert "Prediction evaluation: OK" not in captured.out


class TestCorpusDataError:
    def test_data_error_reported_once_without_validation_or_publication(
        self, tmp_path, monkeypatch, capsys
    ):
        split_file, run_dir, manifest = _success_env(tmp_path)
        _written_dev_csv(run_dir)
        _stub_run_context(monkeypatch, run_dir, manifest)
        for name in (
            "validate_predictions",
            "load_ranked_predictions",
            "compute_ranking_metrics",
            "build_t1_metrics",
            "write_t1_metrics",
        ):
            _forbid_stage_call(monkeypatch, name)

        def fail_tables(lang, data_dir):
            raise DataError("distinctive: corpus tables unavailable")

        monkeypatch.setattr(benchmark_context, "read_corpus_tables", fail_tables)
        status = cli.main(["evaluate-t1-run", "--run-dir", str(run_dir)])
        captured = capsys.readouterr()
        assert status == 1
        assert captured.err.count("distinctive: corpus tables unavailable") == 1
        assert "Traceback" not in captured.err
        assert not (run_dir / "metrics.json").exists()
        assert "Prediction evaluation: OK" not in captured.out


class TestDestinationProtection:
    @pytest.mark.parametrize("kind", ["file", "directory", "dangling-symlink"])
    def test_existing_metrics_refused_before_corpus_loading(
        self, tmp_path, monkeypatch, capsys, kind
    ):
        _forbid_corpus(monkeypatch)
        split_file, run_dir, manifest = _success_env(tmp_path)
        _stub_run_context(monkeypatch, run_dir, manifest)
        metrics = run_dir / "metrics.json"
        if kind == "file":
            metrics.write_text("precious\n", encoding="utf-8")
        elif kind == "directory":
            metrics.mkdir()
            (metrics / "keep.txt").write_text("keep", encoding="utf-8")
        else:
            try:
                metrics.symlink_to(tmp_path / "missing-target")
            except (OSError, NotImplementedError):
                pytest.skip("Symlink creation is unavailable in this environment")
        before = _recursive_entries(tmp_path)
        status = cli.main(["evaluate-t1-run", "--run-dir", str(run_dir)])
        captured = capsys.readouterr()
        assert status == 1
        if kind == "directory":
            assert "is a directory" in captured.err
        else:
            assert "already exists" in captured.err
        assert str(metrics) in captured.err
        assert _recursive_entries(tmp_path) == before
        if kind == "file":
            assert metrics.read_text(encoding="utf-8") == "precious\n"
        elif kind == "directory":
            assert (metrics / "keep.txt").read_text(encoding="utf-8") == "keep"
        else:
            assert metrics.is_symlink()
            assert metrics.readlink() == tmp_path / "missing-target"
            assert not (tmp_path / "missing-target").exists()
        assert "Prediction evaluation: OK" not in captured.out
        assert not list(run_dir.glob(".vepkar-t1-metrics-*.tmp"))


def _recursive_entries(root):
    return {path.relative_to(root).as_posix() for path in root.rglob("*")}


def _read_metrics(run_dir):
    return json.loads(
        (run_dir / "metrics.json").read_text(encoding="utf-8")
    )


class TestSuccessfulEvaluation:
    def test_real_offline_pipeline_via_cli_main(
        self, tmp_path, monkeypatch, capsys
    ):
        tables = _install_tables(monkeypatch)
        split_file, run_dir, manifest = _success_env(tmp_path)
        dev_bytes = _written_dev_csv(run_dir)
        recorded = _stub_run_context(monkeypatch, run_dir, manifest)

        passport_before = (run_dir / "run.json").read_bytes()
        split_before = split_file.read_bytes()
        entries_before = sorted(p.name for p in run_dir.iterdir())
        tables_before = {
            lang: {
                table_name: getattr(tables[lang], table_name).copy(deep=True)
                for table_name in ("texts", "sentences", "words", "candidates")
            }
            for lang in SUPPORTED_LANGUAGES
        }

        status = cli.main(["evaluate-t1-run", "--run-dir", str(run_dir)])
        captured = capsys.readouterr()

        assert status == 0, captured.err
        assert captured.err == ""
        assert recorded["calls"] == 1
        assert recorded["run_dir"] == run_dir
        assert "Prediction evaluation: OK" in captured.out
        assert f"Split: dev" in captured.out
        assert (
            "Evaluated: 3 word instances, 8 candidate rows" in captured.out
        )
        assert "Top-1 accuracy     0.3333" in captured.out
        assert "MRR                0.5833" in captured.out
        assert "Top-3 accuracy     0.6667" in captured.out
        assert f"Run directory: {run_dir.resolve()}" in captured.out
        assert f"Metrics: {run_dir / 'metrics.json'}" in captured.out

        payload = _read_metrics(run_dir)
        assert payload["format"] == "vepkar-metrics"
        assert payload["format_version"] == 2
        assert payload["method"] == "t1.1"
        assert payload["parameters"] == {"C": 0.1}
        assert payload["split"] == "dev"
        assert payload["word_count"] == 3
        assert payload["candidate_count"] == 8
        assert payload["top1_correct"] == 1
        assert payload["top3_correct"] == 2
        assert payload["metrics"]["top1_accuracy"] == pytest.approx(1 / 3)
        assert payload["metrics"]["mrr"] == pytest.approx(7 / 12)
        assert payload["metrics"]["top3_accuracy"] == pytest.approx(2 / 3)
        assert payload["metrics"]["mrr"] != round(payload["metrics"]["mrr"], 4)
        assert payload["data"] == {
            "tag": _TAG,
            "commit": _COMMIT,
            "split_sha256": _SPLIT_SHA256,
        }
        assert payload["predictions_sha256"] == hashlib.sha256(dev_bytes).hexdigest()

        assert (run_dir / "run.json").read_bytes() == passport_before
        assert split_file.read_bytes() == split_before
        assert (run_dir / "dev.csv").read_bytes() == dev_bytes
        assert not (run_dir / "model.bundle").exists()
        assert sorted(p.name for p in run_dir.iterdir()) == sorted(
            entries_before + ["metrics.json"]
        )
        for lang in SUPPORTED_LANGUAGES:
            for table_name in ("texts", "sentences", "words", "candidates"):
                pd.testing.assert_frame_equal(
                    getattr(tables[lang], table_name),
                    tables_before[lang][table_name],
                )
        assert not list(run_dir.glob(".vepkar-t1-metrics-*.tmp"))


class TestInvalidPredictions:
    @pytest.mark.parametrize("kind", ["missing", "empty", "incomplete"])
    def test_no_metrics_publication_or_success(
        self, tmp_path, monkeypatch, capsys, kind
    ):
        _install_tables(monkeypatch)
        split_file, run_dir, manifest = _success_env(tmp_path)
        if kind != "missing":
            (run_dir / "dev.csv").write_bytes(_dev_csv_bytes())
        if kind == "empty":
            (run_dir / "dev.csv").write_text("", encoding="utf-8")
        elif kind == "incomplete":
            (run_dir / "dev.csv").write_text(
                "word_id,wordform_id,gramset,rank,score\n"
                "201,2010,D1,1,2.0\n"
                "201,2011,D1+ALT,2,1.0\n"
                "202,2020,D2,1,3.0\n"
                "202,2021,D2+ALT,2,2.0\n"
                "204,2040,D4,1,4.0\n",
                encoding="utf-8",
            )
        _stub_run_context(monkeypatch, run_dir, manifest)
        status = cli.main(["evaluate-t1-run", "--run-dir", str(run_dir)])
        captured = capsys.readouterr()
        assert status == 1
        if kind == "missing":
            assert "cannot read predictions file" in captured.err
            assert str(run_dir / "dev.csv") in captured.err
        else:
            assert "Predictions validation: FAILED" in captured.out
        assert "Prediction evaluation: OK" not in captured.out
        assert "Top-1 accuracy" not in captured.out
        assert not (run_dir / "metrics.json").exists()
        assert not list(run_dir.glob(".vepkar-t1-metrics-*.tmp"))


class TestStageFailures:
    @pytest.mark.parametrize(
        "case",
        [
            {
                "target": "load_benchmark_context",
                "exc_type": SplitError,
                "exc_message": "cannot read split file: /missing-splits.csv",
                "diag": ("error: cannot load the strict benchmark:",),
                "forbid_later": (
                    "validate_predictions",
                    "load_ranked_predictions",
                    "compute_ranking_metrics",
                    "build_t1_metrics",
                    "write_t1_metrics",
                ),
            },
            {
                "target": "load_ranked_predictions",
                "exc_type": PredictionCsvParseError,
                "exc_message": "unexpected header during reread",
                "diag": (
                    "error: cannot load predictions for evaluation:",
                    "unexpected header during reread",
                ),
                "forbid_later": (
                    "compute_ranking_metrics",
                    "build_t1_metrics",
                    "write_t1_metrics",
                ),
            },
            {
                "target": "compute_ranking_metrics",
                "exc_type": MetricsInputError,
                "exc_message": "the evaluation set is empty",
                "diag": ("error: cannot compute ranking metrics",),
                "forbid_later": ("build_t1_metrics", "write_t1_metrics"),
            },
            {
                "target": "build_t1_metrics",
                "exc_type": T1MetricsError,
                "exc_message": "cannot read T1 dev predictions dev.csv",
                "diag": ("error: cannot read T1 dev predictions",),
                "forbid_later": ("write_t1_metrics",),
            },
        ],
        ids=lambda case: f"{case['target']}-{case['exc_type'].__name__}",
    )
    def test_stage_exception_reported(self, tmp_path, monkeypatch, capsys, case):
        _install_tables(monkeypatch)
        split_file, run_dir, manifest = _success_env(tmp_path)
        _written_dev_csv(run_dir)
        _stub_run_context(monkeypatch, run_dir, manifest)

        for name in case["forbid_later"]:
            _forbid_stage_call(monkeypatch, name)

        def raise_exc(*args, **kwargs):
            raise case["exc_type"](case["exc_message"])

        monkeypatch.setattr(evaluate_pkg, case["target"], raise_exc)
        status = cli.main(["evaluate-t1-run", "--run-dir", str(run_dir)])
        captured = capsys.readouterr()
        assert status == 1
        for fragment in case["diag"]:
            assert fragment in captured.err
        assert case["exc_message"] in captured.err
        assert "Traceback" not in captured.err
        assert "Prediction evaluation: OK" not in captured.out
        assert not (run_dir / "metrics.json").exists()
        assert not list(run_dir.glob(".vepkar-t1-metrics-*.tmp"))


class TestPublishedCleanupFailure:
    def test_metrics_json_reported_when_cleanup_fails(
        self, tmp_path, monkeypatch, capsys
    ):
        _install_tables(monkeypatch)
        split_file, run_dir, manifest = _success_env(tmp_path)
        _written_dev_csv(run_dir)
        _stub_run_context(monkeypatch, run_dir, manifest)
        metrics_path = run_dir / "metrics.json"

        def read_and_raise(payload, output_path):
            write_t1_metrics(payload, output_path)
            temp = run_dir / ".vepkar-t1-metrics-leftover.tmp"
            temp.write_text("leftover", encoding="utf-8")
            raise T1MetricsPublishedCleanupError(
                output_path, temp
            ) from PermissionError("cleanup denied")

        monkeypatch.setattr(evaluate_pkg, "write_t1_metrics", read_and_raise)
        status = cli.main(["evaluate-t1-run", "--run-dir", str(run_dir)])
        captured = capsys.readouterr()
        temp_path = run_dir / ".vepkar-t1-metrics-leftover.tmp"

        assert status == 1
        assert "metrics.json was published" in captured.err
        assert "temporary cleanup failed" in captured.err
        assert str(metrics_path) in captured.err
        assert str(temp_path) in captured.err
        assert "cleanup denied" in captured.err
        assert "Prediction evaluation: OK" not in captured.out
        payload = _read_metrics(run_dir)
        assert payload["word_count"] == 3
        assert payload["candidate_count"] == 8
        assert temp_path.read_text(encoding="utf-8") == "leftover"
        temp_path.unlink()
