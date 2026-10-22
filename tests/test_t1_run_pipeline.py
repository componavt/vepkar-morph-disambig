"""S4f.1 acceptance: one real prepare -> train -> predict -> evaluate pipeline.

Runs the four real T1 command handlers in sequence through ``cli.main`` over
a tiny synthetic corpus in a local temporary Git data repository.  Only the
corpus-table reader is substituted; every other production stage is real.
"""

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import benchmark_context  # noqa: E402
import cli  # noqa: E402
import commands.prepare_t1_run as prepare_pkg  # noqa: E402
from core.data import SUPPORTED_LANGUAGES, CorpusTables  # noqa: E402
from t1_features.model_io import load_model_bundle  # noqa: E402

_TAG = "v2026.09"
_GOLD = {202: (2020, "D1"), 302: (3020, "D2")}


def _git_available() -> bool:
    try:
        result = subprocess.run(
            ["git", "--version"], capture_output=True, text=True
        )
    except OSError:
        return False
    return result.returncode == 0


GIT_AVAILABLE = _git_available()


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=True,
    )


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
        texts=_make_texts([1, 2, 3, 4]),
        sentences=_make_sentences(
            [
                (10, 1, '<s id="10"/>', ""),
                (20, 2, '<s id="20"/>', ""),
                (30, 3, '<s id="30"/>', ""),
                (40, 4, '<s id="40"/>', ""),
            ]
        ),
        words=_make_words(
            [
                (101, 10, 1, "left"),
                (102, 10, 2, "trainkala"),
                (103, 10, 3, "right"),
                (201, 20, 1, "left"),
                (202, 20, 2, "devkala"),
                (203, 20, 3, "right"),
                (301, 30, 1, "left"),
                (302, 30, 2, "devkala2"),
                (303, 30, 3, "right"),
                (401, 40, 1, "left"),
                (402, 40, 2, "testkala"),
                (403, 40, 3, "right"),
            ]
        ),
        candidates=_make_candidates(
            [
                (102, 1020, "SG", 2),
                (102, 1021, "PL", 1),
                (202, 2020, "D1", 2),
                (202, 2021, "D1+ALT", 1),
                (302, 3020, "D2", 2),
                (302, 3021, "D2+ALT", 1),
                (402, 4020, "T1", 2),
                (402, 4021, "T1+ALT", 1),
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


def _snapshot_tables(tables):
    return {
        lang: {
            name: getattr(tables[lang], name).copy(deep=True)
            for name in ("texts", "sentences", "words", "candidates")
        }
        for lang in SUPPORTED_LANGUAGES
    }


def _assert_tables_unchanged(tables, snapshot):
    for lang in SUPPORTED_LANGUAGES:
        for name in ("texts", "sentences", "words", "candidates"):
            pd.testing.assert_frame_equal(
                getattr(tables[lang], name), snapshot[lang][name]
            )


def _write_split_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["language", "text_id", "split"])
        writer.writerows(rows)


def _data_repo(tmp_path):
    root = tmp_path / "dictorpus-data"
    (root / "corpus").mkdir(parents=True)
    (root / "corpus" / "placeholder.txt").write_text("fixture\n", encoding="utf-8")
    (root / "README.md").write_text("synthetic data repo\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    _git(root, "add", "-A")
    _git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.org",
        "commit",
        "-qm",
        "corpus placeholder",
    )
    _git(root, "tag", _TAG)
    return root


def _read_dev_csv(path):
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, strict=True)
        header = next(reader)
        raw = list(reader)
    rows = [
        (int(r[0]), int(r[1]), r[2], int(r[3]), float(r[4]))
        for r in raw
    ]
    return header, rows


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
class TestT1RunPipeline:
    def test_prepare_train_predict_evaluate_via_cli_main(
        self, tmp_path, monkeypatch, capsys
    ):
        project = tmp_path / "project"
        project.mkdir()
        monkeypatch.setattr(prepare_pkg, "_PROJECT_ROOT", project)

        data = _data_repo(tmp_path)
        split_file = tmp_path / "splits.csv"
        _write_split_csv(
            split_file,
            [
                ("krl", 1, "train"),
                ("krl", 2, "dev"),
                ("krl", 3, "dev"),
                ("krl", 4, "test"),
            ],
        )
        tables = _install_tables(monkeypatch)
        table_snapshot = _snapshot_tables(tables)

        run_dir = tmp_path / "runs" / "run-a"
        prepare_args = [
            "prepare-t1-run",
            "--run-dir",
            str(run_dir),
            "--C",
            "0.1",
            "--data-dir",
            str(data),
            "--split-file",
            str(split_file),
        ]

        status = cli.main(prepare_args)
        captured = capsys.readouterr()
        assert status == 0, captured.err
        assert sorted(p.name for p in run_dir.iterdir()) == ["run.json"]
        _assert_tables_unchanged(tables, table_snapshot)

        manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert manifest["format"] == "vepkar-t1-run"
        assert manifest["format_version"] == 4
        assert "code" not in manifest
        assert manifest["data"]["root"] == str(data.resolve())
        assert manifest["data"]["tag"] == _TAG
        assert (
            manifest["data"]["commit"]
            == _git(data, "rev-parse", "HEAD").stdout.strip()
        )
        assert manifest["split"]["path"] == str(split_file.resolve())
        assert manifest["split"]["sha256"] == hashlib.sha256(
            split_file.read_bytes()
        ).hexdigest()
        assert manifest["parameters"] == {"C": 0.1, "target_split": "dev"}

        passport_bytes = (run_dir / "run.json").read_bytes()
        split_bytes = split_file.read_bytes()

        status = cli.main(["train-t1", "--run-dir", str(run_dir)])
        captured = capsys.readouterr()
        assert status == 0, captured.err
        assert sorted(p.name for p in run_dir.iterdir()) == [
            "model.bundle",
            "run.json",
        ]
        assert (run_dir / "run.json").read_bytes() == passport_bytes
        assert split_file.read_bytes() == split_bytes
        _assert_tables_unchanged(tables, table_snapshot)

        bundle_bytes = (run_dir / "model.bundle").read_bytes()
        model = load_model_bundle(run_dir / "model.bundle")
        assert model.classifier.C == pytest.approx(0.1)

        status = cli.main(["predict-t1", "--run-dir", str(run_dir)])
        captured = capsys.readouterr()
        assert status == 0, captured.err
        assert sorted(p.name for p in run_dir.iterdir()) == [
            "dev.csv",
            "model.bundle",
            "run.json",
        ]
        assert (run_dir / "run.json").read_bytes() == passport_bytes
        assert split_file.read_bytes() == split_bytes
        assert (run_dir / "model.bundle").read_bytes() == bundle_bytes
        _assert_tables_unchanged(tables, table_snapshot)

        dev_path = run_dir / "dev.csv"
        dev_bytes = dev_path.read_bytes()
        header, prediction_rows = _read_dev_csv(dev_path)
        assert header == ["word_id", "wordform_id", "gramset", "rank", "score"]
        assert {row[0] for row in prediction_rows} == {202, 302}
        assert {(row[1], row[2]) for row in prediction_rows} == {
            (2020, "D1"),
            (2021, "D1+ALT"),
            (3020, "D2"),
            (3021, "D2+ALT"),
        }
        for word_id in (202, 302):
            assert sorted(
                row[3] for row in prediction_rows if row[0] == word_id
            ) == [1, 2]
        assert all(row[0] not in (102, 402) for row in prediction_rows)

        status = cli.main(["evaluate-t1-run", "--run-dir", str(run_dir)])
        captured = capsys.readouterr()
        assert status == 0, captured.err
        assert sorted(p.name for p in run_dir.iterdir()) == [
            "dev.csv",
            "metrics.json",
            "model.bundle",
            "run.json",
        ]
        assert (run_dir / "run.json").read_bytes() == passport_bytes
        assert split_file.read_bytes() == split_bytes
        assert (run_dir / "model.bundle").read_bytes() == bundle_bytes
        assert (run_dir / "dev.csv").read_bytes() == dev_bytes
        _assert_tables_unchanged(tables, table_snapshot)

        payload = json.loads(
            (run_dir / "metrics.json").read_text(encoding="utf-8")
        )
        assert payload["format"] == "vepkar-metrics"
        assert payload["format_version"] == 2
        assert payload["method"] == "t1.1"
        assert payload["parameters"] == {"C": 0.1}
        assert payload["split"] == "dev"
        assert payload["word_count"] == 2
        assert payload["candidate_count"] == 4
        assert payload["data"] == {
            "tag": _TAG,
            "commit": manifest["data"]["commit"],
            "split_sha256": manifest["split"]["sha256"],
        }
        assert payload["predictions_sha256"] == hashlib.sha256(dev_bytes).hexdigest()

        gold_ranks = []
        for word_id, identity in _GOLD.items():
            rank = next(
                row[3]
                for row in prediction_rows
                if row[0] == word_id and (row[1], row[2]) == identity
            )
            gold_ranks.append(rank)
        top1 = sum(rank == 1 for rank in gold_ranks)
        top3 = sum(rank <= 3 for rank in gold_ranks)
        mrr = sum(1.0 / rank for rank in gold_ranks) / len(gold_ranks)
        assert payload["top1_correct"] == top1
        assert payload["top3_correct"] == top3
        assert payload["metrics"]["top1_accuracy"] == pytest.approx(top1 / 2)
        assert payload["metrics"]["mrr"] == pytest.approx(mrr)
        assert payload["metrics"]["top3_accuracy"] == pytest.approx(top3 / 2)