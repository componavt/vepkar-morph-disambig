"""Offline tests for the T1 comparison file-input layer."""

import csv
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import zstandard

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import benchmark_context  # noqa: E402
from core.data import SUPPORTED_LANGUAGES, CorpusTables  # noqa: E402
from core.metrics import MetricsInputError  # noqa: E402
from t1_features import comparison_inputs  # noqa: E402
from t1_features.comparison_inputs import (  # noqa: E402
    T1ComparisonInputError,
    load_t1_comparison,
)

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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["language", "text_id", "split"])
        writer.writerows(rows)


def _csv_bytes(rows):
    lines = ["word_id,wordform_id,gramset,rank,score"]
    lines.extend(
        f"{word_id},{wordform_id},{gramset},{rank},{score}"
        for word_id, wordform_id, gramset, rank, score in rows
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


INPUT1_ROWS = [
    (201, 2010, "D1", 1, 2.0),
    (201, 2011, "D1+ALT", 2, 1.0),
    (202, 2020, "D2", 1, 3.0),
    (202, 2021, "D2+ALT", 2, 2.0),
    (204, 2040, "D4", 1, 4.0),
    (204, 2041, "D4+ALT", 2, 3.0),
    (204, 2042, "D4+THIRD", 3, 2.0),
    (204, 2043, "D4+FOURTH", 4, 1.0),
]

INPUT2_ROWS = [
    (201, 2010, "D1", 1, 2.0),
    (201, 2011, "D1+ALT", 2, 1.0),
    (202, 2021, "D2+ALT", 1, 2.0),
    (202, 2020, "D2", 2, 1.0),
    (204, 2043, "D4+FOURTH", 1, 4.0),
    (204, 2040, "D4", 2, 3.0),
    (204, 2041, "D4+ALT", 3, 2.0),
    (204, 2042, "D4+THIRD", 4, 1.0),
]

INPUT1_BYTES = _csv_bytes(INPUT1_ROWS)
INPUT2_BYTES = _csv_bytes(INPUT2_ROWS)

_INCOMPLETE_ROWS = [
    (201, 2010, "D1", 1, 2.0),
    (201, 2011, "D1+ALT", 2, 1.0),
    (202, 2020, "D2", 1, 3.0),
    (202, 2021, "D2+ALT", 2, 2.0),
    (204, 2040, "D4", 1, 4.0),
]
_INCOMPLETE_BYTES = _csv_bytes(_INCOMPLETE_ROWS)


def _zstd_bytes(data):
    return zstandard.ZstdCompressor().compress(data)


def _make_package(
    run_dir,
    csv_bytes,
    *,
    compressed=False,
    C=0.1,
    tag=_TAG,
    commit=_COMMIT,
    split_sha256=_SPLIT_SHA256,
    predictions_sha256="f" * 64,
    payload_extra=None,
):
    run_dir.mkdir(parents=True, exist_ok=True)
    if compressed:
        (run_dir / "dev.csv.zst").write_bytes(_zstd_bytes(csv_bytes))
    else:
        (run_dir / "dev.csv").write_bytes(csv_bytes)
    payload = {
        "format": "vepkar-metrics",
        "format_version": 2,
        "method": "t1.1",
        "parameters": {"C": C},
        "split": "dev",
        "word_count": 3,
        "candidate_count": 8,
        "top1_correct": 1,
        "top3_correct": 2,
        "metrics": {"top1_accuracy": 1 / 3, "mrr": 7 / 12, "top3_accuracy": 2 / 3},
        "data": {"tag": tag, "commit": commit, "split_sha256": split_sha256},
        "predictions_sha256": predictions_sha256,
    }
    if payload_extra:
        payload.update(payload_extra)
    (run_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    return run_dir


def _setup_repo(tmp_path, *, tag=_TAG):
    repo_root = tmp_path / "repo"
    split_path = repo_root / "data" / "derived" / f"splits_{tag}.csv"
    _write_split_csv(
        split_path,
        [("krl", 1, "train"), ("krl", 2, "dev"), ("krl", 3, "test")],
    )
    return repo_root, split_path


def _stub_preflight(monkeypatch, data_dir):
    recorded = {}

    def fake(data_dir_override):
        recorded["data_dir"] = data_dir_override
        return data_dir_override, _TAG

    monkeypatch.setattr(comparison_inputs, "_preflight_benchmark", fake)
    return recorded


def _recursive_entries(root):
    return {path.relative_to(root).as_posix() for path in root.rglob("*")}


def _track_temporary_directory(monkeypatch, tmp_path):
    real = comparison_inputs.tempfile.TemporaryDirectory
    owned = tmp_path / "owned-temp"
    owned.mkdir(parents=True)
    recorded = {}

    def wrapped(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["dir"] = str(owned)
        td = real(*args, **kwargs)
        recorded["path"] = Path(td.name)
        return td

    monkeypatch.setattr(comparison_inputs.tempfile, "TemporaryDirectory", wrapped)
    return recorded


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


class TestSuccess:
    def test_two_plain_csv_packages_full_integration(
        self, tmp_path, monkeypatch
    ):
        tables = _install_tables(monkeypatch)
        repo_root, _ = _setup_repo(tmp_path)
        _stub_preflight(
            monkeypatch, repo_root / "data" / "dictorpus-data"
        )

        run_a = _make_package(tmp_path / "runs" / "run-a", INPUT1_BYTES, C=0.1)
        run_b = _make_package(
            tmp_path / "runs" / "run-b", INPUT2_BYTES, C=10.0
        )

        snapshots = {
            "a": {
                "dev": (run_a / "dev.csv").read_bytes(),
                "metrics": (run_a / "metrics.json").read_bytes(),
            },
            "b": {
                "dev": (run_b / "dev.csv").read_bytes(),
                "metrics": (run_b / "metrics.json").read_bytes(),
            },
        }
        split_before = (repo_root / "data" / "derived" / f"splits_{_TAG}.csv").read_bytes()
        tables_before = _snapshot_tables(tables)
        entries_a_before = _recursive_entries(run_a)
        entries_b_before = _recursive_entries(run_b)

        real_validate = comparison_inputs.validate_predictions
        validate_calls = []

        def spy_validate(path, split, instances, split_rows):
            validate_calls.append((path, split, instances, split_rows))
            return real_validate(path, split, instances, split_rows)

        monkeypatch.setattr(comparison_inputs, "validate_predictions", spy_validate)

        real_compare = comparison_inputs.compare_rankings
        compare_calls = {"n": 0}

        def spy_compare(*args, **kwargs):
            compare_calls["n"] += 1
            return real_compare(*args, **kwargs)

        monkeypatch.setattr(comparison_inputs, "compare_rankings", spy_compare)

        loaded = load_t1_comparison(run_a, run_b, repo_root=repo_root)

        assert len(validate_calls) == 2
        assert validate_calls[0][1] == "dev"
        assert validate_calls[1][1] == "dev"
        assert validate_calls[0][2] is validate_calls[1][2]
        assert validate_calls[0][3] is validate_calls[1][3]
        assert compare_calls["n"] == 1

        assert loaded.source_1.run_dir == run_a.resolve()
        assert loaded.source_1.predictions_path == run_a / "dev.csv"
        assert loaded.source_1.method == "t1.1"
        assert loaded.source_1.C == 0.1
        assert loaded.source_1.data_tag == _TAG
        assert loaded.source_1.data_commit == _COMMIT
        assert loaded.source_1.split_sha256 == _SPLIT_SHA256
        assert loaded.source_2.run_dir == run_b.resolve()
        assert loaded.source_2.predictions_path == run_b / "dev.csv"
        assert loaded.source_2.C == 10.0
        assert loaded.source_2.data_tag == _TAG

        comparison = loaded.comparison
        assert comparison.a_input_index == 1
        assert comparison.b_input_index == 2
        assert comparison.selection_reason == "top1"
        assert comparison.overall.word_count == 3
        assert comparison.overall.candidate_count == 8
        assert comparison.overall.both_correct == 1
        assert comparison.overall.corrected == 2
        assert comparison.overall.regressed == 0
        assert comparison.overall.delta_top1 == pytest.approx(2 / 3)

        assert (run_a / "dev.csv").read_bytes() == snapshots["a"]["dev"]
        assert (run_a / "metrics.json").read_bytes() == snapshots["a"]["metrics"]
        assert (run_b / "dev.csv").read_bytes() == snapshots["b"]["dev"]
        assert (run_b / "metrics.json").read_bytes() == snapshots["b"]["metrics"]
        assert (
            (repo_root / "data" / "derived" / f"splits_{_TAG}.csv").read_bytes()
            == split_before
        )
        assert _recursive_entries(run_a) == entries_a_before
        assert _recursive_entries(run_b) == entries_b_before
        _assert_tables_unchanged(tables, tables_before)

    @pytest.mark.parametrize(
        "spec",
        [
            {"a": "plain", "b": "compressed"},
            {"a": "compressed", "b": "plain"},
            {"a": "compressed", "b": "compressed"},
        ],
        ids=["a-plain-b-zst", "a-zst-b-plain", "a-zst-b-zst"],
    )
    def test_compressed_inputs(self, tmp_path, monkeypatch, spec):
        _install_tables(monkeypatch)
        repo_root, _ = _setup_repo(tmp_path)
        _stub_preflight(monkeypatch, repo_root / "data" / "dictorpus-data")

        a_bytes = INPUT1_BYTES if spec["a"] == "plain" else INPUT2_BYTES
        b_bytes = INPUT2_BYTES if spec["a"] == "plain" else INPUT1_BYTES
        run_a = _make_package(
            tmp_path / "runs" / "run-a", a_bytes,
            compressed=(spec["a"] == "compressed"),
        )
        run_b = _make_package(
            tmp_path / "runs" / "run-b", b_bytes,
            compressed=(spec["b"] == "compressed"),
        )
        entries_a_before = _recursive_entries(run_a)
        entries_b_before = _recursive_entries(run_b)
        expected_a = run_a / ("dev.csv.zst" if spec["a"] == "compressed" else "dev.csv")
        expected_b = run_b / ("dev.csv.zst" if spec["b"] == "compressed" else "dev.csv")
        tracked = _track_temporary_directory(monkeypatch, tmp_path)

        loaded = load_t1_comparison(run_a, run_b, repo_root=repo_root)

        assert loaded.source_1.predictions_path == expected_a
        assert loaded.source_2.predictions_path == expected_b
        assert loaded.comparison.overall.word_count == 3
        assert _recursive_entries(run_a) == entries_a_before
        assert _recursive_entries(run_b) == entries_b_before
        assert "path" in tracked
        assert not tracked["path"].exists()


class TestConfig:
    def test_configured_data_dir_honored_custom_split_ignored(
        self, tmp_path, monkeypatch
    ):
        _install_tables(monkeypatch)
        repo_root = tmp_path / "repo"
        configured_data_dir = tmp_path / "elsewhere" / "checkout"
        configured_data_dir.mkdir(parents=True)
        (repo_root / "data" / "derived").mkdir(parents=True)
        (repo_root / "t1.local.toml").write_text(
            "  [paths]\n"
            f"  data_dir = {str(configured_data_dir)!r}\n"
            f"  split_file = {str(tmp_path / 'custom-splits.csv')!r}\n",
            encoding="utf-8",
        )
        (tmp_path / "custom-splits.csv").write_text(
            "language,text_id,split\nkrl,1,train\nkrl,2,train\nkrl,3,train\n",
            encoding="utf-8",
        )
        split_path = repo_root / "data" / "derived" / f"splits_{_TAG}.csv"
        _write_split_csv(
            split_path,
            [("krl", 1, "train"), ("krl", 2, "dev"), ("krl", 3, "test")],
        )
        recorded = _stub_preflight(monkeypatch, configured_data_dir)

        run_a = _make_package(tmp_path / "runs" / "run-a", INPUT1_BYTES)
        run_b = _make_package(tmp_path / "runs" / "run-b", INPUT2_BYTES)

        loaded = load_t1_comparison(run_a, run_b, repo_root=repo_root)

        assert recorded["data_dir"] == configured_data_dir
        assert loaded.comparison.overall.word_count == 3


class TestFailures:
    def test_ambiguous_package_and_missing_input(self, tmp_path, monkeypatch):
        _forbid_corpus(monkeypatch)
        run = tmp_path / "runs" / "run-a"
        run.mkdir(parents=True)
        (run / "dev.csv").write_bytes(INPUT1_BYTES)
        (run / "dev.csv.zst").write_bytes(_zstd_bytes(INPUT1_BYTES))
        (run / "metrics.json").write_text("{}", encoding="utf-8")
        run_b = tmp_path / "runs" / "run-b"
        _make_package(run_b, INPUT2_BYTES)

        with pytest.raises(T1ComparisonInputError, match="ambiguous"):
            load_t1_comparison(run, run_b, repo_root=tmp_path / "repo")

        run2 = tmp_path / "runs" / "missing"
        run2.mkdir(parents=True)
        (run2 / "metrics.json").write_text("{}", encoding="utf-8")
        with pytest.raises(T1ComparisonInputError, match="neither dev.csv"):
            load_t1_comparison(run2, run_b, repo_root=tmp_path / "repo")

        run3 = tmp_path / "runs" / "no-metrics"
        run3.mkdir(parents=True)
        (run3 / "dev.csv").write_bytes(INPUT1_BYTES)
        with pytest.raises(T1ComparisonInputError, match="missing metrics.json"):
            load_t1_comparison(run3, run_b, repo_root=tmp_path / "repo")

        with pytest.raises(T1ComparisonInputError, match="does not exist"):
            load_t1_comparison(
                tmp_path / "runs" / "gone", run_b, repo_root=tmp_path / "repo"
            )

    @pytest.mark.parametrize(
        "mutate, match",
        [
            ({"format": "nope"}, "unsupported T1 metrics format"),
            ({"format_version": True}, "unsupported T1 metrics format_version"),
            ({"parameters": {"C": "high"}}, "parameters.C is not a number"),
            ({"data": {"tag": "a/b", "commit": _COMMIT, "split_sha256": _SPLIT_SHA256}}, "safe filename"),
            ({"data": {"tag": "a\x00b", "commit": _COMMIT, "split_sha256": _SPLIT_SHA256}}, "safe filename"),
            ({"data": {"tag": _TAG, "commit": "", "split_sha256": _SPLIT_SHA256}}, "data.commit must be a nonempty"),
        ],
        ids=["format", "format_version-bool", "bad-C", "bad-tag", "null-tag", "empty-commit"],
    )
    def test_malformed_metadata(self, tmp_path, monkeypatch, mutate, match):
        _forbid_corpus(monkeypatch)
        run_a = _make_package(tmp_path / "runs" / "run-a", INPUT1_BYTES)
        run_b = _make_package(
            tmp_path / "runs" / "run-b", INPUT2_BYTES, payload_extra=mutate
        )

        with pytest.raises(T1ComparisonInputError, match=match):
            load_t1_comparison(run_a, run_b, repo_root=tmp_path / "repo")

    def test_mismatched_identities_do_not_load_corpus(self, tmp_path, monkeypatch):
        _forbid_corpus(monkeypatch)
        run_a = _make_package(
            tmp_path / "runs" / "run-a", INPUT1_BYTES, commit=_COMMIT
        )
        run_b = _make_package(
            tmp_path / "runs" / "run-b", INPUT2_BYTES, commit="e" * 40
        )

        with pytest.raises(T1ComparisonInputError, match="mismatched"):
            load_t1_comparison(run_a, run_b, repo_root=tmp_path / "repo")


class TestInvalidPredictions:
    @pytest.mark.parametrize("compressed", [False, True])
    def test_invalid_rejected_and_comparison_not_called(
        self, tmp_path, monkeypatch, compressed
    ):
        _install_tables(monkeypatch)
        repo_root, _ = _setup_repo(tmp_path)
        _stub_preflight(monkeypatch, repo_root / "data" / "dictorpus-data")

        def forbidden_compare(*args, **kwargs):
            pytest.fail("comparison core must not be called on invalid input")

        monkeypatch.setattr(comparison_inputs, "compare_rankings", forbidden_compare)

        run_a = _make_package(
            tmp_path / "runs" / "run-a", _INCOMPLETE_BYTES, compressed=compressed
        )
        run_b = _make_package(tmp_path / "runs" / "run-b", INPUT2_BYTES)
        entries_a_before = _recursive_entries(run_a)
        tracked = _track_temporary_directory(monkeypatch, tmp_path)

        with pytest.raises(T1ComparisonInputError) as excinfo:
            load_t1_comparison(run_a, run_b, repo_root=repo_root)

        assert excinfo.value.validation is not None
        assert excinfo.value.validation.is_valid is False
        assert excinfo.value.validation.predicted_candidate_count < excinfo.value.validation.expected_candidate_count
        assert _recursive_entries(run_a) == entries_a_before
        assert "path" in tracked
        assert not tracked["path"].exists()

    def test_corrupt_zst_original_path_error(self, tmp_path, monkeypatch):
        _install_tables(monkeypatch)
        repo_root, _ = _setup_repo(tmp_path)
        _stub_preflight(monkeypatch, repo_root / "data" / "dictorpus-data")

        run_a = tmp_path / "runs" / "run-a"
        run_a.mkdir(parents=True)
        (run_a / "dev.csv.zst").write_bytes(b"not a zstd stream")
        (run_a / "metrics.json").write_text(
            json.dumps(
                {
                    "format": "vepkar-metrics",
                    "format_version": 2,
                    "method": "t1.1",
                    "parameters": {"C": 0.1},
                    "split": "dev",
                    "data": {
                        "tag": _TAG,
                        "commit": _COMMIT,
                        "split_sha256": _SPLIT_SHA256,
                    },
                }
            ),
            encoding="utf-8",
        )
        run_b = _make_package(tmp_path / "runs" / "run-b", INPUT2_BYTES)
        entries_a_before = _recursive_entries(run_a)
        tracked = _track_temporary_directory(monkeypatch, tmp_path)

        with pytest.raises(T1ComparisonInputError) as excinfo:
            load_t1_comparison(run_a, run_b, repo_root=repo_root)

        message = str(excinfo.value)
        assert "dev.csv.zst" in message
        assert "dev_input" not in message
        assert _recursive_entries(run_a) == entries_a_before
        assert "path" in tracked
        assert not tracked["path"].exists()


class TestNoFingerprint:
    def test_stored_metrics_disagree_but_core_orientation_wins(
        self, tmp_path, monkeypatch
    ):
        _install_tables(monkeypatch)
        repo_root, _ = _setup_repo(tmp_path)
        _stub_preflight(monkeypatch, repo_root / "data" / "dictorpus-data")

        run_a = _make_package(
            tmp_path / "runs" / "run-a",
            INPUT1_BYTES,
            predictions_sha256="f" * 64,
            split_sha256="1" * 64,
            payload_extra={
                "metrics": {"top1_accuracy": 1.0, "mrr": 1.0, "top3_accuracy": 1.0},
                "top1_correct": 3,
                "top3_correct": 3,
            },
        )
        run_b = _make_package(
            tmp_path / "runs" / "run-b",
            INPUT2_BYTES,
            predictions_sha256="0" * 64,
            split_sha256="1" * 64,
        )

        loaded = load_t1_comparison(run_a, run_b, repo_root=repo_root)

        assert loaded.comparison.a_input_index == 1
        assert loaded.comparison.b_input_index == 2
        assert loaded.comparison.selection_reason == "top1"
        assert loaded.comparison.overall.delta_top1 == pytest.approx(2 / 3)
        assert loaded.source_1.split_sha256 == "1" * 64
        assert loaded.source_2.split_sha256 == "1" * 64


class TestLifecycleAndBenchmark:
    @pytest.mark.parametrize("mode", ["creation", "cleanup"], ids=["creation", "cleanup"])
    def test_temporary_workspace_oserror_wrapped(
        self, tmp_path, monkeypatch, mode
    ):
        _install_tables(monkeypatch)
        repo_root, _ = _setup_repo(tmp_path)
        _stub_preflight(monkeypatch, repo_root / "data" / "dictorpus-data")
        run_a = _make_package(tmp_path / "runs" / "run-a", INPUT1_BYTES)
        run_b = _make_package(tmp_path / "runs" / "run-b", INPUT2_BYTES)
        owned = tmp_path / "owned"
        owned.mkdir()

        real = comparison_inputs.tempfile.TemporaryDirectory
        if mode == "creation":

            def fail_creation(*args, **kwargs):
                raise OSError("temporary creation denied")

            monkeypatch.setattr(
                comparison_inputs.tempfile, "TemporaryDirectory", fail_creation
            )
        else:
            def wrapped(*args, **kwargs):
                kwargs = dict(kwargs)
                kwargs["dir"] = str(owned)
                td = real(*args, **kwargs)

                class FailingTemporaryDirectory:
                    def __enter__(self):
                        return td.name

                    def __exit__(self, etype, value, tb):
                        td.cleanup()
                        raise OSError("temporary cleanup denied")

                return FailingTemporaryDirectory()

            monkeypatch.setattr(
                comparison_inputs.tempfile, "TemporaryDirectory", wrapped
            )

        with pytest.raises(T1ComparisonInputError) as excinfo:
            load_t1_comparison(run_a, run_b, repo_root=repo_root)

        assert "temporary workspace" in str(excinfo.value)
        assert str(run_a / "dev.csv") in str(excinfo.value)
        assert str(run_b / "dev.csv") in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, OSError)

    def test_cleanup_failure_preserves_primary_validation_error(
        self, tmp_path, monkeypatch
    ):
        _install_tables(monkeypatch)
        repo_root, _ = _setup_repo(tmp_path)
        _stub_preflight(monkeypatch, repo_root / "data" / "dictorpus-data")

        def forbidden_compare(*args, **kwargs):
            pytest.fail("comparison core must not be called on invalid input")

        monkeypatch.setattr(comparison_inputs, "compare_rankings", forbidden_compare)

        run_a = _make_package(
            tmp_path / "runs" / "run-a", _INCOMPLETE_BYTES, compressed=True
        )
        run_b = _make_package(tmp_path / "runs" / "run-b", INPUT2_BYTES)

        real_validate_and_load = comparison_inputs._validate_and_load
        captured = {}

        def spy_validate_and_load(readable_path, source, benchmark):
            try:
                return real_validate_and_load(readable_path, source, benchmark)
            except T1ComparisonInputError as exc:
                captured["error"] = exc
                captured["validation"] = exc.validation
                captured["cause"] = exc.__cause__
                raise

        monkeypatch.setattr(
            comparison_inputs, "_validate_and_load", spy_validate_and_load
        )

        owned = tmp_path / "owned"
        owned.mkdir()
        real = comparison_inputs.tempfile.TemporaryDirectory
        tracked = {}

        def wrapped(*args, **kwargs):
            kwargs = dict(kwargs)
            kwargs["dir"] = str(owned)
            td = real(*args, **kwargs)
            tracked["path"] = Path(td.name)

            class FailingTemporaryDirectory:
                def __enter__(self):
                    return td.name

                def __exit__(self, etype, value, tb):
                    td.cleanup()
                    raise OSError("temporary cleanup denied")

            return FailingTemporaryDirectory()

        monkeypatch.setattr(comparison_inputs.tempfile, "TemporaryDirectory", wrapped)

        with pytest.raises(T1ComparisonInputError) as excinfo:
            load_t1_comparison(run_a, run_b, repo_root=repo_root)

        err = excinfo.value
        assert err is captured["error"]
        assert err.validation is captured["validation"]
        assert err.validation is not None
        assert err.validation.is_valid is False
        assert "invalid T1 predictions" in str(err)
        assert str(run_a / "dev.csv.zst") in str(err)
        assert err.__cause__ is captured["cause"]
        assert any("temporary cleanup denied" in note for note in err.__notes__)
        assert not tracked["path"].exists()

    def test_benchmark_split_undecodable(self, tmp_path, monkeypatch):
        _install_tables(monkeypatch)
        repo_root, split_path = _setup_repo(tmp_path)
        _stub_preflight(monkeypatch, repo_root / "data" / "dictorpus-data")
        split_path.write_bytes(b"language,text_id,split\nkrl,1,\xff\xfe\n")
        run_a = _make_package(tmp_path / "runs" / "run-a", INPUT1_BYTES)
        run_b = _make_package(tmp_path / "runs" / "run-b", INPUT2_BYTES)

        with pytest.raises(T1ComparisonInputError) as excinfo:
            load_t1_comparison(run_a, run_b, repo_root=repo_root)

        assert str(split_path) in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, UnicodeDecodeError)

    def test_metrics_input_error_context(self, tmp_path, monkeypatch):
        _install_tables(monkeypatch)
        repo_root, _ = _setup_repo(tmp_path)
        _stub_preflight(monkeypatch, repo_root / "data" / "dictorpus-data")
        run_a = _make_package(tmp_path / "runs" / "run-a", INPUT1_BYTES)
        run_b = _make_package(tmp_path / "runs" / "run-b", INPUT2_BYTES)

        def failing_compare(*args, **kwargs):
            raise MetricsInputError("distinctive core failure")

        monkeypatch.setattr(comparison_inputs, "compare_rankings", failing_compare)

        with pytest.raises(T1ComparisonInputError) as excinfo:
            load_t1_comparison(run_a, run_b, repo_root=repo_root)

        message = str(excinfo.value)
        assert "distinctive core failure" in message
        assert str(run_a / "dev.csv") in message
        assert str(run_b / "dev.csv") in message
        assert isinstance(excinfo.value.__cause__, MetricsInputError)