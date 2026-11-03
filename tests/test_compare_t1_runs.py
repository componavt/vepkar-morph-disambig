"""Offline CLI tests for the compare-t1-runs command."""

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
import zstandard

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import benchmark_context  # noqa: E402
from core.data import SUPPORTED_LANGUAGES, CorpusTables  # noqa: E402
from core.instances import Candidate, Instance  # noqa: E402
from core.ranking_comparison import compare_rankings  # noqa: E402
from t1_features.comparison_inputs import (  # noqa: E402
    T1ComparisonSource,
    T1LoadedComparison,
)
from t1_features.comparison_output import COMPARISON_HEADER  # noqa: E402

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


def _write_split_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["language", "text_id", "split"])
        writer.writerows(rows)


def _setup_repo(tmp_path, tag=_TAG):
    repo_root = tmp_path / "repo"
    split_path = repo_root / "data" / "derived" / f"splits_{tag}.csv"
    _write_split_csv(
        split_path,
        [("krl", 1, "train"), ("krl", 2, "dev"), ("krl", 3, "test")],
    )
    return repo_root, split_path


def _stub_preflight(monkeypatch, data_dir):
    import t1_features.comparison_inputs as input_pkg

    recorded = {}

    def fake(data_dir_override):
        recorded["data_dir"] = data_dir_override
        return data_dir_override, _TAG

    monkeypatch.setattr(input_pkg, "_preflight_benchmark", fake)
    return recorded


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

_INCOMPLETE_ROWS = [
    (201, 2010, "D1", 1, 2.0),
    (201, 2011, "D1+ALT", 2, 1.0),
    (202, 2020, "D2", 1, 3.0),
    (202, 2021, "D2+ALT", 2, 2.0),
    (204, 2040, "D4", 1, 4.0),
]

INPUT1_BYTES = _csv_bytes(INPUT1_ROWS)
INPUT2_BYTES = _csv_bytes(INPUT2_ROWS)
_INCOMPLETE_BYTES = _csv_bytes(_INCOMPLETE_ROWS)


def _make_package(run_dir, csv_bytes, *, compressed=False, C=0.1, tag=_TAG):
    run_dir.mkdir(parents=True, exist_ok=True)
    if compressed:
        (run_dir / "dev.csv.zst").write_bytes(
            zstandard.ZstdCompressor().compress(csv_bytes)
        )
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
        "data": {"tag": tag, "commit": _COMMIT, "split_sha256": _SPLIT_SHA256},
        "predictions_sha256": "f" * 64,
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    return run_dir


def _recursive_entries(root):
    return {path.relative_to(root).as_posix() for path in root.rglob("*")}


def _files(root):
    return {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.reader(fh, strict=True))


def _patch_comparison_unlink(monkeypatch, failure):
    real_unlink = Path.unlink

    def patched_unlink(self, *args, **kwargs):
        if self.name.startswith(".vepkar-t1-comparison-") and self.name.endswith(
            ".tmp"
        ):
            return failure(self, real_unlink)
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", patched_unlink)
    return real_unlink


def _stub_source(run_dir, C):
    run_dir = Path(run_dir)
    return T1ComparisonSource(
        run_dir=run_dir,
        predictions_path=run_dir / "dev.csv",
        method="t1.1",
        C=C,
        data_tag=_TAG,
        data_commit="c",
        split_sha256="s",
    )


@dataclass(frozen=True)
class _PredRow:
    word_id: int
    wordform_id: int
    gramset: str
    rank: int


def _occurrence(word_id, gold_index=1):
    candidates = tuple(Candidate(10 + i, f"G{i}") for i in range(5))
    return Instance(
        language="krl",
        word_id=word_id,
        sentence_id=word_id,
        text_id=word_id,
        word=f"w{word_id}",
        word_number=1,
        sentence_xml="<s/>",
        candidates=candidates,
        gold_analysis=candidates[gold_index],
    )


def _pred_rows(word_id, ranks):
    return [
        _PredRow(word_id, 10 + i, f"G{i}", rank) for i, rank in enumerate(ranks)
    ]


def _negative_delta_comparison():
    """A/B gold ranks 2,2,2 vs 1,5,5: B wins Top-1 but loses MRR and Top-3."""
    instances = [_occurrence(1), _occurrence(2), _occurrence(3)]
    a_rows = (
        _pred_rows(1, [1, 2, 3, 4, 5])
        + _pred_rows(2, [1, 2, 3, 4, 5])
        + _pred_rows(3, [1, 2, 3, 4, 5])
    )
    b_rows = (
        _pred_rows(1, [2, 1, 3, 4, 5])
        + _pred_rows(2, [1, 5, 2, 3, 4])
        + _pred_rows(3, [1, 5, 2, 3, 4])
    )
    return compare_rankings(instances, a_rows, b_rows)


def _stub_loaded(comparison, c1=1.0, c2=0.5):
    return T1LoadedComparison(
        source_1=_stub_source("/experiments/one", c1),
        source_2=_stub_source("/experiments/two", c2),
        comparison=comparison,
    )


def _cli_setup(tmp_path, monkeypatch):
    _install_tables(monkeypatch)
    repo_root, _ = _setup_repo(tmp_path)
    _stub_preflight(monkeypatch, repo_root / "data" / "dictorpus-data")
    import cli
    import commands.compare_t1_runs as compare_pkg

    monkeypatch.setattr(compare_pkg, "_PROJECT_ROOT", repo_root)
    return cli, repo_root


def _cli_packages(tmp_path):
    run_a = _make_package(tmp_path / "runs" / "run-a", INPUT1_BYTES, C=1.0)
    run_b = _make_package(tmp_path / "runs" / "run-b", INPUT2_BYTES, C=0.1)
    return run_a, run_b


class TestParser:
    def test_requires_two_run_dirs_and_optional_output(self):
        import cli

        parser = cli.build_parser()
        args = parser.parse_args(["compare-t1-runs", "--run-dirs", "one", "two"])
        assert args.command == "compare-t1-runs"
        assert args.run_dirs == [Path("one"), Path("two")]
        assert args.output is None

        args = parser.parse_args(
            ["compare-t1-runs", "--run-dirs", "one", "two", "--output", "out.csv"]
        )
        assert args.run_dirs == [Path("one"), Path("two")]
        assert args.output == Path("out.csv")

    def test_rejects_abbreviated_and_unsupported_options(self):
        import cli

        parser = cli.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["compare-t1-runs"])
        for extra in (
            ["--run-dirs", "one"],
            ["--run-dirs", "one", "two", "three"],
        ):
            with pytest.raises(SystemExit):
                parser.parse_args(["compare-t1-runs", *extra])
        for bad in (
            ["--run-dirs", "one", "two", "--force"],
            ["--run-dirs", "one", "two", "--reference", "x"],
            ["--run-dirs", "one", "two", "--candidate", "x"],
            ["--run-dirs", "one", "two", "--id", "x"],
            ["--run-dirs", "one", "two", "--split", "dev"],
            ["--run-dirs", "one", "two", "--data-dir", "x"],
            ["--run-dirs", "one", "two", "--split-file", "x"],
        ):
            with pytest.raises(SystemExit):
                parser.parse_args(["compare-t1-runs", *bad])
        with pytest.raises(SystemExit):
            parser.parse_args(["compare-t1-runs", "--run", "one", "two"])

    def test_cli_main_rejects_unsupported_option(self):
        import cli

        with pytest.raises(SystemExit):
            cli.main(["compare-t1-runs", "--run-dirs", "a", "b", "--force"])


class TestCliSuccess:
    def test_success_automatic_path_real_core(self, tmp_path, monkeypatch, capsys):
        _install_tables(monkeypatch)
        repo_root, split_path = _setup_repo(tmp_path)
        recorded = _stub_preflight(
            monkeypatch, repo_root / "data" / "dictorpus-data"
        )
        import cli
        import commands.compare_t1_runs as compare_pkg

        monkeypatch.setattr(compare_pkg, "_PROJECT_ROOT", repo_root)

        run_a = _make_package(tmp_path / "runs" / "run-a", INPUT1_BYTES, C=1.0)
        run_b = _make_package(
            tmp_path / "runs" / "run-b", INPUT2_BYTES, C=0.1, compressed=True
        )
        entries_a_before = _recursive_entries(run_a)
        entries_b_before = _recursive_entries(run_b)
        split_before = split_path.read_bytes()
        repo_files_before = _files(repo_root)

        status = cli.main(
            ["compare-t1-runs", "--run-dirs", str(run_a), str(run_b)]
        )
        captured = capsys.readouterr()

        assert status == 0, captured.err
        assert captured.err == ""
        out = captured.out
        assert "T1 comparison: OK" in out
        assert f"A: {run_a.resolve()} (t1.1, C=1)" in out
        assert f"B: {run_b.resolve()} (t1.1, C=0.1)" in out
        assert "B selected by Top-1" in out
        assert "Evaluated: 3 word instances, 8 candidate rows" in out
        assert "Top-1 accuracy 0.3333  1.0000  +0.6667" in out
        assert "MRR            0.5833  1.0000  +0.4167" in out
        assert "Top-3 accuracy 0.6667  1.0000  +0.3333" in out
        for line in (
            "both_correct: 1",
            "corrected: 2",
            "regressed: 0",
            "both_wrong_same: 0",
            "both_wrong_changed: 0",
            "top1_changed: 2",
            "gold_rank_improved: 2",
            "gold_rank_worsened: 0",
            "gold_rank_unchanged: 1",
        ):
            assert line in out
        assert "Delta top-1" in out
        assert "+0.6667" in out
        assert "N/A" in out

        expected = (
            repo_root
            / "work"
            / "comparisons"
            / _TAG
            / "t1.1-C1__vs__t1.1-C0.1.csv"
        )
        assert expected.is_file()
        assert expected.parent.is_dir()
        assert f"Comparison CSV: {expected}" in out

        parsed = _read_csv(expected)
        assert parsed[0] == COMPARISON_HEADER
        assert parsed[1] == [
            "krl", "201", "devkala", "2", "2010", "D1",
            "2010", "D1", "2010", "D1", "1", "1", "0",
            "both_correct", "0.0",
        ]
        assert parsed[2] == [
            "krl", "202", "devkala-alt", "2", "2021", "D2+ALT",
            "2020", "D2", "2021", "D2+ALT", "2", "1", "1",
            "corrected", "0.5",
        ]
        assert parsed[3] == [
            "krl", "204", "devkala-fourth", "4", "2043", "D4+FOURTH",
            "2040", "D4", "2043", "D4+FOURTH", "4", "1", "1",
            "corrected", "0.75",
        ]

        assert recorded["data_dir"] == (repo_root / "data" / "dictorpus-data").resolve()
        assert _recursive_entries(run_a) == entries_a_before
        assert _recursive_entries(run_b) == entries_b_before
        assert split_path.read_bytes() == split_before
        added = _files(repo_root) - repo_files_before
        assert added == {"work/comparisons/fixture-tag/t1.1-C1__vs__t1.1-C0.1.csv"}
        assert not list(repo_root.rglob(".vepkar-t1-comparison-*.tmp"))

    def test_reversal_preserves_ab_roles_path_and_csv(
        self, tmp_path, monkeypatch, capsys
    ):
        cli, repo_root = _cli_setup(tmp_path, monkeypatch)
        run_a = _make_package(tmp_path / "runs" / "run-a", INPUT1_BYTES, C=1.0)
        run_b = _make_package(tmp_path / "runs" / "run-b", INPUT2_BYTES, C=0.1)

        status = cli.main(
            ["compare-t1-runs", "--run-dirs", str(run_a), str(run_b)]
        )
        captured = capsys.readouterr()
        assert status == 0, captured.err
        assert f"A: {run_a.resolve()} (t1.1, C=1)" in captured.out
        assert f"B: {run_b.resolve()} (t1.1, C=0.1)" in captured.out
        assert "B selected by Top-1" in captured.out
        expected = (
            repo_root
            / "work"
            / "comparisons"
            / _TAG
            / "t1.1-C1__vs__t1.1-C0.1.csv"
        )
        assert expected.is_file()
        assert f"Comparison CSV: {expected}" in captured.out
        first_bytes = expected.read_bytes()

        expected.unlink()
        assert not expected.exists()

        status = cli.main(
            ["compare-t1-runs", "--run-dirs", str(run_b), str(run_a)]
        )
        captured = capsys.readouterr()
        assert status == 0, captured.err
        assert f"A: {run_a.resolve()} (t1.1, C=1)" in captured.out
        assert f"B: {run_b.resolve()} (t1.1, C=0.1)" in captured.out
        assert "B selected by Top-1" in captured.out
        assert f"Comparison CSV: {expected}" in captured.out
        assert expected.is_file()
        assert expected.read_bytes() == first_bytes

    def test_tie_preserves_input_order_and_equal_c_path(
        self, tmp_path, monkeypatch, capsys
    ):
        cli, repo_root = _cli_setup(tmp_path, monkeypatch)
        run_x = _make_package(tmp_path / "runs" / "run-x", INPUT1_BYTES, C=1.0)
        run_y = _make_package(tmp_path / "runs" / "run-y", INPUT1_BYTES, C=1.0)

        status = cli.main(
            ["compare-t1-runs", "--run-dirs", str(run_x), str(run_y)]
        )
        captured = capsys.readouterr()
        assert status == 0, captured.err
        assert "Tie on all three metrics; input order preserved" in captured.out
        assert f"A: {run_x.resolve()} (t1.1, C=1)" in captured.out
        assert f"B: {run_y.resolve()} (t1.1, C=1)" in captured.out
        assert "+0.0000" in captured.out
        expected = (
            repo_root
            / "work"
            / "comparisons"
            / _TAG
            / "t1.1-C1__vs__t1.1-C1.csv"
        )
        assert expected.is_file()

    def test_explicit_relative_output_uses_caller_cwd(
        self, tmp_path, monkeypatch, capsys
    ):
        cli, repo_root = _cli_setup(tmp_path, monkeypatch)
        run_a, run_b = _cli_packages(tmp_path)
        foreign = tmp_path / "foreign-cwd"
        foreign.mkdir()
        monkeypatch.chdir(foreign)

        status = cli.main(
            [
                "compare-t1-runs",
                "--run-dirs", str(run_a), str(run_b),
                "--output", "relative-out.csv",
            ]
        )
        captured = capsys.readouterr()
        assert status == 0, captured.err
        out_path = foreign / "relative-out.csv"
        assert out_path.is_file()
        assert f"Comparison CSV: {out_path}" in captured.out
        assert not (repo_root / "work").exists()

    def test_report_formats_signed_deltas_and_na(
        self, tmp_path, monkeypatch, capsys
    ):
        import cli
        import commands.compare_t1_runs as compare_pkg

        repo_root = tmp_path / "repo"
        monkeypatch.setattr(compare_pkg, "_PROJECT_ROOT", repo_root)
        loaded = _stub_loaded(_negative_delta_comparison())
        monkeypatch.setattr(
            compare_pkg,
            "load_t1_comparison",
            lambda d1, d2, *, repo_root: loaded,
        )

        status = cli.main(["compare-t1-runs", "--run-dirs", "/x/one", "/x/two"])
        captured = capsys.readouterr()
        assert status == 0, captured.err
        out = captured.out
        assert "T1 comparison: OK" in out
        assert "B selected by Top-1" in out
        assert "Top-1 accuracy 0.0000  0.3333  +0.3333" in out
        assert "MRR            0.5000  0.4667  -0.0333" in out
        assert "Top-3 accuracy 1.0000  0.3333  -0.6667" in out
        assert "N/A" in out
        assert "-0.0333" in out
        assert "+0.3333" in out
        assert "-0.6667" in out
        expected = (
            repo_root
            / "work"
            / "comparisons"
            / _TAG
            / "t1.1-C1__vs__t1.1-C0.5.csv"
        )
        assert expected.is_file()


class TestCliFailures:
    def test_input_error_shows_validation_details_and_cleanup_note(
        self, tmp_path, monkeypatch, capsys
    ):
        _install_tables(monkeypatch)
        repo_root, _ = _setup_repo(tmp_path)
        _stub_preflight(monkeypatch, repo_root / "data" / "dictorpus-data")
        import cli
        import commands.compare_t1_runs as compare_pkg
        import t1_features.comparison_inputs as input_pkg

        monkeypatch.setattr(compare_pkg, "_PROJECT_ROOT", repo_root)

        run_a = _make_package(tmp_path / "runs" / "run-a", _INCOMPLETE_BYTES)
        run_b = _make_package(tmp_path / "runs" / "run-b", INPUT2_BYTES)

        owned = tmp_path / "owned"
        owned.mkdir()
        real = input_pkg.tempfile.TemporaryDirectory

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

        monkeypatch.setattr(input_pkg.tempfile, "TemporaryDirectory", wrapped)

        status = cli.main(
            ["compare-t1-runs", "--run-dirs", str(run_a), str(run_b)]
        )
        captured = capsys.readouterr()
        assert status == 1
        assert "Predictions validation: FAILED" in captured.out
        assert "Expected:" in captured.out
        assert "Found:" in captured.out
        assert "missing candidate" in captured.out
        assert "error: invalid T1 predictions" in captured.err
        assert str(run_a / "dev.csv") in captured.err
        assert "temporary workspace cleanup failed" in captured.err
        assert "temporary cleanup denied" in captured.err
        assert "T1 comparison: OK" not in captured.out
        assert "Comparison CSV:" not in captured.out
        assert not (repo_root / "work").exists()
        assert not list(repo_root.rglob(".vepkar-t1-comparison-*.tmp"))

    def test_automatic_collision_refuses_overwrite(
        self, tmp_path, monkeypatch, capsys
    ):
        cli, repo_root = _cli_setup(tmp_path, monkeypatch)
        run_a, run_b = _cli_packages(tmp_path)

        status = cli.main(
            ["compare-t1-runs", "--run-dirs", str(run_a), str(run_b)]
        )
        capsys.readouterr()
        assert status == 0

        expected = (
            repo_root
            / "work"
            / "comparisons"
            / _TAG
            / "t1.1-C1__vs__t1.1-C0.1.csv"
        )
        first_bytes = expected.read_bytes()
        files_before = _files(repo_root)

        status = cli.main(
            ["compare-t1-runs", "--run-dirs", str(run_a), str(run_b)]
        )
        captured = capsys.readouterr()
        assert status == 1
        assert "destination already exists" in captured.err
        assert str(expected) in captured.err
        assert "--output" in captured.err
        assert expected.read_bytes() == first_bytes
        assert _files(repo_root) == files_before
        assert not list(repo_root.rglob(".vepkar-t1-comparison-*.tmp"))
        assert "T1 comparison: OK" not in captured.out

    def test_explicit_output_requires_existing_parent(
        self, tmp_path, monkeypatch, capsys
    ):
        cli, repo_root = _cli_setup(tmp_path, monkeypatch)
        run_a, run_b = _cli_packages(tmp_path)
        missing_parent = tmp_path / "no-such-dir"

        status = cli.main(
            [
                "compare-t1-runs",
                "--run-dirs", str(run_a), str(run_b),
                "--output", str(missing_parent / "out.csv"),
            ]
        )
        captured = capsys.readouterr()
        assert status == 1
        assert "parent directory does not exist" in captured.err
        assert str(missing_parent) in captured.err
        assert not missing_parent.exists()
        assert "T1 comparison: OK" not in captured.out
        assert not (repo_root / "work").exists()

    def test_writer_error_reports_path_diagnostic(
        self, tmp_path, monkeypatch, capsys
    ):
        cli, repo_root = _cli_setup(tmp_path, monkeypatch)
        run_a, run_b = _cli_packages(tmp_path)

        def failing_link(src, dst):
            raise OSError("link denied")

        monkeypatch.setattr(
            "t1_features.comparison_output.os.link", failing_link
        )

        status = cli.main(
            ["compare-t1-runs", "--run-dirs", str(run_a), str(run_b)]
        )
        captured = capsys.readouterr()
        assert status == 1
        assert "error: cannot write comparison CSV" in captured.err
        assert "link denied" in captured.err
        expected = (
            repo_root
            / "work"
            / "comparisons"
            / _TAG
            / "t1.1-C1__vs__t1.1-C0.1.csv"
        )
        assert not expected.exists()
        assert not list(repo_root.rglob(".vepkar-t1-comparison-*.tmp"))
        assert "T1 comparison: OK" not in captured.out

    def test_publication_collision_hint_on_file_exists_error(
        self, tmp_path, monkeypatch, capsys
    ):
        cli, repo_root = _cli_setup(tmp_path, monkeypatch)
        run_a, run_b = _cli_packages(tmp_path)

        def file_exists_link(src, dst):
            raise FileExistsError("simulated publication collision")

        monkeypatch.setattr(
            "t1_features.comparison_output.os.link", file_exists_link
        )

        status = cli.main(
            ["compare-t1-runs", "--run-dirs", str(run_a), str(run_b)]
        )
        captured = capsys.readouterr()
        assert status == 1
        assert "error: cannot write comparison CSV" in captured.err
        assert "publish" in captured.err
        assert "--output" in captured.err
        assert "T1 comparison: OK" not in captured.out
        assert not list(repo_root.rglob(".vepkar-t1-comparison-*.tmp"))

    def test_published_cleanup_failure_preserves_readable_csv(
        self, tmp_path, monkeypatch, capsys
    ):
        cli, repo_root = _cli_setup(tmp_path, monkeypatch)
        run_a, run_b = _cli_packages(tmp_path)

        def fail_removal(self, real_unlink):
            raise PermissionError("cleanup denied")

        real_unlink = _patch_comparison_unlink(monkeypatch, fail_removal)

        status = cli.main(
            ["compare-t1-runs", "--run-dirs", str(run_a), str(run_b)]
        )
        captured = capsys.readouterr()
        assert status == 1
        assert (
            "error: comparison CSV was published, but temporary cleanup failed:"
            in captured.err
        )
        expected = (
            repo_root
            / "work"
            / "comparisons"
            / _TAG
            / "t1.1-C1__vs__t1.1-C0.1.csv"
        )
        assert f"destination: {expected}" in captured.err
        temps = list(expected.parent.glob(".vepkar-t1-comparison-*.tmp"))
        assert len(temps) == 1
        assert f"temporary: {temps[0]}" in captured.err
        assert "cleanup denied" in captured.err
        assert "T1 comparison: OK" not in captured.out

        parsed = _read_csv(expected)
        assert parsed[0] == COMPARISON_HEADER
        assert len(parsed) - 1 == 3
        real_unlink(temps[0])
