"""Offline tests for the T1 comparison CSV output module."""

import csv
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from core.ranking_comparison import OccurrenceComparison  # noqa: E402
from t1_features.comparison_inputs import (  # noqa: E402
    T1ComparisonSource,
    T1LoadedComparison,
)
from t1_features.comparison_output import (  # noqa: E402
    COMPARISON_HEADER,
    T1ComparisonPublishedCleanupError,
    T1ComparisonWriteError,
    default_comparison_path,
    write_comparison_csv,
)

_TAG = "fixture-tag"


def _row(word_id, **overrides):
    base = dict(
        language="krl",
        word_id=word_id,
        word="kala",
        candidate_count=2,
        gold_wordform_id=10,
        gold_gramset="N+SG+NOM",
        a_top1_wordform_id=10,
        a_top1_gramset="N+SG+NOM",
        b_top1_wordform_id=11,
        b_top1_gramset="N+PL",
        a_gold_rank=1,
        b_gold_rank=2,
        top1_changed=True,
        outcome="regressed",
        delta_rr=-0.5,
    )
    base.update(overrides)
    return OccurrenceComparison(**base)


ROWS = (
    _row(
        1,
        word='kala, "хъ"',
        b_top1_gramset='N+PL,"ALT"',
        top1_changed=True,
        outcome="corrected",
        delta_rr=0.3333333333333333,
    ),
    _row(
        2,
        word="line1\nline2",
        top1_changed=False,
        outcome="both_correct",
        delta_rr=-0.25,
    ),
    _row(3, outcome="both_wrong_same", delta_rr=0.0),
)


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


class TestCsv:
    def test_exact_header_fields_and_full_precision(self, tmp_path):
        out = tmp_path / "out.csv"
        write_comparison_csv(iter(ROWS), out)

        parsed = _read_csv(out)
        assert parsed[0] == COMPARISON_HEADER
        assert len(parsed) - 1 == len(ROWS)

        raw = out.read_text(encoding="utf-8")
        assert "0.3333333333333333" in raw
        assert "-0.25" in raw
        assert "0.0" in raw

        row1 = parsed[1]
        assert row1[0] == "krl"
        assert row1[1] == "1"
        assert row1[2] == 'kala, "хъ"'
        assert row1[3] == "2"
        assert row1[4] == "10"
        assert row1[5] == "N+SG+NOM"
        assert row1[6] == "10"
        assert row1[7] == "N+SG+NOM"
        assert row1[8] == "11"
        assert row1[9] == 'N+PL,"ALT"'
        assert row1[10] == "1"
        assert row1[11] == "2"
        assert row1[12] == "1"
        assert row1[13] == "corrected"
        assert row1[14] == "0.3333333333333333"

        row2 = parsed[2]
        assert row2[2] == "line1\nline2"
        assert row2[12] == "0"
        assert row2[13] == "both_correct"
        assert row2[14] == "-0.25"

        assert not list(tmp_path.glob(".vepkar-t1-comparison-*.tmp"))

    def test_writes_every_row_and_keeps_row_objects(self, tmp_path):
        out = tmp_path / "all.csv"
        write_comparison_csv(ROWS, out)
        parsed = _read_csv(out)
        assert [row[1] for row in parsed[1:]] == ["1", "2", "3"]
        assert [row[13] for row in parsed[1:]] == [
            "corrected",
            "both_correct",
            "both_wrong_same",
        ]
        assert [row[12] for row in parsed[1:]] == ["1", "0", "1"]
        assert not list(tmp_path.glob(".vepkar-t1-comparison-*.tmp"))


class TestAutomaticPath:
    def _source(self, run_dir, method="t1.1", C=1.0, tag=_TAG):
        return T1ComparisonSource(
            run_dir=Path(run_dir),
            predictions_path=Path(run_dir) / "dev.csv",
            method=method,
            C=C,
            data_tag=tag,
            data_commit="c",
            split_sha256="s",
        )

    def _loaded(self, s1, s2):
        return T1LoadedComparison(s1, s2, SimpleNamespace(rows=()))

    def test_path_from_metadata_not_folder_names(self, tmp_path):
        s1 = self._source("somewhere/run-x", C=1.0)
        s2 = self._source("elsewhere/run-y", C=0.1)
        path = default_comparison_path(
            self._loaded(s1, s2), repo_root=tmp_path / "repo"
        )
        assert path == (
            tmp_path
            / "repo"
            / "work"
            / "comparisons"
            / _TAG
            / "t1.1-C1__vs__t1.1-C0.1.csv"
        )

    def test_path_uses_input_order_not_winner(self, tmp_path):
        winner_first = default_comparison_path(
            self._loaded(
                self._source("winner", C=0.1), self._source("loser", C=1.0)
            ),
            repo_root=tmp_path / "repo",
        )
        assert winner_first.name == "t1.1-C0.1__vs__t1.1-C1.csv"
        loser_first = default_comparison_path(
            self._loaded(
                self._source("loser", C=1.0), self._source("winner", C=0.1)
            ),
            repo_root=tmp_path / "repo",
        )
        assert loser_first.name == "t1.1-C1__vs__t1.1-C0.1.csv"

    def test_equal_c_and_nearby_c_labels(self):
        equal = default_comparison_path(
            self._loaded(self._source("a", C=1), self._source("b", C=1.0)),
            repo_root=Path("/repo"),
        )
        assert equal.name == "t1.1-C1__vs__t1.1-C1.csv"

        nearby = default_comparison_path(
            self._loaded(
                self._source("a", C=1.0), self._source("b", C=1.0000000000000002)
            ),
            repo_root=Path("/repo"),
        )
        assert nearby.name == "t1.1-C1__vs__t1.1-C1.0000000000000002.csv"

        scientific = default_comparison_path(
            self._loaded(self._source("a", C=1e-07), self._source("b", C=0.1)),
            repo_root=Path("/repo"),
        )
        assert scientific.name == "t1.1-C1e-07__vs__t1.1-C0.1.csv"


class TestProtection:
    def test_existing_file_rejected(self, tmp_path):
        out = tmp_path / "out.csv"
        out.write_text("original", encoding="utf-8")
        with pytest.raises(T1ComparisonWriteError, match="already exists"):
            write_comparison_csv(ROWS, out)
        assert out.read_text(encoding="utf-8") == "original"
        assert not list(tmp_path.glob(".vepkar-t1-comparison-*.tmp"))

    def test_existing_directory_rejected(self, tmp_path):
        out = tmp_path / "outdir"
        out.mkdir()
        (out / "keep.txt").write_text("keep", encoding="utf-8")
        with pytest.raises(T1ComparisonWriteError, match="already exists"):
            write_comparison_csv(ROWS, out)
        assert out.is_dir()
        assert (out / "keep.txt").read_text(encoding="utf-8") == "keep"

    def test_dangling_symlink_rejected_not_followed(self, tmp_path):
        out = tmp_path / "out.csv"
        missing_target = tmp_path / "missing-target.csv"
        try:
            out.symlink_to(missing_target)
        except (OSError, NotImplementedError):
            pytest.skip("Symlink creation is unavailable in this environment")
        with pytest.raises(T1ComparisonWriteError, match="already exists"):
            write_comparison_csv(ROWS, out)
        assert out.is_symlink()
        assert out.readlink() == missing_target
        assert not missing_target.exists()
        assert not list(tmp_path.glob(".vepkar-t1-comparison-*.tmp"))

    def test_missing_parent_rejected(self, tmp_path):
        out = tmp_path / "no-such-dir" / "out.csv"
        with pytest.raises(T1ComparisonWriteError, match="parent directory does "
                                                        "not exist"):
            write_comparison_csv(ROWS, out)
        assert not out.exists()
        assert not (tmp_path / "no-such-dir").exists()


class TestPublication:
    def test_collision_preserves_competing_bytes(self, tmp_path, monkeypatch):
        out = tmp_path / "out.csv"
        foreign = b"foreign competing destination bytes\n"
        real_link = os.link

        def competing_link(src, dst):
            if Path(dst) == out:
                with open(dst, "wb") as fh:
                    fh.write(foreign)
            return real_link(src, dst)

        monkeypatch.setattr("t1_features.comparison_output.os.link", competing_link)

        with pytest.raises(T1ComparisonWriteError) as excinfo:
            write_comparison_csv(ROWS, out)

        assert "publish" in str(excinfo.value)
        assert out.read_bytes() == foreign
        assert not list(tmp_path.glob(".vepkar-t1-comparison-*.tmp"))

    def test_write_failure_wraps_stage_and_cleans_temp(
        self, tmp_path, monkeypatch
    ):
        out = tmp_path / "out.csv"
        real_writer = csv.writer
        state = {"rows": 0}

        class _FailingWriter:
            def __init__(self, stream):
                self._writer = real_writer(stream)

            def writerow(self, row):
                if state["rows"] >= 1:
                    raise OSError("disk full during comparison write")
                state["rows"] += 1
                self._writer.writerow(row)

            def writerows(self, rows):
                for row in rows:
                    self.writerow(row)

        monkeypatch.setattr(
            "t1_features.comparison_output.csv.writer",
            lambda stream, *args, **kwargs: _FailingWriter(stream),
        )

        with pytest.raises(T1ComparisonWriteError) as excinfo:
            write_comparison_csv(ROWS, out)

        assert "write temporary" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, OSError)
        assert not out.exists()
        assert not list(tmp_path.glob(".vepkar-t1-comparison-*.tmp"))

    def test_post_publication_cleanup_failure_preserves_csv(
        self, tmp_path, monkeypatch
    ):
        out = tmp_path / "out.csv"

        def fail_removal(self, real_unlink):
            raise PermissionError("cleanup denied")

        real_unlink = _patch_comparison_unlink(monkeypatch, fail_removal)

        with pytest.raises(T1ComparisonPublishedCleanupError) as excinfo:
            write_comparison_csv(ROWS, out)

        err = excinfo.value
        assert err.output_path == out
        temps = list(tmp_path.glob(".vepkar-t1-comparison-*.tmp"))
        assert len(temps) == 1
        assert err.temp_path == temps[0]
        assert isinstance(err.__cause__, PermissionError)
        assert "published" in str(err)

        parsed = _read_csv(out)
        assert parsed[0] == COMPARISON_HEADER
        assert len(parsed) - 1 == len(ROWS)
        real_unlink(temps[0])
