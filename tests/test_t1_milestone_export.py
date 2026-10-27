"""Offline tests for exporting one finished T1.1 dev run into a milestone."""

import json
import os
import sys
from pathlib import Path

import pytest
import zstandard

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import t1_features.milestone_export as milestone_export  # noqa: E402
from t1_features.milestone_export import (  # noqa: E402
    T1MilestoneExportError,
    export_t1_milestone,
)

TAG = "v2026.09"
FINGERPRINT = "0" * 64


def _metrics_bytes(
    *,
    tag=TAG,
    C=2.5,
    word_count=4,
    candidate_count=7,
    top1=0.5,
    mrr=0.75,
    top3=1.0,
):
    payload = {
        "format": "vepkar-metrics",
        "format_version": 2,
        "method": "t1.1",
        "split": "dev",
        "parameters": {"C": C},
        "word_count": word_count,
        "candidate_count": candidate_count,
        "top1_correct": 2,
        "top3_correct": 4,
        "metrics": {
            "top1_accuracy": top1,
            "mrr": mrr,
            "top3_accuracy": top3,
        },
        "data": {"tag": tag, "commit": "data-commit", "split_sha256": "c" * 64},
        "predictions_sha256": FINGERPRINT,
    }
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )


def _make_run(tmp_path, *, name="t1-C1-run1", metrics=None):
    run_dir = tmp_path / name
    run_dir.mkdir()
    (run_dir / "dev.csv").write_bytes(b"a,b\n1,2\n")
    (run_dir / "metrics.json").write_bytes(
        metrics if metrics is not None else _metrics_bytes()
    )
    return run_dir


def _export(tmp_path, run_dir, tag=TAG):
    return export_t1_milestone(run_dir, milestones_dir=tmp_path / "milestones")


def _staging_leftovers(root):
    return list(root.rglob(".vepkar-t1-milestone-*.tmp"))


class TestExactDestinationAndFiles:
    def test_export_creates_exact_destination_with_exactly_three_files(
        self, tmp_path
    ):
        run_dir = _make_run(tmp_path)
        destination = _export(tmp_path, run_dir)
        assert destination == tmp_path / "milestones" / TAG / run_dir.name
        assert sorted(p.name for p in destination.iterdir()) == [
            "README.md",
            "dev.csv.zst",
            "metrics.json",
        ]
        assert sorted(p.name for p in run_dir.iterdir()) == [
            "dev.csv",
            "metrics.json",
        ]

    def test_metrics_json_copied_byte_for_byte_with_original_whitespace(
        self, tmp_path
    ):
        raw = (
            '{\n'
            '\t"format": "vepkar-metrics",\n'
            '\t"format_version": 2,\n'
            '\t"method": "t1.1",\n'
            '\t"split": "dev",\n'
            '\t"parameters": {"C": 1.75},\n'
            '\t"word_count": 5,\n'
            '\t"candidate_count": 9,\n'
            '\t"metrics": {"top1_accuracy": 0.6, "mrr": 0.7, '
            '"top3_accuracy": 1.0},\n'
            '\t"data": {"tag": "v2026.09"},\n'
            '\t"predictions_sha256": "' + FINGERPRINT + '"\n'
            '}\n'
        ).encode("utf-8")
        run_dir = _make_run(tmp_path, metrics=raw)
        destination = _export(tmp_path, run_dir)
        assert (destination / "metrics.json").read_bytes() == raw

    def test_dev_csv_zst_round_trips_exact_bytes(self, tmp_path):
        csv_bytes = (
            "word_id,wordform_id,gramset,rank,score\n"
            "501,9001,\"V+PST+äöü\",1,0.5\n"
        ).encode("utf-8")
        run_dir = _make_run(tmp_path)
        (run_dir / "dev.csv").write_bytes(csv_bytes)
        destination = _export(tmp_path, run_dir)
        dctx = zstandard.ZstdDecompressor()
        with open(destination / "dev.csv.zst", "rb") as fh:
            with dctx.stream_reader(fh) as reader:
                assert reader.read() == csv_bytes


class TestCompressionLevel:
    def test_compression_uses_fixed_level_22(self, tmp_path, monkeypatch):
        run_dir = _make_run(tmp_path)
        real = zstandard.ZstdCompressor
        seen = {}

        class SpyCompressor:
            def __init__(self, **kwargs):
                seen.update(kwargs)
                self._inner = real(**kwargs)

            def compressobj(self):
                return self._inner.compressobj()

        monkeypatch.setattr(
            milestone_export.zstandard, "ZstdCompressor", SpyCompressor
        )
        destination = _export(tmp_path, run_dir)
        assert seen == {"level": 22}
        assert (destination / "dev.csv.zst").is_file()


class TestReadme:
    def test_readme_contains_factual_fields_and_qualified_statement(
        self, tmp_path
    ):
        run_dir = _make_run(tmp_path, metrics=_metrics_bytes(C=0.00001))
        destination = _export(tmp_path, run_dir)
        readme = (destination / "README.md").read_text(encoding="utf-8")
        assert "# T1.1: feature-based candidate ranking" in readme
        assert "Binary logistic regression with morphological, suffix, and" in readme
        assert "immediate-context features, without train-frequency features" in readme
        assert f"Data tag: {TAG}" in readme
        assert "C: 1e-05" in readme
        assert "Evaluation split: dev" in readme
        assert (
            "This is a development-set result, not a final test evaluation."
            in readme
        )
        assert "| Word occurrences | 4 |" in readme
        assert "| Candidate rows | 7 |" in readme
        assert "| Top-1 accuracy | 0.5000 |" in readme
        assert "| MRR | 0.7500 |" in readme
        assert "| Top-3 accuracy | 1.0000 |" in readme
        assert "`metrics.json`" in readme
        assert "`dev.csv.zst`" in readme
        assert "The trained model is not included in this directory." in readme
        assert "../../../../README.md#t1-feature-based-model" in readme
        assert readme.endswith("\n")

    def test_readme_does_not_expose_temporary_absolute_paths(self, tmp_path):
        run_dir = _make_run(tmp_path)
        destination = _export(tmp_path, run_dir)
        readme = (destination / "README.md").read_text(encoding="utf-8")
        assert str(run_dir) not in readme
        assert str(tmp_path) not in readme

    def test_readme_metrics_unmodified_in_metrics_json(self, tmp_path):
        run_dir = _make_run(tmp_path)
        destination = _export(tmp_path, run_dir)
        parsed = json.loads(
            (destination / "metrics.json").read_text(encoding="utf-8")
        )
        assert parsed["metrics"]["top1_accuracy"] == 0.5
        assert parsed["metrics"]["mrr"] == 0.75
        assert parsed["metrics"]["top3_accuracy"] == 1.0


class TestSourceUnchanged:
    def test_source_run_unchanged_and_no_leftovers(self, tmp_path):
        run_dir = _make_run(tmp_path, name="t1-C2-run2")
        before_entries = sorted(p.name for p in run_dir.iterdir())
        before_bytes = {p.name: p.read_bytes() for p in run_dir.iterdir()}
        _export(tmp_path, run_dir)
        assert sorted(p.name for p in run_dir.iterdir()) == before_entries
        assert {p.name: p.read_bytes() for p in run_dir.iterdir()} == before_bytes
        assert _staging_leftovers(tmp_path / "milestones") == []


class TestExistingDestinationPreserved:
    @pytest.mark.parametrize(
        "kind",
        ["file", "empty-dir", "nonempty-dir", "dangling-symlink"],
    )
    def test_existing_destination_preserved(self, tmp_path, kind):
        run_dir = _make_run(tmp_path)
        milestones = tmp_path / "milestones"
        destination = milestones / TAG / run_dir.name
        destination.parent.mkdir(parents=True)
        if kind == "file":
            destination.write_bytes(b"precious bytes")
        elif kind == "empty-dir":
            destination.mkdir()
        elif kind == "nonempty-dir":
            destination.mkdir()
            (destination / "keep.txt").write_bytes(b"keep")
        else:
            try:
                destination.symlink_to(tmp_path / "missing-target")
            except (OSError, NotImplementedError):
                pytest.skip("Symlink creation is unavailable in this environment")
        with pytest.raises(T1MilestoneExportError):
            _export(tmp_path, run_dir)
        if kind == "file":
            assert destination.read_bytes() == b"precious bytes"
        elif kind in ("empty-dir", "nonempty-dir"):
            assert destination.is_dir()
            if kind == "nonempty-dir":
                assert (destination / "keep.txt").read_bytes() == b"keep"
        else:
            assert destination.is_symlink()
            assert not (tmp_path / "missing-target").exists()
        assert _staging_leftovers(milestones) == []
        assert sorted(p.name for p in run_dir.iterdir()) == [
            "dev.csv",
            "metrics.json",
        ]


class TestInputFailures:
    def test_run_dir_missing_fails_with_path(self, tmp_path):
        missing = tmp_path / "no-such-run"
        with pytest.raises(T1MilestoneExportError) as exc_info:
            export_t1_milestone(missing, milestones_dir=tmp_path / "m")
        assert str(missing) in str(exc_info.value)
        assert not (tmp_path / "m").exists()

    def test_missing_metrics_json_fails(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "dev.csv").write_bytes(b"x\n")
        with pytest.raises(T1MilestoneExportError) as exc_info:
            _export(tmp_path, run_dir)
        assert "metrics.json" in str(exc_info.value)

    def test_missing_dev_csv_fails(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "metrics.json").write_bytes(_metrics_bytes())
        with pytest.raises(T1MilestoneExportError) as exc_info:
            _export(tmp_path, run_dir)
        assert "dev.csv" in str(exc_info.value)

    def test_malformed_metrics_json_fails(self, tmp_path):
        run_dir = _make_run(tmp_path, metrics=b"not json {")
        with pytest.raises(T1MilestoneExportError) as exc_info:
            _export(tmp_path, run_dir)
        assert "metrics.json" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_invalid_utf8_metrics_fails(self, tmp_path):
        run_dir = _make_run(tmp_path, metrics=b"\xff\xfe")
        with pytest.raises(T1MilestoneExportError):
            _export(tmp_path, run_dir)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("format", "other-format"),
            ("format_version", 3),
            ("format_version", True),
            ("method", "t1.0"),
            ("split", "test"),
        ],
    )
    def test_unsupported_identifiers_fail(self, tmp_path, field, value):
        metrics = json.loads(_metrics_bytes())
        metrics[field] = value
        run_dir = _make_run(
            tmp_path, metrics=(json.dumps(metrics) + "\n").encode("utf-8")
        )
        with pytest.raises(T1MilestoneExportError) as exc_info:
            _export(tmp_path, run_dir)
        assert "unsupported" in str(exc_info.value).lower()

    @pytest.mark.parametrize("tag", ["", ".", "..", "../escape", "a/b", "a\\b"])
    def test_unsafe_tag_rejected(self, tmp_path, tag):
        metrics = json.loads(_metrics_bytes())
        metrics["data"]["tag"] = tag
        run_dir = _make_run(
            tmp_path, metrics=(json.dumps(metrics) + "\n").encode("utf-8")
        )
        with pytest.raises(T1MilestoneExportError):
            _export(tmp_path, run_dir)
        assert not (tmp_path / "milestones").exists()


class TestControlledFailures:
    def test_compression_failure_leaves_no_partial_result(
        self, tmp_path, monkeypatch
    ):
        run_dir = _make_run(tmp_path)
        milestones = tmp_path / "milestones"

        class FailingCompressor:
            def __init__(self, **kwargs):
                pass

            def compressobj(self):
                raise zstandard.ZstdError("simulated compression failure")

        monkeypatch.setattr(
            milestone_export.zstandard, "ZstdCompressor", FailingCompressor
        )
        with pytest.raises(T1MilestoneExportError) as exc_info:
            export_t1_milestone(run_dir, milestones_dir=milestones)
        assert isinstance(exc_info.value.__cause__, zstandard.ZstdError)
        tag_dir = milestones / TAG
        assert not (tag_dir / run_dir.name).exists()
        assert _staging_leftovers(milestones) == []
        assert sorted(p.name for p in run_dir.iterdir()) == [
            "dev.csv",
            "metrics.json",
        ]
        assert (run_dir / "dev.csv").read_bytes() == b"a,b\n1,2\n"

    def test_publication_competing_destination_preserved(
        self, tmp_path, monkeypatch
    ):
        run_dir = _make_run(tmp_path)
        milestones = tmp_path / "milestones"
        destination = milestones / TAG / run_dir.name
        foreign = b"foreign competing destination bytes\n"
        real_mkdir = os.mkdir

        def competing_mkdir(path, *args, **kwargs):
            if Path(path) == destination:
                with open(destination, "wb") as fh:
                    fh.write(foreign)
            return real_mkdir(path, *args, **kwargs)

        monkeypatch.setattr(milestone_export.os, "mkdir", competing_mkdir)
        with pytest.raises(T1MilestoneExportError):
            export_t1_milestone(run_dir, milestones_dir=milestones)
        assert destination.read_bytes() == foreign
        assert _staging_leftovers(milestones) == []
        assert sorted(p.name for p in run_dir.iterdir()) == [
            "dev.csv",
            "metrics.json",
        ]
        assert (run_dir / "dev.csv").read_bytes() == b"a,b\n1,2\n"

    def test_publication_file_collision_preserves_competing_bytes(
        self, tmp_path, monkeypatch
    ):
        run_dir = _make_run(tmp_path)
        milestones = tmp_path / "milestones"
        destination = milestones / TAG / run_dir.name
        foreign = b"foreign competing target bytes\n"
        real_link = os.link
        injected = False

        def competing_link(source, target, *args, **kwargs):
            nonlocal injected
            if not injected and Path(target) == destination / "README.md":
                injected = True
                with open(target, "wb") as fh:
                    fh.write(foreign)
            return real_link(source, target, *args, **kwargs)

        monkeypatch.setattr(milestone_export.os, "link", competing_link)
        with pytest.raises(T1MilestoneExportError):
            export_t1_milestone(run_dir, milestones_dir=milestones)
        assert (destination / "README.md").read_bytes() == foreign
        assert _staging_leftovers(milestones) == []
        assert sorted(p.name for p in run_dir.iterdir()) == [
            "dev.csv",
            "metrics.json",
        ]
        assert (run_dir / "dev.csv").read_bytes() == b"a,b\n1,2\n"