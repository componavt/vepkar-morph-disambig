"""Offline tests for building and publishing the T1 metrics JSON."""

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from core.metrics import RankingMetrics  # noqa: E402
from t1_features.run_context import T1RunContext  # noqa: E402
import t1_features.metrics_io as metrics_io  # noqa: E402
from t1_features.metrics_io import (  # noqa: E402
    T1MetricsError,
    T1MetricsPublishedCleanupError,
    build_t1_metrics,
    write_t1_metrics,
)

TAG = "тест-тэг"


def _manifest(*, C=1.0, prepared_commit="0123abcd", tag=TAG):
    return {
        "format": "vepkar-t1-run",
        "format_version": 3,
        "stage": "prepared",
        "code": {"root": "/code", "commit": prepared_commit},
        "data": {
            "root": "/data",
            "tag": tag,
            "commit": "data-commit-1",
        },
        "split": {"path": "/split.csv", "sha256": "c" * 64},
        "parameters": {"C": C, "target_split": "dev"},
    }


def _context(tmp_path, **manifest_changes):
    run_dir = tmp_path / "run-dir"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = _manifest()
    manifest.update(manifest_changes)
    return T1RunContext(
        run_dir=run_dir,
        manifest=manifest,
        current_code_commit="evaluation-commit-2",
    )


def _metrics(*, word_count=3, top1=2, top3=3, mrr=0.8):
    return RankingMetrics(
        word_count=word_count,
        top1_correct=top1,
        top3_correct=top3,
        top1_accuracy=top1 / word_count,
        mrr=mrr,
        top3_accuracy=top3 / word_count,
    )


def _base_payload(tmp_path):
    context = _context(tmp_path)
    (context.run_dir / "dev.csv").write_bytes(b"word_id,rank\n601,1\n")
    return build_t1_metrics(context, metrics=_metrics(), candidate_count=4)


def _temp_names(root):
    return list(root.glob(".vepkar-t1-metrics-*.tmp"))


class TestExceptionTypes:
    def test_inheritance_and_reported_paths(self):
        assert issubclass(T1MetricsError, ValueError)
        assert issubclass(T1MetricsPublishedCleanupError, T1MetricsError)
        assert T1MetricsError is not ValueError
        output = Path("metrics.json")
        temp = Path("temp.tmp")
        error = T1MetricsPublishedCleanupError(output, temp)
        assert error.output_path == output
        assert error.temp_path == temp
        message = str(error)
        assert "published" in message.lower()
        assert "removal" in message.lower()
        assert str(output) in message
        assert str(temp) in message


class TestExactPayload:
    def test_complete_payload_with_unicode_and_unrounded_metrics(self, tmp_path):
        context = _context(tmp_path)
        raw = "word_id,wordform_id,gramset,rank,score\n601,2001,\"V+IND+PST+äöü\",1,0.1\n".encode(
            "utf-8"
        )
        (context.run_dir / "dev.csv").write_bytes(raw)
        metrics = RankingMetrics(
            word_count=3,
            top1_correct=1,
            top3_correct=2,
            top1_accuracy=1 / 3,
            mrr=0.6111111111111111,
            top3_accuracy=2 / 3,
        )
        payload = build_t1_metrics(context, metrics=metrics, candidate_count=5)

        assert list(payload) == [
            "format",
            "format_version",
            "method",
            "parameters",
            "split",
            "word_count",
            "candidate_count",
            "top1_correct",
            "top3_correct",
            "metrics",
            "data",
            "code",
            "predictions_sha256",
        ]
        assert payload == {
            "format": "vepkar-metrics",
            "format_version": 1,
            "method": "t1.1",
            "parameters": {"C": 1.0},
            "split": "dev",
            "word_count": 3,
            "candidate_count": 5,
            "top1_correct": 1,
            "top3_correct": 2,
            "metrics": {
                "top1_accuracy": 1 / 3,
                "mrr": 0.6111111111111111,
                "top3_accuracy": 2 / 3,
            },
            "data": {
                "tag": TAG,
                "commit": "data-commit-1",
                "split_sha256": "c" * 64,
            },
            "code": {
                "prepared_commit": "0123abcd",
                "evaluation_commit": "evaluation-commit-2",
            },
            "predictions_sha256": hashlib.sha256(raw).hexdigest(),
        }
        assert payload["code"] == {
            "prepared_commit": "0123abcd",
            "evaluation_commit": "evaluation-commit-2",
        }
        assert payload["metrics"]["top1_accuracy"] == 1 / 3
        assert payload["metrics"]["top3_accuracy"] == 2 / 3

    def test_fresh_nested_dictionaries(self, tmp_path):
        context = _context(tmp_path)
        (context.run_dir / "dev.csv").write_bytes(b"x\n")
        payload = build_t1_metrics(context, metrics=_metrics(), candidate_count=4)
        assert payload["parameters"] is not context.manifest["parameters"]
        assert payload["data"] is not context.manifest["data"]
        assert payload["code"] is not context.manifest["code"]
        assert payload["metrics"] is not _metrics()
        payload["data"]["tag"] = "mutated"
        assert context.manifest["data"]["tag"] == TAG


class TestBuildDoesNotModify:
    def test_inputs_and_directory_unchanged_fingerprint_tracks_bytes(self, tmp_path):
        context = _context(tmp_path)
        dev = context.run_dir / "dev.csv"
        original = b"a,b\n1,2\n"
        dev.write_bytes(original)
        manifest_before = copy.deepcopy(context.manifest)
        entries_before = sorted(p.name for p in context.run_dir.iterdir())
        metrics = _metrics()

        first = build_t1_metrics(context, metrics=metrics, candidate_count=4)
        assert context.manifest == manifest_before
        assert dev.read_bytes() == original
        assert sorted(p.name for p in context.run_dir.iterdir()) == entries_before

        dev.write_bytes(b"a,b\n1,3\n")
        second = build_t1_metrics(context, metrics=metrics, candidate_count=4)
        assert second["predictions_sha256"] != first["predictions_sha256"]
        assert second["predictions_sha256"] == hashlib.sha256(b"a,b\n1,3\n").hexdigest()


class TestBuildInputErrors:
    def test_missing_dev_csv_path_aware_chained_error(self, tmp_path):
        context = _context(tmp_path)
        with pytest.raises(T1MetricsError) as exc_info:
            build_t1_metrics(context, metrics=_metrics(), candidate_count=4)
        error = exc_info.value
        assert str(context.run_dir / "dev.csv") in str(error)
        assert isinstance(error.__cause__, OSError)

    def test_unreadable_dev_csv_path_aware_chained_error(self, tmp_path):
        context = _context(tmp_path)
        (context.run_dir / "dev.csv").mkdir()
        with pytest.raises(T1MetricsError) as exc_info:
            build_t1_metrics(context, metrics=_metrics(), candidate_count=4)
        error = exc_info.value
        assert str(context.run_dir / "dev.csv") in str(error)
        assert isinstance(error.__cause__, OSError)

    def test_builder_validates_arguments_before_reading_dev_csv(self, tmp_path):
        context = _context(tmp_path)
        with pytest.raises(T1MetricsError) as exc_info:
            build_t1_metrics(context, metrics=_metrics(), candidate_count=1)
        assert "candidate_count" in str(exc_info.value)
        assert "dev.csv" not in str(exc_info.value)

        bad_c = _context(tmp_path)
        bad_c.manifest["parameters"]["C"] = -2.0
        with pytest.raises(T1MetricsError) as exc_info:
            build_t1_metrics(bad_c, metrics=_metrics(), candidate_count=4)
        assert "parameters.C" in str(exc_info.value)
        assert "dev.csv" not in str(exc_info.value)


def _del(base, key):
    payload = copy.deepcopy(base)
    del payload[key]
    return payload


def _set(base, key, value):
    payload = copy.deepcopy(base)
    payload[key] = value
    return payload


def _set_nested(base, section, key, value):
    payload = copy.deepcopy(base)
    payload[section][key] = value
    return payload


def _del_nested(base, section, key):
    payload = copy.deepcopy(base)
    del payload[section][key]
    return payload


def _extra_key(base):
    payload = copy.deepcopy(base)
    payload["extra"] = 1
    return payload


def _extra_nested(base, section, key):
    payload = copy.deepcopy(base)
    payload[section][key] = key
    return payload


def _not_a_dict(base):
    return ["not", "a", "dict"]


MALFORMED = [
    ("not-a-dict", _not_a_dict),
    ("missing-top-level-key", lambda b: _del(b, "metrics")),
    ("missing-predictions-sha", lambda b: _del(b, "predictions_sha256")),
    ("extra-top-level-key", _extra_key),
    ("wrong-format", lambda b: _set(b, "format", "vepkar-other")),
    ("format-version-2", lambda b: _set(b, "format_version", 2)),
    ("format-version-float", lambda b: _set(b, "format_version", 1.0)),
    ("format-version-bool", lambda b: _set(b, "format_version", True)),
    ("wrong-method", lambda b: _set(b, "method", "t1.2")),
    ("wrong-split", lambda b: _set(b, "split", "test")),
    ("bool-word-count", lambda b: _set(b, "word_count", True)),
    ("zero-word-count", lambda b: _set(b, "word_count", 0)),
    ("bool-candidate-count", lambda b: _set(b, "candidate_count", True)),
    ("candidate-count-below-word-count", lambda b: _set(b, "candidate_count", 2)),
    ("negative-top1-count", lambda b: _set(b, "top1_correct", -1)),
    ("top1-above-top3", lambda b: _set(b, "top3_correct", 1)),
    ("top3-above-word-count", lambda b: _set(b, "top3_correct", 4)),
    ("inconsistent-top1-accuracy", lambda b: _set_nested(b, "metrics", "top1_accuracy", 0.9)),
    ("inconsistent-top3-accuracy", lambda b: _set_nested(b, "metrics", "top3_accuracy", 0.5)),
    ("accuracy-out-of-range", lambda b: _set_nested(b, "metrics", "top1_accuracy", 1.5)),
    ("accuracy-infinite", lambda b: _set_nested(b, "metrics", "top1_accuracy", float("inf"))),
    ("mrr-nan", lambda b: _set_nested(b, "metrics", "mrr", float("nan"))),
    ("mrr-zero", lambda b: _set_nested(b, "metrics", "mrr", 0.0)),
    ("mrr-above-one", lambda b: _set_nested(b, "metrics", "mrr", 1.5)),
    ("bool-accuracy", lambda b: _set_nested(b, "metrics", "top1_accuracy", True)),
    ("bool-C", lambda b: _set_nested(b, "parameters", "C", True)),
    ("nonpositive-C", lambda b: _set_nested(b, "parameters", "C", 0.0)),
    ("oversized-C", lambda b: _set_nested(b, "parameters", "C", 10**10000)),
    ("missing-nested-key", lambda b: _del_nested(b, "data", "tag")),
    ("extra-nested-key", lambda b: _extra_nested(b, "metrics", "precision")),
    ("empty-tag", lambda b: _set_nested(b, "data", "tag", "")),
    ("non-string-tag", lambda b: _set_nested(b, "data", "tag", 42)),
    ("empty-prepared-commit", lambda b: _set_nested(b, "code", "prepared_commit", "")),
    ("empty-evaluation-commit", lambda b: _set_nested(b, "code", "evaluation_commit", "")),
    ("malformed-split-sha", lambda b: _set_nested(b, "data", "split_sha256", "0" * 63)),
    ("uppercase-split-sha", lambda b: _set_nested(b, "data", "split_sha256", "A" * 64)),
    ("malformed-predictions-sha", lambda b: _set(b, "predictions_sha256", "z" * 64)),
    ("non-string-predictions-sha", lambda b: _set(b, "predictions_sha256", 123)),
]


class TestInvalidPayloads:
    @pytest.mark.parametrize(
        ("name", "mutator"), MALFORMED, ids=[name for name, _ in MALFORMED]
    )
    def test_invalid_payload_rejected(self, tmp_path, name, mutator):
        payload = mutator(_base_payload(tmp_path))
        with pytest.raises(T1MetricsError):
            write_t1_metrics(payload, tmp_path / "metrics.json")

    def test_invalid_payload_rejected_before_any_creation(self, tmp_path):
        output = tmp_path / "metrics.json"
        payload = _base_payload(tmp_path)
        payload["format_version"] = 2
        with pytest.raises(T1MetricsError):
            write_t1_metrics(payload, output)
        assert not output.exists()
        assert _temp_names(tmp_path) == []
        assert sorted(p.name for p in tmp_path.iterdir()) == ["run-dir"]


class TestRoundTrip:
    def test_write_read_json_round_trip(self, tmp_path):
        payload = _base_payload(tmp_path)
        payload_before = copy.deepcopy(payload)
        output = tmp_path / "metrics.json"
        write_t1_metrics(payload, output)

        text = output.read_text(encoding="utf-8")
        assert text.endswith("\n")
        assert "\\u" not in text
        assert text == (
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        assert json.loads(text) == payload
        assert payload == payload_before
        assert output.is_file()
        assert sorted(p.name for p in tmp_path.iterdir()) == ["metrics.json", "run-dir"]
        assert _temp_names(tmp_path) == []


class TestDestinationProtection:
    @pytest.mark.parametrize("kind", ["file", "directory", "dangling-symlink"])
    def test_existing_destination_preserved(self, tmp_path, kind):
        payload = _base_payload(tmp_path)
        output = tmp_path / "metrics.json"
        if kind == "file":
            output.write_text("precious\n", encoding="utf-8")
        elif kind == "directory":
            output.mkdir()
        else:
            try:
                output.symlink_to(tmp_path / "missing-target")
            except (OSError, NotImplementedError):
                pytest.skip("Symlink creation is unavailable in this environment")
        with pytest.raises(T1MetricsError) as exc_info:
            write_t1_metrics(payload, output)
        assert str(output) in str(exc_info.value)
        if kind == "file":
            assert output.read_text(encoding="utf-8") == "precious\n"
        elif kind == "directory":
            assert output.is_dir()
        else:
            assert output.is_symlink()
            assert not (tmp_path / "missing-target").exists()
        assert _temp_names(tmp_path) == []
        assert sorted(p.name for p in tmp_path.iterdir()) == ["metrics.json", "run-dir"]

    def test_missing_parent_not_created(self, tmp_path):
        payload = _base_payload(tmp_path)
        output = tmp_path / "missing" / "metrics.json"
        with pytest.raises(T1MetricsError):
            write_t1_metrics(payload, output)
        assert not (tmp_path / "missing").exists()
        assert _temp_names(tmp_path) == []


class FailingWriteStream:
    """Wraps a real text stream; ``write`` raises ``OSError`` immediately."""

    def __init__(self, stream):
        self._stream = stream

    def write(self, data):
        raise OSError("simulated disk full")

    def close(self):
        self._stream.close()

    def __getattr__(self, name):
        return getattr(self._stream, name)


class WriteAndCloseFailureStream:
    """Wraps a real text stream; both ``write`` and ``close`` raise ``OSError``."""

    def __init__(self, stream):
        self._stream = stream

    def write(self, data):
        raise OSError("simulated disk full")

    def close(self):
        self._stream.close()
        raise OSError("simulated close failure")

    def __getattr__(self, name):
        return getattr(self._stream, name)


class ClosingFailureStream:
    """Wraps a real text stream; ``close`` raises a controlled ``OSError``."""

    def __init__(self, stream, close_error):
        self._stream = stream
        self._close_error = close_error

    def write(self, data):
        return self._stream.write(data)

    def close(self):
        self._stream.close()
        raise self._close_error

    def __getattr__(self, name):
        return getattr(self._stream, name)


class TestWriteFailures:
    def test_temporary_creation_failure(self, tmp_path, monkeypatch):
        payload = _base_payload(tmp_path)
        output = tmp_path / "metrics.json"

        def failing_mkstemp(**kwargs):
            raise OSError("simulated temp creation failure")

        monkeypatch.setattr(metrics_io.tempfile, "mkstemp", failing_mkstemp)
        with pytest.raises(T1MetricsError) as exc_info:
            write_t1_metrics(payload, output)
        error = exc_info.value
        assert str(output) in str(error)
        assert isinstance(error.__cause__, OSError)
        assert not output.exists()
        assert _temp_names(tmp_path) == []

    def test_fdopen_failure_closes_descriptor_and_cleans_temp(self, tmp_path, monkeypatch):
        payload = _base_payload(tmp_path)
        output = tmp_path / "metrics.json"
        real_mkstemp = metrics_io.tempfile.mkstemp
        real_close = os.close
        descriptor, name = real_mkstemp(
            prefix=".vepkar-t1-metrics-", suffix=".tmp", dir=str(tmp_path)
        )
        closed = []

        def failing_fdopen(*args, **kwargs):
            raise OSError("simulated fdopen failure")

        def record_close(fd):
            closed.append(fd)
            real_close(fd)

        monkeypatch.setattr(
            metrics_io.tempfile, "mkstemp", lambda **kwargs: (descriptor, name)
        )
        monkeypatch.setattr(metrics_io.os, "fdopen", failing_fdopen)
        monkeypatch.setattr(metrics_io.os, "close", record_close)
        with pytest.raises(T1MetricsError) as exc_info:
            write_t1_metrics(payload, output)
        error = exc_info.value
        assert str(output) in str(error)
        assert isinstance(error.__cause__, OSError)
        assert descriptor in closed
        assert not output.exists()
        assert sorted(p.name for p in tmp_path.iterdir()) == ["run-dir"]

    def test_write_failure_cleans_temporary(self, tmp_path, monkeypatch):
        payload = _base_payload(tmp_path)
        output = tmp_path / "metrics.json"
        real_fdopen = os.fdopen

        def failing_fdopen(descriptor, *args, **kwargs):
            return FailingWriteStream(real_fdopen(descriptor, *args, **kwargs))

        monkeypatch.setattr(metrics_io.os, "fdopen", failing_fdopen)
        with pytest.raises(T1MetricsError) as exc_info:
            write_t1_metrics(payload, output)
        error = exc_info.value
        assert str(output) in str(error)
        assert isinstance(error.__cause__, OSError)
        assert "simulated disk full" in str(error.__cause__)
        assert not output.exists()
        assert _temp_names(tmp_path) == []

    def test_write_and_close_failure_preserves_write_cause(self, tmp_path, monkeypatch):
        payload = _base_payload(tmp_path)
        output = tmp_path / "metrics.json"
        real_fdopen = os.fdopen

        def failing_fdopen(descriptor, *args, **kwargs):
            return WriteAndCloseFailureStream(real_fdopen(descriptor, *args, **kwargs))

        monkeypatch.setattr(metrics_io.os, "fdopen", failing_fdopen)
        with pytest.raises(T1MetricsError) as exc_info:
            write_t1_metrics(payload, output)
        error = exc_info.value
        assert isinstance(error.__cause__, OSError)
        assert "simulated disk full" in str(error.__cause__)
        assert any("simulated close failure" in note for note in error.__notes__)
        assert not output.exists()
        assert _temp_names(tmp_path) == []

    def test_close_failure_after_successful_write(self, tmp_path, monkeypatch):
        payload = _base_payload(tmp_path)
        output = tmp_path / "metrics.json"
        real_fdopen = os.fdopen
        link_calls = []

        def closing_fail_fdopen(descriptor, *args, **kwargs):
            return ClosingFailureStream(
                real_fdopen(descriptor, *args, **kwargs),
                OSError("simulated close failure"),
            )

        original_link = os.link

        def recording_link(source, destination):
            link_calls.append(Path(source))
            return original_link(source, destination)

        monkeypatch.setattr(metrics_io.os, "fdopen", closing_fail_fdopen)
        monkeypatch.setattr(metrics_io.os, "link", recording_link)
        with pytest.raises(T1MetricsError) as exc_info:
            write_t1_metrics(payload, output)
        error = exc_info.value
        assert isinstance(error.__cause__, OSError)
        assert "simulated close failure" in str(error.__cause__)
        assert link_calls == []
        assert not output.exists()
        assert _temp_names(tmp_path) == []

    def test_link_failure_cleans_temporary(self, tmp_path, monkeypatch):
        payload = _base_payload(tmp_path)
        output = tmp_path / "metrics.json"

        def failing_link(source, destination):
            raise OSError("simulated publish failure")

        monkeypatch.setattr(metrics_io.os, "link", failing_link)
        with pytest.raises(T1MetricsError) as exc_info:
            write_t1_metrics(payload, output)
        error = exc_info.value
        assert str(output) in str(error)
        assert isinstance(error.__cause__, OSError)
        assert not output.exists()
        assert _temp_names(tmp_path) == []

    def test_primary_write_failure_with_cleanup_failure(self, tmp_path, monkeypatch):
        payload = _base_payload(tmp_path)
        output = tmp_path / "metrics.json"
        real_fdopen = os.fdopen
        original_unlink = Path.unlink

        def failing_fdopen(descriptor, *args, **kwargs):
            return FailingWriteStream(real_fdopen(descriptor, *args, **kwargs))

        def failing_unlink(self, *args, **kwargs):
            if (
                self.parent == tmp_path
                and self.name.startswith(".vepkar-t1-metrics-")
                and self.name.endswith(".tmp")
            ):
                raise OSError("simulated unlink failure")
            return original_unlink(self, *args, **kwargs)

        monkeypatch.setattr(metrics_io.os, "fdopen", failing_fdopen)
        monkeypatch.setattr(Path, "unlink", failing_unlink)
        with pytest.raises(T1MetricsError) as exc_info:
            write_t1_metrics(payload, output)
        error = exc_info.value
        assert isinstance(error.__cause__, OSError)
        assert "simulated disk full" in str(error.__cause__)
        assert any("simulated unlink failure" in note for note in error.__notes__)
        assert not output.exists()
        temp_files = _temp_names(tmp_path)
        assert len(temp_files) == 1
        for temp in temp_files:
            original_unlink(temp)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["run-dir"]


class TestPublication:
    def test_competing_destination_keeps_foreign_bytes(self, tmp_path, monkeypatch):
        payload = _base_payload(tmp_path)
        output = tmp_path / "metrics.json"
        original_link = os.link

        def competing_link(source, destination):
            with open(destination, "wb") as fh:
                fh.write(b"foreign bytes")
            return original_link(source, destination)

        monkeypatch.setattr(metrics_io.os, "link", competing_link)
        with pytest.raises(T1MetricsError) as exc_info:
            write_t1_metrics(payload, output)
        error = exc_info.value
        assert str(output) in str(error)
        assert isinstance(error.__cause__, OSError)
        assert output.read_bytes() == b"foreign bytes"
        assert _temp_names(tmp_path) == []
        assert sorted(p.name for p in tmp_path.iterdir()) == ["metrics.json", "run-dir"]

    def test_post_publication_unlink_failure(self, tmp_path, monkeypatch):
        payload = _base_payload(tmp_path)
        output = tmp_path / "metrics.json"
        original_link = os.link
        original_unlink = Path.unlink
        published_temps = []

        def recording_link(source, destination):
            published_temps.append(Path(source))
            return original_link(source, destination)

        def failing_unlink(self, *args, **kwargs):
            if self in published_temps:
                raise OSError("simulated permission denied")
            return original_unlink(self, *args, **kwargs)

        monkeypatch.setattr(metrics_io.os, "link", recording_link)
        monkeypatch.setattr(Path, "unlink", failing_unlink)
        with pytest.raises(T1MetricsPublishedCleanupError) as exc_info:
            write_t1_metrics(payload, output)
        error = exc_info.value
        assert "published" in str(error).lower()
        assert error.output_path == output
        assert len(published_temps) == 1
        assert error.temp_path == published_temps[0]
        assert error.__cause__ is not None
        assert json.loads(output.read_text(encoding="utf-8")) == payload
        for temp in published_temps:
            if temp.exists():
                original_unlink(temp)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["metrics.json", "run-dir"]