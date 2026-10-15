"""Offline tests for writing, validating, and publishing dev predictions."""

import csv
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from benchmark_context import BenchmarkContext  # noqa: E402
from core.instances import Candidate, Instance  # noqa: E402
from core.predictions import (  # noqa: E402
    BenchmarkIntegrityError,
    PredictionRow,
    PredictionValidation,
    validate_predictions,
)
import t1_features.prediction_io as prediction_io  # noqa: E402
from t1_features.prediction_io import (  # noqa: E402
    T1PredictionPublishedCleanupError,
    T1PredictionWriteError,
    write_dev_predictions,
)

SPLIT_ROWS = (
    ("krl", 1, "train"),
    ("krl", 2, "dev"),
    ("krl", 3, "test"),
)

TRAIN = {
    501: (
        Candidate(9001, "SG+NOM"),
        Candidate(9001, "SG+ACC"),
        Candidate(9002, "PL+GEN"),
    ),
}
DEV = {
    601: (
        Candidate(2001, "V+IND+PRS,AUX"),
        Candidate(2002, "V+IND+PST+äöü"),
    ),
}
TEST = {
    701: (Candidate(3001, "PRON"), Candidate(3002, "N")),
}

DEV_ROWS = (
    PredictionRow(601, 2001, "V+IND+PRS,AUX", 1, 0.1),
    PredictionRow(601, 2002, "V+IND+PST+äöü", 2, -0.2),
)

EXPECTED_CSV = (
    "word_id,wordform_id,gramset,rank,score\n"
    "601,2001,\"V+IND+PRS,AUX\",1,0.1\n"
    "601,2002,V+IND+PST+äöü,2,-0.2\n"
)


def _instance(word_id, text_id, candidates):
    return Instance(
        language="krl",
        word_id=word_id,
        sentence_id=word_id,
        text_id=text_id,
        word=f"w{word_id}",
        word_number=1,
        sentence_xml="<s/>",
        candidates=candidates,
        gold_analysis=candidates[0],
    )


def _benchmark():
    return BenchmarkContext(
        tag="test-tag",
        split_path=Path("unused-splits.csv"),
        instances=(
            _instance(501, 1, TRAIN[501]),
            _instance(601, 2, DEV[601]),
            _instance(701, 3, TEST[701]),
        ),
        split_rows=SPLIT_ROWS,
    )


def _read_csv(path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.reader(fh))


class CountingIterable:
    def __init__(self, rows):
        self._rows = list(rows)
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return iter(self._rows)


class TestExceptionTypes:
    def test_inheritance(self):
        assert issubclass(T1PredictionWriteError, ValueError)
        assert issubclass(T1PredictionPublishedCleanupError, T1PredictionWriteError)
        assert T1PredictionWriteError is not ValueError

    def test_validation_defaults_to_none(self):
        error = T1PredictionWriteError("message")
        assert error.validation is None

    def test_validation_attached(self):
        validation = PredictionValidation(
            is_valid=False,
            split="dev",
            expected_word_count=1,
            expected_candidate_count=2,
            predicted_word_count=1,
            predicted_candidate_count=1,
            errors={"missing candidate": ("word_id=601; candidate=(2002, 'V+IND+PST+äöü')",)},
            error_counts={"missing candidate": 1},
        )
        error = T1PredictionWriteError("message", validation=validation)
        assert error.validation is validation

    def test_cleanup_error_reports_both_paths(self):
        output = Path("out.csv")
        temp = Path("temp.tmp")
        error = T1PredictionPublishedCleanupError(output, temp)
        assert error.output_path == output
        assert error.temp_path == temp
        message = str(error)
        assert "published" in message.lower()
        assert "removal" in message.lower()
        assert str(output) in message
        assert str(temp) in message


class TestSuccessfulWrite:
    def test_generator_input_exact_csv_and_counts(self, tmp_path):
        output = tmp_path / "dev-predictions.csv"
        benchmark = _benchmark()
        rows = (row for row in DEV_ROWS)
        validation = write_dev_predictions(rows, output, benchmark=benchmark)
        assert validation.is_valid
        assert validation.split == "dev"
        assert validation.expected_word_count == 1
        assert validation.expected_candidate_count == 2
        assert validation.predicted_word_count == 1
        assert validation.predicted_candidate_count == 2
        assert validation.errors == {}
        assert output.read_text(encoding="utf-8") == EXPECTED_CSV
        assert _read_csv(output) == [
            ["word_id", "wordform_id", "gramset", "rank", "score"],
            ["601", "2001", "V+IND+PRS,AUX", "1", "0.1"],
            ["601", "2002", "V+IND+PST+äöü", "2", "-0.2"],
        ]
        assert sorted(p.name for p in tmp_path.iterdir()) == ["dev-predictions.csv"]

    def test_unicode_quoting_no_rounding_inputs_unchanged(self, tmp_path):
        output = tmp_path / "dev-predictions.csv"
        benchmark = _benchmark()
        rows = list(DEV_ROWS)
        original_rows = list(rows)
        original_instances = benchmark.instances
        original_split_rows = benchmark.split_rows
        validation = write_dev_predictions(rows, output, benchmark=benchmark)
        assert validation.is_valid
        assert output.read_text(encoding="utf-8") == EXPECTED_CSV
        assert rows == original_rows
        assert benchmark.instances == original_instances
        assert benchmark.split_rows == original_split_rows

    def test_rows_consumed_exactly_once(self, tmp_path):
        output = tmp_path / "dev-predictions.csv"
        rows = CountingIterable(DEV_ROWS)
        write_dev_predictions(rows, output, benchmark=_benchmark())
        assert rows.iterations == 1
        assert output.read_text(encoding="utf-8") == EXPECTED_CSV


class TestInvalidRows:
    @pytest.mark.parametrize(
        ("rows", "expected_category"),
        [
            ((DEV_ROWS[0],), "missing candidate"),
            (
                (
                    PredictionRow(501, 9001, "SG+NOM", 1, 0.9),
                    PredictionRow(501, 9001, "SG+ACC", 2, 0.5),
                    PredictionRow(501, 9002, "PL+GEN", 3, 0.1),
                ),
                "wrong split",
            ),
        ],
    )
    def test_invalid_rows_rejected_with_validation(
        self, tmp_path, rows, expected_category
    ):
        output = tmp_path / "dev-predictions.csv"
        with pytest.raises(T1PredictionWriteError) as exc_info:
            write_dev_predictions(rows, output, benchmark=_benchmark())
        error = exc_info.value
        assert error.validation is not None
        assert not error.validation.is_valid
        assert expected_category in error.validation.errors
        assert str(output) in str(error)
        assert not output.exists()
        assert sorted(p.name for p in tmp_path.iterdir()) == []


class TestZeroExpectedDev:
    def test_rejected_without_publication(self, tmp_path):
        output = tmp_path / "dev-predictions.csv"
        benchmark = BenchmarkContext(
            tag="test-tag",
            split_path=Path("unused-splits.csv"),
            instances=(
                _instance(501, 1, TRAIN[501]),
                _instance(701, 3, TEST[701]),
            ),
            split_rows=SPLIT_ROWS,
        )
        with pytest.raises(T1PredictionWriteError) as exc_info:
            write_dev_predictions([], output, benchmark=benchmark)
        assert "dev" in str(exc_info.value).lower()
        assert not output.exists()
        assert sorted(p.name for p in tmp_path.iterdir()) == []


class TestDestinationProtection:
    @pytest.mark.parametrize("kind", ["file", "directory", "dangling-symlink"])
    def test_existing_destination_preserved(self, tmp_path, kind):
        output = tmp_path / "dev-predictions.csv"
        if kind == "file":
            output.write_text("precious\n", encoding="utf-8")
        elif kind == "directory":
            output.mkdir()
        else:
            output.symlink_to(tmp_path / "missing-target")
        with pytest.raises(T1PredictionWriteError):
            write_dev_predictions(DEV_ROWS, output, benchmark=_benchmark())
        if kind == "file":
            assert output.read_text(encoding="utf-8") == "precious\n"
        elif kind == "directory":
            assert output.is_dir()
        else:
            assert output.is_symlink()
            assert not (tmp_path / "missing-target").exists()
        assert sorted(p.name for p in tmp_path.iterdir()) == ["dev-predictions.csv"]

    def test_missing_parent_not_created(self, tmp_path):
        output = tmp_path / "missing" / "dev-predictions.csv"
        with pytest.raises(T1PredictionWriteError):
            write_dev_predictions(DEV_ROWS, output, benchmark=_benchmark())
        assert not (tmp_path / "missing").exists()


class FailingWriteStream:
    """Wraps a real text stream; the second write succeeds then raises OSError."""

    def __init__(self, stream):
        self._stream = stream
        self._writes = 0

    def write(self, data):
        self._writes += 1
        self._stream.write(data)
        if self._writes == 2:
            raise OSError("simulated disk full")
        return len(data)

    def close(self):
        self._stream.close()

    def __getattr__(self, name):
        return getattr(self._stream, name)


class TestWriteFailures:
    def test_temporary_creation_failure(self, tmp_path, monkeypatch):
        output = tmp_path / "dev-predictions.csv"

        def failing_mkstemp(**kwargs):
            raise OSError("simulated temp creation failure")

        monkeypatch.setattr(prediction_io.tempfile, "mkstemp", failing_mkstemp)
        with pytest.raises(T1PredictionWriteError) as exc_info:
            write_dev_predictions(DEV_ROWS, output, benchmark=_benchmark())
        error = exc_info.value
        assert str(output) in str(error)
        assert isinstance(error.__cause__, OSError)
        assert "simulated temp creation failure" in str(error.__cause__)
        assert not output.exists()
        assert list(tmp_path.iterdir()) == []

    def test_write_failure_cleans_temporary(self, tmp_path, monkeypatch):
        output = tmp_path / "dev-predictions.csv"
        real_fdopen = os.fdopen

        def failing_fdopen(descriptor, *args, **kwargs):
            return FailingWriteStream(real_fdopen(descriptor, *args, **kwargs))

        monkeypatch.setattr(prediction_io.os, "fdopen", failing_fdopen)
        with pytest.raises(T1PredictionWriteError) as exc_info:
            write_dev_predictions(DEV_ROWS, output, benchmark=_benchmark())
        error = exc_info.value
        assert str(output) in str(error)
        assert isinstance(error.__cause__, OSError)
        assert "simulated disk full" in str(error.__cause__)
        assert not output.exists()
        assert sorted(p.name for p in tmp_path.iterdir()) == []


class TestValidatorFailures:
    def test_benchmark_integrity_error_wrapped(self, tmp_path):
        output = tmp_path / "dev-predictions.csv"
        duplicate = (
            _instance(601, 2, DEV[601]),
            Instance(
                language="vep",
                word_id=601,
                sentence_id=999,
                text_id=999,
                word="duplicate",
                word_number=1,
                sentence_xml="<s/>",
                candidates=DEV[601],
                gold_analysis=DEV[601][0],
            ),
        )
        benchmark = BenchmarkContext(
            tag="test-tag",
            split_path=Path("unused-splits.csv"),
            instances=duplicate,
            split_rows=SPLIT_ROWS,
        )
        with pytest.raises(T1PredictionWriteError) as exc_info:
            write_dev_predictions(DEV_ROWS, output, benchmark=benchmark)
        error = exc_info.value
        assert isinstance(error.__cause__, BenchmarkIntegrityError)
        assert "duplicate word_id" in str(error.__cause__)
        assert str(output) in str(error)
        assert not output.exists()
        assert sorted(p.name for p in tmp_path.iterdir()) == []


class TestPublication:
    def test_competing_destination_keeps_foreign_bytes(self, tmp_path, monkeypatch):
        output = tmp_path / "dev-predictions.csv"
        original_link = os.link

        def competing_link(source, destination):
            with open(destination, "wb") as fh:
                fh.write(b"foreign bytes")
            return original_link(source, destination)

        monkeypatch.setattr(prediction_io.os, "link", competing_link)
        with pytest.raises(T1PredictionWriteError) as exc_info:
            write_dev_predictions(DEV_ROWS, output, benchmark=_benchmark())
        error = exc_info.value
        assert str(output) in str(error)
        assert isinstance(error.__cause__, OSError)
        assert output.read_bytes() == b"foreign bytes"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["dev-predictions.csv"]

    def test_post_publication_unlink_failure(self, tmp_path, monkeypatch):
        output = tmp_path / "dev-predictions.csv"
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

        monkeypatch.setattr(prediction_io.os, "link", recording_link)
        monkeypatch.setattr(Path, "unlink", failing_unlink)
        with pytest.raises(T1PredictionPublishedCleanupError) as exc_info:
            write_dev_predictions(DEV_ROWS, output, benchmark=_benchmark())
        error = exc_info.value
        assert "published" in str(error).lower()
        assert str(output) in str(error)
        assert len(published_temps) == 1
        assert str(published_temps[0]) in str(error)
        assert error.output_path == output
        assert error.temp_path == published_temps[0]
        assert error.__cause__ is not None
        assert output.read_text(encoding="utf-8") == EXPECTED_CSV
        validation = validate_predictions(
            predictions_path=output,
            split="dev",
            instances=_benchmark().instances,
            split_rows=_benchmark().split_rows,
        )
        assert validation.is_valid
        for temp in published_temps:
            if temp.exists():
                original_unlink(temp)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["dev-predictions.csv"]

    def test_primary_write_failure_with_cleanup_failure(self, tmp_path, monkeypatch):
        output = tmp_path / "dev-predictions.csv"
        real_fdopen = os.fdopen
        original_unlink = Path.unlink

        def failing_fdopen(descriptor, *args, **kwargs):
            return FailingWriteStream(real_fdopen(descriptor, *args, **kwargs))

        def failing_unlink(self, *args, **kwargs):
            if (
                self.parent == tmp_path
                and self.name.startswith(".vepkar-t1-predictions-")
                and self.name.endswith(".tmp")
            ):
                raise OSError("simulated unlink failure")
            return original_unlink(self, *args, **kwargs)

        monkeypatch.setattr(prediction_io.os, "fdopen", failing_fdopen)
        monkeypatch.setattr(Path, "unlink", failing_unlink)
        with pytest.raises(T1PredictionWriteError) as exc_info:
            write_dev_predictions(DEV_ROWS, output, benchmark=_benchmark())
        error = exc_info.value
        assert isinstance(error.__cause__, OSError)
        assert "simulated disk full" in str(error.__cause__)
        assert any("simulated unlink failure" in note for note in error.__notes__)
        assert not output.exists()
        temp_files = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
        assert len(temp_files) == 1
        for temp in temp_files:
            original_unlink(temp)
        assert sorted(p.name for p in tmp_path.iterdir()) == []
