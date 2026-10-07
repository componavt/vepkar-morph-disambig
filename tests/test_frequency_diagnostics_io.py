"""Offline tests for reading and verifying frequency-baseline prediction CSVs."""

import csv
import sys
from dataclasses import astuple
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from core.diagnostics import FrequencyDiagnostic  # noqa: E402
from core.frequency import FrequencyPrediction  # noqa: E402
from core.predictions import (  # noqa: E402
    PREDICTION_HEADER,
    PredictionCsvParseError,
    PredictionFileReadError,
)
from frequency_diagnostics_io import (  # noqa: E402
    FrequencyBaselineMismatchError,
    load_verified_frequency_predictions,
    write_frequency_diagnostics,
)


def _expected_rows():
    return (
        FrequencyPrediction(601, 2001, "V+IND+PRS", 1, 3),
        FrequencyPrediction(601, 2002, "V+IND+PST", 2, 1),
        FrequencyPrediction(602, 1001, "N+SG+NOM", 1, 0),
        FrequencyPrediction(602, 1002, "N+SG+GEN", 2, 0),
    )


def _as_csv_rows(rows):
    return [(r.word_id, r.wordform_id, r.gramset, r.rank, r.score) for r in rows]


def _write_csv(tmp_path, rows, header=PREDICTION_HEADER, name="predictions.csv"):
    path = tmp_path / name
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def test_exact_match_integer_output(tmp_path):
    expected = _expected_rows()
    path = _write_csv(tmp_path, _as_csv_rows(expected))
    loaded = load_verified_frequency_predictions(path, expected)
    assert loaded == expected
    assert all(isinstance(row.score, int) for row in loaded)


def test_reordered_rows_keep_input_order(tmp_path):
    expected = _expected_rows()
    path = _write_csv(tmp_path, list(reversed(_as_csv_rows(expected))))
    loaded = load_verified_frequency_predictions(path, expected)
    assert [row.gramset for row in loaded] == [
        "N+SG+GEN",
        "N+SG+NOM",
        "V+IND+PST",
        "V+IND+PRS",
    ]
    assert set(loaded) == set(expected)


def test_single_pass_expected_iterable(tmp_path):
    expected_iter = (
        FrequencyPrediction(r.word_id, r.wordform_id, r.gramset, r.rank, r.score)
        for r in _expected_rows()
    )
    path = _write_csv(tmp_path, _as_csv_rows(_expected_rows()))
    loaded = load_verified_frequency_predictions(path, expected_iter)
    assert loaded == _expected_rows()


@pytest.mark.parametrize("text", ["7", "7.0", "7e0", "7.00", "+7"])
def test_integral_score_representations_accepted(tmp_path, text):
    expected = (FrequencyPrediction(601, 2001, "A", 1, 7),)
    path = _write_csv(tmp_path, [(601, 2001, "A", 1, text)])
    loaded = load_verified_frequency_predictions(path, expected)
    assert loaded == expected


@pytest.mark.parametrize("text", ["7.5", "-1", "NaN", "Infinity", "-Infinity", "nan"])
def test_non_integral_or_nonfinite_scores_rejected(tmp_path, text):
    expected = (FrequencyPrediction(601, 2001, "A", 1, 7),)
    path = _write_csv(tmp_path, [(601, 2001, "A", 1, text)])
    with pytest.raises(FrequencyBaselineMismatchError):
        load_verified_frequency_predictions(path, expected)


def test_near_integer_decimal_rejected_without_rounding(tmp_path):
    expected = (FrequencyPrediction(601, 2001, "A", 1, 7),)
    path = _write_csv(
        tmp_path, [(601, 2001, "A", 1, "7.0000000000000000001")]
    )
    with pytest.raises(FrequencyBaselineMismatchError):
        load_verified_frequency_predictions(path, expected)


def test_score_mismatch_rejected(tmp_path):
    expected = (FrequencyPrediction(601, 2001, "A", 1, 7),)
    path = _write_csv(tmp_path, [(601, 2001, "A", 1, 8)])
    with pytest.raises(FrequencyBaselineMismatchError):
        load_verified_frequency_predictions(path, expected)


def test_rank_mismatch_rejected(tmp_path):
    expected = (FrequencyPrediction(601, 2001, "A", 1, 7),)
    path = _write_csv(tmp_path, [(601, 2001, "A", 2, 7)])
    with pytest.raises(FrequencyBaselineMismatchError):
        load_verified_frequency_predictions(path, expected)


def test_missing_identity_rejected(tmp_path):
    expected = _expected_rows()
    path = _write_csv(tmp_path, _as_csv_rows(expected[:-1]))
    with pytest.raises(FrequencyBaselineMismatchError):
        load_verified_frequency_predictions(path, expected)


def test_unexpected_identity_rejected(tmp_path):
    expected = _expected_rows()
    rows = _as_csv_rows(expected) + [(777, 9999, "X", 1, 0)]
    path = _write_csv(tmp_path, rows)
    with pytest.raises(FrequencyBaselineMismatchError):
        load_verified_frequency_predictions(path, expected)


def test_duplicate_identity_rejected(tmp_path):
    expected = _expected_rows()
    rows = _as_csv_rows(expected) + [(601, 2001, "V+IND+PRS", 1, 3)]
    path = _write_csv(tmp_path, rows)
    with pytest.raises(FrequencyBaselineMismatchError):
        load_verified_frequency_predictions(path, expected)


def test_same_wordform_different_gramset_distinguished(tmp_path):
    expected = (
        FrequencyPrediction(601, 2001, "SG+NOM", 1, 2),
        FrequencyPrediction(601, 2001, "SG+ACC", 2, 1),
    )
    path = _write_csv(
        tmp_path, [(601, 2001, "SG+NOM", 1, 2), (601, 2001, "SG+ACC", 2, 1)]
    )
    assert load_verified_frequency_predictions(path, expected) == expected


def test_same_gramset_different_wordform_distinguished(tmp_path):
    expected = (
        FrequencyPrediction(601, 2001, "N+SG+NOM", 1, 2),
        FrequencyPrediction(601, 2002, "N+SG+NOM", 2, 1),
    )
    path = _write_csv(
        tmp_path, [(601, 2001, "N+SG+NOM", 1, 2), (601, 2002, "N+SG+NOM", 2, 1)]
    )
    assert load_verified_frequency_predictions(path, expected) == expected


def test_header_only_csv_with_empty_expected_returns_empty(tmp_path):
    path = _write_csv(tmp_path, [])
    assert load_verified_frequency_predictions(path, []) == ()


def test_empty_expected_with_data_rows_rejected(tmp_path):
    path = _write_csv(tmp_path, [(601, 2001, "A", 1, 0)])
    with pytest.raises(FrequencyBaselineMismatchError):
        load_verified_frequency_predictions(path, [])


def test_empty_file_rejected_as_parse_error(tmp_path):
    path = tmp_path / "predictions.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(PredictionCsvParseError):
        load_verified_frequency_predictions(path, [])


def test_unreadable_path_raises_file_read_error(tmp_path):
    path = tmp_path / "missing.csv"
    with pytest.raises(PredictionFileReadError):
        load_verified_frequency_predictions(path, [])


def test_malformed_csv_raises_parse_error(tmp_path):
    path = tmp_path / "predictions.csv"
    path.write_text(
        'word_id,wordform_id,gramset,rank,score\n601,2001,"unterminated\n',
        encoding="utf-8",
    )
    with pytest.raises(PredictionCsvParseError):
        load_verified_frequency_predictions(path, [])


def test_unparseable_numeric_fields_raise_parse_error(tmp_path):
    expected = _expected_rows()
    path = _write_csv(tmp_path, [(601, 2001, "A", "x", 3)])
    with pytest.raises(PredictionCsvParseError):
        load_verified_frequency_predictions(path, expected)
    path = _write_csv(tmp_path, [(601, 2001, "A", 1, "abc")])
    with pytest.raises(PredictionCsvParseError):
        load_verified_frequency_predictions(path, expected)


def test_wrong_header_raises_parse_error(tmp_path):
    path = _write_csv(
        tmp_path, [(601, 2001, "A", 1, 3)], header=("a", "b", "c", "d", "e")
    )
    with pytest.raises(PredictionCsvParseError):
        load_verified_frequency_predictions(path, [])


def test_wrong_row_width_raises_parse_error(tmp_path):
    expected = _expected_rows()
    path = _write_csv(tmp_path, [(601, 2001, "A", 1)])
    with pytest.raises(PredictionCsvParseError):
        load_verified_frequency_predictions(path, expected)


def test_expected_inputs_and_source_file_unchanged(tmp_path):
    expected = _expected_rows()
    before_expected = tuple(expected)
    path = _write_csv(tmp_path, _as_csv_rows(expected))
    before_bytes = path.read_bytes()
    loaded = load_verified_frequency_predictions(path, expected)
    assert tuple(expected) == before_expected
    assert path.read_bytes() == before_bytes
    assert loaded == expected


DIAGNOSTICS_HEADER = [
    "language",
    "word_id",
    "word",
    "candidate_count",
    "gold_wordform_id",
    "gold_gramset",
    "top1_wordform_id",
    "top1_gramset",
    "gold_rank",
    "gold_train_frequency",
    "top1_train_frequency",
]


def _diagnostic_rows():
    return (
        FrequencyDiagnostic(
            language="krl",
            word_id=601,
            word="kala",
            candidate_count=3,
            gold_wordform_id=2001,
            gold_gramset="N+SG+NOM",
            top1_wordform_id=2002,
            top1_gramset="N+SG+GEN",
            gold_rank=2,
            gold_train_frequency=5,
            top1_train_frequency=8,
        ),
        FrequencyDiagnostic(
            language="vep",
            word_id=602,
            word="vezi",
            candidate_count=2,
            gold_wordform_id=1001,
            gold_gramset="V+IND+PRS",
            top1_wordform_id=1001,
            top1_gramset="V+IND+PRS",
            gold_rank=1,
            gold_train_frequency=0,
            top1_train_frequency=3,
        ),
    )


def _read_csv(path: Path) -> tuple[list, list[list]]:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, strict=True)
        header = next(reader)
        rows = list(reader)
    return header, rows


def test_write_diagnostics_header_and_rows_unchanged(tmp_path):
    rows = _diagnostic_rows()
    path = tmp_path / "diagnostics.csv"
    count = write_frequency_diagnostics(path, rows)
    assert count == 2
    header, raw = _read_csv(path)
    assert header == DIAGNOSTICS_HEADER
    assert raw == [list(map(str, astuple(row))) for row in rows]


def test_write_diagnostics_empty_rows_header_only(tmp_path):
    path = tmp_path / "diagnostics.csv"
    count = write_frequency_diagnostics(path, ())
    assert count == 0
    header, raw = _read_csv(path)
    assert header == DIAGNOSTICS_HEADER
    assert raw == []


def test_write_diagnostics_single_pass_iterable(tmp_path):
    path = tmp_path / "diagnostics.csv"
    count = write_frequency_diagnostics(path, (row for row in _diagnostic_rows()))
    assert count == 2
    header, raw = _read_csv(path)
    assert len(raw) == 2


def test_write_diagnostics_preserves_unicode_commas_quotes(tmp_path):
    rows = (
        FrequencyDiagnostic(
            language="krl",
            word_id=1,
            word='Šuuruš’a, "päiv"',
            candidate_count=2,
            gold_wordform_id=10,
            gold_gramset="N+SG,NOM",
            top1_wordform_id=10,
            top1_gramset="N+SG,NOM",
            gold_rank=1,
            gold_train_frequency=1,
            top1_train_frequency=1,
        ),
    )
    path = tmp_path / "diagnostics.csv"
    write_frequency_diagnostics(path, rows)
    header, raw = _read_csv(path)
    assert header == DIAGNOSTICS_HEADER
    assert raw[0][2] == 'Šuuruš’a, "päiv"'
    assert raw[0][5] == "N+SG,NOM"
    assert raw[0][7] == "N+SG,NOM"
    content = path.read_text(encoding="utf-8")
    assert content.count('"N+SG,NOM"') == 2


def test_write_diagnostics_rejects_existing_file(tmp_path):
    rows = _diagnostic_rows()
    path = tmp_path / "diagnostics.csv"
    original = "keep\n"
    path.write_text(original, encoding="utf-8")
    with pytest.raises(OSError):
        write_frequency_diagnostics(path, rows)
    assert path.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob(".vepkar-diagnostics-*.tmp"))


def test_write_diagnostics_rejects_directory(tmp_path):
    rows = _diagnostic_rows()
    path = tmp_path / "outdir"
    path.mkdir()
    (path / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(OSError):
        write_frequency_diagnostics(path, rows)
    assert path.is_dir()
    assert (path / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert not list(tmp_path.glob(".vepkar-diagnostics-*.tmp"))


def test_write_diagnostics_rejects_broken_symlink(tmp_path):
    rows = _diagnostic_rows()
    path = tmp_path / "diagnostics.csv"
    target = tmp_path / "missing-target.csv"
    try:
        path.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is unavailable in this environment")
    with pytest.raises(OSError):
        write_frequency_diagnostics(path, rows)
    assert path.is_symlink()
    assert path.readlink() == target
    assert not target.exists()
    assert not list(tmp_path.glob(".vepkar-diagnostics-*.tmp"))


def test_write_diagnostics_missing_parent_rejected_without_creation(tmp_path):
    rows = _diagnostic_rows()
    missing = tmp_path / "no-such-dir"
    path = missing / "diagnostics.csv"
    with pytest.raises(OSError):
        write_frequency_diagnostics(path, rows)
    assert not missing.exists()
    assert not list(tmp_path.glob(".vepkar-diagnostics-*.tmp"))


def test_write_diagnostics_cleans_temp_on_failure(tmp_path):
    def exploding_rows():
        yield _diagnostic_rows()[0]
        raise RuntimeError("boom")

    path = tmp_path / "diagnostics.csv"
    with pytest.raises(RuntimeError):
        write_frequency_diagnostics(path, exploding_rows())
    assert not path.exists()
    assert not list(tmp_path.glob(".vepkar-diagnostics-*.tmp"))