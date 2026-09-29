"""Offline tests for the pure prediction-file validator."""

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from core.instances import Candidate, Instance  # noqa: E402
from core.predictions import (  # noqa: E402
    BenchmarkIntegrityError,
    PredictionCsvParseError,
    PredictionFileReadError,
    validate_predictions,
)

SPLIT_ROWS = (
    ("krl", 1, "train"),
    ("krl", 2, "dev"),
    ("krl", 3, "test"),
)

TRAIN = {
    501: (Candidate(9001, "SG+NOM"), Candidate(9001, "SG+ACC"), Candidate(9002, "PL+GEN")),
    502: (Candidate(7712, "SG+NOM"), Candidate(7712, "SG+GEN")),
    503: (Candidate(1001, "N+SG+NOM"), Candidate(1002, "N+SG+GEN")),
}
DEV = {
    601: (Candidate(2001, "V+IND+PRS"), Candidate(2002, "V+IND+PST")),
}
TEST = {
    701: (Candidate(3001, "PRON"), Candidate(3002, "N")),
}


def _instance(word_id: int, text_id: int, candidates) -> Instance:
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


def _train_instances() -> tuple[Instance, ...]:
    return tuple(_instance(word_id, 1, candidates) for word_id, candidates in TRAIN.items())


TRAIN_INSTANCES = _train_instances()
INSTANCE_501 = TRAIN_INSTANCES[0]


def _all_instances() -> tuple[Instance, ...]:
    return (
        TRAIN_INSTANCES
        + tuple(_instance(word_id, 2, candidates) for word_id, candidates in DEV.items())
        + tuple(_instance(word_id, 3, candidates) for word_id, candidates in TEST.items())
    )


VALID_ROWS = [
    (501, 9001, "SG+NOM", 1, 125),
    (501, 9001, "SG+ACC", 2, 17),
    (501, 9002, "PL+GEN", 3, 3),
    (502, 7712, "SG+NOM", 1, 42),
    (502, 7712, "SG+GEN", 2, 8),
    (503, 1001, "N+SG+NOM", 1, 10),
    (503, 1002, "N+SG+GEN", 2, 5),
]


def _write(path: Path, rows) -> None:
    path.write_text(
        "word_id,wordform_id,gramset,rank,score\n"
        + "".join(f"{w},{f},{g},{r},{s}\n" for w, f, g, r, s in rows),
        encoding="utf-8",
    )


def test_valid_multi_word_file(tmp_path):
    path = tmp_path / "predictions.csv"
    _write(path, VALID_ROWS)
    result = validate_predictions(path, "train", _all_instances(), SPLIT_ROWS)
    assert result.is_valid
    assert result.split == "train"
    assert result.expected_word_count == 3
    assert result.expected_candidate_count == 7
    assert result.predicted_word_count == 3
    assert result.predicted_candidate_count == 7
    assert result.errors == {}
    assert result.error_counts == {}


@pytest.mark.parametrize(
    "header",
    [
        "word_id,rank,wordform_id,gramset,score",
        "word_id,wordform_id,gramset,rank,score,extra",
        "word_id,wordform_id,gramset,rank",
    ],
)
def test_wrong_header_rejected(tmp_path, header):
    path = tmp_path / "predictions.csv"
    path.write_text(header + "\n501,9001,SG+NOM,1,125\n", encoding="utf-8")
    result = validate_predictions(path, "train", TRAIN_INSTANCES, SPLIT_ROWS)
    assert not result.is_valid
    assert "header" in result.errors
    assert result.predicted_word_count == 0


def test_missing_expected_candidate(tmp_path):
    path = tmp_path / "predictions.csv"
    rows = [row for row in VALID_ROWS if not (row[0] == 501 and row[3] == 2)]
    _write(path, rows)
    result = validate_predictions(path, "train", _all_instances(), SPLIT_ROWS)
    assert not result.is_valid
    assert "missing candidate" in result.errors


def test_missing_candidate_full_count(tmp_path):
    path = tmp_path / "predictions.csv"
    _write(
        path,
        [
            (501, 9001, "SG+NOM", 1, 125),
            (502, 7712, "SG+NOM", 1, 42),
            (503, 1001, "N+SG+NOM", 1, 10),
        ],
    )
    result = validate_predictions(path, "train", TRAIN_INSTANCES, SPLIT_ROWS)
    assert not result.is_valid
    assert result.error_counts["missing candidate"] == 4
    assert len(result.errors["missing candidate"]) == 3


def test_unexpected_candidate(tmp_path):
    path = tmp_path / "predictions.csv"
    _write(path, VALID_ROWS + [(501, 9999, "SG+ILL", 4, 1)])
    result = validate_predictions(path, "train", _all_instances(), SPLIT_ROWS)
    assert not result.is_valid
    assert "unexpected candidate" in result.errors


def test_duplicate_candidate_tuple(tmp_path):
    path = tmp_path / "predictions.csv"
    rows = VALID_ROWS[:1] + [(501, 9001, "SG+NOM", 2, 100)] + VALID_ROWS[1:]
    _write(path, rows)
    result = validate_predictions(path, "train", _all_instances(), SPLIT_ROWS)
    assert not result.is_valid
    assert "duplicate candidate" in result.errors


def test_missing_expected_word(tmp_path):
    path = tmp_path / "predictions.csv"
    rows = [row for row in VALID_ROWS if row[0] != 503]
    _write(path, rows)
    result = validate_predictions(path, "train", _all_instances(), SPLIT_ROWS)
    assert not result.is_valid
    assert "missing word" in result.errors
    assert result.predicted_word_count == 2
    assert result.expected_word_count == 3


def test_wrong_split_word(tmp_path):
    path = tmp_path / "predictions.csv"
    _write(
        path,
        VALID_ROWS + [(601, 2001, "V+IND+PRS", 1, 9), (601, 2002, "V+IND+PST", 2, 2)],
    )
    result = validate_predictions(path, "train", _all_instances(), SPLIT_ROWS)
    assert not result.is_valid
    assert "wrong split" in result.errors


def test_unknown_word(tmp_path):
    path = tmp_path / "predictions.csv"
    _write(path, VALID_ROWS + [(999, 5555, "N+PL+NOM", 1, 1)])
    result = validate_predictions(path, "train", _all_instances(), SPLIT_ROWS)
    assert not result.is_valid
    assert "unknown word" in result.errors


@pytest.mark.parametrize("rank", ["0", "-1", "abc"])
def test_invalid_rank(tmp_path, rank):
    path = tmp_path / "predictions.csv"
    _write(
        path,
        [
            (501, 9001, "SG+NOM", rank, 125),
            (501, 9001, "SG+ACC", 2, 17),
            (501, 9002, "PL+GEN", 3, 3),
        ],
    )
    result = validate_predictions(path, "train", [INSTANCE_501], SPLIT_ROWS)
    assert not result.is_valid
    assert "invalid rank" in result.errors


def test_duplicate_rank(tmp_path):
    path = tmp_path / "predictions.csv"
    _write(
        path,
        [
            (501, 9001, "SG+NOM", 1, 125),
            (501, 9001, "SG+ACC", 1, 17),
            (501, 9002, "PL+GEN", 3, 3),
        ],
    )
    result = validate_predictions(path, "train", [INSTANCE_501], SPLIT_ROWS)
    assert not result.is_valid
    assert "rank set" in result.errors


def test_missing_rank(tmp_path):
    path = tmp_path / "predictions.csv"
    _write(
        path,
        [
            (501, 9001, "SG+NOM", 1, 125),
            (501, 9002, "PL+GEN", 3, 3),
        ],
    )
    result = validate_predictions(path, "train", [INSTANCE_501], SPLIT_ROWS)
    assert not result.is_valid
    assert "rank set" in result.errors


def test_score_rank_inconsistency(tmp_path):
    path = tmp_path / "predictions.csv"
    _write(
        path,
        [
            (501, 9001, "SG+NOM", 1, 10),
            (501, 9001, "SG+ACC", 2, 20),
            (501, 9002, "PL+GEN", 3, 30),
        ],
    )
    result = validate_predictions(path, "train", [INSTANCE_501], SPLIT_ROWS)
    assert not result.is_valid
    assert "score order" in result.errors


def test_equal_scores_accepted(tmp_path):
    path = tmp_path / "predictions.csv"
    _write(
        path,
        [
            (501, 9001, "SG+NOM", 1, 5),
            (501, 9001, "SG+ACC", 2, 5),
            (501, 9002, "PL+GEN", 3, 5),
            (502, 7712, "SG+NOM", 1, 5),
            (502, 7712, "SG+GEN", 2, 5),
            (503, 1001, "N+SG+NOM", 1, 5),
            (503, 1002, "N+SG+GEN", 2, 5),
        ],
    )
    result = validate_predictions(path, "train", TRAIN_INSTANCES, SPLIT_ROWS)
    assert result.is_valid
    assert result.errors == {}


@pytest.mark.parametrize("score", ["", "abc", "nan", "inf", "-inf"])
def test_invalid_score(tmp_path, score):
    path = tmp_path / "predictions.csv"
    _write(
        path,
        [
            (501, 9001, "SG+NOM", 1, 125),
            (501, 9001, "SG+ACC", 2, score),
            (501, 9002, "PL+GEN", 3, 3),
        ],
    )
    result = validate_predictions(path, "train", [INSTANCE_501], SPLIT_ROWS)
    assert not result.is_valid
    assert "invalid score" in result.errors


@pytest.mark.parametrize("value", ["", " ", "   ", "\t", "\n"])
def test_empty_gramset(tmp_path, value):
    path = tmp_path / "predictions.csv"
    import csv as _csv

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = _csv.writer(fh)
        writer.writerow(["word_id", "wordform_id", "gramset", "rank", "score"])
        writer.writerow([501, 9001, value, 1, 125])
        writer.writerow([501, 9001, "SG+ACC", 2, 17])
        writer.writerow([501, 9002, "PL+GEN", 3, 3])
    result = validate_predictions(path, "train", [INSTANCE_501], SPLIT_ROWS)
    assert not result.is_valid
    assert "empty gramset" in result.errors
    assert result.error_counts["empty gramset"] == 1


@pytest.mark.parametrize("value", ["0", "-5", "abc", ""])
def test_invalid_word_id(tmp_path, value):
    path = tmp_path / "predictions.csv"
    _write(
        path,
        [
            (value, 9001, "SG+NOM", 1, 125),
            (501, 9001, "SG+ACC", 2, 17),
            (501, 9002, "PL+GEN", 3, 3),
        ],
    )
    result = validate_predictions(path, "train", [INSTANCE_501], SPLIT_ROWS)
    assert not result.is_valid
    assert "invalid word_id" in result.errors


@pytest.mark.parametrize("value", ["0", "-5", "abc", ""])
def test_invalid_wordform_id(tmp_path, value):
    path = tmp_path / "predictions.csv"
    _write(
        path,
        [
            (501, value, "SG+NOM", 1, 125),
            (501, 9001, "SG+ACC", 2, 17),
            (501, 9002, "PL+GEN", 3, 3),
        ],
    )
    result = validate_predictions(path, "train", [INSTANCE_501], SPLIT_ROWS)
    assert not result.is_valid
    assert "invalid wordform_id" in result.errors


def test_unreadable_predictions_path(tmp_path):
    missing_path = tmp_path / "does-not-exist.csv"

    with pytest.raises(PredictionFileReadError) as raised:
        validate_predictions(
            missing_path,
            "train",
            TRAIN_INSTANCES,
            SPLIT_ROWS,
        )

    assert "cannot read predictions file" in str(raised.value)
    assert str(missing_path) in str(raised.value)
    assert isinstance(raised.value.__cause__, FileNotFoundError)


def test_directory_instead_of_predictions_file(tmp_path):
    with pytest.raises(PredictionFileReadError) as raised:
        validate_predictions(
            tmp_path,
            "train",
            TRAIN_INSTANCES,
            SPLIT_ROWS,
        )

    assert isinstance(raised.value.__cause__, IsADirectoryError)


def test_malformed_csv_syntax(tmp_path):
    path = tmp_path / "predictions.csv"
    path.write_text(
        'word_id,wordform_id,gramset,rank,score\n'
        '501,9001,"SG+NOM,1,125\n',
        encoding="utf-8",
    )

    with pytest.raises(PredictionCsvParseError) as raised:
        validate_predictions(
            path,
            "train",
            TRAIN_INSTANCES,
            SPLIT_ROWS,
        )

    assert "cannot parse predictions CSV" in str(raised.value)
    assert str(path) in str(raised.value)
    assert isinstance(raised.value.__cause__, csv.Error)


def test_duplicate_benchmark_word_id(tmp_path):
    duplicate_instances = (
        _instance(501, 1, TRAIN[501]),
        Instance(
            language="vep",
            word_id=501,
            sentence_id=999,
            text_id=999,
            word="duplicate",
            word_number=1,
            sentence_xml="<s/>",
            candidates=TRAIN,
            gold_analysis=TRAIN,
        ),
    )

    with pytest.raises(BenchmarkIntegrityError, match="duplicate word_id"):
        validate_predictions(
            tmp_path / "not-opened.csv",
            "train",
            duplicate_instances,
            SPLIT_ROWS,
        )