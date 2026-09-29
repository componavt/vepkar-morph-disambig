"""Offline tests for the pure train-frequency ranking baseline."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from core.frequency import (  # noqa: E402
    FrequencyPrediction,
    TargetSplitError,
    build_train_frequency,
    rank_by_train_frequency,
)
from core.instances import Candidate, Instance  # noqa: E402


def _instance(
    word_id: int,
    text_id: int,
    candidates,
    gold=None,
    language: str = "krl",
) -> Instance:
    candidates = tuple(candidates)
    if gold is None:
        gold = candidates[0]
    return Instance(
        language=language,
        word_id=word_id,
        sentence_id=word_id,
        text_id=text_id,
        word=f"w{word_id}",
        word_number=1,
        sentence_xml="<s/>",
        candidates=candidates,
        gold_analysis=gold,
    )


def test_train_gold_frequency_only():
    instances = [
        _instance(
            501,
            1,
            [Candidate(9001, "SG+NOM"), Candidate(9001, "SG+ACC")],
            gold=Candidate(9001, "SG+NOM"),
        ),
        _instance(
            502,
            1,
            [Candidate(9001, "SG+NOM"), Candidate(9002, "PL+GEN")],
            gold=Candidate(9001, "SG+NOM"),
        ),
        _instance(
            503,
            1,
            [Candidate(9002, "PL+GEN"), Candidate(9003, "SG+GEN")],
            gold=Candidate(9002, "PL+GEN"),
        ),
    ]
    split_rows = [("krl", 1, "train")]
    frequency = build_train_frequency(instances, split_rows)
    assert frequency[(9001, "SG+NOM")] == 2
    assert frequency[(9002, "PL+GEN")] == 1
    assert frequency[(9001, "SG+ACC")] == 0
    assert frequency[(9003, "SG+GEN")] == 0
    assert len(frequency) == 2


def test_complete_target_output():
    dev_instances = [
        _instance(
            601,
            2,
            [Candidate(2001, "V+IND+PRS"), Candidate(2002, "V+IND+PST")],
        ),
        _instance(
            602,
            2,
            [
                Candidate(1001, "N+SG+NOM"),
                Candidate(1002, "N+SG+GEN"),
                Candidate(1003, "N+SG+PAR"),
            ],
        ),
    ]
    split_rows = [("krl", 2, "dev")]
    rows = rank_by_train_frequency(dev_instances, split_rows, "dev")
    assert {row.word_id for row in rows} == {601, 602}
    assert [row.word_id for row in rows] == [601, 601, 602, 602, 602]
    assert [row.rank for row in rows] == [1, 2, 1, 2, 3]
    identities = [(row.wordform_id, row.gramset) for row in rows]
    assert identities.count((2001, "V+IND+PRS")) == 1
    assert identities.count((2002, "V+IND+PST")) == 1
    assert identities.count((1001, "N+SG+NOM")) == 1
    assert identities.count((1002, "N+SG+GEN")) == 1
    assert identities.count((1003, "N+SG+PAR")) == 1
    assert all(row.score == 0 for row in rows)


def test_scores_come_from_train_frequency():
    train_instances = [
        _instance(501, 1, [Candidate(9001, "A")], gold=Candidate(9001, "A")),
        _instance(502, 1, [Candidate(9001, "A")], gold=Candidate(9001, "A")),
        _instance(503, 1, [Candidate(9001, "A")], gold=Candidate(9001, "A")),
        _instance(504, 1, [Candidate(9002, "B")], gold=Candidate(9002, "B")),
    ]
    target = _instance(
        601,
        2,
        [Candidate(9002, "B"), Candidate(9003, "C"), Candidate(9001, "A")],
    )
    split_rows = [("krl", 1, "train"), ("krl", 2, "dev")]
    rows = rank_by_train_frequency(
        train_instances + [target], split_rows, "dev"
    )
    score_by_identity = {
        (row.wordform_id, row.gramset): row.score for row in rows
    }
    assert score_by_identity[(9001, "A")] == 3
    assert score_by_identity[(9002, "B")] == 1
    assert score_by_identity[(9003, "C")] == 0
    assert [row.rank for row in rows] == [1, 2, 3]
    assert [row.gramset for row in rows] == ["A", "B", "C"]


def test_tie_break_by_wordform_id():
    train_instances = [
        _instance(501, 1, [Candidate(9001, "X")], gold=Candidate(9001, "X")),
        _instance(502, 1, [Candidate(9002, "Y")], gold=Candidate(9002, "Y")),
    ]
    target = _instance(
        601,
        2,
        [Candidate(9002, "Y"), Candidate(9001, "X")],
    )
    split_rows = [("krl", 1, "train"), ("krl", 2, "dev")]
    rows = rank_by_train_frequency(
        train_instances + [target], split_rows, "dev"
    )
    assert [row.wordform_id for row in rows] == [9001, 9002]
    assert [row.rank for row in rows] == [1, 2]
    assert [row.score for row in rows] == [1, 1]


def test_tie_break_by_gramset():
    train_instances = [
        _instance(
            501,
            1,
            [Candidate(9001, "SG+NOM")],
            gold=Candidate(9001, "SG+NOM"),
        ),
        _instance(
            502,
            1,
            [Candidate(9001, "SG+ACC")],
            gold=Candidate(9001, "SG+ACC"),
        ),
    ]
    target = _instance(
        601,
        2,
        [Candidate(9001, "SG+NOM"), Candidate(9001, "SG+ACC")],
    )
    split_rows = [("krl", 1, "train"), ("krl", 2, "dev")]
    rows = rank_by_train_frequency(
        train_instances + [target], split_rows, "dev"
    )
    assert [row.gramset for row in rows] == ["SG+ACC", "SG+NOM"]
    assert [row.rank for row in rows] == [1, 2]
    assert [row.score for row in rows] == [1, 1]


def test_evaluation_gold_must_not_leak():
    split_rows = [("krl", 2, "dev")]
    common = [Candidate(2001, "V+IND+PRS"), Candidate(2002, "V+IND+PST")]
    fixture_a = [
        _instance(601, 2, common, gold=Candidate(2001, "V+IND+PRS"))
    ]
    fixture_b = [
        _instance(601, 2, common, gold=Candidate(2002, "V+IND+PST"))
    ]
    rows_a = rank_by_train_frequency(fixture_a, split_rows, "dev")
    rows_b = rank_by_train_frequency(fixture_b, split_rows, "dev")
    assert rows_a == rows_b
    assert [row.score for row in rows_a] == [0, 0]
    assert [row.rank for row in rows_a] == [1, 2]


def test_dev_and_test_are_isolated():
    train_instances = [
        _instance(501, 1, [Candidate(9001, "A")], gold=Candidate(9001, "A")),
    ]
    dev_instances = [
        _instance(
            601,
            2,
            [
                Candidate(9001, "A"),
                Candidate(2001, "V+IND+PRS"),
                Candidate(2002, "V+IND+PST"),
            ],
        ),
        _instance(602, 2, [Candidate(3001, "PRON"), Candidate(3002, "N")]),
    ]
    test_instances = [
        _instance(701, 3, [Candidate(9001, "A"), Candidate(4002, "V")]),
    ]
    split_rows = [
        ("krl", 1, "train"),
        ("krl", 2, "dev"),
        ("krl", 3, "test"),
    ]
    all_instances = train_instances + dev_instances + test_instances

    dev_rows = rank_by_train_frequency(all_instances, split_rows, "dev")
    assert {row.word_id for row in dev_rows} == {601, 602}
    dev_score = {
        (row.wordform_id, row.gramset): row.score
        for row in dev_rows
        if row.word_id == 601
    }
    assert dev_score[(9001, "A")] == 1
    assert dev_score[(2001, "V+IND+PRS")] == 0

    test_rows = rank_by_train_frequency(all_instances, split_rows, "test")
    assert {row.word_id for row in test_rows} == {701}
    test_score = {
        (row.wordform_id, row.gramset): row.score for row in test_rows
    }
    assert test_score[(9001, "A")] == 1
    assert test_score[(4002, "V")] == 0


@pytest.mark.parametrize("split", ["train", "all", "unknown", ""])
def test_invalid_target_split(split):
    split_rows = [("krl", 1, "train")]
    with pytest.raises(TargetSplitError):
        rank_by_train_frequency([], split_rows, split)


def test_determinism():
    split_rows = [("krl", 1, "train"), ("krl", 2, "dev")]
    instances = [
        _instance(501, 1, [Candidate(9001, "A")], gold=Candidate(9001, "A")),
        _instance(
            601,
            2,
            [Candidate(9001, "A"), Candidate(9002, "B"), Candidate(9003, "C")],
        ),
        _instance(602, 2, [Candidate(9002, "B"), Candidate(9003, "C")]),
    ]
    forward = rank_by_train_frequency(instances, split_rows, "dev")
    reversed_order = rank_by_train_frequency(
        list(reversed(instances)), split_rows, "dev"
    )
    assert forward == reversed_order
    assert all(isinstance(row, FrequencyPrediction) for row in forward)