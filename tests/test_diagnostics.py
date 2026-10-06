"""Offline tests for the pure frequency-baseline diagnostic rows."""

import dataclasses
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from core.diagnostics import (  # noqa: E402
    DiagnosticMappingError,
    FrequencyDiagnostic,
    build_frequency_diagnostics,
)
from core.frequency import FrequencyPrediction  # noqa: E402
from core.instances import Candidate, Instance  # noqa: E402
from core.predictions import BenchmarkIntegrityError  # noqa: E402

FIELDS = (
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
)


def _instance(
    word_id: int,
    word: str,
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
        text_id=word_id,
        word=word,
        word_number=1,
        sentence_xml="<s/>",
        candidates=candidates,
        gold_analysis=gold,
    )


def _predictions(word_id: int, ranked):
    return [
        FrequencyPrediction(word_id, wordform_id, gramset, rank, score)
        for rank, (wordform_id, gramset, score) in enumerate(ranked, start=1)
    ]


def test_top1_and_gold_at_ranks_two_and_three():
    candidates = (
        Candidate(10, "SG+NOM"),
        Candidate(11, "SG+GEN"),
        Candidate(12, "SG+PAR"),
    )
    instances = [
        _instance(1, "kala", candidates, gold=Candidate(10, "SG+NOM")),
        _instance(2, "kala", candidates, gold=Candidate(11, "SG+GEN")),
        _instance(3, "kala", candidates, gold=Candidate(12, "SG+PAR")),
    ]
    ranked = [(10, "SG+NOM", 7), (11, "SG+GEN", 3), (12, "SG+PAR", 0)]
    predictions = [
        *_predictions(1, ranked),
        *_predictions(2, ranked),
        *_predictions(3, ranked),
    ]
    rows = build_frequency_diagnostics(instances, predictions)
    assert len(rows) == 3
    assert rows[0].language == "krl"
    assert rows[0].word_id == 1
    assert rows[0].word == "kala"
    assert rows[0].candidate_count == 3
    assert rows[0].gold_wordform_id == 10
    assert rows[0].gold_gramset == "SG+NOM"
    assert rows[0].top1_wordform_id == 10
    assert rows[0].top1_gramset == "SG+NOM"
    assert rows[0].gold_rank == 1
    assert rows[0].gold_train_frequency == 7
    assert rows[0].top1_train_frequency == 7
    assert rows[1].gold_wordform_id == 11
    assert rows[1].gold_gramset == "SG+GEN"
    assert rows[1].gold_rank == 2
    assert rows[1].gold_train_frequency == 3
    assert rows[1].top1_wordform_id == 10
    assert rows[1].top1_train_frequency == 7
    assert rows[2].gold_wordform_id == 12
    assert rows[2].gold_gramset == "SG+PAR"
    assert rows[2].gold_rank == 3
    assert rows[2].gold_train_frequency == 0
    assert rows[2].top1_train_frequency == 7


def test_equal_positive_and_zero_frequencies():
    candidates = (Candidate(10, "A"), Candidate(11, "B"), Candidate(12, "C"))
    instances = [
        _instance(1, "w1", candidates, gold=Candidate(11, "B")),
        _instance(2, "w2", candidates, gold=Candidate(12, "C")),
    ]
    predictions = [
        *_predictions(1, [(10, "A", 4), (11, "B", 4), (12, "C", 4)]),
        *_predictions(2, [(10, "A", 0), (11, "B", 0), (12, "C", 0)]),
    ]
    rows = build_frequency_diagnostics(instances, predictions)
    assert rows[0].gold_train_frequency == 4
    assert rows[0].top1_train_frequency == 4
    assert rows[1].gold_train_frequency == 0
    assert rows[1].top1_train_frequency == 0
    assert all(type(row.gold_train_frequency) is int for row in rows)
    assert all(type(row.top1_train_frequency) is int for row in rows)


def test_same_wordform_id_different_gramset():
    candidates = (Candidate(100, "SG+NOM"), Candidate(100, "SG+ACC"))
    instance = _instance(1, "kala", candidates, gold=Candidate(100, "SG+ACC"))
    predictions = _predictions(1, [(100, "SG+NOM", 5), (100, "SG+ACC", 2)])
    rows = build_frequency_diagnostics([instance], predictions)
    assert rows[0].gold_wordform_id == 100
    assert rows[0].gold_gramset == "SG+ACC"
    assert rows[0].gold_rank == 2
    assert rows[0].gold_train_frequency == 2
    assert rows[0].top1_wordform_id == 100
    assert rows[0].top1_gramset == "SG+NOM"
    assert rows[0].top1_train_frequency == 5


def test_same_gramset_different_wordform_id():
    candidates = (Candidate(100, "SG+NOM"), Candidate(200, "SG+NOM"))
    instance = _instance(1, "kala", candidates, gold=Candidate(200, "SG+NOM"))
    predictions = _predictions(1, [(100, "SG+NOM", 5), (200, "SG+NOM", 2)])
    rows = build_frequency_diagnostics([instance], predictions)
    assert rows[0].gold_wordform_id == 200
    assert rows[0].gold_gramset == "SG+NOM"
    assert rows[0].gold_rank == 2
    assert rows[0].gold_train_frequency == 2
    assert rows[0].top1_wordform_id == 100
    assert rows[0].top1_train_frequency == 5


def test_order_independence_and_rank1_not_first():
    candidates = (
        Candidate(10, "SG+NOM"),
        Candidate(11, "SG+GEN"),
        Candidate(12, "SG+PAR"),
    )
    instances = [
        _instance(1, "kala", candidates, gold=Candidate(11, "SG+GEN"), language="krl"),
        _instance(2, "vezi", candidates, gold=Candidate(12, "SG+PAR"), language="vep"),
        _instance(3, "magi", candidates, gold=Candidate(10, "SG+NOM"), language="krl"),
    ]
    predictions = [
        *_predictions(1, [(10, "SG+NOM", 7), (11, "SG+GEN", 3), (12, "SG+PAR", 0)]),
        *_predictions(2, [(10, "SG+NOM", 7), (11, "SG+GEN", 3), (12, "SG+PAR", 0)]),
        *_predictions(3, [(10, "SG+NOM", 7), (11, "SG+GEN", 3), (12, "SG+PAR", 0)]),
    ]
    expected = build_frequency_diagnostics(instances, predictions)

    reordered_candidates = (candidates[2], candidates[0], candidates[1])
    shuffled_instances = [
        _instance(
            2, "vezi", reordered_candidates, gold=Candidate(12, "SG+PAR"), language="vep"
        ),
        _instance(3, "magi", candidates, gold=Candidate(10, "SG+NOM"), language="krl"),
        _instance(
            1, "kala", reordered_candidates, gold=Candidate(11, "SG+GEN"), language="krl"
        ),
    ]
    shuffled_predictions = [
        predictions[4], predictions[5], predictions[3],
        predictions[8], predictions[6], predictions[7],
        predictions[2], predictions[0], predictions[1],
    ]
    actual = build_frequency_diagnostics(shuffled_instances, shuffled_predictions)
    assert actual == expected
    assert actual[0].word_id == 1
    assert actual[0].gold_rank == 2
    assert actual[1].word_id == 3
    assert actual[2].word_id == 2


def test_non_ascii_word_preserved():
    word = "Šuuruš’a!"
    instance = _instance(
        1,
        word,
        (Candidate(10, "SG+NOM"), Candidate(11, "SG+GEN")),
        gold=Candidate(10, "SG+NOM"),
    )
    predictions = _predictions(1, [(10, "SG+NOM", 2), (11, "SG+GEN", 1)])
    rows = build_frequency_diagnostics([instance], predictions)
    assert rows[0].word == word


def test_language_first_and_numeric_id_ordering():
    instances = [
        _instance(
            11, "w11", (Candidate(10, "A"), Candidate(11, "B")), language="vep"
        ),
        _instance(2, "w2", (Candidate(10, "A"), Candidate(11, "B")), language="krl"),
        _instance(10, "w10", (Candidate(10, "A"), Candidate(11, "B")), language="krl"),
        _instance(3, "w3", (Candidate(10, "A"), Candidate(11, "B")), language="vep"),
    ]
    predictions = [
        *_predictions(2, [(10, "A", 1), (11, "B", 1)]),
        *_predictions(10, [(10, "A", 1), (11, "B", 1)]),
        *_predictions(3, [(10, "A", 1), (11, "B", 1)]),
        *_predictions(11, [(10, "A", 1), (11, "B", 1)]),
    ]
    rows = build_frequency_diagnostics(instances, predictions)
    assert [(row.language, row.word_id) for row in rows] == [
        ("krl", 2),
        ("krl", 10),
        ("vep", 3),
        ("vep", 11),
    ]


def test_schema_integer_frequencies_and_one_row_per_instance():
    candidates = (
        Candidate(10, "SG+NOM"),
        Candidate(11, "SG+GEN"),
        Candidate(12, "SG+PAR"),
    )
    instances = [
        _instance(1, "kala", candidates, gold=Candidate(11, "SG+GEN")),
        _instance(2, "vezi", candidates, gold=Candidate(10, "SG+NOM")),
        _instance(3, "magi", candidates, gold=Candidate(12, "SG+PAR")),
    ]
    predictions = [
        *_predictions(1, [(10, "SG+NOM", 5), (11, "SG+GEN", 5), (12, "SG+PAR", 0)]),
        *_predictions(2, [(10, "SG+NOM", 5), (11, "SG+GEN", 5), (12, "SG+PAR", 0)]),
        *_predictions(3, [(10, "SG+NOM", 5), (11, "SG+GEN", 5), (12, "SG+PAR", 0)]),
    ]
    rows = build_frequency_diagnostics(instances, predictions)
    assert [field.name for field in dataclasses.fields(FrequencyDiagnostic)] == list(
        FIELDS
    )
    assert len(rows) == len(instances)
    for row in rows:
        assert isinstance(row, FrequencyDiagnostic)
        assert type(row.gold_train_frequency) is int
        assert type(row.top1_train_frequency) is int
        assert row.candidate_count == 3


def test_empty_inputs():
    assert build_frequency_diagnostics([], []) == ()


def test_single_pass_iterables():
    candidates = (Candidate(10, "A"), Candidate(11, "B"))
    instances = [_instance(1, "kala", candidates, gold=Candidate(11, "B"))]
    predictions = _predictions(1, [(10, "A", 3), (11, "B", 1)])
    expected = build_frequency_diagnostics(instances, predictions)
    actual = build_frequency_diagnostics(
        (instance for instance in instances),
        (prediction for prediction in predictions),
    )
    assert actual == expected


def test_duplicate_word_id_raises_integrity_error():
    instances = [
        _instance(1, "a", (Candidate(10, "A"), Candidate(11, "B"))),
        _instance(1, "b", (Candidate(20, "C"), Candidate(21, "D"))),
    ]
    with pytest.raises(BenchmarkIntegrityError, match="duplicate word_id"):
        build_frequency_diagnostics(instances, [])


def test_duplicate_word_id_across_languages_raises_integrity_error():
    instances = [
        _instance(
            1, "a", (Candidate(10, "A"), Candidate(11, "B")), language="krl"
        ),
        _instance(
            1, "b", (Candidate(20, "C"), Candidate(21, "D")), language="vep"
        ),
    ]
    with pytest.raises(BenchmarkIntegrityError, match="duplicate word_id"):
        build_frequency_diagnostics(instances, [])


def test_unknown_prediction_word_id():
    instance = _instance(
        1, "kala", (Candidate(10, "A"), Candidate(11, "B")), gold=Candidate(10, "A")
    )
    predictions = [FrequencyPrediction(99, 10, "A", 1, 5)]
    with pytest.raises(DiagnosticMappingError, match="99"):
        build_frequency_diagnostics([instance], predictions)


def test_duplicate_prediction_identity():
    instance = _instance(
        1, "kala", (Candidate(10, "A"), Candidate(11, "B")), gold=Candidate(10, "A")
    )
    predictions = [
        FrequencyPrediction(1, 10, "A", 1, 5),
        FrequencyPrediction(1, 10, "A", 2, 3),
    ]
    with pytest.raises(DiagnosticMappingError, match="duplicate"):
        build_frequency_diagnostics([instance], predictions)


def test_missing_predictions():
    candidates = (Candidate(10, "A"), Candidate(11, "B"))
    instances = [
        _instance(1, "kala", candidates, gold=Candidate(10, "A")),
        _instance(2, "vezi", candidates, gold=Candidate(10, "A")),
    ]
    predictions = _predictions(1, [(10, "A", 1), (11, "B", 1)])
    with pytest.raises(DiagnosticMappingError, match="2"):
        build_frequency_diagnostics(instances, predictions)


@pytest.mark.parametrize("ranks", [[2, 3], [1, 1, 2]])
def test_missing_or_multiple_rank1(ranks):
    candidates = (Candidate(10, "A"), Candidate(11, "B"), Candidate(12, "C"))
    instance = _instance(1, "kala", candidates, gold=Candidate(11, "B"))
    identities = [(10, "A"), (11, "B"), (12, "C")]
    predictions = [
        FrequencyPrediction(1, identities[index][0], identities[index][1], rank, 1)
        for index, rank in enumerate(ranks)
    ]
    with pytest.raises(DiagnosticMappingError, match="rank-1"):
        build_frequency_diagnostics([instance], predictions)


def test_missing_gold():
    instance = _instance(
        1, "kala", (Candidate(10, "A"), Candidate(11, "B")), gold=Candidate(11, "B")
    )
    predictions = _predictions(1, [(10, "A", 5)])
    with pytest.raises(DiagnosticMappingError, match="gold"):
        build_frequency_diagnostics([instance], predictions)


def test_gold_absent_from_strict_candidates():
    instance = _instance(
        1, "kala", (Candidate(10, "A"), Candidate(11, "B")), gold=Candidate(99, "Z")
    )
    predictions = _predictions(1, [(10, "A", 5), (99, "Z", 3)])
    with pytest.raises(DiagnosticMappingError, match="gold"):
        build_frequency_diagnostics([instance], predictions)


def test_top1_absent_from_strict_candidates():
    instance = _instance(
        1, "kala", (Candidate(10, "A"), Candidate(11, "B")), gold=Candidate(10, "A")
    )
    predictions = _predictions(1, [(99, "Z", 5), (10, "A", 3)])
    with pytest.raises(DiagnosticMappingError, match="top1"):
        build_frequency_diagnostics([instance], predictions)


@pytest.mark.parametrize("rank", [0, 4])
def test_gold_rank_outside_candidate_range(rank):
    candidates = (Candidate(10, "A"), Candidate(11, "B"), Candidate(12, "C"))
    instance = _instance(1, "kala", candidates, gold=Candidate(11, "B"))
    predictions = [
        FrequencyPrediction(1, 10, "A", 1, 5),
        FrequencyPrediction(1, 11, "B", rank, 3),
        FrequencyPrediction(1, 12, "C", 3, 1),
    ]
    with pytest.raises(DiagnosticMappingError, match="rank"):
        build_frequency_diagnostics([instance], predictions)


@pytest.mark.parametrize("score", [2.0, 2.5, -1, -2.0])
def test_gold_score_must_be_nonnegative_int(score):
    instance = _instance(
        1, "kala", (Candidate(10, "A"), Candidate(11, "B")), gold=Candidate(11, "B")
    )
    predictions = [
        FrequencyPrediction(1, 10, "A", 1, 5),
        FrequencyPrediction(1, 11, "B", 2, score),
    ]
    with pytest.raises(DiagnosticMappingError, match="score"):
        build_frequency_diagnostics([instance], predictions)


@pytest.mark.parametrize("score", [2.0, 2.5, -1])
def test_top1_score_must_be_nonnegative_int(score):
    instance = _instance(
        1, "kala", (Candidate(10, "A"), Candidate(11, "B")), gold=Candidate(11, "B")
    )
    predictions = [
        FrequencyPrediction(1, 10, "A", 1, score),
        FrequencyPrediction(1, 11, "B", 2, 3),
    ]
    with pytest.raises(DiagnosticMappingError, match="score"):
        build_frequency_diagnostics([instance], predictions)