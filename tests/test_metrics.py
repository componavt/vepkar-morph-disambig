"""Offline tests for the pure in-memory ranking metrics core."""

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from core.frequency import FrequencyPrediction  # noqa: E402
from core.instances import Candidate, Instance  # noqa: E402
from core.metrics import (  # noqa: E402
    MetricsInputError,
    compute_ranking_metrics,
)


@dataclass(frozen=True)
class BareRow:
    """Minimal test-local prediction row: identity and rank only."""

    word_id: int
    wordform_id: int
    gramset: str
    rank: int


def _instance(
    word_id: int,
    candidates,
    gold=None,
) -> Instance:
    candidates = tuple(candidates)
    if gold is None:
        gold = candidates[0]
    return Instance(
        language="krl",
        word_id=word_id,
        sentence_id=word_id,
        text_id=word_id,
        word=f"w{word_id}",
        word_number=1,
        sentence_xml="<s/>",
        candidates=candidates,
        gold_analysis=gold,
    )


def _ranked(
    word_id: int,
    candidates,
    gold=None,
) -> tuple[Instance, list[FrequencyPrediction]]:
    """Return an instance and consecutive-rank rows ordered like ``candidates``."""
    candidates = tuple(candidates)
    if gold is None:
        gold = candidates[0]
    rows = [
        FrequencyPrediction(word_id, c.wordform_id, c.gramset, rank, 0)
        for rank, c in enumerate(candidates, start=1)
    ]
    return _instance(word_id, candidates, gold=gold), rows


def test_hand_calculated_mixed_result():
    instances = []
    rows = []
    instance, ranked = _ranked(
        601, [Candidate(2001, "V+IND+PRS"), Candidate(2002, "V+IND+PST")],
    )
    instances.append(instance)
    rows.extend(ranked)
    instance, ranked = _ranked(
        602, [Candidate(1001, "N+SG+NOM"), Candidate(1002, "N+SG+GEN")],
        gold=Candidate(1002, "N+SG+GEN"),
    )
    instances.append(instance)
    rows.extend(ranked)
    instance, ranked = _ranked(
        603,
        [
            Candidate(3001, "PRON"),
            Candidate(3002, "N"),
            Candidate(3003, "ADJ"),
            Candidate(3004, "V"),
        ],
        gold=Candidate(3004, "V"),
    )
    instances.append(instance)
    rows.extend(ranked)

    result = compute_ranking_metrics(instances, rows)
    assert result.word_count == 3
    assert result.top1_correct == 1
    assert result.top3_correct == 2
    assert result.top1_accuracy == pytest.approx(1 / 3)
    assert result.top3_accuracy == pytest.approx(2 / 3)
    assert result.mrr == pytest.approx((1 + 1 / 2 + 1 / 4) / 3)


def test_perfect_rankings():
    instances = []
    rows = []
    for word_id in (601, 602, 603):
        instance, ranked = _ranked(
            word_id, [Candidate(100, "A"), Candidate(101, "B")]
        )
        instances.append(instance)
        rows.extend(ranked)

    result = compute_ranking_metrics(instances, rows)
    assert result.word_count == 3
    assert result.top1_correct == 3
    assert result.top1_accuracy == 1.0
    assert result.mrr == 1.0
    assert result.top3_accuracy == 1.0


def test_top3_boundary():
    instance, rows = _ranked(
        601,
        [Candidate(100, "A"), Candidate(101, "B"), Candidate(102, "C")],
        gold=Candidate(102, "C"),
    )
    boundary_success = compute_ranking_metrics([instance], rows)
    assert boundary_success.top3_correct == 1
    assert boundary_success.top3_accuracy == 1.0
    assert boundary_success.top1_correct == 0

    instance, rows = _ranked(
        602,
        [
            Candidate(200, "A"),
            Candidate(201, "B"),
            Candidate(202, "C"),
            Candidate(203, "D"),
        ],
        gold=Candidate(203, "D"),
    )
    boundary_failure = compute_ranking_metrics([instance], rows)
    assert boundary_failure.top3_correct == 0
    assert boundary_failure.top3_accuracy == 0.0
    assert boundary_failure.mrr == pytest.approx(1 / 4)


def test_fewer_than_three_candidates():
    instance, rows = _ranked(
        601,
        [Candidate(100, "SG+NOM"), Candidate(100, "SG+ACC")],
        gold=Candidate(100, "SG+ACC"),
    )
    result = compute_ranking_metrics([instance], rows)
    assert result.top1_correct == 0
    assert result.top1_accuracy == 0.0
    assert result.mrr == pytest.approx(1 / 2)
    assert result.top3_correct == 1
    assert result.top3_accuracy == 1.0


def test_exact_candidate_identity():
    candidates = (Candidate(500, "SG+NOM"), Candidate(500, "SG+ACC"))
    instance, rows = _ranked(601, candidates, gold=Candidate(500, "SG+ACC"))
    result = compute_ranking_metrics([instance], rows)
    assert result.mrr == pytest.approx(1 / 2)
    assert result.top1_correct == 0
    assert result.top3_correct == 1

    instance, rows = _ranked(602, candidates, gold=Candidate(500, "SG+NOM"))
    result = compute_ranking_metrics([instance], rows)
    assert result.mrr == pytest.approx(1.0)
    assert result.top1_correct == 1


def test_equal_word_weighting():
    small, small_rows = _ranked(
        601, [Candidate(100, "A"), Candidate(101, "B")],
        gold=Candidate(101, "B"),
    )
    large, large_rows = _ranked(
        602,
        [
            Candidate(200, "A"),
            Candidate(201, "B"),
            Candidate(202, "C"),
            Candidate(203, "D"),
            Candidate(204, "E"),
        ],
    )
    result = compute_ranking_metrics(
        [small, large], small_rows + large_rows
    )
    assert result.word_count == 2
    assert result.top1_correct == 1
    assert result.top1_accuracy == pytest.approx(1 / 2)
    assert result.mrr == pytest.approx((1 / 2 + 1) / 2)
    assert result.top3_correct == 2
    assert result.top3_accuracy == 1.0


def test_model_independent_rows():
    instances = []
    rows = []
    instance, ranked = _ranked(
        601, [Candidate(2001, "V+IND+PRS"), Candidate(2002, "V+IND+PST")],
    )
    instances.append(instance)
    rows.extend(ranked)
    instance, ranked = _ranked(
        602, [Candidate(1001, "N+SG+NOM"), Candidate(1002, "N+SG+GEN")],
        gold=Candidate(1002, "N+SG+GEN"),
    )
    instances.append(instance)
    rows.extend(ranked)

    frequency_result = compute_ranking_metrics(instances, rows)
    bare_rows = [BareRow(r.word_id, r.wordform_id, r.gramset, r.rank) for r in rows]
    bare_result = compute_ranking_metrics(instances, bare_rows)
    assert all(
        getattr(frequency_result, field) == getattr(bare_result, field)
        for field in (
            "word_count",
            "top1_correct",
            "top3_correct",
            "top1_accuracy",
            "mrr",
            "top3_accuracy",
        )
    )


def test_ordering_and_generators():
    instances = [
        _instance(
            601,
            [Candidate(100, "A"), Candidate(101, "B")],
            gold=Candidate(100, "A"),
        ),
        _instance(
            602,
            [Candidate(200, "A"), Candidate(201, "B"), Candidate(202, "C")],
            gold=Candidate(202, "C"),
        ),
        _instance(
            603,
            [Candidate(300, "A"), Candidate(301, "B"), Candidate(302, "C")],
            gold=Candidate(301, "B"),
        ),
    ]
    rows = [
        FrequencyPrediction(601, 100, "A", 1, 3),
        FrequencyPrediction(601, 101, "B", 2, 2),
        FrequencyPrediction(602, 200, "A", 1, 1),
        FrequencyPrediction(602, 201, "B", 2, 1),
        FrequencyPrediction(602, 202, "C", 3, 1),
        FrequencyPrediction(603, 300, "A", 1, 5),
        FrequencyPrediction(603, 301, "B", 2, 3),
        FrequencyPrediction(603, 302, "C", 3, 2),
    ]

    forward = compute_ranking_metrics(
        (instance for instance in instances), (row for row in rows)
    )
    shuffled = compute_ranking_metrics(
        (instance for instance in reversed(instances)),
        (row for row in reversed(rows)),
    )
    fresh = compute_ranking_metrics(
        (instance for instance in instances), (row for row in rows)
    )
    assert forward == shuffled
    assert forward == fresh


def test_empty_evaluation_set():
    with pytest.raises(MetricsInputError, match="empty"):
        compute_ranking_metrics([], [])


def test_duplicate_instance_word_id():
    instance, rows = _ranked(
        601, [Candidate(100, "A"), Candidate(101, "B")]
    )
    duplicate = _instance(
        601, [Candidate(100, "A"), Candidate(101, "B")]
    )
    with pytest.raises(MetricsInputError, match="duplicate"):
        compute_ranking_metrics([instance, duplicate], rows)


def test_prediction_word_id_outside_evaluation_set():
    instance, rows = _ranked(
        601, [Candidate(100, "A"), Candidate(101, "B")]
    )
    extra = FrequencyPrediction(999, 500, "X", 1, 1)
    with pytest.raises(MetricsInputError, match="not in the supplied"):
        compute_ranking_metrics([instance], rows + [extra])


def test_duplicate_prediction_candidate():
    instance, rows = _ranked(
        601, [Candidate(100, "A"), Candidate(101, "B")]
    )
    duplicate = FrequencyPrediction(601, 100, "A", 1, 1)
    with pytest.raises(MetricsInputError, match="duplicate prediction"):
        compute_ranking_metrics([instance], rows + [duplicate])


def test_non_strict_candidate_rejected():
    instance, rows = _ranked(
        601, [Candidate(100, "A"), Candidate(101, "B")]
    )
    extra = FrequencyPrediction(
        word_id=601,
        wordform_id=999,
        gramset="MADE-UP",
        rank=3,
        score=0,
    )
    with pytest.raises(MetricsInputError, match="strict benchmark") as excinfo:
        compute_ranking_metrics([instance], rows + [extra])
    assert "word_id=601" in str(excinfo.value)


def test_missing_non_gold_strict_candidate_rejected():
    instance, rows = _ranked(
        601,
        [Candidate(100, "A"), Candidate(101, "B"), Candidate(102, "C")],
        gold=Candidate(100, "A"),
    )
    incomplete_rows = [rows[0]]
    with pytest.raises(MetricsInputError, match="incomplete"):
        compute_ranking_metrics([instance], incomplete_rows)


def test_missing_prediction_word_rejected():
    first, first_rows = _ranked(
        601, [Candidate(100, "A"), Candidate(101, "B")]
    )
    second, _ = _ranked(
        602, [Candidate(200, "A"), Candidate(201, "B")]
    )
    with pytest.raises(MetricsInputError, match="incomplete") as excinfo:
        compute_ranking_metrics([first, second], first_rows)
    assert "word_id=602" in str(excinfo.value)


def test_complete_valid_input_unchanged():
    instance, rows = _ranked(
        601,
        [Candidate(100, "A"), Candidate(101, "B"), Candidate(102, "C")],
        gold=Candidate(101, "B"),
    )
    result = compute_ranking_metrics([instance], rows)
    assert result.word_count == 1
    assert result.top1_correct == 0
    assert result.top3_correct == 1
    assert result.top1_accuracy == 0.0
    assert result.mrr == pytest.approx(1 / 2)
    assert result.top3_accuracy == 1.0


def test_missing_gold_candidate():
    instance, rows = _ranked(
        601, [Candidate(100, "A"), Candidate(101, "B")],
        gold=Candidate(101, "B"),
    )
    rows = [row for row in rows if row.gramset != "B"]
    with pytest.raises(MetricsInputError, match="incomplete"):
        compute_ranking_metrics([instance], rows)


def test_nonpositive_or_non_integer_gold_rank():
    instance, rows = _ranked(
        601, [Candidate(100, "A"), Candidate(101, "B")]
    )
    bad = FrequencyPrediction(601, 100, "A", 0, 1)
    rows = [bad, rows[1]]
    with pytest.raises(MetricsInputError, match="nonpositive or non-integer"):
        compute_ranking_metrics([instance], rows)

    bad = FrequencyPrediction(601, 100, "A", 1.5, 1)
    rows = [bad, FrequencyPrediction(601, 101, "B", 2, 1)]
    with pytest.raises(MetricsInputError, match="nonpositive or non-integer"):
        compute_ranking_metrics([instance], rows)


@pytest.mark.parametrize("bad_rank", [0, -1, 1.5, True, False, "1", None])
def test_invalid_non_gold_rank(bad_rank):
    instance, _ = _ranked(
        601,
        [Candidate(100, "A"), Candidate(101, "B"), Candidate(102, "C")],
        gold=Candidate(100, "A"),
    )
    rows = [
        FrequencyPrediction(601, 100, "A", 1, 0),
        FrequencyPrediction(601, 101, "B", bad_rank, 0),
        FrequencyPrediction(601, 102, "C", 3, 0),
    ]
    with pytest.raises(MetricsInputError, match="nonpositive or non-integer"):
        compute_ranking_metrics([instance], rows)


def test_duplicate_ranks_rejected():
    instance, _ = _ranked(
        601,
        [Candidate(100, "A"), Candidate(101, "B"), Candidate(102, "C")],
        gold=Candidate(100, "A"),
    )
    rows = [
        FrequencyPrediction(601, 100, "A", 1, 0),
        FrequencyPrediction(601, 101, "B", 2, 0),
        FrequencyPrediction(601, 102, "C", 2, 0),
    ]
    with pytest.raises(MetricsInputError, match="complete permutation"):
        compute_ranking_metrics([instance], rows)


def test_out_of_range_rank_rejected():
    instance, _ = _ranked(
        601,
        [Candidate(100, "A"), Candidate(101, "B"), Candidate(102, "C")],
        gold=Candidate(100, "A"),
    )
    rows = [
        FrequencyPrediction(601, 100, "A", 1, 0),
        FrequencyPrediction(601, 101, "B", 2, 0),
        FrequencyPrediction(601, 102, "C", 4, 0),
    ]
    with pytest.raises(MetricsInputError, match="complete permutation") as excinfo:
        compute_ranking_metrics([instance], rows)
    assert "word_id=601" in str(excinfo.value)


def test_missing_first_rank_rejected():
    instance, _ = _ranked(
        601,
        [Candidate(100, "A"), Candidate(101, "B"), Candidate(102, "C")],
        gold=Candidate(100, "A"),
    )
    rows = [
        FrequencyPrediction(601, 100, "A", 2, 0),
        FrequencyPrediction(601, 101, "B", 3, 0),
        FrequencyPrediction(601, 102, "C", 4, 0),
    ]
    with pytest.raises(MetricsInputError, match="complete permutation"):
        compute_ranking_metrics([instance], rows)


def test_valid_permutation_shuffled_row_order():
    instance, rows = _ranked(
        601,
        [Candidate(100, "A"), Candidate(101, "B"), Candidate(102, "C")],
        gold=Candidate(102, "C"),
    )
    ordered = compute_ranking_metrics([instance], rows)
    shuffled = compute_ranking_metrics(
        [instance], [rows[2], rows[0], rows[1]]
    )
    assert ordered == shuffled
    assert shuffled.top1_correct == 0
    assert shuffled.mrr == pytest.approx(1 / 3)