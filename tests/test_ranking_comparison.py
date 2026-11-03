"""Offline tests for the pure in-memory ranking comparison core."""

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from core.instances import Candidate, Instance  # noqa: E402
from core.metrics import MetricsInputError  # noqa: E402
from core.ranking_comparison import compare_rankings  # noqa: E402


@dataclass(frozen=True)
class Row:
    """Minimal test-local prediction row: identity and rank only (no score)."""

    word_id: int
    wordform_id: int
    gramset: str
    rank: int


def _make(
    word_id: int,
    candidates,
    gold,
    a_ranks,
    b_ranks,
    language: str = "krl",
):
    candidates = tuple(candidates)
    instance = Instance(
        language=language,
        word_id=word_id,
        sentence_id=word_id,
        text_id=word_id,
        word=f"w{word_id}",
        word_number=1,
        sentence_xml="<s/>",
        candidates=candidates,
        gold_analysis=gold,
    )
    a_rows = [
        Row(word_id, c.wordform_id, c.gramset, rank)
        for c, rank in zip(candidates, a_ranks)
    ]
    b_rows = [
        Row(word_id, c.wordform_id, c.gramset, rank)
        for c, rank in zip(candidates, b_ranks)
    ]
    return instance, a_rows, b_rows


def _candidates(wordform_start: int, count: int):
    return [
        Candidate(wordform_start + i, f"G{i}") for i in range(count)
    ]


def test_mixed_comparison_covers_all_outcomes():
    instances, a_rows, b_rows = [], [], []

    inst, a, b = _make(
        601,
        _candidates(101, 2),
        Candidate(101, "G0"),
        [1, 2],
        [1, 2],
    )
    instances.append(inst)
    a_rows.extend(a)
    b_rows.extend(b)

    shared = (Candidate(200, "N+SG+NOM"), Candidate(200, "N+SG+GEN"))
    inst, a, b = _make(
        602,
        shared,
        Candidate(200, "N+SG+NOM"),
        [2, 1],
        [1, 2],
    )
    instances.append(inst)
    a_rows.extend(a)
    b_rows.extend(b)

    inst, a, b = _make(
        603,
        _candidates(301, 2),
        Candidate(301, "G0"),
        [1, 2],
        [2, 1],
    )
    instances.append(inst)
    a_rows.extend(a)
    b_rows.extend(b)

    inst, a, b = _make(
        604,
        _candidates(401, 3),
        Candidate(401, "G0"),
        [2, 1, 3],
        [2, 1, 3],
    )
    instances.append(inst)
    a_rows.extend(a)
    b_rows.extend(b)

    inst, a, b = _make(
        605,
        _candidates(501, 4),
        Candidate(501, "G0"),
        [3, 1, 2, 4],
        [2, 3, 1, 4],
    )
    instances.append(inst)
    a_rows.extend(a)
    b_rows.extend(b)

    result = compare_rankings(instances, a_rows, b_rows)

    assert result.a_input_index == 1
    assert result.b_input_index == 2
    assert result.selection_reason == "mrr"

    assert result.overall.word_count == 5
    assert result.overall.candidate_count == 13
    assert result.overall.both_correct == 1
    assert result.overall.corrected == 1
    assert result.overall.regressed == 1
    assert result.overall.both_wrong_same == 1
    assert result.overall.both_wrong_changed == 1
    assert result.overall.top1_changed == 3
    assert result.overall.gold_rank_improved == 2
    assert result.overall.gold_rank_worsened == 1
    assert result.overall.gold_rank_unchanged == 2

    assert result.overall.metrics_a.mrr == pytest.approx(2 / 3)
    assert result.overall.metrics_b.mrr == pytest.approx(0.7)
    assert result.overall.delta_top1 == pytest.approx(0.0)
    assert result.overall.delta_mrr == pytest.approx(0.7 - 2 / 3)
    assert result.overall.delta_top3 == pytest.approx(0.0)

    assert (
        result.overall.delta_top1
        == pytest.approx(
            (result.overall.corrected - result.overall.regressed)
            / result.overall.word_count
        )
    )
    mean_delta_rr = sum(r.delta_rr for r in result.rows) / len(result.rows)
    assert result.overall.delta_mrr == pytest.approx(mean_delta_rr)

    summary = result.overall
    assert (
        summary.both_correct
        + summary.corrected
        + summary.regressed
        + summary.both_wrong_same
        + summary.both_wrong_changed
    ) == summary.word_count
    assert summary.top1_changed == (
        summary.corrected
        + summary.regressed
        + summary.both_wrong_changed
    )
    assert (
        summary.gold_rank_improved
        + summary.gold_rank_worsened
        + summary.gold_rank_unchanged
    ) == summary.word_count

    by_word = {row.word_id: row for row in result.rows}
    assert [row.word_id for row in result.rows] == [601, 602, 603, 604, 605]

    w602 = by_word[602]
    assert w602.language == "krl"
    assert w602.word == "w602"
    assert w602.candidate_count == 2
    assert w602.gold_wordform_id == 200
    assert w602.gold_gramset == "N+SG+NOM"
    assert w602.a_top1_wordform_id == 200
    assert w602.a_top1_gramset == "N+SG+GEN"
    assert w602.b_top1_wordform_id == 200
    assert w602.b_top1_gramset == "N+SG+NOM"
    assert w602.a_gold_rank == 2
    assert w602.b_gold_rank == 1
    assert w602.top1_changed is True
    assert w602.outcome == "corrected"
    assert w602.delta_rr == pytest.approx(0.5)

    w604 = by_word[604]
    assert w604.outcome == "both_wrong_same"
    assert w604.top1_changed is False
    assert w604.delta_rr == pytest.approx(0.0)

    w605 = by_word[605]
    assert w605.outcome == "both_wrong_changed"
    assert w605.top1_changed is True
    assert w605.delta_rr == pytest.approx(1 / 2 - 1 / 3)


def test_gold_rank4_to_2_delta_rr_and_top3_change():
    instances, a_rows, b_rows = [], [], []

    inst, a, b = _make(
        701,
        _candidates(11, 4),
        Candidate(11, "G0"),
        [4, 1, 2, 3],
        [2, 1, 3, 4],
    )
    instances.append(inst)
    a_rows.extend(a)
    b_rows.extend(b)

    inst, a, b = _make(
        702,
        _candidates(21, 2),
        Candidate(21, "G0"),
        [1, 2],
        [1, 2],
    )
    instances.append(inst)
    a_rows.extend(a)
    b_rows.extend(b)

    result = compare_rankings(instances, a_rows, b_rows)

    w701 = next(row for row in result.rows if row.word_id == 701)
    assert w701.a_gold_rank == 4
    assert w701.b_gold_rank == 2
    assert w701.outcome == "both_wrong_same"
    assert w701.delta_rr == pytest.approx(0.25)

    assert result.overall.metrics_a.top3_correct == 1
    assert result.overall.metrics_b.top3_correct == 2
    assert result.overall.delta_top3 == pytest.approx(0.5)

    mean_delta_rr = sum(r.delta_rr for r in result.rows) / len(result.rows)
    assert mean_delta_rr == pytest.approx(0.125)
    assert result.overall.delta_mrr == pytest.approx(0.125)
    assert w701.delta_rr == pytest.approx(0.25)


def test_higher_top1_but_lower_mrr_selects_top1():
    instances, a_rows, b_rows = [], [], []
    for word_id in range(701, 711):
        if word_id in (701, 702, 703):
            a_ranks, b_ranks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
            ]
        elif word_id == 704:
            a_ranks, b_ranks = [2, 1, 3, 4, 5, 6, 7, 8, 9, 10], [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
            ]
        else:
            a_ranks, b_ranks = [2, 1, 3, 4, 5, 6, 7, 8, 9, 10], [
                10,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                1,
            ]
        inst, a, b = _make(
            word_id,
            _candidates(word_id * 10, 10),
            Candidate(word_id * 10, "G0"),
            a_ranks,
            b_ranks,
        )
        instances.append(inst)
        a_rows.extend(a)
        b_rows.extend(b)

    result = compare_rankings(instances, a_rows, b_rows)

    assert result.selection_reason == "top1"
    assert result.b_input_index == 2
    assert result.overall.metrics_a.top1_correct == 3
    assert result.overall.metrics_b.top1_correct == 4
    assert result.overall.delta_top1 == pytest.approx(0.1)
    assert result.overall.metrics_a.mrr == pytest.approx(0.65)
    assert result.overall.metrics_b.mrr == pytest.approx(0.46)
    assert result.overall.delta_mrr < 0
    assert result.overall.delta_mrr == pytest.approx(-0.19)


def test_tie_breaking_by_mrr_then_top3():
    instances, a_rows, b_rows = [], [], []
    inst, a, b = _make(
        801, _candidates(11, 3), Candidate(11, "G0"), [1, 2, 3], [2, 1, 3]
    )
    instances.append(inst)
    a_rows.extend(a)
    b_rows.extend(b)
    inst, a, b = _make(
        802, _candidates(21, 3), Candidate(21, "G0"), [2, 1, 3], [1, 2, 3]
    )
    instances.append(inst)
    a_rows.extend(a)
    b_rows.extend(b)
    inst, a, b = _make(
        803, _candidates(31, 3), Candidate(31, "G0"), [3, 1, 2], [2, 1, 3]
    )
    instances.append(inst)
    a_rows.extend(a)
    b_rows.extend(b)

    by_mrr = compare_rankings(instances, a_rows, b_rows)
    assert by_mrr.selection_reason == "mrr"
    assert by_mrr.b_input_index == 2
    assert by_mrr.overall.metrics_a.top1_correct == by_mrr.overall.metrics_b.top1_correct
    assert by_mrr.overall.metrics_a.mrr != by_mrr.overall.metrics_b.mrr

    instances, a_rows, b_rows = [], [], []
    specs = {
        901: ([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]),
        902: ([2, 1, 3, 4, 5, 6], [4, 1, 2, 3, 5, 6]),
        903: ([6, 1, 2, 3, 4, 5], [4, 1, 2, 3, 5, 6]),
        904: ([6, 1, 2, 3, 4, 5], [4, 1, 2, 3, 5, 6]),
        905: ([6, 1, 2, 3, 4, 5], [4, 1, 2, 3, 5, 6]),
    }
    for word_id, (a_ranks, b_ranks) in specs.items():
        inst, a, b = _make(
            word_id,
            _candidates(word_id * 10, 6),
            Candidate(word_id * 10, "G0"),
            a_ranks,
            b_ranks,
        )
        instances.append(inst)
        a_rows.extend(a)
        b_rows.extend(b)

    by_top3 = compare_rankings(instances, a_rows, b_rows)
    assert by_top3.selection_reason == "top3"
    assert by_top3.b_input_index == 1
    assert by_top3.overall.metrics_a.top1_correct == by_top3.overall.metrics_b.top1_correct
    assert by_top3.overall.metrics_a.mrr == pytest.approx(by_top3.overall.metrics_b.mrr)
    assert by_top3.overall.metrics_a.top3_correct == 1
    assert by_top3.overall.metrics_b.top3_correct == 2
    assert by_top3.overall.delta_top3 == pytest.approx(0.2)


def test_metric_tie_keeps_input_order_and_analyses():
    instances, a_rows, b_rows = [], [], []
    inst, a, b = _make(
        1001, _candidates(11, 2), Candidate(11, "G0"), [1, 2], [2, 1]
    )
    instances.append(inst)
    a_rows.extend(a)
    b_rows.extend(b)
    inst, a, b = _make(
        1002, _candidates(21, 2), Candidate(21, "G0"), [2, 1], [1, 2]
    )
    instances.append(inst)
    a_rows.extend(a)
    b_rows.extend(b)

    result = compare_rankings(instances, a_rows, b_rows)

    assert result.selection_reason == "tie"
    assert result.a_input_index == 1
    assert result.b_input_index == 2
    assert result.overall.metrics_a == result.overall.metrics_b
    assert result.overall.corrected == 1
    assert result.overall.regressed == 1


def test_reversed_inputs_preserve_roles_and_deltas():
    instances, a_rows, b_rows = [], [], []
    inst, a, b = _make(
        801, _candidates(11, 3), Candidate(11, "G0"), [1, 2, 3], [2, 1, 3]
    )
    instances.append(inst)
    a_rows.extend(a)
    b_rows.extend(b)
    inst, a, b = _make(
        802, _candidates(21, 3), Candidate(21, "G0"), [2, 1, 3], [1, 2, 3]
    )
    instances.append(inst)
    a_rows.extend(a)
    b_rows.extend(b)
    inst, a, b = _make(
        803, _candidates(31, 3), Candidate(31, "G0"), [3, 1, 2], [2, 1, 3]
    )
    instances.append(inst)
    a_rows.extend(a)
    b_rows.extend(b)

    forward = compare_rankings(instances, a_rows, b_rows)
    reversed_ = compare_rankings(instances, b_rows, a_rows)

    assert (forward.a_input_index, forward.b_input_index) == (1, 2)
    assert (reversed_.a_input_index, reversed_.b_input_index) == (2, 1)
    assert reversed_.selection_reason == forward.selection_reason

    assert reversed_.rows == forward.rows
    assert reversed_.overall == forward.overall
    assert reversed_.by_language == forward.by_language


def test_global_roles_preserved_in_language_deltas_and_empty_group():
    instances, a_rows, b_rows = [], [], []

    for word_id in (1101, 1102):
        inst, a, b = _make(
            word_id,
            _candidates(11, 2),
            Candidate(11, "G0"),
            [1, 2],
            [2, 1],
            language="krl",
        )
        instances.append(inst)
        a_rows.extend(a)
        b_rows.extend(b)

    inst, a, b = _make(
        1201, _candidates(21, 2), Candidate(21, "G0"), [2, 1], [1, 2], language="olo"
    )
    instances.append(inst)
    a_rows.extend(a)
    b_rows.extend(b)

    inst, a, b = _make(
        1301, _candidates(31, 2), Candidate(31, "G0"), [2, 1], [1, 2], language="vep"
    )
    instances.append(inst)
    a_rows.extend(a)
    b_rows.extend(b)
    inst, a, b = _make(
        1302,
        _candidates(41, 3),
        Candidate(41, "G0"),
        [3, 1, 2],
        [1, 2, 3],
        language="vep",
    )
    instances.append(inst)
    a_rows.extend(a)
    b_rows.extend(b)

    result = compare_rankings(instances, a_rows, b_rows)

    assert result.selection_reason == "top1"
    assert result.a_input_index == 1
    assert result.b_input_index == 2
    assert list(result.by_language) == ["krl", "lud", "olo", "vep"]

    krl = result.by_language["krl"]
    assert krl.word_count == 2
    assert krl.delta_top1 < 0
    assert krl.delta_mrr < 0
    assert krl.metrics_a.mrr == 1.0
    assert krl.metrics_b.mrr == 0.5

    olo = result.by_language["olo"]
    assert olo.word_count == 1
    assert olo.delta_top1 == pytest.approx(1.0)

    vep = result.by_language["vep"]
    assert vep.word_count == 2

    lud = result.by_language["lud"]
    assert lud.word_count == 0
    assert lud.candidate_count == 0
    assert lud.metrics_a is None
    assert lud.metrics_b is None
    assert lud.delta_top1 is None
    assert lud.delta_mrr is None
    assert lud.delta_top3 is None
    assert lud.both_correct == 0
    assert lud.corrected == 0
    assert lud.regressed == 0


def test_shuffled_generators_deterministic_and_inputs_unchanged():
    instances, a_rows, b_rows = [], [], []

    inst, a, b = _make(
        601,
        _candidates(101, 2),
        Candidate(101, "G0"),
        [1, 2],
        [1, 2],
    )
    instances.append(inst)
    a_rows.extend(a)
    b_rows.extend(b)
    inst, a, b = _make(
        602,
        (Candidate(200, "N+SG+NOM"), Candidate(200, "N+SG+GEN")),
        Candidate(200, "N+SG+NOM"),
        [2, 1],
        [1, 2],
    )
    instances.append(inst)
    a_rows.extend(a)
    b_rows.extend(b)
    inst, a, b = _make(
        603,
        _candidates(301, 3),
        Candidate(301, "G0"),
        [2, 1, 3],
        [3, 1, 2],
    )
    instances.append(inst)
    a_rows.extend(a)
    b_rows.extend(b)

    instances_snapshot = tuple(instances)
    a_snapshot = tuple(a_rows)
    b_snapshot = tuple(b_rows)

    forward = compare_rankings(instances, a_rows, b_rows)
    shuffled = compare_rankings(
        (i for i in reversed(instances)),
        (r for r in reversed(a_rows)),
        (r for r in reversed(b_rows)),
    )
    generators = compare_rankings(
        (i for i in instances), (r for r in a_rows), (r for r in b_rows)
    )

    assert forward == shuffled
    assert forward == generators
    assert tuple(instances) == instances_snapshot
    assert tuple(a_rows) == a_snapshot
    assert tuple(b_rows) == b_snapshot


def test_invalid_ranking_raises_metrics_error():
    instances, a_rows, b_rows = [], [], []
    inst, a, b = _make(
        601, _candidates(101, 3), Candidate(101, "G0"), [1, 2, 3], [1, 2, 3]
    )
    instances.append(inst)
    a_rows.extend(a)
    b_rows.extend(b)

    incomplete = a_rows[:1]
    with pytest.raises(MetricsInputError, match="incomplete"):
        compare_rankings(instances, incomplete, b_rows)