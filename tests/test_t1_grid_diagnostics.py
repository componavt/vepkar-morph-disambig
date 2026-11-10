"""Tests for word-level T1 C-grid trajectory diagnostics."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from core.instances import Candidate, Instance  # noqa: E402
from core.predictions import PredictionRow  # noqa: E402
from t1_features.grid_diagnostics import (  # noqa: E402
    T1_C_GRID,
    T1GridDiagnosticError,
    summarize_trajectory,
)
from commands.diagnose_t1_grid import _build_output_rows  # noqa: E402


@pytest.mark.parametrize(
    ("correctness", "category", "flips", "direction", "threshold"),
    [
        ([True] * 10, "always_correct", 0, "none", 0.1),
        ([False] * 10, "always_wrong", 0, "none", None),
        ([False] * 3 + [True] * 7, "threshold_x_to_plus", 1, "x_to_plus", 3.0),
        ([True] * 3 + [False] * 7, "regression_plus_to_x", 1, "plus_to_x", None),
        ([False, True, False] + [True] * 7, "unstable", 3, "multiple", None),
    ],
)
def test_summarize_trajectory(correctness, category, flips, direction, threshold):
    summary = summarize_trajectory(correctness)
    assert summary.trajectory == "".join("+" if value else "x" for value in correctness)
    assert summary.n_wrong == correctness.count(False)
    assert summary.n_runs == len(T1_C_GRID)
    assert summary.n_flips == flips
    assert summary.c_threshold == threshold
    assert summary.flip_direction == direction
    assert summary.category == category


@pytest.mark.parametrize(
    ("correctness", "c_values"),
    [([], ()), ([True], (0.1, 0.3)), ([True, False], (0.3, 0.1))],
)
def test_invalid_trajectory_rejected(correctness, c_values):
    with pytest.raises(T1GridDiagnosticError):
        summarize_trajectory(correctness, c_values)


def _instance(word_id, text_id, sentence_id, word, candidates, gold):
    return Instance(
        language="krl",
        word_id=word_id,
        sentence_id=sentence_id,
        text_id=text_id,
        word=word,
        word_number=2,
        sentence_xml="<s/>",
        candidates=candidates,
        gold_analysis=gold,
    )


def _predictions(word_id, *, gold_first, score_top, score_second):
    gold = Candidate(11, "N+SG+NOM")
    other = Candidate(12, "N+SG+GEN")
    first, second = (gold, other) if gold_first else (other, gold)
    return (
        PredictionRow(word_id, first.wordform_id, first.gramset, 1, score_top),
        PredictionRow(word_id, second.wordform_id, second.gramset, 2, score_second),
    )


def test_build_output_rows_keeps_candidate_identity_and_dictionary_links():
    gold = Candidate(11, "N+SG+NOM")
    other = Candidate(12, "N+SG+GEN")
    candidates = (gold, other)
    train_instance = _instance(50, 1, 5, "kala", candidates, gold)
    dev_instance = _instance(100, 2, 7, "kala", candidates, gold)
    correctness = [False, False, False, True, True, True, True, False, True, True]
    predictions_by_c = {
        C: {100: _predictions(100, gold_first=correct, score_top=0.8, score_second=0.2)}
        for C, correct in zip(T1_C_GRID, correctness)
    }
    baseline = {100: _predictions(100, gold_first=False, score_top=0.7, score_second=0.3)}
    candidate_gramset_ids = {
        ("krl", 100, 11, "N+SG+NOM"): "1",
        ("krl", 100, 12, "N+SG+GEN"): "2",
    }
    dictionary_labels = {
        "krl": {
            (11, "1"): (("NOUN",), ("kala", "kala2")),
            (12, "2"): (("NOUN",), ("kala",)),
        }
    }
    rows = _build_output_rows(
        (dev_instance,),
        (train_instance,),
        T1_C_GRID,
        predictions_by_c,
        baseline,
        candidate_gramset_ids,
        {("krl", 7): "Перевод предложения"},
        {("krl", 7): [(1, "вчера"), (2, "kala"), (3, "пришла")]},
        dictionary_labels,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["trajectory"] == "xxx++++x++"
    assert row["n_wrong"] == 4
    assert row["n_runs"] == 10
    assert row["n_flips"] == 3
    assert row["flip_direction"] == "multiple"
    assert row["category"] == "unstable"
    assert row["baseline_correct"] == 0
    assert row["gold_pair_in_train"] == 1
    assert row["gold_rank"] == 1
    assert row["score_margin"] == pytest.approx(0.6)
    assert row["sentence"] == "вчера kala пришла"
    assert row["sentence_ru"] == "Перевод предложения"
    assert row["gold_pos"] == row["top1_pos"] == "NOUN"
    assert row["gold_lemma"] == "kala|kala2"
    assert "1:11:N+SG+NOM" in row["candidates"]
