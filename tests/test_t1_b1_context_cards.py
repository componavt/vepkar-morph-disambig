"""Focused checks for staged B1 questionnaires and selection logic."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "t1_b1_context_cards", ROOT / "scripts/t1-b1-context-cards.py"
)
assert SPEC is not None and SPEC.loader is not None
b1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(b1)


def test_repeated_context_spans_do_not_require_duplicate_answers():
    result = b1.repeated_context_levels(
        {"K0": (3, 3), "K1": (2, 4), "K2": (1, 5), "K3": (0, 6), "FULL": (0, 6)}
    )
    assert result["K0"]["requires_new_answer"] == 1
    assert result["FULL"]["repeat_of_level"] == "K3"
    assert result["FULL"]["requires_new_answer"] == 0


def test_response_schema_allows_multiple_analyses_and_special_statuses():
    b1.validate_response(
        {"status": "analyses_selected", "selected_options": ["A", "B"]},
        {"A", "B"},
    )
    b1.validate_response(
        {"status": "insufficient_context", "selected_options": []}, {"A", "B"}
    )
    with pytest.raises(ValueError, match="requires at least one"):
        b1.validate_response(
            {"status": "analyses_selected", "selected_options": []}, {"A", "B"}
        )
    with pytest.raises(ValueError, match="must not contain"):
        b1.validate_response(
            {"status": "data_problem", "selected_options": ["A"]}, {"A", "B"}
        )
    with pytest.raises(ValueError, match="unknown option"):
        b1.validate_response(
            {"status": "analyses_selected", "selected_options": ["C"]}, {"A", "B"}
        )


def test_primary_target_prefers_error_then_same_form_control():
    targets = [
        {
            "language": "krl", "text_id": 1, "sentence_id": 10, "word_id": 1,
            "target_id": "T1", "target_position": 1, "normalized_form": "same",
            "selection_pair": "SG+ACC ↔ SG+GEN", "gramset_correct": 1,
        },
        {
            "language": "krl", "text_id": 1, "sentence_id": 10, "word_id": 2,
            "target_id": "T2", "target_position": 2, "normalized_form": "same",
            "selection_pair": "SG+ACC ↔ SG+GEN", "gramset_correct": 0,
        },
        {
            "language": "krl", "text_id": 2, "sentence_id": 20, "word_id": 3,
            "target_id": "T3", "target_position": 1, "normalized_form": "same",
            "selection_pair": "SG+ACC ↔ SG+GEN", "gramset_correct": 1,
        },
    ]
    links = b1.assign_primary_targets(targets)
    assert [row["target_id"] for row in targets if row["is_primary_target"]] == ["T2", "T3"]
    assert targets[1]["control_quality"] == "same_form_control"
    assert targets[1]["control_target_ids_internal"] == "T1;T3"
    assert {row["target_id"] for row in links if row["same_form"]} == {"T2"}
