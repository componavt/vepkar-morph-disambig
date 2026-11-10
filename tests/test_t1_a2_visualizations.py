"""Checks for A2 table validation and directed-graph inputs."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/t1-a2-visualizations.py"
SPEC = importlib.util.spec_from_file_location("t1_a2_visualizations", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
a2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a2)


def _conflict_rows():
    rows = []
    for index, (left, right, _title) in enumerate(a2.DIRECTION_PAIRS):
        for gold, alternative, errors, opportunities in (
            (left, right, 10 + index, 20 + index),
            (right, left, 2 + index, 40 + index),
        ):
            rows.append(
                {
                    "gold_gramset": gold,
                    "alternative_gramset": alternative,
                    "selected_error_count": str(errors),
                    "opportunity_count": str(opportunities),
                    "selected_error_rate": str(errors / opportunities),
                }
            )
    return rows


def test_prepare_direction_rows_returns_exact_six_configured_directions():
    plotted = a2.prepare_direction_rows(_conflict_rows())
    assert len(plotted) == 6
    assert [
        (row["gold_gramset"], row["alternative_gramset"])
        for row in plotted
    ] == [
        (left, right)
        for left, right, _title in a2.DIRECTION_PAIRS
        for left, right in ((left, right), (right, left))
    ]


def test_duplicate_gramset_pair_is_rejected():
    rows = _conflict_rows()
    rows.append(dict(rows[0]))
    with pytest.raises(a2.A2InputError, match="duplicate G→H pair"):
        a2.validate_conflict_rows(rows)


@pytest.mark.parametrize(
    "errors,opportunities,rate,match",
    [
        ("4", "3", "1.333", "exceeds opportunities"),
        ("0", "0", "0", "must be positive"),
        ("1", "4", "0.5", "disagrees with counts"),
        ("1", "2", "nan", "disagrees with counts"),
    ],
)
def test_invalid_conflict_counts_and_rates_are_rejected(errors, opportunities, rate, match):
    row = {
        "gold_gramset": "SG+ACC",
        "alternative_gramset": "SG+NOM",
        "selected_error_count": errors,
        "opportunity_count": opportunities,
        "selected_error_rate": rate,
    }
    with pytest.raises(a2.A2InputError, match=match):
        a2.validate_conflict_rows([row])


def test_changed_a1_csv_is_rejected_against_summary_hash(tmp_path):
    conflict_path = tmp_path / "conflict-summary.csv"
    sentence_path = tmp_path / "sentence-summary.csv"
    conflict_path.write_text(
        "gold_gramset,alternative_gramset,opportunity_count,selected_error_count,selected_error_rate\n",
        encoding="utf-8",
    )
    sentence_path.write_text(
        "language,text_id,sentence_id,benchmark_instance_count,error_count,error_rate\n",
        encoding="utf-8",
    )
    summary = {
        "target_split": "dev",
        "prediction_source": {"sha256": "frozen", "expected_sha256": "frozen"},
        "output_sha256": {
            conflict_path.name: hashlib.sha256(conflict_path.read_bytes()).hexdigest(),
            sentence_path.name: hashlib.sha256(sentence_path.read_bytes()).hexdigest(),
        },
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    conflict_path.write_text(conflict_path.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
    with pytest.raises(a2.A2InputError, match="checksum mismatch"):
        a2.load_verified_a1_tables(tmp_path)


def test_sentence_summary_handles_zero_errors_without_division():
    rows = [
        {
            "language": "krl",
            "text_id": "1",
            "sentence_id": "10",
            "benchmark_instance_count": "3",
            "error_count": "0",
            "error_rate": "0.0",
        }
    ]
    assert a2.prepare_sentence_distribution(rows) == [
        {"errors_per_sentence": "0", "sentence_count": 1, "error_instance_count": 0},
        {"errors_per_sentence": "1", "sentence_count": 0, "error_instance_count": 0},
        {"errors_per_sentence": "2+", "sentence_count": 0, "error_instance_count": 0},
    ]


def test_sentence_summary_rejects_duplicate_key_and_wrong_rate():
    row = {
        "language": "krl",
        "text_id": "1",
        "sentence_id": "10",
        "benchmark_instance_count": "2",
        "error_count": "1",
        "error_rate": "0.5",
    }
    with pytest.raises(a2.A2InputError, match="duplicate sentence key"):
        a2.prepare_sentence_distribution([row, dict(row)])
    changed_rate = dict(row, error_rate="0.25")
    with pytest.raises(a2.A2InputError, match="error_rate disagrees"):
        a2.prepare_sentence_distribution([changed_rate])
