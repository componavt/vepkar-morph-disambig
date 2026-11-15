"""Tests for inheriting a previous release's whole-text split."""

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from core.splits import (  # noqa: E402
    SplitError,
    TextWeight,
    inherit_assignments,
    read_split_csv,
    split_csv_bytes,
)


def _weight(text_id: int, language: str = "vep") -> TextWeight:
    return TextWeight(language, text_id, instance_count=2, candidate_row_count=5)


def test_known_texts_keep_split_and_new_texts_go_to_train() -> None:
    previous = {("vep", 1): "dev", ("vep", 2): "test", ("vep", 3): "train"}
    weights = (_weight(1), _weight(2), _weight(3), _weight(4))
    parts = inherit_assignments(weights, previous)
    assert [w.text_id for w in parts["dev"]] == [1]
    assert [w.text_id for w in parts["test"]] == [2]
    assert [w.text_id for w in parts["train"]] == [3, 4]


def test_language_is_part_of_the_key() -> None:
    previous = {("krl", 1): "test"}
    parts = inherit_assignments((_weight(1, "vep"),), previous)
    assert [w.text_id for w in parts["train"]] == [1]
    assert parts["test"] == ()


def test_read_split_csv_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "splits.csv"
    path.write_bytes(split_csv_bytes([("vep", 7, "dev"), ("lud", 9, "test")]))
    assert read_split_csv(path) == {("vep", 7): "dev", ("lud", 9): "test"}


def test_read_split_csv_rejects_bad_header(tmp_path: Path) -> None:
    path = tmp_path / "splits.csv"
    path.write_text("lang,text,part\nvep,1,dev\n", encoding="utf-8")
    with pytest.raises(SplitError):
        read_split_csv(path)
