"""Synthetic checks for the T1 frozen-prediction A1 tables."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.data import CorpusTables  # noqa: E402
from core.instances import Candidate, Instance, build_language_instances  # noqa: E402
from core.predictions import PredictionRow  # noqa: E402
from t1_features.error_analysis import (  # noqa: E402
    T1ErrorAnalysisError,
    build_t1_error_tables,
    validate_t1_error_table_invariants,
)
from t1_features.features import ContextWord  # noqa: E402


def _instance(
    word_id: int,
    word: str,
    candidates: tuple[Candidate, ...],
    gold: Candidate,
    *,
    text_id: int = 1,
    sentence_id: int | None = None,
    word_number: int = 1,
) -> Instance:
    sentence_id = word_id if sentence_id is None else sentence_id
    return Instance(
        language="krl",
        word_id=word_id,
        sentence_id=sentence_id,
        text_id=text_id,
        word=word,
        word_number=word_number,
        sentence_xml=f'<s id="{sentence_id}"/>',
        candidates=candidates,
        gold_analysis=gold,
    )


def _rows(word_id: int, candidates: tuple[Candidate, ...], order, scores=None):
    scores = scores or {key: 1.0 - index for index, key in enumerate(order)}
    return tuple(
        PredictionRow(word_id, wordform_id, gramset, rank, scores[(wordform_id, gramset)])
        for rank, (wordform_id, gramset) in enumerate(order, start=1)
    )


def _context(instances, extra_by_sentence=None):
    extra_by_sentence = extra_by_sentence or {}
    result = []
    seen = set()
    for instance in instances:
        key = (instance.language, instance.sentence_id)
        if key in seen:
            continue
        seen.add(key)
        records = extra_by_sentence.get(key)
        if records is None:
            records = [(instance.word_id, instance.word_number, instance.word)]
        result.extend(
            ContextWord(instance.language, instance.sentence_id, word_id, position, word)
            for word_id, position, word in records
        )
    return tuple(result)


def _build(instances, predictions, context=None):
    return build_t1_error_tables(
        instances,
        predictions,
        _context(instances) if context is None else context,
    )


def test_correct_choice_keeps_available_conflict_denominator():
    candidates = (Candidate(1, "G"), Candidate(2, "H"))
    instance = _instance(10, "form", candidates, Candidate(1, "G"))
    predictions = _rows(10, candidates, ((1, "G"), (2, "H")))
    tables = _build((instance,), predictions)
    summary = tables["conflict_summary"][0]
    assert summary["opportunity_count"] == 1
    assert summary["selected_error_count"] == 0
    assert tables["instances"][0]["is_correct"] == 1


def test_wrong_alternative_is_counted_for_one_available_conflict():
    candidates = (Candidate(1, "G"), Candidate(2, "H"))
    instance = _instance(10, "form", candidates, Candidate(1, "G"))
    predictions = _rows(10, candidates, ((2, "H"), (1, "G")))
    tables = _build((instance,), predictions)
    assert tables["conflict_summary"] == [
        {
            "gold_gramset": "G",
            "alternative_gramset": "H",
            "opportunity_count": 1,
            "selected_error_count": 1,
            "selected_error_rate": 1.0,
        }
    ]


def test_error_identity_breakdown_separates_same_wordform_from_other_ids():
    same_form_candidates = (Candidate(1, "G"), Candidate(1, "H"))
    other_form_candidates = (Candidate(2, "G"), Candidate(3, "H"))
    instances = (
        _instance(10, "a", same_form_candidates, Candidate(1, "G")),
        _instance(11, "b", other_form_candidates, Candidate(2, "G")),
    )
    predictions = (
        *_rows(10, same_form_candidates, ((1, "H"), (1, "G"))),
        *_rows(11, other_form_candidates, ((3, "H"), (2, "G"))),
    )
    tables = _build(instances, predictions)
    categories = {row["word_id"]: row["error_category"] for row in tables["instances"]}
    assert categories == {
        10: "same_wordform_id_different_gramset",
        11: "different_wordform_id_different_gramset",
    }


def test_multiple_candidates_with_same_alternative_gramset_make_one_opportunity():
    candidates = (Candidate(1, "G"), Candidate(2, "H"), Candidate(3, "H"))
    instance = _instance(10, "form", candidates, Candidate(1, "G"))
    predictions = _rows(10, candidates, ((2, "H"), (3, "H"), (1, "G")))
    tables = _build((instance,), predictions)
    assert len(tables["conflict_opportunities"]) == 1
    assert tables["conflict_opportunities"][0]["alternative_candidate_count"] == 2


def test_same_gramset_different_wordform_id_is_separate_from_grammar_conflicts():
    candidates = (Candidate(1, "G"), Candidate(2, "G"), Candidate(3, "H"))
    instance = _instance(10, "form", candidates, Candidate(1, "G"))
    predictions = _rows(10, candidates, ((2, "G"), (1, "G"), (3, "H")))
    tables = _build((instance,), predictions)
    assert tables["same_gramset_alternatives"][0]["selected_error"] == 1
    assert tables["confusions"] == [{
        "gold_gramset": "G", "top1_gramset": "G", "confusion_count": 1
    }]
    assert all(row["alternative_gramset"] != "G" for row in tables["conflict_opportunities"])


def test_zero_margin_wrong_choice_uses_id_tie_break_without_feature_claim():
    candidates = (Candidate(10, "G"), Candidate(5, "G"))
    instance = _instance(10, "form", candidates, Candidate(10, "G"))
    scores = {(5, "G"): 2.0, (10, "G"): 2.0}
    predictions = _rows(10, candidates, ((5, "G"), (10, "G")), scores)
    tables = _build((instance,), predictions)
    row = tables["instances"][0]
    assert row["is_correct"] == 0
    assert row["top1_minus_gold_score"] == 0.0
    assert row["same_gramset_as_gold"] == 1
    assert row["error_category"] == "different_wordform_id_same_gramset"


def test_sentence_tables_keep_correct_instances_and_link_multiple_errors():
    instances = (
        _instance(10, "a", (Candidate(1, "G"), Candidate(2, "H")), Candidate(1, "G"), sentence_id=50, word_number=1),
        _instance(11, "b", (Candidate(3, "G"), Candidate(4, "H")), Candidate(3, "G"), sentence_id=50, word_number=2),
        _instance(12, "c", (Candidate(5, "G"), Candidate(6, "H")), Candidate(5, "G"), sentence_id=50, word_number=3),
    )
    predictions = (
        *_rows(10, instances[0].candidates, ((2, "H"), (1, "G"))),
        *_rows(11, instances[1].candidates, ((3, "G"), (4, "H"))),
        *_rows(12, instances[2].candidates, ((6, "H"), (5, "G"))),
    )
    context = _context(instances, {
        ("krl", 50): [(10, 1, "a"), (11, 2, "b"), (12, 3, "c")]
    })
    tables = _build(instances, predictions, context)
    summary = tables["sentence_summary"][0]
    assert summary["benchmark_instance_count"] == 3
    assert summary["error_count"] == 2
    assert summary["sentence"] == "a b c"
    assert len(tables["sentence_instance_links"]) == 3
    assert len(tables["sentence_error_links"]) == 2


def test_repeated_form_group_includes_correct_and_incorrect_instances():
    candidates_a = (Candidate(1, "G"), Candidate(2, "H"))
    candidates_b = (Candidate(3, "G"), Candidate(4, "H"))
    instances = (
        _instance(10, "oma", candidates_a, Candidate(1, "G"), sentence_id=50),
        _instance(11, "oma", candidates_b, Candidate(3, "G"), sentence_id=51),
    )
    predictions = (
        *_rows(10, candidates_a, ((2, "H"), (1, "G"))),
        *_rows(11, candidates_b, ((3, "G"), (4, "H"))),
    )
    tables = _build(instances, predictions)
    form = tables["form_summary"][0]
    assert form["instance_count"] == 2
    assert form["error_count"] == 1
    assert form["sentence_count"] == 2
    assert len(tables["form_instance_links"]) == 2


def test_missing_prediction_candidate_is_rejected():
    candidates = (Candidate(1, "G"), Candidate(2, "H"))
    instance = _instance(10, "form", candidates, Candidate(1, "G"))
    with pytest.raises(T1ErrorAnalysisError, match="candidate mismatch"):
        _build((instance,), _rows(10, candidates, ((1, "G"),)))


def test_extra_prediction_candidate_is_rejected():
    candidates = (Candidate(1, "G"), Candidate(2, "H"))
    instance = _instance(10, "form", candidates, Candidate(1, "G"))
    predictions = _rows(10, candidates, ((1, "G"), (2, "H"), (3, "J")))
    with pytest.raises(T1ErrorAnalysisError, match="candidate mismatch"):
        _build((instance,), predictions)


def test_duplicate_full_prediction_key_is_rejected():
    candidates = (Candidate(1, "G"), Candidate(2, "H"))
    instance = _instance(10, "form", candidates, Candidate(1, "G"))
    predictions = (
        PredictionRow(10, 1, "G", 1, 2.0),
        PredictionRow(10, 1, "G", 2, 1.0),
    )
    with pytest.raises(T1ErrorAnalysisError, match="duplicate prediction candidate key"):
        _build((instance,), predictions)


def test_gold_missing_from_candidate_list_is_rejected():
    candidates = (Candidate(1, "G"), Candidate(2, "H"))
    instance = _instance(10, "form", candidates, Candidate(3, "X"))
    predictions = _rows(10, candidates, ((1, "G"), (2, "H")))
    with pytest.raises(T1ErrorAnalysisError, match="expected one gold candidate"):
        _build((instance,), predictions)


def test_strict_benchmark_excludes_zero_or_multiple_gold_candidates():
    texts = pd.DataFrame({"text_id": [1], "corpus_ids": [()], "dialect_code": [""], "genre_ids": [()], "year_recorded": [pd.NA]})
    sentences = pd.DataFrame({
        "sentence_id": [10, 11], "text_id": [1, 1],
        "sentence_xml": ["<s/>", "<s/>"], "sentence_ru": ["", ""],
    })
    words = pd.DataFrame({
        "word_id": [100, 101], "sentence_id": [10, 11], "word_number": [1, 1], "word": ["zero", "multiple"],
    })
    candidates = pd.DataFrame({
        "word_id": [100, 100, 101, 101],
        "wordform_id": [1, 2, 3, 4],
        "gramset_id": [1, 2, 3, 4],
        "gramset": ["A", "B", "C", "D"],
        "relevance": [1, 1, 2, 2],
    })
    result = build_language_instances("krl", CorpusTables(texts, sentences, words, candidates))
    assert result.instances == ()
