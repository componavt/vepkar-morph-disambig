"""Pure in-memory comparison of two ranked prediction inputs on one dev set.

The comparison core assigns global A/B roles to two complete, validated
candidate rankings over the same strict instances, builds one row per
occurrence, and summarizes the differences overall and per language variety.
It is deterministic and never touches the corpus, Git, the filesystem, or
split selection.  Metrics reuse and per-group validation delegate to
``core.metrics``; no scores are read and no rankings are refit.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from core.data import SUPPORTED_LANGUAGES
from core.instances import Instance
from core.metrics import (
    RankedPrediction,
    RankingMetrics,
    compute_ranking_metrics,
)

_OUTCOMES = (
    "both_correct",
    "corrected",
    "regressed",
    "both_wrong_same",
    "both_wrong_changed",
)


@dataclass(frozen=True)
class OccurrenceComparison:
    """One compared occurrence of a word across both inputs."""

    language: str
    word_id: int
    word: str
    candidate_count: int
    gold_wordform_id: int
    gold_gramset: str
    a_top1_wordform_id: int
    a_top1_gramset: str
    b_top1_wordform_id: int
    b_top1_gramset: str
    a_gold_rank: int
    b_gold_rank: int
    top1_changed: bool
    outcome: str
    delta_rr: float


@dataclass(frozen=True)
class ComparisonSummary:
    """Aggregated counts and metric deltas for one group of occurrences."""

    word_count: int
    candidate_count: int
    metrics_a: RankingMetrics | None
    metrics_b: RankingMetrics | None
    delta_top1: float | None
    delta_mrr: float | None
    delta_top3: float | None
    both_correct: int
    corrected: int
    regressed: int
    both_wrong_same: int
    both_wrong_changed: int
    top1_changed: int
    gold_rank_improved: int
    gold_rank_worsened: int
    gold_rank_unchanged: int


@dataclass(frozen=True)
class RankingComparison:
    """Full comparison of two ranked prediction inputs."""

    a_input_index: int
    b_input_index: int
    selection_reason: str
    rows: tuple[OccurrenceComparison, ...]
    overall: ComparisonSummary
    by_language: dict[str, ComparisonSummary]


def _build_lookups(
    predictions: tuple[RankedPrediction, ...],
) -> tuple[dict[int, tuple[int, str]], dict[int, dict[tuple[int, str], int]]]:
    top1_by_word: dict[int, tuple[int, str]] = {}
    rank_by_identity: dict[int, dict[tuple[int, str], int]] = {}
    for prediction in predictions:
        identity = (prediction.wordform_id, prediction.gramset)
        rank_by_identity.setdefault(prediction.word_id, {})[identity] = (
            prediction.rank
        )
        if prediction.rank == 1:
            top1_by_word[prediction.word_id] = identity
    return top1_by_word, rank_by_identity


_REASON_BY_CRITERION = {
    "top1_correct": "top1",
    "mrr": "mrr",
    "top3_correct": "top3",
}


def _select_roles(
    metrics_1: RankingMetrics,
    metrics_2: RankingMetrics,
) -> tuple[int, int, str]:
    """Return (a_input_index, b_input_index, reason) with the winner as B."""
    for criterion in ("top1_correct", "mrr", "top3_correct"):
        first = getattr(metrics_1, criterion)
        second = getattr(metrics_2, criterion)
        if first > second:
            return 2, 1, _REASON_BY_CRITERION[criterion]
        if second > first:
            return 1, 2, _REASON_BY_CRITERION[criterion]
    return 1, 2, "tie"


def _summarize(
    rows: list[OccurrenceComparison],
    candidate_count: int,
    metrics_a: RankingMetrics | None,
    metrics_b: RankingMetrics | None,
) -> ComparisonSummary:
    both_correct = sum(row.outcome == "both_correct" for row in rows)
    corrected = sum(row.outcome == "corrected" for row in rows)
    regressed = sum(row.outcome == "regressed" for row in rows)
    both_wrong_same = sum(row.outcome == "both_wrong_same" for row in rows)
    both_wrong_changed = sum(row.outcome == "both_wrong_changed" for row in rows)
    top1_changed = sum(row.top1_changed for row in rows)
    improved = sum(row.b_gold_rank < row.a_gold_rank for row in rows)
    worsened = sum(row.b_gold_rank > row.a_gold_rank for row in rows)
    unchanged = sum(row.b_gold_rank == row.a_gold_rank for row in rows)

    if metrics_a is None or metrics_b is None:
        delta_top1 = delta_mrr = delta_top3 = None
    else:
        delta_top1 = metrics_b.top1_accuracy - metrics_a.top1_accuracy
        delta_mrr = metrics_b.mrr - metrics_a.mrr
        delta_top3 = metrics_b.top3_accuracy - metrics_a.top3_accuracy

    return ComparisonSummary(
        word_count=len(rows),
        candidate_count=candidate_count,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        delta_top1=delta_top1,
        delta_mrr=delta_mrr,
        delta_top3=delta_top3,
        both_correct=both_correct,
        corrected=corrected,
        regressed=regressed,
        both_wrong_same=both_wrong_same,
        both_wrong_changed=both_wrong_changed,
        top1_changed=top1_changed,
        gold_rank_improved=improved,
        gold_rank_worsened=worsened,
        gold_rank_unchanged=unchanged,
    )


def compare_rankings(
    instances: Iterable[Instance],
    predictions_1: Iterable[RankedPrediction],
    predictions_2: Iterable[RankedPrediction],
) -> RankingComparison:
    """Compare two complete ranked inputs over one strict evaluation set."""
    instance_list = tuple(instances)
    preds_1 = tuple(predictions_1)
    preds_2 = tuple(predictions_2)

    metrics_1 = compute_ranking_metrics(instance_list, preds_1)
    metrics_2 = compute_ranking_metrics(instance_list, preds_2)

    a_index, b_index, reason = _select_roles(metrics_1, metrics_2)
    a_preds = preds_1 if a_index == 1 else preds_2
    b_preds = preds_2 if b_index == 2 else preds_1
    a_metrics = metrics_1 if a_index == 1 else metrics_2
    b_metrics = metrics_2 if b_index == 2 else metrics_1

    a_top1, a_ranks = _build_lookups(a_preds)
    b_top1, b_ranks = _build_lookups(b_preds)

    instance_by_word = {instance.word_id: instance for instance in instance_list}
    languages: dict[str, list[Instance]] = defaultdict(list)
    for instance in instance_list:
        languages[instance.language].append(instance)

    rows: list[OccurrenceComparison] = []
    for instance in sorted(instance_list, key=lambda item: (item.language, item.word_id)):
        gold = instance.gold_analysis
        gold_identity = (gold.wordform_id, gold.gramset)
        a_rank = a_ranks[instance.word_id][gold_identity]
        b_rank = b_ranks[instance.word_id][gold_identity]
        a_top1_identity = a_top1[instance.word_id]
        b_top1_identity = b_top1[instance.word_id]
        a_correct = a_top1_identity == gold_identity
        b_correct = b_top1_identity == gold_identity

        if a_correct and b_correct:
            outcome = "both_correct"
        elif b_correct:
            outcome = "corrected"
        elif a_correct:
            outcome = "regressed"
        elif a_top1_identity == b_top1_identity:
            outcome = "both_wrong_same"
        else:
            outcome = "both_wrong_changed"

        rows.append(
            OccurrenceComparison(
                language=instance.language,
                word_id=instance.word_id,
                word=instance.word,
                candidate_count=len(instance.candidates),
                gold_wordform_id=gold.wordform_id,
                gold_gramset=gold.gramset,
                a_top1_wordform_id=a_top1_identity[0],
                a_top1_gramset=a_top1_identity[1],
                b_top1_wordform_id=b_top1_identity[0],
                b_top1_gramset=b_top1_identity[1],
                a_gold_rank=a_rank,
                b_gold_rank=b_rank,
                top1_changed=a_top1_identity != b_top1_identity,
                outcome=outcome,
                delta_rr=1.0 / b_rank - 1.0 / a_rank,
            )
        )

    rows_by_language: dict[str, list[OccurrenceComparison]] = defaultdict(list)
    for row in rows:
        rows_by_language[row.language].append(row)

    overall_candidate_count = sum(len(inst.candidates) for inst in instance_list)
    overall = _summarize(rows, overall_candidate_count, a_metrics, b_metrics)

    by_language: dict[str, ComparisonSummary] = {}
    for language in sorted(SUPPORTED_LANGUAGES):
        lang_instances = languages.get(language, [])
        lang_word_ids = {inst.word_id for inst in lang_instances}
        if not lang_instances:
            by_language[language] = _summarize([], 0, None, None)
            continue
        lang_a_preds = [
            pred for pred in a_preds if pred.word_id in lang_word_ids
        ]
        lang_b_preds = [
            pred for pred in b_preds if pred.word_id in lang_word_ids
        ]
        lang_metrics_a = compute_ranking_metrics(lang_instances, lang_a_preds)
        lang_metrics_b = compute_ranking_metrics(lang_instances, lang_b_preds)
        lang_candidate_count = sum(len(inst.candidates) for inst in lang_instances)
        by_language[language] = _summarize(
            rows_by_language.get(language, []),
            lang_candidate_count,
            lang_metrics_a,
            lang_metrics_b,
        )

    return RankingComparison(
        a_input_index=a_index,
        b_input_index=b_index,
        selection_reason=reason,
        rows=tuple(rows),
        overall=overall,
        by_language=by_language,
    )