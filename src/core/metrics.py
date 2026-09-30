"""Pure in-memory ranking metrics over one strict evaluation split.

The metrics core ranks each evaluated word by the position of its exact gold
candidate identity ``(wordform_id, gramset)`` among complete prediction rows.
It is model-independent (no score is inspected), deterministic, and never
touches the corpus, Git, the filesystem, or split selection.  Full candidate
completeness, rank permutations, and score/rank consistency are prerequisites
enforced by the prediction CSV validator before the CLI calls this function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Protocol

from core.instances import Instance


class MetricsInputError(ValueError):
    """The evaluation set or prediction rows cannot yield meaningful metrics."""


@dataclass(frozen=True)
class RankingMetrics:
    """Evaluation metrics for one complete set of ranked predictions.

    All accuracy values are unrounded fractions in ``[0, 1]``; presentation
    rounding belongs to the later CLI layer.
    """

    word_count: int
    top1_correct: int
    top3_correct: int
    top1_accuracy: float
    mrr: float
    top3_accuracy: float


class RankedPrediction(Protocol):
    """One ranked candidate row, identified by exact candidate identity."""

    @property
    def word_id(self) -> int: ...

    @property
    def wordform_id(self) -> int: ...

    @property
    def gramset(self) -> str: ...

    @property
    def rank(self) -> int: ...


def compute_ranking_metrics(
    instances: Iterable[Instance],
    predictions: Iterable[RankedPrediction],
) -> RankingMetrics:
    """Compute Top-1, MRR, and Top-3 over the supplied instances and rows.

    Each evaluated word contributes exactly one rank: that of its gold
    candidate.  Words are weighted equally regardless of candidate count.
    Inputs are materialized once; lookup structures are built in a single
    pass so generator inputs are not exhausted prematurely.
    """
    instance_list = tuple(instances)
    if not instance_list:
        raise MetricsInputError("the evaluation set is empty")
    word_ids = [instance.word_id for instance in instance_list]
    if len(word_ids) != len(set(word_ids)):
        raise MetricsInputError(
            "duplicate Instance.word_id values across the evaluation set; "
            "each word must appear exactly once"
        )

    evaluated_ids = set(word_ids)
    rank_by_identity: dict[tuple[int, int, str], int] = {}
    for prediction in predictions:
        if prediction.word_id not in evaluated_ids:
            raise MetricsInputError(
                f"prediction word_id={prediction.word_id} is not in the "
                "supplied evaluation set"
            )
        identity = (
            prediction.word_id,
            prediction.wordform_id,
            prediction.gramset,
        )
        if identity in rank_by_identity:
            raise MetricsInputError(
                f"duplicate prediction candidate {identity!r}"
            )
        rank_by_identity[identity] = prediction.rank

    gold_ranks: list[int] = []
    for instance in instance_list:
        gold = instance.gold_analysis
        identity = (instance.word_id, gold.wordform_id, gold.gramset)
        rank = rank_by_identity.get(identity)
        if rank is None:
            raise MetricsInputError(
                f"gold candidate {(gold.wordform_id, gold.gramset)!r} of "
                f"word_id={instance.word_id} is missing from predictions"
            )
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise MetricsInputError(
                f"gold candidate {identity!r} has nonpositive or "
                f"non-integer rank {rank!r}"
            )
        gold_ranks.append(rank)

    word_count = len(gold_ranks)
    top1_correct = sum(rank == 1 for rank in gold_ranks)
    top3_correct = sum(rank <= 3 for rank in gold_ranks)
    top1_accuracy = top1_correct / word_count
    top3_accuracy = top3_correct / word_count
    mrr = math.fsum(1.0 / rank for rank in gold_ranks) / word_count
    return RankingMetrics(
        word_count=word_count,
        top1_correct=top1_correct,
        top3_correct=top3_correct,
        top1_accuracy=top1_accuracy,
        mrr=mrr,
        top3_accuracy=top3_accuracy,
    )