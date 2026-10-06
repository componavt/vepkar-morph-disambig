"""Pure, in-memory diagnostic rows for the frequency baseline on strict dev instances.

Each row summarizes one already selected strict ``dev`` instance: the identity
and train frequency of the frequency baseline's Top-1 candidate and of the
gold analysis, together with the gold rank.  The module is fully
deterministic and in-memory: it performs no split selection, validation,
ranking, or frequency counting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from core.frequency import FrequencyPrediction
from core.instances import Instance
from core.predictions import BenchmarkIntegrityError


class DiagnosticMappingError(ValueError):
    """A prediction-to-instance mapping assumption was broken."""


@dataclass(frozen=True)
class FrequencyDiagnostic:
    """One frequency-baseline diagnostic row for one strict dev instance."""

    language: str
    word_id: int
    word: str
    candidate_count: int
    gold_wordform_id: int
    gold_gramset: str
    top1_wordform_id: int
    top1_gramset: str
    gold_rank: int
    gold_train_frequency: int
    top1_train_frequency: int


def _raise_mapping_error(word_id: int, reason: str) -> None:
    raise DiagnosticMappingError(f"word_id={word_id}: {reason}")


def _require_nonnegative_int(score: object, word_id: int, role: str) -> None:
    if type(score) is not int or score < 0:
        _raise_mapping_error(
            word_id, f"{role} score {score!r} is not a nonnegative integer"
        )


def build_frequency_diagnostics(
    instances: Iterable[Instance],
    predictions: Iterable[FrequencyPrediction],
) -> tuple[FrequencyDiagnostic, ...]:
    """Build one diagnostic row per strict dev instance.

    Caller preconditions, established and not re-verified here: ``instances``
    are already strict dev instances; ``predictions`` have passed
    :func:`core.predictions.validate_predictions`; their dev membership and
    frequency-baseline identity are established.  This function does not
    select a split, call the validator, regenerate rankings, count train
    frequencies, or prove baseline identity.

    Top-1 is the row with ``rank == 1``; gold is the row whose ``(wordform_id,
    gramset)`` equals the instance gold analysis identity.  Output rows are
    ordered by ``(language, word_id)``.  Inputs are not mutated; both
    iterables may be single-pass.
    """
    instance_list = tuple(instances)
    instances_by_id: dict[int, Instance] = {}
    for instance in instance_list:
        if instance.word_id in instances_by_id:
            raise BenchmarkIntegrityError(
                "Strict benchmark has duplicate word_id values across "
                "instances; predictions cannot identify candidates "
                "unambiguously."
            )
        instances_by_id[instance.word_id] = instance

    rows_by_word: dict[int, list[FrequencyPrediction]] = {}
    seen_identities: set[tuple[int, int, str]] = set()
    for prediction in tuple(predictions):
        if prediction.word_id not in instances_by_id:
            _raise_mapping_error(
                prediction.word_id, "prediction references an unknown input word_id"
            )
        identity = (prediction.word_id, prediction.wordform_id, prediction.gramset)
        if identity in seen_identities:
            _raise_mapping_error(
                prediction.word_id, f"duplicate prediction identity {identity!r}"
            )
        seen_identities.add(identity)
        rows_by_word.setdefault(prediction.word_id, []).append(prediction)

    diagnostics: list[FrequencyDiagnostic] = []
    for instance in instance_list:
        word_id = instance.word_id
        rows = rows_by_word.get(word_id)
        if not rows:
            _raise_mapping_error(word_id, "missing predictions")
        top1_rows = [row for row in rows if row.rank == 1]
        if len(top1_rows) != 1:
            _raise_mapping_error(
                word_id, f"expected exactly one rank-1 row; found {len(top1_rows)}"
            )
        gold_key = (
            instance.gold_analysis.wordform_id,
            instance.gold_analysis.gramset,
        )
        gold_rows = [
            row for row in rows if (row.wordform_id, row.gramset) == gold_key
        ]
        if len(gold_rows) != 1:
            _raise_mapping_error(word_id, "missing gold row")
        top1 = top1_rows[0]
        gold = gold_rows[0]
        candidate_keys = {
            (candidate.wordform_id, candidate.gramset)
            for candidate in instance.candidates
        }
        if gold_key not in candidate_keys:
            _raise_mapping_error(
                word_id, "gold analysis absent from strict candidates"
            )
        top1_key = (top1.wordform_id, top1.gramset)
        if top1_key not in candidate_keys:
            _raise_mapping_error(
                word_id, "top1 candidate absent from strict candidates"
            )
        candidate_count = len(instance.candidates)
        if not 1 <= gold.rank <= candidate_count:
            _raise_mapping_error(
                word_id, f"gold rank {gold.rank} outside 1..{candidate_count}"
            )
        _require_nonnegative_int(gold.score, word_id, "gold")
        _require_nonnegative_int(top1.score, word_id, "top1")
        diagnostics.append(
            FrequencyDiagnostic(
                language=instance.language,
                word_id=word_id,
                word=instance.word,
                candidate_count=candidate_count,
                gold_wordform_id=gold.wordform_id,
                gold_gramset=gold.gramset,
                top1_wordform_id=top1.wordform_id,
                top1_gramset=top1.gramset,
                gold_rank=gold.rank,
                gold_train_frequency=gold.score,
                top1_train_frequency=top1.score,
            )
        )

    return tuple(
        sorted(diagnostics, key=lambda row: (row.language, row.word_id))
    )