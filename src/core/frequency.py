"""Pure frequency-baseline ranking over strict benchmark instances.

The baseline learns candidate frequencies from the gold analyses of the train
split only, then ranks every strict candidate of every target ``dev`` or
``test`` word by how often that candidate identity was the gold analysis in
train.  The module is fully deterministic and in-memory: it never touches the
corpus, Git, the filesystem, or any metrics.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Literal

from core.instances import Instance

TARGET_SPLITS = ("dev", "test")


class TargetSplitError(ValueError):
    """The requested target split is not supported by the frequency baseline."""


@dataclass(frozen=True)
class FrequencyPrediction:
    """One ranked candidate row for a target word, ready for CSV output."""

    word_id: int
    wordform_id: int
    gramset: str
    rank: int
    score: int


def _build_split_lookup(
    split_rows: Iterable[tuple[str, int, str]],
) -> dict[tuple[str, int], str]:
    return {
        (language, int(text_id)): split
        for language, text_id, split in split_rows
    }


def _count_train_frequency(
    instances: Iterable[Instance],
    split_by_text: dict[tuple[str, int], str],
) -> Counter[tuple[int, str]]:
    frequency: Counter[tuple[int, str]] = Counter()
    for instance in instances:
        key = (instance.language, int(instance.text_id))
        if split_by_text.get(key) != "train":
            continue
        gold = instance.gold_analysis
        frequency[(gold.wordform_id, gold.gramset)] += 1
    return frequency


def build_train_frequency(
    instances: Iterable[Instance],
    split_rows: Iterable[tuple[str, int, str]],
) -> Counter[tuple[int, str]]:
    """Count gold candidate identities of all train instances.

    ``split_rows`` are ``(language, text_id, split)`` rows in the shape of
    :func:`core.splits.build_split_rows`.  Each train instance contributes
    exactly one count: its strict gold analysis keyed by ``(wordform_id,
    gramset)``.  Candidate identities that never appear as a train gold
    analysis are simply absent from the returned counter.
    """
    return _count_train_frequency(instances, _build_split_lookup(split_rows))


def rank_by_train_frequency(
    instances: Iterable[Instance],
    split_rows: Iterable[tuple[str, int, str]],
    split: Literal["dev", "test"],
) -> tuple[FrequencyPrediction, ...]:
    """Rank every strict candidate of every target word by train frequency.

    Frequencies always come from the ``train`` split.  Each target instance
    contributes all of its candidates, ranked by score descending, then
    ``wordform_id`` ascending, then ``gramset`` ascending; ranks are the
    consecutive integers ``1..N`` per word.  Output rows are ordered by
    ``word_id`` and then rank, independent of the input iteration order.
    """
    if split not in TARGET_SPLITS:
        raise TargetSplitError(
            f"Unsupported target split {split!r}; "
            f"expected one of {TARGET_SPLITS}"
        )
    instance_list = tuple(instances)
    split_by_text = _build_split_lookup(split_rows)
    train_frequency = _count_train_frequency(instance_list, split_by_text)
    target_texts = {
        key for key, part in split_by_text.items() if part == split
    }

    rows_by_word: dict[int, list[FrequencyPrediction]] = {}
    for instance in instance_list:
        key = (instance.language, int(instance.text_id))
        if key not in target_texts:
            continue
        ranked = [
            FrequencyPrediction(
                word_id=instance.word_id,
                wordform_id=candidate.wordform_id,
                gramset=candidate.gramset,
                rank=rank,
                score=train_frequency.get(
                    (candidate.wordform_id, candidate.gramset), 0
                ),
            )
            for rank, candidate in enumerate(
                sorted(
                    instance.candidates,
                    key=lambda candidate: (
                        -train_frequency.get(
                            (candidate.wordform_id, candidate.gramset), 0
                        ),
                        candidate.wordform_id,
                        candidate.gramset,
                    ),
                ),
                start=1,
            )
        ]
        rows_by_word[instance.word_id] = ranked

    output: list[FrequencyPrediction] = []
    for word_id in sorted(rows_by_word):
        output.extend(rows_by_word[word_id])
    return tuple(output)