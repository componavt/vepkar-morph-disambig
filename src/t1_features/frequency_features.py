"""Train-only frequency features for T1 candidate ranking experiments."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Literal

from core.instances import Instance

FrequencyChannel = Literal[
    "analysis_pooled",
    "analysis_language",
    "gramset_pooled",
    "gramset_language",
]

FREQUENCY_CHANNELS: tuple[FrequencyChannel, ...] = (
    "analysis_pooled",
    "analysis_language",
    "gramset_pooled",
    "gramset_language",
)


@dataclass(frozen=True)
class TrainFrequencyCounts:
    """Gold-analysis counts computed from train instances only."""

    analysis_pooled: Counter[tuple[int, str]]
    analysis_language: Counter[tuple[str, int, str]]
    gramset_pooled: Counter[str]
    gramset_language: Counter[tuple[str, str]]


class FrequencyFeatureError(ValueError):
    """Invalid frequency-channel configuration or feature value."""


def count_train_frequencies(
    train_instances: Iterable[Instance],
) -> TrainFrequencyCounts:
    """Count gold analysis and gramset occurrences in the supplied train split."""
    analysis_pooled: Counter[tuple[int, str]] = Counter()
    analysis_language: Counter[tuple[str, int, str]] = Counter()
    gramset_pooled: Counter[str] = Counter()
    gramset_language: Counter[tuple[str, str]] = Counter()
    for instance in train_instances:
        wordform_id = instance.gold_analysis.wordform_id
        gramset = instance.gold_analysis.gramset
        analysis_pooled[(wordform_id, gramset)] += 1
        analysis_language[(instance.language, wordform_id, gramset)] += 1
        gramset_pooled[gramset] += 1
        gramset_language[(instance.language, gramset)] += 1
    return TrainFrequencyCounts(
        analysis_pooled=analysis_pooled,
        analysis_language=analysis_language,
        gramset_pooled=gramset_pooled,
        gramset_language=gramset_language,
    )


def candidate_frequency_features(
    language: str,
    wordform_id: int,
    gramset: str,
    counts: TrainFrequencyCounts,
    channels: Iterable[FrequencyChannel],
) -> dict[str, float]:
    """Encode selected counts as log1p values plus explicit zero-count flags."""
    selected = tuple(channels)
    if len(selected) != len(set(selected)) or any(
        channel not in FREQUENCY_CHANNELS for channel in selected
    ):
        raise FrequencyFeatureError(f"invalid frequency channels: {selected!r}")

    raw_counts = {
        "analysis_pooled": counts.analysis_pooled.get((wordform_id, gramset), 0),
        "analysis_language": counts.analysis_language.get(
            (language, wordform_id, gramset), 0
        ),
        "gramset_pooled": counts.gramset_pooled.get(gramset, 0),
        "gramset_language": counts.gramset_language.get((language, gramset), 0),
    }
    features: dict[str, float] = {}
    for channel in selected:
        count = raw_counts[channel]
        value_feature = json.dumps(
            ["train_frequency", channel, "log1p"],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if count > 0:
            value = math.log1p(count)
            if not math.isfinite(value):
                raise FrequencyFeatureError(
                    f"non-finite log1p frequency for {channel}: {count}"
                )
            features[value_feature] = value
        else:
            zero_feature = json.dumps(
                ["train_frequency", channel, "zero"],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            features[zero_feature] = 1.0
    return features
