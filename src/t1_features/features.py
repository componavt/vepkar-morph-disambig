import json
from dataclasses import dataclass


class FeatureConstructionError(ValueError):
    pass


@dataclass(frozen=True)
class ContextWord:
    language: str
    sentence_id: int
    word_id: int
    word_number: int
    word: str


@dataclass(frozen=True)
class CandidateExample:
    language: str
    word_id: int
    wordform_id: int
    gramset: str
    word: str
    word_length: int
    left_word: str
    right_word: str
    features: dict[str, float]
    label: int | None
    sample_weight: float | None


def _build_features(
    *,
    language: str,
    gramset: str,
    word: str,
    left_word: str,
    right_word: str,
) -> dict[str, float]:
    features: dict[str, float] = {}
    features[json.dumps(["gramset", gramset], ensure_ascii=False, separators=(",", ":"))] = 1.0
    features[json.dumps(["language", language], ensure_ascii=False, separators=(",", ":"))] = 1.0
    features[json.dumps(["language_gramset", language, gramset], ensure_ascii=False, separators=(",", ":"))] = 1.0
    features[json.dumps(["left_gramset", gramset, left_word], ensure_ascii=False, separators=(",", ":"))] = 1.0
    features[json.dumps(["right_gramset", gramset, right_word], ensure_ascii=False, separators=(",", ":"))] = 1.0
    for element in gramset.split("+"):
        features[json.dumps(["element", element], ensure_ascii=False, separators=(",", ":"))] = 1.0
    for length in range(1, min(4, len(word)) + 1):
        suffix = word[-length:]
        features[json.dumps(["suffix_gramset", gramset, str(length), suffix], ensure_ascii=False, separators=(",", ":"))] = 1.0
    return dict(sorted(features.items()))