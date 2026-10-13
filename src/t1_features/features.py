import json
from dataclasses import dataclass
from typing import Iterable

from core.instances import Instance


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


def _resolve_context(
    instances: Iterable[Instance],
    context_words: Iterable[ContextWord],
) -> dict[tuple[str, int], tuple[str, str]]:
    materialized_instances = list(instances)
    if not materialized_instances:
        return {}

    needed_sentences: set[tuple[str, int]] = set()
    instances_by_sentence: dict[tuple[str, int], list[Instance]] = {}
    for instance in materialized_instances:
        sentence_key = (instance.language, instance.sentence_id)
        needed_sentences.add(sentence_key)
        instances_by_sentence.setdefault(sentence_key, []).append(instance)

    records_by_sentence: dict[tuple[str, int], list[ContextWord]] = {}
    for record in context_words:
        sentence_key = (record.language, record.sentence_id)
        if sentence_key in needed_sentences:
            records_by_sentence.setdefault(sentence_key, []).append(record)

    resolved: dict[tuple[str, int], tuple[str, str]] = {}
    for sentence_key, targets in instances_by_sentence.items():
        language, sentence_id = sentence_key
        records = records_by_sentence.get(sentence_key, [])
        affected = targets[0]

        for record in records:
            if record.word_number <= 0:
                raise FeatureConstructionError(
                    f"{language}: sentence {sentence_id}, target word {affected.word_id}: "
                    f"nonpositive word_number {record.word_number} for word {record.word_id} in sentence"
                )

        numbers = [record.word_number for record in records]
        if len(numbers) != len(set(numbers)):
            duplicate = next(number for number in numbers if numbers.count(number) > 1)
            raise FeatureConstructionError(
                f"{language}: sentence {sentence_id}, target word {affected.word_id}: "
                f"duplicate word_number {duplicate} in sentence"
            )

        records.sort(key=lambda record: record.word_number)

        records_by_id: dict[int, list[ContextWord]] = {}
        position_index: dict[int, int] = {}
        for index, record in enumerate(records):
            records_by_id.setdefault(record.word_id, []).append(record)
            position_index[record.word_number] = index

        for target in targets:
            matches = records_by_id.get(target.word_id, [])
            if not matches:
                raise FeatureConstructionError(
                    f"{language}: sentence {sentence_id}, target word {target.word_id}: "
                    "target word missing from sentence context"
                )
            if len(matches) > 1:
                raise FeatureConstructionError(
                    f"{language}: sentence {sentence_id}, target word {target.word_id}: "
                    f"expected exactly one context record, found {len(matches)}"
                )
            match = matches[0]
            if match.word_number != target.word_number:
                raise FeatureConstructionError(
                    f"{language}: sentence {sentence_id}, target word {target.word_id}: "
                    f"position mismatch, expected {target.word_number}, got {match.word_number}"
                )
            if match.word != target.word:
                raise FeatureConstructionError(
                    f"{language}: sentence {sentence_id}, target word {target.word_id}: "
                    f"word mismatch, expected {target.word!r}, got {match.word!r}"
                )
            index = position_index[match.word_number]
            left = records[index - 1].word if index > 0 else "<BOS>"
            right = records[index + 1].word if index + 1 < len(records) else "<EOS>"
            resolved[(language, target.word_id)] = (left, right)

    return dict(sorted(resolved.items()))


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