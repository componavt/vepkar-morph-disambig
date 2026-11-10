import json
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping

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
    *,
    context_window: int = 1,
) -> dict[
    tuple[str, int],
    tuple[tuple[str | None, ...], tuple[str | None, ...]],
]:
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

    resolved: dict[
        tuple[str, int],
        tuple[tuple[str | None, ...], tuple[str | None, ...]],
    ] = {}
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
        counts = Counter(numbers)
        if len(counts) != len(numbers):
            duplicate = next(number for number in numbers if counts[number] > 1)
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
            left_context = []
            right_context = []
            for distance in range(1, context_window + 1):
                left_index = index - distance
                if left_index >= 0:
                    left_context.append(records[left_index].word)
                elif distance == index + 1:
                    left_context.append("<BOS>")
                else:
                    left_context.append(None)

                right_index = index + distance
                if right_index < len(records):
                    right_context.append(records[right_index].word)
                elif distance == len(records) - index:
                    right_context.append("<EOS>")
                else:
                    right_context.append(None)

            resolved[(language, target.word_id)] = (
                tuple(left_context),
                tuple(right_context),
            )

    return dict(sorted(resolved.items()))


def _build_features(
    *,
    language: str,
    gramset: str,
    word: str,
    left_word: str,
    right_word: str,
    left_context: tuple[str | None, ...],
    right_context: tuple[str | None, ...],
    include_left_context: bool = True,
    include_right_context: bool = True,
    suffix_max_length: int = 4,
    affix_values: tuple[str, ...] | None = None,
) -> dict[str, float]:
    if type(suffix_max_length) is not int or suffix_max_length < 0:
        raise FeatureConstructionError(
            "suffix_max_length must be a nonnegative integer, "
            f"got {suffix_max_length!r}"
        )
    features: dict[str, float] = {}
    features[json.dumps(["gramset", gramset], ensure_ascii=False, separators=(",", ":"))] = 1.0
    features[json.dumps(["language", language], ensure_ascii=False, separators=(",", ":"))] = 1.0
    features[json.dumps(["language_gramset", language, gramset], ensure_ascii=False, separators=(",", ":"))] = 1.0
    if len(left_context) != len(right_context):
        raise FeatureConstructionError(
            "left and right context windows must have equal lengths"
        )
    if not left_context:
        raise FeatureConstructionError("context window must contain the nearest words")
    if left_context[0] != left_word or right_context[0] != right_word:
        raise FeatureConstructionError(
            "immediate context fields must match the first context-window positions"
        )

    for side, values, enabled in (
        ("left", left_context, include_left_context),
        ("right", right_context, include_right_context),
    ):
        if not enabled:
            continue
        for distance, context_word in enumerate(values, start=1):
            if context_word is None:
                continue
            if distance == 1:
                feature = [f"{side}_gramset", gramset, context_word]
            else:
                feature = [
                    f"{side}_gramset_offset",
                    gramset,
                    str(distance),
                    context_word,
                ]
            features[
                json.dumps(feature, ensure_ascii=False, separators=(",", ":"))
            ] = 1.0
    for element in gramset.split("+"):
        features[json.dumps(["element", element], ensure_ascii=False, separators=(",", ":"))] = 1.0
    for length in range(1, min(suffix_max_length, len(word)) + 1):
        suffix = word[-length:]
        features[json.dumps(["suffix_gramset", gramset, str(length), suffix], ensure_ascii=False, separators=(",", ":"))] = 1.0
    if affix_values is not None:
        if not affix_values:
            features[
                json.dumps(
                    ["affix_status", gramset, "no_link"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            ] = 1.0
        else:
            for affix in sorted(set(affix_values)):
                if affix == "#":
                    feature = ["affix_status", gramset, "not_stem_remainder"]
                elif affix == r"\N":
                    feature = ["affix_status", gramset, "unknown"]
                else:
                    feature = ["affix_value", gramset, affix]
                features[
                    json.dumps(
                        feature,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                ] = 1.0
    return dict(sorted(features.items()))


def build_candidate_examples(
    instances: Iterable[Instance],
    context_words: Iterable[ContextWord],
    *,
    mode: Literal["train", "predict"],
    include_left_context: bool = True,
    include_right_context: bool = True,
    context_window: int = 1,
    suffix_max_length: int = 4,
    affix_by_candidate: Mapping[
        tuple[str, int, int, str], tuple[str, ...]
    ] | None = None,
    include_affix: bool = False,
) -> tuple[CandidateExample, ...]:
    if mode not in ("train", "predict"):
        raise FeatureConstructionError(f"unsupported mode: {mode!r}")
    if type(include_left_context) is not bool:
        raise FeatureConstructionError(
            "include_left_context must be a boolean, "
            f"got {include_left_context!r}"
        )
    if type(include_right_context) is not bool:
        raise FeatureConstructionError(
            "include_right_context must be a boolean, "
            f"got {include_right_context!r}"
        )
    if type(context_window) is not int or not 1 <= context_window <= 3:
        raise FeatureConstructionError(
            "context_window must be an integer from 1 through 3, "
            f"got {context_window!r}"
        )

    materialized_instances = list(instances)
    if not materialized_instances:
        return ()

    context = _resolve_context(
        materialized_instances,
        context_words,
        context_window=context_window,
    )

    examples: list[CandidateExample] = []
    for instance in materialized_instances:
        if mode == "train":
            matches = [
                candidate
                for candidate in instance.candidates
                if (candidate.wordform_id, candidate.gramset)
                == (instance.gold_analysis.wordform_id, instance.gold_analysis.gramset)
            ]
            if len(matches) != 1:
                raise FeatureConstructionError(
                    f"{instance.language}: sentence {instance.sentence_id}, "
                    f"word {instance.word_id}: expected exactly one candidate matching "
                    f"gold analysis, found {len(matches)}"
                )

        left_context, right_context = context[(instance.language, instance.word_id)]
        left_word = left_context[0]
        right_word = right_context[0]
        if left_word is None or right_word is None:
            raise FeatureConstructionError(
                f"{instance.language}: word {instance.word_id}: "
                "nearest context is missing without a boundary marker"
            )
        for candidate in instance.candidates:
            if mode == "train":
                label = (
                    1
                    if (candidate.wordform_id, candidate.gramset)
                    == (instance.gold_analysis.wordform_id, instance.gold_analysis.gramset)
                    else 0
                )
                sample_weight = 1.0 / len(instance.candidates)
            else:
                label = None
                sample_weight = None
            examples.append(
                CandidateExample(
                    language=instance.language,
                    word_id=instance.word_id,
                    wordform_id=candidate.wordform_id,
                    gramset=candidate.gramset,
                    word=instance.word,
                    word_length=len(instance.word),
                    left_word=left_word,
                    right_word=right_word,
                    features=_build_features(
                        language=instance.language,
                        gramset=candidate.gramset,
                        word=instance.word,
                        left_word=left_word,
                        right_word=right_word,
                        left_context=left_context,
                        right_context=right_context,
                        include_left_context=include_left_context,
                        include_right_context=include_right_context,
                        suffix_max_length=suffix_max_length,
                        affix_values=(
                            (
                                affix_by_candidate.get(
                                    (
                                        instance.language,
                                        instance.word_id,
                                        candidate.wordform_id,
                                        candidate.gramset,
                                    ),
                                    (),
                                )
                                if affix_by_candidate is not None
                                else ()
                            )
                            if include_affix
                            else None
                        ),
                    ),
                    label=label,
                    sample_weight=sample_weight,
                )
            )

    return tuple(
        sorted(
            examples,
            key=lambda example: (
                example.language,
                example.word_id,
                example.wordform_id,
                example.gramset,
            ),
        )
    )
