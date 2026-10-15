import math
from typing import Iterable

from core.predictions import BenchmarkIntegrityError, PredictionRow
from t1_features.features import CandidateExample


def rank_candidate_scores(
    examples: Iterable[CandidateExample],
    scores: Iterable[float],
) -> tuple[PredictionRow, ...]:
    example_list = list(examples)
    score_list = list(scores)

    if len(example_list) != len(score_list):
        raise ValueError(
            "examples and scores must have equal lengths: "
            f"{len(example_list)} examples, {len(score_list)} scores"
        )

    numeric_scores: list[float] = []
    for score in score_list:
        value = float(score)
        if not math.isfinite(value):
            raise ValueError(f"score must be finite: {score!r}")
        numeric_scores.append(value)

    language_of_word: dict[int, str] = {}
    seen_identities: set[tuple[int, int, str]] = set()
    groups: dict[tuple[str, int], list[tuple[CandidateExample, float]]] = {}

    for example, score in zip(example_list, numeric_scores):
        known_language = language_of_word.get(example.word_id)
        if known_language is None:
            language_of_word[example.word_id] = example.language
        elif known_language != example.language:
            raise BenchmarkIntegrityError(
                f"word_id {example.word_id} is associated with different languages: "
                f"{known_language!r} and {example.language!r}"
            )

        identity = (example.word_id, example.wordform_id, example.gramset)
        if identity in seen_identities:
            raise BenchmarkIntegrityError(
                f"duplicate candidate identity: {identity!r}"
            )
        seen_identities.add(identity)

        groups.setdefault((example.language, example.word_id), []).append(
            (example, score)
        )

    rows: list[PredictionRow] = []
    for (_, word_id), entries in groups.items():
        ordered = sorted(
            entries,
            key=lambda entry: (
                -entry[1],
                entry[0].wordform_id,
                entry[0].gramset,
            ),
        )
        for rank, (example, score) in enumerate(ordered, start=1):
            rows.append(
                PredictionRow(
                    word_id,
                    example.wordform_id,
                    example.gramset,
                    rank,
                    score,
                )
            )

    rows.sort(key=lambda row: (row.word_id, row.rank))
    return tuple(rows)
