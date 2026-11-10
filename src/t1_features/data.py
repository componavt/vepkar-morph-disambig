"""T1 input preparation: strict split instances with full sentence context."""

from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import pandas as pd

from benchmark_context import BenchmarkContext, load_benchmark_context
from core.data import DataError, read_corpus_tables
from core.instances import Instance
from core.predictions import BenchmarkIntegrityError
from core.splits import SPLIT_NAMES, SplitError
from t1_features.features import ContextWord

TARGET_SPLITS = ("train", "dev", "test")


@dataclass(frozen=True)
class T1Inputs:
    benchmark: BenchmarkContext
    split: Literal["train", "dev", "test"]
    instances: tuple[Instance, ...]
    context_words: tuple[ContextWord, ...]
    affix_by_candidate: dict[tuple[str, int, int, str], tuple[str, ...]] = field(
        default_factory=dict
    )


def _split_lookup(
    rows: tuple[tuple[str, int, str], ...],
) -> dict[tuple[str, int], str]:
    lookup: dict[tuple[str, int], str] = {}
    for language, text_id, label in rows:
        key = (language, int(text_id))
        if label not in SPLIT_NAMES:
            raise SplitError(
                f"Unsupported split label {label!r} for {key}; "
                f"expected one of {', '.join(SPLIT_NAMES)}"
            )
        if key in lookup:
            raise SplitError(
                f"Duplicate split assignment for {key}: "
                f"assigned {lookup[key]!r} and {label!r}"
            )
        lookup[key] = label
    return lookup


def _require_assigned_instances(
    instances: tuple[Instance, ...],
    lookup: dict[tuple[str, int], str],
) -> None:
    for instance in instances:
        key = (instance.language, int(instance.text_id))
        if key not in lookup:
            raise SplitError(
                f"Strict benchmark instance {instance.language} word "
                f"{instance.word_id} (text {instance.text_id}) has no split assignment"
            )


def _require_unique_word_ids(instances: tuple[Instance, ...]) -> None:
    word_ids = [instance.word_id for instance in instances]
    if len(word_ids) != len(set(word_ids)):
        raise BenchmarkIntegrityError(
            "Strict benchmark has duplicate word_id values across instances; "
            "predictions.csv cannot identify candidates unambiguously."
        )


def _load_affix_values(
    data_dir: Path,
    language: str,
    instances: tuple[Instance, ...],
) -> dict[tuple[str, int, int, str], tuple[str, ...]]:
    """Join dictionary affix values through wordform and gramset IDs."""
    candidate_keys = {
        (language, instance.word_id, candidate.wordform_id, candidate.gramset)
        for instance in instances
        for candidate in instance.candidates
        if instance.language == language
    }
    if not candidate_keys:
        return {}

    try:
        candidates = read_corpus_tables(language, data_dir).candidates
        candidate_ids = {}
        for row in candidates.itertuples(index=False):
            key = (language, int(row.word_id), int(row.wordform_id), str(row.gramset))
            if key not in candidate_keys:
                continue
            if pd.isna(row.gramset_id):
                continue
            candidate_ids[key] = str(int(row.gramset_id))

        links_path = data_dir / "dict" / "morphology" / f"lemma_wordforms_{language}.csv.zst"
        links = pd.read_csv(
            links_path,
            compression="zstd",
            dtype=str,
            keep_default_na=False,
            usecols=["wordform_id", "gramset_id", "affix"],
        )
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        raise DataError(
            f"Could not read dictionary affix values for {language}: {exc}"
        ) from exc

    needed_pairs = {
        (key[2], gramset_id) for key, gramset_id in candidate_ids.items()
    }
    affixes_by_pair: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in links.itertuples(index=False):
        pair = (str(row.wordform_id), str(row.gramset_id))
        if pair in needed_pairs:
            affixes_by_pair[pair].add(str(row.affix))

    result: dict[tuple[str, int, int, str], tuple[str, ...]] = {}
    for key, gramset_id in candidate_ids.items():
        pair = (str(key[2]), gramset_id)
        if pair in affixes_by_pair:
            result[key] = tuple(sorted(affixes_by_pair[pair]))
    return result


def load_t1_inputs(
    data_dir: Path,
    tag: str,
    *,
    split: Literal["train", "dev", "test"],
    split_file_override: Path | None = None,
    include_affix: bool = False,
) -> T1Inputs:
    """Prepare the strict ``split`` instances and their full sentence context."""
    if split not in TARGET_SPLITS:
        raise ValueError(
            f"Unsupported benchmark split {split!r}; "
            f"expected one of {', '.join(TARGET_SPLITS)}"
        )

    benchmark = load_benchmark_context(
        data_dir, tag, split_file_override, include_words=True
    )
    lookup = _split_lookup(benchmark.split_rows)
    _require_unique_word_ids(benchmark.instances)
    _require_assigned_instances(benchmark.instances, lookup)

    selected = tuple(
        sorted(
            (
                instance
                for instance in benchmark.instances
                if lookup[(instance.language, int(instance.text_id))] == split
            ),
            key=lambda instance: (instance.language, instance.word_id),
        )
    )

    needed_sentences: dict[str, set[int]] = {}
    for instance in selected:
        needed_sentences.setdefault(instance.language, set()).add(instance.sentence_id)

    context: list[ContextWord] = []
    for language, table in benchmark.word_tables.items():
        sentence_ids = needed_sentences.get(language)
        if not sentence_ids:
            continue
        rows = table[table["sentence_id"].isin(sentence_ids)]
        for record in rows.itertuples(index=False):
            context.append(
                ContextWord(
                    language=language,
                    sentence_id=int(record.sentence_id),
                    word_id=int(record.word_id),
                    word_number=int(record.word_number),
                    word=record.word,
                )
            )

    affix_by_candidate: dict[tuple[str, int, int, str], tuple[str, ...]] = {}
    if include_affix:
        for language in sorted({instance.language for instance in selected}):
            affix_by_candidate.update(
                _load_affix_values(
                    data_dir,
                    language,
                    selected,
                )
            )

    return T1Inputs(
        benchmark=replace(benchmark, word_tables=None),
        split=split,
        instances=selected,
        context_words=tuple(
            sorted(
                context,
                key=lambda cw: (
                    cw.language,
                    cw.sentence_id,
                    cw.word_number,
                    cw.word_id,
                ),
            )
        ),
        affix_by_candidate=affix_by_candidate,
    )
