"""Validated diagnostic tables for one frozen T1 dev predictions file."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from core.instances import Instance
from core.predictions import PredictionRow


class T1ErrorAnalysisError(ValueError):
    """The frozen predictions or benchmark cannot support diagnostic tables."""


FROZEN_PREDICTIONS_SHA256 = "41451f143cf49da828d02ef64e0804de5501aaa106acdc2f8955b20c27b76c02"


def validate_frozen_prediction_hash(actual_sha256: str) -> None:
    """Reject any prediction file other than the approved frozen A1 input."""
    if actual_sha256 != FROZEN_PREDICTIONS_SHA256:
        raise T1ErrorAnalysisError(
            "frozen dev.csv checksum mismatch: "
            f"got {actual_sha256}, expected {FROZEN_PREDICTIONS_SHA256}"
        )


def resolve_recorded_prediction_path(
    experiment: dict, experiment_dir: Path, *, variant: str, c_value: str
) -> Path:
    """Resolve a prediction path recorded relative to its experiment directory."""
    prediction_files = experiment.get("prediction_files")
    variant_files = prediction_files.get(variant) if isinstance(prediction_files, dict) else None
    recorded_path = variant_files.get(c_value) if isinstance(variant_files, dict) else None
    if not isinstance(recorded_path, str) or not recorded_path.strip():
        raise T1ErrorAnalysisError(
            "experiment summary is missing prediction_files"
            f"[{variant!r}][{c_value!r}]"
        )
    path = Path(recorded_path)
    if not path.is_absolute():
        path = experiment_dir / path
    return path.resolve()


TABLE_NAMES = (
    "instances",
    "candidates",
    "conflict_opportunities",
    "conflict_summary",
    "same_gramset_alternatives",
    "confusions",
    "sentence_summary",
    "sentence_instance_links",
    "sentence_error_links",
    "form_summary",
    "form_instance_links",
)


def _instance_key(instance: Instance) -> tuple[str, int, int, int]:
    return (instance.language, int(instance.text_id), int(instance.sentence_id), int(instance.word_id))


def _candidate_key(row: PredictionRow) -> tuple[int, str]:
    return (int(row.wordform_id), str(row.gramset))


def _validate_and_group(
    instances: tuple[Instance, ...], predictions: tuple[PredictionRow, ...]
) -> dict[int, tuple[PredictionRow, ...]]:
    by_word: dict[int, Instance] = {}
    seen_instance_keys: set[tuple[str, int, int, int]] = set()
    for instance in instances:
        key = _instance_key(instance)
        if key in seen_instance_keys:
            raise T1ErrorAnalysisError(f"duplicate usage key: {key!r}")
        seen_instance_keys.add(key)
        if instance.word_id in by_word:
            raise T1ErrorAnalysisError(f"duplicate benchmark word_id: {instance.word_id}")
        by_word[instance.word_id] = instance

        candidate_keys = [
            (int(candidate.wordform_id), str(candidate.gramset))
            for candidate in instance.candidates
        ]
        if len(candidate_keys) != len(set(candidate_keys)):
            raise T1ErrorAnalysisError(
                f"duplicate full candidate key in benchmark word_id={instance.word_id}"
            )
        gold_key = (
            int(instance.gold_analysis.wordform_id),
            str(instance.gold_analysis.gramset),
        )
        if candidate_keys.count(gold_key) != 1:
            raise T1ErrorAnalysisError(
                f"expected one gold candidate in benchmark word_id={instance.word_id}"
            )

    rows_by_word: dict[int, list[PredictionRow]] = defaultdict(list)
    seen_prediction_keys: set[tuple[int, int, str]] = set()
    for row in predictions:
        identity = (int(row.word_id), int(row.wordform_id), str(row.gramset))
        if identity in seen_prediction_keys:
            raise T1ErrorAnalysisError(f"duplicate prediction candidate key: {identity!r}")
        seen_prediction_keys.add(identity)
        if row.word_id not in by_word:
            raise T1ErrorAnalysisError(f"prediction for unknown dev word_id={row.word_id}")
        if not math.isfinite(float(row.score)):
            raise T1ErrorAnalysisError(
                f"non-finite score for word_id={row.word_id}, candidate={_candidate_key(row)!r}"
            )
        rows_by_word[row.word_id].append(row)

    if set(rows_by_word) != set(by_word):
        missing = sorted(set(by_word) - set(rows_by_word))[:5]
        raise T1ErrorAnalysisError(f"missing predictions for dev word_id values: {missing}")

    ordered_by_word: dict[int, tuple[PredictionRow, ...]] = {}
    for word_id, instance in by_word.items():
        rows = rows_by_word[word_id]
        expected = {
            (int(candidate.wordform_id), str(candidate.gramset))
            for candidate in instance.candidates
        }
        observed = {_candidate_key(row) for row in rows}
        if observed != expected or len(rows) != len(expected):
            missing = sorted(expected - observed)[:3]
            extra = sorted(observed - expected)[:3]
            raise T1ErrorAnalysisError(
                f"candidate mismatch for word_id={word_id}; missing={missing!r}; extra={extra!r}"
            )
        ranked = tuple(sorted(rows, key=lambda row: int(row.rank)))
        expected_ranks = tuple(range(1, len(expected) + 1))
        actual_ranks = tuple(int(row.rank) for row in ranked)
        if actual_ranks != expected_ranks:
            raise T1ErrorAnalysisError(
                f"invalid ranks for word_id={word_id}: {actual_ranks!r}; expected {expected_ranks!r}"
            )

        # Match the project ranking contract: descending score, then ascending
        # wordform_id and gramset. The saved order is retained in the outputs.
        expected_order = tuple(
            sorted(
                rows,
                key=lambda row: (-float(row.score), int(row.wordform_id), str(row.gramset)),
            )
        )
        if tuple(_candidate_key(row) for row in ranked) != tuple(
            _candidate_key(row) for row in expected_order
        ):
            raise T1ErrorAnalysisError(
                f"saved rank order disagrees with score/tie-break rule for word_id={word_id}"
            )
        ordered_by_word[word_id] = ranked
    return ordered_by_word


def _sentence_views(instances: tuple[Instance, ...], context_words: tuple) -> dict:
    records_by_sentence: dict[tuple[str, int], list] = defaultdict(list)
    for record in context_words:
        records_by_sentence[(record.language, int(record.sentence_id))].append(record)
    for key, records in records_by_sentence.items():
        records.sort(key=lambda record: (int(record.word_number), int(record.word_id)))
        positions = [int(record.word_number) for record in records]
        if any(position <= 0 for position in positions):
            raise T1ErrorAnalysisError(f"nonpositive sentence position in {key!r}")
        if len(positions) != len(set(positions)):
            raise T1ErrorAnalysisError(f"duplicate sentence position in {key!r}")
        word_ids = [int(record.word_id) for record in records]
        if len(word_ids) != len(set(word_ids)):
            raise T1ErrorAnalysisError(f"duplicate word_id in sentence context {key!r}")

    result = {}
    for instance in instances:
        sentence_key = (instance.language, int(instance.sentence_id))
        records = records_by_sentence.get(sentence_key, [])
        matches = [record for record in records if int(record.word_id) == int(instance.word_id)]
        if len(matches) != 1:
            raise T1ErrorAnalysisError(
                f"expected one target context record for word_id={instance.word_id}, found {len(matches)}"
            )
        target = matches[0]
        if int(target.word_number) != int(instance.word_number) or target.word != instance.word:
            raise T1ErrorAnalysisError(
                f"context identity mismatch for word_id={instance.word_id}"
            )
        index = records.index(target)
        tokens = [str(record.word) for record in records]
        result[instance.word_id] = {
            "sentence_key": (instance.language, int(instance.text_id), int(instance.sentence_id)),
            "sentence": " ".join(tokens),
            "left_neighbor": tokens[index - 1] if index else "<BOS>",
            "right_neighbor": tokens[index + 1] if index + 1 < len(tokens) else "<EOS>",
            "target_position": int(instance.word_number),
        }
    return result


def build_t1_error_tables(
    instances: Iterable[Instance],
    predictions: Iterable[PredictionRow],
    context_words: Iterable,
) -> dict[str, list[dict]]:
    """Build all A1 tables from already selected strict-dev inputs.

    This function never selects a split, reads files, builds model features,
    trains or scores a model. It preserves the supplied prediction ranks.
    """
    instance_rows = tuple(instances)
    prediction_rows = tuple(predictions)
    context_rows = tuple(context_words)
    if not instance_rows:
        raise T1ErrorAnalysisError("the dev benchmark is empty")
    predictions_by_word = _validate_and_group(instance_rows, prediction_rows)
    sentence_views = _sentence_views(instance_rows, context_rows)

    tables = {name: [] for name in TABLE_NAMES}
    error_rows_by_sentence: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    rows_by_sentence: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    rows_by_form: dict[tuple[str, str], list[dict]] = defaultdict(list)
    conflict_opportunity_counts: Counter = Counter()
    conflict_error_counts: Counter = Counter()
    confusion_counts: Counter = Counter()

    for instance in sorted(instance_rows, key=_instance_key):
        ranked = predictions_by_word[instance.word_id]
        gold_key = (int(instance.gold_analysis.wordform_id), str(instance.gold_analysis.gramset))
        gold_row = next(row for row in ranked if _candidate_key(row) == gold_key)
        top1 = ranked[0]
        top1_key = _candidate_key(top1)
        is_correct = top1_key == gold_key
        same_wordform = int(top1.wordform_id) == gold_key[0]
        same_gramset = str(top1.gramset) == gold_key[1]
        error_category = ""
        if not is_correct:
            if same_wordform and not same_gramset:
                error_category = "same_wordform_id_different_gramset"
            elif not same_wordform and not same_gramset:
                error_category = "different_wordform_id_different_gramset"
            elif not same_wordform and same_gramset:
                error_category = "different_wordform_id_same_gramset"
            else:  # Candidate identity is the pair (wordform_id, gramset).
                raise T1ErrorAnalysisError(
                    f"incorrect candidate has identical gold identity for word_id={instance.word_id}"
                )

        view = sentence_views[instance.word_id]
        sentence_key = view["sentence_key"]
        normalized_form = str(instance.word)
        row = {
            "language": instance.language,
            "text_id": int(instance.text_id),
            "sentence_id": int(instance.sentence_id),
            "word_id": int(instance.word_id),
            "surface_form": str(instance.word),
            "normalized_form": normalized_form,
            "normalization": "identity_from_normalized_corpus_word_field",
            "target_position": view["target_position"],
            "candidate_count": len(instance.candidates),
            "distinct_gramset_count": len({candidate.gramset for candidate in instance.candidates}),
            "gold_wordform_id": gold_key[0],
            "gold_gramset": gold_key[1],
            "gold_rank": int(gold_row.rank),
            "gold_score": float(gold_row.score),
            "top1_wordform_id": int(top1.wordform_id),
            "top1_gramset": str(top1.gramset),
            "top1_rank": int(top1.rank),
            "top1_score": float(top1.score),
            "top1_minus_gold_score": float(top1.score) - float(gold_row.score),
            "same_wordform_id_as_gold": int(same_wordform),
            "same_gramset_as_gold": int(same_gramset),
            "is_correct": int(is_correct),
            "error_category": error_category,
            "sentence": view["sentence"],
            "left_neighbor": view["left_neighbor"],
            "right_neighbor": view["right_neighbor"],
        }
        tables["instances"].append(row)
        rows_by_sentence[sentence_key].append(row)
        rows_by_form[(instance.language, normalized_form)].append(row)
        for candidate_row in ranked:
            candidate_key = _candidate_key(candidate_row)
            tables["candidates"].append(
                {
                    "language": instance.language,
                    "text_id": int(instance.text_id),
                    "sentence_id": int(instance.sentence_id),
                    "word_id": int(instance.word_id),
                    "surface_form": str(instance.word),
                    "wordform_id": candidate_key[0],
                    "gramset": candidate_key[1],
                    "rank": int(candidate_row.rank),
                    "score": float(candidate_row.score),
                    "is_gold": int(candidate_key == gold_key),
                    "is_top1": int(candidate_key == top1_key),
                }
            )
        if not is_correct:
            error_rows_by_sentence[sentence_key].append(row)
            confusion_counts[(gold_key[1], str(top1.gramset))] += 1

        candidates_by_gramset: dict[str, list] = defaultdict(list)
        for candidate in instance.candidates:
            candidates_by_gramset[str(candidate.gramset)].append(candidate)
        for alternative_gramset in sorted(candidates_by_gramset):
            if alternative_gramset == gold_key[1]:
                for alternative in candidates_by_gramset[alternative_gramset]:
                    alternative_key = (int(alternative.wordform_id), str(alternative.gramset))
                    if alternative_key == gold_key:
                        continue
                    selected = int(not is_correct and top1_key == alternative_key)
                    tables["same_gramset_alternatives"].append(
                        {
                            "language": instance.language,
                            "text_id": int(instance.text_id),
                            "sentence_id": int(instance.sentence_id),
                            "word_id": int(instance.word_id),
                            "gold_wordform_id": gold_key[0],
                            "gold_gramset": gold_key[1],
                            "alternative_wordform_id": alternative_key[0],
                            "alternative_gramset": alternative_key[1],
                            "is_top1": int(top1_key == alternative_key),
                            "selected_error": selected,
                        }
                    )
                continue

            alternatives = candidates_by_gramset[alternative_gramset]
            selected_error = int(not is_correct and str(top1.gramset) == alternative_gramset)
            opportunity_key = (gold_key[1], alternative_gramset)
            conflict_opportunity_counts[opportunity_key] += 1
            conflict_error_counts[opportunity_key] += selected_error
            tables["conflict_opportunities"].append(
                {
                    "language": instance.language,
                    "text_id": int(instance.text_id),
                    "sentence_id": int(instance.sentence_id),
                    "word_id": int(instance.word_id),
                    "gold_gramset": gold_key[1],
                    "alternative_gramset": alternative_gramset,
                    "alternative_candidate_count": len(alternatives),
                    "selected_error": selected_error,
                }
            )

    for (gold_gramset, alternative_gramset), opportunities in sorted(conflict_opportunity_counts.items()):
        errors = conflict_error_counts[(gold_gramset, alternative_gramset)]
        tables["conflict_summary"].append(
            {
                "gold_gramset": gold_gramset,
                "alternative_gramset": alternative_gramset,
                "opportunity_count": opportunities,
                "selected_error_count": errors,
                "selected_error_rate": errors / opportunities if opportunities else None,
            }
        )

    for (gold_gramset, top1_gramset), count in sorted(confusion_counts.items()):
        tables["confusions"].append(
            {
                "gold_gramset": gold_gramset,
                "top1_gramset": top1_gramset,
                "confusion_count": count,
            }
        )

    sentence_records: dict[tuple[str, int, int], dict] = {}
    for key, grouped_rows in rows_by_sentence.items():
        errors = error_rows_by_sentence.get(key, [])
        confusion_types = {
            (str(row["gold_gramset"]), str(row["top1_gramset"])) for row in errors
        }
        sentence_records[key] = {
            "language": key[0],
            "text_id": key[1],
            "sentence_id": key[2],
            "benchmark_instance_count": len(grouped_rows),
            "error_count": len(errors),
            "error_rate": len(errors) / len(grouped_rows),
            "distinct_directed_confusion_count": len(confusion_types),
            "sentence": grouped_rows[0]["sentence"],
        }
    tables["sentence_summary"] = [sentence_records[key] for key in sorted(sentence_records)]

    for key, grouped_rows in rows_by_sentence.items():
        for row in sorted(grouped_rows, key=lambda item: int(item["target_position"])):
            link = {
                "language": key[0],
                "text_id": key[1],
                "sentence_id": key[2],
                "word_id": row["word_id"],
                "word": row["surface_form"],
                "target_position": row["target_position"],
                "gold_gramset": row["gold_gramset"],
                "top1_gramset": row["top1_gramset"],
                "gold_rank": row["gold_rank"],
                "is_correct": row["is_correct"],
                "error_category": row["error_category"],
            }
            tables["sentence_instance_links"].append(link)
            if not row["is_correct"]:
                tables["sentence_error_links"].append(link)

    for (language, form), grouped_rows in sorted(rows_by_form.items()):
        errors = [row for row in grouped_rows if not row["is_correct"]]
        confusions = {
            (str(row["gold_gramset"]), str(row["top1_gramset"])) for row in errors
        }
        tables["form_summary"].append(
            {
                "language": language,
                "normalized_form": form,
                "normalization": "identity_from_normalized_corpus_word_field",
                "instance_count": len(grouped_rows),
                "error_count": len(errors),
                "text_count": len({int(row["text_id"]) for row in grouped_rows}),
                "sentence_count": len(
                    {(int(row["text_id"]), int(row["sentence_id"])) for row in grouped_rows}
                ),
                "distinct_gold_gramset_count": len({str(row["gold_gramset"]) for row in grouped_rows}),
                "distinct_directed_error_count": len(confusions),
            }
        )
        for row in sorted(grouped_rows, key=lambda item: (int(item["text_id"]), int(item["sentence_id"]), int(item["word_id"]))):
            tables["form_instance_links"].append(
                {
                    "language": language,
                    "normalized_form": form,
                    "text_id": row["text_id"],
                    "sentence_id": row["sentence_id"],
                    "word_id": row["word_id"],
                    "gold_gramset": row["gold_gramset"],
                    "top1_gramset": row["top1_gramset"],
                    "gold_rank": row["gold_rank"],
                    "is_correct": row["is_correct"],
                    "error_category": row["error_category"],
                }
            )

    return tables


def validate_t1_error_table_invariants(tables: dict[str, list[dict]]) -> dict[str, int]:
    """Check detailed-to-summary consistency and return audit counts."""
    for name in TABLE_NAMES:
        if name not in tables:
            raise T1ErrorAnalysisError(f"missing diagnostic table: {name}")
    instances = tables["instances"]
    if not instances:
        raise T1ErrorAnalysisError("instances table is empty")
    errors = [row for row in instances if not row["is_correct"]]
    correct_count = len(instances) - len(errors)
    candidate_count = len(tables["candidates"])
    if len(instances) != correct_count + len(errors):
        raise T1ErrorAnalysisError("instances != correct + errors")

    instance_by_key = _unique_rows(
        instances,
        lambda row: (
            row["language"], int(row["text_id"]), int(row["sentence_id"]), int(row["word_id"])
        ),
        "instances",
    )

    # Rebuild directed confusion counts from the per-usage records.
    expected_confusions = Counter(
        (str(row["gold_gramset"]), str(row["top1_gramset"])) for row in errors
    )
    actual_confusions = _unique_rows(
        tables["confusions"],
        lambda row: (str(row["gold_gramset"]), str(row["top1_gramset"])),
        "confusions",
    )
    if set(actual_confusions) != set(expected_confusions):
        raise T1ErrorAnalysisError("confusions keys disagree with erroneous instances")
    for key, expected_count in expected_confusions.items():
        if int(actual_confusions[key]["confusion_count"]) != expected_count:
            raise T1ErrorAnalysisError(f"confusion count disagrees with instances for {key!r}")

    different_gramset_errors = sum(
        not row["same_gramset_as_gold"] for row in errors
    )
    same_gramset_errors = len(errors) - different_gramset_errors
    selected_conflict_errors = sum(
        int(row["selected_error_count"]) for row in tables["conflict_summary"]
    )
    if selected_conflict_errors != different_gramset_errors:
        raise T1ErrorAnalysisError(
            "different-gramset errors do not equal selected errors in conflict summary"
        )
    if selected_conflict_errors + same_gramset_errors != len(errors):
        raise T1ErrorAnalysisError("error identity categories do not sum to all errors")

    opportunity_keys = [
        (
            row["language"], row["text_id"], row["sentence_id"], row["word_id"],
            row["gold_gramset"], row["alternative_gramset"],
        )
        for row in tables["conflict_opportunities"]
    ]
    if len(opportunity_keys) != len(set(opportunity_keys)):
        raise T1ErrorAnalysisError("duplicate usage/G/H conflict opportunity")
    expected_conflicts: Counter = Counter()
    expected_selected_errors: Counter = Counter()
    for row in tables["conflict_opportunities"]:
        key = (str(row["gold_gramset"]), str(row["alternative_gramset"]))
        expected_conflicts[key] += 1
        selected = int(row["selected_error"])
        if selected not in (0, 1):
            raise T1ErrorAnalysisError(f"invalid selected_error flag for conflict {key!r}")
        expected_selected_errors[key] += selected
        if int(row["alternative_candidate_count"]) < 1:
            raise T1ErrorAnalysisError(f"non-positive alternative candidate count for {key!r}")

    actual_conflict_summary = _unique_rows(
        tables["conflict_summary"],
        lambda row: (str(row["gold_gramset"]), str(row["alternative_gramset"])),
        "conflict summary",
    )
    if set(actual_conflict_summary) != set(expected_conflicts):
        raise T1ErrorAnalysisError("conflict summary keys disagree with detailed opportunities")
    for key, expected_opportunities in expected_conflicts.items():
        row = actual_conflict_summary[key]
        opportunity_count = int(row["opportunity_count"])
        selected_error_count = int(row["selected_error_count"])
        expected_selected = expected_selected_errors[key]
        if opportunity_count != expected_opportunities:
            raise T1ErrorAnalysisError(f"opportunity count disagrees with details for {key!r}")
        if selected_error_count != expected_selected:
            raise T1ErrorAnalysisError(f"selected-error count disagrees with details for {key!r}")
        if selected_error_count > opportunity_count:
            raise T1ErrorAnalysisError("selected conflict errors exceed opportunities")
        rate = float(row["selected_error_rate"])
        expected_rate = expected_selected / expected_opportunities
        if not math.isfinite(rate) or not math.isclose(rate, expected_rate, rel_tol=0.0, abs_tol=1e-12):
            raise T1ErrorAnalysisError(f"selected-error rate disagrees with counts for {key!r}")

    # Every usage must occur exactly once in the sentence and form link tables;
    # error links must be exactly the erroneous subset.
    link_fields = (
        "language", "text_id", "sentence_id", "word_id", "target_position",
        "gold_gramset", "top1_gramset", "gold_rank", "is_correct", "error_category",
    )
    expected_sentence_links = {
        key: tuple(row[field] for field in link_fields)
        for key, row in instance_by_key.items()
    }
    actual_sentence_links = _unique_rows(
        tables["sentence_instance_links"],
        lambda row: (
            row["language"], int(row["text_id"]), int(row["sentence_id"]), int(row["word_id"])
        ),
        "sentence instance links",
    )
    _assert_link_projection(
        actual_sentence_links, expected_sentence_links, link_fields, "sentence instance links"
    )
    expected_error_links = {
        key: value for key, value in expected_sentence_links.items()
        if not int(instance_by_key[key]["is_correct"])
    }
    actual_error_links = _unique_rows(
        tables["sentence_error_links"],
        lambda row: (
            row["language"], int(row["text_id"]), int(row["sentence_id"]), int(row["word_id"])
        ),
        "sentence error links",
    )
    _assert_link_projection(
        actual_error_links, expected_error_links, link_fields, "sentence error links"
    )

    form_fields = (
        "language", "normalized_form", "text_id", "sentence_id", "word_id",
        "gold_gramset", "top1_gramset", "gold_rank", "is_correct", "error_category",
    )
    expected_form_links = {
        key: tuple(row[field] for field in form_fields)
        for key, row in instance_by_key.items()
    }
    actual_form_links = _unique_rows(
        tables["form_instance_links"],
        lambda row: (
            row["language"], int(row["text_id"]), int(row["sentence_id"]), int(row["word_id"])
        ),
        "form instance links",
    )
    _assert_link_projection(actual_form_links, expected_form_links, form_fields, "form instance links")

    # Rebuild every sentence aggregate from instances, including correct usages.
    sentence_groups: dict[tuple, list[dict]] = defaultdict(list)
    form_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in instances:
        sentence_groups[(row["language"], int(row["text_id"]), int(row["sentence_id"]))].append(row)
        form_groups[(row["language"], str(row["normalized_form"]))].append(row)

    expected_sentence_summary = {}
    for key, rows in sentence_groups.items():
        group_errors = [row for row in rows if not int(row["is_correct"])]
        expected_sentence_summary[key] = {
            "benchmark_instance_count": len(rows),
            "error_count": len(group_errors),
            "error_rate": len(group_errors) / len(rows),
            "distinct_directed_confusion_count": len({
                (str(row["gold_gramset"]), str(row["top1_gramset"])) for row in group_errors
            }),
            "sentence": rows[0]["sentence"],
        }
    actual_sentence_summary = _unique_rows(
        tables["sentence_summary"],
        lambda row: (row["language"], int(row["text_id"]), int(row["sentence_id"])),
        "sentence summary",
    )
    _assert_group_summary(
        actual_sentence_summary,
        expected_sentence_summary,
        ("benchmark_instance_count", "error_count", "distinct_directed_confusion_count", "sentence"),
        ("error_rate",),
        "sentence summary",
    )

    expected_form_summary = {}
    for key, rows in form_groups.items():
        group_errors = [row for row in rows if not int(row["is_correct"])]
        expected_form_summary[key] = {
            "instance_count": len(rows),
            "error_count": len(group_errors),
            "text_count": len({int(row["text_id"]) for row in rows}),
            "sentence_count": len({(int(row["text_id"]), int(row["sentence_id"])) for row in rows}),
            "distinct_gold_gramset_count": len({str(row["gold_gramset"]) for row in rows}),
            "distinct_directed_error_count": len({
                (str(row["gold_gramset"]), str(row["top1_gramset"])) for row in group_errors
            }),
            "normalization": "identity_from_normalized_corpus_word_field",
        }
    actual_form_summary = _unique_rows(
        tables["form_summary"],
        lambda row: (row["language"], str(row["normalized_form"])),
        "form summary",
    )
    _assert_group_summary(
        actual_form_summary,
        expected_form_summary,
        (
            "instance_count", "error_count", "text_count", "sentence_count",
            "distinct_gold_gramset_count", "distinct_directed_error_count", "normalization",
        ),
        (),
        "form summary",
    )

    candidate_counts_by_word: Counter = Counter(
        int(row["word_id"]) for row in tables["candidates"]
    )
    gold_counts_by_word: Counter = Counter()
    top1_counts_by_word: Counter = Counter()
    for row in instances:
        if candidate_counts_by_word[int(row["word_id"])] != int(row["candidate_count"]):
            raise T1ErrorAnalysisError(
                f"candidate rows disagree with instances for word_id={row['word_id']}"
            )
    for row in tables["candidates"]:
        gold_counts_by_word[int(row["word_id"])] += int(row["is_gold"])
        top1_counts_by_word[int(row["word_id"])] += int(row["is_top1"])
    if any(gold_counts_by_word[int(row["word_id"])] != 1 for row in instances):
        raise T1ErrorAnalysisError("candidate table does not contain exactly one gold per instance")
    if any(top1_counts_by_word[int(row["word_id"])] != 1 for row in instances):
        raise T1ErrorAnalysisError("candidate table does not contain exactly one Top-1 per instance")
    selected_same_gramset_errors = sum(
        int(row["selected_error"]) for row in tables["same_gramset_alternatives"]
    )
    if selected_same_gramset_errors != same_gramset_errors:
        raise T1ErrorAnalysisError(
            "same-gramset alternatives do not account for same-gramset errors"
        )
    return {
        "instances": len(instances),
        "correct_top1": correct_count,
        "errors": len(errors),
        "candidates": candidate_count,
        "different_gramset_errors": different_gramset_errors,
        "same_gramset_errors": same_gramset_errors,
        "conflict_opportunities": len(tables["conflict_opportunities"]),
        "distinct_conflicts": len(tables["conflict_summary"]),
        "sentences": len(tables["sentence_summary"]),
        "forms": len(tables["form_summary"]),
    }


def _unique_rows(rows: list[dict], key_fn, label: str) -> dict[tuple, dict]:
    indexed = {}
    for row in rows:
        key = key_fn(row)
        if key in indexed:
            raise T1ErrorAnalysisError(f"duplicate key in {label}: {key!r}")
        indexed[key] = row
    return indexed


def _assert_link_projection(
    actual: dict[tuple, dict], expected: dict[tuple, tuple], fields: tuple[str, ...], label: str
) -> None:
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))[:3]
        extra = sorted(set(actual) - set(expected))[:3]
        raise T1ErrorAnalysisError(f"{label} keys do not match instances; missing={missing!r}, extra={extra!r}")
    for key, expected_values in expected.items():
        actual_values = tuple(actual[key][field] for field in fields)
        if actual_values != expected_values:
            raise T1ErrorAnalysisError(f"{label} values disagree with instance {key!r}")


def _assert_group_summary(
    actual: dict[tuple, dict],
    expected: dict[tuple, dict],
    integer_fields: tuple[str, ...],
    rate_fields: tuple[str, ...],
    label: str,
) -> None:
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))[:3]
        extra = sorted(set(actual) - set(expected))[:3]
        raise T1ErrorAnalysisError(f"{label} keys do not match instances; missing={missing!r}, extra={extra!r}")
    for key, expected_values in expected.items():
        row = actual[key]
        for field in integer_fields:
            if row[field] != expected_values[field]:
                raise T1ErrorAnalysisError(f"{label} {field} disagrees with instances for {key!r}")
        for field in rate_fields:
            value = float(row[field])
            if not math.isfinite(value) or not math.isclose(
                value, float(expected_values[field]), rel_tol=0.0, abs_tol=1e-12
            ):
                raise T1ErrorAnalysisError(f"{label} {field} disagrees with instances for {key!r}")
