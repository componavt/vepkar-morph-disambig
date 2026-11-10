#!/usr/bin/env python3
"""Run the approved T1.2 frequency-feature comparison on dev."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_context import _preflight_benchmark  # noqa: E402
from commands.evaluate_predictions import load_ranked_predictions  # noqa: E402
from core.data import DataError  # noqa: E402
from core.instances import CorpusTagError, Instance  # noqa: E402
from core.predictions import (  # noqa: E402
    BenchmarkIntegrityError,
    PredictionRow,
    validate_predictions,
)
from core.splits import SplitError  # noqa: E402
from t1_features.data import load_t1_inputs  # noqa: E402
from t1_features.features import FeatureConstructionError, build_candidate_examples  # noqa: E402
from t1_features.frequency_features import (  # noqa: E402
    FREQUENCY_CHANNELS,
    FrequencyFeatureError,
    TrainFrequencyCounts,
    candidate_frequency_features,
    count_train_frequencies,
)
from t1_features.grid_diagnostics import T1_C_GRID, summarize_trajectory  # noqa: E402
from t1_features.model import score_candidate_examples, train_candidate_model  # noqa: E402
from t1_features.prediction_io import (  # noqa: E402
    T1PredictionWriteError,
    write_split_predictions,
)
from t1_features.ranking import rank_candidate_scores  # noqa: E402

BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 20261109
METRICS = ("top1", "mrr", "top3")
VARIANT_CHANNELS = {
    "baseline": (),
    "analysis_pooled": ("analysis_pooled",),
    "analysis_language": ("analysis_language",),
    "gramset_pooled": ("gramset_pooled",),
    "gramset_language": ("gramset_language",),
    "all_frequency": FREQUENCY_CHANNELS,
}
NEW_VARIANTS = tuple(name for name in VARIANT_CHANNELS if name != "baseline")
PAIRED_COMPARISONS = (
    ("analysis_pooled_vs_baseline", "analysis_pooled", "baseline"),
    ("analysis_language_vs_baseline", "analysis_language", "baseline"),
    ("gramset_pooled_vs_baseline", "gramset_pooled", "baseline"),
    ("gramset_language_vs_baseline", "gramset_language", "baseline"),
    ("all_frequency_vs_baseline", "all_frequency", "baseline"),
    ("analysis_pooled_vs_language", "analysis_pooled", "analysis_language"),
    ("gramset_pooled_vs_language", "gramset_pooled", "gramset_language"),
)
REFERENCE_VARIANT = "both"


class T1FrequencyAblationError(ValueError):
    """The T1 frequency-feature experiment could not be completed consistently."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _rank_details(
    rows: tuple[PredictionRow, ...],
    instances: tuple[Instance, ...],
) -> tuple[np.ndarray, tuple[PredictionRow, ...]]:
    rows_by_word: dict[int, list[PredictionRow]] = defaultdict(list)
    for row in rows:
        rows_by_word[row.word_id].append(row)
    ranks: list[int] = []
    top_rows: list[PredictionRow] = []
    for instance in instances:
        ordered = sorted(rows_by_word.get(instance.word_id, ()), key=lambda row: row.rank)
        gold = instance.gold_analysis
        gold_ranks = [
            row.rank
            for row in ordered
            if (row.wordform_id, row.gramset) == (gold.wordform_id, gold.gramset)
        ]
        if len(gold_ranks) != 1 or not ordered or ordered[0].rank != 1:
            raise T1FrequencyAblationError(
                f"{instance.language} word_id={instance.word_id}: "
                "prediction rows do not contain exactly one ranked gold"
            )
        ranks.append(gold_ranks[0])
        top_rows.append(ordered[0])
    return np.asarray(ranks, dtype=np.int64), tuple(top_rows)


def _point_metrics(ranks: np.ndarray) -> dict[str, float]:
    return {
        "top1": float(np.mean(ranks == 1)),
        "mrr": float(np.mean(1.0 / ranks)),
        "top3": float(np.mean(ranks <= 3)),
    }


def _bootstrap_metrics(
    ranks_by_model: dict[tuple[str, float], np.ndarray],
    instances: tuple[Instance, ...],
) -> dict[tuple[str, float], dict[str, np.ndarray]]:
    text_keys = sorted({(row.language, int(row.text_id)) for row in instances})
    if not text_keys:
        raise T1FrequencyAblationError("dev has no text groups")
    text_index = {key: index for index, key in enumerate(text_keys)}
    word_text_indices = np.asarray(
        [text_index[(row.language, int(row.text_id))] for row in instances],
        dtype=np.int64,
    )
    bootstrap = {
        key: {metric: np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64) for metric in METRICS}
        for key in ranks_by_model
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for sample_index in range(BOOTSTRAP_SAMPLES):
        sampled = rng.integers(0, len(text_keys), size=len(text_keys))
        multiplicity = np.bincount(sampled, minlength=len(text_keys))
        weights = multiplicity[word_text_indices]
        denominator = int(weights.sum())
        if denominator == 0:
            raise T1FrequencyAblationError("bootstrap sample contains no dev words")
        for key, ranks in ranks_by_model.items():
            bootstrap[key]["top1"][sample_index] = np.dot(weights, ranks == 1) / denominator
            bootstrap[key]["mrr"][sample_index] = np.dot(weights, 1.0 / ranks) / denominator
            bootstrap[key]["top3"][sample_index] = np.dot(weights, ranks <= 3) / denominator
    return bootstrap


def _bootstrap_by_language(
    ranks_by_model: dict[tuple[str, float], np.ndarray],
    instances: tuple[Instance, ...],
) -> dict[str, dict[tuple[str, float], dict[str, np.ndarray]]]:
    languages = sorted({row.language for row in instances})
    results = {}
    for language_index, language in enumerate(languages):
        indices = np.asarray(
            [index for index, row in enumerate(instances) if row.language == language],
            dtype=np.int64,
        )
        lang_instances = tuple(instances[index] for index in indices)
        text_keys = sorted({int(row.text_id) for row in lang_instances})
        text_index = {key: index for index, key in enumerate(text_keys)}
        word_text_indices = np.asarray(
            [text_index[int(row.text_id)] for row in lang_instances], dtype=np.int64
        )
        result = {
            key: {metric: np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64) for metric in METRICS}
            for key in ranks_by_model
        }
        rng = np.random.default_rng(BOOTSTRAP_SEED + 100 + language_index)
        for sample_index in range(BOOTSTRAP_SAMPLES):
            sampled = rng.integers(0, len(text_keys), size=len(text_keys))
            weights = np.bincount(sampled, minlength=len(text_keys))[word_text_indices]
            denominator = int(weights.sum())
            if denominator == 0:
                raise T1FrequencyAblationError(
                    f"bootstrap sample contains no words for {language}"
                )
            for key, ranks in ranks_by_model.items():
                selected = ranks[indices]
                result[key]["top1"][sample_index] = np.dot(weights, selected == 1) / denominator
                result[key]["mrr"][sample_index] = np.dot(weights, 1.0 / selected) / denominator
                result[key]["top3"][sample_index] = np.dot(weights, selected <= 3) / denominator
        results[language] = result
    return results


def _bootstrap_subset_effect(
    pooled_ranks: np.ndarray,
    local_ranks: np.ndarray,
    instances: tuple[Instance, ...],
    subset_indices: np.ndarray,
    *,
    seed: int,
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    subset = [instances[int(index)] for index in subset_indices]
    text_keys = sorted({(row.language, int(row.text_id)) for row in subset})
    text_index = {key: index for index, key in enumerate(text_keys)}
    word_text_indices = np.asarray(
        [text_index[(row.language, int(row.text_id))] for row in subset], dtype=np.int64
    )
    a = pooled_ranks[subset_indices]
    b = local_ranks[subset_indices]
    point_a = _point_metrics(a)
    point_b = _point_metrics(b)
    point = {metric: point_a[metric] - point_b[metric] for metric in METRICS}
    samples = {metric: np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64) for metric in METRICS}
    rng = np.random.default_rng(seed)
    for sample_index in range(BOOTSTRAP_SAMPLES):
        sampled = rng.integers(0, len(text_keys), size=len(text_keys))
        weights = np.bincount(sampled, minlength=len(text_keys))[word_text_indices]
        denominator = int(weights.sum())
        if denominator == 0:
            raise T1FrequencyAblationError("subset bootstrap sample contains no words")
        samples["top1"][sample_index] = (
            np.dot(weights, a == 1) - np.dot(weights, b == 1)
        ) / denominator
        samples["mrr"][sample_index] = (
            np.dot(weights, 1.0 / a) - np.dot(weights, 1.0 / b)
        ) / denominator
        samples["top3"][sample_index] = (
            np.dot(weights, a <= 3) - np.dot(weights, b <= 3)
        ) / denominator
    intervals = {
        metric: (
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        )
        for metric, values in samples.items()
    }
    return point, intervals


def _select_C(variant: str, point: dict[tuple[str, float], dict[str, float]]) -> float:
    return min(
        T1_C_GRID,
        key=lambda C: (
            -point[(variant, C)]["top1"],
            -point[(variant, C)]["mrr"],
            C,
        ),
    )


def _load_reference(
    reference_dir: Path,
    C: float,
    dev_instances: tuple[Instance, ...],
    benchmark_instances: tuple[Instance, ...],
    split_rows: tuple[tuple[str, int, str], ...],
) -> tuple[np.ndarray, tuple[PredictionRow, ...], Path]:
    run_dir = reference_dir / "runs" / REFERENCE_VARIANT / f"c-{C:g}"
    run_path = run_dir / "run.json"
    prediction_path = run_dir / "dev.csv"
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise T1FrequencyAblationError(f"could not read reference run {run_path}: {exc}") from exc
    expected = {
        "C": C,
        "target_split": "dev",
        "suffix_max_length": 5,
        "include_affix": False,
        "include_left_context": True,
        "include_right_context": True,
    }
    parameters = run.get("parameters", {})
    for name, value in expected.items():
        if parameters.get(name) != value:
            raise T1FrequencyAblationError(
                f"reference {run_path} has unexpected {name}: {parameters.get(name)!r}"
            )
    validation = validate_predictions(prediction_path, "dev", benchmark_instances, split_rows)
    if not validation.is_valid:
        raise T1FrequencyAblationError(
            f"invalid reference predictions in {prediction_path}: {validation.errors}"
        )
    ranks, top_rows = _rank_details(load_ranked_predictions(prediction_path), dev_instances)
    return ranks, top_rows, prediction_path


def _augment_examples(examples, counts: TrainFrequencyCounts, channels):
    augmented = []
    for example in examples:
        extra = candidate_frequency_features(
            example.language,
            example.wordform_id,
            example.gramset,
            counts,
            channels,
        )
        augmented.append(
            replace(example, features={**example.features, **extra})
        )
    return tuple(augmented)


def _make_transfer_rows(
    ranks_by_model: dict[tuple[str, float], np.ndarray],
    instances: tuple[Instance, ...],
    counts: TrainFrequencyCounts,
    selected_C: dict[str, float],
) -> list[dict]:
    rows = []
    languages = sorted({row.language for row in instances})
    for language_index, language in enumerate(languages):
        for family, pooled_variant, local_variant in (
            ("analysis", "analysis_pooled", "analysis_language"),
            ("gramset", "gramset_pooled", "gramset_language"),
        ):
            indices = []
            for index, instance in enumerate(instances):
                if instance.language != language:
                    continue
                gold = instance.gold_analysis
                if family == "analysis":
                    local = counts.analysis_language.get(
                        (language, gold.wordform_id, gold.gramset), 0
                    )
                    pooled = counts.analysis_pooled.get(
                        (gold.wordform_id, gold.gramset), 0
                    )
                else:
                    local = counts.gramset_language.get((language, gold.gramset), 0)
                    pooled = counts.gramset_pooled.get(gold.gramset, 0)
                if local == 0 and pooled > 0:
                    indices.append(index)
            subset_indices = np.asarray(indices, dtype=np.int64)
            for C in T1_C_GRID:
                if indices:
                    point_delta, intervals = _bootstrap_subset_effect(
                        ranks_by_model[(pooled_variant, C)],
                        ranks_by_model[(local_variant, C)],
                        instances,
                        subset_indices,
                        seed=BOOTSTRAP_SEED + 1000 + language_index * 100 + (family == "gramset") * 10 + T1_C_GRID.index(C),
                    )
                    pooled_point = _point_metrics(
                        ranks_by_model[(pooled_variant, C)][subset_indices]
                    )
                    local_point = _point_metrics(
                        ranks_by_model[(local_variant, C)][subset_indices]
                    )
                    values = {
                        "pooled_top1": pooled_point["top1"],
                        "language_top1": local_point["top1"],
                        "delta_top1": point_delta["top1"],
                        "delta_top1_ci95_low": intervals["top1"][0],
                        "delta_top1_ci95_high": intervals["top1"][1],
                        "pooled_mrr": pooled_point["mrr"],
                        "language_mrr": local_point["mrr"],
                        "delta_mrr": point_delta["mrr"],
                        "delta_mrr_ci95_low": intervals["mrr"][0],
                        "delta_mrr_ci95_high": intervals["mrr"][1],
                    }
                else:
                    values = {
                        field: ""
                        for field in (
                            "pooled_top1", "language_top1", "delta_top1",
                            "delta_top1_ci95_low", "delta_top1_ci95_high",
                            "pooled_mrr", "language_mrr", "delta_mrr",
                            "delta_mrr_ci95_low", "delta_mrr_ci95_high",
                        )
                    }
                rows.append(
                    {
                        "frequency_family": family,
                        "language": language,
                        "cross_language_only_gold_count": len(indices),
                        "C": C,
                        "pooled_variant": pooled_variant,
                        "language_variant": local_variant,
                        "pooled_selected_C": C == selected_C[pooled_variant],
                        "language_selected_C": C == selected_C[local_variant],
                        **values,
                    }
                )
    return rows


def run_experiment(args: argparse.Namespace) -> dict:
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise T1FrequencyAblationError(f"output already exists: {args.output_dir}")
    data_dir, tag = _preflight_benchmark(args.data_dir)
    if tag != "v2026.11":
        raise T1FrequencyAblationError(f"expected data v2026.11, found {tag!r}")
    split_path = args.split_file or ROOT / "data" / "derived" / f"splits_{tag}.csv"
    split_path = split_path.expanduser().resolve()
    reference_dir = args.reference_dir.expanduser().resolve()
    train_inputs = load_t1_inputs(
        data_dir, tag, split="train", split_file_override=split_path
    )
    dev_inputs = load_t1_inputs(
        data_dir, tag, split="dev", split_file_override=split_path
    )
    train_instances = train_inputs.instances
    dev_instances = dev_inputs.instances
    if len(train_instances) != 20845 or len(dev_instances) != 2613:
        raise T1FrequencyAblationError(
            "unexpected split size: "
            f"train={len(train_instances)}, dev={len(dev_instances)}"
        )
    counts = count_train_frequencies(train_instances)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True)
    run_root = output_dir / "runs"
    run_root.mkdir()
    ranks_by_model: dict[tuple[str, float], np.ndarray] = {}
    top_rows_by_model: dict[tuple[str, float], tuple[PredictionRow, ...]] = {}
    predictions_by_model: dict[tuple[str, float], Path] = {}

    for C in T1_C_GRID:
        ranks, top_rows, prediction_path = _load_reference(
            reference_dir,
            C,
            dev_instances,
            dev_inputs.benchmark.instances,
            dev_inputs.benchmark.split_rows,
        )
        key = ("baseline", C)
        ranks_by_model[key] = ranks
        top_rows_by_model[key] = top_rows
        predictions_by_model[key] = prediction_path

    base_train_examples = build_candidate_examples(
        train_instances,
        train_inputs.context_words,
        mode="train",
        suffix_max_length=5,
        context_window=1,
        include_affix=False,
    )
    base_dev_examples = build_candidate_examples(
        dev_instances,
        dev_inputs.context_words,
        mode="predict",
        suffix_max_length=5,
        context_window=1,
        include_affix=False,
    )

    for variant in NEW_VARIANTS:
        channels = VARIANT_CHANNELS[variant]
        print(f"building frequency features for {variant}", flush=True)
        training_examples = _augment_examples(base_train_examples, counts, channels)
        dev_examples = _augment_examples(base_dev_examples, counts, channels)
        variant_dir = run_root / variant
        variant_dir.mkdir()
        for C in T1_C_GRID:
            print(f"training {variant}, C={C:g}", flush=True)
            model = train_candidate_model(training_examples, C=C)
            scores = score_candidate_examples(model, dev_examples)
            prediction_rows = rank_candidate_scores(dev_examples, scores)
            run_dir = variant_dir / f"c-{C:g}"
            run_dir.mkdir()
            dev_path = run_dir / "dev.csv"
            validation = write_split_predictions(
                prediction_rows,
                dev_path,
                benchmark=dev_inputs.benchmark,
                split="dev",
            )
            if not validation.is_valid:
                raise T1FrequencyAblationError(
                    f"invalid dev predictions for {variant}, C={C:g}"
                )
            (run_dir / "run.json").write_text(
                json.dumps(
                    {
                        "variant": variant,
                        "parameters": {
                            "C": C,
                            "target_split": "dev",
                            "suffix_max_length": 5,
                            "context_window": 1,
                            "include_affix": False,
                            "frequency_channels": list(channels),
                            "frequency_transform": "log1p(count)",
                            "zero_count_indicator": True,
                            "frequency_count_source": "train gold analyses only",
                        },
                        "predictions": "dev.csv",
                        "predicted_word_count": validation.predicted_word_count,
                        "predicted_candidate_count": validation.predicted_candidate_count,
                    },
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            key = (variant, C)
            ranks, top_rows = _rank_details(prediction_rows, dev_instances)
            ranks_by_model[key] = ranks
            top_rows_by_model[key] = top_rows
            predictions_by_model[key] = dev_path

    point = {key: _point_metrics(ranks) for key, ranks in ranks_by_model.items()}
    bootstrap = _bootstrap_metrics(ranks_by_model, dev_instances)
    selected_C = {
        variant: _select_C(variant, point)
        for variant in VARIANT_CHANNELS
    }

    grid_rows = []
    for variant in VARIANT_CHANNELS:
        for C in T1_C_GRID:
            key = (variant, C)
            row = {"variant": variant, "C": C, "selected_C": C == selected_C[variant]}
            for metric in METRICS:
                row[metric] = point[key][metric]
                row[f"{metric}_ci95_low"] = float(np.quantile(bootstrap[key][metric], 0.025))
                row[f"{metric}_ci95_high"] = float(np.quantile(bootstrap[key][metric], 0.975))
            grid_rows.append(row)
    _write_csv(
        output_dir / "grid.csv",
        (
            "variant", "C", "top1", "top1_ci95_low", "top1_ci95_high",
            "mrr", "mrr_ci95_low", "mrr_ci95_high", "top3",
            "top3_ci95_low", "top3_ci95_high", "selected_C",
        ),
        grid_rows,
    )

    effects_rows = []
    effects_summary = {}
    for name, with_variant, without_variant in PAIRED_COMPARISONS:
        effects_summary[name] = {}
        for C in T1_C_GRID:
            row = {
                "effect": name,
                "with_variant": with_variant,
                "without_variant": without_variant,
                "C": C,
            }
            metric_summary = {}
            for metric in METRICS:
                delta = point[(with_variant, C)][metric] - point[(without_variant, C)][metric]
                samples = bootstrap[(with_variant, C)][metric] - bootstrap[(without_variant, C)][metric]
                low = float(np.quantile(samples, 0.025))
                high = float(np.quantile(samples, 0.975))
                row[f"delta_{metric}"] = delta
                row[f"delta_{metric}_ci95_low"] = low
                row[f"delta_{metric}_ci95_high"] = high
                metric_summary[metric] = {"point": delta, "ci95": [low, high]}
            effects_summary[name][str(C)] = metric_summary
            effects_rows.append(row)
    _write_csv(
        output_dir / "effects.csv",
        (
            "effect", "with_variant", "without_variant", "C",
            "delta_top1", "delta_top1_ci95_low", "delta_top1_ci95_high",
            "delta_mrr", "delta_mrr_ci95_low", "delta_mrr_ci95_high",
            "delta_top3", "delta_top3_ci95_low", "delta_top3_ci95_high",
        ),
        effects_rows,
    )

    language_bootstrap = _bootstrap_by_language(ranks_by_model, dev_instances)
    language_rows = []
    for language, per_model in language_bootstrap.items():
        indices = np.asarray(
            [i for i, instance in enumerate(dev_instances) if instance.language == language],
            dtype=np.int64,
        )
        for variant in VARIANT_CHANNELS:
            for C in T1_C_GRID:
                key = (variant, C)
                local_point = _point_metrics(ranks_by_model[key][indices])
                row = {
                    "variant": variant,
                    "C": C,
                    "language": language,
                    "word_count": len(indices),
                    "selected_C": C == selected_C[variant],
                }
                for metric in METRICS:
                    row[metric] = local_point[metric]
                    row[f"{metric}_ci95_low"] = float(np.quantile(per_model[key][metric], 0.025))
                    row[f"{metric}_ci95_high"] = float(np.quantile(per_model[key][metric], 0.975))
                language_rows.append(row)
    _write_csv(
        output_dir / "language-metrics.csv",
        (
            "variant", "C", "language", "word_count", "top1", "top1_ci95_low",
            "top1_ci95_high", "mrr", "mrr_ci95_low", "mrr_ci95_high",
            "top3", "top3_ci95_low", "top3_ci95_high", "selected_C",
        ),
        language_rows,
    )

    transfer_rows = _make_transfer_rows(
        ranks_by_model, dev_instances, counts, selected_C
    )
    _write_csv(
        output_dir / "language-transfer-support.csv",
        (
            "frequency_family", "language", "cross_language_only_gold_count", "C",
            "pooled_variant", "language_variant", "pooled_selected_C",
            "language_selected_C", "pooled_top1", "language_top1",
            "delta_top1", "delta_top1_ci95_low", "delta_top1_ci95_high",
            "pooled_mrr", "language_mrr", "delta_mrr", "delta_mrr_ci95_low",
            "delta_mrr_ci95_high",
        ),
        transfer_rows,
    )

    usage_rows = []
    trajectories = []
    for variant in VARIANT_CHANNELS:
        for instance_index, instance in enumerate(dev_instances):
            correctness = []
            gold_ranks = []
            for C in T1_C_GRID:
                rank = int(ranks_by_model[(variant, C)][instance_index])
                top = top_rows_by_model[(variant, C)][instance_index]
                correct = rank == 1
                correctness.append(correct)
                gold_ranks.append(rank)
                usage_rows.append(
                    {
                        "variant": variant, "C": C,
                        "language": instance.language, "text_id": instance.text_id,
                        "word_id": instance.word_id,
                        "candidate_count": len(instance.candidates),
                        "gold_rank": rank, "top1_wordform_id": top.wordform_id,
                        "top1_gramset": top.gramset, "top1_correct": int(correct),
                    }
                )
            trajectory = summarize_trajectory(correctness, T1_C_GRID)
            trajectory_row = {
                "variant": variant, "language": instance.language,
                "text_id": instance.text_id, "word_id": instance.word_id,
                "top1_trajectory": trajectory.trajectory,
                "n_wrong": trajectory.n_wrong, "n_runs": trajectory.n_runs,
                "n_flips": trajectory.n_flips,
                "c_threshold": "" if trajectory.c_threshold is None else trajectory.c_threshold,
                "flip_direction": trajectory.flip_direction,
                "category": trajectory.category,
            }
            trajectory_row.update(
                {f"gold_rank_C{C:g}": rank for C, rank in zip(T1_C_GRID, gold_ranks)}
            )
            trajectories.append(trajectory_row)

    _write_csv(
        output_dir / "usage-decisions.csv",
        (
            "variant", "C", "language", "text_id", "word_id", "candidate_count",
            "gold_rank", "top1_wordform_id", "top1_gramset", "top1_correct",
        ),
        usage_rows,
    )
    _write_csv(
        output_dir / "trajectories.csv",
        (
            "variant", "language", "text_id", "word_id", "top1_trajectory",
            "n_wrong", "n_runs", "n_flips", "c_threshold", "flip_direction",
            "category", *(f"gold_rank_C{C:g}" for C in T1_C_GRID),
        ),
        trajectories,
    )

    decision_rows = []
    for name, with_variant, without_variant in PAIRED_COMPARISONS:
        for C in T1_C_GRID:
            with_correct = ranks_by_model[(with_variant, C)] == 1
            without_correct = ranks_by_model[(without_variant, C)] == 1
            gains = int(np.count_nonzero(with_correct & ~without_correct))
            losses = int(np.count_nonzero(~with_correct & without_correct))
            decision_rows.append(
                {
                    "effect": name, "C": C,
                    "with_variant": with_variant, "without_variant": without_variant,
                    "improved_words": gains, "worsened_words": losses,
                    "net_correct_words": gains - losses,
                    "unchanged_words": len(dev_instances) - gains - losses,
                }
            )
    _write_csv(
        output_dir / "decision-change-summary.csv",
        (
            "effect", "C", "with_variant", "without_variant", "improved_words",
            "worsened_words", "net_correct_words", "unchanged_words",
        ),
        decision_rows,
    )

    selected = {}
    for variant in VARIANT_CHANNELS:
        C = selected_C[variant]
        key = (variant, C)
        selected[variant] = {
            "C": C,
            "selection_rule": "highest dev Top-1, then highest MRR, then lowest C",
            "point": point[key],
            "ci95_descriptive_after_dev_selection": {
                metric: [
                    float(np.quantile(bootstrap[key][metric], 0.025)),
                    float(np.quantile(bootstrap[key][metric], 0.975)),
                ]
                for metric in METRICS
            },
        }

    code_paths = (
        Path(__file__),
        ROOT / "src" / "t1_features" / "frequency_features.py",
        ROOT / "src" / "t1_features" / "features.py",
        ROOT / "src" / "t1_features" / "model.py",
        ROOT / "src" / "t1_features" / "vectorizer.py",
        ROOT / "src" / "t1_features" / "ranking.py",
        ROOT / "src" / "t1_features" / "data.py",
        ROOT / "src" / "t1_features" / "prediction_io.py",
        ROOT / "src" / "core" / "frequency.py",
        ROOT / "src" / "benchmark_context.py",
        ROOT / "src" / "core" / "instances.py",
    )
    pooled_pair_count = len(counts.analysis_pooled)
    pooled_gramset_count = len(counts.gramset_pooled)
    support_counts = {}
    for language in sorted({instance.language for instance in dev_instances}):
        support_counts[language] = {}
        for family in ("analysis", "gramset"):
            count = 0
            for instance in dev_instances:
                if instance.language != language:
                    continue
                gold = instance.gold_analysis
                if family == "analysis":
                    local = counts.analysis_language.get(
                        (language, gold.wordform_id, gold.gramset), 0
                    )
                    pooled = counts.analysis_pooled.get(
                        (gold.wordform_id, gold.gramset), 0
                    )
                else:
                    local = counts.gramset_language.get((language, gold.gramset), 0)
                    pooled = counts.gramset_pooled.get(gold.gramset, 0)
                count += int(local == 0 and pooled > 0)
            support_counts[language][family] = count

    payload = {
        "protocol": {
            "data_tag": tag,
            "split": "dev",
            "train_instances": len(train_instances),
            "dev_instances": len(dev_instances),
            "languages": sorted({instance.language for instance in train_instances}),
            "suffix_max_length": 5,
            "context_window": 1,
            "include_affix": False,
            "frequency_count_source": "train gold analyses only",
            "frequency_transform": "log1p(count)",
            "zero_count_indicator": True,
            "variants": {name: list(channels) for name, channels in VARIANT_CHANNELS.items()},
            "c_grid": list(T1_C_GRID),
            "selection_rule": "highest dev Top-1, then highest MRR, then lowest C",
            "bootstrap_unit": "(language, text_id)",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "confidence_interval": "percentile 95%",
            "per_language_metrics": True,
            "cross_language_only_gold_support_counts": support_counts,
            "test_used": False,
        },
        "data": {"root": str(data_dir), "tag": tag, "commit": _git_head(data_dir)},
        "split": {"path": str(split_path), "sha256": _sha256(split_path)},
        "training_frequency_counts": {
            "unique_analysis_pairs": pooled_pair_count,
            "unique_gramsets": pooled_gramset_count,
            "analysis_pairs_shared_across_languages": sum(
                1
                for key in counts.analysis_pooled
                if sum(
                    counts.analysis_language.get((language, key[0], key[1]), 0) > 0
                    for language in payload_languages
                ) > 1
            ) if (payload_languages := sorted({i.language for i in train_instances})) else 0,
            "gramsets_shared_across_languages": sum(
                1
                for gramset in counts.gramset_pooled
                if sum(
                    counts.gramset_language.get((language, gramset), 0) > 0
                    for language in sorted({i.language for i in train_instances})
                ) > 1
            ),
        },
        "reference_experiment": str(reference_dir),
        "repository_head": _git_head(ROOT),
        "source_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in code_paths
        },
        "selected_C": selected,
        "same_C_paired_effects": effects_summary,
        "prediction_files": {
            variant: {
                str(C): str(
                    predictions_by_model[(variant, C)].resolve().relative_to(output_dir.parent.resolve())
                )
                for C in T1_C_GRID
            }
            for variant in VARIANT_CHANNELS
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=ROOT / "work" / "experiments" / "v2026.11" / "context-ablation-k5-v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "work" / "experiments" / "v2026.11" / "t1-frequency-k5-window1-v1",
    )
    args = parser.parse_args()
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir
    args.output_dir = args.output_dir.resolve()
    try:
        payload = run_experiment(args)
    except (
        T1FrequencyAblationError,
        CorpusTagError,
        DataError,
        FeatureConstructionError,
        FrequencyFeatureError,
        BenchmarkIntegrityError,
        SplitError,
        T1PredictionWriteError,
        OSError,
        ValueError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("T1.2 frequency-feature ablation: dev")
    for variant, result in payload["selected_C"].items():
        point = result["point"]
        interval = result["ci95_descriptive_after_dev_selection"]["top1"]
        print(
            f"{variant}: C={result['C']:g}, Top-1={point['top1']:.4f} "
            f"[{interval[0]:.4f}, {interval[1]:.4f}], "
            f"MRR={point['mrr']:.4f}, Top-3={point['top3']:.4f}"
        )
    print(f"Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
