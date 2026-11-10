#!/usr/bin/env python3
"""Run the approved cumulative T1 context-window sweep on dev."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
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
VARIANTS = ("no_context", "window_1", "window_2", "window_3")
NEW_WINDOWS = (2, 3)
COMPARISONS = (
    ("window_1_vs_no_context", "window_1", "no_context"),
    ("window_2_vs_window_1", "window_2", "window_1"),
    ("window_3_vs_window_2", "window_3", "window_2"),
    ("window_3_vs_no_context", "window_3", "no_context"),
)
REFERENCE_VARIANTS = {"no_context": "no_context", "window_1": "both"}


class ContextWindowSweepError(ValueError):
    """The T1 context-window sweep could not be completed consistently."""


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
            raise ContextWindowSweepError(
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
        raise ContextWindowSweepError("dev has no text groups")
    text_index = {key: index for index, key in enumerate(text_keys)}
    word_text_indices = np.asarray(
        [text_index[(row.language, int(row.text_id))] for row in instances],
        dtype=np.int64,
    )
    bootstrap = {
        key: {
            metric: np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
            for metric in METRICS
        }
        for key in ranks_by_model
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for sample_index in range(BOOTSTRAP_SAMPLES):
        sampled = rng.integers(0, len(text_keys), size=len(text_keys))
        multiplicity = np.bincount(sampled, minlength=len(text_keys))
        weights = multiplicity[word_text_indices]
        denominator = int(weights.sum())
        if denominator == 0:
            raise ContextWindowSweepError("bootstrap sample contains no dev words")
        for key, ranks in ranks_by_model.items():
            bootstrap[key]["top1"][sample_index] = np.dot(weights, ranks == 1) / denominator
            bootstrap[key]["mrr"][sample_index] = np.dot(weights, 1.0 / ranks) / denominator
            bootstrap[key]["top3"][sample_index] = np.dot(weights, ranks <= 3) / denominator
    return bootstrap


def _interval(values: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _select_C(
    variant: str,
    point: dict[tuple[str, float], dict[str, float]],
) -> float:
    return min(
        T1_C_GRID,
        key=lambda C: (
            -point[(variant, C)]["top1"],
            -point[(variant, C)]["mrr"],
            C,
        ),
    )


def _load_reference_predictions(
    source_dir: Path,
    variant: str,
    C: float,
    dev_instances: tuple[Instance, ...],
    benchmark_instances: tuple[Instance, ...],
    split_rows: tuple[tuple[str, int, str], ...],
) -> tuple[np.ndarray, tuple[PredictionRow, ...], Path]:
    reference_variant = REFERENCE_VARIANTS[variant]
    run_dir = source_dir / "runs" / reference_variant / f"c-{C:g}"
    run_path = run_dir / "run.json"
    prediction_path = run_dir / "dev.csv"
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContextWindowSweepError(f"could not read reference run {run_path}: {exc}") from exc
    parameters = run.get("parameters", {})
    expected = {
        "C": C,
        "target_split": "dev",
        "suffix_max_length": 5,
        "include_affix": False,
        "include_left_context": variant == "window_1",
        "include_right_context": variant == "window_1",
    }
    for name, value in expected.items():
        if parameters.get(name) != value:
            raise ContextWindowSweepError(
                f"reference {run_path} has unexpected {name}: {parameters.get(name)!r}"
            )
    validation = validate_predictions(
        prediction_path, "dev", benchmark_instances, split_rows
    )
    if not validation.is_valid:
        raise ContextWindowSweepError(
            f"invalid reference predictions in {prediction_path}: {validation.errors}"
        )
    return _rank_details(load_ranked_predictions(prediction_path), dev_instances) + (
        prediction_path,
    )


def run_experiment(args: argparse.Namespace) -> dict:
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ContextWindowSweepError(f"output already exists: {args.output_dir}")
    data_dir, tag = _preflight_benchmark(args.data_dir)
    if tag != "v2026.11":
        raise ContextWindowSweepError(f"expected data v2026.11, found {tag!r}")
    split_path = args.split_file or ROOT / "data" / "derived" / f"splits_{tag}.csv"
    split_path = split_path.expanduser().resolve()
    source_dir = args.reference_dir.expanduser().resolve()

    train_inputs = load_t1_inputs(
        data_dir, tag, split="train", split_file_override=split_path
    )
    dev_inputs = load_t1_inputs(
        data_dir, tag, split="dev", split_file_override=split_path
    )
    train_instances = train_inputs.instances
    dev_instances = dev_inputs.instances
    if len(dev_instances) != 2613:
        raise ContextWindowSweepError(
            f"expected 2613 dev instances, found {len(dev_instances)}"
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True)
    run_root = output_dir / "runs"
    run_root.mkdir()
    ranks_by_model: dict[tuple[str, float], np.ndarray] = {}
    top_rows_by_model: dict[tuple[str, float], tuple[PredictionRow, ...]] = {}
    predictions_by_model: dict[tuple[str, float], Path] = {}

    for variant in ("no_context", "window_1"):
        for C in T1_C_GRID:
            ranks, top_rows, prediction_path = _load_reference_predictions(
                source_dir,
                variant,
                C,
                dev_instances,
                dev_inputs.benchmark.instances,
                dev_inputs.benchmark.split_rows,
            )
            key = (variant, C)
            ranks_by_model[key] = ranks
            top_rows_by_model[key] = top_rows
            predictions_by_model[key] = prediction_path

    for window in NEW_WINDOWS:
        variant = f"window_{window}"
        print(f"building examples for {variant}", flush=True)
        training_examples = build_candidate_examples(
            train_instances,
            train_inputs.context_words,
            mode="train",
            context_window=window,
            suffix_max_length=5,
            include_affix=False,
        )
        dev_examples = build_candidate_examples(
            dev_instances,
            dev_inputs.context_words,
            mode="predict",
            context_window=window,
            suffix_max_length=5,
            include_affix=False,
        )
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
                raise ContextWindowSweepError(
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
                            "include_affix": False,
                            "context_window": window,
                            "cumulative_context_positions": [
                                -distance for distance in range(1, window + 1)
                            ]
                            + list(range(1, window + 1)),
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
    selected_C = {variant: _select_C(variant, point) for variant in VARIANTS}

    grid_rows = []
    for variant in VARIANTS:
        for C in T1_C_GRID:
            key = (variant, C)
            row = {"variant": variant, "C": C, "selected_C": C == selected_C[variant]}
            for metric in METRICS:
                low, high = _interval(bootstrap[key][metric])
                row[metric] = point[key][metric]
                row[f"{metric}_ci95_low"] = low
                row[f"{metric}_ci95_high"] = high
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

    effect_rows = []
    effect_summary = {}
    for name, with_variant, without_variant in COMPARISONS:
        effect_summary[name] = {}
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
                low, high = _interval(samples)
                row[f"delta_{metric}"] = delta
                row[f"delta_{metric}_ci95_low"] = low
                row[f"delta_{metric}_ci95_high"] = high
                metric_summary[metric] = {"point": delta, "ci95": [low, high]}
            effect_summary[name][str(C)] = metric_summary
            effect_rows.append(row)
    _write_csv(
        output_dir / "effects.csv",
        (
            "effect", "with_variant", "without_variant", "C",
            "delta_top1", "delta_top1_ci95_low", "delta_top1_ci95_high",
            "delta_mrr", "delta_mrr_ci95_low", "delta_mrr_ci95_high",
            "delta_top3", "delta_top3_ci95_low", "delta_top3_ci95_high",
        ),
        effect_rows,
    )

    usage_rows = []
    trajectories = []
    decision_rows = []
    for variant in VARIANTS:
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

    for name, with_variant, without_variant in COMPARISONS:
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
    _write_csv(
        output_dir / "decision-change-summary.csv",
        (
            "effect", "C", "with_variant", "without_variant", "improved_words",
            "worsened_words", "net_correct_words", "unchanged_words",
        ),
        decision_rows,
    )

    selected = {}
    for variant in VARIANTS:
        C = selected_C[variant]
        key = (variant, C)
        selected[variant] = {
            "C": C,
            "selection_rule": "highest dev Top-1, then highest MRR, then lowest C",
            "point": point[key],
            "ci95_descriptive_after_dev_selection": {
                metric: list(_interval(bootstrap[key][metric])) for metric in METRICS
            },
        }

    code_paths = (
        Path(__file__), ROOT / "scripts" / "t1-context-ablation.py",
        ROOT / "src" / "t1_features" / "features.py",
        ROOT / "src" / "t1_features" / "model.py",
        ROOT / "src" / "t1_features" / "vectorizer.py",
        ROOT / "src" / "t1_features" / "ranking.py",
        ROOT / "src" / "t1_features" / "data.py",
        ROOT / "src" / "t1_features" / "prediction_io.py",
        ROOT / "src" / "benchmark_context.py",
        ROOT / "src" / "core" / "instances.py",
    )
    payload = {
        "protocol": {
            "data_tag": tag, "split": "dev",
            "dev_instances": len(dev_instances), "train_instances": len(train_instances),
            "suffix_max_length": 5, "include_affix": False,
            "window_semantics": "cumulative symmetric offsets; window W includes all distances 1..W",
            "variants": list(VARIANTS), "newly_trained_windows": list(NEW_WINDOWS),
            "c_grid": list(T1_C_GRID),
            "selection_rule": "highest dev Top-1, then highest MRR, then lowest C",
            "bootstrap_unit": "(language, text_id)",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "confidence_interval": "percentile 95%",
            "test_used": False,
        },
        "data": {"root": str(data_dir), "tag": tag, "commit": _git_head(data_dir)},
        "split": {"path": str(split_path), "sha256": _sha256(split_path)},
        "reference_experiment": str(source_dir),
        "repository_head": _git_head(ROOT),
        "source_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in code_paths
        },
        "selected_C": selected,
        "same_C_paired_effects": effect_summary,
        "prediction_files": {
            variant: {
                str(C): str(predictions_by_model[(variant, C)].resolve().relative_to(output_dir.parent.resolve()))
                for C in T1_C_GRID
            }
            for variant in VARIANTS
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
        default=ROOT / "work" / "experiments" / "v2026.11" / "context-window-sweep-k5-v1",
    )
    args = parser.parse_args()
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir
    args.output_dir = args.output_dir.resolve()
    try:
        payload = run_experiment(args)
    except (
        ContextWindowSweepError, CorpusTagError, DataError, FeatureConstructionError,
        BenchmarkIntegrityError, SplitError, T1PredictionWriteError, OSError,
        ValueError, subprocess.CalledProcessError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("T1 cumulative context-window sweep: dev")
    for variant, result in payload["selected_C"].items():
        point = result["point"]
        ci = result["ci95_descriptive_after_dev_selection"]["top1"]
        print(
            f"{variant}: C={result['C']:g}, Top-1={point['top1']:.4f} "
            f"[{ci[0]:.4f}, {ci[1]:.4f}], MRR={point['mrr']:.4f}, "
            f"Top-3={point['top3']:.4f}"
        )
    print(f"Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
