#!/usr/bin/env python3
"""Compare T1 feature configurations on the approved v2026.11 dev C grid."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_context import _preflight_benchmark, load_benchmark_context  # noqa: E402
from commands.evaluate_predictions import load_ranked_predictions  # noqa: E402
from core.data import DataError  # noqa: E402
from core.instances import CorpusTagError, Instance  # noqa: E402
from core.predictions import (  # noqa: E402
    BenchmarkIntegrityError,
    PredictionCsvParseError,
    PredictionFileReadError,
    validate_predictions,
)
from core.splits import SplitError  # noqa: E402
from t1_features.grid_diagnostics import T1_C_GRID  # noqa: E402
from t1_features.run_context import T1RunContextError, load_t1_run_context  # noqa: E402

BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 20261109
METRICS = ("top1", "mrr", "top3")


class AnalysisError(ValueError):
    """The feature-grid comparison inputs are incomplete or inconsistent."""


@dataclass(frozen=True)
class RunInput:
    variant: str
    C: float
    run_dir: Path


def _parse_run(value: str) -> RunInput:
    left, separator, raw_path = value.partition("=")
    variant, variant_separator, raw_C = left.partition(":")
    if not separator or not variant_separator or not variant or not raw_path:
        raise argparse.ArgumentTypeError("expected VARIANT:C=RUN_DIRECTORY")
    try:
        C = float(raw_C)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid C value {raw_C!r}") from exc
    if not np.isfinite(C) or C <= 0:
        raise argparse.ArgumentTypeError(f"C must be finite and positive, got {raw_C!r}")
    return RunInput(variant, C, Path(raw_path))


def _prediction_ranks(path: Path, instances: tuple[Instance, ...]) -> np.ndarray:
    rows_by_word: dict[int, list] = defaultdict(list)
    for row in load_ranked_predictions(path):
        rows_by_word[row.word_id].append(row)
    ranks = []
    for instance in instances:
        gold = instance.gold_analysis
        matches = [
            row.rank
            for row in rows_by_word.get(instance.word_id, ())
            if (row.wordform_id, row.gramset) == (gold.wordform_id, gold.gramset)
        ]
        if len(matches) != 1:
            raise AnalysisError(
                f"{instance.language} word_id={instance.word_id}: expected one gold "
                f"prediction, found {len(matches)} in {path}"
            )
        ranks.append(matches[0])
    return np.asarray(ranks, dtype=np.int64)


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
        raise AnalysisError("dev has no text groups")
    text_index = {key: index for index, key in enumerate(text_keys)}
    word_text_indices = np.asarray(
        [text_index[(row.language, int(row.text_id))] for row in instances],
        dtype=np.int64,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = {
        key: {metric: np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64) for metric in METRICS}
        for key in ranks_by_model
    }
    for sample_index in range(BOOTSTRAP_SAMPLES):
        sampled = rng.integers(0, len(text_keys), size=len(text_keys))
        text_multiplicity = np.bincount(sampled, minlength=len(text_keys))
        word_weights = text_multiplicity[word_text_indices]
        denominator = int(word_weights.sum())
        if denominator == 0:
            raise AnalysisError("bootstrap sample unexpectedly contains no words")
        for key, ranks in ranks_by_model.items():
            bootstrap[key]["top1"][sample_index] = np.dot(word_weights, ranks == 1) / denominator
            bootstrap[key]["mrr"][sample_index] = np.dot(word_weights, 1.0 / ranks) / denominator
            bootstrap[key]["top3"][sample_index] = np.dot(word_weights, ranks <= 3) / denominator
    return bootstrap


def _interval(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def _write_outputs(
    output_dir: Path,
    payload: dict,
    grid_rows: list[dict],
    paired_rows: list[dict],
    factorial_rows: list[dict],
) -> None:
    output_dir.mkdir(parents=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "grid.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = (
            "variant",
            "C",
            "top1",
            "top1_ci95_low",
            "top1_ci95_high",
            "mrr",
            "mrr_ci95_low",
            "mrr_ci95_high",
            "top3",
            "selected_C",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(grid_rows)
    with (output_dir / "paired.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = (
            "variant_a",
            "variant_b",
            "C_a",
            "C_b",
            "delta_top1",
            "delta_top1_ci95_low",
            "delta_top1_ci95_high",
            "delta_mrr",
            "delta_mrr_ci95_low",
            "delta_mrr_ci95_high",
            "delta_top3",
            "delta_top3_ci95_low",
            "delta_top3_ci95_high",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(paired_rows)
    with (output_dir / "factorial.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = (
            "effect",
            "C",
            "delta_top1",
            "delta_top1_ci95_low",
            "delta_top1_ci95_high",
            "delta_mrr",
            "delta_mrr_ci95_low",
            "delta_mrr_ci95_high",
            "delta_top3",
            "delta_top3_ci95_low",
            "delta_top3_ci95_high",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(factorial_rows)


def run_analysis(args: argparse.Namespace) -> dict:
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise AnalysisError(f"output directory already exists: {args.output_dir}")
    data_dir, tag = _preflight_benchmark(args.data_dir)
    if tag != "v2026.11":
        raise AnalysisError(f"analysis is pinned to v2026.11, got {tag!r}")
    benchmark = load_benchmark_context(data_dir, tag, args.split_file)
    split_by_text = {
        (language, int(text_id)): split
        for language, text_id, split in benchmark.split_rows
    }
    instances = tuple(
        row
        for row in benchmark.instances
        if split_by_text.get((row.language, int(row.text_id))) == "dev"
    )
    if len(instances) != 2613:
        raise AnalysisError(f"expected 2613 dev instances, found {len(instances)}")
    if len(args.run) < 2:
        raise AnalysisError("at least two feature variants are required")

    grouped_runs: dict[str, dict[float, tuple[dict, Path]]] = {}
    feature_config_by_variant = {}
    provenance = None
    for run in args.run:
        context = load_t1_run_context(run.run_dir)
        manifest = context.manifest
        parameters = manifest["parameters"]
        if parameters["C"] != run.C:
            raise AnalysisError(
                f"label C={run.C:g} does not match {context.run_dir} passport "
                f"C={parameters['C']:g}"
            )
        if parameters.get("target_split", "dev") != "dev":
            raise AnalysisError(f"run is not a dev experiment: {context.run_dir}")
        if manifest["data"]["tag"] != tag:
            raise AnalysisError(f"run uses data tag {manifest['data']['tag']!r}: {context.run_dir}")
        current_provenance = (manifest["data"], manifest["split"])
        if provenance is None:
            provenance = current_provenance
        elif provenance != current_provenance:
            raise AnalysisError("runs do not share identical data and split provenance")
        features = (
            parameters.get("suffix_max_length", 4),
            parameters.get("include_affix", False),
        )
        prior_features = feature_config_by_variant.setdefault(run.variant, features)
        if prior_features != features:
            raise AnalysisError(f"variant {run.variant!r} mixes feature configurations")
        variant_runs = grouped_runs.setdefault(run.variant, {})
        if run.C in variant_runs:
            raise AnalysisError(f"duplicate run for variant {run.variant!r}, C={run.C:g}")
        predictions = context.run_dir / "dev.csv"
        validation = validate_predictions(
            predictions,
            "dev",
            benchmark.instances,
            benchmark.split_rows,
        )
        if not validation.is_valid:
            raise AnalysisError(f"invalid predictions in {predictions}: {validation.errors}")
        variant_runs[run.C] = (manifest, predictions)

    variants = list(grouped_runs)
    for variant in variants:
        if tuple(sorted(grouped_runs[variant])) != T1_C_GRID:
            raise AnalysisError(
                f"variant {variant!r} must cover the complete approved C grid"
            )

    ranks_by_model = {}
    for variant in variants:
        for C in T1_C_GRID:
            predictions = grouped_runs[variant][C][1]
            ranks_by_model[(variant, C)] = _prediction_ranks(predictions, instances)
    bootstrap = _bootstrap_metrics(ranks_by_model, instances)
    point = {key: _point_metrics(ranks) for key, ranks in ranks_by_model.items()}

    selected_C = {}
    for variant in variants:
        selected_C[variant] = min(
            T1_C_GRID,
            key=lambda C: (
                -point[(variant, C)]["top1"],
                -point[(variant, C)]["mrr"],
                C,
            ),
        )

    grid_rows = []
    summary_variants = {}
    for variant in variants:
        summary_variants[variant] = {
            "features": {
                "suffix_max_length": feature_config_by_variant[variant][0],
                "include_affix": feature_config_by_variant[variant][1],
            },
            "selected_C": selected_C[variant],
            "selection_rule": "highest dev Top-1, then highest MRR, then lowest C",
            "grid": {},
        }
        for C in T1_C_GRID:
            model_key = (variant, C)
            metric_summary = {
                "point": point[model_key],
                "ci95": {metric: _interval(bootstrap[model_key][metric]) for metric in METRICS},
            }
            summary_variants[variant]["grid"][str(C)] = metric_summary
            grid_rows.append(
                {
                    "variant": variant,
                    "C": C,
                    "top1": point[model_key]["top1"],
                    "top1_ci95_low": metric_summary["ci95"]["top1"][0],
                    "top1_ci95_high": metric_summary["ci95"]["top1"][1],
                    "mrr": point[model_key]["mrr"],
                    "mrr_ci95_low": metric_summary["ci95"]["mrr"][0],
                    "mrr_ci95_high": metric_summary["ci95"]["mrr"][1],
                    "top3": point[model_key]["top3"],
                    "selected_C": C == selected_C[variant],
                }
            )

    paired_rows = []
    paired_summary = []
    for index, variant_a in enumerate(variants):
        for variant_b in variants[index + 1 :]:
            for C in T1_C_GRID:
                key_a = (variant_a, C)
                key_b = (variant_b, C)
                deltas = {
                    metric: bootstrap[key_b][metric] - bootstrap[key_a][metric]
                    for metric in METRICS
                }
                row = {
                    "variant_a": variant_a,
                    "variant_b": variant_b,
                    "C_a": C,
                    "C_b": C,
                    "delta_top1": point[key_b]["top1"] - point[key_a]["top1"],
                    "delta_top1_ci95_low": _interval(deltas["top1"])[0],
                    "delta_top1_ci95_high": _interval(deltas["top1"])[1],
                    "delta_mrr": point[key_b]["mrr"] - point[key_a]["mrr"],
                    "delta_mrr_ci95_low": _interval(deltas["mrr"])[0],
                    "delta_mrr_ci95_high": _interval(deltas["mrr"])[1],
                    "delta_top3": point[key_b]["top3"] - point[key_a]["top3"],
                    "delta_top3_ci95_low": _interval(deltas["top3"])[0],
                    "delta_top3_ci95_high": _interval(deltas["top3"])[1],
                }
                paired_rows.append(row)
                paired_summary.append(row)

    selected_pairwise = []
    for index, variant_a in enumerate(variants):
        for variant_b in variants[index + 1 :]:
            C_a = selected_C[variant_a]
            C_b = selected_C[variant_b]
            key_a = (variant_a, C_a)
            key_b = (variant_b, C_b)
            deltas = {
                metric: bootstrap[key_b][metric] - bootstrap[key_a][metric]
                for metric in METRICS
            }
            selected_pairwise.append(
                {
                    "variant_a": variant_a,
                    "C_a": C_a,
                    "variant_b": variant_b,
                    "C_b": C_b,
                    "delta_point": {
                        metric: point[key_b][metric] - point[key_a][metric]
                        for metric in METRICS
                    },
                    "delta_ci95": {
                        metric: _interval(deltas[metric]) for metric in METRICS
                    },
                    "interpretation": "paired dev bootstrap conditional on each variant's selected C",
                }
            )

    factorial_rows = []
    factorial_summary = []
    factor_cells = {}
    for variant, (suffix_max_length, include_affix) in feature_config_by_variant.items():
        cell = (suffix_max_length > 0, include_affix)
        if cell in factor_cells:
            raise AnalysisError(f"duplicate factorial feature cell {cell!r}")
        factor_cells[cell] = variant
    expected_cells = {(False, False), (True, False), (False, True), (True, True)}
    if set(factor_cells) == expected_cells:
        suffix_lengths = {
            config[0]
            for config in feature_config_by_variant.values()
            if config[0] > 0
        }
        if len(suffix_lengths) != 1:
            raise AnalysisError("factorial suffix-on variants must use the same K")
        control = factor_cells[(False, False)]
        suffix_only = factor_cells[(True, False)]
        affix_only = factor_cells[(False, True)]
        both = factor_cells[(True, True)]
        effects = {
            "suffix_without_affix": ((suffix_only, 1), (control, -1)),
            "suffix_with_affix": ((both, 1), (affix_only, -1)),
            "affix_without_suffix": ((affix_only, 1), (control, -1)),
            "affix_with_suffix": ((both, 1), (suffix_only, -1)),
            "interaction": ((both, 1), (affix_only, -1), (suffix_only, -1), (control, 1)),
        }
        for effect_name, terms in effects.items():
            for C in T1_C_GRID:
                effect_point = {
                    metric: sum(
                        sign * point[(variant, C)][metric]
                        for variant, sign in terms
                    )
                    for metric in METRICS
                }
                effect_bootstrap = {
                    metric: sum(
                        sign * bootstrap[(variant, C)][metric]
                        for variant, sign in terms
                    )
                    for metric in METRICS
                }
                row = {
                    "effect": effect_name,
                    "C": C,
                    "delta_top1": effect_point["top1"],
                    "delta_top1_ci95_low": _interval(effect_bootstrap["top1"])[0],
                    "delta_top1_ci95_high": _interval(effect_bootstrap["top1"])[1],
                    "delta_mrr": effect_point["mrr"],
                    "delta_mrr_ci95_low": _interval(effect_bootstrap["mrr"])[0],
                    "delta_mrr_ci95_high": _interval(effect_bootstrap["mrr"])[1],
                    "delta_top3": effect_point["top3"],
                    "delta_top3_ci95_low": _interval(effect_bootstrap["top3"])[0],
                    "delta_top3_ci95_high": _interval(effect_bootstrap["top3"])[1],
                }
                factorial_rows.append(row)
                factorial_summary.append(row)

    payload = {
        "protocol": {
            "data_tag": tag,
            "split": "dev",
            "bootstrap_unit": "(language, text_id)",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "confidence_interval": "percentile 95%",
            "c_grid": list(T1_C_GRID),
            "selection_rule": "highest dev Top-1, then highest MRR, then lowest C",
            "test_used": False,
        },
        "data": provenance[0],
        "split_provenance": provenance[1],
        "variants": summary_variants,
        "paired_differences_same_C": paired_summary,
        "paired_differences_selected_C": selected_pairwise,
        "factorial_effects_same_C": factorial_summary,
    }
    _write_outputs(args.output_dir, payload, grid_rows, paired_rows, factorial_rows)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--run", action="append", type=_parse_run, required=True, metavar="VARIANT:C=RUN_DIR")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = run_analysis(args)
    except (
        AnalysisError,
        CorpusTagError,
        DataError,
        SplitError,
        BenchmarkIntegrityError,
        PredictionCsvParseError,
        PredictionFileReadError,
        T1RunContextError,
        OSError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"T1 feature-grid analysis: {payload['protocol']['data_tag']} dev")
    for variant, details in payload["variants"].items():
        C = details["selected_C"]
        metrics = details["grid"][str(C)]["point"]
        interval = details["grid"][str(C)]["ci95"]["top1"]
        features = details["features"]
        print(
            f"{variant}: K={features['suffix_max_length']}, "
            f"affix={features['include_affix']}, C={C:g}, "
            f"Top-1={metrics['top1']:.4f} [{interval[0]:.4f}, {interval[1]:.4f}], "
            f"MRR={metrics['mrr']:.4f}, Top-3={metrics['top3']:.4f}"
        )
    print(f"Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
