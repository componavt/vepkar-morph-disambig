#!/usr/bin/env python3
"""Summarize the T1.1 C grid or the frozen G1 test comparison."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_context import _preflight_benchmark, load_benchmark_context  # noqa: E402
from commands.evaluate_predictions import load_ranked_predictions  # noqa: E402
from core.data import DataError  # noqa: E402
from core.instances import Instance  # noqa: E402
from core.predictions import validate_predictions  # noqa: E402
from core.splits import SplitError  # noqa: E402
from t1_features.run_context import load_t1_run_context  # noqa: E402

SEED = 20261109
BOOTSTRAP_SAMPLES = 1000
CASES = {"NOM", "ACC", "GEN"}


def _parse_run(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("expected C=RUN_DIRECTORY")
    return label, Path(raw_path)


def _rows_by_word(path: Path) -> dict[int, list]:
    rows_by_word: dict[int, list] = defaultdict(list)
    for row in load_ranked_predictions(path):
        rows_by_word[row.word_id].append(row)
    return rows_by_word


def _gold_rank_map(
    instances: tuple[Instance, ...], rows_by_word: dict[int, list]
) -> dict[int, int]:
    result: dict[int, int] = {}
    for instance in instances:
        gold = instance.gold_analysis
        matches = [
            row.rank
            for row in rows_by_word.get(instance.word_id, ())
            if row.wordform_id == gold.wordform_id and row.gramset == gold.gramset
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{instance.language} word_id={instance.word_id}: "
                f"expected one gold prediction, found {len(matches)}"
            )
        result[instance.word_id] = matches[0]
    return result


def _counts(instances: tuple[Instance, ...], ranks: dict[int, int], multiplicity=None):
    correct = reciprocal = top3 = words = 0.0
    for index, instance in enumerate(instances):
        weight = 1 if multiplicity is None else multiplicity[index]
        if not weight:
            continue
        rank = ranks[instance.word_id]
        words += weight
        correct += weight * (rank == 1)
        reciprocal += weight / rank
        top3 += weight * (rank <= 3)
    if not words:
        raise ValueError("cannot compute metrics for an empty group")
    return {
        "word_count": int(words),
        "top1_accuracy": correct / words,
        "mrr": reciprocal / words,
        "top3_accuracy": top3 / words,
    }


def _summarize_group(instances, rankings, rng):
    point = {name: _counts(instances, ranks) for name, ranks in rankings.items()}
    text_words: dict[tuple[str, int], list[int]] = defaultdict(list)
    for instance in instances:
        text_words[(instance.language, int(instance.text_id))].append(instance.word_id)
    text_keys = sorted(text_words)
    if not text_keys:
        return {"point": point, "intervals": {}, "differences": {}}

    rows_by_model = {
        name: [
            [rankings[name][word_id] for word_id in text_words[key]]
            for key in text_keys
        ]
        for name in rankings
    }
    boot: dict[str, dict[str, list[float]]] = {
        name: {metric: [] for metric in ("top1_accuracy", "mrr", "top3_accuracy")}
        for name in rankings
    }
    model_names = list(rankings)
    differences: dict[str, dict[str, list[float]]] = {}
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled = rng.integers(0, len(text_keys), size=len(text_keys))
        multiplicity = np.bincount(sampled, minlength=len(text_keys))
        denominators = sum(
            int(multiplicity[i]) * len(rows_by_model[model_names[0]][i])
            for i in range(len(text_keys))
        )
        values: dict[str, dict[str, float]] = {}
        for name in model_names:
            top1 = mrr = top3 = 0.0
            for i, weight in enumerate(multiplicity):
                if not weight:
                    continue
                for rank in rows_by_model[name][i]:
                    top1 += int(weight) * (rank == 1)
                    mrr += int(weight) / rank
                    top3 += int(weight) * (rank <= 3)
            values[name] = {
                "top1_accuracy": top1 / denominators,
                "mrr": mrr / denominators,
                "top3_accuracy": top3 / denominators,
            }
            for metric, value in values[name].items():
                boot[name][metric].append(value)
        if len(model_names) > 1:
            for index, name in enumerate(model_names[1:], start=1):
                previous = model_names[index - 1]
                differences.setdefault(name, {m: [] for m in values[name]})
                for metric in values[name]:
                    differences[name][metric].append(
                        values[name][metric] - values[previous][metric]
                    )

    intervals = {
        name: {
            metric: [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]
            for metric, samples in metrics.items()
        }
        for name, metrics in boot.items()
    }
    difference_intervals = {
        name: {
            metric: [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]
            for metric, samples in metrics.items()
        }
        for name, metrics in differences.items()
    }
    return {
        "point": point,
        "intervals": intervals,
        "paired_differences_vs_previous_model": difference_intervals,
        "bootstrap_text_count": len(text_keys),
    }


def _slice_groups(instances, train_instances):
    train_gold_pairs = {
        (row.gold_analysis.wordform_id, row.gold_analysis.gramset)
        for row in train_instances
    }
    train_words = {(row.language, row.word) for row in train_instances}
    groups: dict[str, dict[str, list[Instance]]] = defaultdict(lambda: defaultdict(list))
    for row in instances:
        groups["language"][row.language].append(row)
        pair_known = (row.gold_analysis.wordform_id, row.gold_analysis.gramset) in train_gold_pairs
        groups["gold_pair_seen_in_train"]["yes" if pair_known else "no"].append(row)
        word_known = (row.language, row.word) in train_words
        groups["word_seen_in_train"]["yes" if word_known else "no"].append(row)
        count = len(row.candidates)
        bucket = "2" if count == 2 else "3" if count == 3 else "4-5" if count <= 5 else "6+"
        groups["candidate_count"][bucket].append(row)
        case_tags = {
            token
            for candidate in row.candidates
            for token in candidate.gramset.split("+")
            if token in CASES
        }
        case_group = "NOM_ACC_GEN_ambiguous" if len(case_tags) > 1 else "other"
        groups["NOM_ACC_GEN"][case_group].append(row)
    return groups


def _write_outputs(args, payload, slices, confusions):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "slices.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("slice", "group", "model", "n", "top1", "top1_ci95_low", "top1_ci95_high", "delta_top1_vs_previous", "delta_top1_ci95_low", "delta_top1_ci95_high", "mrr", "top3"))
        for slice_name, group_name, summary in slices:
            for model, metrics in summary["point"].items():
                ci = summary.get("intervals", {}).get(model, {}).get("top1_accuracy", [None, None])
                delta = summary.get("paired_differences_vs_previous_model", {}).get(model, {}).get("top1_accuracy")
                delta_point = None
                if delta is not None:
                    names = list(summary["point"])
                    previous = names[names.index(model) - 1]
                    delta_point = metrics["top1_accuracy"] - summary["point"][previous]["top1_accuracy"]
                writer.writerow((slice_name, group_name, model, metrics["word_count"], metrics["top1_accuracy"], ci[0], ci[1], delta_point, delta[0] if delta else None, delta[1] if delta else None, metrics["mrr"], metrics["top3_accuracy"]))
    with (args.output_dir / "top-confusions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("gold_gramset", "top1_gramset", "count"))
        writer.writerows(confusions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--t1-run", action="append", type=_parse_run, required=True, metavar="C=RUN_DIR")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        data_dir, tag = _preflight_benchmark(args.data_dir)
        benchmark = load_benchmark_context(data_dir, tag, args.split_file)
    except (DataError, SplitError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    split_texts = {(language, int(text_id)) for language, text_id, part in benchmark.split_rows if part == args.split}
    instances = tuple(row for row in benchmark.instances if (row.language, int(row.text_id)) in split_texts)
    train_instances = tuple(row for row in benchmark.instances if (row.language, int(row.text_id)) in {
        (language, int(text_id)) for language, text_id, part in benchmark.split_rows if part == "train"
    })
    model_names = []
    rankings = {}
    rows_by_model = {}
    manifests = []
    for label, run_dir in args.t1_run:
        context = load_t1_run_context(run_dir)
        manifests.append(context.manifest)
        predictions = context.run_dir / f"{args.split}.csv"
        validation = validate_predictions(predictions, args.split, benchmark.instances, benchmark.split_rows)
        if not validation.is_valid:
            raise ValueError(f"invalid {args.split} predictions in {predictions}: {validation.errors}")
        rows_by_word = _rows_by_word(predictions)
        rows_by_model[label] = rows_by_word
        rankings[label] = _gold_rank_map(instances, rows_by_word)
        model_names.append(label)
    if any(
        manifest["data"] != manifests[0]["data"] or manifest["split"] != manifests[0]["split"]
        for manifest in manifests[1:]
    ):
        raise ValueError("T1 runs do not share the same data and split identity")

    if args.baseline:
        validation = validate_predictions(args.baseline, args.split, benchmark.instances, benchmark.split_rows)
        if not validation.is_valid:
            raise ValueError(f"invalid {args.split} baseline predictions: {validation.errors}")
        rows_by_word = _rows_by_word(args.baseline)
        rows_by_model["frequency_baseline"] = rows_by_word
        rankings["frequency_baseline"] = _gold_rank_map(instances, rows_by_word)
        model_names.append("frequency_baseline")

    t1_name = args.t1_run[0][0]
    if args.baseline:
        model_names = ["frequency_baseline", t1_name]
        rankings = {name: rankings[name] for name in model_names}
    elif args.split == "dev" and len(model_names) != 10:
        raise ValueError("the G1 dev grid analysis requires all ten T1 C runs")

    rng = np.random.default_rng(SEED)
    overall = _summarize_group(instances, rankings, rng)
    slices = []
    if args.baseline:
        for slice_name, groups in _slice_groups(instances, train_instances).items():
            for group_name, group_instances in sorted(groups.items()):
                subset_rankings = {
                    model: {row.word_id: ranks[row.word_id] for row in group_instances}
                    for model, ranks in rankings.items()
                }
                summary = _summarize_group(tuple(group_instances), subset_rankings, rng)
                slices.append((slice_name, group_name, summary))

    top1_errors = Counter()
    for row in instances:
        if rankings[t1_name][row.word_id] != 1:
            gold = row.gold_analysis
            top = min(rows_by_model[t1_name][row.word_id], key=lambda prediction: prediction.rank)
            top1_errors[(gold.gramset, top.gramset)] += 1
    confusions = [(gold, top, count) for (gold, top), count in top1_errors.most_common(20)]
    payload = {
        "protocol": {
            "goal": "G1",
            "data_tag": tag,
            "data_commit": manifests[0]["data"]["commit"],
            "split_sha256": manifests[0]["split"]["sha256"],
            "split": args.split,
            "bootstrap_unit": "(language, text_id)",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": SEED,
            "ci": "percentile 95%",
        },
        "overall": overall,
        "models": model_names,
    }
    _write_outputs(args, payload, slices, confusions)
    print(f"G1 analysis: {args.split} {tag}")
    for model in model_names:
        metrics = overall["point"][model]
        ci = overall.get("intervals", {}).get(model, {}).get("top1_accuracy", [None, None])
        print(f"{model}: Top-1={metrics['top1_accuracy']:.4f} ({ci[0]:.4f}, {ci[1]:.4f}); MRR={metrics['mrr']:.4f}; Top-3={metrics['top3_accuracy']:.4f}")
    print(f"Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
