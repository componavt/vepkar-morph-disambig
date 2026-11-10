#!/usr/bin/env python3
"""Describe errors from the selected T1.1 dev prediction file."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.instances import CorpusTagError  # noqa: E402
from t1_features.data import load_t1_inputs  # noqa: E402


def _read_predictions(path: Path) -> dict[int, list[dict[str, str]]]:
    by_word: dict[int, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            by_word[int(row["word_id"])].append(row)
    for rows in by_word.values():
        rows.sort(key=lambda row: int(row["rank"]))
    return by_word


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sentence_views(instances, context_words):
    by_sentence: dict[tuple[str, int], list] = defaultdict(list)
    for word in context_words:
        by_sentence[(word.language, word.sentence_id)].append(word)
    for words in by_sentence.values():
        words.sort(key=lambda word: (word.word_number, word.word_id))

    result = {}
    for instance in instances:
        words = by_sentence[(instance.language, instance.sentence_id)]
        tokens = [word.word for word in words]
        target_index = next(
            index for index, word in enumerate(words) if word.word_id == instance.word_id
        )
        marked = tokens.copy()
        marked[target_index] = f"[{marked[target_index]}]"
        result[instance.word_id] = {
            "sentence": " ".join(tokens),
            "marked_sentence": " ".join(marked),
            "left_neighbor": tokens[target_index - 1] if target_index else "<BOS>",
            "right_neighbor": (
                tokens[target_index + 1]
                if target_index + 1 < len(tokens)
                else "<EOS>"
            ),
        }
    return result


def _count_summary(rows: list[dict], key: str, order: list[str]) -> list[dict]:
    counts = Counter(row[key] for row in rows)
    return [
        {
            key: value,
            "instances": counts[value],
            "errors": sum(row[key] == value and row["correct"] == 0 for row in rows),
            "top1_accuracy": (
                sum(row[key] == value and row["correct"] == 1 for row in rows)
                / counts[value]
            ),
        }
        for value in order
        if counts[value]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/dictorpus-data")
    parser.add_argument("--tag", default="v2026.11")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT
        / "work/experiments/v2026.11/context-ablation-k5-v1/runs/both/c-7/dev.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "work/experiments/v2026.11/t1-dev-error-analysis-v1",
    )
    args = parser.parse_args()

    if not args.predictions.is_file():
        parser.error(f"Prediction file does not exist: {args.predictions}")

    try:
        inputs = load_t1_inputs(args.data_dir, args.tag, split="dev")
    except (CorpusTagError, ValueError) as exc:
        parser.error(str(exc))

    predictions = _read_predictions(args.predictions)
    sentence_views = _sentence_views(inputs.instances, inputs.context_words)
    instance_rows: list[dict] = []
    confusion_counts = Counter()
    rank_counts = Counter()
    same_gramset_errors = 0
    error_rows: list[dict] = []

    for instance in inputs.instances:
        ranked = predictions.get(instance.word_id, [])
        expected = {(candidate.wordform_id, candidate.gramset) for candidate in instance.candidates}
        observed = {(int(row["wordform_id"]), row["gramset"]) for row in ranked}
        if len(ranked) != len(instance.candidates) or observed != expected:
            raise ValueError(f"Prediction candidates do not match dev word_id={instance.word_id}")
        if [int(row["rank"]) for row in ranked] != list(range(1, len(ranked) + 1)):
            raise ValueError(f"Prediction ranks are not contiguous for word_id={instance.word_id}")

        gold_key = (instance.gold_analysis.wordform_id, instance.gold_analysis.gramset)
        gold_row = next(row for row in ranked if (int(row["wordform_id"]), row["gramset"]) == gold_key)
        top_row = ranked[0]
        correct = gold_key == (int(top_row["wordform_id"]), top_row["gramset"])
        rank = int(gold_row["rank"])
        rank_counts[rank] += 1
        view = sentence_views[instance.word_id]
        row = {
            "language": instance.language,
            "text_id": instance.text_id,
            "sentence_id": instance.sentence_id,
            "word_id": instance.word_id,
            "word": instance.word,
            "candidate_count": len(instance.candidates),
            "gold_rank": rank,
            "correct": int(correct),
            "gold_wordform_id": gold_key[0],
            "gold_gramset": gold_key[1],
            "top1_wordform_id": int(top_row["wordform_id"]),
            "top1_gramset": top_row["gramset"],
            "top1_score": float(top_row["score"]),
            "gold_score": float(gold_row["score"]),
            "top1_minus_gold_score": float(top_row["score"]) - float(gold_row["score"]),
            "same_gramset_as_gold": int(top_row["gramset"] == gold_key[1]),
            **view,
        }
        instance_rows.append(row)
        if not correct:
            same_gramset_errors += int(top_row["gramset"] == gold_key[1])
            confusion_counts[(gold_key[1], top_row["gramset"])] += 1
            error_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    languages = ("vep", "krl", "olo", "lud")
    _write_csv(
        args.output_dir / "language-summary.csv",
        ("language", "instances", "errors", "top1_accuracy"),
        _count_summary(instance_rows, "language", list(languages)),
    )
    candidate_values = sorted({row["candidate_count"] for row in instance_rows})
    _write_csv(
        args.output_dir / "candidate-count-summary.csv",
        ("candidate_count", "instances", "errors", "top1_accuracy"),
        _count_summary(instance_rows, "candidate_count", candidate_values),
    )
    _write_csv(
        args.output_dir / "gold-rank-summary.csv",
        ("gold_rank", "instances", "share"),
        [
            {"gold_rank": rank, "instances": rank_counts[rank], "share": rank_counts[rank] / len(instance_rows)}
            for rank in sorted(rank_counts)
        ],
    )
    confusion_rows = [
        {"gold_gramset": gold, "top1_gramset": predicted, "errors": count}
        for (gold, predicted), count in confusion_counts.most_common()
    ]
    _write_csv(
        args.output_dir / "top-confusions.csv",
        ("gold_gramset", "top1_gramset", "errors"),
        confusion_rows,
    )
    _write_csv(
        args.output_dir / "error-instances.csv",
        (
            "language", "text_id", "sentence_id", "word_id", "word", "candidate_count",
            "gold_rank", "gold_wordform_id", "gold_gramset", "top1_wordform_id",
            "top1_gramset", "top1_score", "gold_score", "top1_minus_gold_score",
            "same_gramset_as_gold", "left_neighbor", "right_neighbor", "sentence",
            "marked_sentence", "correct",
        ),
        error_rows,
    )

    totals = {
        "data_tag": args.tag,
        "split": "dev",
        "model": "T1.1, suffix_max_length=5, context_positions=[-1, 1], affix=false, C=7",
        "prediction_file": str(args.predictions),
        "prediction_sha256": _sha256(args.predictions),
        "split_file": str(inputs.benchmark.split_path),
        "split_sha256": _sha256(inputs.benchmark.split_path),
        "instances": len(instance_rows),
        "errors": len(error_rows),
        "top1_correct": len(instance_rows) - len(error_rows),
        "top1_accuracy": (len(instance_rows) - len(error_rows)) / len(instance_rows),
        "errors_same_gramset_as_gold": same_gramset_errors,
        "errors_different_gramset_from_gold": len(error_rows) - same_gramset_errors,
        "test_used": False,
        "interpretation": "Descriptive analysis of the C=7 dev-selected prediction file; not an independent estimate.",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(totals, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(totals, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
