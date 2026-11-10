#!/usr/bin/env python3
"""Render the first A2 diagnostic charts from the checked A1 tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work/.matplotlib-t1-a2"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_INPUT = ROOT / "work/experiments/v2026.11/t1-dev-error-analysis-a1-v5"
DEFAULT_OUTPUT = ROOT / "work/experiments/v2026.11/t1-error-analysis-a2-v3"

DIRECTION_PAIRS = (
    ("SG+NOM", "SG+ACC"),
    ("PL+NOM", "PL+ACC"),
    ("SG+ACC", "SG+GEN"),
    ("SG+ELA", "SG+IN"),
    ("INF", "PRS+PASS"),
    ("PL+3+IND", "SG+3+IND"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def make_direction_chart(rows: list[dict[str, str]], output_dir: Path) -> list[dict]:
    by_pair = {
        (row["gold_gramset"], row["alternative_gramset"]): row
        for row in rows
    }
    plotted = []
    for left, right in DIRECTION_PAIRS:
        for pair in ((left, right), (right, left)):
            if pair not in by_pair:
                raise ValueError(f"A1 conflict table is missing direction {pair[0]} -> {pair[1]}")
            source = by_pair[pair]
            plotted.append(
                {
                    "gold_gramset": pair[0],
                    "alternative_gramset": pair[1],
                    "selected_error_count": int(source["selected_error_count"]),
                    "opportunity_count": int(source["opportunity_count"]),
                    "selected_error_rate": float(source["selected_error_rate"]),
                }
            )

    labels = [f"{row['gold_gramset']} → {row['alternative_gramset']}" for row in plotted]
    rates = [row["selected_error_rate"] * 100 for row in plotted]
    counts = [
        f"{row['selected_error_count']} / {row['opportunity_count']}  ({rate:.2f}%)"
        for row, rate in zip(plotted, rates)
    ]
    colors = [
        "#4C78A8" if index % 2 == 0 else "#F58518"
        for index in range(len(plotted))
    ]
    height = 7.0
    fig, ax = plt.subplots(figsize=(12.4, height), constrained_layout=True)
    positions = list(range(len(plotted)))
    ax.barh(positions, rates, color=colors, height=0.68)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xlabel("Выбор неверной альтернативы H, % случаев с доступным H")
    ax.set_title("Направленные конфликты морфологических кандидатов на dev")
    ax.xaxis.grid(True, color="#D9DDE2", linewidth=0.8)
    ax.set_axisbelow(True)
    for y, rate, count in zip(positions, rates, counts):
        ax.text(max(rate + 1.0, 1.2), y, count, va="center", fontsize=9)
    ax.text(
        0.0,
        -0.09,
        "Подпись: число решений H / число употреблений, где H был доступен. "
        "Обратные направления имеют разные наборы употреблений; график описательный.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#444444",
    )
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"directional-conflicts.{suffix}", dpi=180)
    plt.close(fig)
    return plotted


def make_sentence_chart(rows: list[dict[str, str]], output_dir: Path) -> list[dict]:
    sentences_by_bucket = {"0": 0, "1": 0, "2+": 0}
    errors_by_bucket = {"0": 0, "1": 0, "2+": 0}
    for row in rows:
        error_count = int(row["error_count"])
        bucket = "0" if error_count == 0 else "1" if error_count == 1 else "2+"
        sentences_by_bucket[bucket] += 1
        errors_by_bucket[bucket] += error_count
    buckets = ("0", "1", "2+")
    summary = [
        {
            "errors_per_sentence": bucket,
            "sentence_count": sentences_by_bucket[bucket],
            "error_instance_count": errors_by_bucket[bucket],
        }
        for bucket in buckets
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    sentence_values = [sentences_by_bucket[bucket] for bucket in buckets]
    error_values = [errors_by_bucket[bucket] for bucket in buckets]
    colors = ("#9AA3AD", "#4C78A8", "#E45756")
    for ax, values, title, ylabel in (
        (axes[0], sentence_values, "Предложения", "Число предложений"),
        (axes[1], error_values, "Ошибочные употребления", "Число ошибок Top-1"),
    ):
        bars = ax.bar(buckets, values, color=colors, width=0.62)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True, color="#D9DDE2", linewidth=0.8)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.018, str(value), ha="center", va="bottom")
        ax.set_xlabel("Ошибок в предложении")
    total_errors = sum(error_values)
    multi_error_sentences = sentences_by_bucket["2+"]
    multi_error_instances = errors_by_bucket["2+"]
    fig.suptitle(
        f"Ошибки сосредоточены в {multi_error_sentences} предложениях с 2+ ошибками: "
        f"{multi_error_instances}/{total_errors} = {multi_error_instances / total_errors:.1%} всех ошибок",
        fontsize=12,
    )
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"sentence-error-concentration.{suffix}", dpi=180)
    plt.close(fig)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        parser.error(f"refusing to overwrite existing A2 output: {output_dir}")
    if not (input_dir / "summary.json").is_file():
        parser.error(f"A1 summary is missing from {input_dir}")
    a1_summary = json.loads((input_dir / "summary.json").read_text(encoding="utf-8"))
    if a1_summary.get("prediction_source", {}).get("sha256") != a1_summary.get("prediction_source", {}).get("expected_sha256"):
        parser.error("A1 summary does not confirm the approved frozen prediction checksum")

    conflict_path = input_dir / "conflict-summary.csv"
    sentence_path = input_dir / "sentence-summary.csv"
    output_dir.mkdir(parents=True, exist_ok=False)
    direction_rows = make_direction_chart(read_csv(conflict_path), output_dir)
    sentence_rows = make_sentence_chart(read_csv(sentence_path), output_dir)
    write_csv(
        output_dir / "directional-conflicts.csv",
        ("gold_gramset", "alternative_gramset", "selected_error_count", "opportunity_count", "selected_error_rate"),
        direction_rows,
    )
    write_csv(
        output_dir / "sentence-error-distribution.csv",
        ("errors_per_sentence", "sentence_count", "error_instance_count"),
        sentence_rows,
    )
    summary = {
        "source_a1_output": str(input_dir),
        "source_prediction_sha256": a1_summary["prediction_source"]["sha256"],
        "source_conflict_summary_sha256": sha256(conflict_path),
        "source_sentence_summary_sha256": sha256(sentence_path),
        "visualizations": [
            "directional-conflicts.png",
            "directional-conflicts.svg",
            "sentence-error-concentration.png",
            "sentence-error-concentration.svg",
        ],
        "directional_conflicts": direction_rows,
        "sentence_error_distribution": sentence_rows,
        "test_analyzed": False,
        "model_retrained": False,
        "new_predictions_generated": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
