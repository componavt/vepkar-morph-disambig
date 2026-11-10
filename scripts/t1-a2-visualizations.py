#!/usr/bin/env python3
"""Validate checked A1 tables and render the A2 diagnostic figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work/.matplotlib-t1-a2"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch

DEFAULT_INPUT = ROOT / "work/experiments/v2026.11/t1-dev-error-analysis-a1-v5"
DEFAULT_OUTPUT = ROOT / "work/experiments/v2026.11/t1-error-analysis-a2-v5"

# Pair order follows the requested examples: ACC is the left-hand gold node.
DIRECTION_PAIRS = (
    ("SG+ACC", "SG+NOM", "Единственное число: ACC ↔ NOM"),
    ("PL+ACC", "PL+NOM", "Множественное число: ACC ↔ NOM"),
    ("SG+ACC", "SG+GEN", "Единственное число: ACC ↔ GEN"),
)

MAX_LINEWIDTH = 14.0
COUNT_COLOR = "#3B3B3B"
GOLD_EDGE = "#B58A25"


class A2InputError(ValueError):
    """A1 inputs are missing, corrupt, or internally inconsistent."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise A2InputError(f"cannot read A1 summary {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise A2InputError(f"A1 summary must be a JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream, strict=True))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise A2InputError(f"cannot read A1 table {path}: {exc}") from exc


def verify_table_hash(path: Path, expected_hash: str | None) -> str:
    """Verify an A1 table against the checksum recorded by A1."""
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise A2InputError(f"A1 summary has no valid SHA-256 for {path.name}")
    actual = sha256(path)
    if actual != expected_hash:
        raise A2InputError(
            f"A1 table checksum mismatch for {path.name}: "
            f"got {actual}, expected {expected_hash}"
        )
    return actual


def load_verified_a1_tables(input_dir: Path) -> tuple[dict, list[dict[str, str]], list[dict[str, str]], dict[str, str]]:
    """Read only A1 summary tables whose recorded hashes match their bytes."""
    summary_path = input_dir / "summary.json"
    summary = _read_json(summary_path)
    prediction_source = summary.get("prediction_source", {})
    if not isinstance(prediction_source, dict) or prediction_source.get("sha256") != prediction_source.get("expected_sha256"):
        raise A2InputError("A1 summary does not confirm the approved frozen predictions hash")
    if summary.get("target_split") != "dev":
        raise A2InputError("A2 input must be the checked A1 dev output")
    output_hashes = summary.get("output_sha256")
    if not isinstance(output_hashes, dict):
        raise A2InputError("A1 summary has no output_sha256 table checksums")

    conflict_path = input_dir / "conflict-summary.csv"
    sentence_path = input_dir / "sentence-summary.csv"
    source_hashes = {
        conflict_path.name: verify_table_hash(conflict_path, output_hashes.get(conflict_path.name)),
        sentence_path.name: verify_table_hash(sentence_path, output_hashes.get(sentence_path.name)),
    }
    return (
        summary,
        _read_csv(conflict_path),
        _read_csv(sentence_path),
        source_hashes,
    )


def _parse_nonnegative_int(row: dict[str, str], field: str, table: str) -> int:
    try:
        value = int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise A2InputError(f"invalid integer field {field!r} in {table}: {row!r}") from exc
    if value < 0:
        raise A2InputError(f"negative value for {field!r} in {table}: {value}")
    return value


def validate_conflict_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict]:
    """Reject duplicate, malformed, or arithmetically inconsistent G→H rows."""
    by_pair = {}
    for row in rows:
        try:
            pair = (row["gold_gramset"], row["alternative_gramset"])
            opportunities = _parse_nonnegative_int(row, "opportunity_count", "conflict-summary.csv")
            errors = _parse_nonnegative_int(row, "selected_error_count", "conflict-summary.csv")
            saved_rate = float(row["selected_error_rate"])
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, A2InputError):
                raise
            raise A2InputError(f"malformed conflict-summary.csv row: {row!r}") from exc
        if pair in by_pair:
            raise A2InputError(f"duplicate G→H pair in conflict-summary.csv: {pair!r}")
        if not pair[0] or not pair[1] or pair[0] == pair[1]:
            raise A2InputError(f"invalid G→H pair in conflict-summary.csv: {pair!r}")
        if opportunities <= 0:
            raise A2InputError(f"opportunity_count must be positive for {pair!r}")
        if errors > opportunities:
            raise A2InputError(f"selected_error_count exceeds opportunities for {pair!r}")
        expected_rate = errors / opportunities
        if not math.isfinite(saved_rate) or not math.isclose(
            saved_rate, expected_rate, rel_tol=0.0, abs_tol=1e-12
        ):
            raise A2InputError(f"selected_error_rate disagrees with counts for {pair!r}")
        by_pair[pair] = {
            "gold_gramset": pair[0],
            "alternative_gramset": pair[1],
            "selected_error_count": errors,
            "opportunity_count": opportunities,
            # Recompute the rate from the checked integer fields for plotting.
            "selected_error_rate": expected_rate,
        }
    if not by_pair:
        raise A2InputError("conflict-summary.csv is empty")
    return by_pair


def prepare_direction_rows(rows: list[dict[str, str]]) -> list[dict]:
    """Return the six checked directions represented by the three panels."""
    by_pair = validate_conflict_rows(rows)
    plotted = []
    for left, right, title in DIRECTION_PAIRS:
        for pair in ((left, right), (right, left)):
            if pair not in by_pair:
                raise A2InputError(f"A1 conflict table is missing direction {pair[0]} → {pair[1]}")
            row = dict(by_pair[pair])
            row["panel_title"] = title
            row["panel_order"] = len(plotted) // 2
            row["direction_order"] = len(plotted) % 2
            plotted.append(row)
    return plotted


def prepare_sentence_distribution(rows: list[dict[str, str]]) -> list[dict]:
    """Validate sentence rows and summarize the 0/1/2+ error groups."""
    if not rows:
        raise A2InputError("sentence-summary.csv is empty")
    seen_keys = set()
    sentences_by_bucket = {"0": 0, "1": 0, "2+": 0}
    errors_by_bucket = {"0": 0, "1": 0, "2+": 0}
    for row in rows:
        try:
            key = (row["language"], int(row["text_id"]), int(row["sentence_id"]))
            instance_count = _parse_nonnegative_int(row, "benchmark_instance_count", "sentence-summary.csv")
            error_count = _parse_nonnegative_int(row, "error_count", "sentence-summary.csv")
            saved_rate = float(row["error_rate"])
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, A2InputError):
                raise
            raise A2InputError(f"malformed sentence-summary.csv row: {row!r}") from exc
        if key in seen_keys:
            raise A2InputError(f"duplicate sentence key in sentence-summary.csv: {key!r}")
        seen_keys.add(key)
        if instance_count <= 0 or error_count > instance_count:
            raise A2InputError(f"invalid sentence counts for {key!r}")
        expected_rate = error_count / instance_count
        if not math.isfinite(saved_rate) or not math.isclose(
            saved_rate, expected_rate, rel_tol=0.0, abs_tol=1e-12
        ):
            raise A2InputError(f"sentence error_rate disagrees with counts for {key!r}")
        bucket = "0" if error_count == 0 else "1" if error_count == 1 else "2+"
        sentences_by_bucket[bucket] += 1
        errors_by_bucket[bucket] += error_count
    return [
        {
            "errors_per_sentence": bucket,
            "sentence_count": sentences_by_bucket[bucket],
            "error_instance_count": errors_by_bucket[bucket],
        }
        for bucket in ("0", "1", "2+")
    ]


def validate_a1_count_consistency(
    summary: dict,
    conflict_rows: list[dict[str, str]],
    sentence_rows: list[dict[str, str]],
    distribution: list[dict],
) -> None:
    """Cross-check the rows read by A2 against A1's recorded control totals."""
    counts = summary.get("counts")
    if not isinstance(counts, dict):
        raise A2InputError("A1 summary has no control counts")
    pairs = validate_conflict_rows(conflict_rows)
    total_opportunities = sum(row["opportunity_count"] for row in pairs.values())
    total_sentence_errors = sum(row["error_instance_count"] for row in distribution)
    observed = {
        "distinct_conflicts": len(pairs),
        "conflict_opportunities": total_opportunities,
        "sentences": len(sentence_rows),
        "errors": total_sentence_errors,
    }
    for field, actual in observed.items():
        if counts.get(field) != actual:
            raise A2InputError(
                f"A2 input table {field} total {actual} disagrees with A1 summary {counts.get(field)!r}"
            )


def _percent(value: float) -> str:
    decimals = 2 if 0 < value < 1 else 1
    return f"{value * 100:.{decimals}f}".replace(".", ",") + "%"


def _add_direction_arrow(ax, start: tuple[float, float], end: tuple[float, float], *, rad: float, linewidth: float) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>",
        mutation_scale=17,
        linewidth=linewidth,
        color=COUNT_COLOR,
        shrinkA=47,
        shrinkB=47,
        zorder=2,
    )
    ax.add_patch(arrow)


def draw_conflict_graph(rows: list[dict], output_dir: Path, *, width_metric: str) -> dict[str, str]:
    if width_metric not in {"count", "rate"}:
        raise ValueError("width_metric must be 'count' or 'rate'")
    values = [
        row["selected_error_count"] if width_metric == "count" else row["selected_error_rate"]
        for row in rows
    ]
    maximum = max(values, default=0)
    if maximum <= 0:
        raise A2InputError(f"cannot scale {width_metric} arrows: all displayed values are zero")

    fig, axes = plt.subplots(3, 1, figsize=(11.8, 9.4), squeeze=False)
    for panel_index, (left_gramset, right_gramset, title) in enumerate(DIRECTION_PAIRS):
        ax = axes[panel_index, 0]
        ax.set(xlim=(-3.35, 3.35), ylim=(-1.42, 1.42), aspect="equal")
        ax.axis("off")
        ax.set_title(title, fontsize=12, pad=5)
        forward = rows[2 * panel_index]
        reverse = rows[2 * panel_index + 1]
        left_node = (-2.35, 0.0)
        right_node = (2.35, 0.0)

        for row, start, end, rad, label_y in (
            # With arc3, the same positive radius places the forward path
            # below the nodes and the reverse path above them.
            (forward, left_node, right_node, 0.34, -0.90),
            (reverse, right_node, left_node, 0.34, 0.90),
        ):
            value = row["selected_error_count"] if width_metric == "count" else row["selected_error_rate"]
            if value > 0:
                # Linear mapping preserves exact ratios within each figure.
                linewidth = MAX_LINEWIDTH * value / maximum
                _add_direction_arrow(ax, start, end, rad=rad, linewidth=linewidth)
            label = (
                f"{row['selected_error_count']} / {row['opportunity_count']}"
                f" · {_percent(row['selected_error_rate'])}"
            )
            ax.text(
                0.0,
                label_y,
                label,
                ha="center",
                va="center",
                fontsize=10.5,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.8},
                zorder=4,
            )

        for position, gramset in ((left_node, left_gramset), (right_node, right_gramset)):
            ax.add_patch(
                Circle(
                    position,
                    radius=0.58,
                    facecolor="white",
                    edgecolor=GOLD_EDGE,
                    linewidth=2.2,
                    zorder=3,
                )
            )
            ax.text(*position, gramset, ha="center", va="center", fontsize=11.5, zorder=4)

    metric_label = "числу ошибок" if width_metric == "count" else "доле ошибок среди возможностей"
    fig.suptitle(
        "Направленные ошибки выбора грамсета на dev",
        fontsize=16,
        y=0.985,
    )
    if width_metric == "count":
        legend_values = (10, 20, 40)
        legend_labels = tuple(f"{value} ошибок" for value in legend_values)
        legend_widths = tuple(MAX_LINEWIDTH * value / maximum for value in legend_values)
    else:
        legend_values = (0.10, 0.40, 0.80)
        legend_labels = tuple(_percent(value) for value in legend_values)
        legend_widths = tuple(MAX_LINEWIDTH * value / maximum for value in legend_values)
    handles = [
        Line2D([0], [0], color=COUNT_COLOR, linewidth=width, label=label)
        for width, label in zip(legend_widths, legend_labels)
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.068), ncol=3, frameon=False, title=f"Толщина пропорциональна {metric_label}")
    fig.text(
        0.5,
        0.018,
        "Стрелка идёт от gold-грамсета к неверному Top-1. "
        "Подпись: ошибки / употребления, где альтернатива доступна · доля. "
        "Обратные направления имеют разные наборы употреблений; график описательный.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.subplots_adjust(top=0.94, bottom=0.16, hspace=0.18)

    stem = "directed-error-counts" if width_metric == "count" else "directed-error-rates"
    outputs = {}
    for extension in ("png", "svg", "pdf"):
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        outputs[path.name] = sha256(path)
    plt.close(fig)
    return outputs


def draw_sentence_distribution(distribution: list[dict], output_dir: Path) -> dict[str, str]:
    buckets = [row["errors_per_sentence"] for row in distribution]
    sentence_values = [row["sentence_count"] for row in distribution]
    error_values = [row["error_instance_count"] for row in distribution]
    total_sentences = sum(sentence_values)
    total_errors = sum(error_values)
    multi_sentence_count = sentence_values[2]
    multi_error_count = error_values[2]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2))
    colors = ("#9AA3AD", "#4C78A8", "#E45756")
    for ax, values, title, ylabel in (
        (axes[0], sentence_values, "Предложения", "Число предложений"),
        (axes[1], error_values, "Ошибочные употребления", "Число ошибок Top-1"),
    ):
        bars = ax.bar(buckets, values, color=colors, width=0.62)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Ошибок в предложении")
        ax.yaxis.grid(True, color="#D9DDE2", linewidth=0.8)
        ax.set_axisbelow(True)
        offset = max(values) * 0.018 if max(values, default=0) else 0.1
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                str(value),
                ha="center",
                va="bottom",
            )

    fig.suptitle("Распределение ошибок Top-1 по предложениям dev", fontsize=13)
    fig.text(
        0.5,
        0.045,
        f"Только benchmark-употребления dev. В группе 2+: {multi_sentence_count} из {total_sentences} предложений; "
        f"{multi_error_count} из {total_errors} ошибок.",
        ha="center",
        va="bottom",
        fontsize=9.5,
    )
    fig.subplots_adjust(top=0.84, bottom=0.22, wspace=0.18)
    outputs = {}
    for extension in ("png", "svg", "pdf"):
        path = output_dir / f"sentence-error-distribution.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        outputs[path.name] = sha256(path)
    plt.close(fig)
    return outputs


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = (
        "gold_gramset",
        "alternative_gramset",
        "selected_error_count",
        "opportunity_count",
        "selected_error_rate",
        "panel_title",
        "panel_order",
        "direction_order",
        "absolute_linewidth_pt",
        "rate_linewidth_pt",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() or output_dir.is_symlink():
        parser.error(f"refusing to overwrite existing A2 output: {output_dir}")

    try:
        a1_summary, raw_conflicts, raw_sentences, input_hashes = load_verified_a1_tables(input_dir)
        directions = prepare_direction_rows(raw_conflicts)
        sentence_distribution = prepare_sentence_distribution(raw_sentences)
        validate_a1_count_consistency(
            a1_summary, raw_conflicts, raw_sentences, sentence_distribution
        )
    except (A2InputError, OSError) as exc:
        parser.error(str(exc))

    max_count = max(row["selected_error_count"] for row in directions)
    max_rate = max(row["selected_error_rate"] for row in directions)
    for row in directions:
        row["absolute_linewidth_pt"] = MAX_LINEWIDTH * row["selected_error_count"] / max_count if max_count else 0.0
        row["rate_linewidth_pt"] = MAX_LINEWIDTH * row["selected_error_rate"] / max_rate if max_rate else 0.0

    output_dir.mkdir(parents=True, exist_ok=False)
    output_hashes = {}
    output_hashes.update(draw_conflict_graph(directions, output_dir, width_metric="count"))
    output_hashes.update(draw_conflict_graph(directions, output_dir, width_metric="rate"))
    output_hashes.update(draw_sentence_distribution(sentence_distribution, output_dir))
    direction_csv = output_dir / "directional-conflicts.csv"
    sentence_csv = output_dir / "sentence-error-distribution.csv"
    write_csv(direction_csv, directions)
    with sentence_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("errors_per_sentence", "sentence_count", "error_instance_count"),
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(sentence_distribution)
    output_hashes[direction_csv.name] = sha256(direction_csv)
    output_hashes[sentence_csv.name] = sha256(sentence_csv)

    total_sentences = sum(row["sentence_count"] for row in sentence_distribution)
    total_errors = sum(row["error_instance_count"] for row in sentence_distribution)
    summary = {
        "source_a1_output": str(input_dir),
        "source_prediction_sha256": a1_summary["prediction_source"]["sha256"],
        "source_table_sha256": input_hashes,
        "width_scales": {
            "maximum_linewidth_pt": MAX_LINEWIDTH,
            "absolute_errors": f"{MAX_LINEWIDTH} pt * selected_error_count / max_selected_error_count",
            "error_rate": f"{MAX_LINEWIDTH} pt * (selected_error_count / opportunity_count) / max_rate",
        },
        "displayed_directions": directions,
        "sentence_totals": {"sentences": total_sentences, "errors": total_errors},
        "sentence_distribution": sentence_distribution,
        "test_analyzed": False,
        "model_retrained": False,
        "new_predictions_generated": False,
        "output_sha256": output_hashes,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
