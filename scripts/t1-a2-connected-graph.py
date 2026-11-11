#!/usr/bin/env python3
"""Render checked connected-gramset and gold-vs-Top-1 visualization drafts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[1]
A1_DIR = ROOT / "work/experiments/v2026.11/t1-dev-error-analysis-a1-v5"
DEFAULT_OUTPUT = ROOT / "work/experiments/v2026.11/t1-error-analysis-a2-v12"
GRAMSETS = ("SG+GEN", "SG+ACC", "SG+NOM", "PL+NOM", "PL+ACC")
PAIRS = (
    ("SG+ACC", "SG+NOM"),
    ("SG+ACC", "SG+GEN"),
    ("SG+GEN", "SG+NOM"),
    ("SG+NOM", "PL+NOM"),
    ("PL+NOM", "PL+ACC"),
)
POSITIONS = {
    "SG+GEN": (-4.0, 2.1),
    "SG+ACC": (-4.0, -1.5),
    "SG+NOM": (0.0, 0.4),
    "PL+NOM": (4.1, 0.4),
    "PL+ACC": (4.1, -2.9),
}
GOLD_COLOR = "#B58A25"
MODEL_COLOR = "#476B8F"
RIBBON_COLOR = "#34495E"
MIN_NODE_RADIUS = 0.58
MAX_TO_MIN_NODE_AREA = 3.0
MAX_RIBBON_WIDTH = 4.2


class GraphInputError(ValueError):
    """An A1 input is missing, corrupted, or inconsistent."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GraphInputError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GraphInputError(f"expected a JSON object in {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream, strict=True))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise GraphInputError(f"cannot read {path}: {exc}") from exc


def checked_inputs(a1_dir: Path) -> tuple[dict, dict[str, str], list, list, list]:
    summary = read_json(a1_dir / "summary.json")
    if summary.get("target_split") != "dev":
        raise GraphInputError("connected graph requires the checked A1 dev output")
    prediction = summary.get("prediction_source", {})
    if not isinstance(prediction, dict) or prediction.get("sha256") != prediction.get("expected_sha256"):
        raise GraphInputError("A1 summary does not confirm the frozen prediction checksum")
    output_hashes = summary.get("output_sha256")
    if not isinstance(output_hashes, dict):
        raise GraphInputError("A1 summary has no output checksums")
    names = ("instances.csv", "candidates.csv", "conflict-summary.csv")
    hashes = {}
    tables = []
    for name in names:
        path = a1_dir / name
        expected = output_hashes.get(name)
        actual = sha256(path)
        if actual != expected:
            raise GraphInputError(f"A1 checksum mismatch for {name}: {actual} != {expected}")
        hashes[name] = actual
        tables.append(read_csv(path))
    return summary, hashes, *tables


def assemble_statistics(instances: list, candidates: list, conflict_rows: list) -> tuple[list[dict], list[dict], dict]:
    candidates_by_word: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in candidates:
        candidates_by_word[(row["language"], row["word_id"])].add(row["gramset"])
    seen = set()
    gold_counts = defaultdict(int)
    prediction_counts = defaultdict(int)
    overlap_counts = defaultdict(int)
    selected_errors = defaultdict(int)
    for row in instances:
        key = (row["language"], row["word_id"])
        if key in seen:
            raise GraphInputError(f"duplicate A1 instance: {key!r}")
        seen.add(key)
        gold = row["gold_gramset"]
        top1 = row["top1_gramset"]
        if gold in GRAMSETS:
            gold_counts[gold] += 1
        if top1 in GRAMSETS:
            prediction_counts[top1] += 1
        if gold == top1 and gold in GRAMSETS:
            overlap_counts[gold] += 1
        if gold in GRAMSETS and top1 in GRAMSETS and top1 != gold:
            if top1 not in candidates_by_word[key]:
                raise GraphInputError(f"predicted gramset absent from candidate set for {key!r}")
            selected_errors[(gold, top1)] += 1

    a1_conflicts = {}
    for row in conflict_rows:
        key = (row["gold_gramset"], row["alternative_gramset"])
        if key in a1_conflicts:
            raise GraphInputError(f"duplicate conflict pair: {key!r}")
        try:
            opportunities = int(row["opportunity_count"])
            errors = int(row["selected_error_count"])
            saved_rate = float(row["selected_error_rate"])
        except (KeyError, ValueError) as exc:
            raise GraphInputError(f"invalid conflict row: {row!r}") from exc
        if opportunities <= 0 or not 0 <= errors <= opportunities:
            raise GraphInputError(f"invalid conflict counts for {key!r}")
        if not math.isfinite(saved_rate) or not math.isclose(errors / opportunities, saved_rate, abs_tol=1e-12, rel_tol=0):
            raise GraphInputError(f"conflict rate disagrees with counts for {key!r}")
        a1_conflicts[key] = (opportunities, errors, saved_rate)

    node_rows = []
    for gramset in GRAMSETS:
        gold = gold_counts[gramset]
        predicted = prediction_counts[gramset]
        overlap = overlap_counts[gramset]
        if not gold or not predicted or overlap > min(gold, predicted):
            raise GraphInputError(f"invalid gold/prediction set totals for {gramset}")
        node_rows.append(
            {
                "gramset": gramset,
                "gold_instance_count": gold,
                "top1_prediction_count": predicted,
                "same_gramset_count": overlap,
                "gold_only_count": gold - overlap,
                "prediction_only_count": predicted - overlap,
            }
        )

    edge_rows = []
    for left, right in PAIRS:
        for source, target in ((left, right), (right, left)):
            key = (source, target)
            if key not in a1_conflicts:
                continue
            opportunities, errors, rate = a1_conflicts[key]
            direct = selected_errors[key]
            if direct != errors:
                raise GraphInputError(f"A1 conflict count disagrees with instance rows for {key!r}")
            edge_rows.append(
                {
                    "gold_gramset": source,
                    "alternative_gramset": target,
                    "selected_error_count": errors,
                    "opportunity_count": opportunities,
                    "selected_error_rate": rate,
                    "displayed": int(errors > 0),
                }
            )
    if not edge_rows:
        raise GraphInputError("no selected directed conflicts connect the requested nodes")

    undirected = defaultdict(set)
    for row in edge_rows:
        if row["selected_error_count"]:
            a, b = row["gold_gramset"], row["alternative_gramset"]
            undirected[a].add(b)
            undirected[b].add(a)
    reached = set()
    queue = deque([GRAMSETS[0]])
    while queue:
        node = queue.popleft()
        if node in reached:
            continue
        reached.add(node)
        queue.extend(undirected[node] - reached)
    if reached != set(GRAMSETS):
        raise GraphInputError(f"nonzero-error subgraph is disconnected: {sorted(reached)!r}")

    total_different_gramset_errors = int(summary_count(instances, "different_gramset"))
    covered = sum(row["selected_error_count"] for row in edge_rows)
    if covered > total_different_gramset_errors:
        raise GraphInputError("selected subgraph error count exceeds all gramset errors")
    metadata = {
        "instance_count": len(instances),
        "different_gramset_error_count": total_different_gramset_errors,
        "covered_error_count": covered,
        "covered_error_rate": covered / total_different_gramset_errors if total_different_gramset_errors else 0.0,
        "node_count": len(node_rows),
        "available_directed_pairs": len(edge_rows),
        "displayed_directed_edges": sum(row["displayed"] for row in edge_rows),
        "nonzero_error_graph_connected": True,
    }
    return node_rows, edge_rows, metadata


def summary_count(instances: list, category: str) -> int:
    if category == "different_gramset":
        return sum(row["gold_gramset"] != row["top1_gramset"] for row in instances)
    raise ValueError(category)


def node_area(count: int, minimum: int, maximum: int) -> float:
    """Map log-counts to a shared, bounded area range for all node rings."""
    minimum_area = math.pi * MIN_NODE_RADIUS**2
    maximum_area = minimum_area * MAX_TO_MIN_NODE_AREA
    if maximum <= minimum:
        return minimum_area
    value = (math.log1p(count) - math.log1p(minimum)) / (
        math.log1p(maximum) - math.log1p(minimum)
    )
    value = min(1.0, max(0.0, value))
    return minimum_area + (maximum_area - minimum_area) * value


def node_radius(count: int, minimum: int, maximum: int) -> float:
    return math.sqrt(node_area(count, minimum, maximum) / math.pi)


def format_percent(rate: float) -> str:
    percent = rate * 100
    decimals = 2 if 0 < percent < 1 else 1
    return f"{percent:.{decimals}f}".replace(".", ",") + "%"


def cubic_points(source: tuple[float, float], target: tuple[float, float], curve: float, source_radius: float, target_radius: float):
    sx, sy = source
    tx, ty = target
    dx, dy = tx - sx, ty - sy
    distance = math.hypot(dx, dy)
    ux, uy = dx / distance, dy / distance
    nx, ny = -uy, ux
    bend = curve * distance
    c1 = (sx + dx * 0.31 + nx * bend, sy + dy * 0.31 + ny * bend)
    c2 = (sx + dx * 0.69 + nx * bend, sy + dy * 0.69 + ny * bend)
    t1x, t1y = c1[0] - sx, c1[1] - sy
    t1n = math.hypot(t1x, t1y)
    t2x, t2y = tx - c2[0], ty - c2[1]
    t2n = math.hypot(t2x, t2y)
    start = (sx + t1x / t1n * (source_radius + 0.04), sy + t1y / t1n * (source_radius + 0.04))
    end = (tx - t2x / t2n * (target_radius + 0.04), ty - t2y / t2n * (target_radius + 0.04))
    return start, c1, c2, end


def bezier_value(points, t: float) -> tuple[float, float]:
    p0, p1, p2, p3 = points
    q = 1 - t
    return (
        q**3 * p0[0] + 3 * q**2 * t * p1[0] + 3 * q * t**2 * p2[0] + t**3 * p3[0],
        q**3 * p0[1] + 3 * q**2 * t * p1[1] + 3 * q * t**2 * p2[1] + t**3 * p3[1],
    )


def bezier_derivative(points, t: float) -> tuple[float, float]:
    p0, p1, p2, p3 = points
    q = 1 - t
    return (
        3 * q**2 * (p1[0] - p0[0]) + 6 * q * t * (p2[0] - p1[0]) + 3 * t**2 * (p3[0] - p2[0]),
        3 * q**2 * (p1[1] - p0[1]) + 6 * q * t * (p2[1] - p1[1]) + 3 * t**2 * (p3[1] - p2[1]),
    )


def draw_contour_edge(ax, source, target, r_source, r_target, row, max_count):
    count = row["selected_error_count"]
    if not count:
        return
    width = MAX_RIBBON_WIDTH * count / max_count
    # Reversing the endpoints mirrors a positive curve onto the opposite side.
    points = cubic_points(source, target, 0.22, r_source, r_target)
    codes = (MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4)
    path = MplPath(points, codes)
    ax.add_patch(PathPatch(path, fill=False, linewidth=2 * width, edgecolor=RIBBON_COLOR, capstyle="round", zorder=2))
    ax.add_patch(PathPatch(path, fill=False, linewidth=width, edgecolor="white", capstyle="round", zorder=2.1))
    arrow_start = bezier_value(points, 0.88)
    arrow_end = bezier_value(points, 0.985)
    ax.add_patch(
        FancyArrowPatch(
            arrow_start,
            arrow_end,
            arrowstyle="-|>",
            mutation_scale=20,
            linewidth=1.5,
            edgecolor=RIBBON_COLOR,
            facecolor="white",
            zorder=3,
        )
    )
    return points


def draw_network(node_rows: list[dict], edge_rows: list[dict], output: Path, *, dual: bool = False, partition_labels: bool = False) -> None:
    nodes = {row["gramset"]: row for row in node_rows}
    fig, ax = plt.subplots(figsize=(13.6, 9.2))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-6.8, 6.8)
    ax.set_ylim(-5.0, 4.8)

    all_counts = [
        count
        for row in node_rows
        for count in (row["gold_instance_count"], row["top1_prediction_count"])
    ]
    minimum_count, maximum_count = min(all_counts), max(all_counts)
    gold_radii = {
        g: node_radius(nodes[g]["gold_instance_count"], minimum_count, maximum_count)
        for g in GRAMSETS
    }
    model_radii = {
        g: node_radius(nodes[g]["top1_prediction_count"], minimum_count, maximum_count)
        for g in GRAMSETS
    }
    outer_radii = {
        g: (max(gold_radii[g], model_radii[g]) + 0.16) if dual else gold_radii[g]
        for g in GRAMSETS
    }
    label_positions = {
        ("SG+ACC", "SG+NOM"): (-1.65, -0.35),
        ("SG+NOM", "SG+ACC"): (-1.20, -0.85),
        ("SG+ACC", "SG+GEN"): (-5.55, 0.35),
        ("SG+GEN", "SG+ACC"): (-2.55, 0.35),
        ("SG+GEN", "SG+NOM"): (-1.55, 2.65),
        ("SG+NOM", "PL+NOM"): (2.05, 0.95),
        ("PL+NOM", "SG+NOM"): (2.05, -0.15),
        ("PL+NOM", "PL+ACC"): (4.95, -1.20),
        ("PL+ACC", "PL+NOM"): (3.25, -1.20),
    }
    max_edge_count = max(
        (row["selected_error_count"] for row in edge_rows if row["displayed"]),
        default=1,
    )
    for row in edge_rows:
        source, target = row["gold_gramset"], row["alternative_gramset"]
        if not row["displayed"]:
            continue
        points = draw_contour_edge(
            ax,
            POSITIONS[source],
            POSITIONS[target],
            outer_radii[source],
            outer_radii[target],
            row,
            max_edge_count,
        )
        label_x, label_y = label_positions[(source, target)]
        label = f"{row['selected_error_count']} / {row['opportunity_count']} · {format_percent(row['selected_error_rate'])}"
        ax.text(
            label_x,
            label_y,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.97, "pad": 2.0},
            zorder=4,
        )

    for gramset in GRAMSETS:
        center = POSITIONS[gramset]
        row = nodes[gramset]
        gold_radius = gold_radii[gramset]
        if dual:
            ax.add_patch(Circle(center, gold_radius, facecolor="white", edgecolor=GOLD_COLOR, linewidth=2.1, zorder=5))
            offset = (0.12, 0.10)
            model_center = (center[0] + offset[0], center[1] + offset[1])
            ax.add_patch(
                Circle(
                    model_center,
                    model_radii[gramset],
                    facecolor="none",
                    edgecolor=MODEL_COLOR,
                    linewidth=2.6,
                    linestyle=(0, (5, 2.5)),
                    zorder=5.2,
                )
            )
            if partition_labels:
                label_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.94, "pad": 0.5}
                ax.text(center[0], center[1] + 0.30, gramset, ha="center", va="center", fontsize=8.0, bbox=label_box, zorder=6)
                ax.text(center[0], center[1] + 0.12, f"Gold {row['gold_instance_count']} · Top-1 {row['top1_prediction_count']}", ha="center", va="center", fontsize=6.4, bbox=label_box, zorder=6)
                ax.text(center[0], center[1] - 0.06, f"Gold-only {row['gold_only_count']}", color=GOLD_COLOR, ha="center", va="center", fontsize=6.8, bbox=label_box, zorder=6)
                ax.text(center[0], center[1] - 0.23, f"Совпало {row['same_gramset_count']}", color="#444444", ha="center", va="center", fontsize=6.8, bbox=label_box, zorder=6)
                ax.text(center[0], center[1] - 0.40, f"Top-1-only {row['prediction_only_count']}", color=MODEL_COLOR, ha="center", va="center", fontsize=6.8, bbox=label_box, zorder=6)
            else:
                label = (
                    f"{gramset}\nG={row['gold_instance_count']} · P={row['top1_prediction_count']}\n"
                    f"same={row['same_gramset_count']}"
                )
                fontsize = 8.6
                ax.text(*center, label, ha="center", va="center", fontsize=fontsize, zorder=6)
        else:
            ax.add_patch(Circle(center, gold_radius, facecolor="white", edgecolor=GOLD_COLOR, linewidth=2.1, zorder=5))
            ax.text(
                *center,
                f"{gramset}\nGold N={row['gold_instance_count']}",
                ha="center",
                va="center",
                fontsize=9,
                zorder=6,
            )

    ax.set_title("Связный граф выбранных грамсетных ошибок на dev", fontsize=15, pad=16)
    if dual and partition_labels:
        footer = "Золотой контур — текущий gold; синий пунктир — Top-1. Gold-only / Совпало / Top-1-only — разложение по грамсету; смещение контуров схематично."
    elif dual:
        footer = "Золотой контур — текущий gold; синий пунктир — Top-1. Числа G/P/same точны; площади и перекрытие контуров схематичны."
    else:
        footer = "Размер вершины: площадь по логарифмической шкале числа gold-употреблений. Контурная лента: число ошибок; подпись: ошибки / возможности · доля."
    fig.text(0.5, 0.04, footer, ha="center", va="bottom", fontsize=9.5)
    fig.subplots_adjust(top=0.92, bottom=0.11)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def circle_intersection_area(r1: float, r2: float, distance: float) -> float:
    if distance >= r1 + r2:
        return 0.0
    if distance <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    x1 = max(-1.0, min(1.0, (distance**2 + r1**2 - r2**2) / (2 * distance * r1)))
    x2 = max(-1.0, min(1.0, (distance**2 + r2**2 - r1**2) / (2 * distance * r2)))
    radicand = max(0.0, (-distance + r1 + r2) * (distance + r1 - r2) * (distance - r1 + r2) * (distance + r1 + r2))
    return r1**2 * math.acos(x1) + r2**2 * math.acos(x2) - 0.5 * math.sqrt(radicand)


def exact_circle_distance(r1: float, r2: float, overlap_area: float) -> float:
    max_area = math.pi * min(r1, r2) ** 2
    if overlap_area <= 0:
        return r1 + r2
    if overlap_area >= max_area:
        return abs(r1 - r2)
    low, high = abs(r1 - r2), r1 + r2
    for _ in range(80):
        mid = (low + high) / 2
        if circle_intersection_area(r1, r2, mid) > overlap_area:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def draw_exact_venn(node_rows: list[dict], output: Path) -> None:
    area_per_instance = 0.0050
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.5))
    for index, row in enumerate(node_rows):
        ax = axes[index // 3, index % 3]
        g = row["gold_instance_count"]
        p = row["top1_prediction_count"]
        same = row["same_gramset_count"]
        r_g = math.sqrt(g * area_per_instance / math.pi)
        r_p = math.sqrt(p * area_per_instance / math.pi)
        overlap_area = same * area_per_instance
        distance = exact_circle_distance(r_g, r_p, overlap_area)
        ax.set_aspect("equal")
        ax.set_xlim(-2.65, 2.65)
        ax.set_ylim(-1.75, 1.75)
        ax.axis("off")
        ax.set_title(row["gramset"], fontsize=12, pad=8)
        ax.add_patch(Circle((-distance / 2, 0), r_g, facecolor=(0.85, 0.65, 0.15, 0.28), edgecolor=GOLD_COLOR, linewidth=2.0, zorder=2))
        ax.add_patch(Circle((distance / 2, 0), r_p, facecolor=(0.27, 0.48, 0.67, 0.22), edgecolor=MODEL_COLOR, linewidth=1.8, linestyle=(0, (4, 2)), zorder=2.1))
        ax.text(0, 1.35, f"G={g}  ·  P={p}  ·  G∩P={same}", ha="center", va="center", fontsize=9)
        ax.text(-1.85, -1.42, f"G-only {row['gold_only_count']}", ha="center", fontsize=8.2)
        ax.text(0, -1.42, f"same {same}", ha="center", fontsize=8.2)
        ax.text(1.85, -1.42, f"P-only {row['prediction_only_count']}", ha="center", fontsize=8.2)
    axes[1, 2].axis("off")
    fig.suptitle("Gold и Top-1 по отдельным грамсетам: точные пересечения", fontsize=15, y=0.98)
    fig.text(0.5, 0.025, "Площадь круга пропорциональна числу употреблений; площадь пересечения — числу совпадений по грамсету. Масштаб общий.", ha="center", fontsize=9.5)
    fig.subplots_adjust(top=0.91, bottom=0.10, hspace=0.28, wspace=0.08)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_exact_overlap_network(node_rows: list[dict], edge_rows: list[dict], output: Path) -> None:
    """Draw exact G/P set circles as the nodes of the connected error graph."""
    area_per_instance = 0.0023
    fig, ax = plt.subplots(figsize=(15.2, 10.0))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-7.0, 7.0)
    ax.set_ylim(-5.2, 5.2)

    nodes = {row["gramset"]: row for row in node_rows}
    circles = {}
    extents = {}
    for gramset, row in nodes.items():
        gold_radius = math.sqrt(row["gold_instance_count"] * area_per_instance / math.pi)
        model_radius = math.sqrt(row["top1_prediction_count"] * area_per_instance / math.pi)
        overlap_area = row["same_gramset_count"] * area_per_instance
        distance = exact_circle_distance(gold_radius, model_radius, overlap_area)
        circles[gramset] = (gold_radius, model_radius, distance)
        extents[gramset] = distance / 2 + max(gold_radius, model_radius)

    label_positions = {
        ("SG+ACC", "SG+NOM"): (-1.65, -0.35),
        ("SG+NOM", "SG+ACC"): (-1.20, -0.85),
        ("SG+ACC", "SG+GEN"): (-5.55, 0.35),
        ("SG+GEN", "SG+ACC"): (-2.55, 0.35),
        ("SG+GEN", "SG+NOM"): (-1.55, 2.65),
        ("SG+NOM", "PL+NOM"): (2.05, 0.95),
        ("PL+NOM", "SG+NOM"): (2.05, -0.15),
        ("PL+NOM", "PL+ACC"): (4.95, -1.20),
        ("PL+ACC", "PL+NOM"): (3.25, -1.20),
    }
    for row in edge_rows:
        if not row["displayed"]:
            continue
        source, target = row["gold_gramset"], row["alternative_gramset"]
        draw_contour_edge(
            ax,
            POSITIONS[source],
            POSITIONS[target],
            extents[source],
            extents[target],
            row,
            max(1, max(item["selected_error_count"] for item in edge_rows)),
        )
        label_x, label_y = label_positions[(source, target)]
        ax.text(
            label_x,
            label_y,
            f"{row['selected_error_count']} / {row['opportunity_count']} · {format_percent(row['selected_error_rate'])}",
            ha="center",
            va="center",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.97, "pad": 2.0},
            zorder=4,
        )

    for gramset, row in nodes.items():
        x, y = POSITIONS[gramset]
        gold_radius, model_radius, distance = circles[gramset]
        ax.add_patch(
            Circle(
                (x - distance / 2, y),
                gold_radius,
                facecolor=(0.85, 0.65, 0.15, 0.28),
                edgecolor=GOLD_COLOR,
                linewidth=2.0,
                zorder=4.8,
            )
        )
        ax.add_patch(
            Circle(
                (x + distance / 2, y),
                model_radius,
                facecolor=(0.27, 0.48, 0.67, 0.22),
                edgecolor=MODEL_COLOR,
                linewidth=1.8,
                linestyle=(0, (4, 2)),
                zorder=4.9,
            )
        )
        ax.text(
            x,
            y,
            f"{gramset}\nG={row['gold_instance_count']} · P={row['top1_prediction_count']}\nG∩P={row['same_gramset_count']}",
            ha="center",
            va="center",
            fontsize=8.2,
            path_effects=[path_effects.withStroke(linewidth=3, foreground="white", alpha=0.9)],
            zorder=6,
        )

    ax.set_title("Связный граф с точными пересечениями gold и Top-1", fontsize=15, pad=16)
    fig.text(
        0.5,
        0.035,
        "Узел — пара множеств по грамсету: золотой сплошной круг и Top-1 пунктир. Площади кругов и их пересечения кодируют точные количества; масштаб общий.",
        ha="center",
        va="bottom",
        fontsize=9.2,
    )
    fig.subplots_adjust(top=0.92, bottom=0.11)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_partition_matrix(node_rows: list[dict], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 6.5))
    categories = (
        ("gold_only_count", "Gold-only", "#D6B34A"),
        ("same_gramset_count", "Совпало", "#708A6A"),
        ("prediction_only_count", "Top-1-only", "#6F8FAE"),
    )
    y_positions = list(range(len(node_rows)))
    bar_height = 0.19
    offsets = (-bar_height, 0, bar_height)
    max_count = max(row[key] for row in node_rows for key, _, _ in categories)
    for offset, (key, label, color) in zip(offsets, categories):
        values = [row[key] for row in node_rows]
        bars = ax.barh([y + offset for y in y_positions], values, height=bar_height * 0.86, color=color, label=label)
        for bar, value in zip(bars, values):
            ax.text(value + max_count * 0.018, bar.get_y() + bar.get_height() / 2, str(value), va="center", fontsize=8.5)
    ax.set_yticks(y_positions, [row["gramset"] for row in node_rows])
    ax.invert_yaxis()
    ax.set_xlim(0, max_count * 1.22)
    ax.set_xlabel("Число употреблений (общая линейная шкала)")
    ax.set_title("Разложение gold/Top-1 по каждому грамсету", fontsize=14, pad=12)
    ax.xaxis.grid(True, color="#D9DDE2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, loc="lower right", frameon=False)
    fig.text(0.5, 0.015, "Gold-only = gold-грамсет не выбран Top-1; совпало = совпадение грамсета; Top-1-only = модель выбрала грамсет при другом gold.", ha="center", fontsize=9)
    fig.subplots_adjust(left=0.18, right=0.97, top=0.88, bottom=0.16)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(a1_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise GraphInputError(f"refusing to overwrite existing output: {output_dir}")
    summary, input_hashes, instances, candidates, conflict_rows = checked_inputs(a1_dir)
    nodes, edges, stats = assemble_statistics(instances, candidates, conflict_rows)
    output_dir.mkdir(parents=True)
    counts = [
        count
        for row in nodes
        for count in (row["gold_instance_count"], row["top1_prediction_count"])
    ]
    minimum_count, maximum_count = min(counts), max(counts)
    draw_network(
        nodes,
        edges,
        output_dir / "connected-graph-partitions",
        dual=True,
        partition_labels=True,
    )

    write_csv(
        output_dir / "node-statistics.csv",
        [
            {
                **row,
                "gold_area": node_area(row["gold_instance_count"], minimum_count, maximum_count),
                "gold_radius": node_radius(row["gold_instance_count"], minimum_count, maximum_count),
                "top1_area": node_area(row["top1_prediction_count"], minimum_count, maximum_count),
                "top1_radius": node_radius(row["top1_prediction_count"], minimum_count, maximum_count),
            }
            for row in nodes
        ],
        ("gramset", "gold_instance_count", "top1_prediction_count", "same_gramset_count", "gold_only_count", "prediction_only_count", "gold_area", "gold_radius", "top1_area", "top1_radius"),
    )
    write_csv(
        output_dir / "edge-statistics.csv",
        edges,
        ("gold_gramset", "alternative_gramset", "selected_error_count", "opportunity_count", "selected_error_rate", "displayed"),
    )
    caption = f"""# Подпись к графу: connected-graph-partitions

Направленные ошибки выбора грамсета на dev. Показан связный подграф из пяти грамсетов и девяти ненулевых направлений, охватывающий {stats['covered_error_count']} из {stats['different_gramset_error_count']} ошибок с несовпадением грамсета ({format_percent(stats['covered_error_rate'])}). Золотой сплошной контур каждого узла обозначает множество употреблений, которым данный грамсет назначен в текущей разметке benchmark; синий пунктирный контур — множество употреблений, которым его назначила модель Top-1. Gold-only показывает число употреблений этого gold-грамсета, отнесённых моделью к другим грамсетам; «Совпало» — число совпадений по грамсету; Top-1-only — число употреблений, ошибочно отнесённых моделью к данному грамсету при другом gold.

Размер золотых и синих окружностей рассчитан по одной общей логарифмически преобразованной шкале: максимальная площадь в три раза больше минимальной. Точные количества приведены числами. Взаимное смещение и перекрытие окружностей схематичны и не кодируют величину совпадения. Ленты направлены от gold-грамсета к ошибочно выбранному грамсету Top-1; их ширина отражает число ошибок. Подпись ленты содержит ошибки, число возможностей и долю.

В метках грамсетов `SG` означает единственное число, `PL` — множественное; `NOM`, `GEN` и `ACC` — соответственно именительный, родительный и аккузативный (винительный) падежи. Используются исходные метки корпусной разметки.

Совпадение оценивается по грамсету, а не по полному кандидату `(wordform_id, gramset)`: при другом `wordform_id` полный выбор остаётся ошибочным. Числа внутри узлов рассчитаны по всему рассматриваемому dev-benchmark, тогда как ленты показывают только выбранный подграф; поэтому суммы видимых входящих и исходящих ошибок могут не совпадать с Top-1-only и Gold-only.

Gold означает сохранённую экспертную метку benchmark. Возможный пересмотр этой метки фиксируется отдельно и не изменяет исходные результаты анализа.
"""
    (output_dir / "caption-draft.md").write_text(caption, encoding="utf-8")
    output_hashes = {
        path.name: sha256(path)
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "summary.json"
    }
    summary_out = {
        "study": "A2 connected gramset graph partitions",
        "source_a1_dir": str(a1_dir),
        "source_a1_table_sha256": input_hashes,
        "source_prediction_sha256": summary["prediction_source"]["sha256"],
        "source_split": "dev",
        "node_area_formula": "A = A_min + (3*A_min - A_min) * normalized_log1p(count); shared count range across Gold and Top-1, A_min=pi*0.58^2",
        "edge_visual_encoding": "contour ribbon with white center; outer width is twice the error-count-scaled ribbon width; separate thin outlined arrow head",
        "gold_prediction_sets": "G_g={dev usage: gold gramset g}; P_g={dev usage: frozen top1 gramset g}; overlap counts gramset agreement only",
        "node_area_ratio_max_to_min": MAX_TO_MIN_NODE_AREA,
        "rendered_formats": ["png"],
        "statistics": stats,
        "nodes": nodes,
        "edges": edges,
        "test_used": False,
        "model_retrained": False,
        "new_predictions_generated": False,
        "output_sha256": output_hashes,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_out, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary_out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a1-dir", type=Path, default=A1_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        result = run(args.a1_dir.resolve(), args.output_dir.resolve())
    except (GraphInputError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
