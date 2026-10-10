"""Shared build-instances and make-splits report helpers for CLI command handlers."""

from __future__ import annotations

from pathlib import Path

from core.data import SUPPORTED_LANGUAGES
from core.instances import LanguageInstances
from core.splits import TextWeight, split_overlap_counts


def _language_counts(
    result: LanguageInstances,
) -> tuple[int, int, int]:
    """Return (primary instances, distinct texts, candidate rows)."""
    primary = len(result.instances)
    text_ids = sorted({int(inst.text_id) for inst in result.instances})
    candidate_rows = sum(len(inst.candidates) for inst in result.instances)
    return primary, len(text_ids), candidate_rows


def _print_build_summary(tag: str, results: dict[str, LanguageInstances], path: Path, written: int) -> None:
    print(f"BENCHMARK — dictorpus-data {tag}")
    print()
    print(
        f"{'Lang':<6}{'Before strict':>14}{'Empty-candidate loss':>21}"
        f"{'Primary instances':>18}{'Texts':>7}{'Candidate rows':>15}"
    )
    totals = {"before": 0, "loss": 0, "primary": 0, "texts": 0, "rows": 0}
    for lang in SUPPORTED_LANGUAGES:
        result = results[lang]
        before = result.funnel[6].retained
        loss = result.funnel[7].removed
        primary, texts, rows = _language_counts(result)
        totals["before"] += before
        totals["loss"] += loss
        totals["primary"] += primary
        totals["texts"] += texts
        totals["rows"] += rows
        print(
            f"{lang:<6}{before:>14}{loss:>21}{primary:>18}{texts:>7}{rows:>15}"
        )
    print(
        f"{'ALL':<6}{totals['before']:>14}{totals['loss']:>21}"
        f"{totals['primary']:>18}{totals['texts']:>7}{totals['rows']:>15}"
    )
    print()
    print(
        "Before strict = eligible words before the final gramset check; "
        "Empty-candidate loss = words removed because a candidate has no gramset."
    )
    print(
        "Primary instances = words left for the benchmark; "
        "Candidate rows = the analyses available for those words."
    )
    print()
    print(f"Review CSV: {path} ({written} rows)")


def _print_benchmark_table(tag: str, results: dict[str, LanguageInstances]) -> None:
    print(f"BENCHMARK — dictorpus-data {tag}")
    print()
    print(
        f"{'Lang':<6}{'Primary instances':>18}{'Texts':>7}{'Candidate rows':>15}"
    )
    totals = {"primary": 0, "texts": 0, "rows": 0}
    for lang in SUPPORTED_LANGUAGES:
        primary, texts, rows = _language_counts(results[lang])
        totals["primary"] += primary
        totals["texts"] += texts
        totals["rows"] += rows
        print(f"{lang:<6}{primary:>18}{texts:>7}{rows:>15}")
    print(
        f"{'ALL':<6}{totals['primary']:>18}{totals['texts']:>7}{totals['rows']:>15}"
    )


def _split_stats(
    assignments: dict[str, dict[str, tuple[TextWeight, ...]]],
) -> dict[str, dict[str, tuple[int, int, int]]]:
    """Return per language and part ``(texts, instances, candidates)``."""
    stats: dict[str, dict[str, tuple[int, int, int]]] = {}
    for lang in SUPPORTED_LANGUAGES:
        stats[lang] = {}
        for part in ("train", "dev", "test"):
            weights = assignments[lang][part]
            texts = len(weights)
            instances = sum(weight.instance_count for weight in weights)
            candidates = sum(weight.candidate_row_count for weight in weights)
            stats[lang][part] = (texts, instances, candidates)
    return stats


def _print_split_table(assignments) -> None:
    stats = _split_stats(assignments)
    totals = {"train": [0, 0, 0], "dev": [0, 0, 0], "test": [0, 0, 0]}
    for lang in SUPPORTED_LANGUAGES:
        for part in ("train", "dev", "test"):
            for index in range(3):
                totals[part][index] += stats[lang][part][index]
    print("SPLIT — target by candidate rows: train 80% / dev 10% / test 10%")
    print()
    print(
        f"{'Lang':<6}{'Part':<7}{'Texts':>6}{'Instances':>10}"
        f"{'Candidates':>11}{'Candidate %':>13}"
    )
    for lang in SUPPORTED_LANGUAGES:
        total_rows = sum(stats[lang][part][2] for part in ("train", "dev", "test"))
        for index, part in enumerate(("train", "dev", "test")):
            texts, instances, candidates = stats[lang][part]
            percent = 100.0 * candidates / total_rows if total_rows else 0.0
            name = lang if index == 0 else ""
            print(
                f"{name:<6}{part:<7}{texts:>6}{instances:>10}{candidates:>11}"
                f"{percent:>11.1f}%"
            )
    all_rows = sum(totals[part][2] for part in ("train", "dev", "test"))
    for index, part in enumerate(("train", "dev", "test")):
        texts, instances, candidates = totals[part]
        percent = 100.0 * candidates / all_rows if all_rows else 0.0
        name = "ALL" if index == 0 else ""
        print(
            f"{name:<6}{part:<7}{texts:>6}{instances:>10}{candidates:>11}"
            f"{percent:>11.1f}%"
        )


def _print_split_report(tag, results, assignments, path: Path, written: bool) -> None:
    _print_benchmark_table(tag, results)
    print()
    _print_split_table(assignments)
    print()
    text_overlap, instance_overlap = split_overlap_counts(assignments)
    print(
        f"Checks: text overlap = {text_overlap}; instance overlap = {instance_overlap}"
    )
    status = "created" if written else "unchanged"
    print(f"Split: {path} ({status})")