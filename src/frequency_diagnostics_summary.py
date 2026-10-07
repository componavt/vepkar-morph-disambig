"""Format a compact numerical summary of frequency-baseline diagnostic rows.

The module is pure and deterministic: it reads already constructed
:class:`core.diagnostics.FrequencyDiagnostic` rows, performs no validation,
and neither prints nor reads or writes files.
"""

from __future__ import annotations

from typing import Iterable

from core.diagnostics import FrequencyDiagnostic


def _ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "N/A"
    return f"{numerator / denominator:.4f}"


def format_frequency_diagnostics_summary(
    rows: Iterable[FrequencyDiagnostic],
) -> tuple[str, ...]:
    """Return formatted summary lines for the given diagnostic rows.

    The iterable may be single-pass; inputs are not mutated.  Output lines are
    deterministic and independent of input order.
    """
    collected = tuple(rows)
    total = len(collected)

    correct = 0
    rank_counts = {1: 0, 2: 0, 3: 0}
    rank_ge4 = 0
    zero_gold = 0
    zero_gold_errors = 0
    errors_top1_gt_gold = 0
    errors_equal_positive = 0
    errors_both_zero = 0
    languages: dict[str, list[FrequencyDiagnostic]] = {}
    candidate_counts: dict[int, list[FrequencyDiagnostic]] = {}

    for row in collected:
        is_correct = row.gold_rank == 1
        if is_correct:
            correct += 1
        if row.gold_rank in rank_counts:
            rank_counts[row.gold_rank] += 1
        else:
            rank_ge4 += 1
        if row.gold_train_frequency == 0:
            zero_gold += 1
        if not is_correct:
            if row.gold_train_frequency == 0:
                zero_gold_errors += 1
            if row.top1_train_frequency > row.gold_train_frequency:
                errors_top1_gt_gold += 1
            elif (
                row.top1_train_frequency == row.gold_train_frequency
                and row.gold_train_frequency > 0
            ):
                errors_equal_positive += 1
            elif (
                row.top1_train_frequency == 0 and row.gold_train_frequency == 0
            ):
                errors_both_zero += 1
        languages.setdefault(row.language, []).append(row)
        candidate_counts.setdefault(row.candidate_count, []).append(row)

    errors = total - correct

    lines: list[str] = []
    lines.append("Summary")
    lines.append(f"Occurrences: {total}")
    lines.append(f"Correct Top-1: {correct}")
    lines.append(f"Errors: {errors}")
    lines.append(f"Top-1 accuracy: {_ratio(correct, total)}")
    lines.append("")
    lines.append("Gold rank")
    lines.append(f"Rank 1: {rank_counts[1]}")
    lines.append(f"Rank 2: {rank_counts[2]}")
    lines.append(f"Rank 3: {rank_counts[3]}")
    lines.append(f"Rank >=4: {rank_ge4}")
    lines.append("")
    lines.append("Frequencies")
    lines.append(f"Zero-gold-frequency occurrences: {zero_gold}")
    lines.append(f"Zero-gold-frequency errors: {zero_gold_errors}")
    lines.append(
        f"Zero-gold errors / all errors (E0/E): {_ratio(zero_gold_errors, errors)}"
    )
    lines.append(
        f"Error rate within zero-gold group (E0/Z): "
        f"{_ratio(zero_gold_errors, zero_gold)}"
    )
    lines.append(f"Errors with top1 frequency > gold: {errors_top1_gt_gold}")
    lines.append(f"Errors with equal positive frequencies: {errors_equal_positive}")
    lines.append(f"Errors with both frequencies zero: {errors_both_zero}")
    lines.append("")
    lines.append("By language")
    lines.append("language occurrences errors top1_accuracy")
    for language in sorted(languages):
        group = languages[language]
        group_total = len(group)
        group_errors = sum(1 for row in group if row.gold_rank > 1)
        lines.append(
            f"{language} {group_total} {group_errors} "
            f"{_ratio(group_total - group_errors, group_total)}"
        )
    lines.append("")
    lines.append("By candidate count")
    lines.append("candidate_count occurrences errors top1_accuracy")
    for candidate_count in sorted(candidate_counts):
        group = candidate_counts[candidate_count]
        group_total = len(group)
        group_errors = sum(1 for row in group if row.gold_rank > 1)
        lines.append(
            f"{candidate_count} {group_total} {group_errors} "
            f"{_ratio(group_total - group_errors, group_total)}"
        )
    return tuple(lines)