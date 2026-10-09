"""Shared prediction-report helpers for CLI command handlers."""

from __future__ import annotations


def _print_validation_failure(result) -> None:
    """Print the detailed validation-category report for a failed validation."""
    print("Predictions validation: FAILED")
    print(f"Split: {result.split}")
    print(
        f"Expected: {result.expected_word_count:,} word instances, "
        f"{result.expected_candidate_count:,} candidate rows"
    )
    print(
        f"Found: {result.predicted_word_count:,} word instances, "
        f"{result.predicted_candidate_count:,} prediction rows"
    )
    print()
    # Iterate error_counts in the validator's stable insertion order, which is
    # deterministic for a fixed input; the CLI does not sort or rename
    # categories.  error_counts holds the full count while errors holds at most
    # three representative examples, so the count is never inferred from the
    # example list length.
    for category in result.error_counts:
        count = result.error_counts[category]
        examples = result.errors[category]
        noun = "violation" if count == 1 else "violations"
        print(f"{category}: {count:,} {noun}")
        for example in examples[:3]:
            print(f"  {example}")