"""Offline tests for the compact frequency-diagnostics summary formatter."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from core.diagnostics import FrequencyDiagnostic  # noqa: E402
from frequency_diagnostics_summary import format_frequency_diagnostics_summary  # noqa: E402


def _row(
    language="krl",
    word_id=0,
    word="w",
    candidate_count=2,
    gold_wordform_id=100,
    gold_gramset="G",
    top1_wordform_id=100,
    top1_gramset="G",
    gold_rank=1,
    gold_train_frequency=0,
    top1_train_frequency=0,
):
    return FrequencyDiagnostic(
        language=language,
        word_id=word_id,
        word=word,
        candidate_count=candidate_count,
        gold_wordform_id=gold_wordform_id,
        gold_gramset=gold_gramset,
        top1_wordform_id=top1_wordform_id,
        top1_gramset=top1_gramset,
        gold_rank=gold_rank,
        gold_train_frequency=gold_train_frequency,
        top1_train_frequency=top1_train_frequency,
    )


def _hand_dataset():
    return (
        _row(
            language="vep",
            word_id=2,
            word="vezi",
            candidate_count=10,
            gold_wordform_id=2,
            gold_gramset="V",
            top1_wordform_id=2,
            top1_gramset="V",
            gold_rank=1,
            gold_train_frequency=3,
            top1_train_frequency=3,
        ),
        _row(
            language="krl",
            word_id=1,
            word="kala",
            candidate_count=2,
            gold_wordform_id=1,
            gold_gramset="N",
            top1_wordform_id=1,
            top1_gramset="N",
            gold_rank=1,
            gold_train_frequency=5,
            top1_train_frequency=5,
        ),
        _row(
            language="vep",
            word_id=5,
            candidate_count=10,
            gold_wordform_id=5,
            gold_gramset="V",
            gold_rank=4,
            gold_train_frequency=0,
            top1_train_frequency=0,
        ),
        _row(
            language="krl",
            word_id=3,
            candidate_count=2,
            gold_wordform_id=3,
            gold_gramset="N",
            top1_wordform_id=33,
            top1_gramset="X",
            gold_rank=2,
            gold_train_frequency=2,
            top1_train_frequency=7,
        ),
        _row(
            language="krl",
            word_id=4,
            candidate_count=3,
            gold_wordform_id=4,
            gold_gramset="N",
            top1_wordform_id=44,
            top1_gramset="X",
            gold_rank=3,
            gold_train_frequency=4,
            top1_train_frequency=4,
        ),
    )


HAND_EXPECTED = (
    "Summary",
    "Occurrences: 5",
    "Correct Top-1: 2",
    "Errors: 3",
    "Top-1 accuracy: 0.4000",
    "",
    "Gold rank",
    "Rank 1: 2",
    "Rank 2: 1",
    "Rank 3: 1",
    "Rank >=4: 1",
    "",
    "Frequencies",
    "Zero-gold-frequency occurrences: 1",
    "Zero-gold-frequency errors: 1",
    "Zero-gold errors / all errors (E0/E): 0.3333",
    "Error rate within zero-gold group (E0/Z): 1.0000",
    "Errors with top1 frequency > gold: 1",
    "Errors with equal positive frequencies: 1",
    "Errors with both frequencies zero: 1",
    "",
    "By language",
    "language occurrences errors top1_accuracy",
    "krl 3 2 0.3333",
    "vep 2 1 0.5000",
    "",
    "By candidate count",
    "candidate_count occurrences errors top1_accuracy",
    "2 2 1 0.5000",
    "3 1 1 0.0000",
    "10 2 1 0.5000",
)


def test_hand_dataset_exact_report_lines():
    lines = format_frequency_diagnostics_summary(_hand_dataset())
    assert lines == HAND_EXPECTED


def test_hand_dataset_conservation_invariants():
    lines = format_frequency_diagnostics_summary(_hand_dataset())

    def scalar(prefix):
        for line in lines:
            if line.startswith(prefix):
                return int(line[len(prefix):])
        raise AssertionError(f"missing line prefix {prefix!r}")

    def table_sums(heading, header):
        start = lines.index(heading)
        assert lines[start + 1] == header
        occurrences = 0
        errors = 0
        for line in lines[start + 2:]:
            if not line:
                break
            fields = line.split()
            occurrences += int(fields[1])
            errors += int(fields[2])
        return occurrences, errors

    occurrences = scalar("Occurrences: ")
    correct = scalar("Correct Top-1: ")
    errors = scalar("Errors: ")
    rank_buckets = [
        scalar("Rank 1: "),
        scalar("Rank 2: "),
        scalar("Rank 3: "),
        scalar("Rank >=4: "),
    ]
    error_categories = [
        scalar("Errors with top1 frequency > gold: "),
        scalar("Errors with equal positive frequencies: "),
        scalar("Errors with both frequencies zero: "),
    ]

    assert correct + errors == occurrences
    assert sum(rank_buckets) == occurrences
    assert sum(error_categories) == errors
    assert table_sums(
        "By language", "language occurrences errors top1_accuracy"
    ) == (occurrences, errors)
    assert table_sums(
        "By candidate count", "candidate_count occurrences errors top1_accuracy"
    ) == (occurrences, errors)


EMPTY_EXPECTED = (
    "Summary",
    "Occurrences: 0",
    "Correct Top-1: 0",
    "Errors: 0",
    "Top-1 accuracy: N/A",
    "",
    "Gold rank",
    "Rank 1: 0",
    "Rank 2: 0",
    "Rank 3: 0",
    "Rank >=4: 0",
    "",
    "Frequencies",
    "Zero-gold-frequency occurrences: 0",
    "Zero-gold-frequency errors: 0",
    "Zero-gold errors / all errors (E0/E): N/A",
    "Error rate within zero-gold group (E0/Z): N/A",
    "Errors with top1 frequency > gold: 0",
    "Errors with equal positive frequencies: 0",
    "Errors with both frequencies zero: 0",
    "",
    "By language",
    "language occurrences errors top1_accuracy",
    "",
    "By candidate count",
    "candidate_count occurrences errors top1_accuracy",
)


def test_empty_iterable_keeps_headings_and_headers():
    lines = format_frequency_diagnostics_summary(())
    assert lines == EMPTY_EXPECTED
    assert any(line.endswith("N/A") for line in lines)


NO_ERROR_EXPECTED = (
    "Summary",
    "Occurrences: 2",
    "Correct Top-1: 2",
    "Errors: 0",
    "Top-1 accuracy: 1.0000",
    "",
    "Gold rank",
    "Rank 1: 2",
    "Rank 2: 0",
    "Rank 3: 0",
    "Rank >=4: 0",
    "",
    "Frequencies",
    "Zero-gold-frequency occurrences: 1",
    "Zero-gold-frequency errors: 0",
    "Zero-gold errors / all errors (E0/E): N/A",
    "Error rate within zero-gold group (E0/Z): 0.0000",
    "Errors with top1 frequency > gold: 0",
    "Errors with equal positive frequencies: 0",
    "Errors with both frequencies zero: 0",
    "",
    "By language",
    "language occurrences errors top1_accuracy",
    "krl 1 0 1.0000",
    "vep 1 0 1.0000",
    "",
    "By candidate count",
    "candidate_count occurrences errors top1_accuracy",
    "2 1 0 1.0000",
    "10 1 0 1.0000",
)


def test_no_errors_e0_over_e_is_na():
    rows = (
        _row(language="krl", candidate_count=2, gold_train_frequency=0, top1_train_frequency=0),
        _row(
            language="vep",
            candidate_count=10,
            gold_train_frequency=7,
            top1_train_frequency=7,
        ),
    )
    assert format_frequency_diagnostics_summary(rows) == NO_ERROR_EXPECTED


NO_ZERO_GOLD_EXPECTED = (
    "Summary",
    "Occurrences: 2",
    "Correct Top-1: 1",
    "Errors: 1",
    "Top-1 accuracy: 0.5000",
    "",
    "Gold rank",
    "Rank 1: 1",
    "Rank 2: 1",
    "Rank 3: 0",
    "Rank >=4: 0",
    "",
    "Frequencies",
    "Zero-gold-frequency occurrences: 0",
    "Zero-gold-frequency errors: 0",
    "Zero-gold errors / all errors (E0/E): 0.0000",
    "Error rate within zero-gold group (E0/Z): N/A",
    "Errors with top1 frequency > gold: 0",
    "Errors with equal positive frequencies: 1",
    "Errors with both frequencies zero: 0",
    "",
    "By language",
    "language occurrences errors top1_accuracy",
    "krl 1 0 1.0000",
    "vep 1 1 0.0000",
    "",
    "By candidate count",
    "candidate_count occurrences errors top1_accuracy",
    "2 1 0 1.0000",
    "10 1 1 0.0000",
)


def test_no_zero_gold_frequency_rows_e0_over_z_is_na():
    rows = (
        _row(language="krl", candidate_count=2, gold_train_frequency=5, top1_train_frequency=5),
        _row(
            language="vep",
            candidate_count=10,
            gold_wordform_id=4,
            top1_wordform_id=44,
            top1_gramset="X",
            gold_rank=2,
            gold_train_frequency=4,
            top1_train_frequency=4,
        ),
    )
    assert format_frequency_diagnostics_summary(rows) == NO_ZERO_GOLD_EXPECTED


def test_single_pass_iterable_accepted():
    rows = _hand_dataset()
    lines = format_frequency_diagnostics_summary(
        (row for row in rows)
    )
    assert lines == HAND_EXPECTED


def test_input_unchanged():
    rows = list(_hand_dataset())
    before = list(rows)
    format_frequency_diagnostics_summary(rows)
    assert rows == before


def test_output_order_independent():
    rows = _hand_dataset()
    assert (
        format_frequency_diagnostics_summary(reversed(rows))
        == format_frequency_diagnostics_summary(rows)
    )