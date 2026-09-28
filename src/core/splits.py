"""Whole-text train/dev/test assignment over accepted benchmark instances.

Accepted step-7 instances are grouped by ``(language, text_id)``; each whole
text is then assigned to exactly one split per language, targeting
80/10/10 shares of the accepted candidate rows.  Assignments are computed by a
simple seeded greedy with deterministic tie-breaking and a fixed number of
restarts; the best candidate-row fit is kept.
"""

from __future__ import annotations

import csv
import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from core.instances import Instance

SPLIT_NAMES = ("train", "dev", "test")
TARGET_SHARES = {"train": 0.80, "dev": 0.10, "test": 0.10}
SPLIT_SEED = 20260928
SPLIT_ATTEMPTS = 200

SPLIT_CSV_PREFIX = "splits"
SPLIT_CSV_HEADER = ("language", "text_id", "split")

DEFAULT_SPLIT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "derived"


class SplitError(Exception):
    """Whole-text split assignment or its CSV output failed."""


@dataclass(frozen=True)
class TextWeight:
    """Aggregated accepted-instance weight of one text within one language."""

    language: str
    text_id: int
    instance_count: int
    candidate_row_count: int


def compute_text_weights(instances: Iterable[Instance]) -> tuple[TextWeight, ...]:
    """Group accepted instances by ``(language, text_id)`` and total weights.

    A ``text_id`` appearing in more than one language variety is rejected with
    a diagnostic instead of being assigned to conflicting splits.
    """
    counts: dict[tuple[str, int], list[int]] = {}
    language_by_text: dict[int, set[str]] = {}
    for instance in instances:
        key = (instance.language, int(instance.text_id))
        entry = counts.setdefault(key, [0, 0])
        entry[0] += 1
        entry[1] += len(instance.candidates)
        language_by_text.setdefault(key[1], set()).add(instance.language)
    conflicts = {
        text_id: languages
        for text_id, languages in sorted(language_by_text.items())
        if len(languages) > 1
    }
    if conflicts:
        details = "; ".join(
            f"text_id={text_id} in {', '.join(sorted(languages))}"
            for text_id, languages in conflicts.items()
        )
        raise SplitError(
            "One text_id appears in more than one language variety: " + details
        )
    return tuple(
        TextWeight(language, text_id, entry[0], entry[1])
        for (language, text_id), entry in sorted(counts.items())
    )


def design_text_weights(
    rows: tuple[tuple[str, int, int, int], ...]
) -> tuple[TextWeight, ...]:
    """Build :class:`TextWeight` objects from ``(language, text_id,
    instance_count, candidate_row_count)`` rows, sorted like real weights."""
    return tuple(
        TextWeight(language, text_id, instances, candidates)
        for language, text_id, instances, candidates in sorted(rows)
    )


def _ensure_nonempty(
    bins: dict[str, list[TextWeight]]
) -> dict[str, list[TextWeight]]:
    """Move one smallest text into each empty part when at least three texts
    exist; deterministic and minimal for the 80/10/10 fit."""
    for part in SPLIT_NAMES:
        if bins[part]:
            continue
        source = max(
            (
                candidate
                for candidate in SPLIT_NAMES
                if len(bins[candidate]) > 1
            ),
            key=lambda candidate: (len(bins[candidate]), -SPLIT_NAMES.index(candidate)),
            default=None,
        )
        if source is None:
            continue
        mover = min(
            bins[source],
            key=lambda weight: (weight.candidate_row_count, weight.text_id),
        )
        bins[source].remove(mover)
        bins[part].append(mover)
    return bins


def assign_texts(weights: tuple[TextWeight, ...]) -> dict[str, tuple[TextWeight, ...]]:
    """Assign whole texts to train/dev/test, minimizing candidate-row deviation.

    Deterministic: fixed seed, fixed restarts, and stable tie-breaking.  The
    80/10/10 shares are a target, not an exact guarantee.
    """
    total = sum(weight.candidate_row_count for weight in weights)
    if total == 0:
        raise SplitError("Cannot split a language with no accepted candidate rows")
    targets = {part: total * share for part, share in TARGET_SHARES.items()}
    best: dict[str, tuple[TextWeight, ...]] | None = None
    best_score: float | None = None
    for attempt in range(SPLIT_ATTEMPTS):
        order = list(weights)
        random.Random(SPLIT_SEED + attempt).shuffle(order)
        bins: dict[str, list[TextWeight]] = {part: [] for part in SPLIT_NAMES}
        for weight in order:
            counts = {
                part: sum(entry.candidate_row_count for entry in bins[part])
                for part in SPLIT_NAMES
            }
            chosen = max(
                SPLIT_NAMES,
                key=lambda part: (targets[part] - counts[part]),
            )
            bins[chosen].append(weight)
        _ensure_nonempty(bins)
        score = sum(
            abs(
                sum(entry.candidate_row_count for entry in bins[part])
                - targets[part]
            )
            for part in SPLIT_NAMES
        )
        if best_score is None or score < best_score:
            best_score = score
            best = {
                part: tuple(
                    sorted(bins[part], key=lambda weight: weight.text_id)
                )
                for part in SPLIT_NAMES
            }
    assert best is not None
    return best


def build_split_rows(
    assignments: dict[str, dict[str, tuple[TextWeight, ...]]]
) -> tuple[tuple[str, int, str], ...]:
    """Flatten per-language assignments into sorted ``(language, text_id,
    split)`` rows."""
    rows: set[tuple[str, int, str]] = set()
    for language in assignments:
        for part in SPLIT_NAMES:
            for weight in assignments[language][part]:
                rows.add((language, weight.text_id, part))
    return tuple(sorted(rows, key=lambda row: (row[0], row[1])))


def split_csv_bytes(rows: Iterable[tuple[str, int, str]]) -> bytes:
    """Encode the split CSV with the exact agreed header."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(SPLIT_CSV_HEADER)
    writer.writerows(tuple(rows))
    return buffer.getvalue().encode("utf-8")


def split_csv_path(output_dir: Path, tag: str) -> Path:
    """Return the version-specific split CSV path."""
    return output_dir / f"{SPLIT_CSV_PREFIX}_{tag}.csv"


def write_split_csv(path: Path, content: bytes) -> bool:
    """Write the split CSV unless it already holds identical bytes.

    An existing file with different content is rejected without overwrite.
    Returns True when the file was written, False when it was left unchanged.
    """
    if path.exists():
        if path.read_bytes() == content:
            return False
        raise SplitError(
            f"Existing split file differs from the computed split and will not "
            f"be overwritten: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return True