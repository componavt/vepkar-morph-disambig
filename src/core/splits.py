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
    split)`` rows; a text assigned more than once is rejected."""
    rows: list[tuple[str, int, str]] = []
    seen: set[tuple[str, int]] = set()
    for language in assignments:
        for part in SPLIT_NAMES:
            for weight in assignments[language].get(part, ()):
                key = (language, weight.text_id)
                if key in seen:
                    raise SplitError(f"Text assigned more than once: {key}")
                seen.add(key)
                rows.append((language, weight.text_id, part))
    return tuple(sorted(rows, key=lambda row: (row[0], row[1])))


def _weight_totals(weights: Iterable[TextWeight]) -> tuple[int, int]:
    instances = sum(weight.instance_count for weight in weights)
    candidates = sum(weight.candidate_row_count for weight in weights)
    return instances, candidates


def validate_split_assignments(
    assignments: dict[str, dict[str, tuple[TextWeight, ...]]],
    expected_weights: tuple[TextWeight, ...],
) -> None:
    """Validate whole-text split assignments against the accepted weights.

    Every accepted text must be assigned to exactly one of ``train``, ``dev``,
    ``test`` with unchanged weights; per-language and overall instance and
    candidate-row totals must be conserved.  Raises :class:`SplitError` on the
    first violation.
    """
    expected_by_key = {
        (weight.language, weight.text_id): weight for weight in expected_weights
    }
    expected_by_language: dict[str, tuple[int, int]] = {}
    for weight in expected_weights:
        instances, candidates = expected_by_language.setdefault(
            weight.language, (0, 0)
        )
        expected_by_language[weight.language] = (
            instances + weight.instance_count,
            candidates + weight.candidate_row_count,
        )

    seen: dict[tuple[str, int], str] = {}
    for language, parts in assignments.items():
        unknown = sorted(set(parts) - set(SPLIT_NAMES))
        if unknown:
            raise SplitError(
                f"Unknown split part in {language}: {', '.join(unknown)}"
            )
        for split in SPLIT_NAMES:
            for weight in parts.get(split, ()):
                key = (weight.language, weight.text_id)
                if weight.language != language:
                    raise SplitError(f"Language mismatch in assignment: {key}")
                if key in seen:
                    raise SplitError(
                        f"Text assigned more than once: {key} "
                        f"({seen[key]} and {split})"
                    )
                if key not in expected_by_key:
                    raise SplitError(f"Unexpected assigned text: {key}")
                if weight != expected_by_key[key]:
                    raise SplitError(f"Assigned text weight changed: {key}")
                seen[key] = split

    missing = sorted(set(expected_by_key) - set(seen))
    if missing:
        raise SplitError(
            "Some accepted texts have no split assignment: "
            f"{missing[0]}{' and more' if len(missing) > 1 else ''}"
        )

    for language, total in expected_by_language.items():
        assigned = [
            weight
            for split in SPLIT_NAMES
            for weight in assignments.get(language, {}).get(split, ())
        ]
        if _weight_totals(assigned) != total:
            raise SplitError(
                f"Assigned instance or candidate totals changed for {language}: "
                f"got {_weight_totals(assigned)}, expected {total}"
            )
    all_assigned = [
        weight
        for language in assignments
        for split in SPLIT_NAMES
        for weight in assignments[language].get(split, ())
    ]
    if _weight_totals(all_assigned) != _weight_totals(expected_weights):
        raise SplitError(
            "Assigned instance or candidate totals changed overall: got "
            f"{_weight_totals(all_assigned)}, expected "
            f"{_weight_totals(expected_weights)}"
        )


def split_overlap_counts(
    assignments: dict[str, dict[str, tuple[TextWeight, ...]]]
) -> tuple[int, int]:
    """Return ``(texts, instances)`` assigned to more than one split part."""
    parts_by_key: dict[tuple[str, int], set[str]] = {}
    for language, parts in assignments.items():
        for split in SPLIT_NAMES:
            for weight in parts.get(split, ()):
                parts_by_key.setdefault((language, weight.text_id), set()).add(split)
    overlapping = {
        key for key, parts in parts_by_key.items() if len(parts) > 1
    }
    text_overlap = len(overlapping)
    instance_overlap = sum(
        weight.instance_count
        for language, parts in assignments.items()
        for split in SPLIT_NAMES
        for weight in parts.get(split, ())
        if (language, weight.text_id) in overlapping
    )
    return text_overlap, instance_overlap


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