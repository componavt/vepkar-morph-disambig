#!/usr/bin/env python3
"""Build a small, blinded context-card pilot from the frozen A1 dev tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.data import read_words  # noqa: E402

A1_DIR = ROOT / "work/experiments/v2026.11/t1-dev-error-analysis-a1-v5"
DATA_DIR = ROOT / "data/dictorpus-data"
DEFAULT_OUTPUT = ROOT / "work/experiments/v2026.11/t1-context-cards-b1-v7"
QUESTIONNAIRE_VERSION = "B1-v7"
LANGUAGES = ("vep", "krl", "olo", "lud")
PILOT_LANGUAGES = ("vep", "krl", "olo")
FAMILIES = {
    "case": (
        ("SG+ACC", "SG+NOM"),
        ("SG+ACC", "SG+GEN"),
        ("SG+GEN", "SG+NOM"),
        ("PL+ACC", "PL+NOM"),
    ),
    "number": (("SG+NOM", "PL+NOM"),),
    "verb": (
        ("INF", "PRS+PASS"),
        ("SG+3+IND", "PL+3+IND"),
    ),
}
CONTEXT_LEVELS = ("K0", "K1", "K2", "K3", "FULL")
STATUSES = {"analyses_selected", "insufficient_context", "none_applicable", "data_problem"}


class B1InputError(ValueError):
    """An A1 or corpus input is missing or inconsistent."""


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
        raise B1InputError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise B1InputError(f"expected a JSON object in {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream, strict=True))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise B1InputError(f"cannot read {path}: {exc}") from exc


def verify_a1_inputs(a1_dir: Path) -> tuple[dict, dict[str, str]]:
    summary = read_json(a1_dir / "summary.json")
    if summary.get("target_split") != "dev":
        raise B1InputError("B1 requires the A1 dev analysis")
    expected_prediction = summary.get("prediction_source", {})
    if not isinstance(expected_prediction, dict) or expected_prediction.get("sha256") != expected_prediction.get("expected_sha256"):
        raise B1InputError("A1 summary does not confirm the approved frozen predictions hash")
    outputs = summary.get("output_sha256")
    if not isinstance(outputs, dict):
        raise B1InputError("A1 summary has no output checksums")
    checks = {}
    for name in ("instances.csv", "candidates.csv", "conflict-summary.csv"):
        path = a1_dir / name
        expected = outputs.get(name)
        if not isinstance(expected, str) or sha256(path) != expected:
            raise B1InputError(f"A1 checksum mismatch for {name}")
        checks[name] = expected
    return summary, checks


def load_word_contexts(
    summary: dict,
    data_dir: Path,
    needed_sentences: dict[str, set[int]],
) -> tuple[dict, dict[str, str]]:
    recorded = summary.get("benchmark_source_files_sha256")
    if not isinstance(recorded, dict):
        raise B1InputError("A1 summary has no benchmark source checksums")
    contexts: dict[tuple[str, int], list[dict]] = {}
    checks = {}
    for language in LANGUAGES:
        path = data_dir / "corpus" / f"words_{language}.csv.zst"
        expected = recorded.get(str(path.resolve()))
        if not isinstance(expected, str) or sha256(path) != expected:
            raise B1InputError(f"corpus word-table checksum mismatch for {language}")
        checks[str(path)] = expected
        frame = read_words(language, data_dir)
        by_sentence: dict[int, list[dict]] = defaultdict(list)
        for row in frame.itertuples(index=False):
            sentence_id = int(row.sentence_id)
            if sentence_id not in needed_sentences.get(language, set()):
                continue
            by_sentence[int(row.sentence_id)].append(
                {
                    "word_id": int(row.word_id),
                    "word_number": int(row.word_number),
                    "word": str(row.word),
                }
            )
        for sentence_id, rows in by_sentence.items():
            rows.sort(key=lambda row: row["word_number"])
            positions = [row["word_number"] for row in rows]
            ids = [row["word_id"] for row in rows]
            if len(positions) != len(set(positions)) or len(ids) != len(set(ids)):
                raise B1InputError(f"duplicate token positions or IDs in {(language, sentence_id)!r}")
            contexts[(language, sentence_id)] = rows
    return contexts, checks


def build_registry(
    instances: list[dict[str, str]], candidates: list[dict[str, str]]
) -> list[dict]:
    candidates_by_word: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        candidates_by_word[(row["language"], row["word_id"])].append(row)
    registry = []
    seen = set()
    for instance in instances:
        word_key = (instance["language"], instance["word_id"])
        if word_key in seen:
            raise B1InputError(f"duplicate A1 usage key: {word_key!r}")
        seen.add(word_key)
        gramsets = {row["gramset"] for row in candidates_by_word.get(word_key, [])}
        gold = instance["gold_gramset"]
        top1 = instance["top1_gramset"]
        for family, pairs in FAMILIES.items():
            for pair_left, pair_right in pairs:
                if gold not in (pair_left, pair_right):
                    continue
                alternative = pair_right if gold == pair_left else pair_left
                if alternative not in gramsets:
                    continue
                registry.append(
                    {
                        "language": instance["language"],
                        "family": family,
                        "pair_left": pair_left,
                        "pair_right": pair_right,
                        "gold_gramset": gold,
                        "alternative_gramset": alternative,
                        "text_id": int(instance["text_id"]),
                        "sentence_id": int(instance["sentence_id"]),
                        "word_id": int(instance["word_id"]),
                        "normalized_form": instance["normalized_form"],
                        "target_position": int(instance["target_position"]),
                        "top1_gramset": top1,
                        "gramset_correct": int(top1 == gold),
                        "selected_transition": int(top1 == alternative),
                        "pair_opportunity": 1,
                    }
                )
    return registry


def select_pilot_sentences(registry: list[dict]) -> tuple[list[dict], dict]:
    selected: dict[tuple[str, int, int], dict] = {}
    selection_notes = []
    # Select one frequent error-bearing contrast per family and language,
    # then seek one gramset-correct control from the same unordered contrast.
    for language in PILOT_LANGUAGES:
        language_rows = [row for row in registry if row["language"] == language]
        for family in FAMILIES:
            family_rows = [row for row in language_rows if row["family"] == family]
            by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
            for row in family_rows:
                pair = tuple(sorted((row["pair_left"], row["pair_right"])))
                by_pair[pair].append(row)
            pair_stats = []
            for pair, rows in by_pair.items():
                errors = sum(row["selected_transition"] for row in rows)
                opportunities = len(rows)
                if errors:
                    pair_stats.append((errors, opportunities, pair, rows))
            if not pair_stats:
                selection_notes.append(
                    {"language": language, "family": family, "status": "no observed pairwise error; omitted"}
                )
                continue
            _, _, pair, pair_rows = max(
                pair_stats,
                key=lambda item: (item[0], item[1], tuple(reversed(item[2]))),
            )
            pair_error_rows = [row for row in pair_rows if row["selected_transition"]]
            controls = [row for row in pair_rows if row["gramset_correct"]]
            controls_by_form: dict[str, list[dict]] = defaultdict(list)
            for row in controls:
                controls_by_form[row["normalized_form"]].append(row)
            errors_by_sentence: dict[tuple[int, int], list[dict]] = defaultdict(list)
            for row in pair_error_rows:
                errors_by_sentence[(row["text_id"], row["sentence_id"])].append(row)
            error_sentence_key, error_targets = min(
                errors_by_sentence.items(),
                key=lambda item: (
                    -len({row["normalized_form"] for row in item[1]} & set(controls_by_form)),
                    -len(item[1]),
                    item[0][0],
                    item[0][1],
                ),
            )
            _add_selected_sentence(
                selected,
                language,
                error_sentence_key,
                error_targets,
                family,
                pair,
                "error_case",
            )

            error_forms = {row["normalized_form"] for row in error_targets}
            controls_by_sentence: dict[tuple[int, int], list[dict]] = defaultdict(list)
            for row in controls:
                controls_by_sentence[(row["text_id"], row["sentence_id"])].append(row)
            if controls_by_sentence:
                control_sentence_key, control_targets = min(
                    controls_by_sentence.items(),
                    key=lambda item: (
                        -sum(row["normalized_form"] in error_forms for row in item[1]),
                        -(item[0] != error_sentence_key),
                        -(item[0][0] != error_sentence_key[0]),
                        -len(item[1]),
                        item[0][0],
                        item[0][1],
                    ),
                )
                _add_selected_sentence(
                    selected,
                    language,
                    control_sentence_key,
                    control_targets,
                    family,
                    pair,
                    "gramset_correct_control",
                )
                control_status = "matched control selected"
            else:
                control_status = "no gramset-correct control for selected contrast"
            selection_notes.append(
                {
                    "language": language,
                    "family": family,
                    "pair": list(pair),
                    "pair_opportunities": len(pair_rows),
                    "pairwise_errors": len(pair_error_rows),
                    "error_sentence": list(error_sentence_key),
                    "error_targets_in_sentence": len(error_targets),
                    "control_status": control_status,
                }
            )

    selected_rows = []
    for (language, text_id, sentence_id), group in selected.items():
        for row in group["targets"].values():
            selected_rows.append(row)
    selected_rows.sort(
        key=lambda row: (
            row["language"], row["text_id"], row["sentence_id"], row["target_position"], row["word_id"]
        )
    )
    # Keep at most six distinct sentences per expert language by construction.
    counts = Counter(row["language"] for row in selected_rows)
    sentence_counts = {
        language: len([key for key in selected if key[0] == language])
        for language in PILOT_LANGUAGES
    }
    if any(count > 6 for count in sentence_counts.values()):
        raise B1InputError(f"pilot exceeded six sentences per language: {sentence_counts!r}")
    return selected_rows, {
        "selection_notes": selection_notes,
        "sentence_count_by_language": sentence_counts,
        "target_count_by_language": dict(counts),
    }


def select_lud_deferred_candidates(registry: list[dict]) -> list[dict]:
    """Keep a short, unassigned shortlist for the language without an expert."""
    deferred = []
    for family in FAMILIES:
        rows = [row for row in registry if row["language"] == "lud" and row["family"] == family]
        by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in rows:
            by_pair[tuple(sorted((row["pair_left"], row["pair_right"])))].append(row)
        pair_choices = []
        for pair, pair_rows in by_pair.items():
            errors = [row for row in pair_rows if row["selected_transition"]]
            pair_choices.append((len(errors), len(pair_rows), pair, pair_rows, errors))
        if not pair_choices:
            continue
        _, opportunities, pair, pair_rows, errors = max(
            pair_choices, key=lambda item: (item[0], item[1], item[2])
        )
        error_sentences: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for row in errors:
            error_sentences[(row["text_id"], row["sentence_id"])].append(row)
        chosen_errors = []
        if error_sentences:
            _, chosen_errors = min(
                error_sentences.items(),
                key=lambda item: (-len(item[1]), item[0][0], item[0][1]),
            )
        controls = [row for row in pair_rows if row["gramset_correct"]]
        chosen_control = min(
            controls,
            key=lambda row: (row["text_id"], row["sentence_id"], row["word_id"]),
        ) if controls else None
        def case_text(item: dict) -> str:
            return (
                f"word_id={item['word_id']} form={item['normalized_form']} "
                f"gold={item['gold_gramset']} alt={item['alternative_gramset']} "
                f"text_id={item['text_id']} sentence_id={item['sentence_id']}"
            )
        deferred.append(
            {
                "language": "lud",
                "family": family,
                "pair_left": pair[0],
                "pair_right": pair[1],
                "opportunity_count": opportunities,
                "pairwise_error_count": len(errors),
                "error_sentence_targets": "; ".join(case_text(row) for row in chosen_errors),
                "gramset_correct_control_candidate": case_text(chosen_control) if chosen_control else "",
                "status": "deferred; no expert currently assigned",
            }
        )
    return deferred


def _add_selected_sentence(
    selected: dict,
    language: str,
    sentence_key: tuple[int, int],
    targets: list[dict],
    family: str,
    pair: tuple[str, str],
    role: str,
) -> None:
    key = (language, sentence_key[0], sentence_key[1])
    group = selected.setdefault(key, {"targets": {}})
    for row in targets:
        provenance = {
            "selection_family": family,
            "selection_pair": " ↔ ".join(pair),
            "internal_role": role,
        }
        target = group["targets"].setdefault(
            row["word_id"], {**row, "selection_provenance": []}
        )
        if provenance not in target["selection_provenance"]:
            target["selection_provenance"].append(provenance)


def prepare_pilot(
    selected: list[dict],
    instances: list[dict[str, str]],
    candidates: list[dict[str, str]],
    contexts: dict,
) -> tuple[list[dict], list[dict], list[dict]]:
    instance_by_key = {(r["language"], int(r["word_id"])): r for r in instances}
    candidates_by_word: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        candidates_by_word[(row["language"], int(row["word_id"]))].append(row)
    selected_sentences = sorted(
        {(r["language"], r["text_id"], r["sentence_id"]) for r in selected}
    )
    sentence_ids = {}
    for language in PILOT_LANGUAGES:
        language_sentences = [key for key in selected_sentences if key[0] == language]
        sentence_ids.update(
            {
                key: f"{language.upper()}-B1-S{index:02d}"
                for index, key in enumerate(language_sentences, start=1)
            }
        )
    for row in selected:
        sentence_key = (row["language"], row["text_id"], row["sentence_id"])
        row["card_id"] = sentence_ids[sentence_key]
    targets_by_sentence: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in selected:
        targets_by_sentence[(row["language"], row["text_id"], row["sentence_id"])].append(row)
    cards, context_rows, target_rows = [], [], []
    for sentence_key in selected_sentences:
        language, text_id, sentence_id = sentence_key
        token_rows = contexts.get((language, sentence_id))
        if not token_rows:
            raise B1InputError(f"missing corpus tokens for sentence {sentence_key!r}")
        token_positions = {row["word_id"]: i for i, row in enumerate(token_rows)}
        targets = sorted(
            targets_by_sentence[sentence_key],
            key=lambda row: (row["target_position"], row["word_id"]),
        )
        if len(targets) != len({row["word_id"] for row in targets}):
            raise B1InputError(f"duplicate selected target in {sentence_key!r}")
        marked_indices = {}
        for target_number, target in enumerate(targets, start=1):
            instance = instance_by_key[(language, target["word_id"])]
            index = token_positions.get(target["word_id"])
            if index is None:
                raise B1InputError(f"target word_id missing from sentence context: {target['word_id']}")
            token = token_rows[index]
            if token["word_number"] != target["target_position"] or token["word"] != instance["normalized_form"]:
                raise B1InputError(
                    f"target identity/position mismatch for word_id={target['word_id']}: "
                    f"corpus=({token['word_number']}, {token['word']!r})"
                )
            target_id = f"{sentence_ids[sentence_key]}-T{target_number:02d}"
            target["target_id"] = target_id
            target["token_index"] = index
            marked_indices[index] = target_id
            full_candidates = candidates_by_word[(language, int(target["word_id"]))]
            gramsets = sorted({row["gramset"] for row in full_candidates})
            if len(gramsets) < 2:
                raise B1InputError(f"target has fewer than two gramsets: {target['word_id']}")
            if len(gramsets) > 26:
                raise B1InputError(f"too many distinct gramsets for lettered options: {target['word_id']}")
            option_map = {gramset: chr(ord("A") + i) for i, gramset in enumerate(gramsets)}
            target["neutral_options"] = option_map
            target["candidate_keys_internal"] = sorted(
                {f"{row['wordform_id']}:{row['gramset']}" for row in full_candidates}
            )
            target_rows.append(
                {
                    **target,
                    "selection_family": ";".join(
                        sorted({p["selection_family"] for p in target["selection_provenance"]})
                    ),
                    "selection_pair": ";".join(
                        sorted({p["selection_pair"] for p in target["selection_provenance"]})
                    ),
                    "internal_role": ";".join(
                        sorted({p["internal_role"] for p in target["selection_provenance"]})
                    ),
                    "gold_gramset": instance["gold_gramset"],
                    "top1_gramset": instance["top1_gramset"],
                    "gold_wordform_id": instance["gold_wordform_id"],
                    "top1_wordform_id": instance["top1_wordform_id"],
                    "gramset_correct": int(instance["same_gramset_as_gold"]),
                    "full_candidate_correct": int(instance["is_correct"]),
                    "candidate_keys_internal": ";".join(target["candidate_keys_internal"]),
                    "neutral_option_map_internal": ";".join(
                        f"{letter}:{gramset}" for gramset, letter in option_map.items()
                    ),
                })
            spans = {
                level: _context_span(level, index, len(token_rows))
                for level in CONTEXT_LEVELS
            }
            repeat_meta = repeated_context_levels(spans)
            for level in CONTEXT_LEVELS:
                token_span = spans[level]
                is_full = token_span == (0, len(token_rows) - 1)
                context_rows.append(
                    {
                        "card_id": sentence_ids[sentence_key],
                        "target_id": target_id,
                        "language": language,
                        "text_id": text_id,
                        "sentence_id": sentence_id,
                        "level": level,
                        "span_start": token_span[0],
                        "span_end": token_span[1],
                        **repeat_meta[level],
                        "token_count": token_span[1] - token_span[0] + 1,
                        "same_as_full_sentence": int(is_full),
                        "context_text": _render_tokens(
                            token_rows, token_span, {index: target_id}
                        ),
                    }
                )
        cards.append(
            {
                "card_id": sentence_ids[sentence_key],
                "language": language,
                "text_id": text_id,
                "sentence_id": sentence_id,
                "benchmark_targets_in_sentence": len(targets),
                "targets": targets,
                "token_rows": token_rows,
                "marked_indices": marked_indices,
            }
        )
    return cards, target_rows, context_rows


def _context_span(level: str, index: int, token_count: int) -> tuple[int, int]:
    if level == "FULL":
        return (0, token_count - 1)
    if level == "K0":
        return (index, index)
    window = int(level[1:])
    return (max(0, index - window), min(token_count - 1, index + window))


def repeated_context_levels(level_spans: dict[str, tuple[int, int]]) -> dict[str, dict]:
    first_level_by_span = {}
    result = {}
    for level in CONTEXT_LEVELS:
        span = level_spans[level]
        repeat_of = first_level_by_span.get(span)
        result[level] = {
            "repeat_of_level": repeat_of or "",
            "requires_new_answer": int(repeat_of is None),
            "response_origin": "new_response" if repeat_of is None else "repeated_context",
        }
        if repeat_of is None:
            first_level_by_span[span] = level
    return result


def validate_response(response: dict, allowed_options: set[str]) -> None:
    status = response.get("status")
    selected = response.get("selected_options", [])
    if status not in STATUSES:
        raise ValueError("unknown response status")
    if len(selected) != len(set(selected)):
        raise ValueError("duplicate selected options")
    if not set(selected) <= allowed_options:
        raise ValueError("unknown option code")
    if status == "analyses_selected" and not selected:
        raise ValueError("selected analyses status requires at least one option")
    if status != "analyses_selected" and selected:
        raise ValueError("special status must not contain selected analyses")


def assign_primary_targets(target_rows: list[dict]) -> list[dict]:
    """Mark one deterministic primary target per sentence and link controls."""
    by_sentence = defaultdict(list)
    for row in target_rows:
        by_sentence[(row["language"], row["text_id"], row["sentence_id"])].append(row)

    controls_by_pair = defaultdict(list)
    for row in target_rows:
        if row["gramset_correct"]:
            for pair in row["selection_pair"].split(";"):
                if pair:
                    controls_by_pair[pair].append(row)

    control_links = []
    for row in target_rows:
        pairs = [pair for pair in row["selection_pair"].split(";") if pair]
        controls = {} if row["gramset_correct"] else {
            control["target_id"]: control
            for pair in pairs
            for control in controls_by_pair[pair]
            if control["target_id"] != row["target_id"]
        }
        same_form = sorted(
            (control for control in controls.values() if control["normalized_form"] == row["normalized_form"]),
            key=lambda control: (control["language"], control["text_id"], control["sentence_id"], control["target_position"], control["word_id"]),
        )
        other_form = sorted(
            (control for control in controls.values() if control["normalized_form"] != row["normalized_form"]),
            key=lambda control: (control["language"], control["text_id"], control["sentence_id"], control["target_position"], control["word_id"]),
        )
        if row["gramset_correct"]:
            quality, linked = "not_applicable_control_target", []
        elif same_form:
            quality, linked = "same_form_control", same_form
        elif other_form:
            quality, linked = "same_contrast_other_form_control", other_form
        else:
            quality, linked = "no_control", []
        row["control_quality"] = quality
        row["control_target_ids_internal"] = ";".join(control["target_id"] for control in linked)
        for control in linked:
            control_links.append({
                "target_id": row["target_id"],
                "control_target_id": control["target_id"],
                "control_quality": quality,
                "same_form": int(control["normalized_form"] == row["normalized_form"]),
                "contrast": ";".join(pairs),
            })

    for sentence_rows in by_sentence.values():
        def primary_key(row):
            is_error = row["gramset_correct"] == 0
            has_same_form_control = (
                row["gramset_correct"] == 0
                and row["control_quality"] == "same_form_control"
            )
            return (not is_error, not has_same_form_control, row["target_position"], row["word_id"])

        primary = min(sentence_rows, key=primary_key)
        for row in sentence_rows:
            row["is_primary_target"] = int(row is primary)
            if row is primary:
                if row["gramset_correct"] == 0 and row["control_quality"] == "same_form_control":
                    reason = "error_target_with_same_form_control"
                elif row["gramset_correct"] == 0:
                    reason = "error_target; no same-form control"
                elif row["control_quality"] == "same_form_control":
                    reason = "gramset-correct target with same-form control; no error target in sentence"
                else:
                    reason = "first stable target; no error target in sentence"
            else:
                reason = "additional target retained internally; excluded from staged context judgments"
            row["primary_selection_reason"] = reason
    return control_links


def _render_tokens(
    token_rows: list[dict], span: tuple[int, int], marks: dict[int, str]
) -> str:
    start, end = span
    parts = []
    target_only = start == end and start in marks
    if start > 0 and not target_only:
        parts.append("…")
    for index in range(start, end + 1):
        token = token_rows[index]["word"]
        if index in marks:
            token = f"**[{marks[index]} {token}]**"
        parts.append(token)
    if end < len(token_rows) - 1 and not target_only:
        parts.append("…")
    return " ".join(parts)


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    output_dir: Path,
    cards: list[dict],
    target_rows: list[dict],
    context_rows: list[dict],
    questionnaire_root: Path,
) -> None:
    internal = [
        "# B1: внутренняя версия контекстного пилота", "",
        "Внутренний материал; экспертам не предъявлять.",
        "Контекст — последовательность значений `word` из экспортных таблиц `words_<language>.csv.zst`, отсортированная по `word_number`. Исходная пунктуация и регистр не восстанавливаются.",
        "Разборы дедуплицированы по грамсету только для предъявления грамсетного вопроса; все ключи `(wordform_id, gramset)` сохранены.",
        "Совпадение грамсета не означает совпадение полного кандидата.", "",
    ]
    neutral_by_level = {
        (language, level): [
            f"# Контекстный опросник B1 — `{language}`, {level}", "",
            f"Эксперт: ______  ·  Версия: `{QUESTIONNAIRE_VERSION}`  ·  Этап: `{level}`  ·  Дата заполнения: ______", "",
            (
                "Это заключительный этап. Пожалуйста, заполните его и верните файл с ответами."
                if level == "FULL"
                else "Заполните этот этап в своём экземпляре и сохраните ответы без изменений до следующего этапа. Не ищите исходный текст во внешних источниках."
            ),
            "Для каждой карточки укажите один или несколько допустимых разборов либо один специальный статус. Комментарий необязателен.", "",
        ]
        for language in PILOT_LANGUAGES
        for level in CONTEXT_LEVELS
    }
    contexts_by_target_level = {
        (row["target_id"], row["level"]): row for row in context_rows
    }
    targets_by_sentence = defaultdict(list)
    for row in target_rows:
        targets_by_sentence[(row["language"], row["text_id"], row["sentence_id"])].append(row)

    for card in cards:
        language = card["language"]
        key = (language, card["text_id"], card["sentence_id"])
        targets = sorted(targets_by_sentence[key], key=lambda row: (row["target_position"], row["word_id"]))
        primary = next(row for row in targets if row["is_primary_target"])
        internal.extend([
            f"## {card['card_id']} — `{language}` · text `{card['text_id']}` · sentence `{card['sentence_id']}`", "",
            f"Полная последовательность экспортных токенов: {_render_tokens(card['token_rows'], (0, len(card['token_rows']) - 1), card['marked_indices'])}", "",
        ])
        for target in targets:
            internal.extend([
                f"### {target['target_id']} — {target['normalized_form']}", "",
                f"Основная цель: {bool(target['is_primary_target'])}. Причина: {target['primary_selection_reason']}.",
                "Внутренний отбор: " + "; ".join(
                    f"`{p['selection_family']}`, контраст `{p['selection_pair']}`, роль `{p['internal_role']}`"
                    for p in target["selection_provenance"]
                ) + ".",
                f"Gold `{target['gold_gramset']}` / `{target['gold_wordform_id']}`; Top-1 `{target['top1_gramset']}` / `{target['top1_wordform_id']}`; совпадение грамсета={target['gramset_correct']}, полного кандидата={target['full_candidate_correct']}.",
                f"Качество контроля: `{target['control_quality']}`; связанные контрольные цели: `{target['control_target_ids_internal']}`.",
                "Кандидатные ключи: " + ", ".join(f"`{item}`" for item in target["candidate_keys_internal"].split(";")), "",
            ])
        for level in CONTEXT_LEVELS:
            neutral = neutral_by_level[(language, level)]
            context = contexts_by_target_level[(primary["target_id"], level)]
            neutral.extend([
                f"## {primary['target_id']}", "",
                f"Целевая словоформа: **{primary['normalized_form']}**", "",
                f"Контекст: {context['context_text']}", "",
            ])
            if context["repeat_of_level"]:
                neutral.extend([
                    f"Этот фрагмент совпадает с ранее показанным уровнем {context['repeat_of_level']}; новых слов он не добавляет. Повторный ответ не требуется.", "", "---", "",
                ])
                continue
            if context["same_as_full_sentence"] and level != "FULL":
                neutral.extend([f"Уровень {level} уже охватывает всю последовательность экспортных токенов предложения.", ""])
            options = sorted(primary["neutral_options"].items(), key=lambda item: item[1])
            neutral.append("### Варианты")
            neutral.append("")
            neutral.extend(f"- **{letter}.** `{primary['normalized_form']}` — `{gramset}`" for gramset, letter in options)
            neutral.extend([
                "", "### Ответ", "",
                "Статус (отметьте один):", 
                "- [ ] Выбраны допустимые разборы",
                "- [ ] Недостаточно контекста",
                "- [ ] Ни один вариант не подходит",
                "- [ ] Проблема текста или данных",
                "",
                "Коды выбранных вариантов (A; B; ...; только если выбран первый статус): ______",
                "Более широкий контекст был показан до этого этапа? [ ] нет  [ ] да; если да, какой уровень: ______",
                "Комментарий (необязательно): ______", "", "---", "",
            ])
    (output_dir / "internal-cards.md").write_text("\n".join(internal) + "\n", encoding="utf-8")
    questionnaire_root.mkdir(parents=True, exist_ok=True)
    for (language, level), lines in neutral_by_level.items():
        content = "\n".join(lines).rstrip() + "\n"
        filename = f"{level.lower()}.md"
        (output_dir / f"neutral-{language}-{filename}").write_text(content, encoding="utf-8")
        language_dir = questionnaire_root / language
        language_dir.mkdir(parents=True, exist_ok=True)
        (language_dir / filename).write_text(content, encoding="utf-8")


def _context_rows_for_card(card: dict, target_rows: list[dict], level: str) -> list[dict]:
    result = []
    targets = [
        row for row in target_rows
        if row["card_id"] == card["card_id"]
    ]
    for target in targets:
        index = target["token_index"]
        span = _context_span(level, index, len(card["token_rows"]))
        result.append(
            {
                "target_id": target["target_id"],
                "same_as_full_sentence": int(span == (0, len(card["token_rows"]) - 1)),
                "context_text": _render_tokens(card["token_rows"], span, {index: target["target_id"]}),
            }
        )
    return result


def run(a1_dir: Path, data_dir: Path, output_dir: Path, questionnaire_root: Path) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise B1InputError(f"refusing to overwrite existing output directory: {output_dir}")
    summary, a1_hashes = verify_a1_inputs(a1_dir)
    instances = read_csv(a1_dir / "instances.csv")
    candidates = read_csv(a1_dir / "candidates.csv")
    registry = build_registry(instances, candidates)
    selected, selection_summary = select_pilot_sentences(registry)
    lud_deferred = select_lud_deferred_candidates(registry)
    needed_sentences: dict[str, set[int]] = defaultdict(set)
    for row in selected:
        needed_sentences[row["language"]].add(row["sentence_id"])
    contexts, word_hashes = load_word_contexts(summary, data_dir, needed_sentences)
    cards, target_rows, context_rows = prepare_pilot(selected, instances, candidates, contexts)
    control_links = assign_primary_targets(target_rows)

    output_dir.mkdir(parents=True)
    fields_registry = (
        "language", "family", "pair_left", "pair_right", "gold_gramset", "alternative_gramset",
        "text_id", "sentence_id", "word_id", "normalized_form", "target_position", "top1_gramset",
        "gramset_correct", "selected_transition", "pair_opportunity",
    )
    fields_selected = (
        "card_id", "target_id", "language", "text_id", "sentence_id", "word_id", "normalized_form",
        "target_position", "selection_family", "selection_pair", "internal_role", "gold_gramset",
        "top1_gramset", "gramset_correct", "full_candidate_correct", "candidate_keys_internal",
        "neutral_option_map_internal", "is_primary_target", "primary_selection_reason",
        "control_quality", "control_target_ids_internal",
    )
    write_csv(output_dir / "opportunity-register.csv", registry, fields_registry)
    write_csv(
        output_dir / "deferred-lud-candidates.csv",
        lud_deferred,
        (
            "language", "family", "pair_left", "pair_right", "opportunity_count",
            "pairwise_error_count", "error_sentence_targets", "gramset_correct_control_candidate", "status",
        ),
    )
    write_csv(output_dir / "selected-targets-internal.csv", target_rows, fields_selected)
    write_csv(
        output_dir / "control-links-internal.csv",
        control_links,
        ("target_id", "control_target_id", "control_quality", "same_form", "contrast"),
    )
    write_csv(
        output_dir / "context-views-internal.csv",
        context_rows,
        ("card_id", "target_id", "language", "text_id", "sentence_id", "level", "span_start", "span_end", "repeat_of_level", "requires_new_answer", "response_origin", "token_count", "same_as_full_sentence", "context_text"),
    )
    write_markdown(output_dir, cards, target_rows, context_rows, questionnaire_root)

    errors_by_family = Counter()
    opportunities_by_family = Counter()
    primary_target_ids = {
        target["target_id"] for target in target_rows if target["is_primary_target"]
    }
    for row in registry:
        key = (row["language"], row["family"])
        opportunities_by_family[key] += 1
        errors_by_family[key] += row["selected_transition"]
    summary_out = {
        "study": "B1 context-card pilot",
        "source_a1_dir": str(a1_dir),
        "source_a1_sha256": a1_hashes,
        "source_word_tables_sha256": word_hashes,
        "source_prediction_sha256": summary["prediction_source"]["sha256"],
        "source_split": "dev",
        "questionnaire_version": QUESTIONNAIRE_VERSION,
        "context_definition": "K0 target wordform plus the fixed list of available candidate gramsets; K1/K2/K3 token windows by word_id and word_number plus the same candidate list; FULL all word-table tokens in the sentence plus the same candidate list",
        "context_token_source": "full sequence of exported words_<language>.csv.zst `word` values sorted by word_number; not a recovered original sentence string; punctuation/capitalization are not restored",
        "selection_unit": "one primary target per sentence for staged context judgments; additional targets remain internal",
        "selection_is_purposive": True,
        "selection_summary": selection_summary,
        "lud_deferred_candidate_count": len(lud_deferred),
        "opportunity_counts_by_language_family": {
            f"{language}:{family}": {
                "opportunities": opportunities_by_family[(language, family)],
                "selected_pairwise_errors": errors_by_family[(language, family)],
            }
            for language in LANGUAGES
            for family in FAMILIES
        },
        "sentence_count": len(cards),
        "target_count": len(target_rows),
        "primary_target_count": sum(row["is_primary_target"] for row in target_rows),
        "additional_internal_target_count": sum(not row["is_primary_target"] for row in target_rows),
        "context_view_count": len(context_rows),
        "repeated_context_view_count": sum(not row["requires_new_answer"] for row in context_rows),
        "primary_target_repeated_context_count": sum(
            row["target_id"] in primary_target_ids and not row["requires_new_answer"]
            for row in context_rows
        ),
        "new_response_request_count": sum(
            row["target_id"] in primary_target_ids and row["requires_new_answer"]
            for row in context_rows
        ),
        "response_request_scope": "primary targets only; additional targets remain internal",
        "unique_primary_context_response_count": sum(
            row["target_id"] in primary_target_ids and row["requires_new_answer"]
            for row in context_rows
        ),
        "context_levels": list(CONTEXT_LEVELS),
        "lud_expert_status": "no expert currently assigned; registry retained, no expert pilot cards prepared",
        "test_used": False,
        "model_retrained": False,
        "new_predictions_generated": False,
        "gold_or_candidates_modified": False,
        "experts_contacted": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_out, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary_out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a1-dir", type=Path, default=A1_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--questionnaire-root", type=Path, default=ROOT / "pclaude/experts/questionnaires")
    args = parser.parse_args()
    try:
        summary = run(
            args.a1_dir.resolve(),
            args.data_dir.resolve(),
            args.output_dir.resolve(),
            args.questionnaire_root.resolve(),
        )
    except (B1InputError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
