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
DEFAULT_OUTPUT = ROOT / "work/experiments/v2026.11/t1-context-cards-b1-v4"
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
            for level in CONTEXT_LEVELS:
                token_span = _context_span(level, index, len(token_rows))
                is_full = token_span == (0, len(token_rows) - 1)
                context_rows.append(
                    {
                        "card_id": sentence_ids[sentence_key],
                        "target_id": target_id,
                        "language": language,
                        "text_id": text_id,
                        "sentence_id": sentence_id,
                        "level": level,
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


def write_markdown(output_dir: Path, cards: list[dict], target_rows: list[dict]) -> None:
    internal = [
        "# B1: внутренняя версия контекстного пилота",
        "",
        "Черновик для исследовательской команды. Не предъявлять экспертам в таком виде.",
        "Контекст собран из токенов `dictorpus-data v2026.11`, отсортированных по `word_number`; написание и исходная пунктуация предложения не восстанавливались.",
        "Решения модели заморожены в A1-v5. `gramset_correct` относится к грамсету; полная правильность требует совпадения `(wordform_id, gramset)`. ",
        "",
    ]
    neutral_by_level = {
        (language, level): [
            f"# B1: нейтральные карточки `{language}`, этап {level}",
            "",
            "Показывать только этот файл на текущем этапе. Зафиксировать ответ до открытия следующего уровня контекста.",
            "Варианты грамсетов не ранжированы. Кандидаты с одинаковым грамсетом объединены только для грамсетного вопроса; полный список ключей сохранён во внутренней таблице.",
            "",
        ]
        for language in PILOT_LANGUAGES
        for level in CONTEXT_LEVELS
    }
    targets_by_sentence = defaultdict(list)
    for row in target_rows:
        targets_by_sentence[(row["language"], row["text_id"], row["sentence_id"])].append(row)
    for card in cards:
        language = card["language"]
        key = (language, card["text_id"], card["sentence_id"])
        targets = sorted(targets_by_sentence[key], key=lambda row: row["target_position"])
        internal.extend(
            [
                f"## {card['card_id']} — `{language}` · text `{card['text_id']}` · sentence `{card['sentence_id']}`",
                "",
                f"Полная последовательность токенов: {_render_tokens(card['token_rows'], (0, len(card['token_rows']) - 1), card['marked_indices'])}",
                "",
            ]
        )
        for target in targets:
            internal.extend(
                [
                    f"### {target['target_id']} — {target['normalized_form']}",
                    "",
                    "Внутренний отбор: " + "; ".join(
                        f"`{p['selection_family']}`, контраст `{p['selection_pair']}`, роль `{p['internal_role']}`"
                        for p in target["selection_provenance"]
                    ) + ".",
                    f"Gold `{target['gold_gramset']}` / `{target['gold_wordform_id']}`; Top-1 `{target['top1_gramset']}` / `{target['top1_wordform_id']}`; совпадение грамсета={target['gramset_correct']}, полного кандидата={target['full_candidate_correct']}.",
                    "Кандидатные ключи: " + ", ".join(
                        f"`{item}`" for item in target["candidate_keys_internal"].split(";")
                    ),
                    "",
                ]
            )
        for level in CONTEXT_LEVELS:
            neutral = neutral_by_level[(language, level)]
            neutral.extend([f"## {card['card_id']} — `{language}`", ""])
            context_map = {
                row["target_id"]: row
                for row in [
                    c for c in _context_rows_for_card(card, target_rows, level)
                ]
            }
            for target in targets:
                context = context_map[target["target_id"]]
                neutral.append(f"### {target['target_id']}")
                neutral.append("")
                neutral.append(context["context_text"])
                if context["same_as_full_sentence"] and level != "FULL":
                    neutral.append(f"\n_(Окно {level} уже охватывает всё предложение.)_")
                neutral.append("")
                neutral.append("Варианты: " + " · ".join(
                    f"**{letter}.** `{gramset}`"
                    for gramset, letter in sorted(
                        target["neutral_options"].items(), key=lambda item: item[1]
                    )
                ))
                neutral.append("")
                neutral.append("Ответ: один или несколько вариантов; недостаточно контекста; ни один вариант; проблема текста; требуется более широкий контекст.")
                neutral.append("Уверенность: высокая / средняя / низкая. Краткое основание: ______")
                neutral.append("")
            neutral.append("---\n")
    (output_dir / "internal-cards.md").write_text("\n".join(internal) + "\n", encoding="utf-8")
    for (language, level), lines in neutral_by_level.items():
        (output_dir / f"neutral-{language}-{level.lower()}.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    protocol = """# Проект протокола B1

Цель пилота — выяснить, на каком предъявленном уровне контекста эксперт может обосновать грамсетное суждение для доступных альтернатив. Это целевой пилот диагностических случаев, а не репрезентативная оценка языка или корпуса.

## Последовательность предъявления

1. Для выбранного языка предъявлять только соответствующий файл `neutral-<язык>-k0.md`; до следующего этапа записать ответ, уверенность и фрагмент, на который эксперт опирался.
2. Затем по порядку открыть файлы этого же языка для K1, K2, K3 и FULL. На каждом этапе сохранять ответ до просмотра следующего уровня; изменения ответа записывать отдельно, не затирать прежние решения.
3. Разрешены несколько допустимых грамсетов, «недостаточно контекста», «ни один вариант», замечание о качестве текста и просьба о более широком фрагменте. Не требовать единственного выбора.
4. Порядок вариантов определяется грамсетом и одинаков во всех уровнях одной цели; он не связан с rank или score модели.
5. Одинаковые текстовые окна отмечаются. Окна остаются внутри предложения; соседнее предложение в этом пилоте не подмешивается.

Нейтральные файлы — только макеты. До предъявления нужно согласовать формулировки на соответствующем языке, способ выделения целей и удобство полей ответа. Экспертам материалы не отправлялись.
"""
    (output_dir / "protocol-draft.md").write_text(protocol, encoding="utf-8")


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


def run(a1_dir: Path, data_dir: Path, output_dir: Path) -> dict:
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
        "neutral_option_map_internal",
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
        output_dir / "context-views-internal.csv",
        context_rows,
        ("card_id", "target_id", "language", "text_id", "sentence_id", "level", "token_count", "same_as_full_sentence", "context_text"),
    )
    write_markdown(output_dir, cards, target_rows)

    errors_by_family = Counter()
    opportunities_by_family = Counter()
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
        "context_definition": "K0 target only; K1/K2/K3 token windows by word_id and word_number; FULL all word-table tokens in the sentence",
        "context_token_source": "dictorpus-data v2026.11 words_<language>.csv.zst; sorted by word_number; no punctuation restoration",
        "selection_unit": "sentence with one or more individually identified targets",
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
        "context_view_count": len(context_rows),
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
    args = parser.parse_args()
    try:
        summary = run(args.a1_dir.resolve(), args.data_dir.resolve(), args.output_dir.resolve())
    except (B1InputError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
