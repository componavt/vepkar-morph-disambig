#!/usr/bin/env python3
"""Render the frozen B1-v7 context cards as one .xlsx workbook per stage.

The workbooks are meant to be uploaded to Google Drive and opened as Google
Sheets by the project lead; the script never talks to any online service.

Content is taken unchanged from the B1-v7 internal tables (targets, option
codes, context spans). Only the presentation differs from the Markdown mock-ups:
  * the context is shown in the original corpus spelling (case, punctuation,
    apostrophes) taken from ``sentence_xml`` instead of normalized tokens;
  * gramsets are shown with Russian short names in the order used by the
    VepKar interface (``Gramset::gramsetString`` in dictorpus).
Gold labels, model output, ranks, scores and Russian sentence translations are
never written to the expert workbooks.
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import re
import subprocess
import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.cell.rich_text import CellRichText, TextBlock
from openpyxl.cell.text import InlineFont
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/dictorpus-data"
B1_DIR = ROOT / "work/experiments/v2026.11/t1-context-cards-b1-v7"
DEFAULT_OUTPUT = ROOT / "work/experiments/v2026.11/t1-b1-sheets-v1"
QUESTIONNAIRE_VERSION = "B1-v7"
LEVELS = ("K0", "K1", "K2", "K3", "FULL")
LEVEL_TITLES = {
    "K0": "K0 — только словоформа",
    "K1": "K1 — по одному слову слева и справа",
    "K2": "K2 — по два слова слева и справа",
    "K3": "K3 — по три слова слева и справа",
    "FULL": "FULL — полное предложение",
}
# Order of grammatical categories in the VepKar interface (Gramset::gramsetString).
GRAMSET_ORDER = (
    "reflexive", "infinitive", "mood", "tense", "person",
    "case", "number", "negation", "voice", "participle",
)
STATUS_LABELS = (
    "Выбраны допустимые разборы",
    "Недостаточно контекста",
    "Ни один вариант не подходит",
    "Проблема текста или данных",
)
SEEN_EARLY_LABELS = ("нет", "да: K1", "да: K2", "да: K3", "да: полное предложение")
MAX_OPTIONS = 4
OPTION_CODES = "ABCD"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_zst_csv(path: Path) -> list[dict[str, str]]:
    raw = subprocess.run(["zstdcat", str(path)], check=True, capture_output=True, text=True).stdout
    return list(csv.DictReader(io.StringIO(raw)))


def gramset_labels() -> dict[str, str]:
    """Map gramset code (e.g. 'PL+PRS+3+IND+POS') to its Russian interface label."""
    grams = {row["gram_id"]: row for row in read_csv(DATA_DIR / "dict/grammar/grams.csv")}
    labels = {}
    for row in read_csv(DATA_DIR / "dict/grammar/gramsets.csv"):
        parts = []
        for category in GRAMSET_ORDER:
            gram_id = row.get(f"gram_id_{category}", r"\N")
            if gram_id in ("", r"\N"):
                continue
            gram = grams[gram_id]
            parts.append(gram["name_short_ru"] or gram["name_ru"])
        label = ", ".join(parts)
        if row["gramset_id"] == "282":
            label += " (крат. ф.)"
        labels[row["gramset"]] = label
    return labels


def sentence_tokens(sentence_xml: str) -> tuple[list[str], list[str]]:
    """Split sentence XML into word texts and the separators around them.

    Returns (words, gaps) with len(gaps) == len(words) + 1, so that the original
    sentence is gaps[0] + words[0] + gaps[1] + ... + words[-1] + gaps[-1].
    """
    body = re.sub(r"^<s[^>]*>|</s>$", "", sentence_xml.strip())
    words, gaps, position = [], [], 0
    for match in re.finditer(r"<w[^>]*>(.*?)</w>", body, flags=re.S):
        gaps.append(body[position:match.start()])
        words.append(match.group(1))
        position = match.end()
    gaps.append(body[position:])
    clean = lambda text: html.unescape(re.sub(r"<[^>]+>", "", text))  # noqa: E731
    return [clean(w) for w in words], [clean(g) for g in gaps]


def render_context(words, gaps, start, end, target, level) -> list[tuple[str, bool]]:
    """Return (text, is_target) pieces for the window [start, end] (token indices)."""
    if level == "K0":
        return [(words[target], True)]
    pieces: list[tuple[str, bool]] = []
    if start > 0:
        pieces.append(("… ", False))
    else:
        pieces.append((gaps[0].lstrip(), False))
    for index in range(start, end + 1):
        if index > start:
            pieces.append((gaps[index], False))
        pieces.append((f"[{words[index]}]" if index == target else words[index], index == target))
    if end < len(words) - 1:
        trailing = gaps[end + 1].rstrip()
        pieces.append((trailing + " …" if trailing.strip() else " …", False))
    else:
        pieces.append((gaps[-1].rstrip(), False))
    return [(text, flag) for text, flag in pieces if text]


def rich(pieces: list[tuple[str, bool]]) -> CellRichText:
    bold = InlineFont(b=True, color="C00000")
    return CellRichText(*[TextBlock(bold, text) if flag else text for text, flag in pieces])


def build_language(language: str, output: Path, labels: dict[str, str]) -> list[dict[str, str]]:
    targets = [
        row for row in read_csv(B1_DIR / "selected-targets-internal.csv")
        if row["language"] == language and row["is_primary_target"] == "1"
    ]
    views = {
        (row["target_id"], row["level"]): row
        for row in read_csv(B1_DIR / "context-views-internal.csv")
        if row["language"] == language
    }
    sentence_ids = {row["sentence_id"] for row in targets}
    sentences = {
        row["sentence_id"]: row["sentence_xml"]
        for row in read_zst_csv(DATA_DIR / f"corpus/sentences_{language}.csv.zst")
        if row["sentence_id"] in sentence_ids
    }
    manifest = []
    for level in LEVELS:
        wb = Workbook()
        intro = wb.active
        intro.title = "Инструкция"
        write_instructions(intro, language, level)
        sheet = wb.create_sheet("Карточки")
        header = ["№", "Карточка", "Словоформа", "Контекст"]
        for code in OPTION_CODES[:MAX_OPTIONS]:
            header += [f"Вариант {code}", f"{code} допустим?"]
        header += ["Статус ответа", "Видели более широкий контекст раньше?", "Комментарий (необязательно)"]
        sheet.append(header)
        for cell in sheet[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        sheet.freeze_panes = "E2"

        yes = DataValidation(type="list", formula1='"да"', allow_blank=True)
        status = DataValidation(type="list", formula1='"' + ",".join(STATUS_LABELS) + '"', allow_blank=True)
        seen = DataValidation(type="list", formula1='"' + ",".join(SEEN_EARLY_LABELS) + '"', allow_blank=True)
        for validation in (yes, status, seen):
            sheet.add_data_validation(validation)

        grey = PatternFill("solid", fgColor="E7E6E6")
        for number, target in enumerate(targets, start=1):
            view = views[(target["target_id"], level)]
            words, gaps = sentence_tokens(sentences[target["sentence_id"]])
            # K0 span is the 0-based index of the target inside its sentence.
            index = int(views[(target["target_id"], "K0")]["span_start"])
            if words[index].lower().replace("’", "").replace("'", "") != target["normalized_form"].replace("’", ""):
                # Normalization (apostrophes, w-harmony) may differ; keep a soft check only.
                print(f"note: {target['target_id']} original '{words[index]}' vs normalized '{target['normalized_form']}'", file=sys.stderr)
            options = [item.split(":", 1) for item in target["neutral_option_map_internal"].split(";")]
            row_index = number + 1
            sheet.cell(row_index, 1, number)
            sheet.cell(row_index, 2, target["target_id"])
            sheet.cell(row_index, 3, words[index])
            repeated = view["requires_new_answer"] == "0"
            if repeated:
                sheet.cell(row_index, 4, f"Контекст не изменился по сравнению с этапом {view['repeat_of_level']}. Отвечать повторно не нужно.")
            else:
                pieces = render_context(words, gaps, int(view["span_start"]), int(view["span_end"]), index, level)
                sheet.cell(row_index, 4).value = rich(pieces)
            for slot in range(MAX_OPTIONS):
                label_cell = sheet.cell(row_index, 5 + 2 * slot)
                answer_cell = sheet.cell(row_index, 6 + 2 * slot)
                if slot < len(options):
                    code, gramset = options[slot]
                    label_cell.value = f"{code}. {words[index]} — {labels[gramset]}"
                    if not repeated:
                        yes.add(answer_cell)
                    manifest.append({
                        "language": language, "level": level, "target_id": target["target_id"],
                        "option_code": code, "gramset": gramset, "label_ru": labels[gramset],
                    })
                else:
                    label_cell.fill = grey
                    answer_cell.fill = grey
            last = 5 + 2 * MAX_OPTIONS
            if repeated:
                for column in range(5, last + 3):
                    sheet.cell(row_index, column).fill = grey
            else:
                status.add(sheet.cell(row_index, last))
                seen.add(sheet.cell(row_index, last + 1))
            for column in range(1, last + 3):
                sheet.cell(row_index, column).alignment = Alignment(wrap_text=True, vertical="top")
        widths = [5, 17, 14, 60] + [26, 10] * MAX_OPTIONS + [26, 22, 40]
        for column, width in enumerate(widths, start=1):
            sheet.column_dimensions[sheet.cell(1, column).column_letter].width = width
        path = output / f"{QUESTIONNAIRE_VERSION}-{language}-{level}.xlsx"
        wb.save(path)
        print(f"wrote {path.relative_to(ROOT)}")
    return manifest


def write_instructions(sheet, language: str, level: str) -> None:
    lines = [
        (f"Контекстный опросник {QUESTIONNAIRE_VERSION} · язык: {language} · этап {LEVEL_TITLES[level]}", True),
        ("", False),
        ("Эксперт:", True),
        ("Начало работы над этапом (дата и время):", True),
        ("Окончание работы над этапом (дата и время):", True),
        ("", False),
        ("Что нужно сделать", True),
        ("В листе «Карточки» каждая строка — одна словоформа из текстов ВепКар и предложенные для неё грамматические разборы.", False),
        ("Рассмотрите словоформу в показанном контексте (выделена и заключена в квадратные скобки).", False),
        ("Отметьте «да» в столбце рядом с каждым разбором, который вы считаете допустимым. Можно отметить несколько.", False),
        ("В столбце «Статус ответа» выберите ровно один статус:", False),
        ("  • «Выбраны допустимые разборы» — если отмечен хотя бы один разбор;", False),
        ("  • «Недостаточно контекста» — по показанному фрагменту нельзя решить, какие разборы допустимы;", False),
        ("  • «Ни один вариант не подходит» — в списке нет подходящего разбора;", False),
        ("  • «Проблема текста или данных» — ошибка в словоформе, контексте или карточке (поясните в комментарии).", False),
        ("Комментарий необязателен.", False),
        ("", False),
        ("Порядок работы", True),
        ("Материал выдаётся по этапам: сначала только словоформа, затем всё более широкий контекст, в конце — полное предложение.", False),
        ("Заполняйте только текущий этап. После сдачи этапа ответы не меняются: если мнение изменилось, укажите это на следующем этапе.", False),
        ("Если вы уже видели более широкий контекст (например, узнали текст), отметьте это в столбце «Видели более широкий контекст раньше?».", False),
        ("Серые строки повторяют уже показанный контекст — отвечать на них не нужно.", False),
        ("Не ищите исходное предложение во внешних источниках во время заполнения.", False),
        ("Не нужно угадывать ответ программы или прежнюю разметку корпуса — нас интересует ваше суждение.", False),
        ("Текст показан в исходном написании корпуса ВепКар.", False),
    ]
    for row, (text, bold) in enumerate(lines, start=1):
        cell = sheet.cell(row, 1, text)
        cell.font = Font(bold=bold, size=12 if row == 1 else 11)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    sheet.column_dimensions["A"].width = 110
    for row in (3, 4, 5):
        sheet.cell(row, 2).fill = PatternFill("solid", fgColor="FFF2CC")
    sheet.column_dimensions["B"].width = 30


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--language", default="vep", choices=("vep", "krl", "olo"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output / args.language
    if output.exists() and any(output.iterdir()):
        sys.exit(f"Output directory is not empty, refusing to overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = build_language(args.language, output, gramset_labels())
    with (output / "option-labels-internal.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"wrote {(output / 'option-labels-internal.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
