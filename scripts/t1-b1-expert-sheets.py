#!/usr/bin/env python3
"""Render frozen B1 context cards as one .xlsx workbook per stage.

The workbooks are meant to be uploaded to Google Drive and opened as Google
Sheets by the project lead; the script never talks to any online service.

Content is taken unchanged from the B1 internal selection tables (targets, option
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
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/dictorpus-data"
DEFAULT_B1_DIR = ROOT / "work/experiments/v2026.11/t1-context-cards-b1-v8"
DEFAULT_OUTPUT = ROOT / "work/experiments/v2026.11/t1-b1-sheets-v2"
DEFAULT_VERSION = "B1-v8"
LISTS_SHEET = "Списки"
# Per-option answer; row-level statuses are derived in analysis, not asked:
#   any "да" -> analyses_selected; any "не ясно" -> insufficient context for that option;
#   all "нет" -> none_applicable; data problems go to the comment.
OPTION_ANSWERS = ("да", "нет", "не ясно")
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
# Maria annotated most Veps texts herself, so the realistic "earlier context" is recognition.
RECOGNIZED_LABELS = ("нет", "да")
OPTION_CODES = "ABCDEFGH"
# Muted fills: answer colours (conditional formatting) and the "nothing to do" row tint.
ANSWER_FILLS = {"да": "D9EAD3", "нет": "F4CCCC", "не ясно": "FFF2CC"}
UNUSED_FILL = "E7E6E6"
REPEATED_FILL = "EEF3F8"


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


def list_range(column: str, values) -> str:
    """Absolute reference to a value list on the lists sheet (rows 2..n+1)."""
    return f"'{LISTS_SHEET}'!${column}$2:${column}${len(values) + 1}"


def write_lists(wb) -> None:
    """Hidden sheet holding dropdown values; validations refer to its ranges."""
    sheet = wb.create_sheet(LISTS_SHEET)
    for column, (title, values) in enumerate(
        (("Допустим?", OPTION_ANSWERS), ("Узнаёте?", RECOGNIZED_LABELS)), start=1
    ):
        sheet.cell(1, column, title)
        for row, value in enumerate(values, start=2):
            sheet.cell(row, column, value)
    sheet.sheet_state = "hidden"


def recognition_question(level: str) -> str:
    """At FULL the whole sentence is visible, so only the wider text can be recognized."""
    if level == "FULL":
        return "Узнаёте текст, из которого взято это предложение?"
    return "Узнаёте это предложение или текст?"


def build_language(language: str, output: Path, labels: dict[str, str], b1_dir: Path, version: str, levels=LEVELS) -> list[dict[str, str]]:
    targets = [
        row for row in read_csv(b1_dir / "selected-targets-internal.csv")
        if row["language"] == language and row["is_primary_target"] == "1"
    ]
    views = {
        (row["target_id"], row["level"]): row
        for row in read_csv(b1_dir / "context-views-internal.csv")
        if row["language"] == language
    }
    sentence_ids = {row["sentence_id"] for row in targets}
    sentences = {
        row["sentence_id"]: row["sentence_xml"]
        for row in read_zst_csv(DATA_DIR / f"corpus/sentences_{language}.csv.zst")
        if row["sentence_id"] in sentence_ids
    }
    max_options = max(len(t["neutral_option_map_internal"].split(";")) for t in targets)
    manifest = []
    for level in levels:
        path = output / f"{version}-{language}-{level}.xlsx"
        if path.exists():
            sys.exit(f"Refusing to overwrite an existing (possibly issued) workbook: {path}")
        wb = Workbook()
        intro = wb.active
        intro.title = "Инструкция"
        write_instructions(intro, language, level, version)
        sheet = wb.create_sheet("Карточки")
        header = ["№", "Карточка", "Словоформа", "Контекст"]
        for code in OPTION_CODES[:max_options]:
            header += [f"Вариант {code}", f"{code} допустим?"]
        header += [recognition_question(level), "Комментарий (необязательно)"]
        sheet.append(header)
        for cell in sheet[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        sheet.freeze_panes = "E2"

        write_lists(wb)
        yes = DataValidation(type="list", formula1=list_range("A", OPTION_ANSWERS), allow_blank=True)
        seen = DataValidation(type="list", formula1=list_range("B", RECOGNIZED_LABELS), allow_blank=True)
        for validation in (yes, seen):
            validation.showErrorMessage = True
            validation.error = "Выберите значение из списка"
        for validation in (yes, seen):
            sheet.add_data_validation(validation)

        grey = PatternFill("solid", fgColor=UNUSED_FILL)
        calm = PatternFill("solid", fgColor=REPEATED_FILL)
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
            for slot in range(max_options):
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
            last = 5 + 2 * max_options
            if repeated:
                # Whole row in a calm tint: the expert can skip it.
                for column in range(1, last + 2):
                    sheet.cell(row_index, column).fill = calm
            else:
                seen.add(sheet.cell(row_index, last))
            for column in range(1, last + 2):
                sheet.cell(row_index, column).alignment = Alignment(wrap_text=True, vertical="top")
        # Answer colours via conditional formatting (kept by Excel and Google Sheets import).
        answer_range = " ".join(
            f"{sheet.cell(2, 6 + 2 * slot).column_letter}2:{sheet.cell(2, 6 + 2 * slot).column_letter}{len(targets) + 1}"
            for slot in range(max_options)
        )
        for value, colour in ANSWER_FILLS.items():
            sheet.conditional_formatting.add(
                answer_range,
                CellIsRule(operator="equal", formula=[f'"{value}"'], fill=PatternFill("solid", fgColor=colour, bgColor=colour)),
            )
        # Column widths from content (characters), within sensible bounds.
        for column in range(1, last + 2):
            letter = sheet.cell(1, column).column_letter
            longest = max(
                len(str(sheet.cell(row, column).value or "")) for row in range(2, len(targets) + 2)
            )
            header_words = max(len(w) for w in str(sheet.cell(1, column).value).split())
            if column == last + 1:  # comment: room to write
                width = 40
            elif column == 4 and level != "K0":  # context sentence
                width = min(80, max(30, longest + 2))
            else:
                width = max(header_words + 2, min(45, longest + 2))
            sheet.column_dimensions[letter].width = width
        path = output / f"{version}-{language}-{level}.xlsx"
        wb.save(path)
        print(f"wrote {path.relative_to(ROOT)}")
    return manifest


def write_instructions(sheet, language: str, level: str, version: str) -> None:
    lines = [
        (f"Контекстный опросник {version} · язык: {language} · этап {LEVEL_TITLES[level]}", True),
        ("", False),
        ("Эксперт:", True),
        ("Начало работы над этапом (дата и время):", True),
        ("Окончание работы над этапом (дата и время):", True),
        ("", False),
        ("Что нужно сделать", True),
        ("В листе «Карточки» каждая строка — одна словоформа из текстов ВепКар и предложенные для неё грамматические разборы.", False),
        ("Рассмотрите словоформу в показанном контексте (выделена и заключена в квадратные скобки).", False),
        ("Для каждого разбора выберите в соседнем столбце одно значение:", False),
        ("  • «да» — разбор допустим в показанном контексте (допустимыми можно считать несколько разборов);", False),
        ("  • «нет» — разбор недопустим;", False),
        ("  • «не ясно» — по показанному фрагменту нельзя решить, допустим ли этот разбор.", False),
        ("Если ни один разбор не подходит, отметьте все «нет» и, по желанию, напишите в комментарии, какой разбор нужен.", False),
        ("Если вы заметили ошибку в словоформе, тексте или карточке, опишите её в комментарии.", False),
        ("Комментарий необязателен. Пустые ячейки не считаются ответом «нет».", False),
        ("", False),
        ("Порядок работы", True),
        ("Материал выдаётся по этапам: сначала только словоформа, затем всё более широкий контекст, в конце — полное предложение.", False),
        ("Заполняйте только текущий этап. После сдачи этапа ответы не меняются: если мнение изменилось, укажите это на следующем этапе.", False),
        (("Если вы узнаёте текст, из которого взято предложение (например, размечали его раньше), отметьте «да» в столбце «" + recognition_question(level) + "». Это не ошибка — нам важно это знать.") if level == "FULL" else ("Если вы узнаёте предложение или текст (например, размечали его раньше), отметьте «да» в столбце «" + recognition_question(level) + "». Это не ошибка — нам важно это знать."), False),
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
    parser.add_argument("--b1-dir", type=Path, default=DEFAULT_B1_DIR, help="selection directory (contains <language>/)")
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--levels", nargs="+", default=list(LEVELS), choices=LEVELS,
                        help="stages to (re)build; existing workbooks are never overwritten")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output / args.language
    output.mkdir(parents=True, exist_ok=True)
    b1_dir = args.b1_dir / args.language if (args.b1_dir / args.language).is_dir() else args.b1_dir
    manifest = build_language(args.language, output, gramset_labels(), b1_dir, args.version, tuple(args.levels))
    manifest_path = output / ("option-labels-internal.csv" if len(args.levels) == len(LEVELS) else f"option-labels-internal-{'-'.join(args.levels)}.csv")
    if manifest_path.exists():
        sys.exit(f"Refusing to overwrite {manifest_path}")
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"wrote {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
