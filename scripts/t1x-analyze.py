#!/usr/bin/env python3
"""Summarize T1x dev runs: main table, paired text bootstrap, errors by ambiguity type."""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1x_lib as L  # noqa: E402

RUNS = L.ROOT / "work/experiments/v2026.11/t1x-v1/runs"
OUT = L.ROOT / "work/experiments/v2026.11/t1x-v1/analysis"
A1 = L.ROOT / "work/experiments/v2026.11/t1-dev-error-analysis-a1-v5/instances.csv"

MAIN = [
    ("поточечно, база (T1.1, C=7)", "point-base__C7"),
    ("посписочно, база", "list-base__C1"),
    ("+ исходное написание", "list-orig2__C3"),
    ("+ разложенный грамсет", "list-fact__C1"),
    ("+ кто сосед", "list-nb__C1"),
    ("+ словарная парадигма", "list-lex__C1"),
    ("кто сосед + разложенный", "list-nb-fact__C1"),
    ("всё вместе", "list-all2__C1"),
    ("кто сосед + подсказки (vep)", "list-nb-cues__C1"),
    ("кто сосед + однозначные родственники (w=0,1)", "list-nb-transferctx-w0.1__C1"),
    ("кто сосед + самообучение (p≥0,95, w=0,3)", "list-nb-self-p0.95-w0.3__C1"),
    ("кто сосед + хор (2 прохода)", "list-nb-chorus__C1"),
]
COMPARE = [
    ("посписочно vs поточечно", "point-base__C7", "list-base__C1"),
    ("кто сосед vs посписочная база", "list-base__C1", "list-nb__C1"),
    ("однозначные родственники vs кто сосед", "list-nb__C1", "list-nb-transferctx-w0.1__C1"),
    ("хор vs кто сосед", "list-nb__C1", "list-nb-chorus__C1"),
    ("самообучение vs кто сосед", "list-nb__C1", "list-nb-self-p0.95-w0.3__C1"),
    ("лучшая vs T1.1", "point-base__C7", "list-nb__C1"),
]


def load(name):
    return json.load(open(RUNS / name / "dev_rows.json"))


def pair_types():
    types = {}
    for lang in L.LANGS:
        path = L.ROOT / f"work/experiments/v2026.11/s1-pair-types-v1/{lang}/pair_types_auto.csv"
        with path.open(encoding="utf-8") as h:
            for r in csv.DictReader(h):
                types[(lang, frozenset((r["gramset_a"], r["gramset_b"])))] = r["type_proposed"]
    return types


def error_type(row, types):
    if row["full"]:
        return None
    if row["gold_gramset"] == row["top_gramset"]:
        return "6"
    return types.get((row["lang"], frozenset((row["gold_gramset"], row["top_gramset"]))), "?")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    types = pair_types()
    lines = ["# T1x — сводка по dev (2 613 употреблений; test не использовался)", ""]
    lines += ["| Конфигурация | Run | Полный грамсет | С точностью до синкретизма | MRR | vep | krl | olo | lud |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, name in MAIN:
        res = json.load(open(RUNS / name / "result.json"))["summary"]
        lines.append(f"| {label} | `{name}` | {res['all']['full']:.2%} | {res['all']['sync']:.2%} | "
                     f"{res['all']['mrr']:.3f} | " + " | ".join(f"{res[l]['full']:.1%}" for l in ("vep", "krl", "olo", "lud")) + " |")
    lines += ["", "## Парный bootstrap по текстам (2 000 повторов, 95% интервал разности, п.п.)", "",
              "| Сравнение | Полный грамсет | С точностью до синкретизма |", "|---|---|---|"]
    for label, a, b in COMPARE:
        ra, rb = load(a), load(b)
        cells = []
        for metric in ("full", "sync"):
            d, lo, hi = L.paired_bootstrap(ra, rb, metric)
            cells.append(f"{d*100:+.2f} [{lo*100:+.2f}; {hi*100:+.2f}]")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "## Ошибки по типу пары (gold, Top-1)", "",
              "Тип 1 — обязательный синкретизм; 2 — часть лемм; 3 — диалектный; 4 — разные леммы; "
              "5 — имя/глагол или нет данных словаря; 6 — тот же грамсет (написательные дубли/выбор леммы).", "",
              "| Конфигурация | ошибок | 1 | 2 | 3 | 4 | 5 | 6 | ? |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, name in MAIN:
        c = Counter(error_type(r, types) for r in load(name))
        errs = sum(v for k, v in c.items() if k)
        lines.append(f"| {label} | {errs} | " + " | ".join(str(c.get(t, 0)) for t in "123456?") + " |")
    # learning curves
    curves = defaultdict(list)
    for d in RUNS.glob("curve-*"):
        name = d.name.split("__")[0]
        cfg, frac = name.rsplit("-f", 1)[0].replace("curve-", ""), float(name.rsplit("-f", 1)[1].split("-s")[0])
        s = json.load(open(d / "result.json"))["summary"]["all"]
        curves[(cfg, frac)].append((s["full"], s["sync"]))
    lines += ["", "## Кривые обучения (доля текстов train; среднее по выборкам: 3 для <100%)", "",
              "| Конфигурация | 10% | 25% | 50% | 100% |", "|---|---|---|---|---|"]
    for cfg in ("list-base", "list-nb", "list-nb-cues", "list-nb-transfer"):
        cells = []
        for frac in (0.1, 0.25, 0.5, 1.0):
            v = curves.get((cfg, frac), [])
            if v:
                cells.append(f"{statistics.mean(x[0] for x in v):.2%} / {statistics.mean(x[1] for x in v):.2%}")
            else:
                cells.append("—")
        lines.append(f"| {cfg} (полный / синкр.) | " + " | ".join(cells) + " |")
    # agreement of reproduced pointwise baseline with frozen A1
    a1 = {}
    with A1.open(encoding="utf-8") as h:
        for r in csv.DictReader(h):
            a1[int(r["word_id"])] = (int(r["top1_wordform_id"]), r["top1_gramset"], r["is_correct"] == "1")
    rep = load("point-base__C7")
    same = sum(1 for r in rep if a1[r["word_id"]][:2] == (r["top_wordform"], r["top_gramset"]))
    lines += ["", "## Воспроизведение A1", "",
              f"`point-base__C7` (новый воспроизводимый запуск, модель сохранена) совпадает с замороженным A1 "
              f"по Top-1 в {same} из {len(rep)} употреблений; точность {sum(r['full'] for r in rep)} против "
              f"{sum(v[2] for v in a1.values())} верных."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
