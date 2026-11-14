#!/usr/bin/env python3
"""D: decision transitions between context windows (C=7, saved dev predictions), by ambiguity type.

Windows: none -> +-1 -> +-2 -> +-3 (cumulative). For each benchmark token the
"conflict type" is the senior S1 type of its candidate gramset set (1 obligatory,
2 lexically conditioned, 3 dialectal, 4 cross-lemma, 5 noun/verb or no lexicon,
6 one gramset only). For each window: accuracy (full gramset and modulo
syncretism), and transitions from the previous window (error->correct,
correct->error).
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1x_lib as L  # noqa: E402

BASE = L.ROOT / "work/experiments/v2026.11"
WINDOWS = [
    ("none", BASE / "context-ablation-k5-v1/runs/no_context/c-7/dev.csv"),
    ("±1", BASE / "context-ablation-k5-v1/runs/both/c-7/dev.csv"),
    ("±2", BASE / "context-window-sweep-k5-v1/runs/window_2/c-7/dev.csv"),
    ("±3", BASE / "context-window-sweep-k5-v1/runs/window_3/c-7/dev.csv"),
]
OUT = BASE / "d-context-transitions-v1"


def top1(path):
    best = {}
    with path.open(encoding="utf-8") as h:
        for r in csv.DictReader(h):
            if r["rank"] == "1":
                best[int(r["word_id"])] = (int(r["wordform_id"]), r["gramset"])
    return best


def main():
    corpus = pickle.load(open(BASE / "t1x-v1/corpus.pkl", "rb"))
    pt = {}
    for lang in L.LANGS:
        with (BASE / f"s1-pair-types-v1/{lang}/pair_types_auto.csv").open(encoding="utf-8") as h:
            for r in csv.DictReader(h):
                pt[(lang, frozenset((r["gramset_a"], r["gramset_b"])))] = r["type_proposed"]
    tops = {name: top1(path) for name, path in WINDOWS}
    ctype = {}
    for it in corpus.dev:
        gs = sorted({c[2] for c in it.candidates})
        types = {pt.get((it.lang, frozenset(p)), "?") for p in combinations(gs, 2)}
        ctype[it.word_id] = min(types) if types else "6"
    result = defaultdict(dict)
    rows = []
    for it in corpus.dev:
        status = {}
        for name, _ in WINDOWS:
            wf, g = tops[name][it.word_id]
            top = next(i for i, c in enumerate(it.candidates) if (c[0], c[2]) == (wf, g))
            status[name] = (int(top == it.gold), int(L.correct_modulo_syncretism(corpus, it, top)))
        rows.append((it, status))
    lines = ["# D — переходы решений между окнами контекста (dev, C=7, сохранённые predictions)", "",
             "Тип — основной тип конфликта в списке кандидатов (S1). Ячейка: точность по полному грамсету; "
             "в скобках — исправлено/испорчено относительно предыдущего окна.", "",
             "| Тип | N | без контекста | ±1 | ±2 | ±3 |", "|---|---:|---:|---:|---:|---:|"]
    for ty in "123456":
        sub = [(it, st) for it, st in rows if ctype[it.word_id] == ty]
        if not sub:
            continue
        cells = []
        prev = None
        for name, _ in WINDOWS:
            acc = sum(st[name][0] for _, st in sub) / len(sub)
            if prev is None:
                cells.append(f"{acc:.1%}")
            else:
                fix = sum(1 for _, st in sub if not st[prev][0] and st[name][0])
                hurt = sum(1 for _, st in sub if st[prev][0] and not st[name][0])
                cells.append(f"{acc:.1%} (+{fix}/−{hurt})")
            result[ty][name] = acc
            prev = name
        lines.append(f"| {ty} | {len(sub)} | " + " | ".join(cells) + " |")
    lines += ["", "С точностью до синкретизма:", "", "| Тип | N | без контекста | ±1 | ±2 | ±3 |", "|---|---:|---:|---:|---:|---:|"]
    for ty in "123456":
        sub = [(it, st) for it, st in rows if ctype[it.word_id] == ty]
        if sub:
            lines.append(f"| {ty} | {len(sub)} | " + " | ".join(
                f"{sum(st[n][1] for _, st in sub) / len(sub):.1%}" for n, _ in WINDOWS) + " |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (OUT / "token-transitions.csv").open("w", encoding="utf-8", newline="") as h:
        w = csv.writer(h)
        w.writerow(["language", "word_id", "conflict_type"] + [f"correct_{n}" for n, _ in WINDOWS]
                   + [f"top1_{n}" for n, _ in WINDOWS])
        for it, st in rows:
            w.writerow([it.lang, it.word_id, ctype[it.word_id]] + [st[n][0] for n, _ in WINDOWS]
                       + [f"{tops[n][it.word_id][0]}:{tops[n][it.word_id][1]}" for n, _ in WINDOWS])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
