#!/usr/bin/env python3
"""How experts resolve NOM/ACC homonymy across the whole corpus (test texts excluded).

For every word with an expert-confirmed analysis (relevance 2) whose candidate list
contains the same wordform as both NOM and ACC (of one number), count which case the
expert chose. Confirmed ACC words are also split by form: equal to NOM (zero ending)
or equal to GEN (ending -n). Texts of the test split are skipped; texts outside the
split file (no benchmark words) are included.

Output: work/experiments/<tag>/acc-nom-counts-v1/counts.csv (+ printed table).
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1x_lib as L  # noqa: E402

OUT = L.EXP / "acc-nom-counts-v1"


def main():
    split = {}
    with open(L.ROOT / "data/derived/splits_v2026.13.csv") as h:
        for r in csv.DictReader(h):
            split[(r["language"], int(r["text_id"]))] = r["split"]
    rows_out = []
    for lang in L.LANGS:
        sent_text = {int(r["sentence_id"]): int(r["text_id"])
                     for r in L.zst_rows(L.DATA_DIR / f"corpus/sentences_{lang}.csv.zst")}
        word_text = {int(r["word_id"]): sent_text.get(int(r["sentence_id"]))
                     for r in L.zst_rows(L.DATA_DIR / f"corpus/words_{lang}.csv.zst")}
        cands = defaultdict(list)
        for r in L.zst_rows(L.DATA_DIR / f"corpus/candidate_analyses_{lang}.csv.zst"):
            cands[int(r["word_id"])].append((r["wordform_id"], r["gramset"], r["relevance"]))
        c = Counter()
        for wid, rows in cands.items():
            text = word_text.get(wid)
            if text is None or split.get((lang, text)) == "test":
                continue
            gold = [(wf, g) for wf, g, rel in rows if rel == "2"]
            if len(gold) != 1:
                continue
            wf, g = gold[0]
            same = {gg for w, gg, _ in rows if w == wf}
            for num in ("SG", "PL"):
                nom, acc, gen = f"{num}+NOM", f"{num}+ACC", f"{num}+GEN"
                if g == acc:
                    c[(num, "acc_total")] += 1
                    c[(num, "acc_eq_nom")] += nom in same
                    c[(num, "acc_eq_gen")] += gen in same
                if g in (nom, acc) and nom in same and acc in same:
                    c[(num, "conflict")] += 1
                    c[(num, "conflict_acc")] += g == acc
        for num in ("SG", "PL"):
            row = {"lang": lang, "number": num, **{k: c[(num, k)] for k in
                   ("acc_total", "acc_eq_nom", "acc_eq_gen", "conflict", "conflict_acc")}}
            row["acc_share_in_conflict"] = round(row["conflict_acc"] / row["conflict"], 3) if row["conflict"] else ""
            rows_out.append(row)
            print(row, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "counts.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)


if __name__ == "__main__":
    main()
