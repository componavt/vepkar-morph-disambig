#!/usr/bin/env python3
"""S2: ambiguity map of a language's corpus by syncretism type, dialect, subcorpus.

Population: tokens with >= 2 distinct (wordform_id, gramset) candidates and no
empty gramset, in all texts except the test split. Types come from S1 pair types.
Outputs (work/experiments/v2026.11/s2-ambiguity-map-v1/<lang>/):
  map.json                  population, Q1-Q4 summaries
  type_by_dialect.csv       tokens per (senior type, dialect), all vs annotated
  ambiguity_classes.csv     the 30 most frequent candidate gramset sets
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/dictorpus-data"


def zst_rows(path):
    raw = subprocess.run(["zstdcat", str(path)], check=True, capture_output=True, text=True).stdout
    return csv.DictReader(io.StringIO(raw))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--language", default="vep")
    lang = p.parse_args().language
    out = ROOT / f"work/experiments/v2026.11/s2-ambiguity-map-v1/{lang}"
    out.mkdir(parents=True, exist_ok=True)
    with (ROOT / f"work/experiments/v2026.11/s1-pair-types-v1/{lang}/pair_types_auto.csv").open(encoding="utf-8") as h:
        ptype = {frozenset((r["gramset_a"], r["gramset_b"])): r["type_proposed"] for r in csv.DictReader(h)}
    with (ROOT / "data/derived/splits_v2026.11.csv").open(encoding="utf-8") as h:
        test = {r["text_id"] for r in csv.DictReader(h) if r["language"] == lang and r["split"] == "test"}
    texts = {r["text_id"]: r for r in zst_rows(DATA / f"corpus/texts_{lang}.csv.zst")}
    sent_text = {r["sentence_id"]: r["text_id"] for r in zst_rows(DATA / f"corpus/sentences_{lang}.csv.zst")}
    word_text = {r["word_id"]: sent_text.get(r["sentence_id"]) for r in zst_rows(DATA / f"corpus/words_{lang}.csv.zst")}
    cands = defaultdict(list)
    for r in zst_rows(DATA / f"corpus/candidate_analyses_{lang}.csv.zst"):
        cands[r["word_id"]].append(r)

    pop = Counter()
    by_dialect = Counter()            # (type, dialect, annotated)
    by_corpus = Counter()
    multi = Counter()                 # multi-label: each type the token has
    classes = Counter()
    pl3_pass = Counter()
    for wid, rows in cands.items():
        t = word_text.get(wid)
        if t is None or t in test:
            continue
        if any(r["gramset_id"] in ("0", "7") or not r["gramset"] for r in rows):
            continue
        keys = {(r["wordform_id"], r["gramset"]) for r in rows}
        if len(keys) < 2:
            pop["unambiguous_or_single"] += 1
            continue
        gs = sorted({r["gramset"] for r in rows})
        types = {ptype.get(frozenset(p), "?") for p in combinations(gs, 2)}
        senior = min(types) if types else "6"
        annotated = any(r["relevance"] == "2" for r in rows)
        dialect = texts.get(t, {}).get("dialect_code") or "none"
        corpus = texts.get(t, {}).get("corpus_id") or "none"
        pop["ambiguous"] += 1
        pop["annotated" if annotated else "automatic"] += 1
        by_dialect[(senior, dialect, annotated)] += 1
        by_corpus[(corpus, annotated)] += 1
        for ty in (types or {"6"}):
            multi[(ty, annotated)] += 1
        classes[" | ".join(gs)] += 1
        if "PL+PRS+3+IND+POS" in gs and "PRS+PASS" in gs:
            pl3_pass[dialect] += 1

    dialects = sorted({k[1] for k in by_dialect})
    with (out / "type_by_dialect.csv").open("w", encoding="utf-8", newline="") as h:
        w = csv.writer(h)
        w.writerow(["senior_type", "dialect", "tokens_all", "tokens_annotated"])
        for ty in "123456?":
            for d in dialects:
                a = by_dialect[(ty, d, True)] + by_dialect[(ty, d, False)]
                if a:
                    w.writerow([ty, d, a, by_dialect[(ty, d, True)]])
    with (out / "ambiguity_classes.csv").open("w", encoding="utf-8", newline="") as h:
        w = csv.writer(h)
        w.writerow(["candidate_gramsets", "tokens"])
        w.writerows(classes.most_common(30))

    n_all, n_ann = pop["ambiguous"], pop["annotated"]

    def share(counter_fn):
        return {ty: {"all": round(counter_fn(ty, None) / n_all, 4), "annotated": round(counter_fn(ty, True) / n_ann, 4)}
                for ty in "123456?"}

    senior_share = share(lambda ty, ann: sum(v for (t, d, a), v in by_dialect.items()
                                             if t == ty and (ann is None or a == ann)))
    multi_share = share(lambda ty, ann: (multi[(ty, True)] + multi[(ty, False)]) if ann is None else multi[(ty, True)])
    dial_share = {d: {"all": round(sum(v for (t, dd, a), v in by_dialect.items() if dd == d) / n_all, 4),
                      "annotated": round(sum(v for (t, dd, a), v in by_dialect.items() if dd == d and a) / n_ann, 4)}
                  for d in dialects}
    corp_share = {c: {"all": round((by_corpus[(c, True)] + by_corpus[(c, False)]) / n_all, 4),
                      "annotated": round(by_corpus[(c, True)] / n_ann, 4)}
                  for c in sorted({k[0] for k in by_corpus})}
    tv_dialect = 0.5 * sum(abs(v["all"] - v["annotated"]) for v in dial_share.values())
    tv_type = 0.5 * sum(abs(v["all"] - v["annotated"]) for v in senior_share.values())
    summary = {
        "language": lang, "population": dict(pop),
        "Q1_senior_type_share": senior_share, "Q1_multilabel_type_share": multi_share,
        "Q2_pl3_pass_by_dialect": dict(pl3_pass.most_common()),
        "Q3_dialect_share_all_vs_annotated": dial_share, "Q3_corpus_share_all_vs_annotated": corp_share,
        "Q3_total_variation": {"dialect": round(tv_dialect, 3), "senior_type": round(tv_type, 3)},
        "Q4_top_classes": classes.most_common(10),
    }
    (out / "map.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("population", "Q1_senior_type_share", "Q2_pl3_pass_by_dialect",
                                              "Q3_total_variation")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
