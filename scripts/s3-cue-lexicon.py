#!/usr/bin/env python3
"""S3: draft cue lexicon for Veps INF vs PASS/3PL and its reliability on annotated data.

Cues are taken from the literature (Kosheleva 2023: governors of the I infinitive),
not tuned on data. A cue is "present" when a word within +-WINDOW of the target has
a candidate analysis whose lemma is in the cue lexicon. Reliability is measured on
annotated benchmark tokens of train and dev; test texts are never read (the cached
corpus contains train/dev items only). Frequencies on unannotated tokens are given
for texts outside dev and test.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1x_lib as L  # noqa: E402

OUT = L.ROOT / "work/experiments/v2026.11/s3-cue-lexicon-v1/vep"
WINDOW = 3
# (class, lemma, Russian gloss, lemma_ids in lemmas_vep) -- Kosheleva 2023, I infinitive
CUES = [
    ("deontic", "pidada/pidäda", "надо (pidab)", ("964", "41455", "439")),
    ("deontic", "tarbiž", "нужно", ("55",)),
    ("deontic", "tariž", "нужно", ("43498",)),
    ("modal", "voida", "мочь", ("146",)),
    ("modal", "sada", "мочь, получать", ("209",)),
    ("modal", "mahtta", "уметь", ("545",)),
    ("governing", "zavodida", "начинать", ("39",)),
    ("governing", "antta", "давать", ("17",)),
    ("governing", "käskta", "приказывать", ("959",)),
    ("governing", "ehtta", "успевать", ("1446",)),
    ("governing", "jätta", "оставлять", ("254",)),
    ("governing", "tahtoida/tahtta/tahtoda", "хотеть", ("195", "43401", "77686")),
    ("noun", "aig", "время", ("239",)),
    ("noun", "taht", "желание", ("1071",)),
    ("noun", "mel’/mel", "мысль, охота", ("173", "29622")),
    ("noun", "ofot", "охота, желание", ("1239",)),
]
CUE_LEMMA = {lid: (cls, lemma) for cls, lemma, _, ids in CUES for lid in ids}
INF, PASS, PL3 = "INF", "PRS+PASS", "PL+PRS+3+IND+POS"


def cue_hits(corpus, lang, sentence_id, pos):
    toks = corpus.sentences[(lang, sentence_id)]
    hits = []
    for off in range(-WINDOW, WINDOW + 1):
        j = pos + off
        if off == 0 or not 0 <= j < len(toks):
            continue
        for wf, gid, g, _ in corpus.cands.get((lang, toks[j].word_id), []):
            for lemma in corpus.lemmas.get((lang, wf, gid), ()):
                if lemma in CUE_LEMMA:
                    hits.append((off, *CUE_LEMMA[lemma]))
    return hits


def main():
    corpus = pickle.load(open(L.ROOT / "work/experiments/v2026.11/t1x-v1/corpus.pkl", "rb"))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "cue_lexicon_vep.csv").open("w", encoding="utf-8", newline="") as h:
        w = csv.writer(h)
        w.writerow(["cue_class", "lemma", "gloss_ru", "lemma_ids", "conflict", "predicted_gramset",
                    "window", "source", "status"])
        for cls, lemma, gloss, ids in CUES:
            w.writerow([cls, lemma, gloss, "|".join(ids), "INF vs PRS+PASS / PL+PRS+3", INF,
                        f"±{WINDOW}", "Кошелева 2023 (функции I инфинитива)", "draft"])
    # reliability on annotated benchmark tokens
    table = Counter()
    by_class = Counter()
    for split, items in (("train", corpus.train), ("dev", corpus.dev)):
        for it in items:
            gs = {c[2] for c in it.candidates}
            if it.lang != "vep" or INF not in gs or not ({PASS, PL3} & gs):
                continue
            gold = it.candidates[it.gold][2]
            hits = cue_hits(corpus, it.lang, it.sentence_id, it.pos)
            present = bool(hits)
            table[(split, present, gold == INF)] += 1
            for cls in {h[1] for h in hits}:
                by_class[(cls, gold == INF)] += 1
    # frequency on unannotated ambiguous tokens (allowed texts)
    unann = Counter()
    for it in L.unannotated_items(corpus):
        gs = {c[2] for c in it.candidates}
        if it.lang == "vep" and INF in gs and ({PASS, PL3} & gs):
            unann[bool(cue_hits(corpus, it.lang, it.sentence_id, it.pos))] += 1
    summary = {"window": WINDOW, "annotated": {}, "by_class": {}, "unannotated_inf_pass_tokens": {
        "with_cue": unann[True], "without_cue": unann[False]}}
    for split in ("train", "dev"):
        for present in (True, False):
            n_inf, n_other = table[(split, present, True)], table[(split, present, False)]
            n = n_inf + n_other
            summary["annotated"][f"{split}/{'cue' if present else 'no_cue'}"] = {
                "tokens": n, "gold_INF": n_inf, "share_INF": round(n_inf / n, 3) if n else None}
    for cls in sorted({k[0] for k in by_class}):
        a, b = by_class[(cls, True)], by_class[(cls, False)]
        summary["by_class"][cls] = {"tokens": a + b, "gold_INF": a, "share_INF": round(a / (a + b), 3)}
    (OUT / "reliability.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
