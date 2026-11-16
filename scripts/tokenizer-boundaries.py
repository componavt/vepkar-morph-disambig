#!/usr/bin/env python3
"""Do subword cuts follow the VepKar stem|affix boundary, and does it matter? (dev only)

The boundary comes from the dictionary export: ``affix`` is the remainder of the
wordform after the stem stored for its lemma (paradigm generator). For each dev
target word we take the expert's (gold) candidate, place the boundary at
len(word) - len(affix) and classify the tokenizer's segmentation:

  whole     the word is a single subword
  aligned   one of the internal cuts falls exactly on the stem|affix boundary
  crossing  there are internal cuts, none on the boundary
  zero      empty affix (word = stem); reported separately

Accuracy of seed ensembles (Glot500, N1 BiLSTM) and of the linear model is then
compared per class; N1 reads characters and the linear model reads suffixes, so they
serve as a control for "harder words" versus "worse segmentation". Train frequency of
the word form is reported because frequent words tend to be single subwords.

Output: work/experiments/<tag>/tokenizer-boundaries-v1/{words.csv,summary.csv}
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nn_lib as N  # noqa: E402
import t1x_lib as L  # noqa: E402

OUT = L.EXP / "tokenizer-boundaries-v1"
ENCODERS = {"glot500": "cis-lmu/glot500-base", "xlmr": "FacebookAI/xlm-roberta-base",
            "finbert": "TurkuNLP/bert-base-finnish-cased-v1"}
SYSTEMS = {"glot500-ens5": "hybrid-v1/n2-ens5", "n1-ens5": "hybrid-v1/n1-ens5",
           "list-nb": "t1x-v1/runs/list-nb__C1"}
APOS = str.maketrans("", "", "'’ʼ`´")


def clean(s):
    return s.lower().translate(APOS)


def cut_offsets(tokenizer, word):
    pieces = tokenizer.tokenize(word)
    lengths = []
    for p in pieces:
        p = p.replace("▁", "").replace("##", "")
        lengths.append(len(p))
    if sum(lengths) != len(word) or tokenizer.unk_token in pieces:
        return None, len(pieces)
    cuts, pos = set(), 0
    for n in lengths[:-1]:
        pos += n
        if pos:
            cuts.add(pos)
    cuts.discard(0)
    cuts.discard(len(word))
    return cuts, len(pieces)


def main():
    from transformers import AutoTokenizer
    corpus = N.load_corpus()
    affix = {}
    for lang in L.LANGS:
        for r in L.zst_rows(L.DATA_DIR / f"dict/morphology/lemma_wordforms_{lang}.csv.zst"):
            a = r["affix"]
            if a not in ("#", r"\N"):
                affix[(lang, int(r["wordform_id"]), r["gramset_id"])] = clean(a)
    train_freq = Counter()
    for it in corpus.train:
        train_freq[(it.lang, clean(corpus.sentences[(it.lang, it.sentence_id)][it.pos].norm))] += 1
    rows = {}
    for name, path in SYSTEMS.items():
        for r in json.load(open(L.EXP / path / "dev_rows.json")):
            rows.setdefault((r["lang"], r["word_id"]), {})[name] = r["full"]
    toks = {k: AutoTokenizer.from_pretrained(m) for k, m in ENCODERS.items()}
    words = []
    for it in corpus.dev:
        word = clean(corpus.sentences[(it.lang, it.sentence_id)][it.pos].norm)
        wf, gid, g = it.candidates[it.gold]
        a = affix.get((it.lang, wf, gid))
        rec = {"lang": it.lang, "word_id": it.word_id, "word": word, "gold": g,
               "train_freq": train_freq[(it.lang, word)], **rows.get((it.lang, it.word_id), {})}
        if a is None or not word.endswith(a):
            rec["boundary"] = ""
        else:
            rec["boundary"] = len(word) - len(a)
        for k, tok in toks.items():
            cuts, n = cut_offsets(tok, word)
            b = rec["boundary"]
            if cuts is None:
                cls = "unk_or_mismatch"
            elif b == "":
                cls = "no_boundary"
            elif b == len(word):
                cls = "zero"
            elif not cuts:
                cls = "whole"
            elif b in cuts:
                cls = "aligned"
            else:
                cls = "crossing"
            rec[f"{k}_class"], rec[f"{k}_pieces"] = cls, n
        words.append(rec)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "words.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(words[0]))
        w.writeheader()
        w.writerows(words)
    summary = []
    for k in ENCODERS:
        groups = defaultdict(list)
        for rec in words:
            groups[rec[f"{k}_class"]].append(rec)
        for cls in ("whole", "aligned", "crossing", "zero", "no_boundary", "unk_or_mismatch"):
            g = groups.get(cls, [])
            if not g:
                continue
            row = {"encoder": k, "class": cls, "n": len(g),
                   "median_train_freq": sorted(r["train_freq"] for r in g)[len(g) // 2]}
            for s in SYSTEMS:
                row[s] = round(100 * sum(r.get(s, 0) for r in g) / len(g), 2)
            summary.append(row)
            print(row, flush=True)
    with (OUT / "summary.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)


if __name__ == "__main__":
    main()
