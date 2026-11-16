#!/usr/bin/env python3
"""Hybrid: the final linear model (list-nb, C=1) plus N1 BiLSTM scores (dev only).

N1 scores come from ``n1-bilstm.py --crossfit``: train words are scored out of fold
(by text), dev words by a model trained on all of train; both are averaged over
seeds as log-probabilities. Variants:

  list-nb-rerun   the linear model alone (sanity check, saves dev scores)
  hyb-real        + one real-valued feature: N1 log-probability of the candidate
  hyb-bin         + N1 rank and probability decile as indicator features
  fuse-0.5        no training: mean of the two models' log-probabilities

Usage: hybrid-run.py [--n1 all-lemma-off-oof]
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1x_lib as L  # noqa: E402

T1X = L.EXP / "t1x-v1"
N1 = L.EXP / "n1-bilstm-v1"
OUT = L.EXP / "hybrid-v1"
FLOOR = -10.0
NB = L.FeatureConfig(neighbours=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def logsoftmax(x):
    x = np.asarray(x, dtype=np.float64)
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


def load_n1(run, name):
    """(lang, word_id) -> seed-averaged log-probabilities over the item's candidates."""
    acc, count = {}, defaultdict(int)
    files = sorted(glob.glob(str(N1 / run / "seed-*" / name)))
    for f in files:
        for line in open(f):
            r = json.loads(line)
            key = (r["lang"], r["word_id"])
            lp = logsoftmax(r["scores"])
            acc[key] = acc.get(key, 0) + lp
            count[key] += 1
    log(f"{name}: {len(files)} seeds, {len(acc)} words")
    return {k: v / count[k] for k, v in acc.items()}


def n1_column(items, n1):
    col, missing = [], 0
    for it in items:
        lp = n1.get((it.lang, it.word_id))
        if lp is None or len(lp) != len(it.candidates):
            missing += 1
            lp = np.full(len(it.candidates), -np.log(len(it.candidates)))
        col.extend(np.maximum(lp, FLOOR))
    return np.asarray(col), missing


def bin_features(items, n1):
    extra = []
    for it in items:
        lp = n1.get((it.lang, it.word_id))
        if lp is None or len(lp) != len(it.candidates):
            extra.extend([["nn|missing"]] * len(it.candidates))
            continue
        order = np.argsort(-lp)
        rank = np.empty(len(lp), dtype=int)
        rank[order] = np.arange(1, len(lp) + 1)
        for j in range(len(lp)):
            p = float(np.exp(lp[j]))
            extra.append([f"nnr|{min(rank[j], 3)}", f"nnp|{min(int(p * 10), 9)}", f"nnr|{it.lang}|{min(rank[j], 3)}"])
    return extra


def save(name, corpus, items, scores, extra=None):
    ranked = L.rank(items, scores, None)
    rows = L.evaluate(corpus, items, ranked)
    summary = L.summarize(rows)
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps({"name": name, "summary": summary, **(extra or {})},
                                              ensure_ascii=False, indent=1) + "\n")
    with (d / "dev_rows.json").open("w") as h:
        json.dump(rows, h)
    with (d / "dev_scores.jsonl").open("w") as h:
        k = 0
        for it in items:
            n = len(it.candidates)
            h.write(json.dumps({"lang": it.lang, "word_id": it.word_id, "scores": list(map(float, scores[k:k + n]))}) + "\n")
            k += n
    s = summary["all"]
    log(f"{name:16s} full={s['full']:.4f} sync={s['sync']:.4f} mrr={s['mrr']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n1", default="all-lemma-off-oof")
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()
    corpus = pickle.load(open(T1X / "corpus.pkl", "rb"))
    train, dev = corpus.train, corpus.dev
    log("items", len(train), len(dev))
    n1_train = load_n1(args.n1, "train_oof_scores.jsonl")
    n1_dev = load_n1(args.n1, "dev_scores.jsonl")
    info = {"n1_run": args.n1, "C": args.C}

    ftr, gtr = L.featurize(corpus, train, NB)
    fdv, gdv = L.featurize(corpus, dev, NB)
    vocab = L.Vocab().fit(ftr)
    Xtr, Xdv = vocab.transform(ftr), vocab.transform(fdv)

    base = L.ListwiseModel(args.C).fit(Xtr, train, gtr)
    s_base = base.score(Xdv)
    save("list-nb-rerun", corpus, dev, s_base, info)

    ctr, miss_tr = n1_column(train, n1_train)
    cdv, miss_dv = n1_column(dev, n1_dev)
    log("missing N1 scores: train", miss_tr, "dev", miss_dv)
    info = {**info, "missing_train": miss_tr, "missing_dev": miss_dv}
    Xtr_r = sp.hstack([Xtr, sp.csr_matrix(ctr[:, None])]).tocsr()
    Xdv_r = sp.hstack([Xdv, sp.csr_matrix(cdv[:, None])]).tocsr()
    real = L.ListwiseModel(args.C).fit(Xtr_r, train, gtr)
    save("hyb-real", corpus, dev, real.score(Xdv_r), {**info, "n1_weight": float(real.w[-1]) if hasattr(real, "w") else None})

    btr = [f + e for f, e in zip(ftr, bin_features(train, n1_train))]
    bdv = [f + e for f, e in zip(fdv, bin_features(dev, n1_dev))]
    vocab_b = L.Vocab().fit(btr)
    binm = L.ListwiseModel(args.C).fit(vocab_b.transform(btr), train, gtr)
    save("hyb-bin", corpus, dev, binm.score(vocab_b.transform(bdv)), info)

    fused, k = [], 0
    for it in dev:
        n = len(it.candidates)
        lin = logsoftmax(s_base[k:k + n])
        nn_lp = n1_dev.get((it.lang, it.word_id))
        nn_lp = lin if nn_lp is None or len(nn_lp) != n else nn_lp
        fused.extend(0.5 * lin + 0.5 * nn_lp)
        k += n
    save("fuse-0.5", corpus, dev, np.asarray(fused), info)


if __name__ == "__main__":
    main()
