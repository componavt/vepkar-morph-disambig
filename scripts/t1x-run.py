#!/usr/bin/env python3
"""Run T1x experiments (train -> dev only). See scripts/t1x_lib.py for the axes."""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1x_lib as L  # noqa: E402

OUT = L.ROOT / "work/experiments/v2026.11/t1x-v1"
CACHE = OUT / "corpus.pkl"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def get_corpus():
    if CACHE.exists():
        with CACHE.open("rb") as h:
            return pickle.load(h)
    corpus = L.load_corpus(log)
    OUT.mkdir(parents=True, exist_ok=True)
    with CACHE.open("wb") as h:
        pickle.dump(corpus, h, protocol=pickle.HIGHEST_PROTOCOL)
    return corpus


def train_and_score(corpus, train_items, eval_items, cfg, model_kind, C, predicted_train=None, predicted_eval=None):
    ftr, gtr = L.featurize(corpus, train_items, cfg, predicted_train)
    vocab = L.Vocab().fit(ftr)
    Xtr = vocab.transform(ftr)
    model = (L.PointwiseModel(C) if model_kind == "point" else L.ListwiseModel(C)).fit(Xtr, train_items, gtr)
    fev, gev = L.featurize(corpus, eval_items, cfg, predicted_eval)
    scores = model.score(vocab.transform(fev))
    return model, vocab, L.rank(eval_items, scores, gev)


def run(corpus, name, cfg, model_kind, C, train_items=None, eval_items=None, save=True,
        predicted_train=None, predicted_eval=None):
    train_items = corpus.train if train_items is None else train_items
    eval_items = corpus.dev if eval_items is None else eval_items
    t0 = time.time()
    model, vocab, ranked = train_and_score(corpus, train_items, eval_items, cfg, model_kind, C,
                                           predicted_train, predicted_eval)
    rows = L.evaluate(corpus, eval_items, ranked)
    summary = L.summarize(rows)
    result = {"name": name, "config": asdict(cfg), "model": model_kind, "C": C,
              "train_items": len(train_items), "features": len(vocab.index),
              "seconds": round(time.time() - t0, 1), "summary": summary}
    if save:
        d = OUT / "runs" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
        with (d / "dev_rows.json").open("w") as h:
            json.dump(rows, h)
        with (d / "model.pkl").open("wb") as h:
            pickle.dump({"model": model, "vocab": vocab.index, "config": asdict(cfg), "kind": model_kind, "C": C}, h)
    s = summary["all"]
    log(f"{name:40s} full={s['full']:.4f} sync={s['sync']:.4f} mrr={s['mrr']:.4f} "
        f"feats={len(vocab.index)} t={result['seconds']}s")
    return rows, result, model, vocab, ranked


def c_grid(corpus, prefix, cfg, model_kind, grid, **kw):
    best = None
    table = []
    for C in grid:
        rows, result, *_ = run(corpus, f"{prefix}__C{C}", cfg, model_kind, C, **kw)
        table.append((C, result["summary"]["all"]["full"]))
        if best is None or result["summary"]["all"]["full"] > best[1]["summary"]["all"]["full"]:
            best = (rows, result)
    return best, table


NB = L.FeatureConfig(neighbours=True)


def softmax_top(ranked):
    out = []
    for order, s in ranked:
        e = np.exp(s - s.max())
        p = e / e.sum()
        out.append((order[0], float(p[order[0]])))
    return out


def predict_map(corpus, model_items, target_items, cfg, C):
    """Predicted gramset for each target word (model trained on model_items)."""
    _, _, ranked = train_and_score(corpus, model_items, target_items, cfg, "list", C)
    return {(it.lang, it.word_id): it.candidates[order[0]][2] for it, (order, _) in zip(target_items, ranked)}


def texts_sample(items, fraction, seed):
    texts = sorted({(i.lang, i.text_id) for i in items})
    rng = random.Random(seed)
    rng.shuffle(texts)
    keep = set(texts[:max(1, round(len(texts) * fraction))])
    return [i for i in items if (i.lang, i.text_id) in keep]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage")
    p.add_argument("--weight", default="0.5")
    args = p.parse_args()
    corpus = get_corpus()
    if args.stage == "load":
        log("cached", len(corpus.train), len(corpus.dev))
    elif args.stage == "baseline":
        c_grid(corpus, "point-base", L.FeatureConfig(), "point", [1, 3, 7, 10, 30])
    elif args.stage == "how":
        c_grid(corpus, "list-base", L.FeatureConfig(), "list", [0.3, 1, 3, 10, 30])
    elif args.stage == "orig2":
        c_grid(corpus, "list-orig2", L.FeatureConfig(spelling="orig"), "list", [1, 3, 10])
        c_grid(corpus, "list-all2", L.FeatureConfig(spelling="orig", factored=True, neighbours=True, lexicon=True), "list", [1, 3])
        c_grid(corpus, "list-nb-fact", L.FeatureConfig(factored=True, neighbours=True), "list", [1, 3])
    elif args.stage == "transfer":
        aux = {w: L.transfer_items(corpus, weight=w) for w in (0.2, 0.5, 1.0)}
        log("transfer items", len(aux[0.5]), Counter(i.lang for i in aux[0.5]))
        for w, items in aux.items():
            run(corpus, f"list-nb-transfer-w{w}__C1", NB, "list", 1, train_items=corpus.train + items)
    elif args.stage == "transfer2":
        cfg = replace(NB, transfer_context_only=True)
        for w in (0.1, 0.2, 0.5):
            items = L.transfer_items(corpus, weight=w)
            for C in (1, 3):
                run(corpus, f"list-nb-transferctx-w{w}__C{C}", cfg, "list", C, train_items=corpus.train + items)
    elif args.stage == "selftrain":
        pool = L.unannotated_items(corpus, cap_per_lang=20000)
        log("unannotated pool", len(pool), Counter(i.lang for i in pool))
        _, _, ranked = train_and_score(corpus, corpus.train, pool, NB, "list", 1)
        tops = softmax_top(ranked)
        for threshold, weight in ((0.95, 0.3), (0.9, 0.3), (0.95, 1.0)):
            pseudo = []
            for it, (top, prob) in zip(pool, tops):
                if prob >= threshold:
                    pseudo.append(replace(it, gold=top, weight=weight, source="self"))
            log(f"pseudo-labelled (p>={threshold}):", len(pseudo))
            run(corpus, f"list-nb-self-p{threshold}-w{weight}__C1", NB, "list", 1, train_items=corpus.train + pseudo)
    elif args.stage == "chorus":
        cfg2 = replace(NB, chorus=True)
        texts = sorted({(i.lang, i.text_id) for i in corpus.train})
        random.Random(0).shuffle(texts)
        folds = [set(texts[k::3]) for k in range(3)]
        predicted_train = {}
        for k, fold in enumerate(folds):
            held = [i for i in corpus.train if (i.lang, i.text_id) in fold]
            rest = [i for i in corpus.train if (i.lang, i.text_id) not in fold]
            targets = L.sentence_items(corpus, held)
            predicted_train.update(predict_map(corpus, rest, targets, NB, 1))
            log(f"fold {k}: predicted {len(targets)} words")
        dev_targets = L.sentence_items(corpus, corpus.dev)
        predicted_dev = predict_map(corpus, corpus.train, dev_targets, NB, 1)
        for C in (1, 3):
            run(corpus, f"list-nb-chorus__C{C}", cfg2, "list", C,
                predicted_train=predicted_train, predicted_eval=predicted_dev)
    elif args.stage == "pointnb":
        for C in (3, 7):
            run(corpus, f"point-nb__C{C}", NB, "point", C)
    elif args.stage == "dialect":
        for C in (1, 3):
            run(corpus, f"list-nb-dia__C{C}", replace(NB, dialect=True), "list", C)
        run(corpus, "list-dia__C1", L.FeatureConfig(dialect=True), "list", 1)
    elif args.stage == "cues":
        for C in (1, 3):
            run(corpus, f"list-nb-cues__C{C}", replace(NB, cues=True), "list", C)
    elif args.stage == "curves":
        configs = {"list-base": L.FeatureConfig(), "list-nb": NB,
                   "list-nb-cues": replace(NB, cues=True)}
        transfer = L.transfer_items(corpus, weight=float(args.weight))
        nb_transfer = replace(NB, transfer_context_only=True)
        for fraction in (0.1, 0.25, 0.5, 1.0):
            for seed in ((0, 1, 2) if fraction < 1 else (0,)):
                sub = texts_sample(corpus.train, fraction, seed) if fraction < 1 else corpus.train
                for name, cfg in configs.items():
                    run(corpus, f"curve-{name}-f{fraction}-s{seed}__C1", cfg, "list", 1, train_items=sub)
                run(corpus, f"curve-list-nb-transfer-f{fraction}-s{seed}__C1", nb_transfer, "list", 1,
                    train_items=sub + transfer)
    elif args.stage == "what":
        variants = {
            "orig": L.FeatureConfig(spelling="orig"),
            "fact": L.FeatureConfig(factored=True),
            "nb": L.FeatureConfig(neighbours=True),
            "lex": L.FeatureConfig(lexicon=True),
            "all": L.FeatureConfig(spelling="orig", factored=True, neighbours=True, lexicon=True),
        }
        for name, cfg in variants.items():
            c_grid(corpus, f"list-{name}", cfg, "list", [1, 3, 10])


if __name__ == "__main__":
    main()
