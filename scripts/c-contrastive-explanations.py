#!/usr/bin/env python3
"""C: contrastive feature contributions for selected dev tokens (listwise model).

For a token, score(candidate) = sum of weights of its active features. The
difference score(Top-1) - score(gold) decomposes exactly into weights of the
features present in one candidate but not in the other. The script lists the
largest contributions for the B1-v8 Veps targets under a saved T1x model and
checks that the decomposition reproduces the score difference.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1x_lib as L  # noqa: E402

BASE = L.ROOT / "work/experiments/v2026.11"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="list-nb__C1")
    p.add_argument("--top", type=int, default=6)
    args = p.parse_args()
    corpus = pickle.load(open(BASE / "t1x-v1/corpus.pkl", "rb"))
    saved = pickle.load(open(BASE / f"t1x-v1/runs/{args.run}/model.pkl", "rb"))
    assert saved["kind"] == "list", "contributions are implemented for the listwise model"
    w, vocab = saved["model"].w, saved["vocab"]
    cfg = L.FeatureConfig(**saved["config"])
    by_word = {it.word_id: it for it in corpus.dev}
    with (BASE / "t1-context-cards-b1-v8/vep/selected-targets-internal.csv").open(encoding="utf-8") as h:
        targets = list(csv.DictReader(h))
    lines = [f"# C — контрастные вклады признаков ({args.run}), карточки B1-v8 (vep)", "",
             "Для каждой цели: решение модели и признаки, сильнее всего сдвинувшие его от gold "
             "(положительный вклад — в пользу Top-1). Сумма всех вкладов равна разности оценок.", ""]
    for t in targets:
        it = by_word.get(int(t["word_id"]))
        if it is None:
            continue

        def feats(i):
            return set(L.candidate_features(corpus, it, it.candidates[i], cfg))

        def score(fs):
            return sum(w[vocab[f]] for f in fs if f in vocab)

        scores = [score(feats(i)) for i in range(len(it.candidates))]
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], it.candidates[i][0], it.candidates[i][2]))
        top, gold = order[0], it.gold
        toks = [x.orig for x in corpus.sentences[(it.lang, it.sentence_id)]]
        ctx = " ".join(toks[max(0, it.pos - 3):it.pos] + [f"[{toks[it.pos]}]"] + toks[it.pos + 1:it.pos + 3])
        verdict = "верно" if top == gold else "ошибка"
        lines.append(f"## {t['card_id']} — {t['normalized_form']} ({t['stratum_ru']}, роль в B1: {t['role']})")
        lines.append(f"Контекст: … {ctx} …  ")
        lines.append(f"Gold: `{it.candidates[gold][2]}`; модель: `{it.candidates[top][2]}` — **{verdict}**  ")
        if top != gold:
            a, b = feats(top), feats(gold)
            contrib = [(w[vocab[f]], f, "Top-1") for f in a - b if f in vocab] + \
                      [(-w[vocab[f]], f, "gold") for f in b - a if f in vocab]
            contrib.sort(key=lambda x: -abs(x[0]))
            total = sum(c[0] for c in contrib)
            assert abs(total - (scores[top] - scores[gold])) < 1e-6
            lines.append(f"Разность оценок Top-1 − gold = {total:+.3f}; главные вклады:")
            for value, name, owner in contrib[:args.top]:
                lines.append(f"- `{name}` ({owner}): {value:+.3f}")
        lines.append("")
    out = BASE / "t1x-v1/analysis" / f"contrastive-b1v8-{args.run}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
