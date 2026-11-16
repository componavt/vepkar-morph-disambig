"""Shared data preparation and evaluation for the neural branch (N1, N2).

Items come from the cached T1x corpus (train and dev only; test texts are never
loaded). Results are written in the T1x format (``result.json`` and
``dev_rows.json``) so that ``t1x_lib.paired_bootstrap`` can compare runs.
"""

from __future__ import annotations

import json
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1x_lib as L  # noqa: E402

SCOPES = {"vep": ("vep",), "karelian": ("krl", "olo", "lud"), "all": ("vep", "krl", "olo", "lud")}
HOLDOUT_SHARE = 0.1      # share of train texts used for early stopping
HOLDOUT_SEED = 20261115


def load_corpus():
    """Slim cache (train/dev sentences only, ~1 GB); see nn-make-cache.py."""
    return pickle.load(open(L.EXP / "t1x-v1/corpus-slim.pkl", "rb"))


def lemma_names():
    names = {}
    for lang in L.LANGS:
        for r in L.zst_rows(L.DATA_DIR / f"dict/morphology/lemmas_{lang}.csv.zst"):
            names[(lang, r["lemma_id"])] = r["lemma"].lower()
    return names


def split_items(corpus, scope):
    """Return (fit, holdout, dev) item lists for a data scope.

    The holdout is a fixed 10% of train texts, used only for early stopping.
    """
    langs = SCOPES[scope]
    train = [it for it in corpus.train if it.lang in langs]
    dev = [it for it in corpus.dev if it.lang in langs]
    texts = sorted({(it.lang, it.text_id) for it in train})
    rng = random.Random(HOLDOUT_SEED)
    rng.shuffle(texts)
    held = set(texts[:max(1, int(len(texts) * HOLDOUT_SHARE))])
    fit = [it for it in train if (it.lang, it.text_id) not in held]
    holdout = [it for it in train if (it.lang, it.text_id) in held]
    return fit, holdout, dev


def candidate_lemma(corpus, names, lang, wf, gid):
    ids = sorted(corpus.lemmas.get((lang, wf, gid), ()))
    return names.get((lang, ids[0]), "") if ids else ""


def candidate_string(corpus, lang, wf):
    forms = sorted(corpus.wf_string.get((lang, wf), ()))
    return forms[0] if forms else ""


def token_gramsets(corpus, lang, word_id):
    return sorted({g for _, _, g, _ in corpus.cands.get((lang, word_id), []) if g})


def evaluate_scores(corpus, items, scores):
    """scores: list of per-item score lists (candidate order of the item)."""
    flat = [s for per_item in scores for s in per_item]
    ranked = L.rank(items, flat, None)
    rows = L.evaluate(corpus, items, ranked)
    return rows, summarize(rows)


def summarize(rows):
    """Like ``t1x_lib.summarize`` but skips languages outside the data scope."""
    out = {}
    for key, subset in [("all", rows)] + [(l, [r for r in rows if r["lang"] == l]) for l in L.LANGS]:
        if subset:
            n = len(subset)
            out[key] = {"n": n, "full": sum(r["full"] for r in subset) / n,
                        "sync": sum(r["sync"] for r in subset) / n,
                        "mrr": sum(r["rr"] for r in subset) / n}
    return out


def save_run(out_dir: Path, config: dict, rows, summary, extra=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"config": config, "summary": summary, **(extra or {})}
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    with (out_dir / "dev_rows.json").open("w") as h:
        json.dump(rows, h)


def save_scores(path: Path, items, scores):
    """One JSON line per item: language, word id and candidate scores in item order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as h:
        for it, sc in zip(items, scores):
            h.write(json.dumps({"lang": it.lang, "word_id": it.word_id, "scores": sc}) + "\n")


def vocab(counter: Counter, min_count: int, reserved=("<pad>", "<unk>")):
    itos = list(reserved) + sorted(k for k, v in counter.items() if v >= min_count)
    return {s: i for i, s in enumerate(itos)}
