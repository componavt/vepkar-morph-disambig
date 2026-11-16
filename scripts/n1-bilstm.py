#!/usr/bin/env python3
"""N1: BiLSTM candidate ranker trained from scratch.

Each token of the sentence window is encoded from its characters (char BiLSTM),
a frequent-word embedding and the bag of gramset parts of its candidate analyses
(ambiguity class). A sentence BiLSTM produces the target representation; every
candidate (gramset parts, full gramset, optionally the lemma spelled by
characters) is scored against it, and the softmax over the candidate list is
trained with cross-entropy. Early stopping uses a fixed 10% of train texts;
dev is only evaluated.

Usage:
  n1-bilstm.py --scope vep --lemma off --seed 1
  n1-bilstm.py --grid --seeds 1        # the 6 configurations of the protocol
"""

from __future__ import annotations

import argparse
import json
import sys
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nn_lib as N  # noqa: E402
import t1x_lib as L  # noqa: E402

OUT = L.EXP / "n1-bilstm-v1"
WINDOW = 20           # tokens on each side of the target
MAX_CHARS = 24
GRID = [(scope, lemma) for scope in ("vep", "karelian", "all") for lemma in ("off", "on")]


# ----------------------------------------------------------------------------- data

class Encoder:
    """Turns corpus items into integer tensors."""

    def __init__(self, corpus, names, fit_items):
        self.corpus, self.names = corpus, names
        chars, words, parts, grams = Counter(), Counter(), Counter(), Counter()
        for it in fit_items:
            toks = self.window(it)[0]
            for tok in toks:
                words[tok.norm.lower()] += 1
                chars.update(tok.norm.lower())
                for g in N.token_gramsets(corpus, it.lang, tok.word_id):
                    parts.update(g.split("+"))
            for wf, gid, g in it.candidates:
                grams[g] += 1
                parts.update(g.split("+"))
                chars.update(N.candidate_lemma(corpus, names, it.lang, wf, gid))
        self.char = N.vocab(chars, 1)
        self.word = N.vocab(words, 2)
        self.part = N.vocab(parts, 1)
        self.gram = N.vocab(grams, 1)
        self.lang = {l: i for i, l in enumerate(L.LANGS)}

    def window(self, it):
        toks = self.corpus.sentences[(it.lang, it.sentence_id)]
        lo, hi = max(0, it.pos - WINDOW), min(len(toks), it.pos + WINDOW + 1)
        return toks[lo:hi], it.pos - lo

    def chars_of(self, s):
        ids = [self.char.get(c, 1) for c in s[:MAX_CHARS]]
        return ids or [1]

    def parts_of(self, g):
        return [self.part.get(p, 1) for p in g.split("+") if p] or [1]

    def encode(self, it):
        toks, pos = self.window(it)
        tok_chars = [self.chars_of(t.norm.lower()) for t in toks]
        tok_words = [self.word.get(t.norm.lower(), 1) for t in toks]
        tok_parts = []
        for t in toks:
            ps = []
            for g in N.token_gramsets(self.corpus, it.lang, t.word_id):
                ps += self.parts_of(g)
            tok_parts.append(ps or [0])
        cands = []
        for wf, gid, g in it.candidates:
            lemma = N.candidate_lemma(self.corpus, self.names, it.lang, wf, gid)
            cands.append((self.parts_of(g), self.gram.get(g, 1), self.chars_of(lemma or "?")))
        return {"lang": self.lang[it.lang], "pos": pos, "chars": tok_chars, "words": tok_words,
                "parts": tok_parts, "cands": cands, "gold": it.gold}


def pad(seqs, value=0):
    n = max(len(s) for s in seqs)
    return torch.tensor([s + [value] * (n - len(s)) for s in seqs])


def collate(batch, device):
    """Flatten a batch of encoded items into padded tensors."""
    tok_chars, tok_words, tok_parts, sent_len, pos = [], [], [], [], []
    cand_parts, cand_gram, cand_chars, cand_item, gold, langs = [], [], [], [], [], []
    offset = 0
    for i, ex in enumerate(batch):
        sent_len.append(len(ex["words"]))
        pos.append(ex["pos"])
        tok_chars += ex["chars"]
        tok_words += ex["words"]
        tok_parts += ex["parts"]
        langs.append(ex["lang"])
        for p, g, c in ex["cands"]:
            cand_parts.append(p)
            cand_gram.append(g)
            cand_chars.append(c)
            cand_item.append(i)
        gold.append(offset + (ex["gold"] if ex["gold"] is not None else 0))
        offset += len(ex["cands"])
    t = {
        "tok_chars": pad(tok_chars), "tok_words": torch.tensor(tok_words), "tok_parts": pad(tok_parts),
        "sent_len": torch.tensor(sent_len), "pos": torch.tensor(pos), "lang": torch.tensor(langs),
        "cand_parts": pad(cand_parts), "cand_gram": torch.tensor(cand_gram), "cand_chars": pad(cand_chars),
        "cand_item": torch.tensor(cand_item), "gold": torch.tensor(gold),
    }
    return {k: v.to(device) for k, v in t.items()}


# ---------------------------------------------------------------------------- model

class CharEncoder(nn.Module):
    def __init__(self, n_chars, dim, hidden):
        super().__init__()
        self.emb = nn.Embedding(n_chars, dim, padding_idx=0)
        self.rnn = nn.LSTM(dim, hidden, batch_first=True, bidirectional=True)

    def forward(self, ids):
        lengths = (ids != 0).sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.emb(ids), lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.rnn(packed)
        return torch.cat([h[0], h[1]], dim=-1)


def bag(emb, ids):
    mask = (ids != 0).unsqueeze(-1).float()
    return (emb(ids) * mask).sum(1) / mask.sum(1).clamp(min=1.0)


class Ranker(nn.Module):
    def __init__(self, enc: Encoder, use_lemma: bool, dim=64, hidden=128, dropout=0.3):
        super().__init__()
        self.use_lemma = use_lemma
        self.chars = CharEncoder(len(enc.char), 32, dim)
        self.word = nn.Embedding(len(enc.word), dim, padding_idx=0)
        self.part = nn.Embedding(len(enc.part), dim, padding_idx=0)
        self.gram = nn.Embedding(len(enc.gram), dim)
        self.lang = nn.Embedding(len(enc.lang), 16)
        self.sent = nn.LSTM(2 * dim + dim + dim + 16, hidden, batch_first=True, bidirectional=True)
        self.lemma_chars = CharEncoder(len(enc.char), 32, dim) if use_lemma else None
        cand_dim = dim + dim + (2 * dim if use_lemma else 0)
        self.cand = nn.Sequential(nn.Linear(cand_dim, 2 * hidden), nn.Tanh())
        self.score = nn.Sequential(nn.Linear(6 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, b):
        tok = torch.cat([self.chars(b["tok_chars"]), self.word(b["tok_words"]), bag(self.part, b["tok_parts"])], -1)
        sents = torch.split(tok, b["sent_len"].tolist())
        x = nn.utils.rnn.pad_sequence(sents, batch_first=True)
        lang = self.lang(b["lang"]).unsqueeze(1).expand(-1, x.size(1), -1)
        x = self.drop(torch.cat([x, lang], -1))
        packed = nn.utils.rnn.pack_padded_sequence(x, b["sent_len"].cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.sent(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        target = out[torch.arange(out.size(0), device=out.device), b["pos"]]
        c = [bag(self.part, b["cand_parts"]), self.gram(b["cand_gram"])]
        if self.use_lemma:
            c.append(self.lemma_chars(b["cand_chars"]))
        c = self.cand(torch.cat(c, -1))
        t = target[b["cand_item"]]
        return self.score(torch.cat([t, c, t * c], -1)).squeeze(-1)


def item_logsoftmax(scores, cand_item, n_items):
    m = torch.full((n_items,), -1e9, device=scores.device).scatter_reduce(0, cand_item, scores, "amax")
    z = torch.zeros(n_items, device=scores.device).index_add(0, cand_item, torch.exp(scores - m[cand_item]))
    return scores - (m + torch.log(z))[cand_item]


# ------------------------------------------------------------------------- training

def batches(data, size, shuffle, gen=None):
    order = torch.randperm(len(data), generator=gen).tolist() if shuffle else list(range(len(data)))
    for i in range(0, len(order), size):
        yield [data[j] for j in order[i:i + size]]


def predict(model, data, device, size=256):
    model.eval()
    out = []
    with torch.no_grad():
        for batch in batches(data, size, False):
            b = collate(batch, device)
            s = model(b).cpu().tolist()
            k = 0
            for ex in batch:
                n = len(ex["cands"])
                out.append(s[k:k + n])
                k += n
    return out


def accuracy(data, scores):
    return sum(max(range(len(s)), key=s.__getitem__) == ex["gold"] for ex, s in zip(data, scores)) / len(data)


def train(corpus, names, fit, holdout, lemma, seed, device, epochs=30, patience=4, batch_size=32, lr=2e-3):
    """Train on ``fit`` with early stopping on ``holdout``; returns (model, encoder, history, best)."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    enc = Encoder(corpus, names, fit)
    fit_d, hold_d = ([enc.encode(it) for it in part] for part in (fit, holdout))
    model = Ranker(enc, lemma == "on").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, bad, history = -1.0, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in batches(fit_d, batch_size, True, gen):
            b = collate(batch, device)
            logp = item_logsoftmax(model(b), b["cand_item"], len(batch))
            loss = -logp[b["gold"]].mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * len(batch)
        acc = accuracy(hold_d, predict(model, hold_d, device))
        history.append({"epoch": epoch, "loss": total / len(fit_d), "holdout_top1": acc})
        print(f"  epoch {epoch} loss {total / len(fit_d):.4f} holdout {acc:.4f}", flush=True)
        if acc > best:
            best, bad = acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, enc, history, best


def run(corpus, names, scope, lemma, seed, device, epochs=30, save_scores=False):
    t0 = time.time()
    fit, holdout, dev = N.split_items(corpus, scope)
    model, enc, history, best = train(corpus, names, fit, holdout, lemma, seed, device, epochs)
    scores = predict(model, [enc.encode(it) for it in dev], device)
    rows, summary = N.evaluate_scores(corpus, dev, scores)
    config = {"model": "n1-bilstm", "scope": scope, "lemma": lemma, "seed": seed, "window": WINDOW,
              "params": sum(p.numel() for p in model.parameters())}
    extra = {"history": history, "holdout_best": best, "seconds": round(time.time() - t0, 1),
             "fit_items": len(fit), "holdout_items": len(holdout), "dev_items": len(dev)}
    out = OUT / f"{scope}-lemma-{lemma}" / f"seed-{seed}"
    N.save_run(out, config, rows, summary, extra)
    if save_scores:
        N.save_scores(out / "dev_scores.jsonl", dev, scores)
    return summary


def crossfit(corpus, names, scope, lemma, seed, device, folds=5, epochs=30):
    """Out-of-fold scores for train (folds by text) plus dev scores of a full-train model.

    Used as a stacking feature for the linear model: no train word is scored by a
    model that saw its text.
    """
    out = OUT / f"{scope}-lemma-{lemma}-oof" / f"seed-{seed}"
    langs = N.SCOPES[scope]
    train_items = [it for it in corpus.train if it.lang in langs]
    texts = sorted({(it.lang, it.text_id) for it in train_items})
    random.Random(N.HOLDOUT_SEED).shuffle(texts)
    fold_of = {t: k % folds for k, t in enumerate(texts)}
    oof_items, oof_scores = [], []
    for k in range(folds):
        rest = [it for it in train_items if fold_of[(it.lang, it.text_id)] != k]
        part = [it for it in train_items if fold_of[(it.lang, it.text_id)] == k]
        rest_texts = sorted({(it.lang, it.text_id) for it in rest})
        random.Random(N.HOLDOUT_SEED + k).shuffle(rest_texts)
        held = set(rest_texts[:max(1, int(len(rest_texts) * N.HOLDOUT_SHARE))])
        fit = [it for it in rest if (it.lang, it.text_id) not in held]
        holdout = [it for it in rest if (it.lang, it.text_id) in held]
        model, enc, _, best = train(corpus, names, fit, holdout, lemma, seed, device, epochs)
        oof_items += part
        oof_scores += predict(model, [enc.encode(it) for it in part], device)
        print(f"  fold {k}: {len(part)} items, holdout {best:.4f}", flush=True)
    N.save_scores(out / "train_oof_scores.jsonl", oof_items, oof_scores)
    fit, holdout, dev = N.split_items(corpus, scope)
    model, enc, _, _ = train(corpus, names, fit, holdout, lemma, seed, device, epochs)
    dev_scores = predict(model, [enc.encode(it) for it in dev], device)
    rows, summary = N.evaluate_scores(corpus, dev, dev_scores)
    N.save_scores(out / "dev_scores.jsonl", dev, dev_scores)
    N.save_run(out, {"model": "n1-bilstm-oof", "scope": scope, "lemma": lemma, "seed": seed,
                     "folds": folds}, rows, summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=sorted(N.SCOPES), default="vep")
    ap.add_argument("--lemma", choices=("off", "on"), default="off")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--crossfit", action="store_true",
                    help="out-of-fold train scores + dev scores for stacking (seeds 1..--seeds)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    corpus, names = N.load_corpus(), N.lemma_names()
    if args.crossfit:
        for seed in range(1, args.seeds + 1):
            if (OUT / f"{args.scope}-lemma-{args.lemma}-oof" / f"seed-{seed}" / "result.json").exists():
                continue
            print(f"== crossfit {args.scope} lemma={args.lemma} seed={seed}", flush=True)
            s = crossfit(corpus, names, args.scope, args.lemma, seed, device, epochs=args.epochs)
            print(json.dumps({k: round(v["full"], 4) for k, v in s.items()}), flush=True)
        return
    plan = GRID if args.grid else [(args.scope, args.lemma)]
    seeds = range(1, args.seeds + 1) if args.grid else [args.seed]
    for scope, lemma in plan:
        for seed in seeds:
            if (OUT / f"{scope}-lemma-{lemma}" / f"seed-{seed}" / "result.json").exists():
                continue
            print(f"== {scope} lemma={lemma} seed={seed}", flush=True)
            s = run(corpus, names, scope, lemma, seed, device, epochs=args.epochs)
            print(json.dumps({k: round(v["full"], 4) for k, v in s.items() if v["n"]}), flush=True)


if __name__ == "__main__":
    main()
