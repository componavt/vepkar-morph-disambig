#!/usr/bin/env python3
"""N2: fine-tuned pretrained encoder as a candidate ranker.

The sentence window is encoded by a pretrained encoder; the target word vector is
the mean of the hidden states covering its characters. Each candidate is
represented by its gramset (full tag + bag of tag parts) and, optionally, by its
lemma encoded with the same encoder. A small scorer ranks the candidates; the
softmax over the candidate list is trained with cross-entropy. Early stopping
uses a fixed 10% of train texts; dev is only evaluated.

Stage 1 of the protocol: four encoders, scope vep, lemma off, one seed.
  n2-encoder.py --stage1
Single run:
  n2-encoder.py --encoder glot500 --scope vep --lemma off --seed 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nn_lib as N  # noqa: E402
import t1x_lib as L  # noqa: E402

OUT = L.EXP / "n2-encoder-v1"
ENCODERS = {
    "xlmr": "FacebookAI/xlm-roberta-base",
    "glot500": "cis-lmu/glot500-base",
    "canine": "google/canine-s",
    "finbert": "TurkuNLP/bert-base-finnish-cased-v1",
}
WINDOW = {"canine": 12}   # words on each side of the target; default below
DEFAULT_WINDOW = 20
MAX_LEN = {"canine": 512}
DEFAULT_MAX_LEN = 160


def item_text(corpus, it, window):
    """Joined sentence window and the character span of the target."""
    toks = corpus.sentences[(it.lang, it.sentence_id)]
    lo, hi = max(0, it.pos - window), min(len(toks), it.pos + window + 1)
    parts, span, offset = [], None, 0
    for i in range(lo, hi):
        w = toks[i].norm
        if i == it.pos:
            span = (offset, offset + len(w))
        parts.append(w)
        offset += len(w) + 1
    return " ".join(parts), span


class Data:
    def __init__(self, corpus, names, fit, encoder_key):
        self.corpus, self.names = corpus, names
        self.window = WINDOW.get(encoder_key, DEFAULT_WINDOW)
        grams, parts = Counter(), Counter()
        for it in fit:
            for _, _, g in it.candidates:
                grams[g] += 1
                parts.update(g.split("+"))
        self.gram = N.vocab(grams, 1)
        self.part = N.vocab(parts, 1)
        self.lang = {l: i for i, l in enumerate(L.LANGS)}

    def encode(self, it):
        text, span = item_text(self.corpus, it, self.window)
        cands = []
        for wf, gid, g in it.candidates:
            lemma = N.candidate_lemma(self.corpus, self.names, it.lang, wf, gid) or "?"
            cands.append((self.gram.get(g, 1), [self.part.get(p, 1) for p in g.split("+") if p] or [1], lemma))
        return {"text": text, "span": span, "lang": self.lang[it.lang], "cands": cands, "gold": it.gold}


def bag(emb, ids):
    mask = (ids != 0).unsqueeze(-1).float()
    return (emb(ids) * mask).sum(1) / mask.sum(1).clamp(min=1.0)


class Ranker(nn.Module):
    def __init__(self, name, data: Data, use_lemma, dim=128, dropout=0.2):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        h = self.enc.config.hidden_size
        self.use_lemma = use_lemma
        self.gram = nn.Embedding(len(data.gram), dim)
        self.part = nn.Embedding(len(data.part), dim, padding_idx=0)
        self.lang = nn.Embedding(len(data.lang), 16)
        self.target = nn.Sequential(nn.Dropout(dropout), nn.Linear(h + 16, 2 * dim), nn.Tanh())
        self.cand = nn.Sequential(nn.Linear(2 * dim + (h if use_lemma else 0), 2 * dim), nn.Tanh())
        self.score = nn.Sequential(nn.Linear(6 * dim, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, 1))

    def pooled(self, enc_inputs, masks):
        hidden = self.enc(**enc_inputs).last_hidden_state
        m = masks.unsqueeze(-1).float()
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1.0)

    def forward(self, b):
        t = self.target(torch.cat([self.pooled(b["sent"], b["target_mask"]), self.lang(b["lang"])], -1))
        c = [self.gram(b["cand_gram"]), bag(self.part, b["cand_parts"])]
        if self.use_lemma:
            c.append(self.pooled(b["lemma"], b["lemma"]["attention_mask"]))
        c = self.cand(torch.cat(c, -1))
        t = t[b["cand_item"]]
        return self.score(torch.cat([t, c, t * c], -1)).squeeze(-1)


class Collator:
    def __init__(self, tokenizer, encoder_key, use_lemma, device):
        self.tok, self.key, self.use_lemma, self.device = tokenizer, encoder_key, use_lemma, device
        self.max_len = MAX_LEN.get(encoder_key, DEFAULT_MAX_LEN)

    def __call__(self, batch):
        texts = [ex["text"] for ex in batch]
        if self.key == "canine":
            enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
            mask = torch.zeros_like(enc["input_ids"])
            for i, ex in enumerate(batch):     # CLS at 0, characters from 1
                a, b = ex["span"]
                mask[i, 1 + a:min(1 + b, mask.size(1))] = 1
        else:
            enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_len,
                           return_offsets_mapping=True, return_tensors="pt")
            offsets = enc.pop("offset_mapping")
            mask = torch.zeros_like(enc["input_ids"])
            for i, ex in enumerate(batch):
                a, b = ex["span"]
                s, e = offsets[i, :, 0], offsets[i, :, 1]
                mask[i] = ((e > a) & (s < b) & (e > s)).long()
        cand_gram, cand_parts, cand_item, lemmas, gold, offset = [], [], [], [], [], 0
        for i, ex in enumerate(batch):
            for g, p, lemma in ex["cands"]:
                cand_gram.append(g)
                cand_parts.append(p)
                cand_item.append(i)
                lemmas.append(lemma)
            gold.append(offset + (ex["gold"] or 0))
            offset += len(ex["cands"])
        n = max(len(p) for p in cand_parts)
        out = {"sent": {k: v.to(self.device) for k, v in enc.items()}, "target_mask": mask.to(self.device),
               "lang": torch.tensor([ex["lang"] for ex in batch], device=self.device),
               "cand_gram": torch.tensor(cand_gram, device=self.device),
               "cand_parts": torch.tensor([p + [0] * (n - len(p)) for p in cand_parts], device=self.device),
               "cand_item": torch.tensor(cand_item, device=self.device),
               "gold": torch.tensor(gold, device=self.device)}
        if self.use_lemma:
            lem = self.tok(lemmas, padding=True, truncation=True, max_length=32, return_tensors="pt")
            out["lemma"] = {k: v.to(self.device) for k, v in lem.items()}
        return out


def item_logsoftmax(scores, cand_item, n_items):
    m = torch.full((n_items,), -1e9, device=scores.device).scatter_reduce(0, cand_item, scores, "amax")
    z = torch.zeros(n_items, device=scores.device).index_add(0, cand_item, torch.exp(scores - m[cand_item]))
    return scores - (m + torch.log(z))[cand_item]


def batches(data, size, shuffle, gen=None):
    order = torch.randperm(len(data), generator=gen).tolist() if shuffle else list(range(len(data)))
    for i in range(0, len(order), size):
        yield [data[j] for j in order[i:i + size]]


def predict(model, data, collate, size=64):
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in batches(data, size, False):
            s = model(collate(batch)).float().cpu().tolist()
            k = 0
            for ex in batch:
                out.append(s[k:k + len(ex["cands"])])
                k += len(ex["cands"])
    return out


def accuracy(data, scores):
    return sum(max(range(len(s)), key=s.__getitem__) == ex["gold"] for ex, s in zip(data, scores)) / len(data)


def run(corpus, names, key, scope, lemma, seed, device, epochs=8, patience=2, batch_size=16,
        lr_encoder=3e-5, lr_head=1e-3, suffix=""):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    fit, holdout, dev = N.split_items(corpus, scope)
    data = Data(corpus, names, fit, key)
    fit_d, hold_d, dev_d = ([data.encode(it) for it in part] for part in (fit, holdout, dev))
    tokenizer = AutoTokenizer.from_pretrained(ENCODERS[key])
    model = Ranker(ENCODERS[key], data, lemma == "on").to(device)
    collate = Collator(tokenizer, key, lemma == "on", device)
    enc_params = list(model.enc.parameters())
    enc_ids = {id(p) for p in enc_params}
    head = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": lr_encoder}, {"params": head, "lr": lr_head}],
                            weight_decay=0.01)
    steps = epochs * ((len(fit_d) + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr_encoder, lr_head], total_steps=steps,
                                                pct_start=0.1, anneal_strategy="linear")
    best, best_state, bad, history, t0 = -1.0, None, 0, [], time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in batches(fit_d, batch_size, True, gen):
            b = collate(batch)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                scores = model(b).float()
            loss = -item_logsoftmax(scores, b["cand_item"], len(batch))[b["gold"]].mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(batch)
        acc = accuracy(hold_d, predict(model, hold_d, collate))
        history.append({"epoch": epoch, "loss": total / len(fit_d), "holdout_top1": acc})
        print(f"  epoch {epoch} loss {total / len(fit_d):.4f} holdout {acc:.4f}", flush=True)
        if acc > best:
            best, bad = acc, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    scores = predict(model, dev_d, collate)
    rows, summary = N.evaluate_scores(corpus, dev, scores)
    config = {"model": "n2-encoder", "encoder": ENCODERS[key], "scope": scope, "lemma": lemma, "seed": seed,
              "epochs_max": epochs, "lr_encoder": lr_encoder, "lr_head": lr_head, "batch": batch_size}
    extra = {"history": history, "holdout_best": best, "seconds": round(time.time() - t0, 1),
             "fit_items": len(fit), "holdout_items": len(holdout), "dev_items": len(dev)}
    out = OUT / f"{key}-{scope}-lemma-{lemma}{suffix}" / f"seed-{seed}"
    N.save_run(out, config, rows, summary, extra)
    N.save_scores(out / "dev_scores.jsonl", dev, scores)
    del model
    torch.cuda.empty_cache()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=sorted(ENCODERS), default="glot500")
    ap.add_argument("--scope", choices=sorted(N.SCOPES), default="vep")
    ap.add_argument("--lemma", choices=("off", "on"), default="off")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--stage1", action="store_true", help="four encoders, vep, lemma off, seed 1")
    ap.add_argument("--stage2", nargs="+", metavar="ENCODER",
                    help="given encoders x {vep, karelian, all} x {lemma off, on} x 5 seeds, lr 1e-5")
    ap.add_argument("--lr-encoder", type=float, default=3e-5)
    ap.add_argument("--suffix", default="", help="output directory suffix for non-default settings")
    args = ap.parse_args()
    device = "cuda"
    corpus, names = N.load_corpus(), N.lemma_names()
    plan = [(k, "vep", "off", 1) for k in ("xlmr", "glot500", "canine", "finbert")] if args.stage1 \
        else [(args.encoder, args.scope, args.lemma, args.seed)]
    if args.stage2:
        args.lr_encoder, args.suffix = 1e-5, "-lr1e-5"
        plan = [(k, scope, lemma, seed) for seed in range(1, 6) for k in args.stage2
                for scope in ("vep", "karelian", "all") for lemma in ("off", "on")]
    for key, scope, lemma, seed in plan:
        if (OUT / f"{key}-{scope}-lemma-{lemma}{args.suffix}" / f"seed-{seed}" / "result.json").exists():
            continue
        print(f"== {key} {scope} lemma={lemma} seed={seed}", flush=True)
        s = run(corpus, names, key, scope, lemma, seed, device, lr_encoder=args.lr_encoder, suffix=args.suffix)
        print(json.dumps({k: round(v["full"], 4) for k, v in s.items()}), flush=True)


if __name__ == "__main__":
    main()
