#!/usr/bin/env python3
"""N4: a large language model chooses among candidate analyses without training.

The model sees a Veps sentence with the target word marked and a numbered list of
candidate analyses (lemma + grammatical tags) and answers with one number per
item. Items are sent in batches to reduce per-call overhead.

Backends:
  claude  -- Claude Code in non-interactive mode (``claude -p``) with all tools
             disabled, our own system prompt, no settings or MCP servers, run
             from an empty temporary directory so the model cannot read the
             repository; the exact model id is taken from the JSON reply.
  local   -- an open model on the local GPU (4-bit), same system prompt and
             tag note, one item per prompt; the assistant turn is prefilled with
             '{"item": 1, "answer": ' and the candidates are ranked by the
             next-token probability of their numbers (deterministic);
             --permute cyclic averages over all cyclic shifts of the list
             to cancel the position bias of small models.

Only dev is used here; test runs follow pclaude test protocol section 7.

Usage:
  n4-llm-rank.py --backend claude --n 100 --batch 10 --run-name pilot-100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nn_lib as N  # noqa: E402
import t1x_lib as L  # noqa: E402

OUT = L.EXP / "n4-llm-v1"
SAMPLE_SEED = 20261115
LANG_NAME = {"vep": "Veps", "krl": "Karelian Proper", "olo": "Livvi Karelian", "lud": "Ludian"}

SYSTEM_PROMPT = """You are an expert in Finnic languages and morphological annotation.
You will receive numbered items. Each item is a sentence from a corpus with one target word
marked as [[word]], followed by the candidate morphological analyses of that word produced by
a dictionary: lemma and grammatical tags (Leipzig-style abbreviations joined by '+').
For each item choose the single analysis that is correct in this context.
Answer only with JSON lines, one per item, in the form {"item": <item number>, "answer": <analysis number>}.
Do not add explanations."""

TAGS_NOTE = """Tag abbreviations: SG/PL number; NOM nominative, GEN genitive, ACC accusative, PRT partitive,
ESS essive, TRANS translative, INS instructive, COM comitative, AB abessive, APPRX approximative,
IN inessive, ELA elative, ILL illative, AD adessive, ABL ablative, ALL allative;
INF first infinitive, INF2 second infinitive, INF3 third infinitive; PRS present, IPFV.PST imperfect (past),
IND indicative, COND conditional, IMP imperative, POS affirmative, CON connegative (form used with
the negative verb), PASS passive, ACT active, IPFV.PTCP / PFV.PTCP participles, 1/2/3 person."""


def sample_items(corpus, lang, n):
    items = [it for it in corpus.dev if it.lang == lang]
    rng = random.Random(SAMPLE_SEED)
    return items if n >= len(items) else rng.sample(items, n)


def render_item(corpus, names, it, k, order=None):
    toks = corpus.sentences[(it.lang, it.sentence_id)]
    words = [f"[[{t.norm}]]" if i == it.pos else t.norm for i, t in enumerate(toks)]
    lines = [f"Item {k}. Language: {LANG_NAME[it.lang]}.", "Sentence: " + " ".join(words), "Analyses:"]
    cands = it.candidates if order is None else [it.candidates[i] for i in order]
    for j, (wf, gid, g) in enumerate(cands, 1):
        lemma = N.candidate_lemma(corpus, names, it.lang, wf, gid) or "?"
        lines.append(f"  {j}. {lemma}: {g}")
    return "\n".join(lines)


def build_prompt(blocks):
    return TAGS_NOTE + "\n\n" + "\n\n".join(blocks) + "\n"


def call_claude(prompt, model):
    cmd = ["claude", "-p", "--model", model, "--tools", "", "--system-prompt", SYSTEM_PROMPT,
           "--setting-sources", "", "--strict-mcp-config", "--no-session-persistence",
           "--output-format", "json"]
    with tempfile.TemporaryDirectory(prefix="n4-") as empty:
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=empty, timeout=900)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:] or proc.stdout[-2000:])
    reply = json.loads(proc.stdout)
    return reply.get("result", ""), reply


def parse_answers(text, n_items):
    answers = {}
    for m in re.finditer(r'\{\s*"item"\s*:\s*(\d+)\s*,\s*"answer"\s*:\s*(\d+)\s*\}', text):
        answers[int(m.group(1))] = int(m.group(2))
    return [answers.get(k) for k in range(1, n_items + 1)]


class LocalScorer:
    def __init__(self, model_name):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant,
                                                          device_map="cuda", low_cpu_mem_usage=True)
        self.model.eval()
        self.digit_ids = [self.tok.encode(str(d), add_special_tokens=False)[0] for d in range(1, 10)]

    def score(self, block, n):
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt([block])}]
        try:
            text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
        except TypeError:
            text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text += '{"item": 1, "answer": '
        ids = self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        with self.torch.no_grad():
            logits = self.model(ids).logits[0, -1].float()
        logp = self.torch.log_softmax(logits, -1)
        return [logp[self.digit_ids[j]].item() for j in range(n)], ids.size(1)


def run_local(args, corpus, names, items, out, prompt_hash):
    scorer = LocalScorer(args.model)
    scores, t0, n_tokens = [], time.time(), 0
    with (out / "scores.jsonl").open("w") as log:
        for k, it in enumerate(items, 1):
            n = len(it.candidates)
            shifts = range(n) if args.permute == "cyclic" else range(1)
            s = [0.0] * n
            for shift in shifts:
                # Cyclic shifts put every candidate in every position once; averaging
                # the per-candidate log-probabilities cancels the position bias.
                order = [(i + shift) % n for i in range(n)]
                part, n_tok = scorer.score(render_item(corpus, names, it, 1, order), n)
                n_tokens += n_tok
                for pos, i in enumerate(order):
                    s[i] += part[pos] / len(shifts)
            scores.append(s)
            log.write(json.dumps({"word_id": it.word_id, "lang": it.lang, "logp": s}) + "\n")
            if k % 100 == 0:
                print(f"  items {k}/{len(items)}", flush=True)
    rows, summary = N.evaluate_scores(corpus, items, scores)
    config = {"model": "n4-local", "requested_model": args.model, "quantization": "bnb-nf4",
              "permute": args.permute, "lang": args.lang, "n": len(items), "sample_seed": SAMPLE_SEED, "prompt_sha256": prompt_hash}
    extra = {"seconds": round(time.time() - t0, 1), "prompt_tokens": n_tokens,
             "word_ids": [it.word_id for it in items]}
    N.save_run(out, config, rows, summary, extra)
    print(json.dumps({"summary": summary, "seconds": extra["seconds"]}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("claude", "local"), default="claude")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--lang", default="vep")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--permute", choices=("none", "cyclic"), default="none",
                    help="local backend: average scores over cyclic shifts of the candidate list")
    args = ap.parse_args()
    corpus, names = N.load_corpus(), N.lemma_names()
    items = sample_items(corpus, args.lang, args.n)
    out = OUT / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    prompt_hash = hashlib.sha256((SYSTEM_PROMPT + "\n" + TAGS_NOTE).encode()).hexdigest()
    if args.backend == "local":
        return run_local(args, corpus, names, items, out, prompt_hash)
    log = (out / "calls.jsonl").open("a")
    scores, calls, t0 = [], [], time.time()
    for start in range(0, len(items), args.batch):
        chunk = items[start:start + args.batch]
        prompt = build_prompt([render_item(corpus, names, it, k) for k, it in enumerate(chunk, 1)])
        text, reply = call_claude(prompt, args.model)
        answers = parse_answers(text, len(chunk))
        for it, a in zip(chunk, answers):
            n = len(it.candidates)
            # invalid or missing answer: no candidate preferred -> scored as an error below
            s = [0.0] * n
            if a is not None and 1 <= a <= n:
                s[a - 1] = 1.0
            else:
                s = [-1.0 if i == it.gold else 0.0 for i in range(n)]
            scores.append(s)
        record = {"start": start, "n": len(chunk), "answers": answers, "result": text,
                  "usage": reply.get("usage"), "modelUsage": reply.get("modelUsage"),
                  "total_cost_usd": reply.get("total_cost_usd"), "duration_ms": reply.get("duration_ms")}
        log.write(json.dumps(record, ensure_ascii=False) + "\n")
        log.flush()
        calls.append(record)
        print(f"  items {start + len(chunk)}/{len(items)}; invalid {sum(a is None for a in answers)}", flush=True)
    rows, summary = N.evaluate_scores(corpus, items, scores)
    models = sorted({m for c in calls for m in (c.get("modelUsage") or {})})
    config = {"model": "n4-" + args.backend, "requested_model": args.model, "model_ids": models,
              "lang": args.lang, "n": len(items), "batch": args.batch, "sample_seed": SAMPLE_SEED,
              "prompt_sha256": prompt_hash}
    extra = {"seconds": round(time.time() - t0, 1), "calls": len(calls),
             "total_cost_usd_equivalent": round(sum(c.get("total_cost_usd") or 0 for c in calls), 4),
             "input_tokens": sum((c.get("usage") or {}).get("input_tokens", 0) for c in calls),
             "output_tokens": sum((c.get("usage") or {}).get("output_tokens", 0) for c in calls),
             "word_ids": [it.word_id for it in items]}
    N.save_run(out, config, rows, summary, extra)
    print(json.dumps({"summary": summary, **{k: v for k, v in extra.items() if k != "word_ids"}}, indent=1))


if __name__ == "__main__":
    main()
