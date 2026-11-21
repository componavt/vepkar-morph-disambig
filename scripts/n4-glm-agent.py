#!/usr/bin/env python3
"""N4 experiment via ZCode subagents (GLM-5.3-Flash), same cards as the Opus run.

The model text is identical to n4-llm-rank.py (SYSTEM_PROMPT + TAGS_NOTE + items,
same SAMPLE_SEED), but the call is made by ZCode subagents instead of `claude -p`:
each batch prompt file is pasted by the orchestrator into a fresh subagent whose
final message is the reply. Deviation from the Opus run (recorded in the config):
the harness prepends its own agent instructions; our task text goes into the
user message, and the agent is told not to use any tools.

Phases:
  build   -- write batch prompt files + manifest.json for one language
  ingest  -- parse replies/batch-NNN.txt into answers.jsonl (checkpointable)
  eval    -- evaluate answers and save a run dir under n4-llm-v1/<run-name>

Usage:
  n4-glm-agent.py build --lang vep --n 841 --batch 40 --run glm53flash-dev-vep
  n4-glm-agent.py ingest --run glm53flash-dev-vep
  n4-glm-agent.py eval --run glm53flash-dev-vep --model-id "account:zai-start-plan/GLM-5.3-Flash"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nn_lib as N  # noqa: E402
import t1x_lib as L  # noqa: E402

import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location("n4_llm_rank", Path(__file__).resolve().parent / "n4-llm-rank.py")
_n4 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_n4)
OUT, SAMPLE_SEED = _n4.OUT, _n4.SAMPLE_SEED
SYSTEM_PROMPT, TAGS_NOTE = _n4.SYSTEM_PROMPT, _n4.TAGS_NOTE
parse_answers, render_item, sample_items = _n4.parse_answers, _n4.render_item, _n4.sample_items

ROOT = Path(__file__).resolve().parent.parent
NIGHT = ROOT / "work" / "night-records" / "n4-glm"
DEVIATIONS = [
    "backend: ZCode subagent (no `claude -p`); our SYSTEM_PROMPT is prepended to the user message instead of replacing the system prompt",
    "agents instructed not to use any tools; harness instructions still precede the task text",
    "prompt text, card format, item sampling seed and answer parsing are byte-identical to the Opus run",
]


def batch_text(corpus, names, items):
    # Byte-identical to the Opus run's content: system prompt + tags note + cards.
    return SYSTEM_PROMPT + "\n\n" + TAGS_NOTE + "\n\n" + "\n\n".join(
        render_item(corpus, names, it, k) for k, it in enumerate(items, 1)) + "\n"


def run_dir(run):
    d = NIGHT / run
    (d / "batches").mkdir(parents=True, exist_ok=True)
    (d / "replies").mkdir(parents=True, exist_ok=True)
    return d


def phase_build(args):
    corpus, names = N.load_corpus(), N.lemma_names()
    items = sample_items(corpus, args.lang, args.n) if args.n else sample_items(corpus, args.lang, 10**9)
    d = run_dir(args.run)
    meta = []
    for b, start in enumerate(range(0, len(items), args.batch), 1):
        chunk = items[start:start + args.batch]
        f = d / "batches" / f"batch-{b:03d}.txt"
        f.write_text(batch_text(corpus, names, chunk))
        meta.append({"batch": b, "file": f.name,
                     "items": [{"k": start + i, "word_id": it.word_id} for i, it in enumerate(chunk)]})
    manifest = {"run": args.run, "lang": args.lang, "n": len(items), "batch": args.batch,
                "sample_seed": SAMPLE_SEED,
                "prompt_sha256": hashlib.sha256((SYSTEM_PROMPT + "\n" + TAGS_NOTE).encode()).hexdigest(),
                "batches": meta}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(json.dumps({"run": args.run, "batches": len(meta), "items": len(items)}))


def phase_ingest(args):
    d = NIGHT / args.run
    manifest = json.loads((d / "manifest.json").read_text())
    answers, invalid, missing = {}, 0, []
    for b in manifest["batches"]:
        rf = d / "replies" / b["file"]
        if not rf.exists() or not rf.read_text().strip():
            missing.append(b["batch"])
            continue
        parsed = parse_answers(rf.read_text(), len(b["items"]))
        for rec, a in zip(b["items"], parsed):
            answers[rec["k"]] = a
            if a is None:
                invalid += 1
    with (d / "answers.jsonl").open("w") as out:
        for b in manifest["batches"]:
            for rec in b["items"]:
                out.write(json.dumps({"k": rec["k"], "word_id": rec["word_id"],
                                      "answer": answers.get(rec["k"])}, ensure_ascii=False) + "\n")
    print(json.dumps({"run": args.run, "items": manifest["n"], "answered": len(answers),
                      "invalid": invalid, "missing_batches": missing}))


def phase_eval(args):
    corpus, names = N.load_corpus(), N.lemma_names()
    d = NIGHT / args.run
    manifest = json.loads((d / "manifest.json").read_text())
    answers = {int(json.loads(l)["k"]): json.loads(l)["answer"]
               for l in (d / "answers.jsonl").read_text().splitlines() if l.strip()}
    items = sample_items(corpus, manifest["lang"], manifest["n"])
    scores = []
    for k, it in enumerate(items):
        n = len(it.candidates)
        a = answers.get(k)
        s = [0.0] * n
        if a is not None and 1 <= a <= n:
            s[a - 1] = 1.0
        else:  # same convention as the Opus backend: invalid/missing answer cannot win
            s = [-1.0 if i == it.gold else 0.0 for i in range(n)]
        scores.append(s)
    rows, summary = N.evaluate_scores(corpus, items, scores)
    out = OUT / args.run
    out.mkdir(parents=True, exist_ok=True)
    config = {"model": "n4-agent", "requested_model": args.model_id,
              "backend": "zcode-subagent", "lang": manifest["lang"], "n": manifest["n"],
              "batch": manifest["batch"], "sample_seed": manifest["sample_seed"],
              "prompt_sha256": manifest["prompt_sha256"], "deviations": DEVIATIONS}
    extra = {"word_ids": [it.word_id for it in items],
             "invalid_answers": sum(a is None for a in answers.values())}
    N.save_run(out, config, rows, summary, extra)
    print(json.dumps({"summary": summary}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("build", "ingest", "eval"))
    ap.add_argument("--lang", default="vep")
    ap.add_argument("--n", type=int, default=0, help="0 = all dev items of the language")
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model-id", default="account:zai-start-plan/GLM-5.3-Flash")
    args = ap.parse_args()
    if args.phase == "build":
        phase_build(args)
    elif args.phase == "ingest":
        phase_ingest(args)
    else:
        phase_eval(args)


if __name__ == "__main__":
    main()
