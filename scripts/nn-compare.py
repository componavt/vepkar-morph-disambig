#!/usr/bin/env python3
"""Compare neural / LLM runs on dev with the linear models (seed-averaged, paired bootstrap).

Each system is a directory with seed-*/dev_rows.json (rows are averaged over seeds
word by word) or a single dev_rows.json; several directories joined with '+' are
concatenated (e.g. per-language N4 runs). Every system is compared with the
reference runs on the words it covers, by a text-level paired bootstrap.

Usage:
  nn-compare.py [--ref point-base__C7 list-nb__C1] SYSTEM [SYSTEM ...]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
import t1x_lib as L  # noqa: E402

EXP = "work/experiments/v2026.13"
LANGS = ("vep", "krl", "olo", "lud")
SEED = 20261115


def load_rows(spec):
    rows = []
    for part in spec.split("+"):
        path = part if os.path.exists(part) else os.path.join(EXP, part)
        files = sorted(glob.glob(os.path.join(path, "seed-*", "dev_rows.json"))) or [os.path.join(path, "dev_rows.json")]
        acc = defaultdict(lambda: {"full": 0.0, "sync": 0.0, "rr": 0.0})
        meta = {}
        for f in files:
            for r in json.load(open(f)):
                key = (r["lang"], r["word_id"])
                for m in ("full", "sync", "rr"):
                    acc[key][m] += r[m] / len(files)
                meta[key] = r
        for key, v in acc.items():
            rows.append({**meta[key], **v})
    return {(r["lang"], r["word_id"]): r for r in rows}, len(files)


def mean(rows, m, lang=None):
    xs = [r[m] for r in rows if lang is None or r["lang"] == lang]
    return 100 * sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", nargs="+", default=["t1x-v1/runs/point-base__C7", "t1x-v1/runs/list-nb__C1"])
    ap.add_argument("systems", nargs="+")
    args = ap.parse_args()
    refs = {r: load_rows(r)[0] for r in args.ref}
    for spec in args.systems:
        rows, n_seeds = load_rows(spec)
        keys = [k for k in next(iter(refs.values())) if k in rows]
        sys_rows = [rows[k] for k in keys]
        langs = " ".join(f"{l} {mean(sys_rows, 'full', l):.2f}" for l in LANGS if any(k[0] == l for k in keys))
        print(f"\n{spec}  (seeds {n_seeds}, n {len(keys)})")
        print(f"  full {mean(sys_rows, 'full'):.2f}  sync {mean(sys_rows, 'sync'):.2f}  mrr {mean(sys_rows, 'rr'):.2f}  | {langs}")
        for ref, ref_rows in refs.items():
            ra = [ref_rows[k] for k in keys]
            out = []
            for m in ("full", "sync"):
                d, lo, hi = L.paired_bootstrap(ra, sys_rows, m, seed=SEED)
                out.append(f"{m} {100 * d:+.2f} [{100 * lo:+.2f}; {100 * hi:+.2f}]")
            print(f"  vs {os.path.basename(ref):16s} " + "   ".join(out))


if __name__ == "__main__":
    main()
