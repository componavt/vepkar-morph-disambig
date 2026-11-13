#!/usr/bin/env python3
"""Select the stratified B1-v8 context-card targets for one language (default vep).

Design (fixed before selection):
  * five ambiguity strata derived from the paradigm-generation rules;
  * per stratum 3 model-error targets (Top-1 gramset != gold) and 3 gramset-correct
    control targets; if fewer errors exist, all are taken and controls fill up to 6;
  * the six primary B1-v7 targets are kept and counted first;
  * one target per sentence, at most two targets per text, sentences of at most
    MAX_TOKENS tokens; dev split only (frozen A1-v5 predictions), test never read;
  * random choice inside each stratum with a fixed seed; controls sharing the
    surface form of an already chosen error target are preferred;
  * card order is shuffled with the same seed so that strata are interleaved.
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/dictorpus-data"
A1_DIR = ROOT / "work/experiments/v2026.11/t1-dev-error-analysis-a1-v5"
B1_V7_DIR = ROOT / "work/experiments/v2026.11/t1-context-cards-b1-v7"
DEFAULT_OUTPUT = ROOT / "work/experiments/v2026.11/t1-context-cards-b1-v8"
VERSION = "B1-v8"
SEED = 20261113
PER_STRATUM = 6
ERRORS_PER_STRATUM = 3
MAX_TOKENS = 40
MAX_PER_TEXT = 2
LEVELS = ("K0", "K1", "K2", "K3", "FULL")
WINDOW = {"K1": 1, "K2": 2, "K3": 3}

STRATA = {
    "1-obligatory": {frozenset(p) for p in (
        ("SG+ACC", "SG+GEN"), ("SG+ACC", "SG+NOM"), ("SG+ESS", "SG+GEN"),
        ("SG+ACC", "SG+ESS"), ("PL+ACC", "PL+NOM"))},
    "2-stem-class": {frozenset(p) for p in (("SG+PRT", "PL+NOM"), ("PL+ACC", "SG+PRT"))},
    "3-dialectal": {frozenset(("PL+PRS+3+IND+POS", "PRS+PASS"))},
    "4-cross-lemma": {frozenset(p) for p in (("INF", "PRS+PASS"), ("PL+NOM", "SG+NOM"))},
    # stratum 5 (noun/verb homonymy) is detected from gramset categories, see is_cross_pos()
    "5-cross-pos": set(),
}
STRATUM_RU = {
    "1-obligatory": "обязательный синкретизм",
    "2-stem-class": "синкретизм по типу основы",
    "3-dialectal": "диалектный синкретизм",
    "4-cross-lemma": "межлексемная омонимия",
    "5-cross-pos": "омонимия имени и глагола",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_zst_csv(path: Path) -> list[dict[str, str]]:
    raw = subprocess.run(["zstdcat", str(path)], check=True, capture_output=True, text=True).stdout
    return list(csv.DictReader(io.StringIO(raw)))


def gramset_kinds() -> dict[str, str]:
    """Classify gramsets as 'nominal' (case, no verbal category) or 'verbal'."""
    kinds = {}
    for row in read_csv(DATA_DIR / "dict/grammar/gramsets.csv"):
        has = lambda c: row[f"gram_id_{c}"] not in ("", r"\N")  # noqa: E731
        verbal = any(has(c) for c in ("mood", "infinitive", "participle", "voice", "tense"))
        kinds[row["gramset"]] = "verbal" if verbal else ("nominal" if has("case") else "other")
    return kinds


def stratum_of(pair: frozenset, kinds: dict[str, str]) -> str | None:
    for name, pairs in STRATA.items():
        if pair in pairs:
            return name
    if {kinds.get(g) for g in pair} == {"nominal", "verbal"}:
        return "5-cross-pos"
    return None


def sentence_words(sentence_xml: str) -> list[str]:
    return [html.unescape(re.sub(r"<[^>]+>", "", w)) for w in re.findall(r"<w[^>]*>(.*?)</w>", sentence_xml, flags=re.S)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--language", default="vep")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    lang = args.language
    out = args.output / lang
    if out.exists() and any(out.iterdir()):
        sys.exit(f"Output directory is not empty, refusing to overwrite: {out}")

    kinds = gramset_kinds()
    instances = [r for r in read_csv(A1_DIR / "instances.csv") if r["language"] == lang]
    cand = defaultdict(set)
    for r in read_csv(A1_DIR / "candidates.csv"):
        if r["language"] == lang:
            cand[r["word_id"]].add(r["gramset"])
    sentences = {r["sentence_id"]: r for r in read_zst_csv(DATA_DIR / f"corpus/sentences_{lang}.csv.zst")}
    dialect = {r["text_id"]: r["dialect_code"] for r in read_zst_csv(DATA_DIR / f"corpus/texts_{lang}.csv.zst")}

    # Candidate pool: every dev instance with its stratum, role and sentence length.
    pool = []
    for r in instances:
        words = sentence_words(sentences[r["sentence_id"]]["sentence_xml"])
        gold, top1 = r["gold_gramset"], r["top1_gramset"]
        if gold != top1:
            role, strata = "error", {stratum_of(frozenset((gold, top1)), kinds)} - {None}
        else:
            # A control can illustrate every stratum whose pair it contains.
            role = "control"
            strata = {stratum_of(frozenset((gold, g)), kinds) for g in cand[r["word_id"]] - {gold}} - {None}
        for stratum in sorted(strata):
            pool.append({**r, "role": role, "stratum": stratum, "n_tokens": len(words)})

    old = {r["word_id"]: (r["target_id"], frozenset(x.strip() for x in r["selection_pair"].split("↔")))
           for r in read_csv(B1_V7_DIR / "selected-targets-internal.csv")
           if r["language"] == lang and r["is_primary_target"] == "1"}
    rng = random.Random(SEED)
    chosen, used_sentences, per_text = [], set(), Counter()

    def take(row, old_id=""):
        chosen.append({**row, "old_target_id": old_id})
        used_sentences.add(row["sentence_id"])
        per_text[row["text_id"]] += 1

    # Old B1-v7 targets keep the stratum of the pair they were originally selected for.
    for word_id, (old_id, pair) in old.items():
        wanted = stratum_of(pair, kinds)
        take(next(r for r in pool if r["word_id"] == word_id and r["stratum"] == wanted), old_id)

    # Scarce strata first, so that rare model errors are not crowded out by text limits.
    scarcity = Counter(r["stratum"] for r in pool if r["role"] == "error")
    for stratum in sorted(STRATA, key=lambda name: (scarcity[name], name)):
        def free(row):
            return (row["stratum"] == stratum and row["sentence_id"] not in used_sentences
                    and per_text[row["text_id"]] < MAX_PER_TEXT and row["n_tokens"] <= MAX_TOKENS)
        have = Counter(c["role"] for c in chosen if c["stratum"] == stratum)
        errors = [r for r in pool if r["role"] == "error" and free(r)]
        rng.shuffle(errors)
        for row in errors:
            if have["error"] >= ERRORS_PER_STRATUM:
                break
            if free(row):
                take(row); have["error"] += 1
        error_forms = {c["normalized_form"] for c in chosen if c["stratum"] == stratum and c["role"] == "error"}
        controls = [r for r in pool if r["role"] == "control" and free(r)]
        rng.shuffle(controls)
        controls.sort(key=lambda r: r["normalized_form"] not in error_forms)  # stable: same-form first
        need = PER_STRATUM - have["error"] - have["control"]
        for row in controls:
            if need <= 0:
                break
            if free(row):
                take(row); need -= 1

    rng.shuffle(chosen)
    out.mkdir(parents=True, exist_ok=True)
    targets, views = [], []
    for number, row in enumerate(chosen, start=1):
        card = f"VEP-B1v8-{number:02d}" if lang == "vep" else f"{lang.upper()}-B1v8-{number:02d}"
        options = sorted(cand[row["word_id"]])
        sentence = sentences[row["sentence_id"]]
        words = sentence_words(sentence["sentence_xml"])
        index = int(row["target_position"]) - 1
        assert words[index].lower().replace("’", "").replace("'", "") == row["normalized_form"].replace("’", ""), card
        targets.append({
            "card_id": card, "target_id": card, "language": lang, "is_primary_target": "1",
            "stratum": row["stratum"], "stratum_ru": STRATUM_RU[row["stratum"]], "role": row["role"],
            "old_target_id": row["old_target_id"], "text_id": row["text_id"], "sentence_id": row["sentence_id"],
            "word_id": row["word_id"], "dialect_code": dialect.get(row["text_id"], ""),
            "normalized_form": row["normalized_form"], "target_position": row["target_position"],
            "gold_gramset": row["gold_gramset"], "top1_gramset": row["top1_gramset"],
            "neutral_option_map_internal": ";".join(f"{chr(65 + i)}:{g}" for i, g in enumerate(options)),
            "n_tokens": len(words), "sentence_ru": sentence.get("sentence_ru", ""),
        })
        first_span = {}
        for level in LEVELS:
            if level == "K0":
                span = (index, index)
            elif level == "FULL":
                span = (0, len(words) - 1)
            else:
                span = (max(0, index - WINDOW[level]), min(len(words) - 1, index + WINDOW[level]))
            repeat = first_span.get(span, "") if level != "K0" else ""
            first_span.setdefault(span, level)
            views.append({"card_id": card, "target_id": card, "language": lang, "level": level,
                          "span_start": span[0], "span_end": span[1], "repeat_of_level": repeat,
                          "requires_new_answer": "0" if repeat else "1"})

    for name, rows in (("selected-targets-internal.csv", targets), ("context-views-internal.csv", views)):
        with (out / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "version": VERSION, "language": lang, "seed": SEED, "max_tokens": MAX_TOKENS,
        "max_per_text": MAX_PER_TEXT, "targets": len(targets),
        "new_answers": sum(v["requires_new_answer"] == "1" for v in views),
        "by_stratum_role": {f"{s}/{r}": n for (s, r), n in sorted(Counter((t["stratum"], t["role"]) for t in targets).items())},
        "source_predictions": "A1-v5 frozen dev predictions (both / C=7)",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
