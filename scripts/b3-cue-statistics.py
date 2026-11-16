#!/usr/bin/env python3
"""B3: corpus statistics for Veps infinitive cues and candidate new cues.

Part (a): for each cue lemma of the S3 lexicon (Kosheleva 2023, governors of the
I infinitive) count
  * cue tokens in texts outside dev and test;
  * nearby (+-WINDOW) unambiguous infinitives and INF/PASS/3PL conflict tokens there;
  * annotated conflict tokens of train+dev near the cue and their gold analyses.
Part (b): context lemmas that are not in the lexicon but occur near annotated
conflict tokens whose gold analysis is INF ("discovered cues"), with examples.
Test texts are never read: the cached corpus holds train/dev items only and the
unannotated scan is restricted to texts outside dev and test.
"""

from __future__ import annotations

import csv
import re
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1x_lib as L  # noqa: E402

LANG = "vep"
WINDOW = 3
INF, PASS, PL3 = "INF", "PRS+PASS", "PL+PRS+3+IND+POS"
OUT = L.EXP / "b3-cue-statistics-v1" / LANG
LEXICON = L.EXP / "s3-cue-lexicon-v1" / LANG / "cue_lexicon_vep.csv"
MAX_EXAMPLES = 3
# Part (b): candidate cues found earlier, matched by spelling (some forms have no
# dictionary lemma); "right" restricts the search to words after the target.
FINDINGS = [
    ("tahtoišk-", r"^tahtoišk", "both"),
    ("abuta", r"^abut", "both"),
    ("voimuz/voimust", r"^voimu", "both"),
    ("kebn", r"^kebn", "both"),
    ("tariž (справа)", r"^tariž", "right"),
]


def load_lexicon():
    cues, by_lemma = [], {}
    with LEXICON.open(encoding="utf-8") as h:
        for row in csv.DictReader(h):
            cues.append(row)
            for lid in row["lemma_ids"].split("|"):
                by_lemma[lid] = row["lemma"]
    return cues, by_lemma


def lemma_names():
    return {r["lemma_id"]: r["lemma"]
            for r in L.zst_rows(L.DATA_DIR / f"dict/morphology/lemmas_{LANG}.csv.zst")}


def token_lemmas(corpus, word_id):
    out = set()
    for wf, gid, _, _ in corpus.cands.get((LANG, word_id), []):
        out |= corpus.lemmas.get((LANG, wf, gid), frozenset())
    return out


def token_gramsets(corpus, word_id):
    return {g for _, _, g, _ in corpus.cands.get((LANG, word_id), []) if g}


def is_conflict(gramsets):
    return INF in gramsets and bool({PASS, PL3} & gramsets)


def neighbours(tokens, pos):
    for off in range(-WINDOW, WINDOW + 1):
        j = pos + off
        if off and 0 <= j < len(tokens):
            yield off, j


def render(tokens, target, cue):
    words = []
    for i, tok in enumerate(tokens):
        w = tok.norm  # orig spelling may be misaligned with word ids
        if i == target:
            w = f"**{w}**"
        elif i == cue:
            w = f"_{w}_"
        words.append(w)
    return " ".join(words)


def gold_label(gramset):
    return {INF: "INF", PASS: "PASS", PL3: "3PL"}.get(gramset, "other")


def main():
    corpus = pickle.load(open(L.EXP / "t1x-v1/corpus.pkl", "rb"))
    cues, cue_of = load_lexicon()
    names = lemma_names()
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- (a) corpus-wide counts in texts outside dev and test
    corpus_counts = defaultdict(Counter)
    for (lang, sid), tokens in corpus.sentences.items():
        if lang != LANG or (lang, sid) not in corpus.sentence_text or not L.allowed_aux_text(
                corpus, lang, corpus.sentence_text[(lang, sid)]):
            continue
        for pos, tok in enumerate(tokens):
            hit = {cue_of[l] for l in token_lemmas(corpus, tok.word_id) if l in cue_of}
            if not hit:
                continue
            near_inf = near_conf = False
            for _, j in neighbours(tokens, pos):
                gs = token_gramsets(corpus, tokens[j].word_id)
                if gs == {INF}:
                    near_inf = True
                elif is_conflict(gs):
                    near_conf = True
            for cue in hit:
                c = corpus_counts[cue]
                c["tokens"] += 1
                c["near_unambiguous_INF"] += near_inf
                c["near_conflict"] += near_conf

    # ---- annotated conflict tokens (train + dev)
    annotated = defaultdict(Counter)
    context = defaultdict(Counter)          # lemma_id -> gold labels (+ side)
    examples = defaultdict(list)            # lemma_id -> rendered examples
    base = Counter()
    for split, items in (("train", corpus.train), ("dev", corpus.dev)):
        for it in items:
            gs = {c[2] for c in it.candidates}
            if it.lang != LANG or not is_conflict(gs):
                continue
            gold = gold_label(it.candidates[it.gold][2])
            base[gold] += 1
            tokens = corpus.sentences[(LANG, it.sentence_id)]
            seen = set()
            for off, j in neighbours(tokens, it.pos):
                for lemma in token_lemmas(corpus, tokens[j].word_id):
                    if lemma in seen:
                        continue
                    seen.add(lemma)
                    context[lemma][gold] += 1
                    context[lemma]["left" if off < 0 else "right"] += 1
                    if lemma in cue_of:
                        annotated[cue_of[lemma]][gold] += 1
                    if len(examples[lemma]) < MAX_EXAMPLES or gold == "INF" and not any(
                            e[1] == "INF" for e in examples[lemma]):
                        dialect = corpus.text_dialect.get((LANG, it.text_id), "")
                        examples[lemma].append((split, gold, it.text_id, dialect, render(tokens, it.pos, j)))

    with (OUT / "cue_statistics.csv").open("w", encoding="utf-8", newline="") as h:
        w = csv.writer(h)
        w.writerow(["cue_class", "lemma", "gloss_ru", "corpus_tokens", "near_unambiguous_INF",
                    "near_conflict", "annotated_conflict", "gold_INF", "gold_PASS", "gold_3PL"])
        for cue in cues:
            c, a = corpus_counts[cue["lemma"]], annotated[cue["lemma"]]
            w.writerow([cue["cue_class"], cue["lemma"], cue["gloss_ru"], c["tokens"],
                        c["near_unambiguous_INF"], c["near_conflict"],
                        a["INF"] + a["PASS"] + a["3PL"] + a["other"], a["INF"], a["PASS"], a["3PL"]])

    cue_ids = set(cue_of)
    found = [(lid, c) for lid, c in context.items() if lid not in cue_ids and c["INF"] >= 1
             and c["INF"] / (c["INF"] + c["PASS"] + c["3PL"] + c["other"]) >= 0.5]
    found.sort(key=lambda x: (-x[1]["INF"], names.get(x[0], "")))
    with (OUT / "discovered_cues.csv").open("w", encoding="utf-8", newline="") as h:
        w = csv.writer(h)
        w.writerow(["lemma_id", "lemma", "pos", "conflict_tokens", "gold_INF", "gold_PASS", "gold_3PL",
                    "left", "right"])
        for lid, c in found:
            n = c["INF"] + c["PASS"] + c["3PL"] + c["other"]
            w.writerow([lid, names.get(lid, ""), corpus.lemma_pos.get((LANG, lid), ""), n,
                        c["INF"], c["PASS"], c["3PL"], c["left"], c["right"]])

    with (OUT / "examples.md").open("w", encoding="utf-8") as h:
        h.write(f"# B3 examples ({LANG}, train+dev, window ±{WINDOW}); target in bold, context word in italics\n\n")
        h.write(f"Baseline gold among annotated conflict tokens: {dict(base)}\n\n")
        for lid, c in [(l, context[l]) for l in sorted(cue_ids)] + found:
            if not examples[lid]:
                continue
            h.write(f"## {names.get(lid, lid)} ({lid}) — INF {c['INF']}, PASS {c['PASS']}, 3PL {c['3PL']}\n\n")
            for split, gold, text_id, dialect, sent in examples[lid]:
                h.write(f"- [{split}, text {text_id}, {dialect}, gold {gold}] {sent}\n")
            h.write("\n")
    # ---- (b) five findings by spelling, all expert-annotated conflict tokens of train+dev
    found_by_string = defaultdict(list)
    for (lang, sid), tokens in corpus.sentences.items():
        if lang != LANG or (lang, sid) not in corpus.sentence_text:
            continue
        text_id = corpus.sentence_text[(lang, sid)]
        split = corpus.split_of_text.get((lang, text_id))
        if split not in ("train", "dev"):
            continue
        for pos, tok in enumerate(tokens):
            rows = corpus.cands.get((LANG, tok.word_id), [])
            gold = [g for _, _, g, rel in rows if rel == "2"]
            if not gold or not is_conflict({g for _, _, g, _ in rows if g}):
                continue
            for name, pattern, side in FINDINGS:
                for off, j in neighbours(tokens, pos):
                    if side == "right" and off < 0:
                        continue
                    if re.search(pattern, tokens[j].norm):
                        dialect = corpus.text_dialect.get((LANG, text_id), "")
                        found_by_string[name].append(
                            (split, gold_label(gold[0]), text_id, dialect, render(tokens, pos, j)))
                        break
    with (OUT / "findings_examples.md").open("w", encoding="utf-8") as h:
        h.write("# B3 (b): five candidate cues, matched by spelling (train+dev, expert-annotated)\n\n")
        for name, _, _ in FINDINGS:
            ex = found_by_string[name]
            h.write(f"## {name} — {len(ex)} examples, INF {sum(e[1] == 'INF' for e in ex)}\n\n")
            for split, gold, text_id, dialect, sent in ex:
                h.write(f"- [{split}, text {text_id}, {dialect}, gold {gold}] {sent}\n")
            h.write("\n")
    print("baseline", dict(base))
    print(open(OUT / "cue_statistics.csv", encoding="utf-8").read())
    print("discovered:", len(found))
    for lid, c in found[:40]:
        print(lid, names.get(lid), dict(c))


if __name__ == "__main__":
    main()
