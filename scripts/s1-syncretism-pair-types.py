#!/usr/bin/env python3
"""S1: measure competing gramset pairs in a language and propose a syncretism type.

For every unordered pair {G, H} of gramsets that co-occur in the candidate list of
an ambiguous corpus token, the script reports
  * corpus counts: tokens (all / expert-annotated / automatic only), by dialect;
  * token-level lemma evidence: share of tokens where one lemma has candidates
    for both G and H (same-paradigm syncretism) versus only different lemmas;
  * lexicon evidence: among lemmas whose dictionary paradigm has both cells,
    the share where the two cells share a word form, overall and by dialect;
  * a rule-based type proposal (1 obligatory, 2 lexically conditioned,
    3 dialectal, 4 cross-lemma, 5 noun/verb homonymy or no lexicon evidence) for later manual review.

Population: corpus tokens with >= 2 distinct (wordform_id, gramset) candidates and
no empty gramset; texts assigned to the test split are excluded entirely.
No predictions, gold labels or models are used for the type proposal.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/dictorpus-data"
SPLIT_FILE = ROOT / "data/derived/splits_v2026.11.csv"
DEFAULT_OUTPUT = ROOT / "work/experiments/v2026.11/s1-pair-types-v1"
EMPTY_GRAMSET_IDS = {"0", "7"}
MIN_DIALECT_LEMMAS = 30  # dialect lexicon rates are reported only above this support
SAME_LEMMA_LOW = 0.2     # token share below which a pair is treated as cross-lemma
OBLIGATORY = 0.95        # lexicon share at or above which a pair is obligatory


def zst_rows(path: Path):
    raw = subprocess.run(["zstdcat", str(path)], check=True, capture_output=True, text=True).stdout
    return csv.DictReader(io.StringIO(raw))


def gramset_kinds() -> dict[str, str]:
    kinds = {}
    with (DATA_DIR / "dict/grammar/gramsets.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            has = lambda c: row[f"gram_id_{c}"] not in ("", r"\N")  # noqa: E731
            verbal = any(has(c) for c in ("mood", "infinitive", "participle", "voice", "tense"))
            kinds[row["gramset"]] = "verbal" if verbal else ("nominal" if has("case") else "other")
    return kinds


def propose_type(pair, kinds, same_lemma_share, lexicon_share, dialect_shares) -> tuple[str, str]:
    g, h = pair
    cross_pos = {kinds.get(g), kinds.get(h)} == {"nominal", "verbal"}
    if same_lemma_share is not None and same_lemma_share < SAME_LEMMA_LOW:
        return ("5", "noun/verb homonymy") if cross_pos else ("4", "different lemmas in tokens")
    if lexicon_share is None:
        return "5", "no lexicon evidence"
    if lexicon_share >= OBLIGATORY:
        return "1", f"cells coincide in {lexicon_share:.0%} of lemmas"
    supported = [v for v in dialect_shares.values() if v is not None]
    if len(supported) >= 2 and max(supported) >= 0.8 and min(supported) <= 0.2:
        return "3", "coincidence depends on dialect"
    # Same lemma in the tokens, but only part of the lexicon: stem class or a few
    # (often frequent) lemmas such as pronouns; mixed pairs are flagged.
    mixed = " (mixed with cross-lemma tokens)" if same_lemma_share < 0.8 else ""
    return "2", f"cells coincide in {lexicon_share:.1%} of lemmas{mixed}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--language", default="vep")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    lang = args.language
    out = args.output / lang
    if out.exists() and any(out.iterdir()):
        sys.exit(f"Output directory is not empty, refusing to overwrite: {out}")

    with SPLIT_FILE.open(encoding="utf-8") as handle:
        test_texts = {r["text_id"] for r in csv.DictReader(handle) if r["language"] == lang and r["split"] == "test"}
    dialect_of_text = {r["text_id"]: (r["dialect_code"] or "none") for r in zst_rows(DATA_DIR / f"corpus/texts_{lang}.csv.zst")}
    text_of_sentence = {r["sentence_id"]: r["text_id"] for r in zst_rows(DATA_DIR / f"corpus/sentences_{lang}.csv.zst")}
    text_of_word = {r["word_id"]: text_of_sentence.get(r["sentence_id"]) for r in zst_rows(DATA_DIR / f"corpus/words_{lang}.csv.zst")}

    # Lexicon: (wordform_id, gramset_id) -> lemmas; lemma -> dialect -> gramset -> wordforms
    lemmas_of = defaultdict(set)
    paradigm = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    gramset_name = {}
    with (DATA_DIR / "dict/grammar/gramsets.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            gramset_name[row["gramset_id"]] = row["gramset"]
    # Row IDs in the published generation rules equal VepKar gramset_id.
    gramset_id = {name: gid for gid, name in gramset_name.items()}
    for r in zst_rows(DATA_DIR / f"dict/morphology/lemma_wordforms_{lang}.csv.zst"):
        lemmas_of[(r["wordform_id"], r["gramset_id"])].add(r["lemma_id"])
        g = gramset_name.get(r["gramset_id"])
        if g:
            dialect = r["dialect_code"] if r["dialect_code"] not in ("", r"\N") else "none"
            paradigm[r["lemma_id"]][dialect][g].add(r["wordform_id"])
            paradigm[r["lemma_id"]]["*"][g].add(r["wordform_id"])

    # Candidate lists per token
    candidates = defaultdict(list)
    for r in zst_rows(DATA_DIR / f"corpus/candidate_analyses_{lang}.csv.zst"):
        candidates[r["word_id"]].append(r)

    pair_tokens = Counter()
    pair_gold = Counter()
    pair_same_lemma = Counter()
    pair_same_wordform = Counter()
    pair_dialect = defaultdict(Counter)
    pair_examples = defaultdict(Counter)
    population = Counter()
    for word_id, rows in candidates.items():
        text_id = text_of_word.get(word_id)
        if text_id is None or text_id in test_texts:
            continue
        if any(r["gramset_id"] in EMPTY_GRAMSET_IDS or not r["gramset"] for r in rows):
            population["excluded_empty_gramset"] += 1
            continue
        keys = {(r["wordform_id"], r["gramset"]) for r in rows}
        if len(keys) < 2:
            continue
        annotated = any(r["relevance"] == "2" for r in rows)
        population["ambiguous_tokens"] += 1
        population["annotated" if annotated else "automatic_only"] += 1
        by_gramset = defaultdict(lambda: {"wf": set(), "lemmas": set()})
        for r in rows:
            entry = by_gramset[r["gramset"]]
            entry["wf"].add(r["wordform_id"])
            entry["lemmas"] |= lemmas_of.get((r["wordform_id"], r["gramset_id"]), set())
        dialect = dialect_of_text.get(text_id, "none")
        for g, h in combinations(sorted(by_gramset), 2):
            pair = (g, h)
            pair_tokens[pair] += 1
            pair_gold[pair] += annotated
            pair_same_lemma[pair] += bool(by_gramset[g]["lemmas"] & by_gramset[h]["lemmas"])
            pair_same_wordform[pair] += bool(by_gramset[g]["wf"] & by_gramset[h]["wf"])
            pair_dialect[pair][dialect] += 1
    # Example surface forms (from words table) for readability
    surface = {}
    for r in zst_rows(DATA_DIR / f"corpus/words_{lang}.csv.zst"):
        surface[r["word_id"]] = r["word"]
    for word_id, rows in candidates.items():
        text_id = text_of_word.get(word_id)
        if text_id is None or text_id in test_texts or any(r["gramset_id"] in EMPTY_GRAMSET_IDS or not r["gramset"] for r in rows):
            continue
        gramsets = sorted({r["gramset"] for r in rows})
        if len({(r["wordform_id"], r["gramset"]) for r in rows}) < 2:
            continue
        for pair in combinations(gramsets, 2):
            pair_examples[pair][surface.get(word_id, "")] += 1

    kinds = gramset_kinds()
    dialects = sorted({d for lemma in paradigm.values() for d in lemma if d != "*"})
    total = population["ambiguous_tokens"]
    rows_out = []
    for pair, n in pair_tokens.most_common():
        g, h = pair
        both = same = 0
        dial = {d: [0, 0] for d in dialects}
        for lemma in paradigm.values():
            for d, cells in lemma.items():
                if g in cells and h in cells:
                    hit = bool(cells[g] & cells[h])
                    if d == "*":
                        both += 1
                        same += hit
                    else:
                        dial[d][0] += 1
                        dial[d][1] += hit
        lexicon_share = same / both if both else None
        dialect_shares = {d: (s / b if b >= MIN_DIALECT_LEMMAS else None) for d, (b, s) in dial.items()}
        same_lemma_share = pair_same_lemma[pair] / n
        type_id, reason = propose_type(pair, kinds, same_lemma_share, lexicon_share, dialect_shares)
        rows_out.append({
            "gramset_a": g, "gramset_b": h,
            "rule_id_a": gramset_id.get(g, ""), "rule_id_b": gramset_id.get(h, ""), "kind_a": kinds.get(g, ""), "kind_b": kinds.get(h, ""),
            "type_proposed": type_id, "type_reason": reason,
            "tokens_all": n, "tokens_share": round(n / total, 5),
            "tokens_annotated": pair_gold[pair], "tokens_automatic": n - pair_gold[pair],
            "token_same_lemma_share": round(same_lemma_share, 4),
            "token_same_wordform_share": round(pair_same_wordform[pair] / n, 4),
            "lexicon_lemmas_both": both, "lexicon_lemmas_same": same,
            "lexicon_same_share": "" if lexicon_share is None else round(lexicon_share, 4),
            **{f"lexicon_share_{d}": ("" if v is None else round(v, 3)) for d, v in dialect_shares.items()},
            **{f"lexicon_lemmas_{d}": dial[d][0] for d in dialects},
            "tokens_by_dialect": ";".join(f"{d}:{c}" for d, c in pair_dialect[pair].most_common()),
            "example_forms": ";".join(f for f, _ in pair_examples[pair].most_common(5)),
        })

    out.mkdir(parents=True, exist_ok=True)
    with (out / "pair_types_auto.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    summary = {
        "language": lang, "population": dict(population), "pairs": len(rows_out),
        "test_texts_excluded": len(test_texts),
        "pairs_with_20_plus_tokens": sum(r["tokens_all"] >= 20 for r in rows_out),
        "type_counts_pairs_20_plus": dict(Counter(r["type_proposed"] for r in rows_out if r["tokens_all"] >= 20)),
        "pair_token_mass_by_type": {t: sum(r["tokens_all"] for r in rows_out if r["type_proposed"] == t) for t in "12345"},
        "thresholds": {"obligatory": OBLIGATORY, "same_lemma_low": SAME_LEMMA_LOW, "min_dialect_lemmas": MIN_DIALECT_LEMMAS},
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
