"""T1x: experiment framework for candidate ranking along four axes.

Axes
  * how we learn   -- pointwise logistic regression (as T1.1) or listwise
                      conditional logit (softmax over the candidates of a token);
  * what we see    -- base T1.1 features, original spelling from sentence_xml,
                      factored gramset x context, neighbour candidates, lexicon;
  * where data from-- benchmark train, unambiguous relatives (paradigm transfer),
                      self-training on unannotated tokens, learning curves;
  * how we decide  -- per word, or a two-pass "chorus" that sees predicted
                      gramsets of neighbouring words.

Only train and dev are used. Test texts are excluded from every auxiliary data
source (unannotated and unambiguous tokens included); dev texts are excluded
from auxiliary training data as well.
"""

from __future__ import annotations

import csv
import html
import io
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.optimize
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_context import load_benchmark_context  # noqa: E402

DATA_DIR = ROOT / "data/dictorpus-data"
TAG = "v2026.11"
LANGS = ("krl", "lud", "olo", "vep")
EMPTY_GRAMSET_IDS = {"0", "7"}
SUFFIX_K = 5


def zst_rows(path: Path):
    raw = subprocess.run(["zstdcat", str(path)], check=True, capture_output=True, text=True).stdout
    return csv.DictReader(io.StringIO(raw))


def xml_tokens(sentence_xml: str) -> list[str]:
    return [html.unescape(re.sub(r"<[^>]+>", "", body))
            for body in re.findall(r"<w[^>]*>(.*?)</w>", sentence_xml, flags=re.S)]


# ----------------------------------------------------------------------------- data

@dataclass
class Token:
    word_id: int
    norm: str
    orig: str          # original spelling, lower-cased (apostrophes, hyphens kept)
    cap: bool          # capitalised and not sentence-initial


@dataclass
class Item:
    """One ranking problem: a token with its candidate list."""
    lang: str
    word_id: int
    text_id: int
    sentence_id: int
    pos: int                                  # 0-based index in sentence tokens
    candidates: list[tuple[int, str, str]]    # (wordform_id, gramset_id, gramset)
    gold: int | None                          # index into candidates
    weight: float = 1.0
    context_only: bool = False                # transfer items: no form features
    source: str = "benchmark"


@dataclass
class Corpus:
    sentences: dict[tuple[str, int], list[Token]] = field(default_factory=dict)
    sentence_text: dict[tuple[str, int], int] = field(default_factory=dict)
    word_pos: dict[tuple[str, int], tuple[int, int]] = field(default_factory=dict)
    cands: dict[tuple[str, int], list[tuple[int, str, str, str]]] = field(default_factory=dict)
    lemmas: dict[tuple[str, int, str], frozenset] = field(default_factory=dict)
    wf_string: dict[tuple[str, int], set] = field(default_factory=dict)
    lemma_pos: dict[tuple[str, str], str] = field(default_factory=dict)
    split_of_text: dict[tuple[str, int], str] = field(default_factory=dict)
    obligatory_pairs: dict[str, set] = field(default_factory=dict)
    cell_multiplicity: dict[tuple[str, str, str], int] = field(default_factory=dict)
    text_dialect: dict[tuple[str, int], str] = field(default_factory=dict)
    wf_dialects: dict[tuple[str, int, str], frozenset] = field(default_factory=dict)
    train: list[Item] = field(default_factory=list)
    dev: list[Item] = field(default_factory=list)


def load_corpus(log=print) -> Corpus:
    corpus = Corpus()
    log("loading benchmark instances and split ...")
    bench = load_benchmark_context(DATA_DIR, TAG, include_words=False)
    for lang, text_id, split in bench.split_rows:
        corpus.split_of_text[(lang, int(text_id))] = split
    for lang in LANGS:
        log(f"loading {lang} tables ...")
        xml_by_sentence = {}
        for r in zst_rows(DATA_DIR / f"corpus/sentences_{lang}.csv.zst"):
            sid = int(r["sentence_id"])
            corpus.sentence_text[(lang, sid)] = int(r["text_id"])
            xml_by_sentence[sid] = r["sentence_xml"]
        words = defaultdict(list)
        for r in zst_rows(DATA_DIR / f"corpus/words_{lang}.csv.zst"):
            words[int(r["sentence_id"])].append((int(r["word_number"]), int(r["word_id"]), r["word"]))
        for sid, rows in words.items():
            rows.sort()
            orig = xml_tokens(xml_by_sentence.get(sid, ""))
            aligned = len(orig) == len(rows)
            tokens = []
            for index, (_, wid, norm) in enumerate(rows):
                o = orig[index] if aligned else norm
                tokens.append(Token(wid, norm, o.lower(), aligned and index > 0 and o[:1].isupper()))
                corpus.word_pos[(lang, wid)] = (sid, index)
            corpus.sentences[(lang, sid)] = tokens
        for r in zst_rows(DATA_DIR / f"corpus/candidate_analyses_{lang}.csv.zst"):
            corpus.cands.setdefault((lang, int(r["word_id"])), []).append(
                (int(r["wordform_id"]), r["gramset_id"], r["gramset"], r["relevance"]))
        for r in zst_rows(DATA_DIR / f"dict/morphology/lemmas_{lang}.csv.zst"):
            corpus.lemma_pos[(lang, r["lemma_id"])] = r["pos"]
        for r in zst_rows(DATA_DIR / f"corpus/texts_{lang}.csv.zst"):
            corpus.text_dialect[(lang, int(r["text_id"]))] = r["dialect_code"] or ""
        lem = defaultdict(set)
        dia = defaultdict(set)
        for r in zst_rows(DATA_DIR / f"dict/morphology/lemma_wordforms_{lang}.csv.zst"):
            wf = int(r["wordform_id"])
            lem[(lang, wf, r["gramset_id"])].add(r["lemma_id"])
            dia[(lang, wf, r["gramset_id"])].add(r["dialect_code"])
            corpus.wf_string.setdefault((lang, wf), set()).add(r["wordform"].lower())
        for key, value in lem.items():
            corpus.lemmas[key] = frozenset(value)
        for key, value in dia.items():
            corpus.wf_dialects[key] = frozenset(value)
        # paradigm cell multiplicity: how many cells of the lemma share this string
        by_lemma_string = Counter()
        for (lg, wf, gid), lemmas in lem.items():
            for lemma in lemmas:
                by_lemma_string[(lemma, wf)] += 1
        for (lg, wf, gid), lemmas in lem.items():
            corpus.cell_multiplicity[(lang, str(wf), gid)] = max(by_lemma_string[(l, wf)] for l in lemmas)
        pair_file = ROOT / f"work/experiments/v2026.11/s1-pair-types-v1/{lang}/pair_types_auto.csv"
        with pair_file.open(encoding="utf-8") as handle:
            corpus.obligatory_pairs[lang] = {
                frozenset((r["gramset_a"], r["gramset_b"])) for r in csv.DictReader(handle) if r["type_proposed"] == "1"}
    log("building benchmark items ...")
    for inst in bench.instances:
        split = corpus.split_of_text.get((inst.language, int(inst.text_id)))
        if split not in ("train", "dev"):
            continue
        item = make_item(corpus, inst.language, int(inst.word_id),
                         gold_key=(int(inst.gold_analysis.wordform_id), inst.gold_analysis.gramset),
                         allowed=[(int(c.wordform_id), c.gramset) for c in inst.candidates])
        if item is None:
            continue
        (corpus.train if split == "train" else corpus.dev).append(item)
    log(f"train items {len(corpus.train)}, dev items {len(corpus.dev)}")
    return corpus


def make_item(corpus, lang, word_id, gold_key=None, allowed=None, **kw) -> Item | None:
    where = corpus.word_pos.get((lang, word_id))
    if where is None or (lang, where[0]) not in corpus.sentence_text:
        return None
    sid, pos = where
    raw = corpus.cands.get((lang, word_id), [])
    seen, cands = set(), []
    for wf, gid, g, _ in raw:
        if (wf, g) in seen or not g:
            continue
        if allowed is not None and (wf, g) not in allowed:
            continue
        seen.add((wf, g))
        cands.append((wf, gid, g))
    if allowed is not None:  # keep benchmark order/identity exactly
        lookup = {(wf, g): gid for wf, gid, g in cands}
        cands = [(wf, lookup.get((wf, g), ""), g) for wf, g in allowed]
    cands.sort(key=lambda c: (c[0], c[2]))
    gold = None
    if gold_key is not None:
        gold = next(i for i, c in enumerate(cands) if (c[0], c[2]) == gold_key)
    return Item(lang, word_id, corpus.sentence_text[(lang, sid)], sid, pos, cands, gold, **kw)


# ------------------------------------------------------------------------- features

@dataclass(frozen=True)
class FeatureConfig:
    spelling: str = "norm"        # "norm" (T1.1) or "orig" (sentence_xml)
    factored: bool = False        # gramset elements x context
    neighbours: bool = False      # candidate lemmas / gramsets of +-1 neighbours
    lexicon: bool = False         # dictionary paradigm features of the candidate
    chorus: bool = False          # predicted gramsets of +-1 neighbours (2nd pass)
    suffix_k: int = SUFFIX_K
    transfer_context_only: bool = False  # transfer items do not touch prior features
    cues: bool = False            # S3 literature cue lexicon (vep), window +-3
    dialect: bool = False         # does the candidate's dictionary record fit the text dialect?


def candidate_features(corpus, item: Item, cand, cfg: FeatureConfig, predicted=None) -> list[str]:
    wf, gid, g = cand
    toks = corpus.sentences[(item.lang, item.sentence_id)]
    use_orig = cfg.spelling == "orig"
    tok = toks[item.pos]
    word = tok.orig if use_orig else tok.norm

    def neighbour(offset):
        j = item.pos + offset
        if j < 0:
            return "<BOS>", None
        if j >= len(toks):
            return "<EOS>", None
        t = toks[j]
        return (t.orig if use_orig else t.norm), t.word_id

    left, left_id = neighbour(-1)
    right, right_id = neighbour(1)
    elements = g.split("+")
    f = [f"L|{g}|{left}", f"R|{g}|{right}"]
    if not (item.context_only and cfg.transfer_context_only):
        # prior (bias) features; transfer items may be barred from shifting priors,
        # because their gramset distribution differs from benchmark candidate lists
        f += [f"g|{g}", f"lang|{item.lang}", f"lg|{item.lang}|{g}"]
        f += [f"el|{e}" for e in elements]
    if not item.context_only:
        for k in range(1, min(cfg.suffix_k, len(word)) + 1):
            f.append(f"suf|{g}|{k}|{word[-k:]}")
        # NB: a "candidate spelling == text spelling" feature was tried and removed:
        # among same-lemma spelling duplicates experts pick the smaller wordform_id
        # (~98% in vep train), not the record matching the text spelling.
        if use_orig and tok.cap:
            f.append(f"cap|{g}")
    if cfg.factored:
        for e in elements:
            f += [f"Le|{e}|{left}", f"Re|{e}|{right}"]
    if cfg.neighbours:
        for side, nid in (("L", left_id), ("R", right_id)):
            if nid is None:
                continue
            ncands = corpus.cands.get((item.lang, nid), [])
            ngs = {c[2] for c in ncands if c[2]}
            nlem = set()
            for nwf, ngid, _, _ in ncands:
                nlem |= corpus.lemmas.get((item.lang, nwf, ngid), frozenset())
            for ng in ngs:
                f.append(f"{side}ng|{g}|{ng}")
            for lemma in list(nlem)[:6]:
                f.append(f"{side}lem|{g}|{lemma}")
            if len(ngs) == 1:
                f.append(f"{side}ng1|{g}|{next(iter(ngs))}")
    if cfg.lexicon and not item.context_only:
        lemmas = corpus.lemmas.get((item.lang, wf, gid), frozenset())
        for lemma in lemmas:
            f.append(f"lemg|{lemma}|{g}")
            f.append(f"pos|{corpus.lemma_pos.get((item.lang, lemma), '?')}|{g}")
        mult = corpus.cell_multiplicity.get((item.lang, str(wf), gid), 0)
        f.append(f"mult|{min(mult, 4)}|{g.split('+')[0]}")
    if cfg.dialect and not item.context_only:
        td = corpus.text_dialect.get((item.lang, item.text_id), "")
        ds = corpus.wf_dialects.get((item.lang, wf, gid), frozenset())
        if not td or "|" in td or not ds or r"\N" in ds:
            status = "unknown"
        else:
            status = "match" if td in ds else "other"
        kind = "new" if td.endswith("-new") else "dial"
        f.append(f"dia|{item.lang}|{kind}|{status}")
    if cfg.cues and item.lang == "vep":
        for cls in cue_classes(corpus, item):
            f.append(f"cue|{cls}|{g}")
            f.append(f"cue_any|{g}")
    if cfg.chorus and predicted is not None:
        for side, nid in (("L", left_id), ("R", right_id)):
            pg = predicted.get((item.lang, nid), "<B>") if nid is not None else "<B>"
            f.append(f"{side}pg|{g}|{pg}")
            for e in elements:
                f.append(f"{side}pge|{e}|{pg}")
    return f


class Vocab:
    def __init__(self):
        self.index = {}

    def fit(self, feature_lists):
        for feats in feature_lists:
            for name in feats:
                if name not in self.index:
                    self.index[name] = len(self.index)
        return self

    def transform(self, feature_lists):
        rows, cols = [], []
        for r, feats in enumerate(feature_lists):
            for name in set(feats):
                c = self.index.get(name)
                if c is not None:
                    rows.append(r)
                    cols.append(c)
        data = np.ones(len(rows), dtype=np.float64)
        return sp.csr_matrix((data, (rows, cols)), shape=(len(feature_lists), len(self.index)))


def featurize(corpus, items, cfg, predicted=None):
    feats, group = [], []
    for gi, item in enumerate(items):
        for cand in item.candidates:
            feats.append(candidate_features(corpus, item, cand, cfg, predicted))
            group.append(gi)
    return feats, np.asarray(group, dtype=np.int64)


# --------------------------------------------------------------------------- models

class PointwiseModel:
    def __init__(self, C):
        self.C = C

    def fit(self, X, items, group):
        y = np.zeros(X.shape[0])
        w = np.zeros(X.shape[0])
        offset = 0
        for item in items:
            n = len(item.candidates)
            y[offset + item.gold] = 1
            w[offset:offset + n] = item.weight / n
            offset += n
        self.clf = LogisticRegression(C=self.C, solver="lbfgs", max_iter=2000)
        self.clf.fit(X, y, sample_weight=w)
        return self

    def score(self, X):
        return self.clf.decision_function(X)


class ListwiseModel:
    """Conditional logit: softmax over the candidates of each token, L2 penalty."""

    def __init__(self, C):
        self.C = C

    def fit(self, X, items, group):
        X = X.tocsr()
        n_groups = len(items)
        gold_rows = np.zeros(n_groups, dtype=np.int64)
        weights = np.zeros(n_groups)
        starts = np.zeros(n_groups, dtype=np.int64)
        offset = 0
        for i, item in enumerate(items):
            starts[i] = offset
            gold_rows[i] = offset + item.gold
            weights[i] = item.weight
            offset += len(item.candidates)
        row_weight = weights[group]
        Xt = X.T.tocsr()

        def objective(w):
            s = X @ w
            gmax = np.maximum.reduceat(s, starts)
            e = np.exp(s - gmax[group])
            z = np.add.reduceat(e, starts)
            p = e / z[group]
            loss = -np.sum(weights * (s[gold_rows] - gmax - np.log(z)))
            r = row_weight * p
            r[gold_rows] -= weights
            grad = Xt @ r
            return self.C * loss + 0.5 * w @ w, self.C * grad + w

        w0 = np.zeros(X.shape[1])
        res = scipy.optimize.minimize(objective, w0, jac=True, method="L-BFGS-B",
                                      options={"maxiter": 3000})
        self.w = res.x
        self.converged = bool(res.success)
        return self

    def score(self, X):
        return X @ self.w


def rank(items, scores, group):
    """Top-1 index per item with T1 tie-break (score desc, wordform_id, gramset)."""
    best = []
    offset = 0
    for item in items:
        n = len(item.candidates)
        s = scores[offset:offset + n]
        order = sorted(range(n), key=lambda i: (-s[i], item.candidates[i][0], item.candidates[i][2]))
        best.append((order, s))
        offset += n
    return best


# -------------------------------------------------------------------------- metrics

def correct_modulo_syncretism(corpus, item, top) -> bool:
    if top == item.gold:
        return True
    gwf, ggid, gg = item.candidates[item.gold]
    twf, tgid, tg = item.candidates[top]
    if gwf == twf and frozenset((gg, tg)) in corpus.obligatory_pairs[item.lang]:
        return True
    if gg == tg:  # spelling duplicates of the same lemma
        a = corpus.lemmas.get((item.lang, gwf, ggid), frozenset())
        b = corpus.lemmas.get((item.lang, twf, tgid), frozenset())
        return bool(a & b)
    return False


def evaluate(corpus, items, ranked):
    rows = []
    for item, (order, _) in zip(items, ranked):
        top = order[0]
        rows.append({
            "lang": item.lang, "text_id": item.text_id, "word_id": item.word_id,
            "full": int(top == item.gold),
            "sync": int(correct_modulo_syncretism(corpus, item, top)),
            "rr": 1.0 / (order.index(item.gold) + 1),
            "top_gramset": item.candidates[top][2], "gold_gramset": item.candidates[item.gold][2],
            "top_wordform": item.candidates[top][0], "gold_wordform": item.candidates[item.gold][0],
        })
    return rows


def summarize(rows):
    out = {}
    for key, subset in [("all", rows)] + [(l, [r for r in rows if r["lang"] == l]) for l in LANGS]:
        n = len(subset)
        out[key] = {"n": n, "full": sum(r["full"] for r in subset) / n,
                    "sync": sum(r["sync"] for r in subset) / n,
                    "mrr": sum(r["rr"] for r in subset) / n}
    return out


def paired_bootstrap(rows_a, rows_b, metric="full", reps=2000, seed=0):
    """Text-level paired bootstrap of mean(b) - mean(a); returns (delta, lo, hi)."""
    by_text = defaultdict(lambda: [0.0, 0.0, 0])
    for a, b in zip(rows_a, rows_b):
        assert a["word_id"] == b["word_id"]
        t = by_text[(a["lang"], a["text_id"])]
        t[0] += a[metric]
        t[1] += b[metric]
        t[2] += 1
    arr = np.asarray(list(by_text.values()))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(reps, len(arr)))
    sa, sb, n = arr[idx, 0].sum(1), arr[idx, 1].sum(1), arr[idx, 2].sum(1)
    deltas = (sb - sa) / n
    delta = (arr[:, 1].sum() - arr[:, 0].sum()) / arr[:, 2].sum()
    return delta, float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


# ------------------------------------------------------------- auxiliary data sources

def allowed_aux_text(corpus, lang, text_id) -> bool:
    """Auxiliary data may come from train texts and texts outside the split file."""
    return corpus.split_of_text.get((lang, text_id)) not in ("dev", "test")


def gramset_ids(corpus):
    ids = {}
    for (lang, word_id), rows in corpus.cands.items():
        for wf, gid, g, _ in rows:
            if g:
                ids[g] = gid
    return ids


def transfer_items(corpus, cap_per_gramset=2000, min_pair=3, weight=0.5, seed=0):
    """Unambiguous relatives: single-candidate tokens with pseudo-competitors.

    The competitors are gramsets that conflict with the token's gramset in
    benchmark train candidate lists; features are context-only, so the item
    teaches in which contexts a gramset occurs, not what its form looks like.
    """
    import random
    partners = defaultdict(Counter)
    for item in corpus.train:
        gs = {c[2] for c in item.candidates}
        for a in gs:
            for b in gs:
                if a != b:
                    partners[(item.lang, a)][b] += 1
    gid = gramset_ids(corpus)
    pools = defaultdict(list)
    for (lang, word_id), rows in corpus.cands.items():
        keys = {(wf, g) for wf, _, g, _ in rows}
        if len(keys) != 1 or not rows[0][2] or rows[0][1] in EMPTY_GRAMSET_IDS:
            continue
        where = corpus.word_pos.get((lang, word_id))
        if where is None or (lang, where[0]) not in corpus.sentence_text:
            continue  # orphan words (sentence missing in export)
        text_id = corpus.sentence_text[(lang, where[0])]
        if not allowed_aux_text(corpus, lang, text_id):
            continue
        g = rows[0][2]
        comp = [h for h, n in partners[(lang, g)].items() if n >= min_pair
                and frozenset((g, h)) not in corpus.obligatory_pairs[lang]]
        if comp:
            pools[(lang, g)].append((word_id, rows[0][0], rows[0][1], comp))
    rng = random.Random(seed)
    items = []
    for key, pool in sorted(pools.items()):
        rng.shuffle(pool)
        for word_id, wf, g_id, comp in pool[:cap_per_gramset]:
            lang, g = key
            sid, pos = corpus.word_pos[(lang, word_id)]
            cands = [(wf, g_id, g)] + [(0, gid.get(h, ""), h) for h in comp]
            cands.sort(key=lambda c: (c[0], c[2]))
            gold = next(i for i, c in enumerate(cands) if c[2] == g and c[0] == wf)
            items.append(Item(lang, word_id, corpus.sentence_text[(lang, sid)], sid, pos, cands, gold,
                              weight=weight, context_only=True, source="transfer"))
    return items


def unannotated_items(corpus, cap_per_lang=None, seed=0):
    """Ambiguous tokens without an expert choice, from allowed auxiliary texts."""
    import random
    out = defaultdict(list)
    for (lang, word_id), rows in corpus.cands.items():
        if any(r[3] in ("0", "2") for r in rows):
            continue
        if any((not r[2]) or r[1] in EMPTY_GRAMSET_IDS for r in rows):
            continue
        if len({(r[0], r[2]) for r in rows}) < 2:
            continue
        where = corpus.word_pos.get((lang, word_id))
        if where is None or (lang, where[0]) not in corpus.sentence_text or not allowed_aux_text(corpus, lang, corpus.sentence_text[(lang, where[0])]):
            continue
        item = make_item(corpus, lang, word_id)
        if item is not None:
            out[lang].append(item)
    rng = random.Random(seed)
    items = []
    for lang in sorted(out):
        pool = out[lang]
        rng.shuffle(pool)
        items += pool[:cap_per_lang] if cap_per_lang else pool
    return items


def sentence_items(corpus, items):
    """Generic items for every word with candidates in the sentences of ``items``."""
    seen, out = set(), []
    for item in items:
        key = (item.lang, item.sentence_id)
        if key in seen:
            continue
        seen.add(key)
        for tok in corpus.sentences[key]:
            it = make_item(corpus, item.lang, tok.word_id)
            if it is not None and it.candidates:
                out.append(it)
    return out


_CUE_LEMMA = None


def cue_classes(corpus, item, window=3):
    global _CUE_LEMMA
    if _CUE_LEMMA is None:
        path = ROOT / "work/experiments/v2026.11/s3-cue-lexicon-v1/vep/cue_lexicon_vep.csv"
        _CUE_LEMMA = {}
        with path.open(encoding="utf-8") as h:
            for r in csv.DictReader(h):
                for lid in r["lemma_ids"].split("|"):
                    _CUE_LEMMA[lid] = r["cue_class"]
    toks = corpus.sentences[(item.lang, item.sentence_id)]
    found = set()
    for off in range(-window, window + 1):
        j = item.pos + off
        if off == 0 or not 0 <= j < len(toks):
            continue
        for wf, gid, g, _ in corpus.cands.get((item.lang, toks[j].word_id), []):
            for lemma in corpus.lemmas.get((item.lang, wf, gid), ()):
                if lemma in _CUE_LEMMA:
                    found.add(_CUE_LEMMA[lemma])
    return found
