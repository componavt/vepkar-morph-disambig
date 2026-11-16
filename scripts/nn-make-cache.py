#!/usr/bin/env python3
"""Build a slim corpus cache for the neural branch and N4.

The full T1x cache (``t1x-v1/corpus.pkl``) holds the whole corpus and needs about
10 GB of RAM once unpickled. The neural models and N4 only need the sentences of
train and dev benchmark items, the candidates of their words and the tables used
by the evaluation. The slim cache keeps exactly that.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1x_lib as L  # noqa: E402

SLIM = L.EXP / "t1x-v1/corpus-slim.pkl"


def main():
    full = pickle.load(open(L.EXP / "t1x-v1/corpus.pkl", "rb"))
    slim = L.Corpus()
    keys = {(it.lang, it.sentence_id) for it in full.train + full.dev}
    words = set()
    for key in keys:
        slim.sentences[key] = full.sentences[key]
        slim.sentence_text[key] = full.sentence_text[key]
        words.update((key[0], t.word_id) for t in full.sentences[key])
    wfs = set()
    for key in words:
        rows = full.cands.get(key)
        if rows is not None:
            slim.cands[key] = rows
            wfs.update((key[0], r[0], r[1]) for r in rows)
    for lang, wf, gid in wfs:
        k = (lang, wf, gid)
        if k in full.lemmas:
            slim.lemmas[k] = full.lemmas[k]
        if k in full.wf_dialects:
            slim.wf_dialects[k] = full.wf_dialects[k]
        if (lang, wf) in full.wf_string:
            slim.wf_string[(lang, wf)] = full.wf_string[(lang, wf)]
    slim.lemma_pos = full.lemma_pos
    slim.split_of_text = full.split_of_text
    slim.obligatory_pairs = full.obligatory_pairs
    slim.text_dialect = full.text_dialect
    slim.train, slim.dev = full.train, full.dev
    with SLIM.open("wb") as h:
        pickle.dump(slim, h, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"sentences {len(slim.sentences)}, words {len(slim.cands)}, wordforms {len(slim.wf_string)} -> {SLIM}")


if __name__ == "__main__":
    main()
