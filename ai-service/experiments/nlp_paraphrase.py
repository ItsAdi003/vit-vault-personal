"""
NLP experiment 1: paraphrase retrieval, sparse vs dense.

48 campus-event phrases = 12 intents x 4 paraphrases. Every phrase is used as a
query against the other 47; the three phrases of the same intent are relevant.

Methods:  TF-IDF (word overlap) | BM25 | CLIP text tower (dense) | Hybrid RRF(BM25 + CLIP)
Optional: --minilm adds a dedicated sentence encoder (all-MiniLM-L6-v2, ~90 MB download on first use).

Run:  python experiments/nlp_paraphrase.py [--minilm]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _report import print_table, save_report  # noqa: E402

from ml import retrieval as R  # noqa: E402
from ml import zeroshot  # noqa: E402
from models.clip_model import load_model  # noqa: E402

INTENTS = {
    "dance": ["students performing a dance on stage", "cultural fest dance competition",
              "a group of people dancing at the annual day", "choreography show with performers"],
    "hackathon": ["participants coding on laptops at a hackathon", "overnight programming contest",
                  "teams building software late at night", "developers competing to build an app"],
    "seminar": ["a guest lecture in the auditorium", "speaker presenting to a seated audience",
                "seminar with slides on a projector", "professor delivering a talk to students"],
    "graduation": ["convocation ceremony with caps and gowns", "students receiving their degrees",
                   "graduation day group photo", "degree distribution function"],
    "sports": ["football match on the college ground", "athletes running in a track race",
               "inter-college cricket tournament", "students playing outdoor games"],
    "award": ["trophy being handed over on stage", "prize distribution ceremony",
              "winners receiving medals and certificates", "felicitation of the toppers"],
    "food": ["canteen food stall at the fest", "students eating lunch together",
             "snacks and refreshments being served", "food court during the event"],
    "lab": ["students doing experiments in the laboratory", "electronics workshop with circuit boards",
            "practical session in the computer lab", "hands-on lab work with instruments"],
    "music": ["band performing live on stage", "singing competition with a microphone",
              "musical night with guitar and drums", "concert by the college music club"],
    "poster": ["poster presentation exhibition", "project display with charts on boards",
               "science exhibition stalls", "research posters explained to visitors"],
    "group": ["group photo of the whole team", "everyone posing together for a picture",
              "class photograph outdoors", "friends smiling together for a snapshot"],
    "night": ["evening celebration with lights", "night event with colourful stage lighting",
              "festival after dark", "illuminated campus at night"],
}


def evaluate(score_fn, n, labels):
    ap, rr, p3, nd = [], [], [], []
    for q in range(n):
        scores = np.array(score_fn(q), dtype=float)
        ranked = [i for i in R.ranking(scores) if i != q]
        rel = {j for j in range(n) if labels[j] == labels[q] and j != q}
        ap.append(R.average_precision(ranked, rel))
        rr.append(R.reciprocal_rank(ranked, rel))
        p3.append(R.precision_at_k(ranked, rel, 3))
        nd.append(R.ndcg_at_k(ranked, rel, 3))
    return [float(np.mean(ap)), float(np.mean(rr)), float(np.mean(p3)), float(np.mean(nd))]


def main():
    use_minilm = "--minilm" in sys.argv
    items = [(intent, text) for intent, texts in INTENTS.items() for text in texts]
    rng = np.random.default_rng(7)
    rng.shuffle(items)                       # avoid tie-break bias from grouped ordering
    labels = [i for i, _ in items]
    texts = [t for _, t in items]
    n = len(texts)

    toks = [R.tokenize(t) for t in texts]
    bm25 = R.BM25Index(toks)
    load_model()
    E = zeroshot.encode_texts(texts)
    dense = E @ E.T

    def f_tfidf(q): return R.tfidf_cosine_scores(toks[q], toks)
    def f_bm25(q): return bm25.scores(toks[q])
    def f_dense(q): return dense[q]

    def f_hybrid(q):
        sparse_rank = [i for i in R.ranking(bm25.scores(toks[q])) if i != q and bm25.scores(toks[q])[i] > 0]
        dense_rank = [i for i in R.ranking(dense[q]) if i != q]
        fused = R.rrf([sparse_rank, dense_rank])
        s = np.zeros(n)
        for i, v in fused.items():
            s[i] = v
        return s

    methods = [("TF-IDF (sparse)", f_tfidf), ("BM25 (sparse)", f_bm25),
               ("CLIP text tower (dense)", f_dense), ("Hybrid RRF (BM25 + CLIP)", f_hybrid)]

    if use_minilm:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        M = m.encode(texts, normalize_embeddings=True)
        mini = M @ M.T
        methods.append(("MiniLM sentence encoder (dense)", lambda q: mini[q]))

    # How often do two paraphrases share no content word at all? (the ceiling for sparse methods)
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n) if labels[a] == labels[b]]
    no_overlap = float(np.mean([len(set(toks[a]) & set(toks[b])) == 0 for a, b in pairs]))

    rows = [[name] + evaluate(fn, n, labels) for name, fn in methods]
    cols = ["method", "MAP", "MRR", "P@3", "nDCG@3"]
    print(f"{n} phrases, {len(INTENTS)} intents; paraphrase pairs with zero shared stems: {no_overlap:.1%}")
    print_table(cols, rows)

    path = save_report("nlp_paraphrase", {
        "title": "Paraphrase retrieval on campus-event phrases: sparse vs dense",
        "description": "48 phrases (12 intents x 4 paraphrases) retrieve each other. Sparse methods can only match shared "
                       "words; dense methods compare meaning in embedding space.",
        "meta": {"phrases": n, "intents": len(INTENTS), "relevant per query": 3,
                 "paraphrase pairs with no shared word": pct(no_overlap)},
        "table": {"columns": cols, "rows": rows},
        "notes": "Dense CLIP beats both sparse rankers by a wide margin because most paraphrase pairs share no word. "
                 "Equal-weight RRF is worse than dense alone here: when one ranker is mostly uninformative it dilutes "
                 "the good one, so fusion should be weighted or gated. Fusion is expected to pay off when queries do "
                 "reuse document vocabulary (exact tag names). CLIP's text tower is trained to align text with "
                 "images, not as a general sentence encoder; a dedicated model such as MiniLM may do better on pure "
                 "text-text similarity (run with --minilm to measure it).",
    })
    print("saved", path)


def pct(x):
    return f"{x * 100:.1f}%"


if __name__ == "__main__":
    main()
