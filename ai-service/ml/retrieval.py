"""
Sparse vs dense retrieval, fusion and ranking metrics (NLP lab).

Sparse  : BM25 / TF-IDF over *words* (exact-term overlap).
Dense   : cosine similarity of CLIP embeddings (meaning as geometry).
Hybrid  : Reciprocal Rank Fusion of the two rankings.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

import numpy as np

STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were with "
    "photo photos picture pictures image images".split()
)
_WORD = re.compile(r"[a-z0-9]+")


def stem(word: str) -> str:
    """Tiny suffix stripper (enough to match dance/dancing/dances)."""
    for suf in ("ing", "ed", "es", "s"):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            word = word[: -len(suf)]
            break
    if word.endswith("e") and len(word) > 3:
        word = word[:-1]
    return word


def tokenize(text: str, use_stem: bool = True, drop_stop: bool = True) -> list[str]:
    toks = _WORD.findall((text or "").lower())
    if drop_stop:
        toks = [t for t in toks if t not in STOPWORDS]
    return [stem(t) for t in toks] if use_stem else toks


class BM25Index:
    """Okapi BM25 over pre-tokenised documents."""

    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [list(d) for d in docs]
        self.N = len(self.docs)
        self.doc_len = np.array([len(d) for d in self.docs], dtype=float)
        self.avgdl = float(self.doc_len.mean()) if self.N else 0.0
        self.tf = [Counter(d) for d in self.docs]
        df = Counter()
        for d in self.docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        s = np.zeros(self.N)
        for t in set(query_tokens):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f:
                    norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
                    s[i] += idf * f * (self.k1 + 1) / (f + norm)
        return s


def tfidf_cosine_scores(query_tokens: Sequence[str], docs: Sequence[Sequence[str]]) -> np.ndarray:
    """Plain TF-IDF (smooth idf) + cosine similarity between the query and every document."""
    N = len(docs)
    df = Counter()
    for d in docs:
        df.update(set(d))
    idf = {t: math.log((1 + N) / (1 + n)) + 1 for t, n in df.items()}

    def vec(tokens):
        c = Counter(tokens)
        return {t: (1 + math.log(f)) * idf.get(t, 0.0) for t, f in c.items() if t in idf}

    q = vec(query_tokens)
    qn = math.sqrt(sum(v * v for v in q.values())) or 1.0
    out = np.zeros(N)
    for i, d in enumerate(docs):
        dv = vec(d)
        dn = math.sqrt(sum(v * v for v in dv.values())) or 1.0
        out[i] = sum(w * dv.get(t, 0.0) for t, w in q.items()) / (qn * dn)
    return out


def cosine_scores(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = np.asarray(query_vec, dtype=float)
    M = np.asarray(matrix, dtype=float)
    q = q / (np.linalg.norm(q) or 1.0)
    M = M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-12, None)
    return M @ q


def ranking(scores: np.ndarray) -> list[int]:
    """Indices sorted by descending score (stable)."""
    return list(np.argsort(-np.asarray(scores), kind="stable"))


def rrf(rankings: Sequence[Sequence[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = Σ_r 1 / (k + rank_r(d))."""
    fused: dict[int, float] = {}
    for r in rankings:
        for pos, d in enumerate(r, start=1):
            fused[d] = fused.get(d, 0.0) + 1.0 / (k + pos)
    return fused


# ------------------------------------------------------------------ metrics
def precision_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    return sum(1 for d in ranked[:k] if d in relevant) / k if k else 0.0


def recall_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    return sum(1 for d in ranked[:k] if d in relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank(ranked: Sequence[int], relevant: set[int]) -> float:
    for pos, d in enumerate(ranked, start=1):
        if d in relevant:
            return 1.0 / pos
    return 0.0


def average_precision(ranked: Sequence[int], relevant: set[int]) -> float:
    hits, total = 0, 0.0
    for pos, d in enumerate(ranked, start=1):
        if d in relevant:
            hits += 1
            total += hits / pos
    return total / len(relevant) if relevant else 0.0


def ndcg_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    dcg = sum(1.0 / math.log2(pos + 1) for pos, d in enumerate(ranked[:k], start=1) if d in relevant)
    ideal = sum(1.0 / math.log2(pos + 1) for pos in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0
