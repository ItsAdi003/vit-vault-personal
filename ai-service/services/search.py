"""
Hybrid semantic search over the approved gallery images stored in Supabase.

  dense  : cosine(CLIP text embedding of the query, stored CLIP image embedding)
  sparse : BM25 over AI tags + caption + event title
  fusion : Reciprocal Rank Fusion of the two rankings
"""
import json

import numpy as np

from config import supabase
from ml import retrieval


def _as_vector(value):
    if value is None:
        return None
    if isinstance(value, str):          # pgvector columns are returned as "[0.1,0.2,...]"
        try:
            value = json.loads(value)
        except ValueError:
            return None
    arr = np.asarray(value, dtype=float)
    return arr if arr.ndim == 1 and arr.size else None


def _doc_text(row: dict) -> str:
    meta = row.get("metadata") or {}
    event = row.get("events") or {}
    parts = [" ".join(row.get("ai_tags") or []), meta.get("custom_name", ""),
             meta.get("description", ""), event.get("title", "")]
    return " ".join(p for p in parts if p)


def rank_media(query: str, query_embedding, limit: int = 60, margin: float = 0.06) -> list[dict]:
    rows = (supabase.table("media")
            .select("id, ai_tags, image_embedding, metadata, events(title)")
            .eq("status", "APPROVED").execute().data) or []
    if not rows:
        return []

    docs = [[t for t in retrieval.tokenize(_doc_text(r)) if not t.isdigit()] for r in rows]
    bm25 = retrieval.BM25Index(docs).scores(retrieval.tokenize(query))

    vectors = [_as_vector(r.get("image_embedding")) for r in rows]
    has_vec = [i for i, v in enumerate(vectors) if v is not None]
    dense = np.full(len(rows), -1.0)
    if has_vec:
        dense[has_vec] = retrieval.cosine_scores(np.asarray(query_embedding),
                                                 np.stack([vectors[i] for i in has_vec]))

    sparse_rank = [i for i in retrieval.ranking(bm25) if bm25[i] > 0]
    dense_rank = [i for i in retrieval.ranking(dense) if dense[i] > -1.0]
    fused = retrieval.rrf([sparse_rank, dense_rank])

    best = max((dense[i] for i in has_vec), default=None)
    keep = [i for i in fused
            if bm25[i] > 0 or (best is not None and dense[i] >= best - margin)]
    keep.sort(key=lambda i: -fused[i])
    return [{"id": rows[i]["id"], "score": round(fused[i], 5),
             "dense": round(float(dense[i]), 4) if dense[i] > -1.0 else None,
             "bm25": round(float(bm25[i]), 4)} for i in keep[:limit]]
