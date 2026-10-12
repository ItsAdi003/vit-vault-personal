"""
AI Lab API — database-independent endpoints that expose every model in the
project for the AML / DL / NLP lab demos.

Nothing here touches Supabase: images are analysed in memory, so the lab keeps
working even when the hosted database is unavailable.
"""
import base64
import io
import json
import os
import threading
import time
import uuid

import imagehash
import numpy as np
from flask import Blueprint, jsonify, request
from PIL import Image, ImageOps

from config import CANDIDATE_LABELS
from ml import explain, retrieval, zeroshot
from ml.em_gmm import DiagGMM, pca_project
from ml.nb_model import BernoulliNaiveBayes
from models.clip_model import get_model
from services.classifier import (DEFAULT_LIKELIHOOD, PRIOR_PROBABILITIES,
                                 SMOOTHING_ALPHA, TAG_LIKELIHOODS, classify_category)

lab_bp = Blueprint("lab", __name__)

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml", "artifacts")
MAX_UPLOAD_IMAGES = 60
MAX_CORPUS = 200

_corpus: list[dict] = []
_lock = threading.RLock()


# ----------------------------------------------------------------- helpers
def _open_image(file_storage) -> Image.Image:
    img = Image.open(file_storage.stream)
    img.load()
    return ImageOps.exif_transpose(img).convert("RGB")  # phone photos carry a rotation flag


def _thumb_b64(img: Image.Image, size: int = 160) -> str:
    t = img.copy()
    t.thumbnail((size, size))
    buf = io.BytesIO()
    t.save(buf, format="JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _tags_from_probs(probs: np.ndarray, top_n: int = 5, min_conf: float = 0.05) -> list[str]:
    """Same rule as services/tagger.py: top-5 labels above 5 % confidence."""
    order = np.argsort(-probs)[:top_n]
    return [CANDIDATE_LABELS[i] for i in order if probs[i] > min_conf]


def _learned_nb():
    path = os.path.join(ARTIFACT_DIR, "nb_learned.json")
    if not os.path.exists(path):
        return None
    try:
        return BernoulliNaiveBayes.load(path)
    except Exception:
        return None


def _nb_payload(tags: list[str]) -> dict:
    hand = classify_category(tags)
    out = {"handcrafted": {
        "predicted": hand["predicted_category"], "confidence": hand["confidence"],
        "posterior": hand["posteriors"], "evidence": hand.get("evidence", []),
        "prior": PRIOR_PROBABILITIES,
    }, "learned": None}
    nb = _learned_nb()
    if nb is not None:
        out["learned"] = nb.explain(tags)
    return out


def _doc_tokens(item: dict) -> list[str]:
    name = os.path.splitext(item["name"])[0].replace("_", " ").replace("-", " ")
    toks = retrieval.tokenize(name + " " + " ".join(item["tags"]))
    return [t for t in toks if not t.isdigit()]


def _read_artifact_json(name: str):
    path = os.path.join(ARTIFACT_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------- info
@lab_bp.route("/api/lab/info", methods=["GET"])
def info():
    model, _ = get_model()
    vc, tc = model.config.vision_config, model.config.text_config
    count = lambda m: int(sum(p.numel() for p in m.parameters()))
    nb = _learned_nb()
    return jsonify({
        "model": "openai/clip-vit-base-patch32",
        "logit_scale": zeroshot.logit_scale(),
        "parameters": {"total": count(model), "vision": count(model.vision_model),
                       "text": count(model.text_model)},
        "vision": {"layers": vc.num_hidden_layers, "heads": vc.num_attention_heads,
                   "hidden": vc.hidden_size, "patch": vc.patch_size, "image_size": vc.image_size},
        "text": {"layers": tc.num_hidden_layers, "heads": tc.num_attention_heads,
                 "hidden": tc.hidden_size, "context": tc.max_position_embeddings},
        "embedding_dim": model.config.projection_dim,
        "labels": CANDIDATE_LABELS,
        "templates": zeroshot.PROMPT_TEMPLATES,
        "cpt": {"likelihoods": TAG_LIKELIHOODS, "priors": PRIOR_PROBABILITIES,
                "smoothing_alpha": SMOOTHING_ALPHA, "default_likelihood": DEFAULT_LIKELIHOOD},
        "learned_nb": None if nb is None else {
            "classes": nb.classes, "alpha": nb.alpha,
            "class_counts": nb.class_counts_.tolist(),
        },
        "corpus_size": len(_corpus),
    })


# ------------------------------------------------------------------ analyze
@lab_bp.route("/api/lab/analyze", methods=["POST"])
def analyze():
    f = request.files.get("image")
    if f is None:
        return jsonify({"error": "Send an image file in the 'image' form field"}), 400
    template = request.form.get("template", "bare")
    if template not in zeroshot.PROMPT_TEMPLATES:
        return jsonify({"error": f"Unknown template '{template}'"}), 400
    want_attention = request.form.get("attention", "1") != "0"

    try:
        img = _open_image(f)
    except Exception as e:  # unreadable upload
        return jsonify({"error": f"Could not read image: {e}"}), 400

    timings = {}
    t = time.perf_counter()
    feat = zeroshot.encode_images([img])
    timings["image_encoder_ms"] = round((time.perf_counter() - t) * 1000, 1)

    t = time.perf_counter()
    W = zeroshot.class_text_matrix(CANDIDATE_LABELS, zeroshot.PROMPT_TEMPLATES[template])
    probs = zeroshot.zero_shot_probs(feat, W)[0]
    timings["zero_shot_ms"] = round((time.perf_counter() - t) * 1000, 1)

    order = np.argsort(-probs)
    labels = [{"label": CANDIDATE_LABELS[i], "prob": float(probs[i]),
               "cosine": float(feat[0] @ W[i])} for i in order]
    tags = _tags_from_probs(probs)

    ph = imagehash.phash(img)
    bits = format(int(str(ph), 16), "064b")
    with _lock:
        dists = [{"name": c["name"], "distance": int(ph - c["phash"])} for c in _corpus]
    nearest = min(dists, key=lambda d: d["distance"]) if dists else None

    result = {
        "template": template,
        "labels": labels,
        "tags": tags,
        "naive_bayes": _nb_payload(tags),
        "phash": {"hex": str(ph), "bits": bits, "nearest_in_corpus": nearest, "threshold": 10},
        "embedding": {"dims": int(feat.shape[1]), "norm": float(np.linalg.norm(feat[0])),
                      "head": [round(float(v), 4) for v in feat[0][:16]]},
        "thumb": _thumb_b64(img, 320),
        "image_size": list(img.size),
    }
    if want_attention:
        try:
            t = time.perf_counter()
            ar = explain.attention_rollout(img)
            timings["attention_ms"] = round((time.perf_counter() - t) * 1000, 1)
            result["attention"] = {"overlay": explain.overlay_png_b64(img, ar["grid"]),
                                   "layers": ar["layers"], "heads": ar["heads"], "patches": ar["patches"],
                                   "grid": [[round(float(v), 3) for v in row] for row in ar["grid"]]}
        except Exception as e:
            result["attention"] = {"error": str(e)}
    result["timings"] = timings
    return jsonify(result)


@lab_bp.route("/api/lab/nb", methods=["POST"])
def nb_infer():
    data = request.get_json(silent=True) or {}
    tags = [t.lower().strip() for t in data.get("tags", []) if isinstance(t, str)]
    return jsonify({"tags": tags, **_nb_payload(tags)})


@lab_bp.route("/api/lab/tokenize", methods=["POST"])
def tokenize_text():
    text = (request.get_json(silent=True) or {}).get("text", "")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Missing text"}), 400
    _, processor = get_model()
    tk = processor.tokenizer
    full = tk(text)["input_ids"]
    ids = tk(text, truncation=True, max_length=77)["input_ids"]
    emb = zeroshot.encode_texts([text])[0]
    return jsonify({
        "text": text, "ids": ids, "tokens": tk.convert_ids_to_tokens(ids),
        "num_tokens": len(full), "truncated": len(full) > 77, "context": 77,
        "word_tokens": retrieval.tokenize(text),
        "embedding": {"dims": int(emb.shape[0]), "norm": float(np.linalg.norm(emb)),
                      "head": [round(float(v), 4) for v in emb[:12]]},
    })


# ------------------------------------------------------------------- corpus
@lab_bp.route("/api/lab/corpus", methods=["GET"])
def corpus_list():
    with _lock:
        return jsonify({"items": [{"id": c["id"], "name": c["name"], "thumb": c["thumb"],
                                   "tags": c["tags"], "phash": c["phash_hex"]} for c in _corpus]})


@lab_bp.route("/api/lab/corpus", methods=["POST"])
def corpus_add():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "Send one or more files in the 'images' form field"}), 400
    files = files[:MAX_UPLOAD_IMAGES]
    imgs, names, skipped = [], [], []
    for f in files:
        try:
            imgs.append(_open_image(f))
            names.append(f.filename or "image")
        except Exception:
            skipped.append(f.filename)
    if not imgs:
        return jsonify({"error": "None of the files could be read as images", "skipped": skipped}), 400

    feats = zeroshot.encode_images(imgs)
    W = zeroshot.class_text_matrix(CANDIDATE_LABELS, zeroshot.PROMPT_TEMPLATES["bare"])
    probs = zeroshot.zero_shot_probs(feats, W)
    added = []
    with _lock:
        for img, name, feat, p in zip(imgs, names, feats, probs):
            if len(_corpus) >= MAX_CORPUS:
                break
            ph = imagehash.phash(img)
            item = {"id": uuid.uuid4().hex[:10], "name": name, "thumb": _thumb_b64(img),
                    "tags": _tags_from_probs(p), "embedding": feat, "phash": ph, "phash_hex": str(ph)}
            _corpus.append(item)
            added.append({"id": item["id"], "name": name, "tags": item["tags"]})
    return jsonify({"added": added, "skipped": skipped, "corpus_size": len(_corpus)})


@lab_bp.route("/api/lab/corpus", methods=["DELETE"])
def corpus_clear():
    with _lock:
        _corpus.clear()
    return jsonify({"corpus_size": 0})


# ------------------------------------------------------------------- search
@lab_bp.route("/api/lab/search", methods=["POST"])
def lab_search():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Missing query"}), 400
    k = int(data.get("k", 8))
    template = data.get("template", "bare")
    if template not in zeroshot.PROMPT_TEMPLATES:
        template = "bare"

    with _lock:
        items = list(_corpus)
    if not items:
        return jsonify({"error": "The corpus is empty. Add some photos first."}), 400
    q_low = query.lower()
    keyword = np.array([1.0 if (q_low in it["name"].lower() or any(q_low in t for t in it["tags"])) else 0.0
                        for it in items])

    docs = [_doc_tokens(it) for it in items]
    q_tokens = retrieval.tokenize(query)
    bm25 = retrieval.BM25Index(docs).scores(q_tokens)
    tfidf = retrieval.tfidf_cosine_scores(q_tokens, docs)

    phrases = [t.format(query) for t in zeroshot.PROMPT_TEMPLATES[template]]
    q_emb = zeroshot.encode_texts(phrases).mean(axis=0)
    dense = retrieval.cosine_scores(q_emb, np.stack([it["embedding"] for it in items]))

    sparse_rank = [i for i in retrieval.ranking(bm25) if bm25[i] > 0]
    fused = retrieval.rrf([sparse_rank, retrieval.ranking(dense)])
    hybrid = np.zeros(len(items))
    for i, s in fused.items():
        hybrid[i] = s

    def pack(scores, only_positive=False):
        out = []
        for rank, i in enumerate(retrieval.ranking(scores)[:k], start=1):
            if only_positive and scores[i] <= 0:
                break
            out.append({"id": items[i]["id"], "name": items[i]["name"],
                        "score": float(scores[i]), "rank": rank})
        return out

    return jsonify({
        "query": query,
        "query_tokens": q_tokens,
        "raw_tokens": retrieval.tokenize(query, use_stem=False, drop_stop=False),
        "template": template,
        "keyword": pack(keyword, only_positive=True),
        "tfidf": pack(tfidf, only_positive=True),
        "bm25": pack(bm25, only_positive=True),
        "dense": pack(dense),
        "hybrid": pack(hybrid),
        "corpus_size": len(items),
    })


@lab_bp.route("/api/lab/embedding-map", methods=["GET"])
def embedding_map():
    with _lock:
        items = list(_corpus)
    n = len(items)
    if n < 4:
        return jsonify({"error": "Add at least 4 photos to build the embedding map"}), 400
    E = np.stack([it["embedding"] for it in items])
    Z, ratio, _, _ = pca_project(E, 2)
    # EM works in a 10-D PCA subspace: 512-D diagonal Gaussians are far too flexible for a handful of photos.
    P, *_ = pca_project(E, min(10, n - 1))
    max_k = max(1, min(6, n // 2))
    requested = request.args.get("k", "auto")
    bics, fits = {}, {}
    for kk in range(1, max_k + 1):
        g = DiagGMM(kk, seed=0).fit(P)
        bics[kk], fits[kk] = g.bic(P), g
    chosen = int(requested) if requested.isdigit() and int(requested) in fits else min(bics, key=bics.get)
    g = fits[chosen]
    labels = g.predict(P)
    clusters = []
    for c in range(chosen):
        members = [items[i] for i in range(n) if labels[i] == c]
        counts: dict[str, int] = {}
        for m in members:
            for t in m["tags"]:
                counts[t] = counts.get(t, 0) + 1
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
        clusters.append({"cluster": c, "size": len(members), "top_tags": [t for t, _ in top],
                         "weight": float(g.weights_[c])})
    return jsonify({
        "points": [{"id": it["id"], "name": it["name"], "x": float(Z[i, 0]), "y": float(Z[i, 1]),
                    "cluster": int(labels[i]), "tags": it["tags"]} for i, it in enumerate(items)],
        "explained_variance": [float(r) for r in ratio],
        "bic": {str(k): float(v) for k, v in bics.items()},
        "chosen_k": chosen, "auto_k": min(bics, key=bics.get),
        "log_likelihood_curve": [float(v) for v in g.log_likelihoods_],
        "em_iterations": g.n_iter_, "clusters": clusters,
    })


@lab_bp.route("/api/lab/benchmarks", methods=["GET"])
def benchmarks():
    names = ["dl_cifar10", "aml_nb_recovery", "aml_em_gmm", "nlp_paraphrase", "nlp_crossmodal", "nb_learned_eval"]
    return jsonify({n: _read_artifact_json(n + ".json") for n in names})
