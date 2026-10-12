"""
Batch CLIP encoding and zero-shot classification with prompt templates (DL + NLP labs).

Zero-shot classifier built from text:
    w_c = normalise( mean_t  normalise( TextEncoder(template_t.format(label_c)) ) )
    p(c | image) = softmax_c( s · <normalise(ImageEncoder(image)), w_c> )
where s = exp(logit_scale) ≈ 100 is the temperature learned during CLIP pre-training.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from models.clip_model import get_model

PROMPT_TEMPLATES: dict[str, list[str]] = {
    "bare": ["{}"],
    "photo": ["a photo of {}."],
    "event": ["a photo of {} at a college event."],
    "ensemble": [
        "{}",
        "a photo of {}.",
        "a photo showing {}.",
        "a picture of {}.",
        "a photo taken at a college event showing {}.",
    ],
}

_text_cache: dict[tuple, np.ndarray] = {}


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def logit_scale() -> float:
    model, _ = get_model()
    return float(model.logit_scale.exp().item())


@torch.no_grad()
def encode_texts(texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
    model, processor = get_model()
    out = []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i:i + batch_size])
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True)
        out.append(model.get_text_features(**inputs).cpu().numpy())
    return _l2(np.concatenate(out)) if out else np.zeros((0, 512), dtype=np.float32)


@torch.no_grad()
def encode_images(images: Sequence, batch_size: int = 32) -> np.ndarray:
    model, processor = get_model()
    out = []
    for i in range(0, len(images), batch_size):
        inputs = processor(images=list(images[i:i + batch_size]), return_tensors="pt")
        out.append(model.get_image_features(pixel_values=inputs["pixel_values"]).cpu().numpy())
    return _l2(np.concatenate(out)) if out else np.zeros((0, 512), dtype=np.float32)


def class_text_matrix(labels: Sequence[str], templates: Sequence[str]) -> np.ndarray:
    """(L, 512) classifier weights: template-averaged, re-normalised text embeddings."""
    key = (tuple(labels), tuple(templates))
    if key not in _text_cache:
        per_template = [encode_texts([t.format(l) for l in labels]) for t in templates]
        _text_cache[key] = _l2(np.mean(per_template, axis=0))
    return _text_cache[key]


def zero_shot_probs(image_feats: np.ndarray, text_matrix: np.ndarray) -> np.ndarray:
    logits = logit_scale() * (np.atleast_2d(image_feats) @ text_matrix.T)
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)
