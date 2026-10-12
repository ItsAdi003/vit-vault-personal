"""
Attention rollout for CLIP's Vision Transformer (DL lab).

A ViT-B/32 cuts a 224x224 image into 7x7 = 49 patches (+1 [CLS] token). Every
layer has 12 heads of self-attention. Rollout (Abnar & Zuidema, 2020) multiplies
the head-averaged attention matrices of all layers (with the residual connection
added as an identity) to estimate how much each patch contributed to [CLS].
"""
from __future__ import annotations

import base64
import io

import numpy as np
import torch
from PIL import Image

from models.clip_model import get_model

_SIZE = 224


def _center_crop_224(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    s = _SIZE / min(w, h)
    image = image.resize((max(_SIZE, round(w * s)), max(_SIZE, round(h * s))), Image.BICUBIC)
    w, h = image.size
    l, t = (w - _SIZE) // 2, (h - _SIZE) // 2
    return image.crop((l, t, l + _SIZE, t + _SIZE))


@torch.no_grad()
def attention_rollout(image: Image.Image) -> dict:
    model, processor = get_model()
    pixel_values = processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"]
    out = model.vision_model(pixel_values=pixel_values, output_attentions=True)
    atts = out.attentions
    if not atts or atts[0] is None:
        raise RuntimeError("Vision model did not return attention weights")
    T = atts[0].shape[-1]
    eye = torch.eye(T)
    rollout = eye.clone()
    for layer in atts:
        a = layer.mean(dim=1)[0] + eye          # average heads, add residual
        a = a / a.sum(dim=-1, keepdim=True)
        rollout = a @ rollout
    side = int(round((T - 1) ** 0.5))
    grid = rollout[0, 1:].reshape(side, side).numpy()
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-12)
    return {"grid": grid, "layers": len(atts), "heads": atts[0].shape[1], "patches": side * side}


def _colormap(x: np.ndarray) -> np.ndarray:
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def overlay_png_b64(image: Image.Image, grid: np.ndarray, alpha: float = 0.55) -> str:
    base = _center_crop_224(image)
    heat = Image.fromarray((grid * 255).astype(np.uint8)).resize((_SIZE, _SIZE), Image.BICUBIC)
    h = np.asarray(heat, dtype=float) / 255.0
    rgb = (_colormap(h) * 255).astype(np.uint8)
    blended = (np.asarray(base, dtype=float) * (1 - alpha) + rgb * alpha).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
