"""CIFAR-10 loader + cached CLIP embeddings, shared by the DL / AML / NLP experiments."""
import hashlib
import os
import pickle
import sys
import tarfile
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA_DIR = os.path.join(ROOT, "data")
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
MD5 = "c58f30108f718f92721af3b95e74349a"
ARCHIVE = os.path.join(DATA_DIR, "cifar-10-python.tar.gz")
FOLDER = os.path.join(DATA_DIR, "cifar-10-batches-py")
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


TOTAL_BYTES = 170498071


def _download_resumable(url: str, dest: str, retries: int = 20):
    """Download with HTTP Range resume so an interrupted run continues instead of restarting."""
    last_pct = -1
    for attempt in range(retries):
        pos = os.path.getsize(dest) if os.path.exists(dest) else 0
        if pos >= TOTAL_BYTES:
            return
        req = urllib.request.Request(url, headers={"Range": f"bytes={pos}-"} if pos else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if pos and r.status != 206:           # server ignored Range: start over
                    pos = 0
                    open(dest, "wb").close()
                with open(dest, "ab") as f:
                    while True:
                        chunk = r.read(1 << 16)
                        if not chunk:
                            break
                        f.write(chunk)
                        pos += len(chunk)
                        pct = pos * 100 // TOTAL_BYTES
                        if pct // 10 != last_pct // 10:
                            last_pct = pct
                            print(f"  {pct}%", flush=True)
            if os.path.getsize(dest) >= TOTAL_BYTES:
                return
        except (OSError, ConnectionError) as e:
            print(f"  connection problem ({e}); resuming from {os.path.getsize(dest) if os.path.exists(dest) else 0} bytes "
                  f"(attempt {attempt + 1}/{retries})", flush=True)
    raise SystemExit("Could not finish the CIFAR-10 download; run the script again to resume.")


def ensure_cifar():
    if os.path.isdir(FOLDER):
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    have = os.path.getsize(ARCHIVE) if os.path.exists(ARCHIVE) else 0
    if have < TOTAL_BYTES:
        print(f"Downloading CIFAR-10 (~163 MB) from {URL} (have {have // 2**20} MB)", flush=True)
        _download_resumable(URL, ARCHIVE)
    got = _md5(ARCHIVE)
    if got != MD5:
        os.remove(ARCHIVE)
        raise SystemExit(f"Checksum mismatch for CIFAR-10 archive ({got}); deleted it, please retry.")
    with tarfile.open(ARCHIVE) as tar:
        tar.extractall(DATA_DIR)
    print("CIFAR-10 ready.")


def _load_batch(name):
    with open(os.path.join(FOLDER, name), "rb") as f:
        d = pickle.load(f, encoding="bytes")
    X = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return X, np.array(d[b"labels"])


def load_subset(split: str, per_class: int, seed: int = 0):
    """Stratified subset: (images uint8 [n,32,32,3], labels [n])."""
    ensure_cifar()
    if split == "train":
        parts = [_load_batch(f"data_batch_{i}") for i in range(1, 6)]
        X = np.concatenate([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
    else:
        X, y = _load_batch("test_batch")
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.permutation(np.flatnonzero(y == c))[:per_class] for c in range(10)])
    idx = rng.permutation(idx)
    return X[idx], y[idx]


def clip_embeddings(split: str, per_class: int, seed: int = 0):
    """(embeddings [n,512] float32 L2-normalised, labels, images) — embeddings cached on disk."""
    X, y = load_subset(split, per_class, seed)
    cache = os.path.join(DATA_DIR, f"cifar10_clip_{split}_{per_class}_{seed}.npz")
    if os.path.exists(cache):
        return np.load(cache)["emb"], y, X
    from PIL import Image
    from ml import zeroshot
    from models.clip_model import load_model
    load_model()
    print(f"Encoding {len(X)} {split} images with CLIP (CPU)...", flush=True)
    emb = []
    for i in range(0, len(X), 64):
        emb.append(zeroshot.encode_images([Image.fromarray(a) for a in X[i:i + 64]]))
        if (i // 64) % 10 == 0:
            print(f"  {i + 64}/{len(X)}", flush=True)
    emb = np.concatenate(emb).astype(np.float32)
    np.savez_compressed(cache, emb=emb)
    return emb, y, X
