"""
DL experiment: what do pretrained representations buy you?

On CIFAR-10 (stratified subset; same train/test images for every method):
  1. CLIP zero-shot with three prompt strategies          (0 trainable parameters, 0 labelled images)
  2. CLIP frozen features + linear probe / k-NN            (5 130 trainable parameters)
  3. few-shot linear probe (k labelled images per class)
  4. a small CNN trained from scratch                       (~0.3 M trainable parameters)

Run:  python experiments/dl_cifar10_benchmark.py [--train-per-class 500] [--test-per-class 200] [--epochs 40]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _cifar import CLASSES, clip_embeddings  # noqa: E402
from _report import print_table, save_report  # noqa: E402


def zero_shot_accuracy(E_test, y_test, templates):
    from ml import zeroshot
    W = zeroshot.class_text_matrix(CLASSES, templates)
    probs = zeroshot.zero_shot_probs(E_test, W)
    pred = probs.argmax(1)
    return float((pred == y_test).mean()), pred


def train_cnn(X_train, y_train, X_test, y_test, epochs, seed=0):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    mean = X_train.reshape(-1, 3).mean(0) / 255.0
    std = X_train.reshape(-1, 3).std(0) / 255.0

    def prep(X):
        t = torch.from_numpy(X).float().div(255.0).permute(0, 3, 1, 2)
        return (t - torch.tensor(mean).view(1, 3, 1, 1).float()) / torch.tensor(std).view(1, 3, 1, 1).float()

    Xtr, Xte = prep(X_train), prep(X_test)
    ytr, yte = torch.from_numpy(y_train).long(), torch.from_numpy(y_test).long()

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(),
                             nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(),
                             nn.MaxPool2d(2), nn.Dropout(0.2))

    net = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), nn.Flatten(), nn.Linear(128 * 16, 10))
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=4e-3, epochs=epochs, steps_per_epoch=(len(Xtr) + 127) // 128)
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 128):
            idx = perm[i:i + 128]
            xb = Xtr[idx]
            flip = torch.rand(len(xb)) < 0.5
            xb = torch.where(flip.view(-1, 1, 1, 1), xb.flip(3), xb)       # random horizontal flip
            shift = np.random.randint(-3, 4, 2)                             # random translation
            xb = torch.roll(xb, shifts=(int(shift[0]), int(shift[1])), dims=(2, 3))
            loss = F.cross_entropy(net(xb), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if (ep + 1) % 10 == 0:
            net.eval()
            with torch.no_grad():
                acc = (net(Xte).argmax(1) == yte).float().mean().item()
            print(f"  CNN epoch {ep + 1}/{epochs}: test acc {acc:.3f}", flush=True)
    net.eval()
    with torch.no_grad():
        acc = (net(Xte).argmax(1) == yte).float().mean().item()
    return float(acc), n_params, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-per-class", type=int, default=500)
    ap.add_argument("--test-per-class", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier

    E_tr, y_tr, X_tr = clip_embeddings("train", args.train_per_class)
    E_te, y_te, X_te = clip_embeddings("test", args.test_per_class)
    n_tr, n_te = len(y_tr), len(y_te)
    rows = []

    for name, tpl in [("bare label: \"airplane\"", ["{}"]),
                      ("template: \"a photo of a airplane.\"", ["a photo of a {}."]),
                      ("ensemble of 5 templates", ["{}", "a photo of a {}.", "a photo showing a {}.",
                                                   "a picture of a {}.", "a low resolution photo of a {}."])]:
        acc, _ = zero_shot_accuracy(E_te, y_te, tpl)
        rows.append([f"CLIP zero-shot, {name}", acc, 0, 0, "no labelled training images"])

    # Linear probe on frozen CLIP features (C picked on a held-out 20 % of the training set)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_tr)
    cut = int(0.8 * n_tr)
    best_c, best_v = None, -1
    for C in (0.1, 1, 10, 100):
        v = LogisticRegression(C=C, max_iter=2000).fit(E_tr[perm[:cut]], y_tr[perm[:cut]]).score(E_tr[perm[cut:]], y_tr[perm[cut:]])
        if v > best_v:
            best_c, best_v = C, v
    probe = LogisticRegression(C=best_c, max_iter=3000).fit(E_tr, y_tr)
    probe_acc = float(probe.score(E_te, y_te))
    rows.append([f"CLIP features + linear probe (C={best_c})", probe_acc, 512 * 10 + 10, n_tr, "frozen encoder"])
    knn = KNeighborsClassifier(5, metric="cosine").fit(E_tr, y_tr)
    rows.append(["CLIP features + 5-NN (cosine)", float(knn.score(E_te, y_te)), 0, n_tr, "no training at all"])

    for k in (1, 5, 20, 100):
        idx = np.concatenate([np.flatnonzero(y_tr == c)[:k] for c in range(10)])
        m = LogisticRegression(C=best_c, max_iter=3000).fit(E_tr[idx], y_tr[idx])
        rows.append([f"few-shot linear probe, {k} image(s) per class", float(m.score(E_te, y_te)), 5130, len(idx), ""])

    cnn_acc, n_params, secs = train_cnn(X_tr, y_tr, X_te, y_te, args.epochs)
    rows.append([f"small CNN from scratch ({args.epochs} epochs, {secs:.0f}s)", cnn_acc, n_params, n_tr, "random init"])

    cols = ["method", "test accuracy", "trainable params", "labelled train images", "note"]
    print(f"\nCIFAR-10: {n_tr} train / {n_te} test images (stratified)")
    print_table(cols, rows)

    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_te, probe.predict(E_te)):
        cm[t, p] += 1
    fewshot = [r[1] for r in rows if r[0].startswith("few-shot")]
    path = save_report("dl_cifar10", {
        "title": "CIFAR-10: pretrained CLIP features vs a CNN trained from scratch",
        "description": "Same test images for every method. Zero-shot uses only text prompts; the probe trains a single "
                       "linear layer on frozen 512-D CLIP features; the CNN learns everything from pixels.",
        "meta": {"train images": n_tr, "test images": n_te, "classes": 10, "CNN epochs": args.epochs},
        "table": {"columns": cols, "rows": rows},
        "curve": {"label": "Few-shot linear-probe accuracy vs labelled images per class (1, 5, 20, 100)", "y": fewshot},
        "confusion": {"classes": CLASSES, "matrix": cm.tolist()},
        "notes": "CIFAR-10 images are 32x32 and are up-sampled to 224x224 for CLIP, which hurts zero-shot accuracy "
                 "relative to natural-resolution photos. The test split is a 2 000-image subset, so differences of "
                 "about 2 percentage points are within sampling noise.",
    })
    print("saved", path)


if __name__ == "__main__":
    main()
