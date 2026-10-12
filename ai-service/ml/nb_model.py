"""
Bernoulli Naive Bayes classifier, written from scratch (AML lab).

Bayesian-network view
---------------------
    Category C  ──►  X_1, X_2, ..., X_V        X_j ∈ {0,1}  "tag j present"

Naive assumption: the tags are conditionally independent given the category,
    P(C, X) = P(C) · Π_j P(X_j | C)
so the posterior is
    P(C | x) ∝ P(C) · Π_j θ_cj^x_j · (1 − θ_cj)^(1 − x_j)

Unlike the hand-written table in services/classifier.py, the parameters θ and
the prior here are *learned* from labelled data by maximum likelihood with
Laplace (add-alpha) smoothing:
    θ_cj = (count(tag j in class c) + α) / (n_c + 2α)
    P(c) = n_c / n                       (empirical class frequency)
"""
from __future__ import annotations

import json
import math
from typing import Iterable, Sequence

import numpy as np


class BernoulliNaiveBayes:
    def __init__(self, vocab: Sequence[str], classes: Sequence[str], alpha: float = 1.0):
        self.vocab = list(vocab)
        self.classes = list(classes)
        self.alpha = float(alpha)
        self._index = {t: i for i, t in enumerate(self.vocab)}
        self.log_prior_: np.ndarray | None = None  # (K,)
        self.theta_: np.ndarray | None = None      # (K, V)  P(tag present | class)
        self.class_counts_: np.ndarray | None = None

    # ------------------------------------------------------------------ data
    def vectorize(self, tag_sets: Iterable[Iterable[str]]) -> np.ndarray:
        rows = list(tag_sets)
        X = np.zeros((len(rows), len(self.vocab)), dtype=np.uint8)
        for r, tags in enumerate(rows):
            for t in tags:
                j = self._index.get(str(t).lower().strip())
                if j is not None:
                    X[r, j] = 1
        return X

    # ----------------------------------------------------------------- train
    def fit(self, tag_sets: Sequence[Sequence[str]], y: Sequence[str]) -> "BernoulliNaiveBayes":
        X = self.vectorize(tag_sets)
        y = np.asarray(list(y))
        K, V = len(self.classes), len(self.vocab)
        counts = np.zeros(K)
        tag_counts = np.zeros((K, V))
        for k, c in enumerate(self.classes):
            mask = y == c
            counts[k] = mask.sum()
            if counts[k]:
                tag_counts[k] = X[mask].sum(axis=0)
        self.class_counts_ = counts
        self.theta_ = (tag_counts + self.alpha) / (counts[:, None] + 2 * self.alpha)
        prior = counts / counts.sum() if counts.sum() else np.full(K, 1.0 / K)
        self.log_prior_ = np.log(np.clip(prior, 1e-12, None))
        return self

    # ------------------------------------------------------------- inference
    def joint_log_proba(self, X: np.ndarray) -> np.ndarray:
        """log P(c) + Σ_j log P(x_j | c)  — shape (n, K)."""
        X = X.astype(float)
        log_t, log_nt = np.log(self.theta_), np.log1p(-self.theta_)
        return self.log_prior_[None, :] + X @ log_t.T + (1 - X) @ log_nt.T

    def predict_proba(self, tag_sets) -> np.ndarray:
        jll = self.joint_log_proba(self.vectorize(tag_sets))
        jll -= jll.max(axis=1, keepdims=True)          # log-sum-exp trick
        p = np.exp(jll)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, tag_sets) -> list[str]:
        return [self.classes[i] for i in self.predict_proba(tag_sets).argmax(axis=1)]

    # ---------------------------------------------------------- explainability
    def explain(self, tags: Sequence[str]) -> dict:
        """Posterior plus the weight of evidence each tag contributes.

        For two classes the contribution of a tag is the log-likelihood ratio
            log P(x_j | c1) − log P(x_j | c0)
        (positive pushes towards classes[1]). Absent tags also contribute in a
        Bernoulli model; only the strongest few are returned.
        """
        x = self.vectorize([tags])[0]
        post = self.predict_proba([tags])[0]
        out = {
            "classes": self.classes,
            "prior": {c: float(math.exp(lp)) for c, lp in zip(self.classes, self.log_prior_)},
            "posterior": {c: float(p) for c, p in zip(self.classes, post)},
            "predicted": self.classes[int(post.argmax())],
            "confidence": float(post.max()),
            "evidence": [],
        }
        if len(self.classes) == 2:
            t0, t1 = self.theta_[0], self.theta_[1]
            llr_present = np.log(t1) - np.log(t0)
            llr_absent = np.log1p(-t1) - np.log1p(-t0)
            for j, tag in enumerate(self.vocab):
                present = bool(x[j])
                llr = float(llr_present[j] if present else llr_absent[j])
                out["evidence"].append({
                    "tag": tag, "present": present, "llr": llr,
                    "p_given_" + self.classes[0]: float(t0[j]),
                    "p_given_" + self.classes[1]: float(t1[j]),
                })
            out["evidence"].sort(key=lambda e: -abs(e["llr"]))
            out["evidence"] = [e for e in out["evidence"] if e["present"]][:8] or out["evidence"][:8]
        return out

    # ----------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {"vocab": self.vocab, "classes": self.classes, "alpha": self.alpha,
                "log_prior": self.log_prior_.tolist(), "theta": self.theta_.tolist(),
                "class_counts": self.class_counts_.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "BernoulliNaiveBayes":
        m = cls(d["vocab"], d["classes"], d.get("alpha", 1.0))
        m.log_prior_ = np.asarray(d["log_prior"])
        m.theta_ = np.asarray(d["theta"])
        m.class_counts_ = np.asarray(d.get("class_counts", [0] * len(m.classes)))
        return m

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "BernoulliNaiveBayes":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# --------------------------------------------------------------------------
def classification_report(y_true: Sequence[str], y_pred: Sequence[str],
                          proba: np.ndarray | None, classes: Sequence[str]) -> dict:
    """Accuracy, per-class P/R/F1, confusion matrix, log-loss, Brier score."""
    classes = list(classes)
    idx = {c: i for i, c in enumerate(classes)}
    K = len(classes)
    cm = np.zeros((K, K), dtype=int)               # rows = truth, cols = prediction
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    n = cm.sum()
    per_class = {}
    for c, i in idx.items():
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1, "support": int(cm[i].sum())}
    rep = {"n": int(n), "accuracy": float(np.trace(cm) / n) if n else 0.0,
           "macro_f1": float(np.mean([v["f1"] for v in per_class.values()])),
           "per_class": per_class, "confusion_matrix": cm.tolist(), "classes": classes}
    if proba is not None and n:
        yi = np.array([idx[t] for t in y_true])
        p = np.clip(proba[np.arange(len(yi)), yi], 1e-12, 1.0)
        rep["log_loss"] = float(-np.log(p).mean())
        onehot = np.eye(K)[yi]
        rep["brier"] = float(((proba - onehot) ** 2).sum(axis=1).mean())
    return rep
