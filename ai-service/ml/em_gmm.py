"""
Gaussian Mixture Model fitted with the EM algorithm, from scratch (AML lab).

Generative story for an embedding x ∈ R^d:
    z ~ Categorical(π)            (hidden "theme" of the photo)
    x | z=k ~ N(μ_k, diag(σ²_k))

EM alternates
    E-step:  r_nk = π_k N(x_n | μ_k, σ²_k) / Σ_j π_j N(x_n | μ_j, σ²_j)
    M-step:  N_k = Σ_n r_nk;  π_k = N_k/N;  μ_k = Σ r_nk x_n / N_k;
             σ²_k = Σ r_nk (x_n − μ_k)² / N_k
and the data log-likelihood is guaranteed not to decrease after any iteration.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def pca_project(X: np.ndarray, k: int = 2):
    """Return (Z, explained_variance_ratio, components, mean) via SVD."""
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(k, Vt.shape[0])
    var = (S ** 2) / max(len(X) - 1, 1)
    ratio = var / var.sum() if var.sum() > 0 else var
    return Xc @ Vt[:k].T, ratio[:k], Vt[:k], mean


def _kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    centers = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        probs = d2 / d2.sum() if d2.sum() > 0 else np.full(len(X), 1 / len(X))
        centers.append(X[rng.choice(len(X), p=probs)])
    return np.array(centers)


class DiagGMM:
    def __init__(self, n_components: int, max_iter: int = 200, tol: float = 1e-5,
                 reg: float = 1e-6, seed: int = 0):
        self.k, self.max_iter, self.tol, self.reg, self.seed = n_components, max_iter, tol, reg, seed
        self.weights_ = self.means_ = self.vars_ = None
        self.log_likelihoods_: list[float] = []   # average log-likelihood after each iteration
        self.n_iter_ = 0

    def _log_prob_components(self, X: np.ndarray) -> np.ndarray:
        # log N(x | μ_k, σ²_k) for diagonal covariance  — shape (n, K)
        d = X.shape[1]
        out = np.empty((len(X), self.k))
        for j in range(self.k):
            diff2 = (X - self.means_[j]) ** 2 / self.vars_[j]
            out[:, j] = -0.5 * (d * np.log(2 * np.pi) + np.log(self.vars_[j]).sum() + diff2.sum(1))
        return out

    def _e_step(self, X: np.ndarray):
        lp = self._log_prob_components(X) + np.log(self.weights_)[None, :]
        ll = logsumexp(lp, axis=1)
        return np.exp(lp - ll[:, None]), float(ll.mean())

    def fit(self, X: np.ndarray) -> "DiagGMM":
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self.means_ = _kmeans_pp_init(X, self.k, rng)
        self.vars_ = np.tile(X.var(axis=0) + self.reg, (self.k, 1))
        self.weights_ = np.full(self.k, 1.0 / self.k)
        self.log_likelihoods_ = []
        prev = -np.inf
        for it in range(self.max_iter):
            resp, ll = self._e_step(X)
            Nk = resp.sum(axis=0) + 1e-10
            self.weights_ = Nk / n
            self.means_ = (resp.T @ X) / Nk[:, None]
            self.vars_ = (resp.T @ (X ** 2)) / Nk[:, None] - self.means_ ** 2 + self.reg
            self.vars_ = np.maximum(self.vars_, self.reg)
            self.log_likelihoods_.append(ll)
            self.n_iter_ = it + 1
            if abs(ll - prev) < self.tol:
                break
            prev = ll
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._e_step(np.asarray(X, dtype=float))[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def score(self, X: np.ndarray) -> float:
        return self._e_step(np.asarray(X, dtype=float))[1]

    def n_parameters(self, d: int) -> int:
        return self.k * d * 2 + (self.k - 1)

    def bic(self, X: np.ndarray) -> float:
        n, d = np.asarray(X).shape
        return float(-2 * self.score(X) * n + self.n_parameters(d) * np.log(n))
