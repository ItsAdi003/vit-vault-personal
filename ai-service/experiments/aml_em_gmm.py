"""
AML experiment 3: unsupervised discovery with our own EM-fitted Gaussian mixture.

CLIP embeddings of CIFAR-10 images (labels are NOT used for fitting) are reduced
with PCA and clustered with ml.em_gmm.DiagGMM. We report
  * BIC across k (model selection)
  * agreement with the hidden classes: NMI and purity
  * the same numbers for k-means as a baseline
  * the EM log-likelihood trace (must be non-decreasing)

Run:  python experiments/aml_em_gmm.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _cifar import clip_embeddings  # noqa: E402
from _report import print_table, save_report  # noqa: E402

from ml.em_gmm import DiagGMM, pca_project  # noqa: E402


def purity(y, c):
    return float(sum(np.bincount(y[c == k]).max() for k in np.unique(c)) / len(y))


def main():
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score as nmi

    E_tr, y_tr, _ = clip_embeddings("train", 500)
    E_te, y_te, _ = clip_embeddings("test", 200)
    E, y = np.concatenate([E_tr, E_te]), np.concatenate([y_tr, y_te])
    Z, ratio, _, _ = pca_project(E, 10)

    rows = []
    fits = {}
    for k in (2, 4, 6, 8, 10, 12, 14):
        g = DiagGMM(k, seed=0, max_iter=300).fit(Z)
        fits[k] = g
        lab = g.predict(Z)
        rows.append([k, g.bic(Z), g.n_iter_, float(nmi(y, lab)), purity(y, lab)])
    cols = ["k", "BIC", "EM iterations", "NMI vs classes", "purity"]
    best_k = min(fits, key=lambda k: [r for r in rows if r[0] == k][0][1])
    print(f"{len(E)} embeddings, PCA(10) keeps {ratio.sum():.1%} of variance; BIC picks k={best_k}")
    print_table(cols, rows)

    # seeds + k-means baseline at k = 10 (the true number of classes)
    seeds = [DiagGMM(10, seed=s, max_iter=300).fit(Z) for s in range(5)]
    seed_nmi = [float(nmi(y, g.predict(Z))) for g in seeds]
    km = KMeans(10, n_init=10, random_state=0).fit(Z)
    km_nmi, km_pur = float(nmi(y, km.labels_)), purity(y, km.labels_)
    ll = np.array(fits[10].log_likelihoods_)
    monotone = bool(np.all(np.diff(ll) >= -1e-9))
    print(f"k=10 over 5 seeds: NMI {np.mean(seed_nmi):.3f} +/- {np.std(seed_nmi):.3f}; "
          f"k-means NMI {km_nmi:.3f}, purity {km_pur:.3f}; EM log-likelihood monotone: {monotone}")

    save_report("aml_em_gmm", {
        "title": "EM Gaussian mixture on CLIP embeddings (no labels used)",
        "description": "7 000 CIFAR-10 CLIP embeddings, PCA to 10 dimensions, diagonal-covariance GMM fitted by our own "
                       "EM. Class labels are used only afterwards to measure agreement.",
        "meta": {"points": len(E), "PCA dims": 10, "variance kept": f"{ratio.sum():.1%}", "BIC-selected k": best_k,
                 "k=10 NMI (5 seeds)": f"{np.mean(seed_nmi):.3f} +/- {np.std(seed_nmi):.3f}",
                 "k-means NMI (k=10)": f"{km_nmi:.3f}", "log-likelihood monotone": monotone},
        "table": {"columns": cols, "rows": rows},
        "curve": {"label": "Average log-likelihood per EM iteration (k = 10)", "y": ll.tolist()},
        "notes": "NMI and purity compare the unsupervised clusters with the hidden CIFAR-10 classes. BIC need not "
                 "pick k = 10: it prefers the number of Gaussian blobs that fit the 10-D projection best, which is "
                 "not the same thing as the number of semantic classes.",
    })


if __name__ == "__main__":
    main()
