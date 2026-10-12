"""Unit tests for the ML toolkit. Run from ai-service/:  python -m unittest discover -s tests -v"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.nb_model import BernoulliNaiveBayes, classification_report
from ml.em_gmm import DiagGMM, pca_project
from ml import retrieval as R


def _toy_tagsets(n=400, seed=0):
    rng = np.random.default_rng(seed)
    vocab = ["stage", "seminar", "coding", "hackathon", "dance", "food"]
    classes = ["INSTITUTIONAL", "COMMITTEE"]
    theta = np.array([[0.8, 0.7, 0.1, 0.05, 0.2, 0.2],
                      [0.2, 0.1, 0.7, 0.8, 0.6, 0.5]])
    y = rng.integers(0, 2, n)
    X = (rng.random((n, len(vocab))) < theta[y]).astype(int)
    tag_sets = [[vocab[j] for j in np.flatnonzero(row)] for row in X]
    return vocab, classes, theta, tag_sets, [classes[i] for i in y], X, y


class NaiveBayesTests(unittest.TestCase):
    def test_matches_sklearn_bernoulli_nb(self):
        from sklearn.naive_bayes import BernoulliNB
        vocab, classes, _, tag_sets, y, X, y_idx = _toy_tagsets()
        mine = BernoulliNaiveBayes(vocab, classes, alpha=1.0).fit(tag_sets, y)
        ref = BernoulliNB(alpha=1.0, binarize=None).fit(X, y_idx)
        np.testing.assert_allclose(mine.predict_proba(tag_sets), ref.predict_proba(X), atol=1e-9)

    def test_parameter_recovery_improves_with_data(self):
        errs = []
        for n in (50, 5000):
            vocab, classes, theta, tag_sets, y, *_ = _toy_tagsets(n=n, seed=1)
            m = BernoulliNaiveBayes(vocab, classes).fit(tag_sets, y)
            errs.append(np.abs(m.theta_ - theta).mean())
        self.assertLess(errs[1], errs[0])
        self.assertLess(errs[1], 0.03)

    def test_posterior_sums_to_one_and_unknown_tags_ignored(self):
        vocab, classes, _, tag_sets, y, *_ = _toy_tagsets()
        m = BernoulliNaiveBayes(vocab, classes).fit(tag_sets, y)
        p = m.predict_proba([["stage", "not-a-tag"], []])
        np.testing.assert_allclose(p.sum(axis=1), 1.0)

    def test_explain_direction(self):
        vocab, classes, _, tag_sets, y, *_ = _toy_tagsets()
        m = BernoulliNaiveBayes(vocab, classes).fit(tag_sets, y)
        e = m.explain(["hackathon", "coding"])
        self.assertEqual(e["predicted"], "COMMITTEE")
        self.assertGreater(e["evidence"][0]["llr"], 0)  # favours classes[1]

    def test_save_load_roundtrip(self):
        vocab, classes, _, tag_sets, y, *_ = _toy_tagsets()
        m = BernoulliNaiveBayes(vocab, classes).fit(tag_sets, y)
        m2 = BernoulliNaiveBayes.from_dict(m.to_dict())
        np.testing.assert_allclose(m.predict_proba(tag_sets), m2.predict_proba(tag_sets))

    def test_report_metrics(self):
        rep = classification_report(["a", "a", "b", "b"], ["a", "b", "b", "b"], None, ["a", "b"])
        self.assertAlmostEqual(rep["accuracy"], 0.75)
        self.assertEqual(rep["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertAlmostEqual(rep["per_class"]["b"]["recall"], 1.0)
        self.assertAlmostEqual(rep["per_class"]["b"]["precision"], 2 / 3)


class GMMTests(unittest.TestCase):
    def _blobs(self, seed=0):
        rng = np.random.default_rng(seed)
        centers = np.array([[0, 0], [6, 6], [-6, 6]], dtype=float)
        X = np.concatenate([c + rng.normal(scale=0.6, size=(150, 2)) for c in centers])
        y = np.repeat([0, 1, 2], 150)
        return X, y

    def test_loglikelihood_is_monotone(self):
        X, _ = self._blobs()
        g = DiagGMM(3, seed=1).fit(X)
        ll = np.array(g.log_likelihoods_)
        self.assertTrue(np.all(np.diff(ll) >= -1e-9), "EM must never decrease the likelihood")

    def test_recovers_clusters_and_matches_sklearn(self):
        from sklearn.metrics import normalized_mutual_info_score
        from sklearn.mixture import GaussianMixture
        X, y = self._blobs()
        g = DiagGMM(3, seed=2).fit(X)
        self.assertGreater(normalized_mutual_info_score(y, g.predict(X)), 0.95)
        ref = GaussianMixture(3, covariance_type="diag", random_state=0).fit(X)
        self.assertAlmostEqual(g.score(X), ref.score(X), delta=0.05)

    def test_bic_prefers_true_k(self):
        X, _ = self._blobs()
        bics = {k: DiagGMM(k, seed=0).fit(X).bic(X) for k in (1, 2, 3, 5)}
        self.assertEqual(min(bics, key=bics.get), 3)

    def test_pca_shapes(self):
        Z, ratio, comps, mean = pca_project(np.random.default_rng(0).normal(size=(30, 10)), 2)
        self.assertEqual(Z.shape, (30, 2))
        self.assertTrue(0 < ratio.sum() <= 1)


class RetrievalTests(unittest.TestCase):
    def test_stemming_matches_inflections(self):
        self.assertEqual(R.tokenize("students dancing"), R.tokenize("student dance"))

    def test_bm25_ranks_matching_doc_first(self):
        docs = [R.tokenize(t) for t in ["hackathon coding laptops", "stage dance cultural event", "food canteen"]]
        idx = R.BM25Index(docs)
        s = idx.scores(R.tokenize("people dancing on stage"))
        self.assertEqual(int(np.argmax(s)), 1)

    def test_bm25_zero_when_no_overlap(self):
        idx = R.BM25Index([R.tokenize("hackathon coding")])
        self.assertEqual(idx.scores(R.tokenize("sunset beach"))[0], 0.0)

    def test_tfidf_cosine_in_unit_range(self):
        docs = [R.tokenize("stage dance"), R.tokenize("coding hackathon")]
        s = R.tfidf_cosine_scores(R.tokenize("dance"), docs)
        self.assertGreater(s[0], s[1])
        self.assertTrue(np.all((s >= 0) & (s <= 1 + 1e-9)))

    def test_rrf_prefers_consensus(self):
        fused = R.rrf([[2, 0, 1], [2, 1, 0]])
        self.assertEqual(max(fused, key=fused.get), 2)

    def test_metrics(self):
        ranked, rel = [3, 1, 4, 0], {1, 0}
        self.assertAlmostEqual(R.precision_at_k(ranked, rel, 2), 0.5)
        self.assertAlmostEqual(R.recall_at_k(ranked, rel, 2), 0.5)
        self.assertAlmostEqual(R.reciprocal_rank(ranked, rel), 0.5)
        self.assertAlmostEqual(R.average_precision(ranked, rel), (1 / 2 + 2 / 4) / 2)
        self.assertAlmostEqual(R.ndcg_at_k([1, 0], {1, 0}, 2), 1.0)

    def test_cosine_scores(self):
        s = R.cosine_scores(np.array([1.0, 0.0]), np.array([[2.0, 0.0], [0.0, 3.0]]))
        np.testing.assert_allclose(s, [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
