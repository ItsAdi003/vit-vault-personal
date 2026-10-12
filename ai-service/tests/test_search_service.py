"""Tests for services/search.rank_media using a fake Supabase client (no network)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import search


class _FakeQuery:
    def __init__(self, rows):
        self.rows = rows

    def select(self, *_):
        return self

    def eq(self, *_):
        return self

    def execute(self):
        class R:  # noqa: D401
            pass
        r = R()
        r.data = self.rows
        return r


class _FakeSupabase:
    def __init__(self, rows):
        self.rows = rows

    def table(self, _):
        return _FakeQuery(self.rows)


def _unit(v):
    v = np.asarray(v, dtype=float)
    return (v / np.linalg.norm(v)).tolist()


class RankMediaTests(unittest.TestCase):
    def setUp(self):
        self._orig = search.supabase
        rows = [
            {"id": "dance", "ai_tags": ["dance", "stage"], "metadata": {}, "events": {"title": "Cultural Fest"},
             "image_embedding": _unit([1, 0, 0])},
            {"id": "code", "ai_tags": ["coding"], "metadata": {"custom_name": "Hack night"}, "events": {"title": "Hackathon"},
             # pgvector comes back from PostgREST as a string
             "image_embedding": str(_unit([0, 1, 0]))},
            {"id": "food", "ai_tags": ["food"], "metadata": {}, "events": {"title": "Canteen"},
             "image_embedding": _unit([0, 0, 1])},
            {"id": "no_vec", "ai_tags": ["dance"], "metadata": {}, "events": None, "image_embedding": None},
        ]
        search.supabase = _FakeSupabase(rows)

    def tearDown(self):
        search.supabase = self._orig

    def test_dense_finds_image_without_matching_words(self):
        # query vector is close to the "dance" image; no word overlap with its tags
        res = search.rank_media("people performing", _unit([0.95, 0.1, 0.05]))
        self.assertEqual(res[0]["id"], "dance")

    def test_sparse_match_is_kept_even_without_embedding(self):
        res = search.rank_media("dance", _unit([0, 0, 1]))
        ids = [r["id"] for r in res]
        self.assertIn("no_vec", ids)       # BM25 hit, no embedding stored
        self.assertIn("dance", ids)

    def test_margin_drops_clearly_irrelevant_images(self):
        res = search.rank_media("zzz", _unit([1, 0, 0]))
        self.assertEqual([r["id"] for r in res], ["dance"])

    def test_empty_table(self):
        search.supabase = _FakeSupabase([])
        self.assertEqual(search.rank_media("x", [1, 0, 0]), [])


if __name__ == "__main__":
    unittest.main()
