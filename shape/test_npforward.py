"""Tests of the parts of shape/npforward.py that need no model file. Run from the root of a checkout: python3 -m unittest shape/test_npforward.py"""
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import npforward as F  # noqa: E402


class TestTokenizer(unittest.TestCase):
    def setUp(self):
        names = F._byte_names()
        singles = [names[b] for b in range(256)]
        self.tok = F.Tokenizer(singles + ["th", "the", "Ġthe", "Ġt"], ["t h", "th e", "Ġ the", "Ġ t"])

    def test_bytes_have_256_names(self):
        self.assertEqual(len(set(F._byte_names().values())), 256)
        self.assertEqual(F._byte_names()[ord(" ")], "Ġ")

    def test_merges_by_rank(self):
        ids = self.tok.encode("the the")
        self.assertEqual([i - 256 for i in ids], [1, 2])               # "the", then " the": the merge "t h" comes before "Ġ t"

    def test_a_digit_is_its_own_piece(self):
        ids = self.tok.encode("the 12")
        self.assertEqual(len(ids), 4)                                  # the, space, 1, 2

    def test_every_text_has_tokens(self):
        text = "Naïve — 3.5 km²\n\tthe_end"
        ids = self.tok.encode(text)
        names = {v: k for k, v in self.tok.ids.items()}
        back = {v: k for k, v in self.tok.names.items()}
        self.assertEqual(bytes(back[c] for i in ids for c in names[i]).decode("utf-8"), text)


class TestScoring(unittest.TestCase):
    def test_passages_and_scored(self):
        P = F.passages(list(range(100)), 8, 3)
        self.assertEqual(P.shape, (3, 8))
        self.assertEqual(list(F.scored(8)), [4, 5, 6])
        self.assertEqual(len(F.scored(512)), 255)
        with self.assertRaises(ValueError):
            F.passages(list(range(10)), 8, 3)

    def test_kl_is_zero_for_the_same_and_positive_otherwise(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(5, 9))
        a -= np.log(np.exp(a).sum(axis=1, keepdims=True))
        b = rng.normal(size=(5, 9))
        b -= np.log(np.exp(b).sum(axis=1, keepdims=True))
        self.assertTrue(np.allclose(F.kl_rows(a, a), 0.0))
        self.assertTrue(np.all(F.kl_rows(a, b) > 0))
        p = np.log(np.array([[0.5, 0.5]])), np.log(np.array([[0.25, 0.75]]))
        self.assertAlmostEqual(float(F.kl_rows(*p)[0]), 0.5 * np.log(2.0) + 0.5 * np.log(2.0 / 3.0), places=12)


class TestRotary(unittest.TestCase):
    def test_keeps_lengths_and_depends_on_the_gap_only(self):
        m = F.Model.__new__(F.Model)
        m.base = 130000.0
        rng = np.random.default_rng(1)
        q = rng.normal(size=(1, 6, 2, 8)).astype(np.float32)
        r = m._rotate(q, 6)
        self.assertTrue(np.allclose(r[0, 0], q[0, 0]))                 # position 0 is not turned
        self.assertTrue(np.allclose((r ** 2).sum(axis=-1), (q ** 2).sum(axis=-1), rtol=1e-5))
        same = np.repeat(q[:, :1], 6, axis=1)                          # the same vector at every position
        t = m._rotate(same, 6)
        self.assertAlmostEqual(float((t[0, 1, 0] * t[0, 3, 0]).sum()), float((t[0, 2, 0] * t[0, 4, 0]).sum()), places=4)


if __name__ == "__main__":
    unittest.main()
