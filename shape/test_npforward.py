"""Tests of the parts of shape/npforward.py that need no model file. Run from the root of a checkout: python3 -m unittest shape/test_npforward.py"""
import sys
import unicodedata
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import npforward as F  # noqa: E402


class _StrSub(str):
    pass


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

    def test_unicode_number_categories_split_individually(self):
        # Exact Pass-1 \p{N} span splitting across Nd (0-9, ٣), Nl (Ⅰ, Ⅱ, 〇), and No (½, ², ①)
        self.assertEqual(F._split_unicode_numbers(""), [])
        self.assertEqual(F._split_unicode_numbers("1"), [(True, "1")])
        self.assertEqual(F._split_unicode_numbers("12a"), [(True, "1"), (True, "2"), (False, "a")])
        self.assertEqual(F._split_unicode_numbers("a1"), [(False, "a"), (True, "1")])
        self.assertEqual(
            F._split_unicode_numbers("4 ½ km² ⅡⅢ ① 〇 ٣"),
            [
                (True, "4"),
                (False, " "),
                (True, "½"),
                (False, " km"),
                (True, "²"),
                (False, " "),
                (True, "Ⅱ"),
                (True, "Ⅲ"),
                (False, " "),
                (True, "①"),
                (False, " "),
                (True, "〇"),
                (False, " "),
                (True, "٣"),
            ],
        )

        # Reproduce the exact wiki.test.raw '4 ½' SmolLM2 BPE divergence:
        # In SmolLM2, 'Ġ Â' has higher merge priority than 'Â ½'; without splitting '½' in Pass 1,
        # ' ½' stays a single piece and merges to [ĠÂ, ½] instead of [Ġ, Â½].
        names = F._byte_names()
        singles = [names[b] for b in range(256)]
        b_space = names[ord(" ")]
        b_c2, b_bd = (names[b] for b in "½".encode("utf-8"))
        b_b2 = names["²".encode("utf-8")[1]]
        tok_frac = F.Tokenizer(
            singles + [b_space + b_c2, b_c2 + b_bd, "km", "m" + b_c2],
            [f"{b_space} {b_c2}", f"{b_c2} {b_bd}", "k m", f"m {b_c2}"],
        )
        # "4 ½" must split into ["4", " ", "½"] -> [4, Ġ, Â½], NEVER merging Ġ+Â!
        ids_half = tok_frac.encode("4 ½")
        self.assertEqual(ids_half, [ord("4"), ord(" "), 256 + 1])
        # "km²" must split into ["km", "²"] -> [km, Â, ²], NEVER merging m+Â!
        ids_km2 = tok_frac.encode("km²")
        self.assertEqual(ids_km2, [256 + 2, 0xC2, 0xB2])
        # Consecutive Nl Roman numerals and leading space stay separate pieces: [" ", "Ⅱ", "Ⅲ"]
        ids_roman = tok_frac.encode(" ⅡⅢ")
        expected_roman = [ord(" ")] + list("Ⅱ".encode("utf-8")) + list("Ⅲ".encode("utf-8"))
        self.assertEqual(ids_roman, expected_roman)
        # Unused helper var check for byte table sanity
        self.assertEqual(b_b2, names[0xB2])

    def test_cjk_numeric_ideographs_are_letters_not_unicode_numbers(self):
        # U+4E94 '五' and U+4E8C '二' have str.isnumeric()==True but General Category Lo (not \p{N}).
        # llama.cpp maps only Nd|Nl|No to \p{N}, so Lo characters must remain in non-number spans.
        self.assertEqual(unicodedata.category("五"), "Lo")
        self.assertEqual(unicodedata.category("二"), "Lo")
        self.assertEqual(F._split_unicode_numbers(" 五二"), [(False, " 五二")])

    def test_exhaustive_unicode_p_n_category_parity(self):
        # Verify exact parity with Unicode General Category N (Nd|Nl|No) over all 1,114,112 codepoints
        # for the running interpreter's unicodedata.unidata_version (1,831 on Unicode 15.1 / Py3.13;
        # 1,911 on Unicode 16.0 / Py3.14).
        all_cps = "".join(chr(cp) for cp in range(0x110000))
        spans = F._split_unicode_numbers(all_cps)
        self.assertEqual("".join(ch for _, ch in spans), all_cps)
        self.assertTrue(all(len(ch) > 0 for _, ch in spans))
        self.assertTrue(all(len(ch) == 1 for is_n, ch in spans if is_n))
        got_n = [ord(ch) for is_n, ch in spans if is_n]
        n_cats = ("Nd", "Nl", "No")
        expected_n = [cp for cp in range(0x110000) if unicodedata.category(chr(cp)) in n_cats]
        self.assertEqual(got_n, expected_n)
        self.assertGreaterEqual(len(got_n), 1800)
        got_set = set(got_n)
        for ch in ("0", "9", "٣", "५", "½", "²", "①", "Ⅰ", "Ⅻ", "〇"):
            self.assertIn(ord(ch), got_set)
        for ch in ("一", "二", "五", "零", "万"):
            self.assertNotIn(ord(ch), got_set)

    def test_edge_cases_surrogates_and_strict_str_guard(self):
        self.assertEqual(self.tok.encode(""), [])
        self.assertEqual(len(self.tok.encode("1234")), 4)
        with self.assertRaises(UnicodeEncodeError):
            self.tok.encode(chr(0xD800))
        for bad in (None, 123, b"12", ["1"], _StrSub("12")):
            with self.assertRaises(TypeError):
                F._split_unicode_numbers(bad)  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                self.tok.encode(bad)  # type: ignore[arg-type]

    def test_every_text_has_tokens(self):
        text = "Naïve — 3.5 km² 4 ½ ⅡⅢ ① 〇 ٣ 五二\n\tthe_end"
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
