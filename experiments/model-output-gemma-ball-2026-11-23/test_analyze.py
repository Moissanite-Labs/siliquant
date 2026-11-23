"""Synthetic controls for impact arithmetic and strict runner-shaped parsing."""

from fractions import Fraction as Q
import importlib.util
import json
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location("gemma_impact", Path(__file__).with_name("analyze.py"))
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


def synthetic_row(**changes):
    row = dict(model_id="gemma-3-270m-it", model_sha256="a" * 64, variant=a.MATRIX,
               ngl=999, bridge=False, text="wikitext", text_sha256="b" * 64,
               imatrix_sha256="c" * 64, timestamp_utc="2026-01-01T00:00:00+00:00",
               matrices=126, weights=123456, matrices_active=126, weights_active=123456,
               bits_per_weight=4.52, bits_per_weight_active=4.52, weight_err_db_mean=-10,
               weight_err_db_pooled=-11, weight_err_db_mean_active=-10,
               weight_err_db_pooled_active=-11, weight_err_db_by_kind={"ffn_down": -9},
               share_out_of_range=0, kept_columns=0, build_seconds=2, score_seconds=3,
               kl_mean=0.123, kl_mean_se=0.01, same_top=90.5, ppl=8)
    row.update(changes)
    return (json.dumps(row) + "\n").encode()


class ImpactTests(unittest.TestCase):
    def test_runner_numbers_and_isolation(self):
        primary = synthetic_row()
        bridge = synthetic_row(model_id=a.PRIMARY, text="bridge", bridge=True, variant=a.BRIDGE[0],
                               model_sha256="f" * 64)
        rows = a.load_rows([primary, bridge])
        self.assertEqual(a.select(rows, "gemma-3-270m-it", "wikitext")[a.MATRIX]["kl_mean"], Q(123, 1000))
        self.assertEqual(a.select(rows, a.PRIMARY, "wikitext"), {})
        for bad in (primary + primary, primary.replace(b'0.123', b'NaN'),
                    primary.replace(b'0.123', b'-Infinity'), primary.replace(b'0.123', b'true'),
                    primary.replace(b'"ngl": 999', b'"ngl": 999, "ngl": 999'),
                    primary + synthetic_row(variant=a.PLAIN, model_sha256="d" * 64)):
            with self.assertRaises(ValueError):
                a.load_rows([bad])

    def test_match_interpolation_bounds_and_nonmonotonic(self):
        points = [(Q(4), Q(1, 4)), (Q(9, 2), Q(1, 5)), (Q(5), Q(1, 10))]
        self.assertEqual(a.matched_rate(points, Q(3, 20)), ("interpolated", Q(19, 4)))
        self.assertEqual(a.matched_rate(points, Q(1, 2)), ("bound", Q(4)))
        self.assertEqual(a.matched_rate(points, Q(1, 20)), ("not reached", None))
        with self.assertRaises(ValueError):
            a.matched_rate([(4, 1), (5, 2)], 1)

    def test_exact_deltas_and_se_boundary(self):
        base = dict(kl_mean=Q("0.2"), kl_mean_se=Q("0.04"), same_top=80, ppl=10)
        arm = dict(kl_mean=Q("0.1"), kl_mean_se=Q("0.03"), same_top=85, ppl=8)
        c = a.comparison(arm, base, Q(10))
        self.assertEqual((c["gain"], c["ratio"], c["keep"], c["ppl"]),
                         (Q(1, 10), Q(1, 2), Q(5), (-20, 0)))
        self.assertTrue(c["screen"])
        arm["kl_mean"] += Q("0.00000000000000001")
        self.assertFalse(a.comparison(arm, base, None)["screen"])

    def test_scenario_and_transcription(self):
        spec = ("Synthetic", 8, 100, 100, 1, 50, 2)
        label, before, after, memory, _start, _end, fleet, dollars = a.scenario(Q(6), Q(1), spec)
        self.assertEqual((label, before, after, memory, fleet, dollars),
                         ("Synthetic", 26 * 10**9, 25 * 10**9, Q(1, 26), Q(5, 72), Q(3650, 3)))
        row = ("\x60\x60\x60text\n" + a.PLAIN + " | wikitext | 4.5 | 0.2 | 0.01 | 5.5 | 90\n"
               "\x60\x60\x60\n").encode()
        cells = a.load_transcribed_table(row, "gemma-3-270m-it")
        self.assertNotIn("model_sha256", next(iter(cells.values())))
        with self.assertRaises(ValueError):
            a.load_transcribed_table(row.replace(b"90", b"190"), "gemma-3-270m-it")


if __name__ == "__main__":
    unittest.main()
