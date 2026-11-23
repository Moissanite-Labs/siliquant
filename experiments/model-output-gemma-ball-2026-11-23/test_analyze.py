"""Synthetic controls for impact arithmetic and strict runner-shaped parsing."""

from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction as Q
import importlib.util
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
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


def fenced(body):
    return ("\x60\x60\x60text\n" + body + "\x60\x60\x60\n").encode()


TRANSCRIBED_ROW = a.PLAIN + " | wikitext | 4.5 | 0.2 | 0.01 | 5.5 | 90\n"


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

    def test_exact_decoder_refusals(self):
        for token in ("0." + "1" * 257, "1e513", "1e-513", "1e000000"):
            with self.subTest(token=token), self.assertRaisesRegex(ValueError, "decimal outside runner range"):
                a.read_json(token)
        for token in ("NaN", "Infinity"):
            with self.subTest(token=token), self.assertRaisesRegex(ValueError, "illegal JSON constant"):
                a.read_json(token)
        self.assertIs(a.read_json("-Infinity"), a.NEG_INF)
        valid = synthetic_row().replace(b'"weight_err_db_mean": -10', b'"weight_err_db_mean": -Infinity')
        self.assertIs(next(iter(a.load_rows([valid]).values()))["weight_err_db_mean"], a.NEG_INF)
        raw = synthetic_row(extra={"nested": ["replace"]}).replace(b'"replace"', b"-Infinity")
        with self.assertRaisesRegex(ValueError, "illegal infinity in extra"):
            a.load_rows([raw])

    def test_runner_schema_refuses_malformed_objects(self):
        baseline = dict(variant=a.Q4, file_sha256="d" * 64, other_tensors_that_differ=[])
        cases = [
            (b"[]\n", "row must be a JSON object"),
            (synthetic_row(model_id="otro"), "unknown or missing model_id"),
            (synthetic_row(bridge=0), "text/bridge mismatch"),
            (synthetic_row(variant="unknown-arm"), "variant outside preregistered"),
            (synthetic_row(variant=a.BRIDGE[0], text="bridge", bridge=True), "bridge requires primary model"),
            (synthetic_row(model_sha256="z" * 64), "invalid model_sha256"),
            (synthetic_row(text_sha256=None), "invalid text_sha256"),
            (synthetic_row(imatrix_sha256=""), "missing or invalid imatrix_sha256"),
            (synthetic_row(variant=a.Q4, other_tensors_that_differ=[]), "missing or invalid file_sha256"),
            (synthetic_row(variant=a.PLAIN, imatrix_sha256="bad"), "invalid imatrix_sha256"),
            (synthetic_row(ngl=True), "invalid ngl"),
            (synthetic_row(timestamp_utc="2026-01-01T00:00:00Z"), "invalid timestamp_utc"),
            (synthetic_row(matrices=True), "invalid or missing matrices"),
            (synthetic_row(matrices_active=127), "active counts exceed total"),
            (synthetic_row().replace(b'"build_seconds": 2, ', b""), "missing build_seconds"),
            (synthetic_row(kl_mean_se=-1), "negative kl_mean_se"),
            (synthetic_row(weight_err_db_mean="not numeric"), "invalid weight_err_db_mean"),
            (synthetic_row(weight_err_db_by_kind={"bad": False}), "invalid weight_err_db_by_kind"),
            (synthetic_row(**(baseline | {"other_tensors_that_differ": [1]})), "invalid other_tensors_that_differ"),
            (synthetic_row().replace(b'"kl_mean_se": 0.01, ', b""), "KL and KL SE must appear together"),
            (synthetic_row().replace(b'"kl_mean": 0.123, "kl_mean_se": 0.01, ', b""), "score metadata without KL"),
            (synthetic_row(same_top_se=1).replace(b'"same_top": 90.5, ', b""), "top-1 SE without top-1"),
            (synthetic_row(same_top=101), "top-1 outside"),
        ]
        for raw, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic), self.assertRaisesRegex(ValueError, diagnostic):
                a.load_rows([raw])
        self.assertEqual(len(a.load_rows([synthetic_row(**baseline)])), 1)

    def test_transcription_structure_refusals(self):
        valid = fenced(TRANSCRIBED_ROW)
        cases = [
            (b"no table", "expected one fenced text table"),
            (valid + valid, "expected one fenced text table"),
            (fenced("missing columns\n"), "expected variant, text"),
            (valid.replace(b"wikitext", b"bridge"), "invalid transcribed headline cell"),
            (fenced(TRANSCRIBED_ROW * 2), "duplicate transcribed headline cell"),
            (valid.replace(b"0.01", b"1/100"), "invalid transcribed numeric field"),
            (fenced(""), "empty transcription"),
        ]
        for raw, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic), self.assertRaisesRegex(ValueError, diagnostic):
                a.load_transcribed_table(raw, "gemma-3-270m-it")

    def test_match_interpolation_bounds_and_nonmonotonic(self):
        points = [(Q(4), Q(1, 4)), (Q(9, 2), Q(1, 5)), (Q(5), Q(1, 10))]
        self.assertEqual(a.matched_rate(points, Q(3, 20)), ("interpolated", Q(19, 4)))
        self.assertEqual(a.matched_rate(points, Q(1, 2)), ("bound", Q(4)))
        self.assertEqual(a.matched_rate(points, Q(1, 20)), ("not reached", None))
        self.assertEqual(a.matched_rate(points[1:], Q(3, 20)), ("interpolated", Q(19, 4)))
        with self.assertRaisesRegex(ValueError, "at least two rates required"):
            a.matched_rate(points[:1], Q(1, 4))
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
        row = fenced(TRANSCRIBED_ROW)
        cells = a.load_transcribed_table(row, "gemma-3-270m-it")
        self.assertNotIn("model_sha256", next(iter(cells.values())))
        with self.assertRaises(ValueError):
            a.load_transcribed_table(row.replace(b"90", b"190"), "gemma-3-270m-it")

    def test_cli_refuses_before_emitting_report(self):
        with TemporaryDirectory() as tmp:
            raw, table = Path(tmp) / "synthetic.jsonl", Path(tmp) / "synthetic.md"
            raw.write_bytes(synthetic_row())
            table.write_bytes(fenced(TRANSCRIBED_ROW))
            cases = [
                ([str(table), str(table), "--transcribed-table", "--model", "gemma-3-270m-it"],
                 "provide a single transcription"),
                ([str(raw), "--model", "gemma-3-270m-it", "--f16-python", "0"], "F16 PPL must be positive"),
                ([str(raw), "--model", "gemma-3-1b-it"], "no headline cells for model"),
            ]
            for argv, diagnostic in cases:
                out, err = StringIO(), StringIO()
                with self.subTest(diagnostic=diagnostic), redirect_stdout(out), redirect_stderr(err):
                    with self.assertRaises(SystemExit) as exit_info:
                        a.main(argv)
                    self.assertEqual(exit_info.exception.code, 2)
                    self.assertIn(diagnostic, err.getvalue())
                    self.assertEqual(out.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
