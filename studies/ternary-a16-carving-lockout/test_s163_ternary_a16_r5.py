#!/usr/bin/env python3
"""R5 test suite for Track S163 EXP-Q42a ternary A_16 vs Model Carving lockout verifier."""
from __future__ import annotations

import ast
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import unittest

BASE = Path(os.path.abspath(str(Path(__file__).parent)))
sys.path.insert(0, str(BASE))
import verify_s163_ternary_a16_carving as vmod

ERR = vmod.VerifyError
VERIFIER = "verify_s163_ternary_a16_carving.py"
CERT = "s163_ternary_a16_carving_certificate.json"
_PAYLOAD = {}


def payload():
    if "p" not in _PAYLOAD:
        _PAYLOAD["p"] = vmod.build_payload(base=BASE)
    return _PAYLOAD["p"]


def good_runner(_lean):
    lines = ["'TernaryA16Carving.t" + str(i) + "' depends on axioms: [propext, Quot.sound]" for i in range(vmod.EXPECTED_THEOREMS - 2)]
    lines.append("'TernaryA16Carving.a' does not depend on any axioms")
    lines.append("'TernaryA16Carving.b' depends on axioms: [propext]")
    return {"version_rc": 0, "version_output": "Lean (version 4.36.0)", "build_rc": 0, "build_output": "ok", "axiom_rc": 0, "axiom_stdout": "\n".join(lines) + "\n"}


def runner_with(**kw):
    def run(lean):
        out = good_runner(lean)
        out.update(kw)
        return out
    return run


def copy_lean(tmp):
    dest = Path(tmp) / "lean"
    shutil.copytree(BASE / "lean", dest, ignore=shutil.ignore_patterns(".lake"))
    return Path(tmp)


class Patch:
    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.orig = None

    def __enter__(self):
        self.orig = getattr(vmod, self.name)
        setattr(vmod, self.name, self.value)
        return self

    def __exit__(self, *exc):
        setattr(vmod, self.name, self.orig)
        return False


def codeword_block(k):
    return vmod.make_block(vmod.zero_syndrome_codewords()[k], k)


class OsProxy:
    """Stand-in for the verifier's `os` module that overrides selected calls only."""

    def __init__(self, **over):
        self.over = over

    def __getattr__(self, name):
        if name in self.over:
            return self.over[name]
        return getattr(os, name)


def run_main_capture(argv):
    with tempfile.TemporaryFile() as cap:
        saved = os.dup(2)
        try:
            os.dup2(cap.fileno(), 2)
            rc = vmod.main(argv)
        finally:
            os.dup2(saved, 2)
            os.close(saved)
        cap.seek(0)
        return rc, cap.read().decode("utf-8")


def cert_bytes():
    return vmod.canonical_cert_bytes(payload())


class TestPillar1(unittest.TestCase):
    def test_happy(self):
        res = vmod.verify_pillar1()
        self.assertEqual(res["vectors_checked"], 65536)
        self.assertEqual(res["codewords"], 2048)
        self.assertEqual(res["codewords_closed_under_flip"], 2048)
        self.assertEqual(res["syndrome_0x0000"], 0)
        self.assertEqual(res["syndrome_0xFFFF"], 0)
        self.assertEqual(res["syndrome_0x0001"], 1)
        self.assertEqual(res["bipolar_norm_sq"], 16)
        self.assertEqual(res["mask_weights"], [16, 8, 8, 8, 8])
        self.assertEqual(vmod.syndrome16(2), 3)
        self.assertEqual(vmod.complement16(0x1234), 0xEDCB)
        self.assertEqual(vmod.bipolar16(0x8001), [-1] + [1] * 14 + [-1])

    def test_guards(self):
        with self.assertRaisesRegex(ERR, "^P1_WEIGHTS:"):
            vmod.verify_pillar1(masks=(0xFFFF, 0xAAAB, 0xCCCC, 0xF0F0, 0xFF00))
        with self.assertRaisesRegex(ERR, "^P1_SYN_INVARIANT:"):
            vmod.verify_pillar1(flip=0x7FFF)
        with self.assertRaisesRegex(ERR, "^P1_BIPOLAR_NEG:"):
            vmod.verify_pillar1(flip=0x0FF0)
        with Patch("complement16", lambda m, flip=0xFFFF: 0xFFFF if m == 0 else 0x1234):
            with self.assertRaisesRegex(ERR, "^P1_INVOLUTION:"):
                vmod.verify_pillar1()
        with Patch("bipolar16", lambda m: [0] * 16):
            with self.assertRaisesRegex(ERR, "^P1_BIPOLAR_NORM:"):
                vmod.verify_pillar1()
        with self.assertRaisesRegex(ERR, "^P1_CODEWORDS:"):
            vmod.verify_pillar1(masks=(0xFFFF, 0xAAAA, 0xCCCC, 0xF0F0, 0xAAAA))


class TestPillar2(unittest.TestCase):
    def test_happy(self):
        res = payload()["pillar2_blockh4_scale_flip"]
        self.assertEqual(res["blocks"], 2048)
        self.assertEqual(res["sum_h4_sq"], 4 * res["sum_c_sq"])
        self.assertEqual(res["flip_norm_drift"], "0/1")
        self.assertEqual(res["flip_payload_bytes_changed"], 0)
        self.assertEqual(res["round_norm_drift"], "0/1")
        self.assertEqual(res["even_identity_returns"], 4)
        self.assertEqual(res["odd_negations"], 4)
        self.assertEqual(res["h4_gram_diag"], [4, 4, 4, 4])
        self.assertEqual(Fraction(res["sum_w_sq"]), Fraction(9, 64) * res["sum_c_sq"])
        h, w, w_neg = vmod.verify_block([0] * 16, Fraction(1))
        self.assertEqual((h, w, w_neg), ([0] * 16, [0] * 16, [0] * 16))
        edge = [7, 7, 7, 7, -8] + [0] * 11
        self.assertEqual(vmod.syndrome16(vmod.lsb_mask(edge)), 0)
        self.assertEqual(len(vmod.verify_block(edge, Fraction(-5, 3))[0]), 16)
        self.assertEqual(vmod.apply_round(Fraction(3), -1), Fraction(-3))

    def test_block_guards(self):
        good = codeword_block(9)
        with self.assertRaisesRegex(ERR, "^P2_BLOCK_LEN:"):
            vmod.verify_block(good[:15], Fraction(1))
        hi = list(good)
        hi[0] = 8
        with self.assertRaisesRegex(ERR, "^P2_NIBBLE_RANGE:"):
            vmod.verify_block(hi, Fraction(1))
        lo = list(good)
        lo[0] = -10
        with self.assertRaisesRegex(ERR, "^P2_NIBBLE_RANGE:"):
            vmod.verify_block(lo, Fraction(1))
        bad = list(good)
        bad[0] = bad[0] ^ 1
        with self.assertRaisesRegex(ERR, "^P2_BLOCK_SYN:"):
            vmod.verify_block(bad, Fraction(1))
        with self.assertRaisesRegex(ERR, "^P2_PARSEVAL:"):
            vmod.verify_block(good, Fraction(1), rows=((1, 1, 1, 1),) * 4)
        with Patch("block_weights", lambda h, s: [s * x for x in h]):
            with self.assertRaisesRegex(ERR, "^P2_WEIGHT_NORM:"):
                vmod.verify_block(good, Fraction(1))
        with Patch("block_weights", lambda h, s: [Fraction(1, 2) * abs(s) * x for x in h]):
            with self.assertRaisesRegex(ERR, "^P2_FLIP_NEG:"):
                vmod.verify_block(good, Fraction(1))

    def test_pillar_guards(self):
        with self.assertRaisesRegex(ERR, "^P2_SCALE_ZERO:"):
            vmod.verify_pillar2(scale=Fraction(0))
        with self.assertRaisesRegex(ERR, "^P2_H4_GRAM:"):
            vmod.verify_pillar2(rows=((1, 1, 1, 1),) * 4)
        with self.assertRaisesRegex(ERR, "^P2_SIGMA:"):
            vmod.verify_pillar2(sequences=((1, 2),))
        with self.assertRaisesRegex(ERR, "^P2_SIGMA:"):
            vmod.verify_pillar2(sequences=((0,),))
        with Patch("apply_round", lambda s, sigma: s):
            with self.assertRaisesRegex(ERR, "^P2_ROUND_NET:"):
                vmod.verify_pillar2(sequences=((-1,),))

    def test_payload_snapshot_detects_mutation(self):
        real = vmod.verify_block
        touched = []

        def mutating(c, s, rows=vmod.H4_ROWS):
            out = real(c, s, rows)
            c[0] = c[0] ^ 2
            touched.append(1)
            return out
        with Patch("verify_block", mutating):
            with self.assertRaisesRegex(ERR, "^P2_PAYLOAD_MUTATED:"):
                vmod.verify_pillar2()
        self.assertEqual(len(touched), 2048)
        calm = []

        def observing(c, s, rows=vmod.H4_ROWS):
            calm.append(vmod.pack_nibbles(c))
            return real(c, s, rows)
        with Patch("verify_block", observing):
            self.assertEqual(vmod.verify_pillar2()["flip_payload_bytes_changed"], 0)
        self.assertEqual(len(calm), 2048)


class TestTernary(unittest.TestCase):
    def test_happy(self):
        res = payload()["pillar2_ternary"]
        self.assertEqual(res["blocks"], 32)
        self.assertEqual(res["state_syndromes_zero"], 96)
        self.assertEqual(res["exact_restores"], 64)
        self.assertEqual(res["zero_state_norm_sq"], "0/1")
        self.assertIs(res["neg_state_is_negation"], True)
        self.assertEqual(res["bounced_invalid_requests"], 3)
        c = codeword_block(4)
        self.assertEqual(vmod.ternary_view(-1, c, Fraction(1))[0], vmod.complement16(vmod.lsb_mask(c)))
        self.assertEqual(vmod.ternary_view(0, c, Fraction(1))[0], 0)
        self.assertEqual(vmod.switch_state(-1, 2), -1)
        self.assertEqual(vmod.switch_state(0, -1), -1)

    def test_guards(self):
        c = codeword_block(4)
        with self.assertRaisesRegex(ERR, "^T_STATE:"):
            vmod.ternary_view(2, c, Fraction(1))
        bad = list(c)
        bad[0] = bad[0] ^ 1
        with self.assertRaisesRegex(ERR, "^T_SYNDROME:"):
            vmod.verify_ternary([bad])
        with Patch("switch_state", lambda cur, req: cur):
            with self.assertRaisesRegex(ERR, "^T_REVERSE:"):
                vmod.verify_ternary([c])


class TestPillar3(unittest.TestCase):
    def test_happy(self):
        res = payload()["pillar3_weft_rollback"]
        self.assertEqual(res["checkpoint_bytes"], 16)
        self.assertEqual(res["d_sq"], vmod.CANON_D * vmod.CANON_D)
        self.assertEqual(res["spinor_flips"], 9)
        self.assertEqual(res["pre_rollback_spinor"], [-x for x in vmod.CANON_SPINOR])
        self.assertIn(0, res["pre_rollback_switches"])
        self.assertIn(-1, res["pre_rollback_switches"])
        self.assertEqual(res["post_rollback_switches"], [1] * 8)
        self.assertEqual(res["bounced_switch_requests"], 2)
        self.assertEqual(res["bounced_assignments"], 2)
        self.assertEqual(res["bounced_tampered_checkpoints"], 3)
        self.assertEqual(res["spinor_norm_drift"], "0/1")
        self.assertEqual(res["weight_norm_drift"], "0/1")
        self.assertEqual(bytes.fromhex(res["checkpoint_hex"]), struct.pack("<4i", *vmod.CANON_SPINOR))
        edge = vmod.verify_pillar3(spinor=(2147483647, 0, 0, 0), d=2147483647)
        self.assertEqual(edge["checkpoint_bytes"], 16)
        quiet = vmod.verify_pillar3(ops=())
        self.assertEqual(quiet["pre_rollback_switches"], [1] * 8)
        self.assertEqual(quiet["spinor_flips"], 0)

    def test_guards(self):
        with self.assertRaisesRegex(ERR, "^W_INT32_RANGE:"):
            vmod.verify_pillar3(spinor=(2147483648, 0, 0, 0), d=2147483648)
        with self.assertRaisesRegex(ERR, "^W_INT32_RANGE:"):
            vmod.verify_pillar3(spinor=(-2147483649, 0, 0, 0), d=2147483649)
        with self.assertRaisesRegex(ERR, "^W_UNIT_NORM:"):
            vmod.verify_pillar3(d=vmod.CANON_D + 1)
        real_pack = vmod.pack16B
        with Patch("pack16B", lambda q: real_pack(q)[:15]):
            with self.assertRaisesRegex(ERR, "^W_PACK_LEN:"):
                vmod.verify_pillar3()
        real_unpack = vmod.unpack16B
        with Patch("unpack16B", lambda raw: tuple(x + 1 for x in real_unpack(raw))):
            with self.assertRaisesRegex(ERR, "^W_ROUNDTRIP:"):
                vmod.verify_pillar3()
        probes = set([vmod.CANON_SPINOR, vmod.flip_spinor(vmod.CANON_SPINOR), (-2147483648, 2147483647, 0, -1), (1, -1, 2147483647, -2147483647)])
        with Patch("pack16B", lambda q: real_pack(q) if tuple(q) in probes else bytes(16)):
            with self.assertRaisesRegex(ERR, "^W_BYTES_ROUNDTRIP:"):
                vmod.verify_pillar3()
        with self.assertRaisesRegex(ERR, "^W_OP_KIND:"):
            vmod.verify_pillar3(ops=(("bogus", 0, 0),))
        with Patch("flip_spinor", lambda q: (2 * q[0], q[1], q[2], q[3])):
            with self.assertRaisesRegex(ERR, "^W_ROUND_NORM:"):
                vmod.verify_pillar3(ops=(("switch", 0, -1),))
        with Patch("assign_base", lambda cur, cand: (list(cand), 0)):
            with self.assertRaisesRegex(ERR, "^W_ROUND_SYN:"):
                vmod.verify_pillar3(ops=(("assign", 0, "tamper_lsb"),))
        with Patch("rollback", lambda state, ckpt, digest, sw, dsq: (state, 1)):
            with self.assertRaisesRegex(ERR, "^W_ROLLBACK_BOUNCED:"):
                vmod.verify_pillar3()
        with Patch("rollback", lambda state, ckpt, digest, sw, dsq: (state, 0)):
            with self.assertRaisesRegex(ERR, "^W_RESTORE_BYTES:"):
                vmod.verify_pillar3()
        q0 = vmod.CANON_SPINOR
        with Patch("rollback", lambda state, ckpt, digest, sw, dsq: ({"switches": state["switches"], "spinor": q0}, 0)):
            with self.assertRaisesRegex(ERR, "^W_RESTORE_WEIGHTS:"):
                vmod.verify_pillar3()

    def test_rollback_and_assign_units(self):
        q0 = vmod.CANON_SPINOR
        ckpt = vmod.pack16B(q0)
        import hashlib
        digest = hashlib.sha256(ckpt).hexdigest()
        state = {"switches": [0], "spinor": vmod.flip_spinor(q0)}
        dsq = vmod.CANON_D * vmod.CANON_D
        self.assertEqual(vmod.rollback(state, ckpt[:15], digest, [1], dsq), (state, 1))
        self.assertEqual(vmod.rollback(state, bytes([ckpt[0] ^ 1]) + ckpt[1:], digest, [1], dsq), (state, 1))
        other = vmod.pack16B((1, 2, 3, 4))
        self.assertEqual(vmod.rollback(state, other, hashlib.sha256(other).hexdigest(), [1], dsq), (state, 1))
        self.assertEqual(vmod.rollback(state, ckpt, digest, [1], dsq), ({"switches": [1], "spinor": q0}, 0))
        c = codeword_block(3)
        self.assertEqual(vmod.assign_base(c, c[:15]), (c, 1))
        self.assertEqual(vmod.assign_base(c, vmod.tamper_candidate(c, "tamper_lsb")), (c, 1))
        d = codeword_block(8)
        self.assertEqual(vmod.assign_base(c, d), (d, 0))
        self.assertEqual(vmod.sign_of(-1), -1)
        self.assertEqual(vmod.sign_of(0), 1)


class TestPillar4(unittest.TestCase):
    def test_chantless(self):
        res = payload()["pillar4_chantless"]
        self.assertEqual(res["retention_drops"], ["397/2500", "537/5000", "829/10000"])
        self.assertEqual(res["inkling"], "3/23")
        self.assertEqual(res["witness_blocks"], 64)
        self.assertEqual(res["strict_collapses"], 64)
        self.assertEqual(res["distinct_images"], 1)
        edge = vmod.verify_chantless(retention=((10000, 9999),), blocks=[[1], [2]])
        self.assertEqual(edge["retention_drops"], ["1/10000"])
        with self.assertRaisesRegex(ERR, "^C_RETENTION_DROP:"):
            vmod.verify_chantless(retention=((8333, 9921),))
        with self.assertRaisesRegex(ERR, "^C_RETENTION_DROP:"):
            vmod.verify_chantless(retention=((5000, 5000),))
        with self.assertRaisesRegex(ERR, "^C_RETENTION_RANGE:"):
            vmod.verify_chantless(retention=((10001, 5),))
        with self.assertRaisesRegex(ERR, "^C_INKLING:"):
            vmod.verify_chantless(inkling=(7, 46))
        with self.assertRaisesRegex(ERR, "^C_DISTINCT:"):
            vmod.verify_chantless(blocks=[[1, 2], [1, 2]])
        with self.assertRaisesRegex(ERR, "^C_NONZERO:"):
            vmod.verify_chantless(blocks=[[1, 2], [0, 0]])
        with Patch("up_row_zero", lambda w: list(w)):
            with self.assertRaisesRegex(ERR, "^C_ZERO_NORM:"):
                vmod.verify_chantless(blocks=[[1, 2]])
        with self.assertRaisesRegex(ERR, "^C_COLLISION:"):
            vmod.verify_chantless(blocks=[[1, 2], [1, 2, 3]])

    def test_carving(self):
        res = payload()["pillar4_carving"]
        self.assertEqual(res["locked_reopened"], "0/193")
        self.assertEqual(res["locked_reopened_count"], 0)
        self.assertEqual(res["hold_open_reopened_count"], 193)
        self.assertEqual(res["hold_open_eps"], "1/1000000")
        self.assertEqual(res["kept_half_reopen"], "141/184")
        self.assertEqual(res["kept_tenth_reopen"], "33/35")
        self.assertEqual(res["positive_gate_after_step"], "11/15")
        self.assertEqual(vmod.carving_step(Fraction(0), Fraction(0), Fraction(1)), 0)
        self.assertEqual(vmod.hold_open_step(Fraction(0), Fraction(0), Fraction(1), Fraction(1, 7)), Fraction(1, 7))
        edge = vmod.verify_carving(removed=2, kept=184, half=(184, 184), tenth=(184, 184), pressures=(Fraction(0),))
        self.assertEqual(edge["locked_reopened_count"], 0)
        with self.assertRaisesRegex(ERR, "^K_PRESSURE_NEG:"):
            vmod.verify_carving(pressures=(Fraction(0), Fraction(-1, 10)))
        with self.assertRaisesRegex(ERR, "^K_REOPEN_GRAD:"):
            vmod.verify_carving(reopen=Fraction(0))
        with self.assertRaisesRegex(ERR, "^K_KEPT_HALF:"):
            vmod.verify_carving(half=(184, 185))
        with self.assertRaisesRegex(ERR, "^K_KEPT_HALF:"):
            vmod.verify_carving(kept=183)
        with self.assertRaisesRegex(ERR, "^K_KEPT_TENTH:"):
            vmod.verify_carving(tenth=(35, 36))
        with self.assertRaisesRegex(ERR, "^K_KEPT_TENTH:"):
            vmod.verify_carving(tenth=(185, 33))
        with Patch("gate_grad", lambda g, raw: raw):
            with self.assertRaisesRegex(ERR, "^K_LOCKOUT:"):
                vmod.verify_carving()
        with Patch("hold_open_step", lambda g, u, p, eps: Fraction(0)):
            with self.assertRaisesRegex(ERR, "^K_HOLD_OPEN:"):
                vmod.verify_carving()

    def test_duality(self):
        res = payload()["pillar4_duality"]
        self.assertEqual(res["ternary_recovered"], "193/193")
        self.assertEqual(res["ternary_lockout"], "0/193")
        self.assertEqual(res["carving_lockout"], "193/193")
        with Patch("switch_state", lambda cur, req: 0):
            with self.assertRaisesRegex(ERR, "^D_RECOVERY:"):
                vmod.verify_duality(n=3)


class TestStrictIO(unittest.TestCase):
    def test_strict_io(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "good.json"
            good.write_bytes(b'{"a": 1}\n')
            self.assertEqual(vmod.read_strict_json(good), {"a": 1})
            with self.assertRaisesRegex(ERR, "^READ_OPEN:"):
                vmod.read_strict_bytes(root / "missing.json")
            fifo = root / "pipe"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(ERR, "^NOT_REGULAR:"):
                vmod.read_strict_bytes(fifo)
            link = root / "link.json"
            link.symlink_to(good)
            with self.assertRaisesRegex(ERR, "^SYMLINK_ANCESTOR:"):
                vmod.read_strict_bytes(link)
            exact_big = root / "exact_big.json"
            exact_big.write_bytes(b"{}" + b" " * (vmod.MAX_BYTES - 3) + b"\n")
            self.assertEqual(len(exact_big.read_bytes()), vmod.MAX_BYTES)
            self.assertEqual(vmod.read_strict_json(exact_big), {})
            big = root / "big.bin"
            big.write_bytes(b"a" * (vmod.MAX_BYTES + 1))
            with self.assertRaisesRegex(ERR, "^TOO_LARGE:"):
                vmod.read_strict_bytes(big)
            for code, raw in [("BAD_UTF8", b"\xff\n"), ("BAD_BOM", '{"a": 1}\n'.encode("utf-8-sig")), ("BAD_LF", b'{"a": 1}\r\n'), ("BAD_FINAL_LF", b'{"a": 1}')]:
                p = root / (code + ".json")
                p.write_bytes(raw)
                with self.assertRaisesRegex(ERR, "^" + code + ":"):
                    vmod.read_strict_text(p)
            real_sub = root / "real" / "sub"
            real_sub.mkdir(parents=True)
            link_anc = root / "real" / "sym"
            link_anc.symlink_to(real_sub)
            (real_sub / "target.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ERR, "^SYMLINK_ANCESTOR:"):
                vmod.read_strict_bytes(str(link_anc) + "/../sub/target.json")
        cwd_orig = os.getcwd()
        try:
            os.chdir(str(BASE))
            rel_cert = vmod.read_strict_json(CERT)
            self.assertEqual(rel_cert["queue_id"], "EXP-Q42a")
        finally:
            os.chdir(cwd_orig)
        with self.assertRaisesRegex(ERR, "^READ_OPEN:"):
            vmod.read_strict_bytes("/")
        dfd, leaf = vmod._open_parent_nofollow("/tmp")
        os.close(dfd)
        self.assertEqual(leaf, "tmp")
        with self.assertRaisesRegex(ERR, "^READ_OPEN:"):
            vmod.read_strict_bytes("/nonexistent_s163_top_entry")
        with self.assertRaisesRegex(ERR, "^READ_OPEN:"):
            vmod.read_strict_bytes(str(BASE / "missing_subdir" / CERT))

    def test_strict_json_parse(self):
        cases = [
            ("DEPTH", "{" + '"a": {' * 33 + "1" + "}" * 34 + "\n"),
            ("UNBALANCED", "]\n"),
            ("UNBALANCED_FINAL", "{\n"),
            ("INT_DIGITS", '{"a": ' + "1" * 257 + "}\n"),
            ("FLOAT_TOKEN", '{"a": 12' + "." + "5}\n"),
            ("FLOAT_TOKEN", '{"a": 1e5}\n'),
            ("FLOAT_TOKEN", '{"a": 1E5}\n'),
            ("UNTERMINATED_JSON_STRING", '"abc\n'),
            ("DUP_KEY", '{"a": 1, "a": 2}\n'),
            ("BAD_CONSTANT", '{"a": NaN}\n'),
            ("BAD_JSON", '{"a": }\n'),
            ("NOT_OBJECT", "[1, 2]\n"),
        ]
        for code, text in cases:
            with self.assertRaisesRegex(ERR, "^" + code + ":"):
                vmod.parse_strict_json(text)
        self.assertEqual(vmod.parse_strict_json('{"a": {"b": [1, 2]}, "s": "x\\"1.5e3"}\n'), {"a": {"b": [1, 2]}, "s": 'x"1.5e3'})
        d32 = '{"a": ' + "[" * 31 + "0" + "]" * 31 + "}\n"
        self.assertIsInstance(vmod.parse_strict_json(d32), dict)
        n256 = '{"a": ' + "9" * 256 + "}\n"
        self.assertEqual(vmod.parse_strict_json(n256)["a"], int("9" * 256))
        orig_loads = vmod.json.loads
        try:
            vmod.json.loads = lambda *a, **kw: (_ for _ in ()).throw(ValueError("limit"))
            with self.assertRaisesRegex(ERR, "^BAD_JSON:"):
                vmod.parse_strict_json('{"a": 1}\n')
        finally:
            vmod.json.loads = orig_loads


class TestLean(unittest.TestCase):
    def test_lean_strip_and_axioms(self):
        self.assertNotIn("sorry", vmod.strip_lean_comments_and_strings("-- sorry\ntheorem x := 1\n"))
        self.assertNotIn("sorry", vmod.strip_lean_comments_and_strings('def s := "sorry"\n'))
        self.assertNotIn("sorry", vmod.strip_lean_comments_and_strings("/- a /- sorry -/ b -/ x\n"))
        self.assertIn("x", vmod.strip_lean_comments_and_strings("/- a /- sorry -/ b -/ x\n"))
        self.assertIn("y", vmod.strip_lean_comments_and_strings('def s := "a\\"sorry" ++ y\n'))
        with self.assertRaisesRegex(ERR, "^UNTERMINATED_BLOCK_COMMENT:"):
            vmod.strip_lean_comments_and_strings("/- abc")
        with self.assertRaisesRegex(ERR, "^UNTERMINATED_STRING_LITERAL:"):
            vmod.strip_lean_comments_and_strings('"abc')
        n = vmod.EXPECTED_THEOREMS
        good = good_runner(Path("x"))["axiom_stdout"].splitlines()
        counts = vmod.validate_axiom_lines(good, n)
        self.assertEqual(counts["zero_axiom"], 1)
        self.assertEqual(counts["listed"], n - 1)
        with self.assertRaisesRegex(ERR, "^AXIOM_LINE_COUNT:"):
            vmod.validate_axiom_lines([], n)
        for code, line in [("AXIOM_LINE_FORMAT", "garbage"), ("AXIOM_LBRACKET", "x depends on axioms: propext]"), ("AXIOM_RBRACKET", "x depends on axioms: [propext"), ("FORBIDDEN_AXIOM", "x depends on axioms: [Classical.choice]")]:
            with self.assertRaisesRegex(ERR, "^" + code + ":"):
                vmod.validate_axiom_lines([line] * n, n)

    def test_lean_kernel_escapes_rejected(self):
        basic = Path("lean") / "TernaryA16Carving" / "Basic.lean"
        appended = [
            "set_option debug.skipKernelTC true in\ntheorem esc1 : 1 = 1 := rfl\n",
            "theorem esc2 : True := by\n  classical\n  trivial\n",
            "theorem esc3 : True := Lean.ofReduceBool _ _ rfl\n",
            "noncomputable def esc4 : Nat := 0\n",
            "partial def esc5 (n : Nat) : Nat := esc5 n\n",
            "#reduce (1 : Nat) + 1\n",
            "#  eval 1\n",
            "#guard_msgs in\n#check 1\n",
            "def esc6 : Nat := panic! \"x\"\n",
            "def esc7 := IO . println 1\n",
            "opaque esc8 : Nat\n",
            "macro_rules | `(x) => `(y)\n",
        ]
        for snippet in appended:
            with tempfile.TemporaryDirectory() as tmp:
                root = copy_lean(tmp)
                p = root / basic
                p.write_text(p.read_text(encoding="utf-8") + snippet, encoding="utf-8")
                with self.assertRaisesRegex(ERR, "^LEAN_ESCAPE:"):
                    vmod.verify_lean_package(root)
        for header in ["public import Mathlib\n", "private import Lean.Elab\n", "meta public import Std\n", "import all Init\n"]:
            with tempfile.TemporaryDirectory() as tmp:
                root = copy_lean(tmp)
                p = root / basic
                p.write_text(header + p.read_text(encoding="utf-8"), encoding="utf-8")
                with self.assertRaisesRegex(ERR, "^BAD_IMPORT:"):
                    vmod.verify_lean_package(root)
        with tempfile.TemporaryDirectory() as tmp:
            root = copy_lean(tmp)
            p = root / "lean" / "TernaryA16Carving.lean"
            p.write_text("public import TernaryA16Carving.Basic\n" + p.read_text(encoding="utf-8"), encoding="utf-8")
            q = root / basic
            q.write_text(q.read_text(encoding="utf-8") + "-- sorry classical set_option partial #reduce\n/- noncomputable ofReduceBool -/\ndef okStr : String := \"#eval IO.println\"\n", encoding="utf-8")
            self.assertEqual(vmod.verify_lean_package(root)["public_theorems"], vmod.EXPECTED_THEOREMS)
        tokens = ["sorry", "admit", "axiom", "unsafe", "native_decide", "sorryAx", "Classical", "classical",
                  "ofReduceBool", "ofReduceNat", "trustCompiler", "implemented_by", "extern", "initialize",
                  "builtin_initialize", "run_cmd", "run_meta", "run_tac", "syntax", "declare_syntax_cat", "macro",
                  "macro_rules", "elab", "elab_rules", "notation", "dbg_trace", "dbgTrace", "dbgTraceVal",
                  "dbgTraceIfShared", "dbgSleep", "dbgStackTrace", "panicWithPosWithDecl", "throwError", "csimp",
                  "opaque", "mkSorry", "bv_decide", "set_option", "noncomputable", "partial", "prelude", "panic!",
                  "unreachable!", "#eval", "# reduce", "#exit", "#check", "#check_failure", "#synth", "#guard",
                  "#guard_msgs", "IO.println", "IO . print", "debug.skipKernelTC", "debug . byAsSorry",
                  "debug.proofAsSorry", "debug.sorryAsError", "warn.sorry", "warn . sorry"]
        for tok in tokens:
            self.assertIsNotNone(vmod._FORBIDDEN_LEAN_RE.search("x " + tok + " y"), tok)
        for benign in ["partial_sum", "Classically", "simp", "csimp_x", "#print axioms Foo", "axioms", "macroX", "opaqueness"]:
            self.assertIsNone(vmod._FORBIDDEN_LEAN_RE.search("x " + benign + " y"), benign)
        mods = [m.group(1) for m in vmod._ANY_IMPORT_RE.finditer("public import A\n  private import B.C\nmeta public import D\nimport E\n")]
        self.assertEqual(mods, ["A", "B.C", "D", "E"])

    def test_lean_package_fake(self):
        info = vmod.verify_lean_package(BASE, lean_runner=good_runner)
        self.assertEqual(info["public_theorems"], vmod.EXPECTED_THEOREMS)
        self.assertEqual(info["lean_files"], 3)
        self.assertIn("carving_zero_absorbing", info["theorem_names"])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ERR, "^MISSING_LEAN_DIR:"):
                vmod.verify_lean_package(Path(tmp))
        basic = Path("lean") / "TernaryA16Carving" / "Basic.lean"
        variants = []

        def bad_toolchain(root):
            (root / "lean" / "lean-toolchain").write_text("leanprover/lean4:v0\n", encoding="utf-8")
        variants.append(("BAD_TOOLCHAIN", bad_toolchain))

        def bad_manifest(root):
            (root / "lean" / "lake-manifest.json").write_text('{"packages": [{"x": 1}]}\n', encoding="utf-8")
        variants.append(("NONEMPTY_PACKAGES", bad_manifest))

        def bad_lakefile(root):
            (root / "lean" / "lakefile.toml").write_text('name = "Other"\n', encoding="utf-8")
        variants.append(("BAD_LAKEFILE", bad_lakefile))

        def extra_file(root):
            (root / "lean" / "Extra.lean").write_text("theorem extra : 1 = 1 := rfl\n", encoding="utf-8")
        variants.append(("LEAN_FILE_COUNT", extra_file))

        def escape(root):
            p = root / basic
            p.write_text(p.read_text(encoding="utf-8") + "theorem bad : 1 = 1 := sorry\n", encoding="utf-8")
        variants.append(("LEAN_ESCAPE", escape))

        def eval_escape(root):
            p = root / basic
            p.write_text(p.read_text(encoding="utf-8") + "#eval 1\n", encoding="utf-8")
        variants.append(("LEAN_ESCAPE", eval_escape))

        def bad_import(root):
            p = root / basic
            p.write_text("import Mathlib\n" + p.read_text(encoding="utf-8"), encoding="utf-8")
        variants.append(("BAD_IMPORT", bad_import))

        def thm_count(root):
            p = root / basic
            p.write_text(p.read_text(encoding="utf-8").replace("theorem carving_lockout_witness", "theoremX carving_lockout_witness", 1), encoding="utf-8")
        variants.append(("THM_COUNT", thm_count))

        def audit_count(root):
            p = root / "lean" / "AxiomAudit.lean"
            lines = p.read_text(encoding="utf-8").splitlines(True)
            p.write_text("".join(lines[:-1]), encoding="utf-8")
        variants.append(("AUDIT_COUNT", audit_count))

        def audit_set(root):
            p = root / "lean" / "AxiomAudit.lean"
            p.write_text(p.read_text(encoding="utf-8").replace("carving_lockout_witness", "nope_name", 1), encoding="utf-8")
        variants.append(("AUDIT_SET", audit_set))

        for code, mutate in variants:
            with tempfile.TemporaryDirectory() as tmp:
                root = copy_lean(tmp)
                mutate(root)
                with self.assertRaisesRegex(ERR, "^" + code + ":"):
                    vmod.verify_lean_package(root)
        with tempfile.TemporaryDirectory() as tmp:
            root = copy_lean(tmp)
            (root / "lean" / ".lake" / "build").mkdir(parents=True)
            (root / "lean" / ".lake" / "build" / "Stray.lean").write_text("theorem s : 1 = 1 := rfl\n", encoding="utf-8")
            self.assertEqual(vmod.verify_lean_package(root)["lean_files"], 3)
        for code, kw in [("LEAN_VERSION_RC", {"version_rc": 1}), ("BAD_LEAN_VERSION", {"version_output": "Lean (version 0)"}), ("LAKE_BUILD_FAIL", {"build_rc": 1}), ("LAKE_BUILD_WARNING", {"build_output": "Warning: x"}), ("AXIOM_RC", {"axiom_rc": 1}), ("AXIOM_LINE_COUNT", {"axiom_stdout": ""})]:
            with self.assertRaisesRegex(ERR, "^" + code + ":"):
                vmod.verify_lean_package(BASE, lean_runner=runner_with(**kw))
        seen = []

        def fake_default(lean):
            seen.append(lean)
            return good_runner(lean)
        with Patch("default_live_lean_runner", fake_default):
            vmod.verify_lean_package(BASE, run_live_lean=True)
            vmod.verify_lean_package(BASE, run_live_lean=False)
        self.assertEqual(len(seen), 1)


def _tree_snapshot(root):
    out = []
    for dirpath, dirnames, filenames in os.walk(str(root)):
        dirnames.sort()
        for name in sorted(dirnames + filenames):
            full = Path(dirpath) / name
            info = os.lstat(str(full))
            out.append((str(full.relative_to(root)), stat.S_IMODE(info.st_mode), stat.S_IFMT(info.st_mode)))
    return out


def _set_tree_writable(root, writable):
    for dirpath, dirnames, filenames in os.walk(str(root), topdown=False):
        for name in filenames:
            full = Path(dirpath) / name
            if full.is_symlink() is False:
                os.chmod(str(full), 0o644 if writable else 0o444)
        os.chmod(dirpath, 0o755 if writable else 0o555)


class TestLiveLeanStaging(unittest.TestCase):
    def test_readonly_snapshot_staging(self):
        rel_files = sorted(str(Path(*parts)) for parts in vmod.LEAN_PACKAGE_FILES)
        self.assertEqual(len(rel_files), 6)
        with tempfile.TemporaryDirectory() as scratch_s:
            scratch = Path(scratch_s)
            tmp_root = scratch / "tmproot"
            tmp_root.mkdir()

            class TempProxy:
                @staticmethod
                def TemporaryDirectory():
                    return tempfile.TemporaryDirectory(dir=str(tmp_root))

            calls = []

            def fake_run(cmd, cwd=None, env=None, capture_output=None, text=None, timeout=None, check=None):
                work = Path(cwd)
                calls.append((list(cmd), env, timeout, check))
                staged = sorted(str(p.relative_to(work)) for p in work.rglob("*") if p.is_file())
                if len(calls) == 1:
                    self.assertEqual(staged, rel_files)
                    self.assertEqual(stat.S_IMODE(os.stat(str(work)).st_mode), 0o755)
                    self.assertEqual(stat.S_IMODE(os.stat(str(work / "TernaryA16Carving")).st_mode), 0o755)
                    for rel in rel_files:
                        self.assertEqual((work / rel).read_bytes(), (BASE / "lean" / rel).read_bytes())
                        self.assertEqual(stat.S_IMODE(os.stat(str(work / rel)).st_mode), 0o644)
                if cmd[:2] == ["lake", "build"]:
                    (work / ".lake" / "build").mkdir(parents=True)
                    (work / ".lake" / "build" / "x.olean").write_bytes(b"o")
                return subprocess.CompletedProcess(cmd, 0, "out:" + cmd[0], "")

            ro_root = scratch / "ro"
            ro_root.mkdir()
            copy_lean(str(ro_root))
            ro_lean = ro_root / "lean"
            (ro_lean / "NOTES.txt").write_text("not staged\n", encoding="utf-8")
            (ro_lean / "extra_dir").mkdir()
            (ro_lean / "extra_dir" / "Junk.lean").write_text("theorem j : 1 = 1 := rfl\n", encoding="utf-8")
            before_bytes = {rel: (ro_lean / rel).read_bytes() for rel in rel_files}
            _set_tree_writable(ro_lean, False)
            try:
                before = _tree_snapshot(ro_lean)
                self.assertEqual(stat.S_IMODE(os.stat(str(ro_lean)).st_mode), 0o555)
                with Patch("subprocess", OsProxy(run=fake_run, SubprocessError=subprocess.SubprocessError)), Patch("tempfile", TempProxy):
                    res = vmod.default_live_lean_runner(ro_lean)
                self.assertEqual([c[0] for c in calls], [["lean", "--version"], ["lake", "build"], ["lake", "env", "lean", "AxiomAudit.lean"]])
                self.assertEqual([c[2] for c in calls], [60, 300, 180])
                self.assertEqual([c[3] for c in calls], [False, False, False])
                self.assertEqual(calls[0][1]["CUDA_VISIBLE_DEVICES"], "")
                self.assertEqual(calls[0][1]["PYTHONDONTWRITEBYTECODE"], "1")
                self.assertEqual(res["version_rc"], 0)
                self.assertEqual(res["build_rc"], 0)
                self.assertEqual(res["axiom_rc"], 0)
                self.assertEqual(res["build_output"], "out:lake")
                self.assertEqual(res["axiom_stdout"], "out:lake")
                self.assertEqual(_tree_snapshot(ro_lean), before)
                self.assertFalse((ro_lean / ".lake").exists())
                self.assertEqual(list(tmp_root.iterdir()), [])
                for rel in rel_files:
                    self.assertEqual((ro_lean / rel).read_bytes(), before_bytes[rel])
            finally:
                _set_tree_writable(ro_lean, True)

            dst = scratch / "direct"
            self.assertEqual(vmod.stage_lean_package(BASE / "lean", dst), 6)
            self.assertEqual(sorted(str(p.relative_to(dst)) for p in dst.rglob("*") if p.is_file()), rel_files)

            def plant_symlink(lean):
                target = lean / "TernaryA16Carving" / "Basic.lean"
                target.unlink()
                os.symlink(str(BASE / "lean" / "TernaryA16Carving" / "Basic.lean"), str(target))

            def plant_fifo(lean):
                target = lean / "lakefile.toml"
                target.unlink()
                os.mkfifo(str(target))

            def plant_dir_symlink(lean):
                sub = lean / "TernaryA16Carving"
                shutil.rmtree(str(sub))
                os.symlink(str(BASE / "lean" / "TernaryA16Carving"), str(sub))

            def drop_file(lean):
                (lean / "AxiomAudit.lean").unlink()

            for idx, (code, plant) in enumerate([("SYMLINK_ANCESTOR", plant_symlink), ("NOT_REGULAR", plant_fifo), ("SYMLINK_ANCESTOR", plant_dir_symlink), ("READ_OPEN", drop_file)]):
                root = scratch / ("bad" + str(idx))
                root.mkdir()
                copy_lean(str(root))
                lean = root / "lean"
                plant(lean)
                _set_tree_writable(lean, False)
                calls.clear()
                try:
                    with Patch("subprocess", OsProxy(run=fake_run)), Patch("tempfile", TempProxy):
                        with self.assertRaisesRegex(ERR, "^" + code + ":"):
                            vmod.default_live_lean_runner(lean)
                    self.assertEqual(calls, [])
                    self.assertEqual(list(tmp_root.iterdir()), [])
                    self.assertFalse((lean / ".lake").exists())
                finally:
                    _set_tree_writable(lean, True)

            def timeout_run(cmd, **kw):
                raise subprocess.TimeoutExpired(cmd, kw["timeout"])

            with Patch("subprocess", OsProxy(run=timeout_run, SubprocessError=subprocess.SubprocessError)), Patch("tempfile", TempProxy), Patch("build_payload", lambda **kw: vmod.verify_lean_package(BASE, run_live_lean=True)):
                rc, err = run_main_capture(["--standalone", "--live-lean"])
            self.assertEqual(rc, 2)
            self.assertEqual(json.loads(err), {"code": "SUBPROCESS_ERROR", "message": "TimeoutExpired", "status": "FAIL"})
            self.assertEqual(list(tmp_root.iterdir()), [])


class TestWriteCert(unittest.TestCase):
    def assert_cli_rejects(self, target, code):
        with Patch("build_payload", lambda **kw: payload()):
            rc, err = run_main_capture(["--write", str(target)])
        self.assertEqual(rc, 2)
        self.assertEqual(len(err.splitlines()), 1)
        parsed = json.loads(err)
        self.assertEqual(parsed["status"], "FAIL")
        self.assertEqual(parsed["code"], code)

    def test_atomic_write_positive(self):
        raw = cert_bytes()
        old_mask = os.umask(0o077)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                fresh = root / "fresh.json"
                vmod.write_cert(fresh, payload())
                self.assertEqual(fresh.read_bytes(), raw)
                self.assertEqual(oct(fresh.stat().st_mode & 0o777), oct(0o644))
                self.assertEqual(sorted(os.listdir(tmp)), ["fresh.json"])
                old = root / "old.json"
                old.write_bytes(b"stale\n")
                os.chmod(old, 0o600)
                ino = old.stat().st_ino
                vmod.write_cert(old, payload())
                self.assertEqual(old.read_bytes(), raw)
                self.assertEqual(oct(old.stat().st_mode & 0o777), oct(0o644))
                self.assertNotEqual(old.stat().st_ino, ino)
                self.assertEqual(sorted(os.listdir(tmp)), ["fresh.json", "old.json"])
                sizes = []

                def small_write(fd, data):
                    sizes.append(len(data))
                    return os.write(fd, bytes(data[:3]))
                chunked = root / "chunked.json"
                with Patch("os", OsProxy(write=small_write)):
                    vmod.write_cert(chunked, payload())
                self.assertEqual(chunked.read_bytes(), raw)
                self.assertGreater(len(sizes), len(raw) // 3 - 1)
                self.assertEqual(sizes[0], len(raw))
                with Patch("build_payload", lambda **kw: payload()):
                    rc, err = run_main_capture(["--write", str(root / "cli.json")])
                self.assertEqual((rc, err), (0, ""))
                self.assertEqual((root / "cli.json").read_bytes(), raw)
        finally:
            os.umask(old_mask)

    def test_write_failures_are_atomic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "cert.json"
            target.write_bytes(b"keep\n")
            calls = []

            def zero_write(fd, data):
                calls.append(1)
                if len(calls) > 1:
                    raise RuntimeError("write retried without progress")
                return 0
            with Patch("os", OsProxy(write=zero_write)):
                with self.assertRaisesRegex(ERR, "^WRITE_SHORT:"):
                    vmod.write_cert(target, payload())
            self.assertEqual(len(calls), 1)
            self.assertEqual(target.read_bytes(), b"keep\n")
            self.assertEqual(os.listdir(tmp), ["cert.json"])

            def broken_replace(*a, **kw):
                raise OSError("replace refused")
            with Patch("os", OsProxy(replace=broken_replace)):
                with self.assertRaisesRegex(ERR, "^WRITE_REPLACE:"):
                    vmod.write_cert(target, payload())
            self.assertEqual(target.read_bytes(), b"keep\n")
            self.assertEqual(os.listdir(tmp), ["cert.json"])
            victim = root / "victim.txt"
            victim.write_bytes(b"victim\n")
            os.symlink(str(victim), str(root / ".cert.json.fixed.tmp"))
            with Patch("_temp_token", lambda: "fixed"):
                with self.assertRaisesRegex(ERR, "^WRITE_TEMP:"):
                    vmod.write_cert(target, payload())
            self.assertEqual(victim.read_bytes(), b"victim\n")
            self.assertEqual(target.read_bytes(), b"keep\n")
            self.assertEqual(len(vmod._temp_token()), 16)
            self.assertNotEqual(vmod._temp_token(), vmod._temp_token())
            with self.assertRaisesRegex(ERR, "^WRITE_STAT:"):
                vmod.write_cert(root / ("x" * 300), payload())
            with self.assertRaisesRegex(ERR, "^WRITE_OPEN:"):
                vmod.write_cert(root / "missing" / "cert.json", payload())
            with self.assertRaisesRegex(ERR, "^WRITE_OPEN:"):
                vmod.write_cert("/", payload())
            self.assertEqual(sorted(os.listdir(tmp)), [".cert.json.fixed.tmp", "cert.json", "victim.txt"])

    def test_write_rejects_special_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            victim = root / "victim.txt"
            victim.write_bytes(b"victim\n")
            link = root / "link.json"
            link.symlink_to(victim)
            with self.assertRaisesRegex(ERR, "^SYMLINK_ANCESTOR:"):
                vmod.write_cert(link, payload())
            self.assert_cli_rejects(link, "SYMLINK_ANCESTOR")
            self.assertTrue(link.is_symlink())
            self.assertEqual(victim.read_bytes(), b"victim\n")
            dfd, leaf = vmod._open_parent_nofollow(link, "WRITE_OPEN")
            try:
                with self.assertRaisesRegex(ERR, "^WRITE_NOT_REGULAR:"):
                    vmod._check_write_leaf(dfd, leaf)
                self.assertEqual(vmod._check_write_leaf(dfd, "victim.txt"), 1)
                self.assertEqual(vmod._check_write_leaf(dfd, "absent.json"), 0)
            finally:
                os.close(dfd)
            real = root / "a" / "real"
            real.mkdir(parents=True)
            (root / "a" / "link").symlink_to(real)
            sneaky = str(root / "a" / "link") + "/../cert.json"
            with self.assertRaisesRegex(ERR, "^SYMLINK_ANCESTOR:"):
                vmod.write_cert(sneaky, payload())
            self.assert_cli_rejects(sneaky, "SYMLINK_ANCESTOR")
            self.assertEqual(sorted(os.listdir(str(root / "a"))), ["link", "real"])
            self.assertEqual(os.listdir(str(real)), [])
            fifo = root / "pipe.json"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(ERR, "^WRITE_NOT_REGULAR:"):
                vmod.write_cert(fifo, payload())
            self.assert_cli_rejects(fifo, "WRITE_NOT_REGULAR")
            self.assertTrue(stat_is_fifo(fifo))
            folder = root / "dir.json"
            folder.mkdir()
            with self.assertRaisesRegex(ERR, "^WRITE_NOT_REGULAR:"):
                vmod.write_cert(folder, payload())
            self.assert_cli_rejects(folder, "WRITE_NOT_REGULAR")
            self.assertEqual(os.listdir(folder), [])
            first = root / "first.json"
            first.write_bytes(b"shared\n")
            second = root / "second.json"
            os.link(first, second)
            with self.assertRaisesRegex(ERR, "^WRITE_HARDLINK:"):
                vmod.write_cert(second, payload())
            self.assert_cli_rejects(second, "WRITE_HARDLINK")
            self.assertEqual(first.read_bytes(), b"shared\n")
            self.assertEqual(second.read_bytes(), b"shared\n")
            self.assertEqual(sorted(os.listdir(tmp)), ["a", "dir.json", "first.json", "link.json", "pipe.json", "second.json", "victim.txt"])


def stat_is_fifo(p):
    return stat.S_ISFIFO(os.lstat(p).st_mode)


def bool_int_leaf_paths(node, path=()):
    """Every path to a bool leaf or an int 0/1 leaf, i.e. every leaf that `==` cannot tell from its bool/int twin."""
    out = []
    if type(node) is dict:
        for key in sorted(node):
            out.extend(bool_int_leaf_paths(node[key], path + (key,)))
    elif type(node) is list:
        for idx in range(len(node)):
            out.extend(bool_int_leaf_paths(node[idx], path + (idx,)))
    elif type(node) is bool:
        out.append(path)
    elif type(node) is int:
        if node in (0, 1):
            out.append(path)
    return out


def swap_bool_int(doc, path):
    """Deep copy of `doc` with the leaf at `path` swapped bool <-> int (True <-> 1, False <-> 0)."""
    clone = json.loads(json.dumps(doc))
    cur = clone
    for step in path[:-1]:
        cur = cur[step]
    leaf = cur[path[-1]]
    cur[path[-1]] = int(leaf) if type(leaf) is bool else bool(leaf)
    return clone


def write_spoof(path, doc):
    Path(path).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class TestExactJsonEqual(unittest.TestCase):
    def test_unit_cases(self):
        eq = vmod._json_exact_equal
        for lhs, rhs in [(True, 1), (1, True), (False, 0), (0, False), (None, 0), (0, None), (False, None),
                         ("1", 1), ([1], (1,)), ({}, []), ([], {}), ({"a": 1}, [["a", 1]])]:
            self.assertIs(eq(lhs, rhs), False, (lhs, rhs))
        for value in [True, False, 0, 1, -1, 2 ** 70, "", "x", None, [], {}, [1, [True, None]], {"a": {"b": [0, False, "s"]}}]:
            self.assertIs(eq(value, json.loads(json.dumps(value))), True, value)
        self.assertIs(eq({"a": 1, "b": [2]}, {"b": [2], "a": 1}), True)
        for lhs, rhs in [({"a": 1}, {"a": 1, "b": 2}), ({"a": 1, "b": 2}, {"a": 1}), ({"a": 1}, {"b": 1}), ({1: "x"}, {True: "x"})]:
            self.assertIs(eq(lhs, rhs), False, ("dict keys", lhs, rhs))
        for lhs, rhs in [({"a": 1}, {"a": 2}), ({"a": True}, {"a": 1}), ({"a": {"b": 0}}, {"a": {"b": False}})]:
            self.assertIs(eq(lhs, rhs), False, ("dict value", lhs, rhs))
        for lhs, rhs in [([1], [1, 2]), ([1, 2], [1]), ([], [0])]:
            self.assertIs(eq(lhs, rhs), False, ("list length", lhs, rhs))
        for lhs, rhs in [([1, 2], [1, 3]), ([0, 1], [0, True]), ([[0]], [[False]])]:
            self.assertIs(eq(lhs, rhs), False, ("list element", lhs, rhs))
        for lhs, rhs in [(1, 2), ("a", "b"), (True, False), (2 ** 70, 2 ** 70 + 1)]:
            self.assertIs(eq(lhs, rhs), False, ("scalar", lhs, rhs))

    def test_every_bool_int_leaf_spoof_rejected(self):
        p = payload()
        cert = vmod.read_strict_json(BASE / CERT)
        self.assertIs(vmod._json_exact_equal(cert, p), True)
        paths = bool_int_leaf_paths(p)
        self.assertEqual(len(paths), 20)
        self.assertEqual([path for path in paths if path[-1] == "neg_state_is_negation"], [("pillar2_ternary", "neg_state_is_negation")])
        for path in paths:
            spoof = swap_bool_int(p, path)
            self.assertEqual(spoof, p, path)
            self.assertIs(vmod._json_exact_equal(spoof, p), False, path)
            self.assertIs(vmod._json_exact_equal(p, spoof), False, path)

    def test_cli_bool_spoof_rejected(self):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        mismatch = {"code": "CERT_MISMATCH", "message": "CERT_MISMATCH: certificate mismatch", "status": "FAIL"}
        with tempfile.TemporaryDirectory() as tmp:
            data = json.loads((BASE / CERT).read_text(encoding="utf-8"))
            self.assertIs(data["pillar1_a16_complement"]["syndrome_0x0000"], 0)
            data["pillar1_a16_complement"]["syndrome_0x0000"] = False
            spoof = Path(tmp) / "bool_spoof.json"
            write_spoof(spoof, data)
            self.assertIn('"syndrome_0x0000": false', spoof.read_text(encoding="utf-8"))
            self.assertEqual(vmod.read_strict_json(spoof), payload())
            bad = subprocess.run([sys.executable, "-I", "-B", VERIFIER, "--check", str(spoof), "--json"], cwd=str(BASE), env=env, capture_output=True, text=True, timeout=120, check=False)
            self.assertEqual(bad.returncode, 2)
            self.assertEqual(bad.stdout, "")
            self.assertEqual(len(bad.stderr.splitlines()), 1)
            self.assertEqual(json.loads(bad.stderr), mismatch)
            with Patch("build_payload", lambda **kw: payload()):
                for path in [("pillar3_weft_rollback", "post_rollback_switches", 0), ("pillar2_ternary", "neg_state_is_negation")]:
                    nested = Path(tmp) / "nested.json"
                    write_spoof(nested, swap_bool_int(payload(), path))
                    rc, err = run_main_capture(["--check", str(nested), "--json"])
                    self.assertEqual(rc, 2, path)
                    self.assertEqual(json.loads(err), mismatch)
                control = Path(tmp) / "control.json"
                write_spoof(control, payload())
                self.assertEqual(run_main_capture(["--check", str(control)]), (0, ""))


class TestPayloadAndCli(unittest.TestCase):
    def test_payload_and_ast(self):
        p = payload()
        self.assertEqual(p["queue_id"], "EXP-Q42a")
        self.assertEqual(p["schema"], "s163.ternary_a16_carving.v1")
        self.assertNotIn("live_lean", json.dumps(p))
        src = (BASE / VERIFIER).read_text(encoding="utf-8")
        tree = ast.parse(src)
        self.assertEqual(sum(1 for n in ast.walk(tree) if isinstance(n, ast.BoolOp)), 0)
        self.assertEqual(sum(1 for n in ast.walk(tree) if isinstance(n, ast.Assert)), 0)
        self.assertEqual(sum(1 for n in ast.walk(tree) if isinstance(n, ast.BinOp) if isinstance(n.op, ast.Div)), 0)
        floats = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) if isinstance(n.value, float)]
        self.assertEqual(floats, [])
        calls = [n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) if isinstance(n.func, ast.Name)]
        self.assertNotIn("float", calls)
        host_home = "/" + "home" + "/"
        self.assertNotIn(host_home, src)
        self.assertNotIn(host_home, (BASE / "test_s163_ternary_a16_r5.py").read_text(encoding="utf-8"))
        cert = vmod.read_strict_json(BASE / CERT)
        self.assertEqual(cert, p)
        self.assertIs(vmod._json_exact_equal(cert, p), True)
        self.assertEqual(vmod.canonical_cert_bytes(p), (BASE / CERT).read_bytes())
        guard_args = [n.args[0] for n in ast.walk(tree) if isinstance(n, ast.Call) if isinstance(n.func, ast.Name) if n.func.id == "require_ok" if ast.unparse(n.args[1]) == "'CERT_MISMATCH'"]
        self.assertEqual([ast.unparse(a) for a in guard_args], ["_json_exact_equal(cert, payload)"])

    def test_main_paths(self):
        orig = vmod.build_payload
        try:
            vmod.build_payload = lambda **kw: (_ for _ in ()).throw(RecursionError("x"))
            self.assertEqual(vmod.main(["--standalone"]), 2)
            vmod.build_payload = lambda **kw: (_ for _ in ()).throw(OSError("x"))
            self.assertEqual(vmod.main(["--standalone"]), 2)
            vmod.build_payload = lambda **kw: (_ for _ in ()).throw(ERR("X_CODE", "y"))
            self.assertEqual(vmod.main(["--standalone"]), 2)
            vmod.build_payload = lambda **kw: payload()
            self.assertEqual(vmod.main([]), 0)
            self.assertEqual(vmod.main(["--standalone"]), 0)
            self.assertEqual(vmod.main(["--check", "--standalone"]), 2)
            self.assertEqual(vmod.main(["--write", "--standalone"]), 2)
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "w.json"
                self.assertEqual(vmod.main(["--write", str(out)]), 0)
                self.assertEqual(out.read_bytes(), (BASE / CERT).read_bytes())
                self.assertEqual(oct(out.stat().st_mode & 0o777), oct(0o644))
                self.assertEqual(vmod.main(["--check", str(out)]), 0)
        finally:
            vmod.build_payload = orig
        self.assertEqual(vmod.resolve_cert_path(BASE, None), BASE / CERT)
        self.assertEqual(vmod.resolve_cert_path(BASE, "x.json"), BASE / "x.json")

    def test_cli_subprocess(self):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        run = subprocess.run([sys.executable, "-I", "-B", VERIFIER, "--check", "--json"], cwd=str(BASE), env=env, capture_output=True, text=True, timeout=120, check=False)
        self.assertEqual(run.returncode, 0)
        self.assertEqual(run.stderr, "")
        parsed = json.loads(run.stdout)
        self.assertEqual(parsed, payload())
        with tempfile.TemporaryDirectory() as tmp:
            tampered = Path(tmp) / "cert.json"
            data = json.loads((BASE / CERT).read_text(encoding="utf-8"))
            data["pillar4_carving"]["locked_reopened_count"] = 1
            tampered.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            bad = subprocess.run([sys.executable, "-I", "-B", VERIFIER, "--check", str(tampered), "--json"], cwd=str(BASE), env=env, capture_output=True, text=True, timeout=120, check=False)
            self.assertEqual(bad.returncode, 2)
            self.assertEqual(bad.stdout, "")
            err = json.loads(bad.stderr)
            self.assertEqual(err["code"], "CERT_MISMATCH")
            self.assertEqual(err["status"], "FAIL")
            self.assertEqual(len(bad.stderr.splitlines()), 1)
            big = Path(tmp) / "big.json"
            big.write_text('{"a": ' + "9" * 400 + "}\n", encoding="utf-8")
            env2 = dict(env)
            env2["PYTHONINTMAXSTRDIGITS"] = "0"
            bigrun = subprocess.run([sys.executable, "-I", "-B", VERIFIER, "--check", str(big), "--json"], cwd=str(BASE), env=env2, capture_output=True, text=True, timeout=120, check=False)
            self.assertEqual(bigrun.returncode, 2)
            self.assertIn("INT_DIGITS", bigrun.stderr)
            self.assertNotIn("Traceback", bigrun.stderr)


if __name__ == "__main__":
    unittest.main()
