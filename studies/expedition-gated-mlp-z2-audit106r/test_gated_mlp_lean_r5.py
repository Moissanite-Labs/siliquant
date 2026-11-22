"""Assertion-free, -O-safe R5 verification suite for Track S9 (AUDIT-106):
Gated-MLP Z_2 Gauge Conservation (SwiGLU + Separate/Fused Muon + Learnable Multipliers),
Symmetry-Breaking Controls, and weft-loom#44 / AUDIT-059 Exact Verification.
"""

import copy
from fractions import Fraction
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gated_mlp_lean_oracle as o
import gated_mlp_lean_audit as a


class TestGatedMlpLeanR5(unittest.TestCase):
    def _assert_err(self, code, fn, *args, **kwargs):
        with self.assertRaises(o.GatedMlpLeanError) as ctx:
            fn(*args, **kwargs)
        self.assertEqual(ctx.exception.code, code)

    def test_primitives_and_label_guard(self):
        o.verify_no_floats({"a": [1, Fraction(1, 2), ("s", True, None)]})
        self._assert_err("FLOAT_FORBIDDEN", o.verify_no_floats, 0.25)
        self._assert_err("FLOAT_FORBIDDEN", o.verify_no_floats, 1 + 2j)
        self._assert_err("FLOAT_FORBIDDEN", o.verify_no_floats, {1.5: "x"})
        self._assert_err("INVALID_CONTAINER", o.verify_no_floats, {1, 2})
        self._assert_err("INVALID_CONTAINER", o.verify_no_floats, frozenset([1]))

        self.assertEqual(o.require_non_empty_str("ok", "f"), "ok")
        self._assert_err("EMPTY_STRING", o.require_non_empty_str, "", "f")
        self._assert_err("EMPTY_STRING", o.require_non_empty_str, " ok ", "f")
        self._assert_err("EMPTY_STRING", o.require_non_empty_str, " standalone_fixture ", "f")
        self._assert_err("INVALID_TYPE", o.require_non_empty_str, 7, "f")
        self._assert_err("CONTROL_CHAR", o.require_non_empty_str, "a\x1fb", "f")
        self._assert_err("CONTROL_CHAR", o.require_non_empty_str, "a\x7fb", "f")
        self._assert_err("CONTROL_CHAR", o.require_non_empty_str, "a\u200bb", "f")
        self._assert_err("EMPTY_STRING", o.require_non_empty_str, "   ", "f")

        self.assertIs(o.require_exact_bool(True, "b"), True)
        self._assert_err("INVALID_BOOL", o.require_exact_bool, 1, "b")
        self.assertEqual(o.require_exact_int(5, "x", 5, 5), 5)
        self._assert_err("INVALID_INT", o.require_exact_int, True, "x")
        self._assert_err("RANGE_ERROR", o.require_exact_int, 4, "x", min_value=5)
        self._assert_err("RANGE_ERROR", o.require_exact_int, 6, "x", max_value=5)

        self.assertEqual(o.require_exact_fraction(3, "q"), Fraction(3, 1))
        self.assertEqual(o.require_exact_fraction(Fraction(1, 2), "q"), Fraction(1, 2))
        self._assert_err("INVALID_FRACTION", o.require_exact_fraction, False, "q")
        self._assert_err("INVALID_FRACTION", o.require_exact_fraction, "1/2", "q")

        self.assertEqual(o.frac_str(Fraction(4, 6)), "2/3")
        self.assertEqual(o.format_exact_decimal(Fraction(3, 2), 0), "2")
        self.assertEqual(o.format_exact_decimal(Fraction(-3, 2), 0), "-2")
        self.assertEqual(o.format_exact_decimal(Fraction(-1, 3), 0), "0")
        self.assertEqual(o.format_exact_decimal(Fraction(1, 20), 1), "0.1")
        self.assertEqual(o.format_exact_decimal(Fraction(49, 1000), 1), "0.0")
        self.assertEqual(o.format_exact_decimal(Fraction(-1, 200), 2), "-0.01")
        self.assertEqual(o.format_exact_decimal(Fraction(-1, 10000), 2), "0.00")
        self.assertEqual(o.canonical_json_bytes({"b": 1, "a": 2}), b'{"a":2,"b":1}')
        self.assertEqual(o.canonical_sha256({"x": 1}), hashlib.sha256(b'{"x":1}').hexdigest())

        self.assertFalse(o._strict_equal(1, True))
        self.assertFalse(o._strict_equal({"a": 1}, {"a": 1, "b": 2}))
        self.assertFalse(o._strict_equal({"a": 1}, {"a": True}))
        self.assertFalse(o._strict_equal([1], [1, 2]))
        self.assertFalse(o._strict_equal([1], [True]))
        self.assertFalse(o._strict_equal(1, 2))
        self.assertTrue(o._strict_equal({"k": [1, True, "s"]}, {"k": [1, True, "s"]}))

        g = o.enforce_law5_label_hygiene(["SwiGLU_Z2", "Muon-NS"])
        self.assertTrue(g["label_filter_ok"])
        self._assert_err("INVALID_LABELS", o.enforce_law5_label_hygiene, [])
        self._assert_err("INVALID_LABELS", o.enforce_law5_label_hygiene, "abc")
        self._assert_err("EMPTY_STRING", o.enforce_law5_label_hygiene, ["---"])
        self._assert_err("DUPLICATE_LABEL", o.enforce_law5_label_hygiene, ["a-b", "A_B"])
        for bad in ("miracle", "quantum_consciousness", "magical", "numerology", "telepathy"):
            self._assert_err("METAPHOR_FORBIDDEN", o.enforce_law5_label_hygiene, [bad])

    def test_pillar1_z2_conservation_muon_lrm(self):
        p1 = o.audit_z2_conservation_muon_lrm_exact()
        self.assertTrue(p1["passed"])
        scope = p1["operator_scope"]
        self.assertIs(scope["gate_is_transcendental_silu"], False)
        self.assertIs(scope["norm_is_sqrt_rmsnorm"], False)
        self.assertIs(scope["rms_like_backprop_and_muon_verified"], True)

        branch_keys = (
            "forward_output_max_diff_exact",
            "loss_diff_exact",
            "lrm_shared_row_grad_max_diff_exact",
            "grad_wg_max_diff_exact",
            "grad_wu_equivariance_diff_exact",
            "grad_wd_equivariance_diff_exact",
            "muon_separate_equivariance_diff_exact",
            "muon_fused_gate_up_equivariance_diff_exact",
        )
        for branch_name in ("unnormalized_branch", "rms_like_norm_branch"):
            branch = p1[branch_name]
            self.assertTrue(branch["passed"], branch_name)
            for k in branch_keys:
                self.assertEqual(branch[k], "0/1", f"{branch_name}.{k}")
        self.assertTrue(p1["gate_up_inner_product_sign_flip_exact"])

        # Independent exact pure-Q dual-number Jacobian check across all 3 hidden_mode branches
        # (proves exact 0/1 analytical backprop for none, rms_like_sq_denom, and layernorm_center)
        def d_add(a, b):
            return (a[0] + b[0], a[1] + b[1])
        def d_sub(a, b):
            return (a[0] - b[0], a[1] - b[1])
        def d_mul(a, b):
            return (a[0] * b[0], a[0] * b[1] + a[1] * b[0])
        def d_div(a, b):
            return (a[0] / b[0], (a[1] * b[0] - a[0] * b[1]) / (b[0] * b[0]))
        def d_abs(a):
            return a if a[0] >= 0 else (-a[0], -a[1])
        def dual_loss(w_g, w_u, w_d, r_sh, s_g, s_u, s_d, x, target, mode, asym_alpha):
            h_dim, d_dim = len(w_g), len(x)
            z = Fraction(0, 1)
            h_raw = []
            for i in range(h_dim):
                gp = (z, z)
                up = (z, z)
                for j in range(d_dim):
                    xj = (x[j], z)
                    gp = d_add(gp, d_mul(w_g[i][j], xj))
                    up = d_add(up, d_mul(w_u[i][j], xj))
                gp = d_mul(d_mul((s_g, z), r_sh[i]), gp)
                up = d_mul(d_mul((s_u, z), r_sh[i]), up)
                denom = d_add((Fraction(1, 1), z), d_abs(gp))
                sig = d_mul((Fraction(1, 2), z), d_add((Fraction(1, 1), z), d_div(gp, denom)))
                h_raw.append(d_mul(d_mul(gp, sig), up))
            if mode == "none":
                h_eff = h_raw
            elif mode == "rms_like_sq_denom":
                msq = (z, z)
                for v in h_raw:
                    msq = d_add(msq, d_mul(v, v))
                msq = d_mul(msq, (Fraction(1, h_dim), z))
                q = d_add((Fraction(1, 1), z), msq)
                h_eff = [d_div(v, q) for v in h_raw]
            else:
                mh = (z, z)
                for v in h_raw:
                    mh = d_add(mh, v)
                mh = d_mul(mh, (Fraction(1, h_dim), z))
                h_eff = [d_sub(v, mh) for v in h_raw]
            loss_d = (z, z)
            for out_i in range(d_dim):
                yo = (z, z)
                for i in range(h_dim):
                    yo = d_add(yo, d_mul(w_d[out_i][i], h_eff[i]))
                yo = d_mul((s_d, z), yo)
                dy = d_sub(yo, (target[out_i], z))
                loss_d = d_add(loss_d, d_mul((Fraction(1, 2), z), d_mul(dy, dy)))
            sum_h = (z, z)
            for v in h_raw:
                sum_h = d_add(sum_h, v)
            return d_sub(loss_d, d_mul((asym_alpha, z), sum_h))

        wg0 = [[Fraction(1, 2), Fraction(-1, 3)], [Fraction(-2, 5), Fraction(1, 2)], [Fraction(3, 5), Fraction(1, 4)]]
        wu0 = [[Fraction(2, 3), Fraction(1, 5)], [Fraction(1, 4), Fraction(-2, 3)], [Fraction(-1, 2), Fraction(3, 4)]]
        wd0 = [[Fraction(1, 2), Fraction(-1, 4), Fraction(2, 5)], [Fraction(-1, 3), Fraction(3, 5), Fraction(-1, 2)]]
        r0 = [Fraction(6, 5), Fraction(4, 5), Fraction(7, 5)]
        sg0, su0, sd0 = Fraction(5, 4), Fraction(3, 4), Fraction(7, 8)
        x0 = [Fraction(2, 1), Fraction(-3, 2)]
        t0 = [Fraction(1, 2), Fraction(-1, 4)]
        z0 = Fraction(0, 1)
        one0 = Fraction(1, 1)
        for mode in ("none", "rms_like_sq_denom", "layernorm_center"):
            res_m = o._swiglu_forward_and_grads(wg0, wu0, wd0, r0, sg0, su0, sd0, x0, t0, hidden_mode=mode, asym_alpha=Fraction(1, 3))
            for i in range(3):
                for j in range(2):
                    dg = [[(wg0[r][c], one0 if (r, c) == (i, j) else z0) for c in range(2)] for r in range(3)]
                    du = [[(wu0[r][c], z0) for c in range(2)] for r in range(3)]
                    dd = [[(wd0[r][c], z0) for c in range(3)] for r in range(2)]
                    dr = [(r0[r], z0) for r in range(3)]
                    lv, dlv = dual_loss(dg, du, dd, dr, sg0, su0, sd0, x0, t0, mode, Fraction(1, 3))
                    self.assertEqual(lv, res_m["loss"])
                    self.assertEqual(dlv, res_m["grad_wg"][i][j])
                    dg0 = [[(wg0[r][c], z0) for c in range(2)] for r in range(3)]
                    du1 = [[(wu0[r][c], one0 if (r, c) == (i, j) else z0) for c in range(2)] for r in range(3)]
                    _, dlv_u = dual_loss(dg0, du1, dd, dr, sg0, su0, sd0, x0, t0, mode, Fraction(1, 3))
                    self.assertEqual(dlv_u, res_m["grad_wu"][i][j])
                dg0 = [[(wg0[r][c], z0) for c in range(2)] for r in range(3)]
                du0 = [[(wu0[r][c], z0) for c in range(2)] for r in range(3)]
                dd0 = [[(wd0[r][c], z0) for c in range(3)] for r in range(2)]
                dr1 = [(r0[r], one0 if r == i else z0) for r in range(3)]
                _, dlv_r = dual_loss(dg0, du0, dd0, dr1, sg0, su0, sd0, x0, t0, mode, Fraction(1, 3))
                self.assertEqual(dlv_r, res_m["grad_r"][i])
            for oi in range(2):
                for i in range(3):
                    dg0 = [[(wg0[r][c], z0) for c in range(2)] for r in range(3)]
                    du0 = [[(wu0[r][c], z0) for c in range(2)] for r in range(3)]
                    dd1 = [[(wd0[r][c], one0 if (r, c) == (oi, i) else z0) for c in range(3)] for r in range(2)]
                    dr0 = [(r0[r], z0) for r in range(3)]
                    _, dlv_d = dual_loss(dg0, du0, dd1, dr0, sg0, su0, sd0, x0, t0, mode, Fraction(1, 3))
                    self.assertEqual(dlv_d, res_m["grad_wd"][oi][i])

        # Guards
        self._assert_err("INVALID_SIGNS", o.audit_z2_conservation_muon_lrm_exact, "1234")
        self._assert_err("INVALID_SIGNS", o.audit_z2_conservation_muon_lrm_exact, (1, -1))
        self._assert_err("INVALID_SIGNS", o.audit_z2_conservation_muon_lrm_exact, (1, 0, -1, 1))
        self._assert_err("INVALID_MATRIX", o.newton_schulz_odd_poly_exact, 42)
        self._assert_err("INVALID_MATRIX", o.newton_schulz_odd_poly_exact, [[1, 2], "ab"])
        self._assert_err("INVALID_MATRIX", o.newton_schulz_odd_poly_exact, [])
        self._assert_err("INVALID_MATRIX", o.newton_schulz_odd_poly_exact, [[]])
        self._assert_err("INVALID_MATRIX", o.newton_schulz_odd_poly_exact, [[1, 2], [3]])
        self._assert_err("INVALID_MODE", o._swiglu_forward_and_grads, [[1]], [[1]], [[1]], [1], 1, 1, 1, [1], [1], hidden_mode="bad")

        # Pin EVERY branch residual (both unnormalized & rms_like_norm) and inner_flip_exact individually!
        for idx in range(4):
            seq = [Fraction(0, 1)] * 4
            seq[idx] = Fraction(1, 1)
            with mock.patch.object(o, "_vec_max_abs_diff", side_effect=seq):
                self.assertFalse(o.audit_z2_conservation_muon_lrm_exact()["passed"])
        for idx in range(12):
            seq = [Fraction(0, 1)] * 12
            seq[idx] = Fraction(1, 1)
            with mock.patch.object(o, "_mat_max_abs_diff", side_effect=seq):
                self.assertFalse(o.audit_z2_conservation_muon_lrm_exact()["passed"])
        orig_fw = o._swiglu_forward_and_grads
        for bad_call in (2, 4):
            calls = [0]
            def _bad_loss(*args, _bc=bad_call, _c=calls, **kwargs):
                r = orig_fw(*args, **kwargs)
                _c[0] += 1
                if _c[0] == _bc:
                    r = dict(r, loss=r["loss"] + Fraction(1, 1))
                return r
            with mock.patch.object(o, "_swiglu_forward_and_grads", side_effect=_bad_loss):
                self.assertFalse(o.audit_z2_conservation_muon_lrm_exact()["passed"])
        # Clean isolation of inner_flip_exact: patch _dot on the last call only
        orig_dot = o._dot
        n_dots = [0]
        def _count_dot(x, y):
            n_dots[0] += 1
            return orig_dot(x, y)
        with mock.patch.object(o, "_dot", side_effect=_count_dot):
            o.audit_z2_conservation_muon_lrm_exact()
        total_dots = n_dots[0]
        n_dots[0] = 0
        def _corrupt_last_dot(x, y):
            n_dots[0] += 1
            v = orig_dot(x, y)
            return v + Fraction(1, 1) if n_dots[0] == total_dots else v
        with mock.patch.object(o, "_dot", side_effect=_corrupt_last_dot):
            self.assertFalse(o.audit_z2_conservation_muon_lrm_exact()["passed"])

    def test_pillar2_symmetry_breaking_controls(self):
        p2 = o.audit_z2_symmetry_breaking_controls_exact()
        self.assertTrue(p2["passed"])
        self.assertTrue(p2["b1_layernorm_breaks_z2"])
        self.assertTrue(p2["b2_asym_penalty_breaks_z2"])
        self.assertTrue(p2["b3_matches_orthogonal_invariant_signature"])
        self._assert_err("INVALID_PARAMS", o.audit_z2_symmetry_breaking_controls_exact, asym_alpha=0)
        self._assert_err("INVALID_PARAMS", o.audit_z2_symmetry_breaking_controls_exact, step_lr=0)

        # Pin EVERY conjunct in Pillar 2 individually
        with mock.patch.object(o, "_vec_max_abs_diff", return_value=Fraction(0, 1)):
            self.assertFalse(o.audit_z2_symmetry_breaking_controls_exact()["passed"])
        orig_dot = o._dot
        n_dots = [0]
        def _cnt(x, y):
            n_dots[0] += 1
            return orig_dot(x, y)
        with mock.patch.object(o, "_dot", side_effect=_cnt):
            o.audit_z2_symmetry_breaking_controls_exact()
        tot = n_dots[0]
        # Zero out the 8 twin_inner_sum dots (indices tot-11 .. tot-4) -> b2 False, b3 True!
        n_dots[0] = 0
        def _zero_b2(x, y):
            n_dots[0] += 1
            return Fraction(0, 1) if (tot - 11 <= n_dots[0] <= tot - 4) else orig_dot(x, y)
        with mock.patch.object(o, "_dot", side_effect=_zero_b2):
            r_b2 = o.audit_z2_symmetry_breaking_controls_exact()
            self.assertFalse(r_b2["b2_asym_penalty_breaks_z2"])
            self.assertTrue(r_b2["b3_matches_orthogonal_invariant_signature"])
            self.assertFalse(r_b2["passed"])
        # Zero out the last dot (corr_gu_pos -> 3 != 4) -> b3 False!
        n_dots[0] = 0
        def _zero_b3(x, y):
            n_dots[0] += 1
            return Fraction(0, 1) if n_dots[0] == tot else orig_dot(x, y)
        with mock.patch.object(o, "_dot", side_effect=_zero_b3):
            r_b3 = o.audit_z2_symmetry_breaking_controls_exact()
            self.assertFalse(r_b3["b3_matches_orthogonal_invariant_signature"])
            self.assertFalse(r_b3["passed"])
        # Pin corr_up_rowsum_pos == 2 conjunct independently via mocked sum on 3-element row
        orig_sum = sum
        with mock.patch("builtins.sum", side_effect=lambda it, *ar, **kw: Fraction(1, 1) if isinstance(it, list) and len(it) == 3 and it[0] == Fraction(-2, 1) else orig_sum(it, *ar, **kw)):
            r_b3b = o.audit_z2_symmetry_breaking_controls_exact()
            self.assertFalse(r_b3b["b3_matches_orthogonal_invariant_signature"])
            self.assertFalse(r_b3b["passed"])

    def test_pillar3_and_pillar4_certification_and_live_json(self):
        self.assertEqual(o.exact_binomial_two_sided(1, 0), Fraction(1, 1))
        self.assertEqual(o.exact_binomial_two_sided(24, 21), Fraction(2325, 8388608))
        self.assertEqual(o.exact_hypergeom_tail(4, 2, 2, 2), Fraction(1, 6))

        cert = a.run_audit(live=False)
        self.assertTrue(cert["all_gates_passed"])
        self.assertEqual(cert["verdict"], "AUDITED_GATED_MLP_SURROGATE_Z2_EQUIVARIANCE_AND_RULINGS")
        self.assertEqual(cert["mode"], "standalone_fixture_replay")
        self.assertIs(cert["live_json_recounted"], False)
        self.assertIs(cert["causal_and_statistical_scope"]["rules_out_finite_run_asymmetry"], False)
        self.assertIs(cert["causal_and_statistical_scope"]["identifies_falcon_unreleased_training_cause"], False)
        body_no_sha = {k: v for k, v in cert.items() if k != "certificate_sha256"}
        self.assertEqual(cert["certificate_sha256"], o.canonical_sha256(body_no_sha))

        p1 = cert["pillar1_z2_equivariance_surrogate_witness"]
        p2 = cert["pillar2_z2_symmetry_breaking_controls"]
        p3 = cert["pillar3_weft_loom_weight_lean_and_audit059"]
        axes = dict(cert["three_axis_audit_checks"])
        fx = copy.deepcopy(a.STANDALONE_WEIGHT_LEAN_FIXTURE)

        # Pillar 3 guards
        self._assert_err("INVALID_SUMMARY", o.audit_weight_lean_and_audit059_exact, "bad")
        self._assert_err("INVALID_SUMMARY", o.audit_weight_lean_and_audit059_exact, {})
        bad_s = copy.deepcopy(fx); bad_s["diag_source"] = "forged"
        self._assert_err("INVALID_SUMMARY", o.audit_weight_lean_and_audit059_exact, bad_s)
        bad_m = copy.deepcopy(fx); bad_m["models"] = {}
        self._assert_err("INVALID_MODELS", o.audit_weight_lean_and_audit059_exact, bad_m)
        bad_m2 = copy.deepcopy(fx); bad_m2["models"] = []
        self._assert_err("INVALID_MODELS", o.audit_weight_lean_and_audit059_exact, bad_m2)
        bad_r = copy.deepcopy(fx); bad_r["models"]["Qwen__Qwen3.5-0.8B"] = {"gu_pos": 1}
        self._assert_err("INVALID_MODEL_REC", o.audit_weight_lean_and_audit059_exact, bad_r)
        bad_r2 = copy.deepcopy(fx); bad_r2["models"]["Qwen__Qwen3.5-0.8B"] = []
        self._assert_err("INVALID_MODEL_REC", o.audit_weight_lean_and_audit059_exact, bad_r2)

        # Pin EVERY conjunct in Pillar 3 passed individually!
        # 1. Falcon 0.6B layers_gt_half < 44
        f_l = copy.deepcopy(fx)
        f_l["models"]["tiiuae__Falcon-H1-Tiny-R-0.6B"]["layers_gt_half"] = 43
        f_l["models"]["tiiuae__Falcon-H1-Tiny-R-0.6B-pre-GRPO"]["layers_gt_half"] = 43
        self.assertFalse(o.audit_weight_lean_and_audit059_exact(f_l)["passed"])
        # 2. Falcon 0.6B gu_pos/units <= 3/4
        f_g = copy.deepcopy(fx)
        f_g["models"]["tiiuae__Falcon-H1-Tiny-R-0.6B"]["gu_pos"] = 67584
        f_g["models"]["tiiuae__Falcon-H1-Tiny-R-0.6B-pre-GRPO"]["gu_pos"] = 67584
        self.assertFalse(o.audit_weight_lean_and_audit059_exact(f_g)["passed"])
        # 3. Falcon pre-GRPO differs from 0.6B
        f_d = copy.deepcopy(fx)
        f_d["models"]["tiiuae__Falcon-H1-Tiny-R-0.6B-pre-GRPO"]["gu_pos"] = 70499
        self.assertFalse(o.audit_weight_lean_and_audit059_exact(f_d)["passed"])
        # 4. Falcon 90M layers_gt_half < 24 and gu_pos/units <= 3/5
        f_90l = copy.deepcopy(fx); f_90l["models"]["tiiuae__Falcon-H1-Tiny-90M-Base"]["layers_gt_half"] = 23
        self.assertFalse(o.audit_weight_lean_and_audit059_exact(f_90l)["passed"])
        f_90g = copy.deepcopy(fx); f_90g["models"]["tiiuae__Falcon-H1-Tiny-90M-Base"]["gu_pos"] = 11000
        self.assertFalse(o.audit_weight_lean_and_audit059_exact(f_90g)["passed"])
        # S9-BUG-01: exact_stratified_two_block_tail validates all 6 block args + obs_total
        self._assert_err("INVALID_INT", o.exact_stratified_two_block_tail, True, 0, 0, 1, 0, 0, 0)
        self._assert_err("INVALID_INT", o.exact_stratified_two_block_tail, 5.0, 0, 0, 1, 0, 0, 0)
        self._assert_err("RANGE_ERROR", o.exact_stratified_two_block_tail, -5, 0, 0, 1, 0, 0, 0)
        self._assert_err("RANGE_ERROR", o.exact_stratified_two_block_tail, 5, 6, 1, 5, 1, 1, 0)
        self._assert_err("RANGE_ERROR", o.exact_stratified_two_block_tail, 5, 1, 6, 5, 1, 1, 0)
        self._assert_err("RANGE_ERROR", o.exact_stratified_two_block_tail, 5, 1, 1, 0, 0, 0, 0)
        self._assert_err("RANGE_ERROR", o.exact_stratified_two_block_tail, 5, 1, 1, 5, 6, 1, 0)
        self._assert_err("RANGE_ERROR", o.exact_stratified_two_block_tail, 5, 1, 1, 5, 1, 6, 0)
        self._assert_err("RANGE_ERROR", o.exact_stratified_two_block_tail, 5, 1, 1, 5, 1, 1, 3)

        # S9-BUG-03: Falcon up_pos / dn_pos must remain near 50% (rejects 100% skew!)
        f_up = copy.deepcopy(fx)
        f_up["models"]["tiiuae__Falcon-H1-Tiny-R-0.6B"]["up_pos"] = 90112
        f_up["models"]["tiiuae__Falcon-H1-Tiny-R-0.6B-pre-GRPO"]["up_pos"] = 90112
        self.assertFalse(o.audit_weight_lean_and_audit059_exact(f_up)["passed"])
        f_dn = copy.deepcopy(fx)
        f_dn["models"]["tiiuae__Falcon-H1-Tiny-R-0.6B"]["dn_pos"] = 90112
        f_dn["models"]["tiiuae__Falcon-H1-Tiny-R-0.6B-pre-GRPO"]["dn_pos"] = 90112
        self.assertFalse(o.audit_weight_lean_and_audit059_exact(f_dn)["passed"])

        # 5. Free-sign control off half or all layers > half
        f_c1 = copy.deepcopy(fx); f_c1["models"]["Qwen__Qwen3.5-0.8B"]["gu_pos"] = 50000
        self.assertFalse(o.audit_weight_lean_and_audit059_exact(f_c1)["passed"])
        f_c2 = copy.deepcopy(fx); f_c2["models"]["Qwen__Qwen3.5-0.8B"]["layers_gt_half"] = 24
        self.assertFalse(o.audit_weight_lean_and_audit059_exact(f_c2)["passed"])
        # 6. AUDIT-059 exact tail conjuncts isolated
        with mock.patch.object(o, "exact_binomial_two_sided", return_value=Fraction(1, 2)):
            self.assertFalse(o.audit_weight_lean_and_audit059_exact(fx)["passed"])
        with mock.patch.object(o, "exact_stratified_two_block_tail", side_effect=[Fraction(1, 2), Fraction(1, 1000)]):
            self.assertFalse(o.audit_weight_lean_and_audit059_exact(fx)["passed"])
        with mock.patch.object(o, "exact_stratified_two_block_tail", side_effect=[Fraction(1, 10 ** 10), Fraction(1, 2)]):
            self.assertFalse(o.audit_weight_lean_and_audit059_exact(fx)["passed"])
        with mock.patch.object(o, "exact_hypergeom_tail", side_effect=[Fraction(1, 1000), Fraction(1, 20)]):
            self.assertFalse(o.audit_weight_lean_and_audit059_exact(fx)["passed"])
        with mock.patch.object(o, "exact_hypergeom_tail", side_effect=[Fraction(1, 20), Fraction(1, 1000)]):
            self.assertFalse(o.audit_weight_lean_and_audit059_exact(fx)["passed"])
        with mock.patch.object(o, "_gemma_l10_l9_margin_exact", return_value=Fraction(1, 10)):
            self.assertFalse(o.audit_weight_lean_and_audit059_exact(fx)["passed"])

        # Honest REJECTED_GATE_FAILED when p3 fails and axes reflect it
        p3_fail = o.audit_weight_lean_and_audit059_exact(f_d)
        rej = o.certify_gated_mlp_lean_audit_exact(
            "standalone_fixture_replay", False, a.OPERATOR_LABELS, f_d, p1, p2, p3_fail,
            dict(axes, weft_loom_weight_lean_and_audit059=False),
        )
        self.assertFalse(rej["all_gates_passed"])
        self.assertEqual(rej["verdict"], "REJECTED_GATE_FAILED")

        # Pillar 4 fail-closed guards & provenance binding
        self._assert_err("INVALID_MODE", o.certify_gated_mlp_lean_audit_exact, "bad", False, a.OPERATOR_LABELS, fx, p1, p2, p3, axes)
        self._assert_err("MODE_MISMATCH", o.certify_gated_mlp_lean_audit_exact, "standalone_fixture_replay", True, a.OPERATOR_LABELS, fx, p1, p2, p3, axes)
        self._assert_err("INVALID_AXES", o.certify_gated_mlp_lean_audit_exact, "standalone_fixture_replay", False, a.OPERATOR_LABELS, fx, p1, p2, p3, {"x": True})
        self._assert_err("INVALID_AXES", o.certify_gated_mlp_lean_audit_exact, "standalone_fixture_replay", False, a.OPERATOR_LABELS, fx, p1, p2, p3, [])
        self._assert_err("INVALID_PILLAR", o.certify_gated_mlp_lean_audit_exact, "standalone_fixture_replay", False, a.OPERATOR_LABELS, fx, "bad", p2, p3, axes)
        self._assert_err("INVALID_PILLAR", o.certify_gated_mlp_lean_audit_exact, "standalone_fixture_replay", False, a.OPERATOR_LABELS, fx, p1, "bad", p3, axes)
        self._assert_err("INVALID_PILLAR", o.certify_gated_mlp_lean_audit_exact, "standalone_fixture_replay", False, a.OPERATOR_LABELS, fx, p1, p2, "bad", axes)
        for bad_t in (
            (dict(p1, passed=1), p2, p3),
            (p1, dict(p2, passed=1), p3),
            (p1, p2, dict(p3, passed=1)),
        ):
            self._assert_err("PILLAR_CONTRADICTION", o.certify_gated_mlp_lean_audit_exact, "standalone_fixture_replay", False, a.OPERATOR_LABELS, fx, bad_t[0], bad_t[1], bad_t[2], axes)
        self._assert_err("PROVENANCE_MISMATCH", a.certify_from_summary, "live_json_recount", True, fx)
        fx_lv = copy.deepcopy(fx); fx_lv["diag_source"] = "live_json_file"
        self._assert_err("PROVENANCE_MISMATCH", a.certify_from_summary, "standalone_fixture_replay", False, fx_lv)
        self._assert_err("AXIS_CONTRADICTION", o.certify_gated_mlp_lean_audit_exact, "standalone_fixture_replay", False, a.OPERATOR_LABELS, fx, p1, p2, p3, dict(axes, z2_symmetry_breaking_controls=False))

        # Live JSON parser & CLI tests (synthetic known-answer weight_lean.json)
        self._assert_err("PATH_REQUIRED", a.run_audit, True, None)
        with tempfile.TemporaryDirectory() as td:
            self._assert_err("FILE_MISSING", a.load_live_weight_lean_summary, os.path.join(td, "missing.json"))
            bad_j = os.path.join(td, "bad.json")
            with open(bad_j, "w", encoding="utf-8") as fh:
                json.dump([1, 2], fh)
            self._assert_err("INVALID_JSON", a.load_live_weight_lean_summary, bad_j)
            with open(bad_j, "w", encoding="utf-8") as fh:
                json.dump({}, fh)
            self._assert_err("MISSING_MODEL", a.load_live_weight_lean_summary, bad_j)

            # Build full valid synthetic weight_lean.json matching exact per-layer sums & counts!
            syn = {}
            for mk, r in fx["models"].items():
                n_l, l_gt = r["n_layers"], r["layers_gt_half"]
                u_per = r["units"] // n_l
                half = u_per // 2
                vals = [half + 1] * l_gt + [half] * (n_l - l_gt)
                diff = r["gu_pos"] - sum(vals)
                if diff > 0:
                    q, rem = divmod(diff, l_gt)
                    for i in range(l_gt):
                        vals[i] += q + (1 if i < rem else 0)
                elif diff < 0:
                    q, rem = divmod(-diff, n_l - l_gt)
                    for i in range(n_l - l_gt):
                        vals[l_gt + i] -= q + (1 if i < rem else 0)
                pl = [{"layer": i, "gate_up_inner_positive": vals[i], "units": u_per} for i in range(n_l)]
                syn[mk] = {
                    "gate_up_inner_positive": {"num": r["gu_pos"], "den": r["units"]},
                    "up_rowsum_positive": {"num": r["up_pos"], "den": r["units"]},
                    "down_colsum_positive": {"num": r["dn_pos"], "den": r["units"]},
                    "per_layer": pl,
                }
            for bad_mut, exp_code in (
                (lambda s: s.__setitem__("Qwen__Qwen3.5-0.8B", []), "MISSING_MODEL"),
                (lambda s: s["Qwen__Qwen3.5-0.8B"].__setitem__("per_layer", 123), "INVALID_MODEL_REC"),
                (lambda s: s["Qwen__Qwen3.5-0.8B"].__setitem__("per_layer", []), "INVALID_MODEL_REC"),
                (lambda s: s["Qwen__Qwen3.5-0.8B"]["per_layer"].__setitem__(0, "bad"), "INVALID_MODEL_REC"),
                (lambda s: s["Qwen__Qwen3.5-0.8B"].__setitem__("gate_up_inner_positive", 0), "INVALID_MODEL_REC"),
                (lambda s: s["Qwen__Qwen3.5-0.8B"].__setitem__("up_rowsum_positive", 0), "INVALID_MODEL_REC"),
                (lambda s: s["Qwen__Qwen3.5-0.8B"].__setitem__("down_colsum_positive", 0), "INVALID_MODEL_REC"),
                (lambda s: s["Qwen__Qwen3.5-0.8B"]["gate_up_inner_positive"].__setitem__("den", 0), "RANGE_ERROR"),
                (lambda s: s["Qwen__Qwen3.5-0.8B"]["up_rowsum_positive"].__setitem__("den", 0), "RANGE_ERROR"),
                (lambda s: s["Qwen__Qwen3.5-0.8B"]["down_colsum_positive"].__setitem__("den", 0), "RANGE_ERROR"),
                (lambda s: s["Qwen__Qwen3.5-0.8B"]["up_rowsum_positive"].__setitem__("den", 123456789), "DEN_MISMATCH"),
                (lambda s: s["Qwen__Qwen3.5-0.8B"]["down_colsum_positive"].__setitem__("den", 123456789), "DEN_MISMATCH"),
                (lambda s: s["Qwen__Qwen3.5-0.8B"]["gate_up_inner_positive"].__setitem__("num", -1), "RANGE_ERROR"),
                (lambda s: s["Qwen__Qwen3.5-0.8B"]["up_rowsum_positive"].__setitem__("num", -1), "RANGE_ERROR"),
                (lambda s: s["Qwen__Qwen3.5-0.8B"]["down_colsum_positive"].__setitem__("num", -1), "RANGE_ERROR"),
                (lambda s: s["Qwen__Qwen3.5-0.8B"]["per_layer"][0].__setitem__("units", 0), "RANGE_ERROR"),
                (lambda s: s["Qwen__Qwen3.5-0.8B"]["per_layer"][0].__setitem__("gate_up_inner_positive", -1), "RANGE_ERROR"),
                (lambda s: s["Qwen__Qwen3.5-0.8B"]["per_layer"][0].__setitem__("gate_up_inner_positive", 4000), "RANGE_ERROR"),
                (lambda s: s["Qwen__Qwen3.5-0.8B"]["per_layer"][0].__setitem__("gate_up_inner_positive", 1.5), "INVALID_INT"),
                (lambda s: s["Qwen__Qwen3.5-0.8B"]["per_layer"][0].__setitem__("gate_up_inner_positive", 0), "SUM_MISMATCH"),
                (lambda s: s["Qwen__Qwen3.5-0.8B"]["per_layer"][0].__setitem__("units", 4000), "SUM_MISMATCH"),
            ):
                bad_s = copy.deepcopy(syn)
                bad_mut(bad_s)
                with open(bad_j, "w", encoding="utf-8") as fh:
                    json.dump(bad_s, fh)
                self._assert_err(exp_code, a.load_live_weight_lean_summary, bad_j)

            good_j = os.path.join(td, "weight_lean.json")
            with open(good_j, "w", encoding="utf-8") as fh:
                json.dump(syn, fh)
            c_lv = a.run_audit(live=False, weight_lean_json=good_j)
            self.assertTrue(c_lv["all_gates_passed"])
            self.assertIs(c_lv["live_json_recounted"], True)
            out_p = os.path.join(td, "cert.json")
            buf = io.StringIO()
            with mock.patch.object(sys, "stdout", buf):
                rc = a.main(["--standalone", "--json", "--output", out_p])
            self.assertEqual(rc, 0)
            with open(out_p, "r", encoding="utf-8") as fh:
                self.assertEqual(fh.read(), buf.getvalue())
            with mock.patch.object(a, "run_audit", return_value={"all_gates_passed": False}), mock.patch.object(sys, "stdout", io.StringIO()):
                self.assertEqual(a.main(["--standalone"]), 1)
            with mock.patch.object(a, "run_audit", return_value={"all_gates_passed": True}) as m_ra, mock.patch.object(sys, "stdout", io.StringIO()):
                self.assertEqual(a.main(["--live"]), 0)
                m_ra.assert_called_once_with(live=True, weight_lean_json=None)
            with mock.patch.object(a, "run_audit", return_value={"all_gates_passed": True}) as m_ra, mock.patch.object(sys, "stdout", io.StringIO()):
                self.assertEqual(a.main(["--weight-lean-json", good_j]), 0)
                m_ra.assert_called_once_with(live=True, weight_lean_json=good_j)
            with mock.patch.object(a, "run_audit", return_value={"all_gates_passed": True}) as m_ra, mock.patch.object(sys, "stdout", io.StringIO()):
                self.assertEqual(a.main(["--live", "--weight-lean-json", good_j]), 0)
                m_ra.assert_called_once_with(live=True, weight_lean_json=good_j)


def run_all():
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=0, failfast=True).run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
    print("ALL_AUDIT106_GATED_MLP_LEAN_TESTS_PASSED")


if __name__ == "__main__":
    run_all()
