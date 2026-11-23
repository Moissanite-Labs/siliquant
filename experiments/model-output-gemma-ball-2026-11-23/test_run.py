#!/usr/bin/env python3
"""Unit tests for experiments/model-output-gemma-ball-2026-11-23/run.py."""
from __future__ import annotations

import enum
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
RUN_PATH = HERE / "run.py"

spec = importlib.util.spec_from_file_location("gemma_ball_run", RUN_PATH)
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


class TestGemmaBallRunner(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)
        # 16 x 256 synthetic matrix divisible by 128 and 32
        self.W = (rng.standard_normal((16, 256)) * 0.05).astype(np.float32)
        # Inject loud columns in cols 3, 17, 42
        self.W[0, 3] = 2.5
        self.W[1, 17] = -2.2
        self.W[2, 42] = 1.9

    def test_hadamard8_orthogonal_symmetric(self):
        h = R.H8
        self.assertEqual(h.shape, (8, 8))
        np.testing.assert_allclose(h, h.T, atol=1e-12)
        np.testing.assert_allclose(h @ h.T, np.eye(8), atol=1e-12)

    def test_zero_codeword_preservation(self):
        checks = R.verify_zero_codeword_matrix(self.W)
        for label in ("ZBG128:b560s16", "VE8Z32:16", "HBG128:b560s16", "HVE8Z32:16"):
            self.assertIn(label, checks)
            self.assertTrue(checks[label]["passed"], f"failed zero check: {label}")
            self.assertEqual(checks[label]["synth_zero_ghost_rate"], 0.0)
            self.assertEqual(checks[label]["real_zero_ghost_rate"], 0.0)

    def test_all_prereg_variants_dispatch_and_bits(self):
        def fake_colscale(name, cols, power, which="real"):
            return np.linspace(0.5, 1.5, cols, dtype=np.float64)

        cases = [
            ("ZBG128:b560s16", "blk.0.attn_q.weight", 4.45, 4.55),
            ("hZBG128:b560s16", "blk.0.attn_q.weight", 4.50, 5.60),
            ("hZBG128:b560s16~rand", "blk.0.attn_q.weight", 4.50, 5.60),
            ("ZBG128:b560s16+K10", "blk.0.ffn_down.weight", 4.50, 5.50),
            ("hZBG128:b560s16+K10", "blk.0.ffn_down.weight", 4.50, 6.50),
            ("EBG128:b560s16", "blk.0.attn_q.weight", 4.45, 4.55),
            ("hEBG128:b560s16", "blk.0.attn_q.weight", 4.50, 5.60),
            ("hHBG128:b560s16", "blk.0.attn_q.weight", 4.50, 5.60),
            ("iEBG128:b560s16", "blk.0.attn_q.weight", 4.50, 5.60),
            ("HBG128:b560s16", "blk.0.attn_q.weight", 4.45, 4.55),
            ("VE8Z32:16", "blk.0.attn_q.weight", 4.49, 4.51),
            ("VE832:16", "blk.0.attn_q.weight", 4.49, 4.51),
            ("ZBG128:b496s16", "blk.0.attn_q.weight", 3.95, 4.05),
            ("hZBG128:b496s16", "blk.0.attn_q.weight", 4.00, 5.10),
            ("ZBG128:b624s16", "blk.0.attn_q.weight", 4.95, 5.05),
            ("hZBG128:b624s16", "blk.0.attn_q.weight", 5.00, 6.10),
            ("ZB128@R:16", "blk.0.attn_q.weight", 4.00, 4.10),
            ("ZB128@R:32", "blk.0.attn_q.weight", 5.00, 5.10),
        ]
        for variant, name, lo_bits, hi_bits in cases:
            Wh, info = R.round_matrix_variant(variant, name, self.W, colscale_fn=fake_colscale)
            self.assertEqual(Wh.shape, self.W.shape)
            self.assertEqual(Wh.dtype, np.float16)
            self.assertGreaterEqual(info["bits"], lo_bits, f"{variant} bits too low")
            self.assertLessEqual(info["bits"], hi_bits, f"{variant} bits too high")
            self.assertGreaterEqual(info["rel_err"], 0.0)

        # Verify +K10 preserves the 10 loud columns of ffn_down bit-for-bit as float16
        cols = R.loud_columns(self.W, k=10)
        self.assertIn(3, cols)
        self.assertIn(17, cols)
        self.assertIn(42, cols)
        Wh_k10, info_k10 = R.round_matrix_variant("ZBG128:b560s16+K10", "blk.0.ffn_down.weight", self.W)
        self.assertEqual(info_k10["kept_columns"], 10)
        np.testing.assert_array_equal(Wh_k10[:, cols], self.W[:, cols].astype(np.float16))

    def test_unread_gemma_e4b_accounting(self):
        infos = {
            "blk.0.attn_k.weight": {"rel_err": 0.01, "bits": 4.5, "overload": 0.0, "energy": 1.0, "size": 1000},
            "blk.24.attn_k.weight": {"rel_err": 0.02, "bits": 4.5, "overload": 0.0, "energy": 1.0, "size": 1000},
            "blk.24.attn_v.weight": {"rel_err": 0.02, "bits": 4.5, "overload": 0.0, "energy": 1.0, "size": 1000},
            "blk.24.ffn_down.weight": {"rel_err": 0.01, "bits": 4.5, "overload": 0.0, "energy": 1.0, "size": 2000},
        }
        agg = R.stats_of(infos, "gemma-4-E4B-it", n_layers=42)
        self.assertEqual(agg["matrices"], 4)
        self.assertEqual(agg["matrices_active"], 2)
        self.assertEqual(agg["weights"], 5000)
        self.assertEqual(agg["weights_active"], 3000)

        # Non-E4B model keeps all matrices active
        agg3 = R.stats_of(infos, "gemma-3-4b-it", n_layers=34)
        self.assertEqual(agg3["matrices_active"], 4)

    def test_quantize_tensor_type_flags(self):
        class FakeType(enum.IntEnum):
            F32 = 0
            F16 = 1
            Q8_0 = 8
            BF16 = 30

        def fake_tensor(name, ttype):
            return types.SimpleNamespace(name=name, tensor_type=ttype)

        # String-only call defaults to f16
        base_flags = R.quantize_tensor_type_flags(["token_embd.weight", "blk.0.attn_q.weight"])
        self.assertEqual(base_flags, ["--tensor-type", r"^token_embd\.weight$=f16"])

        # Enum tensor_type inspection preserves BF16, F16, and F32
        bf16_flags = R.quantize_tensor_type_flags([
            fake_tensor("token_embd.weight", FakeType.BF16),
            fake_tensor("blk.0.attn_q.weight", FakeType.BF16),
        ])
        self.assertEqual(bf16_flags, ["--tensor-type", r"^token_embd\.weight$=bf16"])

        f16_flags = R.quantize_tensor_type_flags([
            fake_tensor("token_embd.weight", FakeType.F16),
            fake_tensor("blk.0.attn_q.weight", FakeType.F16),
        ])
        self.assertEqual(f16_flags, ["--tensor-type", r"^token_embd\.weight$=f16"])

        f32_flags = R.quantize_tensor_type_flags([
            fake_tensor("token_embd.weight", FakeType.F32),
            fake_tensor("blk.0.attn_q.weight", FakeType.F32),
        ])
        self.assertEqual(f32_flags, ["--tensor-type", r"^token_embd\.weight$=f32"])

        # Raw GGML integer type codes also map cleanly
        self.assertEqual(
            R.quantize_tensor_type_flags([fake_tensor("token_embd.weight", 30)]),
            ["--tensor-type", r"^token_embd\.weight$=bf16"],
        )
        self.assertEqual(
            R.quantize_tensor_type_flags([fake_tensor("token_embd.weight", 0)]),
            ["--tensor-type", r"^token_embd\.weight$=f32"],
        )

        # E4B BF16 checkpoint keeps bf16 on token_embd plus f32 on E4B proj/gate tensors
        e4b_tensors = [
            fake_tensor("token_embd.weight", FakeType.BF16),
            fake_tensor("per_layer_model_proj.weight", FakeType.F32),
            fake_tensor("blk.0.inp_gate.weight", FakeType.F32),
            fake_tensor("blk.0.alt.proj.weight", FakeType.F32),
        ]
        e4b_flags = R.quantize_tensor_type_flags(e4b_tensors)
        self.assertEqual(e4b_flags[:2], ["--tensor-type", r"^token_embd\.weight$=bf16"])
        self.assertIn("per_layer_model_proj=f32", e4b_flags)
        self.assertIn("inp_gate=f32", e4b_flags)
        self.assertIn(r"\.proj\.=f32", e4b_flags)

        # ensure_baseline_gguf forwards the inspected BF16 flag in its llama-quantize command
        cmd = R.ensure_baseline_gguf(
            "model-bf16.gguf",
            "out-Q4_K_M.gguf",
            "Q4_K_M",
            "llama-quantize",
            e4b_tensors,
            dry_run=True,
        )
        self.assertIn(r"^token_embd\.weight$=bf16", cmd)

        # Unsupported token_embd.weight dtype fails closed
        with self.assertRaises(ValueError):
            R.quantize_tensor_type_flags([fake_tensor("token_embd.weight", FakeType.Q8_0)])

    def test_arm_needs_real_imat_and_provenance(self):
        for arm in (
            "file:Q4_K_M-imat",
            "hZBG128:b560s16",
            "hEBG128:b560s16",
            "hHBG128:b560s16",
            "iZBG128:b560s16",
            "iEBG128:b560s16",
            "iHBG128:b560s16",
            "hEBG128:b496s16",
            "hHBG128:b624s16",
            "hZBG128:b560s16+K10",
        ):
            self.assertTrue(R.arm_needs_real_imat(arm), f"expected real imatrix for {arm}")

        for arm in (
            "file:Q4_0",
            "file:Q4_K_M",
            "ZBG128:b560s16",
            "EBG128:b560s16",
            "HBG128:b560s16",
            "ZBG128:b560s16+K10",
            "hZBG128:b560s16~rand",
            "hEBG128:b560s16~rand",
            "hHBG128:b560s16~rand",
            "iEBG128:b560s16~rand",
            "VE8Z32:16",
            "ZB128@R:16",
        ):
            self.assertFalse(R.arm_needs_real_imat(arm), f"did not expect real imatrix for {arm}")

        # End-to-end sweep check (metrics-only, mocked GGUF reader): hEBG128 and hHBG128
        # trigger real imatrix and record imat_real_sha; ~rand records imat_rand_sha.
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, "fake-bf16.gguf")
            Path(model_path).write_bytes(b"fake-gguf")
            cache_dir = os.path.join(td, "cache")
            os.makedirs(cache_dir, exist_ok=True)
            real_imat = os.path.join(cache_dir, "fake-model.wikitext.imatrix.gguf")
            rand_txt = os.path.join(cache_dir, "fake-model.rand.txt")
            rand_imat = os.path.join(cache_dir, "fake-model.rand.imatrix.gguf")
            Path(real_imat).write_bytes(b"real-imatrix-bytes")
            Path(rand_txt).write_text("rand", encoding="utf-8")
            Path(rand_imat).write_bytes(b"rand-imatrix-bytes")

            fake_tensor = types.SimpleNamespace(
                name="blk.0.attn_q.weight",
                shape=(256, 16),
                data=self.W,
                tensor_type=types.SimpleNamespace(name="BF16"),
            )
            fake_embd = types.SimpleNamespace(
                name="token_embd.weight",
                shape=(256, 16),
                data=self.W,
                tensor_type=types.SimpleNamespace(name="BF16"),
            )
            fake_reader = types.SimpleNamespace(tensors=[fake_embd, fake_tensor])
            fake_ggufedit = types.SimpleNamespace(
                open_model=lambda _p: fake_reader,
                tensor_float=lambda t: t.data,
            )

            seen_which = []

            def fake_col_weights(name, cols, power, which="real"):
                seen_which.append(which)
                return np.ones(cols, dtype=np.float64)

            args = R.parse_args([
                "--model-id", "fake-model",
                "--model-path", model_path,
                "--out-dir", td,
                "--texts", "wikitext",
                "--arms", "hEBG128:b560s16,hHBG128:b560s16,hEBG128:b560s16~rand",
                "--workers", "1",
                "--metrics-only",
            ])
            with mock.patch.object(R, "_load_ggufedit", return_value=fake_ggufedit), \
                 mock.patch.object(R, "_column_weights", side_effect=fake_col_weights), \
                 mock.patch.dict(sys.modules, {"evalkl": types.SimpleNamespace()}):
                R.run_sweep(args)

            rows = [
                json.loads(line)
                for line in Path(os.path.join(td, "results.jsonl")).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            by_var = {r["variant"]: r for r in rows}
            expected_real_sha = R.sha256_of(real_imat)
            expected_rand_sha = R.sha256_of(rand_imat)
            self.assertEqual(by_var["hEBG128:b560s16"]["imatrix_sha256"], expected_real_sha)
            self.assertEqual(by_var["hHBG128:b560s16"]["imatrix_sha256"], expected_real_sha)
            self.assertEqual(by_var["hEBG128:b560s16~rand"]["imatrix_sha256"], expected_rand_sha)
            self.assertEqual(seen_which, ["real", "real", "rand"])

            # Verify missing real imatrix without --wikitext-train fails closed for hEBG128
            os.remove(real_imat)
            args_missing = R.parse_args([
                "--model-id", "fake-model",
                "--model-path", model_path,
                "--out-dir", td,
                "--texts", "wikitext",
                "--arms", "hEBG128:b560s16",
                "--workers", "1",
                "--metrics-only",
            ])
            with mock.patch.object(R, "_load_ggufedit", return_value=fake_ggufedit), \
                 mock.patch.dict(sys.modules, {"evalkl": types.SimpleNamespace()}):
                with self.assertRaises(SystemExit):
                    R.run_sweep(args_missing)

    def test_fail_closed_validation(self):
        with self.assertRaises(ValueError):
            R.round_matrix_variant("UNKNOWN_ARM", "blk.0.attn_q.weight", self.W)
        with self.assertRaises(ValueError):
            R.voronoi_e8_round(self.W, 16, 32, centred="false")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            R.group_ball_round(np.zeros((4, 100), dtype=np.float32), "Z8", 128, 560, 16)

    def test_cli_help_and_dry_run_and_render_results(self):
        p_help = subprocess.run([sys.executable, "-B", str(RUN_PATH), "--help"], capture_output=True, text=True, check=True)
        self.assertIn("--pilot-only", p_help.stdout)
        self.assertIn("--bridge-eval", p_help.stdout)

        with tempfile.TemporaryDirectory() as td:
            p_dry = subprocess.run(
                [
                    sys.executable, "-B", str(RUN_PATH),
                    "--model-id", "gemma-4-E4B-it",
                    "--model-path", os.path.join(td, "gemma-4-E4B-it-f16.gguf"),
                    "--out-dir", td,
                    "--texts", "wikitext,python",
                    "--ngl", "999",
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            plan = json.loads(p_dry.stdout)
            self.assertEqual(plan["mode"], "sweep")
            self.assertEqual(plan["ngl"], 999)
            self.assertEqual(plan["arms"], list(R.DEFAULT_ARMS))
            self.assertEqual(len(plan["kld_cmds"]), 2)
            self.assertIn("-ngl", plan["kld_cmds"][0])

            p_bridge = subprocess.run(
                [
                    sys.executable, "-B", str(RUN_PATH),
                    "--model-id", "gemma-4-E4B-it",
                    "--model-path", os.path.join(td, "gemma-4-E4B-it-Q8_0.gguf"),
                    "--out-dir", td,
                    "--bridge-eval", os.path.join(td, "eval.txt"),
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            bplan = json.loads(p_bridge.stdout)
            self.assertEqual(bplan["mode"], "bridge")
            self.assertEqual(bplan["arms"], list(R.BRIDGE_ARMS))

            # Test render_results_out formatting
            jsonl = os.path.join(td, "results.jsonl")
            out = os.path.join(td, "results.out")
            with open(jsonl, "w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "model_id": "gemma-4-E4B-it",
                    "text": "wikitext",
                    "variant": "ZBG128:b560s16",
                    "bits_per_weight": 4.5,
                    "bits_per_weight_active": 4.5,
                    "weight_err_db_mean": -23.5,
                    "kl_mean": 0.0412,
                    "kl_mean_se": 0.0005,
                    "same_top": 0.9123,
                    "build_seconds": 12.3,
                    "score_seconds": 45.6,
                }) + "\n")
            R.render_results_out(jsonl, out)
            content = Path(out).read_text(encoding="utf-8")
            self.assertIn("ZBG128:b560s16", content)
            self.assertIn("0.041200", content)


if __name__ == "__main__":
    unittest.main()
