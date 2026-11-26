#!/usr/bin/env python3
"""Unit tests for experiments/model-output-gemma-ball-2026-11-23/run.py."""
from __future__ import annotations

import enum
import importlib.util
import json
import os
import re
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
            Q4_K = 12
            BF16 = 30
            UNKNOWN = 999

        def fake_tensor(name, ttype, shape=None, data=None):
            ns = types.SimpleNamespace(name=name, tensor_type=ttype)
            if shape is not None:
                ns.shape = shape
            if data is not None:
                ns.data = data
                ns.n_bytes = data.nbytes
            return ns

        # String-only call defaults to f16 for token embeddings
        base_flags = R.quantize_tensor_type_flags(["token_embd.weight", "blk.0.attn_q.weight"])
        self.assertEqual(base_flags, ["--tensor-type", r"^token_embd\.weight$=f16"])

        # Enum tensor_type inspection preserves BF16, F16, F32, and Q8_0
        for ttype, flag_val in (
            (FakeType.BF16, "bf16"),
            (FakeType.F16, "f16"),
            (FakeType.F32, "f32"),
            (FakeType.Q8_0, "q8_0"),
        ):
            flags = R.quantize_tensor_type_flags([
                fake_tensor("token_embd.weight", ttype),
                fake_tensor("per_layer_token_embd.weight", ttype),
                fake_tensor("blk.0.attn_q.weight", ttype),
            ])
            self.assertEqual(
                flags,
                [
                    "--tensor-type", rf"^token_embd\.weight$={flag_val}",
                    "--tensor-type", rf"^per_layer_token_embd\.weight$={flag_val}",
                ],
            )

        # Raw GGML integer type codes also map cleanly
        self.assertEqual(
            R.quantize_tensor_type_flags([fake_tensor("token_embd.weight", 30)]),
            ["--tensor-type", r"^token_embd\.weight$=bf16"],
        )
        self.assertEqual(
            R.quantize_tensor_type_flags([fake_tensor("per_layer_token_embd.weight", 8)]),
            ["--tensor-type", r"^per_layer_token_embd\.weight$=q8_0"],
        )
        self.assertEqual(
            R.quantize_tensor_type_flags([fake_tensor("token_embd.weight", 0)]),
            ["--tensor-type", r"^token_embd\.weight$=f32"],
        )

        # E4B BF16 checkpoint preserves native source dtypes (never hardcoding =f32)
        # and collapses uniform multi-layer non-block tensors into ^blk\.\d+\.<suffix>$
        e4b_bf16_tensors = [
            fake_tensor("token_embd.weight", FakeType.BF16, shape=(256, 16)),
            fake_tensor("per_layer_token_embd.weight", FakeType.BF16, shape=(256, 16)),
            fake_tensor("per_layer_model_proj.weight", FakeType.BF16, shape=(256, 16)),
            fake_tensor("blk.0.inp_gate.weight", FakeType.BF16, shape=(256, 16)),
            fake_tensor("blk.1.inp_gate.weight", FakeType.BF16, shape=(256, 16)),
            fake_tensor("blk.0.proj.weight", FakeType.BF16, shape=(256, 16)),
            fake_tensor("blk.1.proj.weight", FakeType.BF16, shape=(256, 16)),
            fake_tensor("blk.0.alt.proj.weight", FakeType.F32, shape=(256, 16)),
            fake_tensor("blk.0.attn_q.weight", FakeType.BF16, shape=(256, 16)),
            fake_tensor("blk.1.attn_q.weight", FakeType.BF16, shape=(256, 16)),
        ]
        e4b_flags = R.quantize_tensor_type_flags(e4b_bf16_tensors)
        self.assertEqual(
            e4b_flags,
            [
                "--tensor-type", r"^token_embd\.weight$=bf16",
                "--tensor-type", r"^per_layer_token_embd\.weight$=bf16",
                "--tensor-type", r"^per_layer_model_proj\.weight$=bf16",
                "--tensor-type", r"^blk\.\d+\.inp_gate\.weight$=bf16",
                "--tensor-type", r"^blk\.\d+\.proj\.weight$=bf16",
                "--tensor-type", r"^blk\.0\.alt\.proj\.weight$=f32",
            ],
        )

        # Mixed dtypes across layers stay separate exact anchored flags instead of collapsing
        mixed_flags = R.quantize_tensor_type_flags([
            fake_tensor("blk.0.inp_gate.weight", FakeType.BF16),
            fake_tensor("blk.1.inp_gate.weight", FakeType.F32),
        ])
        self.assertEqual(
            mixed_flags,
            [
                "--tensor-type", r"^blk\.0\.inp_gate\.weight$=bf16",
                "--tensor-type", r"^blk\.1\.inp_gate\.weight$=f32",
            ],
        )

        # ensure_baseline_gguf forwards all preserved non-block flags
        cmd = R.ensure_baseline_gguf(
            "model-bf16.gguf",
            "out-Q4_K_M.gguf",
            "Q4_K_M",
            "llama-quantize",
            e4b_bf16_tensors,
            dry_run=True,
        )
        self.assertIn(r"^token_embd\.weight$=bf16", cmd)
        self.assertIn(r"^per_layer_token_embd\.weight$=bf16", cmd)
        self.assertIn(r"^per_layer_model_proj\.weight$=bf16", cmd)
        self.assertIn(r"^blk\.\d+\.inp_gate\.weight$=bf16", cmd)
        self.assertIn(r"^blk\.\d+\.proj\.weight$=bf16", cmd)

        # Unsupported or missing tensor_type fails closed
        with self.assertRaises(ValueError):
            R.quantize_tensor_type_flags([fake_tensor("token_embd.weight", FakeType.UNKNOWN)])
        with self.assertRaises(ValueError):
            R.quantize_tensor_type_flags([fake_tensor("per_layer_token_embd.weight", None)])
        with self.assertRaises(ValueError):
            R.quantize_tensor_type_flags([fake_tensor("per_layer_model_proj.weight", "BOGUS")])

        # End-to-end compare_file check across BF16, F16, and F32 models:
        # simulate llama-quantize applying --tensor-type regex overrides (first match wins,
        # unmatched 2D tensors quantized to Q4_K) and verify other_tensors_that_differ == [].
        flag_to_enum = {
            "bf16": FakeType.BF16,
            "f16": FakeType.F16,
            "f32": FakeType.F32,
            "q8_0": FakeType.Q8_0,
            "q4_k": FakeType.Q4_K,
        }
        norm_vec = np.ones((16,), dtype=np.float32)
        for model_dtype in (FakeType.BF16, FakeType.F16, FakeType.F32):
            ref_tensors = [
                fake_tensor("token_embd.weight", model_dtype, shape=(256, 16), data=self.W),
                fake_tensor("per_layer_token_embd.weight", model_dtype, shape=(256, 16), data=self.W),
                fake_tensor("per_layer_model_proj.weight", model_dtype, shape=(256, 16), data=self.W),
                fake_tensor("output.weight", model_dtype, shape=(256, 16), data=self.W),
                fake_tensor("output_norm.weight", FakeType.F32, shape=(16,), data=norm_vec),
                fake_tensor("blk.0.attn_norm.weight", FakeType.F32, shape=(16,), data=norm_vec),
                fake_tensor("blk.1.attn_norm.weight", FakeType.F32, shape=(16,), data=norm_vec),
                fake_tensor("blk.0.inp_gate.weight", model_dtype, shape=(256, 16), data=self.W),
                fake_tensor("blk.1.inp_gate.weight", model_dtype, shape=(256, 16), data=self.W),
                fake_tensor("blk.0.proj.weight", model_dtype, shape=(256, 16), data=self.W),
                fake_tensor("blk.1.proj.weight", model_dtype, shape=(256, 16), data=self.W),
                fake_tensor("blk.0.attn_q.weight", model_dtype, shape=(256, 16), data=self.W),
                fake_tensor("blk.1.ffn_down.weight", model_dtype, shape=(256, 16), data=self.W),
            ]
            flags = R.quantize_tensor_type_flags(ref_tensors)
            overrides = []
            for idx in range(0, len(flags), 2):
                self.assertEqual(flags[idx], "--tensor-type")
                pat, tname = flags[idx + 1].rsplit("=", 1)
                overrides.append((re.compile(pat), flag_to_enum[tname]))

            quant_tensors = []
            for t in ref_tensors:
                override_type = None
                for pat, target_type in overrides:
                    if pat.search(t.name):
                        override_type = target_type
                        break
                if override_type is not None:
                    out_type = override_type
                    out_data = t.data.copy() if out_type == t.tensor_type else (t.data * 0.99)
                else:
                    out_type = FakeType.Q4_K
                    out_data = (t.data * 0.98).astype(np.float32)
                quant_tensors.append(fake_tensor(t.name, out_type, shape=t.shape, data=out_data))

            # Verify block targets were quantized while every non-block tensor kept its source dtype
            quant_by_name = {t.name: t for t in quant_tensors}
            self.assertEqual(quant_by_name["blk.0.attn_q.weight"].tensor_type, FakeType.Q4_K)
            self.assertEqual(quant_by_name["blk.1.ffn_down.weight"].tensor_type, FakeType.Q4_K)
            for t in ref_tensors:
                if not (R.BLOCK.match(t.name) and len(t.shape) == 2):
                    self.assertEqual(quant_by_name[t.name].tensor_type, t.tensor_type, f"{model_dtype.name}: {t.name}")

            ref_reader = types.SimpleNamespace(tensors=ref_tensors)
            quant_reader = types.SimpleNamespace(tensors=quant_tensors)
            fake_ggufedit = types.SimpleNamespace(
                open_model=lambda _p, qr=quant_reader: qr,
                tensor_float=lambda t: t.data,
            )
            with mock.patch.object(R, "_load_ggufedit", return_value=fake_ggufedit):
                row = R.compare_file(ref_reader, "quant.gguf", "gemma-4-E4B-it", n_layers=2)
            self.assertEqual(row["other_tensors_that_differ"], [], f"differing non-block tensors for {model_dtype.name}")
            self.assertEqual(row["matrices"], 2)

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

    def test_missing_imatrix_tensor_falls_back_to_unweighted(self):
        # Unit tests for _column_weights with missing keys, zero counts, zero sum2, and length mismatch
        cols = self.W.shape[1]
        in_sum2_present = np.linspace(1.0, 4.0, cols, dtype=np.float64)
        fake_im_tensors = [
            types.SimpleNamespace(name="blk.0.attn_q.weight.in_sum2", data=in_sum2_present),
            types.SimpleNamespace(name="blk.0.attn_q.weight.counts", data=np.array([10.0], dtype=np.float64)),
            types.SimpleNamespace(name="blk.0.ffn_down.weight.in_sum2", data=in_sum2_present),
            types.SimpleNamespace(name="blk.0.ffn_down.weight.counts", data=np.array([10.0], dtype=np.float64)),
            types.SimpleNamespace(name="blk.1.attn_q.weight.in_sum2", data=in_sum2_present),
            types.SimpleNamespace(name="blk.2.attn_q.weight.in_sum2", data=in_sum2_present),
            types.SimpleNamespace(name="blk.2.attn_q.weight.counts", data=np.array([0.0], dtype=np.float64)),
            types.SimpleNamespace(name="blk.3.attn_q.weight.in_sum2", data=np.zeros(cols, dtype=np.float64)),
            types.SimpleNamespace(name="blk.3.attn_q.weight.counts", data=np.array([5.0], dtype=np.float64)),
            types.SimpleNamespace(name="blk.4.attn_q.weight.in_sum2", data=np.ones(cols // 2, dtype=np.float64)),
            types.SimpleNamespace(name="blk.4.attn_q.weight.counts", data=np.array([5.0], dtype=np.float64)),
        ]
        fake_gguf_mod = types.SimpleNamespace(
            GGUFReader=lambda _p: types.SimpleNamespace(tensors=fake_im_tensors),
        )
        fake_ggufedit = types.SimpleNamespace()

        with mock.patch.object(R, "_load_ggufedit", return_value=fake_ggufedit), \
             mock.patch.dict(sys.modules, {"gguf": fake_gguf_mod}):
            R._worker_init("fake.gguf", "real.imatrix.gguf", None)
            present = R._column_weights("blk.0.attn_q.weight", cols, 0.25, "real")
            self.assertIsNotNone(present)
            self.assertEqual(present.shape, (cols,))
            # Missing .in_sum2 and .counts (e.g. KV-shared blk.24.attn_k.weight)
            self.assertIsNone(R._column_weights("blk.24.attn_k.weight", cols, 0.25, "real"))
            # Missing .counts only
            self.assertIsNone(R._column_weights("blk.1.attn_q.weight", cols, 0.25, "real"))
            # Zero count
            self.assertIsNone(R._column_weights("blk.2.attn_q.weight", cols, 0.25, "real"))
            # Zero mean importance
            self.assertIsNone(R._column_weights("blk.3.attn_q.weight", cols, 0.25, "real"))
            # Length mismatch still fails closed
            with self.assertRaises(ValueError):
                R._column_weights("blk.4.attn_q.weight", cols, 0.25, "real")
            # Missing imatrix path for rand still fails closed
            with self.assertRaises(ValueError):
                R._column_weights("blk.0.attn_q.weight", cols, 0.25, "rand")

        # round_matrix_variant matches unweighted output bit-for-bit on fallback
        none_colscale = lambda _n, _c, _p, _w="real": None
        ones_colscale = lambda _n, c, _p, _w="real": np.ones(c, dtype=np.float64)

        Wh_unw, info_unw = R.round_matrix_variant("ZBG128:b560s16", "blk.24.attn_k.weight", self.W)
        Wh_fb, info_fb = R.round_matrix_variant("hZBG128:b560s16", "blk.24.attn_k.weight", self.W, colscale_fn=none_colscale)
        np.testing.assert_array_equal(Wh_fb, Wh_unw)
        self.assertAlmostEqual(info_fb["bits"], info_unw["bits"], places=12)
        self.assertTrue(info_fb["imatrix_fallback"])
        self.assertFalse(info_unw["imatrix_fallback"])

        _, info_weighted = R.round_matrix_variant("hZBG128:b560s16", "blk.0.attn_q.weight", self.W, colscale_fn=ones_colscale)
        self.assertFalse(info_weighted["imatrix_fallback"])

        # Loud-column weighted arm falls back cleanly for both non-ffn_down and ffn_down tensors
        for tname in ("blk.24.attn_k.weight", "blk.24.ffn_down.weight"):
            Wh_lk_unw, info_lk_unw = R.round_matrix_variant("ZBG128:b560s16+K10", tname, self.W)
            Wh_lk_fb, info_lk_fb = R.round_matrix_variant("hZBG128:b560s16+K10", tname, self.W, colscale_fn=none_colscale)
            np.testing.assert_array_equal(Wh_lk_fb, Wh_lk_unw)
            self.assertAlmostEqual(info_lk_fb["bits"], info_lk_unw["bits"], places=12)
            self.assertTrue(info_lk_fb["imatrix_fallback"])
            self.assertFalse(info_lk_unw["imatrix_fallback"])

        # End-to-end sweep provenance: missing tensors recorded in row metadata
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, "fake-e4b.gguf")
            Path(model_path).write_bytes(b"fake-gguf")
            cache_dir = os.path.join(td, "cache")
            os.makedirs(cache_dir, exist_ok=True)
            real_imat = os.path.join(cache_dir, "gemma-4-E4B-it.wikitext.imatrix.gguf")
            rand_txt = os.path.join(cache_dir, "gemma-4-E4B-it.rand.txt")
            rand_imat = os.path.join(cache_dir, "gemma-4-E4B-it.rand.imatrix.gguf")
            Path(real_imat).write_bytes(b"real-imatrix-bytes")
            Path(rand_txt).write_text("rand", encoding="utf-8")
            Path(rand_imat).write_bytes(b"rand-imatrix-bytes")

            def mk_tensor(name):
                return types.SimpleNamespace(
                    name=name,
                    shape=(cols, self.W.shape[0]),
                    data=self.W,
                    tensor_type=types.SimpleNamespace(name="BF16"),
                )

            fake_reader = types.SimpleNamespace(
                tensors=[
                    mk_tensor("token_embd.weight"),
                    mk_tensor("blk.0.attn_q.weight"),
                    mk_tensor("blk.0.ffn_down.weight"),
                    mk_tensor("blk.24.attn_k.weight"),
                    mk_tensor("blk.24.attn_v.weight"),
                ]
            )
            fake_ggufedit_sweep = types.SimpleNamespace(
                open_model=lambda _p: fake_reader,
                tensor_float=lambda t: t.data,
            )
            args = R.parse_args([
                "--model-id", "gemma-4-E4B-it",
                "--model-path", model_path,
                "--out-dir", td,
                "--texts", "wikitext",
                "--arms", "ZBG128:b560s16,hZBG128:b560s16,hZBG128:b560s16~rand,hZBG128:b560s16+K10",
                "--workers", "1",
                "--metrics-only",
            ])
            with mock.patch.object(R, "_load_ggufedit", return_value=fake_ggufedit_sweep), \
                 mock.patch.dict(sys.modules, {"gguf": fake_gguf_mod, "evalkl": types.SimpleNamespace()}):
                R.run_sweep(args)

            rows = [
                json.loads(line)
                for line in Path(os.path.join(td, "results.jsonl")).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            by_var = {r["variant"]: r for r in rows}
            expected_missing = ["blk.24.attn_k.weight", "blk.24.attn_v.weight"]
            for arm in ("hZBG128:b560s16", "hZBG128:b560s16~rand", "hZBG128:b560s16+K10"):
                self.assertEqual(by_var[arm]["imatrix_fallback_count"], 2, arm)
                self.assertEqual(by_var[arm]["imatrix_fallback_tensors"], expected_missing, arm)
            self.assertEqual(by_var["ZBG128:b560s16"]["imatrix_fallback_count"], 0)
            self.assertEqual(by_var["ZBG128:b560s16"]["imatrix_fallback_tensors"], [])

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


class TestAddendumArms(unittest.TestCase):
    """2026-11-24 addendum: A_n Hamming-lattice ball arms and stock files decoded to 16-bit weights."""

    def setUp(self):
        rng = np.random.default_rng(321)
        self.W = (rng.standard_normal((8, 256)) * 0.05).astype(np.float32)

    def test_arm_regexes_and_imatrix_routing(self):
        for arm in ("A16BG128:b560s16", "hA16BG128:b560s16", "iA16BG128:b560s16~rand", "A32BG128:b368s16"):
            self.assertTrue(R.HAMMING_BALL_RE.match(arm), arm)
            self.assertFalse(R.WEIGHTED_BALL_RE.match(arm), arm)
            self.assertFalse(R.GROUP_BALL_RE.match(arm), arm)
        self.assertFalse(R.arm_needs_real_imat("A16BG128:b560s16"))
        self.assertTrue(R.arm_needs_real_imat("hA16BG128:b560s16"))
        self.assertFalse(R.arm_needs_real_imat("iA16BG128:b560s16~rand"))
        for arm in ("file:Q4_0@16", "file:Q4_K_M@16", "file:Q4_K_M-imat@16"):
            self.assertTrue(R.STOCK16_RE.match(arm), arm)
            self.assertFalse(R.STOCK16_RE.match(arm.replace("@16", "")), arm)
        self.assertFalse(R.arm_needs_real_imat("file:Q4_K_M@16"))
        self.assertTrue(R.arm_needs_real_imat("file:Q4_K_M-imat@16"))

    def test_hamming_ball_arms_dispatch_bits_and_lattice(self):
        import math

        def fake_colscale(name, cols, power, which="real"):
            return np.linspace(0.5, 1.5, cols, dtype=np.float64)

        ham = R.hamming_module()
        for variant, lo, hi in (("A16BG128:b560s16", 4.45, 4.55), ("hA16BG128:b560s16", 4.45, 6.60),
                                ("A16BG128:b368s16", 2.95, 3.05)):
            Wh, info = R.round_matrix_variant(variant, "blk.0.attn_q.weight", self.W, colscale_fn=fake_colscale)
            self.assertEqual(Wh.shape, self.W.shape)
            self.assertEqual(Wh.dtype, np.float16)
            self.assertGreaterEqual(info["bits"], lo, variant)
            self.assertLessEqual(info["bits"], hi, variant)
            # every group of the rounded matrix lies in the ball of the lattice, at its own scale
            W64 = np.asarray(self.W, dtype=np.float64)
            n, bbits = 128, int(R.HAMMING_BALL_RE.match(variant).group(4))
            bound, points = ham.ball_bound(4, n, bbits)
            X = W64.reshape(-1, n)
            t = R._hamming_block_scales(X, 4, bound, math.sqrt(ham.covering_sq(4, n)))
            Y = ham.closest(X / t[:, None], 4)
            self.assertTrue(bool(np.all((Y ** 2).sum(axis=1) <= bound + 1e-9)), variant)
            self.assertTrue(bool(np.array_equal(ham.closest(Y, 4), Y)), variant)

    def test_hamming_decoder_c_numpy_parity(self):
        ham = R.hamming_module()
        rng = np.random.default_rng(7)
        X = rng.standard_normal((200, 128)) * 2.0
        self.assertTrue(np.array_equal(ham.closest(X, 4, use_c=True), ham.closest(X, 4, use_c=False)))

    def test_hamming_scales_match_addendum_ball_round_bit_for_bit(self):
        """The port's scale bisection equals the 2026-11-24 record's ball_round on the same input."""
        import math

        ham = R.hamming_module()
        bd = R.ball_module()
        W64 = np.asarray(self.W, dtype=np.float64)
        X = W64.reshape(-1, 128)
        bound, points = ham.ball_bound(4, 128, 560)
        margin = math.sqrt(ham.covering_sq(4, 128))
        # the record's bisection (experiments/hamming-lattices-real-weights-2026-11-24/run.py)
        g = np.maximum(np.sqrt((X ** 2).sum(axis=1)), 1e-30)
        r = math.sqrt(bound)
        lo = g / (r + margin) * 0.99
        hi = g / max(r - margin, 0.05 * r) * 1.01
        for _ in range(bd.BISECT_STEPS):
            mid = np.sqrt(lo * hi)
            ok = (ham.closest(X / mid[:, None], 4) ** 2).sum(axis=1) <= bound + 1e-9
            hi = np.where(ok, mid, hi)
            lo = np.where(ok, lo, mid)
        t = R._hamming_block_scales(X, 4, bound, margin)
        np.testing.assert_array_equal(t, hi)
        # bpw of A16BG128:b560s16 must equal the stored counted bits_per_weight of the committed
        # receipt rows (out/ccf045d/gemma-3-12b-it-hebg/results.jsonl, variant A16BG128:b560s16:
        # bits_per_weight = 4.49973479628501; PREREG-ADDENDUM-2026-11-24.md line 61: B = 560 -> 4.4997 bits)
        self.assertAlmostEqual((math.log2(points) + 16) / 128, 4.49973479628501, places=12)



class TestStreamArms(unittest.TestCase):
    """--stream-arms: sidecar-based arm writing must be byte-identical to write_copy."""

    def _tiny_gguf(self, path):
        import gguf
        rng = np.random.default_rng(7)
        writer = gguf.GGUFWriter(path, arch="gemma3")
        writer.add_tensor_info("token_embd.weight", (8, 4), np.dtype(np.float32), 8 * 4 * 4,
                               gguf.GGMLQuantizationType.F32)
        writer.add_tensor_info("blk.0.attn_q.weight", (4, 128), np.dtype(np.float32), 4 * 128 * 4,
                               gguf.GGMLQuantizationType.F32)
        writer.add_tensor_info("blk.0.ffn_down.weight", (4, 128), np.dtype(np.float32), 4 * 128 * 4,
                               gguf.GGMLQuantizationType.F32)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        writer.write_tensor_data(rng.standard_normal((8, 4)).astype(np.float32))
        writer.write_tensor_data(rng.standard_normal((4, 128)).astype(np.float32))
        writer.write_tensor_data(rng.standard_normal((4, 128)).astype(np.float32))
        writer.close()

    def test_streaming_writer_byte_parity(self):
        ggufedit = R._load_ggufedit()
        with tempfile.TemporaryDirectory() as td:
            model = os.path.join(td, "tiny.gguf")
            self._tiny_gguf(model)
            rd = ggufedit.open_model(model)
            names = [t.name for t in R._targets(rd)]
            self.assertEqual(len(names), 2)
            replace = {}
            for n in names:
                W = R._matrix_of(next(t for t in rd.tensors if t.name == n))
                Wh, _ = R.round_matrix_variant("ZBG128:b560s16", n, W)
                replace[n] = Wh
            out_a = os.path.join(td, "a.gguf")
            ggufedit.write_copy(rd, out_a, replace)
            files = {}
            for n in names:
                os.makedirs(os.path.join(td, "sc"), exist_ok=True)
                p = os.path.join(td, "sc", re.sub(r"[^A-Za-z0-9_.+-]", "_", n) + ".f16.npy")
                np.save(p, replace[n])
                files[n] = p
            out_b = os.path.join(td, "b.gguf")
            ggufedit.write_copy_streaming(rd, out_b, files)
            self.assertEqual(R.sha256_of(out_a), R.sha256_of(out_b))

    def test_round_one_worker_sidecar(self):
        with tempfile.TemporaryDirectory() as td:
            model = os.path.join(td, "tiny.gguf")
            self._tiny_gguf(model)
            tdir = os.path.join(td, "tensors")
            try:
                R._worker_init(model, None, None, tdir)
                name = R._targets(R._get_reader())[0].name
                got_name, Wh, info, sidecar = R._round_one_worker(("ZBG128:b560s16", name))
                self.assertEqual(got_name, name)
                self.assertIsNone(Wh)
                self.assertTrue(sidecar.endswith(".f16.npy") and os.path.exists(sidecar))
                W = R._matrix_of(next(t for t in R._get_reader().tensors if t.name == name))
                Wh_ref, info_ref = R.round_matrix_variant("ZBG128:b560s16", name, W)
                self.assertTrue(np.array_equal(np.load(sidecar), Wh_ref))
                self.assertEqual(info, info_ref)
            finally:
                R._worker_init(model, None, None, None)

    def test_stream_arms_flag_in_dry_run(self):
        with tempfile.TemporaryDirectory() as td:
            p = subprocess.run(
                [sys.executable, "-B", str(RUN_PATH), "--model-id", "m", "--model-path", os.path.join(td, "m.gguf"),
                 "--out-dir", td, "--stream-arms", "--dry-run"],
                capture_output=True, text=True, check=True)
            self.assertIn('"arms"', p.stdout)


class TestQ8RefStratum(unittest.TestCase):
    """Q8_0-reference stratum: --allow-requantize and GGUF-magic cache validation."""

    def _fake_tensor(self, name, code):
        return types.SimpleNamespace(name=name, tensor_type=code, shape=(128, 4))

    def test_source_is_quantized(self):
        self.assertTrue(R._source_is_quantized([self._fake_tensor("blk.0.attn_q.weight", 8)]))
        self.assertFalse(R._source_is_quantized([self._fake_tensor("blk.0.attn_q.weight", 30)]))
        self.assertFalse(R._source_is_quantized([self._fake_tensor("token_embd.weight", 0)]))

    def test_valid_gguf_magic(self):
        with tempfile.TemporaryDirectory() as td:
            good = os.path.join(td, "good.gguf")
            bad = os.path.join(td, "bad.gguf")
            with open(good, "wb") as f:
                f.write(b"GGUF" + b"\x00" * 100)
            with open(bad, "wb") as f:
                f.write(b"\x00" * 100)
            self.assertTrue(R._valid_gguf(good))
            self.assertFalse(R._valid_gguf(bad))
            self.assertFalse(R._valid_gguf(os.path.join(td, "missing.gguf")))

    def test_baseline_requant_flag_and_corrupt_cache_removal(self):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.gguf")
            with open(out, "wb") as f:
                f.write(b"\x00" * 64)  # zero-header partial from a killed run
            q_tensors = [self._fake_tensor("blk.0.attn_q.weight", 8), self._fake_tensor("token_embd.weight", 8)]
            cmd = R.ensure_baseline_gguf("model.gguf", out, "Q4_0", "/bin/echo", q_tensors, threads=4)
            self.assertIn("--allow-requantize", cmd)
            self.assertFalse(os.path.exists(out))  # corrupt cache was removed
            f_tensors = [self._fake_tensor("blk.0.attn_q.weight", 30), self._fake_tensor("token_embd.weight", 30)]
            out2 = os.path.join(td, "out2.gguf")
            cmd2 = R.ensure_baseline_gguf("model.gguf", out2, "Q4_0", "/bin/echo", f_tensors, threads=4, dry_run=True)
            self.assertNotIn("--allow-requantize", cmd2)

    def test_kld_cache_trusted_by_size_not_gguf_magic(self):
        with tempfile.TemporaryDirectory() as td:
            kld = os.path.join(td, "m.wikitext.kld")
            with open(kld, "wb") as f:
                f.write(b"_log" + b"\x00" * 64)  # .kld files start with "_log", not GGUF
            cmd = R.ensure_kld("model.gguf", "text.raw", kld, "/bin/echo", 40, 512, 512, 40, 4)
            self.assertTrue(os.path.exists(kld))  # non-empty kld cache is reused, not deleted
            self.assertIn("--kl-divergence-base", cmd)

    def test_drop_file_cache_best_effort(self):
        with tempfile.TemporaryDirectory() as td:
            f = os.path.join(td, "x.bin")
            with open(f, "wb") as fh:
                fh.write(b"\x00" * 4096)
            R._drop_file_cache(f)                      # real file: must not raise
            R._drop_file_cache(os.path.join(td, "nope"))  # missing file: must not raise


if __name__ == "__main__":
    unittest.main()
