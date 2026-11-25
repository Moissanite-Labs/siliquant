#!/usr/bin/env python3
"""Comprehensive CPU-only unit & end-to-end integration test suite for Track S74 (`siliquant-hf-v1`).

Covers:
  1. Bit-exact `A_16` Extended Hamming [16, 11, 4] codec round-trip (`0.0` diff, `0` syndrome errors)
     across `axis`, `block_h2`, `block_h4`, `h16`, `g=32` & `g=128`, `bpw=4.5` & `3.0`,
     `k_bits in {2, 3, 4}`, optional `colscale` & `perm_in`, odd `N_blk` tail padding,
     and large overflow values (`|k| > 127` triggering `int16` overflow path).
  2. Fail-closed negative tests: invalid `A_16` parity rejected by `pack_a16_tensor`,
     non-zero `qham11` tail padding rejected by `unpack_a16_integers`, SHA-256 tamper rejected,
     `/home/` path leak rejected, and `Qwen2.5-3B` mislabeled `apache-2.0` rejected.
  3. Dtype immunity: `SiliquantLinear` buffers stay bit-identical (`0.0` diff in FP32 dequant)
     after `.to(torch.bfloat16)`, `.to(torch.float16)`, and `.to(torch.float32)`.
  4. End-to-end synthetic tiny `Qwen2` (both `tie_word_embeddings=True` with `quad_lpt_down_wcol`
     offline fold AND `tie_word_embeddings=False` with explicit `perm_in` + `colscale`) cold-reloaded
     via `AutoModelForCausalLM.from_pretrained(tmp_dir, trust_remote_code=True, dtype="auto")`:
     0 missing/unexpected keys, 0 syndrome errors, 0.0 weight diff, 0.0 logits diff,
     live `.generate()` non-empty finite tokens, genuine compression vs BF16,
     `siliquant_dequant.py` CLI PASS, `verify_siliquant_hf.py` PASS.
  5. End-to-end synthetic tiny `Llama` (`SmolLM2`-like, `g=32`, `3.0 bpw` & `4.5 bpw`,
     `cache_dequantized=False` on-the-fly mode) cold-reloaded via `AutoModelForCausalLM.from_pretrained`:
     0 syndrome errors, 0.0 weight diff, 0.0 logits diff, `.generate()` PASS, verifier PASS.
  6. Exact upstream license YAML compliance for all 4 Wave 1-2 HF repos (`SmolLM2-360M`,
     `Qwen2.5-0.5B`, `Qwen2.5-1.5B`, `Qwen2.5-3B`).
  7. Reference AVX-512 C kernel live CPU compilation + bit-exact self-test (in ephemeral tmp dir)
     and hygiene check on both reference kernels.
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from configuration_siliquant import SiliquantConfig
from export_siliquant_hf import (
    export_model_to_siliquant_hf,
    make_quad_lpt_perm,
    make_quad_snake_perm,
    quantize_a16_matrix,
    resolve_upstream_license,
)
from modeling_siliquant import (
    SiliquantForCausalLM,
    SiliquantLinear,
    check_a16_syndrome_errors,
    dequantize_a16_weight,
    pack_a16_tensor,
    unpack_a16_integers,
)
from siliquant_dequant import inspect_model, unpack_to_safetensors, verify_receipt_and_syndromes
from verify_siliquant_hf import (
    VerificationError,
    scan_export_hygiene,
    verify_readme_license,
    verify_siliquant_export,
)


def _make_tiny_qwen2(*, tie_word_embeddings: bool = True, seed: int = 20261124) -> Qwen2ForCausalLM:
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=tie_word_embeddings,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = Qwen2ForCausalLM(cfg)
    finally:
        torch.set_default_dtype(prev_dtype)
    model.eval()
    return model


def _make_tiny_llama(*, tie_word_embeddings: bool = True, seed: int = 20261125) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=256,
        hidden_size=96,
        intermediate_size=192,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=tie_word_embeddings,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = LlamaForCausalLM(cfg)
    finally:
        torch.set_default_dtype(prev_dtype)
    model.eval()
    return model


def _build_quantized_reference_model(
    orig_model: torch.nn.Module,
    ref_dequant_f32: dict[str, torch.Tensor],
    work_biases: dict[str, torch.Tensor | None],
) -> torch.nn.Module:
    """Copy original BF16 model (preserving FP32 RoPE buffers) with exact dequantized BF16 weights."""
    ref = copy.deepcopy(orig_model)
    with torch.no_grad():
        for name, w_f32 in ref_dequant_f32.items():
            mod = ref.get_submodule(name)
            mod.weight.copy_(w_f32.to(torch.bfloat16))
            if work_biases.get(name) is not None and mod.bias is not None:
                mod.bias.copy_(work_biases[name].to(torch.bfloat16))
    ref.eval()
    return ref


class TestSiliquantCodec(unittest.TestCase):
    def test_codec_roundtrip_all_rotations_groups_and_bpw(self) -> None:
        torch.manual_seed(42)
        combos = [
            (128, "4.5", "axis", False, False),
            (128, "4.5", "block_h2", False, False),
            (128, "4.5", "block_h4", True, True),
            (128, "4.5", "h16", True, False),
            (32, "4.5", "block_h4", False, True),
            (128, "3.0", "block_h4", False, False),
            (32, "3.0", "block_h2", True, True),
        ]
        for g, bpw, rot, use_cs, use_perm in combos:
            m, k = 32, 128
            w = torch.randn((m, k), dtype=torch.float32) * 0.08
            cs = (torch.rand(k, dtype=torch.float32) + 0.5) if use_cs else None
            perm = make_quad_lpt_perm((w ** 2).mean(dim=0), g, block_w=4) if use_perm else None
            y_int, sc_f16, wq_ref, syn_q = quantize_a16_matrix(
                w, group_size=g, bpw_key=bpw, rotation=rot, colscale=cs, perm_in=perm
            )
            self.assertEqual(syn_q, 0)
            cs_store = cs.index_select(0, perm.to(torch.long)) if (cs is not None and perm is not None) else cs

            # Test auto k_bits AND explicit k_bits in (2, 3, 4).
            for kb in (None, 2, 3, 4):
                bufs, meta = pack_a16_tensor(
                    y_int,
                    sc_f16,
                    group_size=g,
                    rotation=rot,
                    colscale_f32=cs_store,
                    perm_in=perm,
                    k_bits=kb,
                )
                wq_rt, syn_rt = dequantize_a16_weight(
                    bufs["qquot"],
                    bufs["qham11"],
                    bufs["scales_fp16_bits"],
                    out_features=m,
                    in_features=k,
                    group_size=g,
                    rotation=rot,
                    k_bits=meta["k_bits"],
                    qov_rowptr=bufs.get("qov_rowptr"),
                    qov_col=bufs.get("qov_col"),
                    qov_val=bufs.get("qov_val"),
                    colscale_fp32_bits=bufs.get("colscale_fp32_bits"),
                    perm_in=bufs.get("perm_in"),
                    verify_syndrome=True,
                    target_dtype=torch.float32,
                )
                self.assertEqual(syn_rt, 0)
                diff = float((wq_rt - wq_ref).abs().max().item())
                self.assertEqual(diff, 0.0, f"Non-zero diff for {(g, bpw, rot, use_cs, use_perm, kb)}")

    def test_odd_block_padding_and_int16_overflow(self) -> None:
        # M=17, K=32 => N_blk=34 (not divisible by 8!) AND inject |k| = 200 (> 127 => int16 overflow!).
        torch.manual_seed(7)
        m, k = 17, 32
        w = torch.randn((m, k), dtype=torch.float32) * 0.1
        y_int, sc_f16, _, syn_q = quantize_a16_matrix(w, group_size=32, bpw_key="4.5", rotation="axis")
        self.assertEqual(syn_q, 0)
        # Adding an even integer (400 = 2*200) preserves parity bits c = y & 1 so A_16 validity holds!
        y_big = y_int.clone()
        y_big[0, 3] = y_big[0, 3] + 400
        y_big[16, 31] = y_big[16, 31] - 420
        self.assertEqual(check_a16_syndrome_errors(y_big), 0)

        bufs, meta = pack_a16_tensor(y_big, sc_f16, group_size=32, rotation="axis", k_bits=4)
        self.assertEqual(meta["ov_val_dtype"], "int16")
        y_rt, syn_rt = unpack_a16_integers(
            bufs["qquot"],
            bufs["qham11"],
            out_features=m,
            in_features=k,
            k_bits=meta["k_bits"],
            qov_rowptr=bufs["qov_rowptr"],
            qov_col=bufs["qov_col"],
            qov_val=bufs["qov_val"],
            verify_syndrome=True,
        )
        self.assertEqual(syn_rt, 0)
        self.assertTrue(torch.equal(y_rt, y_big.to(torch.int32)))

        # Corrupting unused padding bits in the final byte of qham11 MUST fail closed.
        bad_qham = bufs["qham11"].clone()
        bad_qham[-1] = bad_qham[-1] ^ 0x80
        with self.assertRaises(ValueError):
            unpack_a16_integers(
                bufs["qquot"],
                bad_qham,
                out_features=m,
                in_features=k,
                k_bits=meta["k_bits"],
                qov_rowptr=bufs["qov_rowptr"],
                qov_col=bufs["qov_col"],
                qov_val=bufs["qov_val"],
                verify_syndrome=True,
            )

    def test_pack_rejects_corrupted_a16_syndrome(self) -> None:
        torch.manual_seed(11)
        w = torch.randn((16, 128), dtype=torch.float32) * 0.05
        y_int, sc_f16, _, syn = quantize_a16_matrix(w, group_size=128, bpw_key="4.5", rotation="axis")
        self.assertEqual(syn, 0)
        y_bad = y_int.clone()
        y_bad[3, 5] += 1  # flips 1 parity bit -> guaranteed Extended Hamming [16,11,4] syndrome violation
        self.assertEqual(check_a16_syndrome_errors(y_bad), 1)
        with self.assertRaises(ValueError):
            pack_a16_tensor(y_bad, sc_f16, group_size=128, rotation="axis")

    def test_siliquant_linear_dtype_immunity(self) -> None:
        torch.manual_seed(19)
        m, k = 32, 128
        w = torch.randn((m, k), dtype=torch.float32) * 0.05
        cs = torch.linspace(0.7, 1.4, k, dtype=torch.float32)
        y_int, sc_f16, wq_ref, _ = quantize_a16_matrix(
            w, group_size=128, bpw_key="4.5", rotation="block_h4", colscale=cs
        )
        bufs, meta = pack_a16_tensor(
            y_int, sc_f16, group_size=128, rotation="block_h4", colscale_f32=cs
        )
        layer = SiliquantLinear(
            in_features=k,
            out_features=m,
            group_size=128,
            rotation="block_h4",
            k_bits=meta["k_bits"],
            n_overflow=meta["n_overflow"],
            ov_col_dtype=meta["ov_col_dtype"],
            ov_val_dtype=meta["ov_val_dtype"],
            has_colscale=True,
            has_perm_in=False,
            has_bias=True,
        )
        layer.load_state_dict({**bufs, "bias": torch.zeros(m, dtype=torch.float32)}, strict=True)

        # Casting module repeatedly across bfloat16/float16/float32 MUST NEVER corrupt scales/colscale.
        for dt in (torch.bfloat16, torch.float16, torch.float32):
            layer.to(dtype=dt)
            w_rt = layer.dequantize(dtype=torch.float32, verify_syndrome=True)
            self.assertEqual(float((w_rt - wq_ref).abs().max().item()), 0.0)


class TestSiliquantEndToEndHF(unittest.TestCase):
    def test_qwen2_tied_quad_lpt_down_offline_fold_end_to_end(self) -> None:
        orig = _make_tiny_qwen2(tie_word_embeddings=True, seed=101)
        with tempfile.TemporaryDirectory(prefix="s74_qwen2_tied_") as tmp:
            out_dir = Path(tmp) / "Qwen2.5-3B-Siliquant-A16-BlockH4"
            res = export_model_to_siliquant_hf(
                orig,
                out_dir,
                base_model_id="Qwen/Qwen2.5-3B",
                repo_name="siliquant-test/Qwen2.5-3B-Siliquant-A16-BlockH4",
                group_size=128,
                bpw_key="4.5",
                rotation="block_h4",
                perm_mode="quad_lpt_down_wcol",
                benchmarks={"wt2_mean_kl": 0.036518, "wt2_ppl": 8.2104},
                device=torch.device("cpu"),
            )
            receipt = res["receipt"]
            summary = receipt["summary"]
            self.assertEqual(summary["total_syndrome_errors"], 0)
            self.assertEqual(summary["max_abs_roundtrip_diff"], 0.0)
            self.assertTrue(summary["tie_word_embeddings"])
            # Genuine byte compression vs BF16 on quantized weights (> 2.5x!).
            self.assertGreater(summary["quant_compression_ratio_vs_bf16"], 2.5)
            # Confirm tied lm_head was not duplicated in safetensors.
            st_keys = set(load_file(str(out_dir / "model.safetensors"), device="cpu").keys())
            self.assertIn("model.embed_tokens.weight", st_keys)
            self.assertNotIn("lm_head.weight", st_keys)

            # Build exact BF16 reference quantized model to assert 0.0 max abs logits diff.
            ref_model = _build_quantized_reference_model(
                orig, res["ref_dequant_f32"], res["work_biases"]
            )
            input_ids = torch.tensor([[5, 19, 42, 77, 13, 88]], dtype=torch.long)
            with torch.no_grad():
                ref_logits = ref_model(input_ids, use_cache=False).logits

            # Cold reload + full fail-closed verification via verify_siliquant_hf.py!
            ver = verify_siliquant_export(
                out_dir,
                ref_weights_f32=res["ref_dequant_f32"],
                ref_logits_input_ids=input_ids,
                ref_logits=ref_logits,
                max_new_tokens=32,
                device="cpu",
                expected_benchmarks={"wt2_mean_kl": 0.036518, "wt2_ppl": 8.2104},
            )
            self.assertEqual(ver["status"], "PASS")
            self.assertEqual(ver["syndrome_report"]["total_syndrome_errors"], 0)
            self.assertEqual(ver["max_abs_weight_diff"], 0.0)
            self.assertEqual(ver["max_abs_logits_diff"], 0.0)
            self.assertEqual(ver["generated_new_tokens"], 32)
            self.assertEqual(ver["license_check"]["license"], "other")
            self.assertEqual(ver["license_check"]["license_name"], "qwen-research")

            # Test standalone siliquant_dequant.py inspect / verify / unpack!
            ins = inspect_model(out_dir)
            self.assertEqual(ins["total_syndrome_errors"], 0)
            vcli = verify_receipt_and_syndromes(out_dir)
            self.assertEqual(vcli["status"], "PASS")
            unpacked_path = Path(tmp) / "unpacked.safetensors"
            unp = unpack_to_safetensors(out_dir, unpacked_path, dtype_name="bfloat16")
            self.assertEqual(unp["total_syndrome_errors"], 0)
            unp_tensors = load_file(str(unpacked_path), device="cpu")
            for lname, ref_w in res["ref_dequant_f32"].items():
                self.assertTrue(torch.equal(unp_tensors[f"{lname}.weight"], ref_w.to(torch.bfloat16)))

    def test_qwen2_untied_explicit_perm_and_colscale_end_to_end(self) -> None:
        orig = _make_tiny_qwen2(tie_word_embeddings=False, seed=202)
        h_diag = {}
        torch.manual_seed(203)
        for name, mod in orig.named_modules():
            if isinstance(mod, torch.nn.Linear) and name.startswith("model.layers."):
                h_diag[name] = torch.rand(mod.in_features, dtype=torch.float32) + 0.1

        with tempfile.TemporaryDirectory(prefix="s74_qwen2_untied_") as tmp:
            out_dir = Path(tmp) / "Qwen2.5-1.5B-Siliquant-hA16-QuadSnake"
            res = export_model_to_siliquant_hf(
                orig,
                out_dir,
                base_model_id="Qwen/Qwen2.5-1.5B",
                repo_name="siliquant-test/Qwen2.5-1.5B-Siliquant-hA16-QuadSnake",
                group_size=128,
                bpw_key="4.5",
                rotation="projtype_uniform",
                perm_mode="quad_snake_hreal",
                hessian_diag=h_diag,
                use_colscale=True,
                device=torch.device("cpu"),
            )
            st_keys = set(load_file(str(out_dir / "model.safetensors"), device="cpu").keys())
            self.assertIn("lm_head.weight", st_keys)
            self.assertIn("model.embed_tokens.weight", st_keys)

            ref_model = _build_quantized_reference_model(
                orig, res["ref_dequant_f32"], res["work_biases"]
            )
            input_ids = torch.tensor([[3, 7, 15, 31, 63]], dtype=torch.long)
            with torch.no_grad():
                ref_logits = ref_model(input_ids, use_cache=False).logits

            ver = verify_siliquant_export(
                out_dir,
                ref_weights_f32=res["ref_dequant_f32"],
                ref_logits_input_ids=input_ids,
                ref_logits=ref_logits,
                max_new_tokens=16,
                device="cpu",
            )
            self.assertEqual(ver["status"], "PASS")
            self.assertEqual(ver["max_abs_weight_diff"], 0.0)
            self.assertEqual(ver["max_abs_logits_diff"], 0.0)
            self.assertEqual(ver["license_check"]["license"], "apache-2.0")

    def test_llama_smollm2_g32_3bpw_and_uncached_end_to_end(self) -> None:
        orig = _make_tiny_llama(tie_word_embeddings=True, seed=303)
        with tempfile.TemporaryDirectory(prefix="s74_llama_3bpw_") as tmp:
            out_dir = Path(tmp) / "SmolLM2-360M-Siliquant-A16-3bpw"
            res = export_model_to_siliquant_hf(
                orig,
                out_dir,
                base_model_id="HuggingFaceTB/SmolLM2-360M",
                repo_name="siliquant-test/SmolLM2-360M-Siliquant-A16-3bpw",
                group_size=32,
                bpw_key="3.0",
                rotation="block_h4",
                perm_mode="quad_lpt_wcol",
                cache_dequantized=False,
                device=torch.device("cpu"),
            )
            # 3.0 bpw g=32 must compress > 3.4x vs BF16 even on tiny 96x96 layers (and < 4.0 bpw at real width!).
            eff_bpw = res["receipt"]["summary"]["effective_packed_bpw"]
            self.assertGreater(res["receipt"]["summary"]["quant_compression_ratio_vs_bf16"], 3.4)
            self.assertLess(eff_bpw, 4.8)

            # Also verify realistic SmolLM2-width (960x960) g=32 (< 4.30 bpw) & Qwen-width (1024x1024) g=128 (< 3.95 bpw) for 3.0 bpw!
            torch.manual_seed(304)
            w_g32 = torch.randn((960, 960), dtype=torch.float32) * 0.04
            y_r, sc_r, _, syn_r = quantize_a16_matrix(w_g32, group_size=32, bpw_key="3.0", rotation="block_h4")
            self.assertEqual(syn_r, 0)
            bufs_r, _ = pack_a16_tensor(y_r, sc_r, group_size=32, rotation="block_h4")
            b_r = sum(int(t.numel()) * int(t.element_size()) for t in bufs_r.values())
            self.assertLess(8.0 * b_r / float(960 * 960), 4.30)

            w_g128 = torch.randn((1024, 1024), dtype=torch.float32) * 0.04
            y_q, sc_q, _, syn_q = quantize_a16_matrix(w_g128, group_size=128, bpw_key="3.0", rotation="block_h4")
            self.assertEqual(syn_q, 0)
            bufs_q, _ = pack_a16_tensor(y_q, sc_q, group_size=128, rotation="block_h4")
            b_q = sum(int(t.numel()) * int(t.element_size()) for t in bufs_q.values())
            self.assertLess(8.0 * b_q / float(1024 * 1024), 3.95)

            ref_model = _build_quantized_reference_model(
                orig, res["ref_dequant_f32"], res["work_biases"]
            )
            input_ids = torch.tensor([[9, 18, 27, 36, 45]], dtype=torch.long)
            with torch.no_grad():
                ref_logits = ref_model(input_ids, use_cache=False).logits

            ver = verify_siliquant_export(
                out_dir,
                ref_weights_f32=res["ref_dequant_f32"],
                ref_logits_input_ids=input_ids,
                ref_logits=ref_logits,
                max_new_tokens=32,
                device="cpu",
            )
            self.assertEqual(ver["status"], "PASS")
            self.assertEqual(ver["syndrome_report"]["total_syndrome_errors"], 0)
            self.assertEqual(ver["max_abs_weight_diff"], 0.0)
            self.assertEqual(ver["max_abs_logits_diff"], 0.0)
            self.assertEqual(ver["license_check"]["license"], "apache-2.0")

            # Confirm on-the-fly uncached mode does not populate _cached_weight.
            reloaded = AutoModelForCausalLM.from_pretrained(
                str(out_dir), trust_remote_code=True, dtype="auto", local_files_only=True
            )
            _ = reloaded(input_ids, use_cache=False)
            for mod in reloaded.modules():
                if isinstance(mod, SiliquantLinear):
                    self.assertIsNone(mod._cached_weight)


class TestComplianceAndNegativeGuards(unittest.TestCase):
    def test_upstream_license_registry_exact(self) -> None:
        s360 = resolve_upstream_license("HuggingFaceTB/SmolLM2-360M")
        q05 = resolve_upstream_license("Qwen/Qwen2.5-0.5B")
        q15 = resolve_upstream_license("Qwen/Qwen2.5-1.5B")
        q3 = resolve_upstream_license("Qwen/Qwen2.5-3B")

        self.assertEqual(s360["license"], "apache-2.0")
        self.assertEqual(q05["license"], "apache-2.0")
        self.assertEqual(q15["license"], "apache-2.0")
        self.assertEqual(q3["license"], "other")
        self.assertEqual(q3["license_name"], "qwen-research")
        self.assertEqual(q3["license_link"], "https://huggingface.co/Qwen/Qwen2.5-3B/blob/main/LICENSE")

        # Attempt to mislabel Qwen2.5-3B as apache-2.0 MUST fail closed.
        with self.assertRaises(ValueError):
            resolve_upstream_license("Qwen/Qwen2.5-3B", override_license="apache-2.0")

    def test_verifier_fails_closed_on_leaks_tamper_and_wrong_license(self) -> None:
        orig = _make_tiny_qwen2(tie_word_embeddings=True, seed=404)
        with tempfile.TemporaryDirectory(prefix="s74_neg_") as tmp:
            out_dir = Path(tmp) / "Qwen2.5-3B-NegTest"
            export_model_to_siliquant_hf(
                orig,
                out_dir,
                base_model_id="Qwen/Qwen2.5-3B",
                repo_name="siliquant-test/Qwen2.5-3B-NegTest",
                group_size=128,
                bpw_key="4.5",
                rotation="axis",
                device=torch.device("cpu"),
            )

            # 1. Tampering with README.md breaks SHA-256 receipt check.
            readme = out_dir / "README.md"
            orig_readme = readme.read_text(encoding="utf-8")
            readme.write_text(orig_readme + "\ntampered\n", encoding="utf-8")
            with self.assertRaises(VerificationError):
                verify_siliquant_export(out_dir, max_new_tokens=4)

            # 2. Injecting a home-directory path leak fails hygiene scan.
            fake_home_leak = "".join(["/", "home", "/alice/secret"])
            readme.write_text(orig_readme + "\n" + fake_home_leak + "\n", encoding="utf-8")
            with self.assertRaises(VerificationError):
                scan_export_hygiene(out_dir)

            # 3. Mislabeled Qwen2.5-3B apache-2.0 fails license compliance check.
            bad_lic = orig_readme.replace("license: other", "license: apache-2.0")
            readme.write_text(bad_lic, encoding="utf-8")
            with self.assertRaises(VerificationError):
                verify_readme_license(out_dir, "Qwen/Qwen2.5-3B")

    def test_no_bare_asserts_in_verifier_and_no_leaks_in_package(self) -> None:
        hygiene = scan_export_hygiene(SCRIPT_DIR)
        self.assertEqual(hygiene["findings"], [])
        verifier_src = (SCRIPT_DIR / "verify_siliquant_hf.py").read_text(encoding="utf-8")
        for line in verifier_src.splitlines():
            stripped = line.lstrip()
            self.assertFalse(
                stripped.startswith("assert "),
                f"Bare assert found in verify_siliquant_hf.py: {line!r}",
            )

    def test_reference_avx512_kernel_builds_and_passes_cpu_selftest(self) -> None:
        gcc = shutil.which("gcc")
        if gcc is None:
            self.skipTest("gcc not available on PATH")
        src = SCRIPT_DIR / "kernels" / "avx512_a16_blockh4.c"
        self.assertTrue(src.is_file())
        with tempfile.TemporaryDirectory(prefix="s74_avx512_") as tmp:
            # Test BOTH native AVX-512 VNNI AND scalar fallback under ASan + UBSan + -Wall -Wextra
            # across K=1024, K=960 (SmolLM2-360M, K%128!=0), and K=112 (pure tail).
            for tag, flags in (
                ("avx512_asan", ["-O2", "-g", "-Wall", "-Wextra", "-Werror", "-march=native", "-fsanitize=address,undefined"]),
                ("scalar_asan", ["-O2", "-g", "-Wall", "-Wextra", "-Werror", "-mno-avx512f", "-fsanitize=address,undefined"]),
            ):
                exe = Path(tmp) / f"avx512_a16_blockh4_{tag}"
                comp = subprocess.run(
                    [gcc, *flags, str(src), "-lm", "-o", str(exe)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(comp.returncode, 0, f"gcc build ({tag}) failed: {comp.stderr}")
                run = subprocess.run([str(exe)], capture_output=True, text=True, check=False)
                self.assertEqual(run.returncode, 0, f"self-test ({tag}) failed: {run.stdout}\n{run.stderr}")
                self.assertIn("K=1024", run.stdout)
                self.assertIn("K=960", run.stdout)
                self.assertIn("K=112", run.stdout)
                self.assertNotIn("mismatches=1", run.stdout)


class TestAudit174ARegressions(unittest.TestCase):
    """Explicit fail-closed regression tests for AUDIT-174A findings S74-F1 through S74-F7."""

    def test_s74_f1_perm_in_validation(self) -> None:
        torch.manual_seed(501)
        w = torch.randn((16, 32), dtype=torch.float32) * 0.05
        y_int, sc_f16, _, _ = quantize_a16_matrix(w, group_size=32, bpw_key="4.5", rotation="axis")
        bufs, meta = pack_a16_tensor(y_int, sc_f16, group_size=32, rotation="axis")

        dup_p = torch.arange(32, dtype=torch.int32)
        dup_p[1] = 0
        neg_p = torch.tensor([-1] + list(range(31)), dtype=torch.int32)
        zero_p = torch.zeros(32, dtype=torch.int32)
        flt_p = torch.arange(32, dtype=torch.float32)

        for bad_p in (dup_p, neg_p, zero_p, flt_p):
            with self.assertRaises(ValueError):
                pack_a16_tensor(y_int, sc_f16, group_size=32, rotation="axis", perm_in=bad_p)
            with self.assertRaises(ValueError):
                dequantize_a16_weight(
                    bufs["qquot"],
                    bufs["qham11"],
                    bufs["scales_fp16_bits"],
                    out_features=16,
                    in_features=32,
                    group_size=32,
                    rotation="axis",
                    k_bits=meta["k_bits"],
                    qov_rowptr=bufs.get("qov_rowptr"),
                    qov_col=bufs.get("qov_col"),
                    qov_val=bufs.get("qov_val"),
                    perm_in=bad_p,
                )
            with self.assertRaises(ValueError):
                quantize_a16_matrix(w, group_size=32, bpw_key="4.5", rotation="axis", perm_in=bad_p)

    def test_s74_f2_csr_and_dimension_validation(self) -> None:
        torch.manual_seed(502)
        w = torch.randn((16, 32), dtype=torch.float32) * 0.05
        y_int, sc_f16, _, _ = quantize_a16_matrix(w, group_size=32, bpw_key="4.5", rotation="axis")
        # Force a known overflow entry at (0, 1) so CSR buffers exist.
        y_ov = y_int.clone()
        y_ov[0, 1] += 40
        bufs, meta = pack_a16_tensor(y_ov, sc_f16, group_size=32, rotation="axis", k_bits=4)
        base_kw = dict(
            out_features=16,
            in_features=32,
            k_bits=meta["k_bits"],
        )

        # Negative qov_col
        bad_col_neg = bufs["qov_col"].clone()
        bad_col_neg[0] = -1
        with self.assertRaises(ValueError):
            unpack_a16_integers(
                bufs["qquot"], bufs["qham11"], qov_rowptr=bufs["qov_rowptr"],
                qov_col=bad_col_neg, qov_val=bufs["qov_val"], **base_kw
            )

        # Out-of-bounds qov_col (>= K)
        bad_col_oob = bufs["qov_col"].clone()
        bad_col_oob[0] = 32
        with self.assertRaises(ValueError):
            unpack_a16_integers(
                bufs["qquot"], bufs["qham11"], qov_rowptr=bufs["qov_rowptr"],
                qov_col=bad_col_oob, qov_val=bufs["qov_val"], **base_kw
            )

        # Mismatched qov_col vs qov_val lengths (prevent silent broadcast!)
        rp2 = torch.zeros(17, dtype=torch.int32)
        rp2[1:] = 2
        with self.assertRaises(ValueError):
            unpack_a16_integers(
                bufs["qquot"], bufs["qham11"], qov_rowptr=rp2,
                qov_col=torch.tensor([0, 1], dtype=torch.int16),
                qov_val=torch.tensor([50], dtype=torch.int8), **base_kw
            )

        # Duplicate (row, col) in CSR
        with self.assertRaises(ValueError):
            unpack_a16_integers(
                bufs["qquot"], bufs["qham11"], qov_rowptr=rp2,
                qov_col=torch.tensor([1, 1], dtype=torch.int16),
                qov_val=torch.tensor([50, 51], dtype=torch.int8), **base_kw
            )

        # Non-zero rowptr[0]
        with self.assertRaises(ValueError):
            unpack_a16_integers(
                bufs["qquot"], bufs["qham11"], qov_rowptr=bufs["qov_rowptr"] + 5,
                qov_col=bufs["qov_col"], qov_val=bufs["qov_val"], **base_kw
            )

        # Wrong rowptr length
        with self.assertRaises(ValueError):
            unpack_a16_integers(
                bufs["qquot"], bufs["qham11"], qov_rowptr=torch.tensor([0, int(bufs["qov_col"].numel())], dtype=torch.int32),
                qov_col=bufs["qov_col"], qov_val=bufs["qov_val"], **base_kw
            )

        # Non-monotonic rowptr (must raise ValueError, NOT raw RuntimeError)
        bad_rp = bufs["qov_rowptr"].clone()
        bad_rp[1] = bad_rp[-1] + 5
        with self.assertRaises(ValueError):
            unpack_a16_integers(
                bufs["qquot"], bufs["qham11"], qov_rowptr=bad_rp,
                qov_col=bufs["qov_col"], qov_val=bufs["qov_val"], **base_kw
            )

        # Partial CSR buffers
        with self.assertRaises(ValueError):
            unpack_a16_integers(
                bufs["qquot"], bufs["qham11"], qov_rowptr=bufs["qov_rowptr"],
                qov_col=None, qov_val=None, **base_kw
            )

        # Odd non-multiple-of-16 dimension
        with self.assertRaises(ValueError):
            unpack_a16_integers(
                bufs["qquot"], bufs["qham11"], out_features=16, in_features=17, k_bits=4
            )

    def test_s74_f3_int32_no_wrap_and_finite_nonzero_scales(self) -> None:
        torch.manual_seed(503)
        w = torch.randn((16, 32), dtype=torch.float32) * 0.05
        y_int, sc_f16, _, _ = quantize_a16_matrix(w, group_size=32, bpw_key="4.5", rotation="axis")
        # Exact S74-F3 reproducer: y_int = 39985 (k = 19992 <= 32767) MUST NOT wrap in int16!
        y_large = y_int.to(torch.int32).clone()
        y_large[0, 0] = 39984 + (y_large[0, 0] & 1)
        expected_val = int(y_large[0, 0].item())
        self.assertGreaterEqual(expected_val, 39984)
        bufs, meta = pack_a16_tensor(y_large, sc_f16, group_size=32, rotation="axis", k_bits=4)
        y_rt, syn_rt = unpack_a16_integers(
            bufs["qquot"], bufs["qham11"], out_features=16, in_features=32, k_bits=meta["k_bits"],
            qov_rowptr=bufs["qov_rowptr"], qov_col=bufs["qov_col"], qov_val=bufs["qov_val"],
            verify_syndrome=True,
        )
        self.assertEqual(syn_rt, 0)
        self.assertEqual(int(y_rt[0, 0].item()), expected_val)
        self.assertTrue(torch.equal(y_rt, y_large))

        # Zero colscale and NaN scales_fp16 rejected in both pack and dequant.
        cs_zero = torch.ones(32, dtype=torch.float32)
        cs_zero[0] = 0.0
        with self.assertRaises(ValueError):
            pack_a16_tensor(y_int, sc_f16, group_size=32, rotation="axis", colscale_f32=cs_zero)
        sc_nan = sc_f16.clone()
        sc_nan[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            pack_a16_tensor(y_int, sc_nan, group_size=32, rotation="axis")

        bufs_ok, meta_ok = pack_a16_tensor(
            y_int, sc_f16, group_size=32, rotation="axis", colscale_f32=torch.ones(32, dtype=torch.float32)
        )
        bad_sc_bits = bufs_ok["scales_fp16_bits"].clone()
        bad_sc_bits[0, 0] = 0x7E00  # FP16 NaN
        with self.assertRaises(ValueError):
            dequantize_a16_weight(
                bufs_ok["qquot"], bufs_ok["qham11"], bad_sc_bits,
                out_features=16, in_features=32, group_size=32, rotation="axis",
                k_bits=meta_ok["k_bits"], qov_rowptr=bufs_ok.get("qov_rowptr"),
                qov_col=bufs_ok.get("qov_col"), qov_val=bufs_ok.get("qov_val"),
            )
        bad_cs_bits = bufs_ok["colscale_fp32_bits"].clone()
        bad_cs_bits[0] = 0  # FP32 +0.0
        with self.assertRaises(ValueError):
            dequantize_a16_weight(
                bufs_ok["qquot"], bufs_ok["qham11"], bufs_ok["scales_fp16_bits"],
                out_features=16, in_features=32, group_size=32, rotation="axis",
                k_bits=meta_ok["k_bits"], qov_rowptr=bufs_ok.get("qov_rowptr"),
                qov_col=bufs_ok.get("qov_col"), qov_val=bufs_ok.get("qov_val"),
                colscale_fp32_bits=bad_cs_bits,
            )

    def test_s74_f5_load_state_dict_cache_invalidation_and_submodule_guard(self) -> None:
        # 1. Zero-overflow path (exact AUDIT-174A reproducer: layer(x) -> load_state_dict -> layer(x)).
        torch.manual_seed(505)
        w0 = torch.randn((16, 32), dtype=torch.float32) * 0.2
        y0, sc0, _, _ = quantize_a16_matrix(w0, group_size=32, bpw_key="3.0", rotation="axis")
        bufs0, meta0 = pack_a16_tensor(y0, sc0, group_size=32, rotation="axis", k_bits=4)
        self.assertEqual(meta0["n_overflow"], 0)
        layer0 = SiliquantLinear(
            in_features=32,
            out_features=16,
            group_size=32,
            rotation="axis",
            k_bits=meta0["k_bits"],
            n_overflow=0,
            cache_dequantized=True,
        )
        x = torch.randn((2, 32), dtype=torch.float32)
        y_before = layer0(x)
        self.assertEqual(float(y_before.abs().max().item()), 0.0)
        self.assertIsNotNone(layer0._cached_weight)
        layer0.load_state_dict(bufs0, strict=True)
        self.assertIsNone(layer0._cached_weight)
        self.assertFalse(layer0._syndrome_verified)
        y_after = layer0(x)
        y_expected = torch.nn.functional.linear(x, layer0.dequantize(torch.float32))
        self.assertGreater(float(y_after.abs().max().item()), 0.01)
        self.assertEqual(float((y_after - y_expected).abs().max().item()), 0.0)

        # 2. Non-zero-overflow path: swapping from state_dict A to state_dict B invalidates cache.
        w1 = torch.randn((16, 32), dtype=torch.float32) * 0.2
        y1, sc1, _, _ = quantize_a16_matrix(w1, group_size=32, bpw_key="4.5", rotation="axis")
        bufs1, meta1 = pack_a16_tensor(y1, sc1, group_size=32, rotation="axis", k_bits=4)
        layer1 = SiliquantLinear(
            in_features=32,
            out_features=16,
            group_size=32,
            rotation="axis",
            k_bits=meta1["k_bits"],
            n_overflow=meta1["n_overflow"],
            ov_col_dtype=meta1["ov_col_dtype"],
            ov_val_dtype=meta1["ov_val_dtype"],
            cache_dequantized=True,
        )
        layer1.load_state_dict(bufs1, strict=True)
        y_a = layer1(x)
        bufs1_neg = {k: v.clone() for k, v in bufs1.items()}
        # Negate all FP16 scales bit-exact -> exact negated output after reload!
        bufs1_neg["scales_fp16_bits"] = (-bufs1["scales_fp16_bits"].view(torch.float16)).view(torch.int16)
        layer1.load_state_dict(bufs1_neg, strict=True)
        self.assertIsNone(layer1._cached_weight)
        y_b = layer1(x)
        self.assertEqual(float((y_a + y_b).abs().max().item()), 0.0)

        # _no_split_modules present on SiliquantForCausalLM
        self.assertEqual(
            SiliquantForCausalLM._no_split_modules, ["Qwen2DecoderLayer", "LlamaDecoderLayer"]
        )

        # Unconstrained submodule replacement (e.g. model.norm) MUST fail closed.
        orig = _make_tiny_qwen2(tie_word_embeddings=True, seed=506)
        bad_cfg = SiliquantConfig.from_base_config(
            orig.config,
            siliquant_layer_map={
                "model.norm": {
                    "in_features": 128,
                    "out_features": 128,
                    "group_size": 128,
                    "rotation": "axis",
                    "k_bits": 4,
                    "n_overflow": 0,
                }
            },
        )
        with self.assertRaises(ValueError):
            SiliquantForCausalLM(bad_cfg)

    def test_s74_f6_receipt_unlisted_traversal_symlink_and_dup_yaml(self) -> None:
        orig = _make_tiny_qwen2(tie_word_embeddings=True, seed=507)
        with tempfile.TemporaryDirectory(prefix="s74_f6_") as tmp:
            out_dir = Path(tmp) / "Qwen2.5-3B-F6"
            export_model_to_siliquant_hf(
                orig,
                out_dir,
                base_model_id="Qwen/Qwen2.5-3B",
                repo_name="siliquant-test/Qwen2.5-3B-F6",
                group_size=128,
                bpw_key="4.5",
                rotation="axis",
                device=torch.device("cpu"),
            )

            # Unlisted extra file MUST fail siliquant_dequant.verify_receipt_and_syndromes.
            rogue = out_dir / "ROGUE_EXTRA_FILE.py"
            rogue.write_text("print('rogue')\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_receipt_and_syndromes(out_dir)
            rogue.unlink()

            # Symlink inside export dir MUST fail both verifiers.
            outside = Path(tmp) / "outside_file.txt"
            outside.write_text("secret\n", encoding="utf-8")
            sym = out_dir / "link.txt"
            sym.symlink_to(outside)
            with self.assertRaises(ValueError):
                verify_receipt_and_syndromes(out_dir)
            with self.assertRaises(VerificationError):
                verify_siliquant_export(out_dir, max_new_tokens=4)
            sym.unlink()

            # Path traversal entry "../outside_file.txt" in file_sha256 MUST fail closed.
            rec_path = out_dir / "siliquant_verification_receipt.json"
            orig_rec_txt = rec_path.read_text(encoding="utf-8")
            rec_data = json.loads(orig_rec_txt)
            rec_data["file_sha256"]["../outside_file.txt"] = "0" * 64
            rec_path.write_text(json.dumps(rec_data, indent=2), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_receipt_and_syndromes(out_dir)
            with self.assertRaises(VerificationError):
                verify_siliquant_export(out_dir, max_new_tokens=4)
            rec_path.write_text(orig_rec_txt, encoding="utf-8")

            # Duplicate YAML license key bypass MUST fail verify_readme_license.
            readme = out_dir / "README.md"
            orig_rd = readme.read_text(encoding="utf-8")
            dup_rd = orig_rd.replace("license: other\n", "license: apache-2.0\nlicense: other\n", 1)
            readme.write_text(dup_rd, encoding="utf-8")
            with self.assertRaises(VerificationError):
                verify_readme_license(out_dir, "Qwen/Qwen2.5-3B")

            # Spoofed license_link prefix MUST fail exact URL check.
            spoof_rd = orig_rd.replace(
                "https://huggingface.co/Qwen/Qwen2.5-3B/blob/main/LICENSE",
                "https://huggingface.co/Qwen/Qwen2.5-3B/spoof",
            )
            readme.write_text(spoof_rd, encoding="utf-8")
            with self.assertRaises(VerificationError):
                verify_readme_license(out_dir, "Qwen/Qwen2.5-3B")
            readme.write_text(orig_rd, encoding="utf-8")

    def test_s74_f6b_hf_download_artifacts_ignored_and_other_unlisted_fail(self) -> None:
        """Fresh `snapshot_download(local_dir=...)` artifacts must not trip the unlisted-file check.

        Ignored (both verifiers PASS): top-level `.gitattributes`, anything under
        `.cache/huggingface/`, and `__pycache__/**/*.pyc`. Everything else unlisted
        (and any symlink, even one named `.gitattributes`) must still fail closed.
        """
        orig = _make_tiny_qwen2(tie_word_embeddings=True, seed=517)
        with tempfile.TemporaryDirectory(prefix="s74_f6b_") as tmp:
            out_dir = Path(tmp) / "Qwen2.5-3B-F6B"
            export_model_to_siliquant_hf(
                orig,
                out_dir,
                base_model_id="Qwen/Qwen2.5-3B",
                repo_name="siliquant-test/Qwen2.5-3B-F6B",
                group_size=128,
                bpw_key="4.5",
                rotation="axis",
                device=torch.device("cpu"),
            )

            # (1) HF download metadata + __pycache__ MUST be ignored by BOTH verifiers.
            gitattributes = out_dir / ".gitattributes"
            gitattributes.write_text("* text=auto\n", encoding="utf-8")
            cache_meta = out_dir / ".cache" / "huggingface" / "download" / "x.metadata"
            cache_meta.parent.mkdir(parents=True, exist_ok=True)
            cache_meta.write_text("{}\n", encoding="utf-8")
            pyc = out_dir / "__pycache__" / "m.cpython-312.pyc"
            pyc.parent.mkdir(parents=True, exist_ok=True)
            pyc.write_bytes(b"\x00" * 16)
            vcli = verify_receipt_and_syndromes(out_dir)
            self.assertEqual(vcli["status"], "PASS")
            ver = verify_siliquant_export(out_dir, max_new_tokens=4)
            self.assertEqual(ver["status"], "PASS")
            gitattributes.unlink()
            shutil.rmtree(out_dir / ".cache")
            shutil.rmtree(out_dir / "__pycache__")

            # (2) Any OTHER unlisted file MUST still fail BOTH verifiers.
            for rogue_rel in ("notes.txt", ".cache/other/x", "extra.gitattributes"):
                rogue = out_dir / rogue_rel
                rogue.parent.mkdir(parents=True, exist_ok=True)
                rogue.write_text("rogue\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    verify_receipt_and_syndromes(out_dir)
                with self.assertRaises(VerificationError):
                    verify_siliquant_export(out_dir, max_new_tokens=4)
                if rogue_rel.startswith(".cache/"):
                    shutil.rmtree(out_dir / ".cache")
                else:
                    rogue.unlink()

            # (3) A symlink named .gitattributes (inside an ignored path) MUST still fail BOTH verifiers.
            outside = Path(tmp) / "outside_gitattributes.txt"
            outside.write_text("secret\n", encoding="utf-8")
            sym = out_dir / ".gitattributes"
            sym.symlink_to(outside)
            with self.assertRaises(ValueError):
                verify_receipt_and_syndromes(out_dir)
            with self.assertRaises(VerificationError):
                verify_siliquant_export(out_dir, max_new_tokens=4)
            sym.unlink()

            # Sanity: with everything cleaned up, both verifiers PASS again.
            self.assertEqual(verify_receipt_and_syndromes(out_dir)["status"], "PASS")
            self.assertEqual(verify_siliquant_export(out_dir, max_new_tokens=4)["status"], "PASS")

    def test_s74_f7_permuter_divisibility_and_conflicting_license_override(self) -> None:
        with self.assertRaises(ValueError):
            make_quad_snake_perm(torch.arange(100, dtype=torch.float32), 32)
        with self.assertRaises(ValueError):
            make_quad_lpt_perm(torch.arange(100, dtype=torch.float32), 32)
        with self.assertRaises(ValueError):
            resolve_upstream_license("Qwen/Qwen2.5-1.5B", override_license="mit")
        with self.assertRaises(ValueError):
            resolve_upstream_license("HuggingFaceTB/SmolLM2-360M", override_license="mit")


if __name__ == "__main__":
    unittest.main()
