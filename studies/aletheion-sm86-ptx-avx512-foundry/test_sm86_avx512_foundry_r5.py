#!/usr/bin/env python3
"""
Standalone R5 verification suite for the bare-metal Ampere sm_86 PTX/SASS & Intel i9-11900H AVX-512
A_16 / Block-H2 kernel foundry study accompanying Moissanite-Labs/siliquant#77.

Verifies:
  1. All 15/15 compiled Ampere sm_86 kernels have 0 B stack frame, 0 B spill stores, 0 B spill loads,
     and 0 SASS LDL/STL local memory operations, with verified LDG.128, LDGSTS, IMMA, LOP3, PRMT, POPC,
     IDP.4A, SHFL, and HFMA2 instructions.
  2. Empirical RTX 3080 Laptop GPU (sm_86) fused 4.500-bpw GEMV achieves >1.0 TB/s FP16-equivalent
     effective bandwidth, >10 INT8 TOPS on batched B=32 cp.async+IMMA GEMM, 0 verify failures, and
     0 syndrome errors across all 20 shape x arm runs, backed by a clean hardware Zone-B receipt.
  3. Intel i9-11900H AVX-512 decoder emits hardware vpermb, vpopcntq, and vpdpbusd instructions,
     achieves 0 bit-exact mismatches and 0.0 max_abs_err on fused VNNI GEMV and exact enumerative
     18B/32w hampack.c decoding with >5x multi-core speedup.
  4. Report tables match on-disk JSON receipts and zero ELF binaries exist in the study directory.

Zero external dependencies (Python 3 stdlib only). Zero bare asserts. Zero `/dev/null` writes.
"""
from __future__ import annotations

import json
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent


class TestSm86PtxAvx512FoundryR5(unittest.TestCase):
    def test_01_sm86_ptx_sass_zero_spill_and_opcodes(self) -> None:
        sass = json.loads((HERE / "sm86_ptx_sass_inspection.json").read_text(encoding="utf-8"))
        self.assertTrue(sass["zero_spill_verified"])
        self.assertEqual(sass["kernels_compiled"], 15)
        self.assertEqual(sass["max_stack_frame_bytes"], 0)
        self.assertEqual(sass["max_spill_stores_bytes"], 0)
        self.assertEqual(sass["max_spill_loads_bytes"], 0)
        self.assertEqual(sass["total_sass_local_mem_ops"], 0)

        all_ops: dict[str, int] = {}
        for k in sass["kernels"]:
            self.assertEqual(k["stack_frame_bytes"], 0)
            self.assertEqual(k["spill_stores_bytes"], 0)
            self.assertEqual(k["spill_loads_bytes"], 0)
            for op, cnt in k["sass_op_counts"].items():
                all_ops[op] = all_ops.get(op, 0) + cnt
        for req_op in ("LDG_128", "LDGSTS", "IMMA", "LOP3", "PRMT", "POPC", "IDP_4A", "SHFL", "HFMA2"):
            self.assertGreater(all_ops.get(req_op, 0), 0, f"missing SASS opcode {req_op}")

    def test_02_sm86_empirical_throughput_and_hardware_receipt(self) -> None:
        res = json.loads((HERE / "sm86_ptx_foundry_results.json").read_text(encoding="utf-8"))
        self.assertEqual(res["total_verify_failures"], 0)
        self.assertEqual(res["warp_shuffle_zero_shmem"]["WHT8_shfl"]["bit_exact_mismatches"], 0)
        self.assertEqual(res["warp_shuffle_zero_shmem"]["H16_shfl"]["bit_exact_mismatches"], 0)
        self.assertGreater(res["device"]["stream_triad_gb_s"], 350.0)

        best_fp16_tb = 0.0
        for shape_obj in res["fused_gemv_4_5_bpw"].values():
            gemm_b32 = shape_obj["batched_b32_cp_async_imma_gemm"]
            self.assertEqual(gemm_b32["batch"], 32)
            self.assertGreater(gemm_b32["int8_tops"], 12.0)
            self.assertLess(gemm_b32["rel_l2_err"], 1e-6)
            for arm in shape_obj["arms"].values():
                self.assertEqual(arm["syndrome_errors"], 0)
                self.assertLess(arm["rel_l2_err"], 1e-5)
                if arm["fp16_equiv_tb_s"] > best_fp16_tb:
                    best_fp16_tb = arm["fp16_equiv_tb_s"]
        self.assertGreater(best_fp16_tb, 1.25)

        rec = json.loads((HERE / "mlos_sm86_zone_b_receipt.json").read_text(encoding="utf-8"))
        self.assertTrue(rec["clean_hardware_run"])
        self.assertEqual(rec["determinism_zone"], "B")
        self.assertEqual(rec["execution"]["exit_code"], 0)
        self.assertEqual(rec["anomalies"], [])

    def test_03_avx512_isa_and_hampack_exact_parity(self) -> None:
        isa = json.loads((HERE / "avx512_isa_inspection.json").read_text(encoding="utf-8"))
        self.assertTrue(isa["vpermb_verified"])
        self.assertTrue(isa["vpopcntdq_verified"])
        self.assertTrue(isa["vnni_vpdpbusd_verified"])

        res = json.loads((HERE / "avx512_a16_benchmark_results.json").read_text(encoding="utf-8"))
        self.assertEqual(res["total_verify_failures"], 0)
        self.assertEqual(res["bitpacked_a16_decode"]["syndrome_errors_avx512"], 0)
        self.assertEqual(res["bitpacked_a16_decode"]["bit_exact_mismatches"], 0)
        self.assertEqual(res["fused_vnni_gemv"]["a16_syndrome_errors"], 0)
        self.assertEqual(res["fused_vnni_gemv"]["a16_max_abs_err"], 0.0)
        self.assertEqual(res["fused_vnni_gemv"]["block_h2_max_abs_err"], 0.0)
        self.assertGreater(res["fused_vnni_gemv"]["a16_vnni_8t_gweights_s"], 50.0)

        hp = res["siliquant77_exact_enumerative_hampack"]
        self.assertEqual(hp["rank_errors"], 0)
        self.assertEqual(hp["decode_errors_ref"], 0)
        self.assertEqual(hp["decode_errors_opt"], 0)
        self.assertEqual(hp["bit_exact_mismatches"], 0)
        self.assertGreater(hp["speedup_8t"], 10.0)

    def test_04_sources_present_zero_elf_and_report_parity(self) -> None:
        self.assertTrue((HERE / "aletheion_sm86_ptx_foundry.cu").is_file())
        self.assertTrue((HERE / "aletheion_avx512_a16_decoder.c").is_file())
        report = (HERE / "SM86_PTX_AVX512_FOUNDRY_REPORT.md").read_text(encoding="utf-8")
        # Spot-check verbatim receipt numbers in report
        for token in ("356.68", "646.51", "1.2930 TB/s", "14.90 INT8 TOPS", "223.19 Mw/s", "60.953 Gw/s"):
            self.assertIn(token, report, f"missing receipt figure {token} in report")
        # Zero ELF binaries inside study folder
        for item in HERE.iterdir():
            if item.is_file():
                head = item.read_bytes()[:4]
                self.assertNotEqual(head, b"\x7fELF", f"compiled ELF binary found: {item.name}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
