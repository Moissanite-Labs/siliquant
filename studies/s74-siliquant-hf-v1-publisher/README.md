# Siliquant HF v1 (`siliquant-hf-v1`): Bit-Packed `A_16` Lattice Model Publisher & `transformers` Loader

Hugging Face `AutoModelForCausalLM` (`trust_remote_code=True`) publisher and loader, bit-packed `safetensors` codec, standalone CLI unpacker/verifier, and reference Ampere `sm_86` / Intel AVX-512 VNNI kernels for **`A_16` Extended Hamming `[16, 11, 4]`** lattice quantization (`siliquant-hf-v1`).

- **Format Version:** `siliquant-hf-v1`
- **Supported Base Architectures:** `qwen2` (`Qwen2Config` / `Qwen2Model`) and `llama` (`LlamaConfig` / `LlamaModel`, e.g., `SmolLM2`), with both tied and untied `lm_head` embeddings and both `transformers` 4.x (`past_key_values=`) and 5.x (`cache=`) compatibility.
- **Zero Extra Runtime Dependencies:** Inference via `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True, dtype="auto")` requires only standard `torch`, `transformers`, and `safetensors` (optional `device_map="auto"` with `accelerate`, `_no_split_modules = ["Qwen2DecoderLayer", "LlamaDecoderLayer"]`).
- **Live Public Hugging Face Collection:** [ahrazzle/siliquant-stratified-lattice-quantized-llms](https://huggingface.co/collections/ahrazzle/siliquant-stratified-lattice-quantized-llms-6b060e6988afd65942cd7768)

## Live Public Wave-1A Models (Exported & Cold-Reload Verified)

| Model Repo | Base Model | Upstream License | Recipe (`g`, Target) | Packed Size vs BF16 | Eff. Packed `bpw` | `A_16` Unpack Codec Check | Max Abs Round-Trip Diff |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [`ahrazzle/SmolLM2-360M-Siliquant-A16-BlockH4-AdaptZeroCal-4.5bpw`](https://huggingface.co/ahrazzle/SmolLM2-360M-Siliquant-A16-BlockH4-AdaptZeroCal-4.5bpw) | `HuggingFaceTB/SmolLM2-360M` | `apache-2.0` | `A16_AdaptZeroCal_1bit` (`g=32`, `4.5 bpw`, `97 BlockH4 + 127 axis`, zero-text) | `292.91 MiB` vs `690.12 MiB` (`2.36x` total / `2.96x` quant) | `5.4037` | `0 / 19,660,800` (`0` by construction; SHA-256 = integrity) | `0.0` |
| [`ahrazzle/Qwen2.5-0.5B-Siliquant-A16-BlockH4-QuadLPT-4.5bpw`](https://huggingface.co/ahrazzle/Qwen2.5-0.5B-Siliquant-A16-BlockH4-QuadLPT-4.5bpw) | `Qwen/Qwen2.5-0.5B` | `apache-2.0` | `hA16_BlockH4_quad_lpt_real` (`g=128`, `4.5 bpw`, `168 BlockH4 + QuadLPT + colscale`) | `483.99 MiB` vs `942.24 MiB` (`1.95x` total / `3.05x` quant) | `5.2523` | `0 / 22,364,160` (`0` by construction; SHA-256 = integrity) | `0.0` |

**Pinned revisions (checked 2026-11-26):** [`ahrazzle/SmolLM2-360M-Siliquant-A16-BlockH4-AdaptZeroCal-4.5bpw@e52d331c7ba141f77b78e4f84ffd72983fe2faf4`](https://huggingface.co/ahrazzle/SmolLM2-360M-Siliquant-A16-BlockH4-AdaptZeroCal-4.5bpw/tree/e52d331c7ba141f77b78e4f84ffd72983fe2faf4) and [`ahrazzle/Qwen2.5-0.5B-Siliquant-A16-BlockH4-QuadLPT-4.5bpw@26a0419d4025058e9e7775f53a6f1ae3b52d6b75`](https://huggingface.co/ahrazzle/Qwen2.5-0.5B-Siliquant-A16-BlockH4-QuadLPT-4.5bpw/tree/26a0419d4025058e9e7775f53a6f1ae3b52d6b75). At each revision, `siliquant_verification_receipt.json` records `siliquant_dequant.py` at SHA-256 `b391f4f36f3b5fed835c1a6f6b846b706d17cdc335072c0c0db5360999ebb953`, and the file at that revision hashes to the same value. Load a pinned copy with `from_pretrained(..., revision="<sha>")`. Hugging Face collections are not versioned, so the collection link above is not pinned.

> **Integrity & Codec Provenance Note:** On unpack, `_unpack_hamming11` algebraically reconstructs the 5 parity bits from the 11 stored systematic info bits, so the unpack syndrome count is `0` by construction and serves strictly as a decoder/mask self-consistency cross-check (`qquot` and info-bit tampering are not detected by unpack syndromes). Actual file tamper/bit-flip detection on Hugging Face exports is enforced by the SHA-256 manifest in `siliquant_verification_receipt.json`. Wave-1A HF receipts pin `siliquant_dequant.py` at SHA-256 `b391f4f3...`; in this tree it is SHA-256 `468daf7261b6d8fb4a2ae435bf39abf6fbc85f02da94e35806531037ff29f794` (git blob `10b0c1678a26e5cfe9b45beec26285c74f598092`): verifier hardening (ignores HF-download `.gitattributes`, `.cache/huggingface/` and `__pycache__`) plus docstrings; the unpack/wire format is unchanged.

## Mathematical Container (`siliquant-hf-v1`)

Every quantized 16-weight block $y \in A_{16} = 2\mathbb{Z}^{16} + C_{[16, 11, 4]}$ is decomposed exactly as $y = 2k + c$ with $c = y \bmod 2 \in C_{[16, 11, 4]}$ and $k = (y - c)/2 \in \mathbb{Z}^{16}$:

- **`qham11` (`uint8`, `0.6875 bpw`):** Packs the 11 systematic information bits at positions `(3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15)` (8 blocks = 88 bits = 11 bytes exact), algebraically reconstructing the 5 parity bits at `{0, 1, 2, 4, 8}`. Unused tail padding bits are fail-closed verified zero.
- **`qquot` (`uint8`) + CSR Overflow (`qov_rowptr`, `qov_col`, `qov_val`):** Auto-selects per-layer minimum-byte width `k_bits in {2, 3, 4}` for $k$, storing rare tail entries in strictly validated CSR overflow buffers (rejecting non-monotonic `rowptr`, negative/out-of-bounds/duplicate columns, partial CSR buffers, and shape mismatches). Reconstructed in `int32` (`(k << 1) + c`) with zero `int16` shift wrap-around.
- **Dtype-Immune Scale Buffers (`scales_fp16_bits: int16`, optional `colscale_fp32_bits: int32`):** Exact IEEE-754 bit patterns stored as integer buffers so `.to(torch.bfloat16)` / `from_pretrained(..., dtype="auto")` never truncates FP16/FP32 scale mantissas (`0.0` max abs diff across `bfloat16`, `float16`, and `float32`).
- **Tamper Integrity & Upstream License Enforcement:** `qham11` syndrome verification enforces codec/lattice consistency, while end-to-end payload tamper integrity across `qquot`, `qham11`, scales, and metadata is verified by `siliquant_verification_receipt.json` SHA-256 manifests (rejecting tampered, missing, unlisted, symlinked, or path-traversal entries). Upstream licenses are enforced fail-closed (including `license: other`, `license_name: qwen-research` with exact upstream `license_link` for `Qwen/Qwen2.5-3B`, never `apache-2.0`).

## Bundle Contents (`8` Verified Source Artifacts + This README)

- `configuration_siliquant.py` — `SiliquantConfig(PretrainedConfig)` wrapping `Qwen2Config` and `LlamaConfig` via `auto_map`.
- `modeling_siliquant.py` — Vectorized pure-PyTorch `A_16` bit-packing codec (`pack_a16_tensor`, `unpack_a16_integers`, `dequantize_a16_weight`), `SiliquantLinear` (with `load_state_dict` cache invalidation and strict `perm_in`/CSR/finiteness validation), and `SiliquantForCausalLM` (with constrained `model.layers.*` `nn.Linear` replacement).
- `export_siliquant_hf.py` — Exact `A_16` L2-ball quantizer (`4.5 bpw` & `3.0 bpw`, `g=128` & `g=32`, `axis`/`block_h2`/`block_h4`/`h16`/`projtype_uniform`, `quad_lpt`/`quad_snake`/`quad_lpt_down_wcol` offline fold), bit-exact `safetensors` exporter (round-trip max abs diff `0.0`), Model Card generator, and SHA-256 receipt writer.
- `siliquant_dequant.py` — Standalone CLI (`inspect`, `verify`, `unpack`) to verify SHA-256 receipts & `A_16` syndromes or export unpacked dense BF16/FP16/FP32 `safetensors`.
- `verify_siliquant_hf.py` — Fail-closed cold-reload verifier (`0` bare `assert`s) checking SHA-256 manifest, privacy/secret hygiene, duplicate-key-resistant YAML license compliance, `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` load, unpack codec consistency (`0` syndrome errors), `bf16`/`fp16`/`fp32` dtype immunity, live `.generate()` + KV-cache step parity, and optional receipt-metadata benchmark cross-checks.
- `test_siliquant_hf_exporter.py` — Comprehensive CPU-only unit, adversarial regression, end-to-end synthetic `Qwen2`/`Llama` cold-reload, and ASan/UBSan AVX-512 test suite (`19` test methods).
- `kernels/sm86_a16_blockh4.cu` — Standalone reference zero-spill Ampere `sm_86` PTX fused fixed-4-bit nibble-packed `A_16` + Block-H4 `dp4a.s32.s32` GEMV kernel (compile-verified via `nvcc -arch=sm_86 --ptxas-options=-v`: `0 B` stack frame, `0 B` spill, `PTX_LDL=0`, `PTX_STL=0`; includes an on-device `main()` self-test for `sm_86` hardware).
- `kernels/avx512_a16_blockh4.c` — Standalone reference Intel AVX-512 (`vpermb` + `vpopcntq` + `vpdpbusd` VNNI) fixed-4-bit nibble-packed `A_16` + Block-H4 GEMV decoder with scalar tail handling, live-executed and verified ASan/UBSan-clean across `K=1024`, `K=960`, and `K=112`.

## Quick Verification

Run from inside `studies/s74-siliquant-hf-v1-publisher/`:

```bash
CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -I -B test_siliquant_hf_exporter.py -v
CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -I -O -B test_siliquant_hf_exporter.py -v
CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -I -B -W error test_siliquant_hf_exporter.py -v
```

Note: the end-to-end exact-logits check (`max_abs_logits_diff == 0.0`) can flake under heavy CPU contention; run the suite with `OMP_NUM_THREADS=1` on an idle host.
