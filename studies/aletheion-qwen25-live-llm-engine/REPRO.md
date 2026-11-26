> Reviewed by grk-tensor 7:45 PM CT: spot-checked bench.py:1489-1498, 751-752 and report L28/L33 against main ba6213a1, and they match. Two fixes made: nvcc takes `-Xcompiler -fPIC`, not a bare `-fPIC`, and the box CPU-flag note was wrong.

# REPRO — ALETHEION_LIVE_LLM_ENGINE_REPORT.md §3.2 on RTX 3080 Laptop (sm_86)

Source: a `siliquant` checkout of main `ba6213a1`, study
`studies/aletheion-qwen25-live-llm-engine`, added by commit `67c7706` (= PR #88,
"studies: add live Qwen2.5-0.5B A_16 PTX/SASS vs AVX-512 silicon engine",
`git show --stat 67c7706`: 12 files, 6,656 insertions). Working copy used for all
on-box runs: a scratch copy of the study directory (source checkout untouched).

Targets to reproduce (§3.2, report lines 53-63):
- **Arm B1** `sm_86` PTX `A_16` + BF16 `lm_head` (full E2E decode): **400.580 tok/s**, 2.49638 ms/tok
- **Arm A** PyTorch BF16 (full E2E decode): **71.576 tok/s**, 13.97115 ms/tok

Command marks: **VERIFIED-ON-BOX** = ran successfully on grk-tensor's check machine (gcc, no nvcc, no GPU);
**STATIC** = derived from code with file:line citation, not run; **NEEDS-SM86** = only
confirmable on an sm_86 GPU machine.

---

## 1. Prerequisites

| Item | Value | Source |
|---|---|---|
| GPU | NVIDIA GeForce RTX 3080 Laptop (GA104M, sm_86 / cc 8.6, 8 GiB) | `mlos_zone_b_receipt.json:31,79` (`driver_version`), `qwen25_live_engine_empirical_results.json` `hardware_telemetry_method` (cc "8.6", gpu_name) |
| Driver | 615.78.08 | `mlos_zone_b_receipt.json:31,79` |
| OS (recorded) | Linux 7.2.15-1-omarchy | `mlos_zone_b_receipt.json:58` |
| Python (recorded) | 3.14.8 | `mlos_zone_b_receipt.json:59` |
| CUDA toolkit | **not recorded anywhere** — GAP-2 | — |
| nvcc | **path not specified in study**; sibling study defaults to `/opt/cuda/bin/nvcc` (`studies/rtx3080-live-llm-a16-eval/rtx3080_live_llm_eval.py:909`) — GAP-2 | — |
| Python deps | `numpy` (bench.py:36), `torch` (bench.py:286,548,746), `transformers` (bench.py:748-749), `safetensors` (bench.py:532); versions **not recorded** — GAP-3. No pynvml: NVML via `ctypes.CDLL("libnvidia-ml.so.1")` (bench.py:144) | — |
| CPU energy | readable `/sys/class/powercap/intel-rapl:0{,:0,:1}` and `:1` `energy_uj` (bench.py:97-100,104-117) — needs root/powercap perms; unreadable domains are silently skipped (bench.py:114-117) | STATIC |
| GPU energy | NVML `nvmlDeviceGetPowerUsage` sampled at 100 Hz, trapezoidal integration (bench.py:141,144,156-224); `nvmlDeviceGetTotalEnergyConsumption` deliberately rejected (bench.py:135-138) | STATIC |
| `nvidia-smi` | not used by the bench; only appears inside the external `mlos` wrapper that produced the receipt | STATIC |
| Ring-0 `mlos` | provenance wrapper only; not invoked by bench/audit code — GAP-6 | `mlos_zone_b_receipt.json`, `aletheion_live_llm_audit.py:720-762` |

Offline/determinism env pins used for the recorded run (from the mlos receipt,
`mlos_zone_b_receipt.json:6-18`): `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `TORCH_USE_DETERMINISTIC_ALGORITHMS=1`,
`PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`. The bench itself loads the model
with `local_files_only=True` (bench.py:765, 783), so a pre-populated HF cache is enough.

## 2. Model download

Repo id `Qwen/Qwen2.5-0.5B`, pinned snapshot `060db6499f32faf8b98477b0a26969ef7d8b9987`
(`qwen25_live_engine_empirical_results.json` `model_identity`; report line 4).
Expected SHA-256 (`aletheion_live_llm_oracle.py:45-61` `PINNED_SHA256`):
`model.safetensors` = `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`,
`config.json` = `479dcf0c5286339e41ad3992cd08ae88a467c4187587936248e2b7c96283484b`,
`tokenizer.json` = `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`.

The download command is **not written anywhere in the repo** (no `huggingface-cli` /
`snapshot_download` in code — GAP-4). Inferred, copy-pasteable:

```bash
# STATIC (inferred — command itself is a GAP; repo id/snapshot/sha pins are STATIC from oracle.py:45-61)
huggingface-cli download Qwen/Qwen2.5-0.5B --revision 060db6499f32faf8b98477b0a26969ef7d8b9987
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export MODEL_DIR="$HF_HOME/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
sha256sum "$MODEL_DIR/model.safetensors"   # must equal 88c14255...ed342
```

The bench takes `--model-dir` = that snapshot dir; it requires `model.safetensors`,
`config.json`, `tokenizer.json` inside it, hashes them (bench.py:758-763), records
`"snapshot_id": model_dir.name` (bench.py:1416), and asserts the tensor inventory
168 linear matrices / 357,826,560 weights / 122 non-block BF16 tensors
(bench.py:788-798). VERIFIED-ON-BOX: `TECH_PROMPT` SHA-256 recomputed =
`a79058dd...6b8c`, matches `aletheion_live_llm_oracle.py:50` (bench.py:70-74).

## 3. Build commands for the two engines

The bench does **not** build anything: it takes prebuilt shared objects
`--sm86-so`, `--avx512-so` (bench.py:1492-1493) and loads them with
`ctypes.CDLL` (bench.py:388-389, 461-462). The only compiler flags written in the
study are in the report (STATIC):

- sm_86 engine — report line 28: `nvcc -O3 -std=c++17 -arch=sm_86 --ptxas-options=-v`
  plus `cuobjdump -sass` for the SASS receipt (`sm86_engine_sass_inspection.json`, 16 kernels).
- AVX-512 engine — report line 33:
  `gcc -O3 -march=native -mavx512f -mavx512bw -mavx512vl -mavx512vbmi -mavx512vpopcntdq -mavx512vnni -fopenmp`
  (`avx512_engine_isa_inspection.json`).

GAP-1: neither the report nor any script writes the `-shared -fPIC -o *.so` link
step (sibling study `rtx3080_live_llm_eval.py:851` builds an *executable* with the same
nvcc flags). The `.so` form below adds `-shared -fPIC -o … -lm`, which the ctypes loader
requires:

```bash
# NEEDS-SM86 (grk-tensor's check machine (no GPU) has no nvcc; flags STATIC from report line 28, -shared/-o are GAP-1)
nvcc -O3 -std=c++17 -arch=sm_86 --ptxas-options=-v -shared -Xcompiler -fPIC \
  qwen25_sm86_a16_engine.cu -o libqwen25_sm86_a16_engine.so
cuobjdump -sass libqwen25_sm86_a16_engine.so   # expect 16 kernels, 0 spill/LDL/STL

# VERIFIED-ON-BOX (compile only, gcc 14.2.0, rc=0, 0 warnings, exports present;
#  NOT executed. (Correction by grk-tensor: on grk-tensor's check machine (no GPU), /proc/cpuinfo DOES list avx512_vnni and avx512_vpopcntdq; it was simply not run.)
#  flags STATIC from report line 33; -shared -fPIC -o -lm are GAP-1)
gcc -O3 -march=native -mavx512f -mavx512bw -mavx512vl -mavx512vbmi -mavx512vpopcntdq \
  -mavx512vnni -fopenmp -shared -fPIC \
  qwen25_avx512_a16_engine.c -o libqwen25_avx512_a16_engine.so -lm
```

On-box compile check produced `/tmp/qwen25_avx512_a16_engine.so` (29,280 B) exporting
`qwen25_cpu_create/destroy/load_globals/load_layer_norms_biases/load_proj/reset_kv/
get_syn_errors/forward_token/bench_decode_sustained` (matches `load_avx512_lib`,
bench.py:461-506); `objdump -d` shows `vpermb`/`vpopcntq`/`vpdpbusd` emitted (counts
differ from `avx512_engine_isa_inspection.json` because this gcc/target differs — the
receipt values are only reproducible on the i9-11900H-class target, NEEDS-SM86-class).

## 4. Bench invocation (Arm B1 and Arm A come from ONE run)

There is **no per-arm flag** (GAP-8): `--phase all` runs Arm A → quantize → Arm D →
Arm B → auto-spawned Arm C subprocess, in that order (bench.py:739-1481). Argparse
(bench.py:1489-1498): `--phase {all,arm_c}` (default `all`), `--model-dir` (required),
`--sm86-so` (required for `all`, bench.py:1516), `--avx512-so` (required),
`--out-json` (required), `--cpu-threads` (default 8), plus `--disk-18b-path`,
`--expected-disk-sha`, `--parity-dir` used only by the self-spawned `--phase arm_c`
child (bench.py:1319-1341, env `CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS`,
`OMP_PROC_BIND=spread`, `OMP_PLACES=cores`, bench.py:1311-1316).

```bash
# NEEDS-SM86 (invocation STATIC from bench.py:1489-1523; run aborts without CUDA at
# bench.py:751-752 "CUDA GPU not available")
python3 aletheion_live_llm_bench.py \
  --phase all \
  --model-dir "$MODEL_DIR" \
  --sm86-so ./libqwen25_sm86_a16_engine.so \
  --avx512-so ./libqwen25_avx512_a16_engine.so \
  --out-json ./qwen25_live_engine_empirical_results_repro.json \
  --cpu-threads 8
```

Files written (STATIC): only `--out-json` (bench.py:1474) plus the pareto summary
printed to stdout (bench.py:1475); ephemeral `s58_qwen25_18b_*.bin` in `/tmp`
(bench.py:988, deleted at :1008) and `s58_armc_18b_*.bin` + `s58_armc_par_*` in
`/dev/shm` (or `/tmp`, bench.py:1290-1306), removed in the `finally` (bench.py:1476-1481).

Expected output fields vs. reported §3.2 values (report lines 53-63;
`qwen25_live_engine_empirical_results.json`; oracle decimals `oracle.py:69,99`):

| JSON path | field | reported |
|---|---|---|
| `arm_b_sm86_ptx_a16_gpu.b1_pure_dp4a.full_e2e_with_bf16_lm_head` | `gpu_event_tok_s` / `gpu_event_ms_per_tok` | **400.580** / 2.49638 (bench.py:1188-1198) |
| same | `gpu_mj_per_unit` / `cpu_pkg_mj_per_unit` / `combined_gpu_plus_cpu_pkg_mj_per_unit` / `psys_mj_per_unit` | 191.40979 / 121.37732 / 312.78711 / 18.12172 (energy: uncontrolled, provisional) |
| `arm_a_pytorch_bf16_gpu.a2_full_e2e_with_bf16_lm_head` | `units_per_s` / `ms_per_unit` | **71.576** / 13.97115 (bench.py:865-868) |
| same | combined mJ/tok | 1535.12265 (energy: uncontrolled, provisional) |
| `arm_b_sm86_ptx_a16_gpu.b1_pure_dp4a.backbone_24layers` | `gpu_event_tok_s` | 571.367 (§3.1) |
| `arm_a_pytorch_bf16_gpu.a1_backbone_24layers` | `units_per_s` | 76.921 (§3.1) |
| `pareto_latency_energy_summary.decode_full_e2e_with_bf16_lm_head` | `speedup_b1_vs_a_bf16` | 5.597 (bench.py:1385) |

Energy rows above (`mJ/tok`): uncontrolled, provisional. They come from one laptop run with one measurement window per arm, no repeats and no reported spread; CPU energy is the `intel-rapl` package counter for the whole CPU. No controlled rerun has been done yet (see the report's §3 note).

## 5. Test / audit / receipt verification (no GPU needed)

```bash
# VERIFIED-ON-BOX (python3 3.13.5): "Ran 6 tests in 2.215s — OK", rc=0
python3 -I -B test_aletheion_live_llm_r5.py          # unittest.main at test file end; imports only oracle+audit (stdlib)

# VERIFIED-ON-BOX: {"valid":true, ... "full_replay":true} rc=0
python3 aletheion_live_llm_audit.py --check aletheion_live_llm_certificate.json

# VERIFIED-ON-BOX: {"valid":true,"verified_files":6,...} rc=0
python3 aletheion_live_llm_audit.py --verify-receipts .
```

CLI modes (STATIC, `aletheion_live_llm_audit.py:780-818`): no args / `--standalone --json`
emit the certificate; `--write <path>` (0o600 atomic write, audit.py:379);
`--check <path>`; `--verify-receipts <dir>` re-hashes the 6 receipt files against
`PINNED_SHA256` and cross-checks key numbers as exact `Fraction`s (audit.py:720-762).
`--help` of the bench (VERIFIED-ON-BOX, rc=0, needs only `numpy` importable):
`python3 aletheion_live_llm_bench.py --help`.

## 6. On-box verification log (grk-tensor's check machine (no GPU): gcc 14.2.0, Python 3.13.5, numpy 2.5.5, 8 cores, no nvcc, no GPU; /proc/cpuinfo has avx512f/bw/vl/vbmi/vnni/vpopcntdq (corrected by grk-tensor))

| Check | Result | Mark |
|---|---|---|
| `python3 -m py_compile` on all 4 `.py` | OK | VERIFIED-ON-BOX |
| `python3 aletheion_live_llm_bench.py --help` | rc=0, usage printed | VERIFIED-ON-BOX |
| `audit.py --check aletheion_live_llm_certificate.json` | `valid:true`, full_replay, rc=0 | VERIFIED-ON-BOX |
| `audit.py --verify-receipts .` | `valid:true`, 6 files, rc=0 | VERIFIED-ON-BOX |
| `python3 -I -B test_aletheion_live_llm_r5.py` | 6 tests OK in 2.215s, rc=0 | VERIFIED-ON-BOX |
| `TECH_PROMPT` sha256 recompute | matches pin `a79058dd…` (oracle.py:50) | VERIFIED-ON-BOX |
| gcc compile of `qwen25_avx512_a16_engine.c` with report flags + `-shared -fPIC -lm` | rc=0, `.so` built, exports present; **not executed** | VERIFIED-ON-BOX (compile only) |
| nvcc build of `.cu`; running the bench | not possible here | NEEDS-SM86 |

## 7. GAPS (what the code does not define)

1. **GAP-1 — no `.so` link step**: build flags exist only as report prose (report lines 28, 33); no script emits `-shared -fPIC -o *.so`. Bench requires prebuilt `.so` (bench.py:1492-1493, 388-389, 461-462). Reader must add the link flags.
2. **GAP-2 — CUDA toolkit version and nvcc path unrecorded**; only sibling default `/opt/cuda/bin/nvcc` (`rtx3080_live_llm_eval.py:909`). Driver recorded: 615.78.08 (`mlos_zone_b_receipt.json:31`).
3. **GAP-3 — Python package versions unrecorded**; only Python 3.14.8 (`mlos_zone_b_receipt.json:59`). Imports: numpy/torch/transformers/safetensors (bench.py:36,286,532,746-749).
4. **GAP-4 — model download command/cache path not in code**; only repo id + snapshot + sha pins (oracle.py:45-61; report line 4). `--model-dir` must be the snapshot dir; `snapshot_id = model_dir.name` (bench.py:1416); `local_files_only=True` (bench.py:765,783); offline pins in receipt (`mlos_zone_b_receipt.json:6,13`).
5. **GAP-5 — RAPL access**: `/sys/class/powercap/intel-rapl:*` must be readable or CPU mJ/tok silently reads 0 (bench.py:97-117). NVML soname hardcoded `libnvidia-ml.so.1` (bench.py:144).
6. **GAP-6 — Ring-0 `mlos` tool and profile `ux582hs-rtx3080-8gb.toml` not in repo**; the bench never calls mlos — the receipt is only checked as a file (`audit.py:720-762`, `mlos_zone_b_receipt.json`; profile_sha256 `566c9192…`, receipt_sha256 `208e14b2…` in oracle.py:59-60). `nvidia-smi` appears only inside mlos telemetry, not in bench code.
7. **GAP-7 — hardcoded temp paths**: `/tmp` for the cold 18 B file (bench.py:988), `/dev/shm`-or-`/tmp` for the Arm C handoff (bench.py:1290-1292).
8. **GAP-8 — no per-arm CLI flag**: B1/A numbers require the full `--phase all` run (both `.so` files mandatory); `--phase arm_c` alone is not runnable without the parent-provided `--disk-18b-path`/`--parity-dir` (bench.py:1501-1506). Arm C requires the GPU to reach idle P8 ≤ 20 W first (bench.py:508-517, 596, plus `time.sleep(1.5)` at :1310).
9. **GAP-9 — energy method**: GPU mJ/tok uses 100 Hz trapezoidal `nvmlDeviceGetPowerUsage` (bench.py:141,156-224), NOT `nvmlDeviceGetTotalEnergyConsumption` (rejected, bench.py:135-138); CPU uses RAPL microjoule deltas (bench.py:119-131, 236-249).

## 8. One-line answer

**No — the B1 400.580 tok/s (§3.2) command cannot be confirmed without an sm_86 GPU:** the bench hard-fails without CUDA (`bench.py:751-752`), the `qwen25_sm86_a16_engine.cu` `.so` needs nvcc `-arch=sm_86` (absent here; report line 28), and grk-tensor's check machine (no GPU) has no GPU at all, so every B1/A number in §3.2 is NEEDS-SM86 (only the Python checks, audit/receipt verification, and the AVX-512 compile were VERIFIED-ON-BOX).
