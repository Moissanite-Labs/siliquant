# RTX 3080 `sm_86` `A_16` / `Block-H2` / `18B` & `72B` Wire-Format CUDA & Anisotropic Calibration Study

Live bare-metal NVIDIA RTX 3080 Laptop (`GA104M`, `sm_86`, `110W` AC `P0`) CUDA & PyTorch empirical study accompanying `Moissanite-Labs/siliquant` (`#58`, `#77`, `#80`), paired with a pure-$\mathbb{Z}/\mathbb{Q}$ zero-float verifier and deterministic certificate (`measured_live_model_weights: false`).

## Highlights

1. **Zero-Spill `flip_mask` Decode (`24/24` `sm_86` Kernels, `0 B` Stack / `0 B` Spills):**
   - Branchless register-resident `flip_mask` `[16,11,4]` syndrome decoding reaches `178.19 Gweights/s` (`356.39 GB/s` = `99.95%` of STREAM Triad) standalone with `0` syndrome errors (`+21.06%` over spill-bound `v[syn]` indexing).
2. **`4.5000 bpw` Wire-Format Roofline Shootout (`1,240x` Serving Cliff):**
   - Exact GPU implementations of `siliquant#77` (`49283ff`) `18B` `hampack` (`30.62 MB` DP table = `7.30x` L2) and `72B` `hammingindex` (`24.38 MB` compressed table = `5.81x` L2) achieve `3.73 Gweights/s` and `0.49 Gweights/s` fused GEMV (`0` bit mismatches, `0` syndrome errors, CPU FP64 `rel_l2 < 1.9e-7`).
   - Direct `72`-byte / `128`-weight SoA pre-encoded `A_16` + 5-`__popc` parity verify + native `__dp4a` (`A16_axis_preencoded_dp4a`) reaches **`611.21 Gweights/s` (`344.13 GB/s` = `96.51%` STREAM Triad)** at `4096x14336` — `163.7x` faster than GPU `18B` GEMV and `1,240x` faster than GPU `72B` GEMV, matching `Block_H2_dp4a_int8` (`609.07 Gweights/s`, `96.17%` Triad).
   - **Serving Architecture Recommendation:** Keep spherical-ball `18B`/`72B` encoding for cold/disk storage and perform a one-time load-time transcode to VRAM SoA nibbles for `96.5%`-Triad GPU inference.
3. **Exact `f64` Euclidean-Ball Refinement for `72B` `A_16^8`:**
   - Although `72B` `hammingindex` (`bound=5213`) reaches single-coordinate magnitude `72 > 64` (still signed `int8_t`-safe `<= 127`), the even-norm theorem (`<= 5212`) + Cauchy–Schwarz caps `||v||_1 <= 816`, proving an exact `10.04x` `f64` exact-integer safety margin (`41.82x` for `18B` `hampack`).
4. **5-Seed Anisotropy Sign Flip & `colscale` Calibration Verification (`measured_live_model_weights: false`):**
   - At exact `0.000000 dB` matched weight MSE, Regime I (coherent block-DC) favors `A16_axis` over `A16_H16` by `3.71x` (`5/5` seeds) while Regime II (sparse axis-aligned spikes) reverses cleanly (`-66.9%`, `5/5` seeds).
   - Experiment 3B (`bound=5213` L2-ball `colscale`): `hA16_real < hA16_rand < plain < hA16_colproxy` on `5/5` seeds (`P5 = 27.01% < 70%`, data-free `1/col_rms^2` proxy `-69.41%` retention, Spearman `-0.182` vs. `+0.335`).

## Reproduce Verification (Zero External Dependencies)

```bash
python3 -I -B studies/rtx3080-siliquant-a16/rtx3080_siliquant_a16_audit.py --check studies/rtx3080-siliquant-a16/rtx3080_siliquant_a16_certificate.json
python3 -I -B studies/rtx3080-siliquant-a16/rtx3080_siliquant_a16_audit.py --verify-receipts studies/rtx3080-siliquant-a16
python3 -I -B studies/rtx3080-siliquant-a16/test_rtx3080_siliquant_a16_r5.py
```

## File Manifest

| File | Lines | Bytes | Mode | SHA-256 |
|---|---:|---:|---:|---|
| `rtx3080_siliquant_a16_kernels.cu` | `2074` | `91268` | `0o644` | `0bd8df2e79f64ea2bed48c83ee11b6b597b816d458228a0f2420e2b3092f593d` |
| `rtx3080_siliquant_a16_benchmark.py` | `916` | `41064` | `0o644` | `b8ff97bdd9ba8cff172a05b10498404abb2e95f9716b84b620ae8a5514149f3b` |
| `rtx3080_siliquant_a16_empirical_results.json` | `914` | `28722` | `0o644` | `62fff8638cae3158fa3394a4c8be24b8827c7251be5c6ebb1d481bc8acbb13dd` |
| `mlos_zone_b_receipt.json` | `91` | `2627` | `0o600` | `d79ccc851ba80e33fb1b0db21c7eb8998eb1a3563d28e6304be8e1e307f68738` |
| `rtx3080_siliquant_a16_oracle.py` | `630` | `27362` | `0o644` | `c5a9383a56d970f99f1b2352d2f30ce2aaa779910937dc6f982e07dd7d2cf4de` |
| `rtx3080_siliquant_a16_audit.py` | `469` | `19013` | `0o644` | `9786f8e4bf0a7b4a8daa685e96487327a25e4dd692508b3f9f7a69c4aa4cc7a3` |
| `rtx3080_siliquant_a16_certificate.json` | `1` | `4136` | `0o600` | `3a3ed2ac2a6bcb59bd215ffe0abc48f5651adffc19c80d620ceb32d7c0dac789` |
| `test_rtx3080_siliquant_a16_r5.py` | `799` | `40727` | `0o644` | `aab15f6b837304153c6a68f820a01afc4528c190a25a06b23dc7a1419265a403` |
| `RTX3080_SILIQUANT_A16_REPORT.md` | `159` | `21884` | `0o644` | `d64a1f543ae16bb75d1fbd42a358a323d25998d86993a2543cf4dd3e00d92dfa` |
