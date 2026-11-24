# Live RTX 3080 `sm_86` `110W` AC CUDA & PyTorch Empirical Study of $A_{16}$ Wire Formats, `Block-H2`, and Anisotropic Calibration for `Moissanite-Labs/siliquant` (`#58` / `#77` / `#80`)

## Epistemic Taxonomy & Hardware Provenance

Every claim in this report is explicitly classified into one of three non-overlapping epistemic tiers:

1. **`exact` (Pure-$\mathbb{Z}/\mathbb{Q}$ Certified Mathematics):** Extended Hamming `[16,11,4]` and `[8,4,4]` codeword enumerations and bitmask bijections, `Block-H2` vs. `WHT_4`/`WHT_8`/`WHT_16` `INT8` dynamic-range separation, exact `f64` accumulator bounds (both $L_\infty$ hypercube and $L_2$ Euclidean-ball Cauchy–Schwarz refinements for `18B`/`72B` wire formats), `2×` parity-state / even-norm compression proofs, and exact `RTX 3080` `GA104M` theoretical GDDR6 roofline arithmetic. Verified in `rtx3080_siliquant_a16_oracle.py` and `rtx3080_siliquant_a16_audit.py` with zero floating-point literals, zero `float()` casts, and zero bare `assert`s.
2. **`sampled` (Live RTX 3080 `sm_86` Empirical Silicon Measurements — `measured_live_model_weights: false`):** Bare-metal `sm_86` CUDA roofline, standalone decode, `7` direct `4.5000 bpw` fused GEMV arms, exact `siliquant#77` (`49283ff`) `18`-byte `hampack` & `72`-byte `hammingindex` enumerative GPU decoders, `PyTorch 2.15.0+cu132` Tensor Core GEMMs, and `5`-seed (`2026112401..2026112405`) anisotropic Hessian & Fisher-calibration shootouts on synthetic heavy-tailed Transformer matrices with strict calibration/held-out evaluation splits.
3. **`anchor` (Pinned Upstream & Hardware Reference Baselines):** First-light `~55W` battery probe (`STREAM Triad = 356.58 GB/s`, `BF16 = 31.88 TFLOPS`, `A_16` standalone with local-memory spill `0.4559 ms` / `147.19 Gweights/s`), `Moissanite-Labs/siliquant#77` (`49283ff`) 1-core CPU `hampack.c` decode throughput (`9.4–10.1 M weights/s`), and `Moissanite-Labs/siliquant#58` Gemma 3 `12B` (`ba5ec5a6`) unconfounded `P5` (`53.74% < 70%`) and `HBG128` vs. `EBG128` (`+4.24%` Wiki KL at `0.00037 dB` MSE parity) measurements.

### Hardware & Ring-0 `mlos` Zone-B Stamp (`sampled`)

| Telemetry Field | Measured Value (`mlos_zone_b_receipt.json`) |
|---|---|
| GPU Silicon / Bus | `NVIDIA GeForce RTX 3080 Laptop GPU` (`GA104M`, `CC 8.6` `sm_86`, `48` SMs, `100 KiB` shmem/SM, `4 MiB` L2, `8,192 MiB` GDDR6, `256`-bit @ `6001 MHz`) |
| Exact Peak Bandwidth (`exact`) | `48008 / 125 = 384.0640 GB/s` (`4.50 bpw` weight ceiling `768128 / 1125 = 682.7804 Gweights/s`) |
| Driver / VBIOS / Toolchain | NVIDIA `615.78.08`, VBIOS `94.04.49.00.51`, `CUDA 13.4` (`nvcc -O3 -arch=sm_86`), `PyTorch 2.15.0+cu132`, `NumPy 2.5.5`, Linux `7.2.15-1-omarchy` |
| Ring-0 `mlos` Verdict | `clean_hardware_run: true`, `anomalies: []`, `determinism_zone: "B"`, `exit_code: 0`, `duration_ms: 17217.833` (`2026-11-24T12:29:40Z..12:29:57Z`) |
| Desktop Isolation & Thermal | `gpu_processes_queried: true`, `graphics_processes: []`, `linux_kmd_baseline_vram_mib: 2` (`<= 4 MiB`), end state `P0`, `6001 MHz` mem, `1710 MHz` SM, `65 °C`, `thermal_throttled: false`, `slowdown_active: false` |
| `mlos` Receipt SHA-256 | `de1a1b18b65497c629cfa6116eb0be70d4fcdf399e8c2cb2d157b005c051102d` (file SHA-256 `d79ccc851ba80e33fb1b0db21c7eb8998eb1a3563d28e6304be8e1e307f68738`) |
| `ptxas -v` Register/Spill Audit | **`24 / 24` compiled `sm_86` kernels: `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`** (`15–78` registers) |

---

## Executive Summary of Key Findings

1. **Zero-Spill `flip_mask` Microarchitectural Breakthrough (`+21.06%` Standalone Decode, `0` Spills Across `24/24` Kernels — `sampled` + `anchor`):** Replacing dynamic local-array indexing `v[syn]` (which forced `nvcc` local-memory register spills in the first-light probe at `0.4559 ms` / `147.19 Gweights/s`) with a branchless register-resident `flip_mask` bitmask yields **`0` spill bytes across all `24/24` compiled `sm_86` kernels**, lifting standalone `67.1M`-weight `A_16` `[16,11,4]` syndrome decode to **`0.376606 ms` (`178.19 Gweights/s`, `356.39 GB/s` = `99.95%` of STREAM Triad)** — matching `Block-H2` (`178.21 Gweights/s`) and `WHT_8` (`178.20 Gweights/s`) identically with `0` syndrome errors.
2. **`siliquant#77` (`49283ff`) `4.5000 bpw` Wire-Format Shootout — The `1,240×` Serving Cliff (`sampled` + `anchor`):** Implementing `@orda-aska`'s exact `18`-byte `hampack.py` (`bound=1207`, `30.62 MB` DP table) and exact `72`-byte `hammingindex.py` (`bound=5213`, `24.38 MB` parity/even-compressed `shell`+`after`) bit-for-bit on RTX 3080 (`0` bit mismatches, `0` syndrome errors, CPU FP64 `rel_l2 < 1.9e-7`) proves:
   - GPU `hampack_18B` reaches **`1.65–1.91 Gweights/s`** standalone (`165–191×` over `siliquant#77`'s `9.4–10.1 M/s` 1-core CPU) and **`3.73–4.16 Gweights/s`** fused GEMV (`0.59%–0.66%` Triad), while GPU `hammingindex_72B` reaches **`0.29–0.30 Gweights/s`** standalone and **`0.49–0.55 Gweights/s`** fused GEMV (`0.08%–0.09%` Triad) — fundamentally throttled by DP tables `5.81×`–`7.30×` larger than RTX 3080's `4 MiB` L2 cache plus serial multi-precision unranking.
   - In contrast, **direct `72`-byte / `128`-weight SoA pre-encoded `A_16` + 5-`__popc` hardware parity verify + native `__dp4a` (`A16_axis_preencoded_dp4a`)** reaches **`611.21 Gweights/s` (`344.13 GB/s` = `96.51%` of STREAM Triad)** at `4096×14336` (`576.50 Gweights/s` / `91.14%` Triad at `4096×4096`) — **`163.7×` faster than GPU `hampack_18B` GEMV**, **`1,240×` faster than GPU `hammingindex_72B` GEMV**, **`2.73×` faster than weight-side `A16_H16`**, **`2.82×` faster than `E8_WHT8`**, and **`~60,500×` faster than 1-core CPU `hampack.c`**, matching `Block_H2_dp4a_int8` (`609.07 Gweights/s`, `96.17%` Triad).
   - **Actionable Recommendation for `siliquant#77`:** Separate **cold/disk archive storage** (`18B`/`72B` enumerative spherical balls for peak KL/byte on disk) from **VRAM-resident serving layout** (one-time load-time transcode into direct `72B` pre-encoded `A_16` or `Block-H2` SoA nibbles for `96.5%`-Triad GPU inference).
3. **Exact `f64` Euclidean-Ball Refinement for `72B` `A_16^8` (`exact`):** Although `72B` `hammingindex` (`bound=5213`) admits single-coordinate magnitudes up to $\lfloor\sqrt{5213}\rfloor = 72 > 64$ (exceeding the hypercube $L_\infty$ threshold $W_{\max} \le 64$, while remaining signed `int8_t`-safe since $72 \le 127$), every point of $A_{16}^8$ has **even** squared norm ($\le 5212$), so Cauchy–Schwarz gives $\sum_{i=0}^{127} |v_i| \le \lfloor\sqrt{128 \cdot 5212}\rfloor = 816 \ll 8192$ — certifying an exact **`10.04×` (`2^53 / (816 * 65504 * 2^24)`) `f64` accumulator safety margin** (`41.82×` for `18B` `hampack` where $\max |v| = 34$, $\max \|v\|_1 \le 196$).
4. **5-Seed Anisotropy Sign Flip & Fisher Calibration Verification (`sampled`):** Across all `5/5` logged seeds (`0` syndrome errors):
   - **Regime I (Coherent Block-DC Subspace, mirroring Gemma 3 `12B` `HBG > EBG`):** At exact `0.000000 dB` matched weight MSE, `A16_axis` beats `A16_H16` by **`3.71×`** (`0.1090` vs. `0.4041`) and `E8_axis` beats `E8_H8` by **`2.08×`** (`0.0856` vs. `0.1782`), with `Block-H2` staying close to axis (`0.1293` / `0.1006`).
   - **Regime II (Sparse Axis-Aligned Spikes, mirroring Gemma 3 `270M` `HBG < EBG`):** At exact `0.000000 dB` matched weight MSE, the sign reverses `5/5`: `A16_H16` beats `A16_axis` by **`-66.9%`** (`0.3721` vs. `1.1244`) and `E8_H8` beats `E8_axis` by **`-53.5%`** (`0.3445` vs. `0.7406`).
   - **Experiment 3B (`bound=5213` L2-Ball `colscale` Calibration):** Strict ordering **`hA16_real` (`0.002401`, `-32.3%`) < `hA16_rand` (`0.003239`) < `plain` (`0.003549`) < `hA16_colproxy` (`0.004345`, `+22.4%`)** holds on `5/5` seeds, with `P5` random-token retention **`27.01% < 70%`**, data-free `1/col_rms^2` proxy retention **`-69.41%`** (actively degrading KL!), and Spearman sign flip **`-0.182` vs. `+0.335`**.

---

## Section 1 (`exact`) — Pure-$\mathbb{Z}/\mathbb{Q}$ Lattice, Accumulator & Wire-Format Invariants

All values below are deterministically proved from first principles in `rtx3080_siliquant_a16_oracle.py` (`0` floating-point operations):

| Invariant / Parameter | Exact Pure-$\mathbb{Z}/\mathbb{Q}$ Value | Significance |
|---|---|---|
| `[16,11,4]` Extended Hamming Code | `2,048` codewords, $d_{\min} = 4$, all-even weight (`A_0=1, A_4=140, A_6=448, A_8=870, A_10=448, A_12=140, A_16=1`), kissing number `140*16 + 16*2 = 2,272` | Proves `16/16` single-flip syndrome bijection (`0xFFFF/0xAAAA/0xCCCC/0xF0F0/0xFF00`) and guarantees every $A_{16}$ point has **even** squared norm |
| `[8,4,4]` Extended Hamming Code ($A_8 \cong E_8$) | `16` codewords, $d_{\min} = 4$, kissing number `14*16 + 8*2 = 240`, `8/8` single-flip bijection | Exact baseline parity for `E8_WHT8` |
| `INT8` Dynamic Range (`|x| <= 39`) | `Block-H2` unscaled peak `78 <= 127` (**SAFE**); `WHT_4 = 156`, `WHT_8 = 312`, `WHT_16 = 624` (**ALL OVERFLOW `INT8`**) | Explains why `Block-H2` runs natively on `sm_86` INT8 `__dp4a` without runtime rescaling |
| `f64` $L_\infty$ Hypercube Boundary (`128` wts × `f16`) | `W_max = 64`: `9,002,801,208,229,888 <= 2^53` (**SAFE**); `W_max = 65`: `9,143,469,977,108,480 > 2^53` (**EXCEEDS**) | Exact `#80` (`studies/hamming-construction-a-lattices`) hypercube cutoff |
| `18B` `hampack` (`bound = 1207`, `n = 32`) | `count = 85249697968064323459110312223931592722113` (`136` bits `<= 2^136`); `max |v| = 34`, `max ||v||_1 <= 196`, `f64` margin **`41.82×`**; table `30,615,552 B` = **`7.30×` L2** | Matches `shape/hampack.py` (`49283ff`) bit-for-bit; proves DP table exceeds `4 MiB` L2 by `7.3×` |
| `72B` `hammingindex` (`bound = 5213`, `group = 128`) | `count` `560` bits (`0.9767 * 2^560`); `max |v| = 72` (`> 64`, `<= 127` INT8); even ball `<= 5212` -> `max ||v||_1 <= 816`, **`f64` margin `10.04×`**; compressed table `22,691,328 + 1,689,336 = 24,380,664 B` = **`5.81×` L2** | Proves exact `2×` parity compression (`s>>4 == t&1`) + `2×` even-norm compression and proves `72B` ball remains `f64`-exact despite `max |v| = 72 > 64` |
| `4.5000 bpw` Exact Packing | `18 B / 32 = 72 B / 128 = 9/2 bpw`; `4096×4096 = 9,437,184 B`; `4096×14336 = 33,030,144 B` | Exact byte parity across direct SoA, `18B` `hampack`, and `72B` `hammingindex` |

---

## Section 2 (`sampled` + `anchor`) — RTX 3080 `sm_86` `110W` AC `P0` Roofline vs. `55W` Battery Anchor

| Benchmark | `55W` Battery Anchor (`anchor`) | `110W` AC `P0` Live (`sampled`) | Exact Ratio / Delta | Live Telemetry (`sampled`) |
|---|---:|---:|---:|---|
| Bare-Metal `STREAM Triad` (`256 MiB × 3`) | `356.58 GB/s` | **`356.5679 GB/s`** (`92.84%` peak) | `0.99997×` | `6001 MHz` GDDR6, `0` spill |
| Bare-Metal `PCIe Gen4 H2D / D2H` (`128 MiB`) | — | **`12.2908 / 13.1275 GB/s`** | — | Pinned host DMA |
| Bare-Metal `FP32 FMA` / `INT8 __dp4a` ALU | `— / 41.29 TOPS` | **`17.1444 TFLOPS` / `41.2554 TOPS`** | `0.9992×` | `15 / 20` regs, `0` spill |
| `PyTorch BF16` Tensor Core GEMM (`4096³`) | `31.88 TFLOPS` | **`35.0310 TFLOPS`** (`3.923 ms`) | **`+9.88%`** | `1500 MHz` SM, `68.59W`, `66 °C`, `0x4` (`SwPowerCap`) |
| `PyTorch FP16` Tensor Core GEMM (`4096³`) | — | **`33.7220 TFLOPS`** (`4.076 ms`) | — | `1455 MHz` SM, `62.39W`, `67 °C`, `0x4` |
| `PyTorch TF32` Tensor Core GEMM (`4096³`) | — | **`17.0938 TFLOPS`** (`8.040 ms`) | — | `1470 MHz` SM, `76.33W`, `68 °C`, `0x4` |
| `PyTorch INT8` Tensor Core (`_int_mm`, `4096³`) | — | **`22.5759 TOPS`** (`6.088 ms`) | — | `1065 MHz` SM, `88.03W`, `68 °C`, `0x4` |
| `PyTorch FP32` Strict GEMM (`4096³`) | — | **`7.6297 TFLOPS`** (`18.014 ms`) | — | `930 MHz` SM, `89.67W`, `68 °C`, `0x4` (`0` thermal throttle) |
| Standalone `67.1M` `A_16` Decode | `0.4559 ms` (`147.19 Gwt/s`, *spill*) | **`0.376606 ms` (`178.1938 Gwt/s`)** | **`+21.06%`** | `356.39 GB/s` (`99.95%` Triad), `35` regs, `0` spill, `0` syn err |
| Standalone `67.1M` `Block-H2` / `WHT_8` | — | **`178.2062` / `178.2035 Gwt/s`** | `1.0000×` | `356.41 GB/s`, `24 / 25` regs, `0` spill |

*Note on sustained power telemetry:* On `UX582HS` AC power, sustained dense `4096³` Tensor Core / FP32 GEMMs draw up to `89.67W` (`66–68 °C`, `0` thermal throttling) and register NVML `SwPowerCap` (`0x4`), whereas memory-bound `4.50-bpw` GEMV saturates `96.5%` of GDDR6 bandwidth well below the power envelope (`~58W`, `1710 MHz` SM).

---

## Section 3 (`sampled` + `anchor`) — Complete `4.5000 bpw` Bare-Metal Wire-Format & Fused GEMV Shootout

All `9` formats store weights at **exact `4.5000 bpw` (`9/2` bits/weight)** (`9,437,184 B` at `4096×4096`; `33,030,144 B` at `4096×14336`), compile with **`0` register spills**, record **`0` syndrome errors** and **`0` round-trip bit mismatches**, and pass independent **CPU FP64 reference parity (`rel_l2 < 1.9e-7`)**:

### 3.1 `4096 × 14336` (`58.72M` Weights, `33.03 MB` — MLP Down/Up Shape)

| Arm / Wire Format (`4.5000 bpw`) | Regs / Spill | Latency (`ms`) | Throughput (`Gweights/s`) | Effective BW (`GB/s`) | `%` STREAM Triad | CPU FP64 `rel_l2` | vs. `18B` GPU GEMV |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`A16_axis_preencoded_dp4a`** *(72B SoA + 5-`__popc` + `__dp4a`)* | `38 / 0 B` | **`0.096072`** | **`611.2129`** | **`344.1270`** | **`96.511%`** | `7.20e-8` | **`163.66×`** |
| **`Block_H2_dp4a_int8`** *(72B SoA + `Block-H2` + `__dp4a`)* | `38 / 0 B` | **`0.096410`** | **`609.0706`** | **`342.9209`** | **`96.173%`** | `1.34e-7` | `163.09×` |
| `Block_H2_fp32act` *(72B SoA + `Block-H2` + FP32 FMA)* | `38 / 0 B` | `0.136838` | `429.1222` | `241.9200` | `67.847%` | `1.34e-7` | `114.90×` |
| `A16_axis` *(72B SoA + live 1/2-flip repair + FP32)* | `44 / 0 B` | `0.199793` | `293.9060` | `165.6912` | `46.468%` | `7.20e-8` | `78.70×` |
| `A16_axis_dp4a_live_repair` *(72B SoA + live repair + `__dp4a`)* | `44 / 0 B` | `0.208210` | `282.0243` | `158.7862` | `44.532%` | `7.20e-8` | `75.52×` |
| `A16_H16` *(72B SoA + live repair + weight-side $H_{16}$)* | `44 / 0 B` | `0.262194` | `223.9571` | `126.2571` | `35.409%` | `7.75e-8` | `59.97×` |
| `E8_WHT8` *(72B SoA + $A_8$ repair + weight-side $H_8$)* | `46 / 0 B` | `0.270674` | `216.9409` | `122.3017` | `34.300%` | `1.09e-7` | `58.09×` |
| **`hampack_18B_fused_gemv`** *(exact `hampack.py` `49283ff`, `30.6 MB` DP)* | `37 / 0 B` | `15.723251` | **`3.7346`** | `2.1054` | `0.590%` | `1.20e-7` | `1.00×` |
| `hampack_18B_standalone` *(exact `hampack.c` decode, `0` bit mismatch)* | `32 / 0 B` | `35.685550` | **`1.6455`** | — | — | `0` bit err | **`164.5×` CPU** |
| **`hammingindex_72B_fused_gemv`** *(exact `hammingindex.py`, `24.4 MB` DP)* | `78 / 0 B` | `119.127492` | **`0.4929`** | `0.2779` | `0.078%` | `1.77e-7` | `0.132×` |
| `hammingindex_72B_standalone` *(exact 2-level decode, `0` bit mismatch)* | `72 / 0 B` | `193.361206` | **`0.3037`** | — | — | `0` bit err | **`30.4×` CPU** |
| *Anchor (`siliquant#77` `49283ff` 1-core CPU `hampack.c`)* | CPU | — | *`0.0094–0.0101`* | — | — | `0` bit err | *`0.0027×`* |

### 3.2 `4096 × 4096` (`16.78M` Weights, `9.44 MB` — Attention Projection Shape)

| Arm / Wire Format (`4.5000 bpw`) | Latency (`ms`) | Throughput (`Gweights/s`) | Effective BW (`GB/s`) | `%` STREAM Triad | CPU FP64 `rel_l2` |
|---|---:|---:|---:|---:|---:|
| **`Block_H2_dp4a_int8`** | **`0.029041`** | **`577.7151`** | **`325.6700`** | **`91.335%`** | `8.90e-8` |
| **`A16_axis_preencoded_dp4a`** | **`0.029102`** | **`576.4954`** | **`324.9824`** | **`91.142%`** | `5.33e-8` |
| `Block_H2_fp32act` | `0.034774` | `482.4677` | `272.3304` | `76.375%` | `8.90e-8` |
| `A16_axis_dp4a_live_repair` | `0.057303` | `292.7806` | `165.0465` | `46.288%` | `5.33e-8` |
| `A16_axis` | `0.059986` | `279.6859` | `157.8696` | `44.275%` | `5.33e-8` |
| `E8_WHT8` | `0.062495` | `268.4581` | `151.5320` | `42.497%` | `8.87e-8` |
| `A16_H16` | `0.064686` | `259.3636` | `146.3986` | `41.058%` | `5.80e-8` |
| **`hampack_18B_fused_gemv`** | `4.031021` | **`4.1620`** | `2.3493` | `0.659%` | `1.57e-7` |
| `hampack_18B_standalone` | `8.780489` | **`1.9107`** | — | — | `0` bit err (`191.1×` CPU) |
| **`hammingindex_72B_fused_gemv`** | `30.303961` | **`0.5536`** | `0.3125` | `0.088%` | `1.81e-7` |
| `hammingindex_72B_standalone` | `56.996332` | **`0.2944`** | — | — | `0` bit err (`29.4×` CPU) |

**Microarchitectural Takeaway:** Eliminating runtime weight-side syndrome repair (`A16_axis_preencoded_dp4a`) doubles fused GEMV throughput (`293.91 -> 611.21 Gweights/s`, `96.51%` Triad) while still verifying all `5` extended Hamming parity checks per `16`-weight block via `__popc`. By contrast, enumerative spherical-ball unranking (`hampack_18B` / `hammingindex_72B`), though `30×–191×` faster on RTX 3080 than on a single CPU core, remains `164×–1,240×` below the GDDR6 roofline due to `30.6 MB` / `24.4 MB` DP-table L2 misses (`7.30×` / `5.81×` L2) and serial multi-limb arithmetic.

---

## Section 4 (`sampled` + `anchor`) — 5-Seed Anisotropy & Calibration Shootout

Executed in GPU FP64 (`1024×1024`, `N_cal = 256`, `N_eval = 256` held-out tokens, seeds `2026112401..2026112405`, **`0` total $A_{16}$ syndrome errors across all seeds and arms**):

### 4.1 Regime I vs. Regime II — Opposite-Sign Hadamard Reversal at Exact `0.000000 dB` Matched Weight MSE

| Regime (`5/5` Seeds, `max |ΔdB| = 0.000000`) | `A16_axis` | `A16_BlockH2` | `A16_H16` | `A16_H16 / A16_axis` | `E8_axis` | `E8_BlockH2` | `E8_H8` | `E8_H8 / E8_axis` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Regime I** *(Coherent block-DC subspace — `12B` `HBG > EBG`)* | **`0.109016`** | `0.129252` | `0.404052` | **`3.706×` (`H16` worse)** | **`0.085557`** | `0.100587` | `0.178230` | **`2.083×` (`H8` worse)** |
| **Regime II** *(Sparse axis-aligned spikes — `270M` `HBG < EBG`)* | `1.124439` | `0.776422` | **`0.372072`** | **`0.331×` (`H16` better)** | `0.740603` | `0.573121` | **`0.344469`** | **`0.465×` (`H8` better)** |

*Natural-MSE weight error & KL note:* Under finite `int4` `[-7, 7]` optimal clipped scaling, `H_16` concentrates coherent Regime I block-DC weights by `4×` onto rotated coordinate `0` (`-13.00 dB` vs. `A16_axis` `-18.60 dB`, natural KL `1.4693` vs. `0.1090`), whereas in Regime II `H_16` disperses lane-0 spikes by `1/4` (`-19.00 dB` vs. `A16_axis` `-11.61 dB`, natural KL `0.0679` vs. `1.1244`). Crucially, even after rescaling residuals to **exact `0.000000 dB` weight-MSE parity**, the error covariance anisotropy alone produces the full opposite-sign reversal on `5/5` seeds, with `Block-H2` acting as a low-variance `INT8`-safe compromise in both regimes.

### 4.2 Experiment 3B — Real Fisher vs. Random-Token (`P5`) vs. Data-Free `1/col_rms^2` Proxy (`bound = 5213` L2-Ball `colscale`)

| Metric (`5/5` Seeds, Held-Out `X_eval`) | `hA16_real` (Real Fisher) | `hA16_rand` (Random `P5`) | `plain` (Uncalibrated) | `hA16_colproxy` (`1/col_rms^2`) |
|---|---:|---:|---:|---:|
| Mean $A_{16}$ Held-Out `KL_quad` | **`0.002401`** (`-32.34%` vs. plain) | `0.003239` (`-8.73%`) | `0.003549` | **`0.004345`** (**`+22.44%` worse!**) |
| Mean $E_8$ Held-Out `KL_quad` | **`0.003087`** (`-32.34%` vs. plain) | `0.004159` (`-8.82%`) | `0.004562` | **`0.005570`** (**`+22.11%` worse!**) |
| Gain Retention vs. Real Fisher | `100.00%` | **`27.01%`** (`< 70%` `P5` floor; `12B` anchor `53.74%`) | `0.00%` | **`-69.41%`** *(net degradation)* |
| Spearman $\rho$ vs. True Held-Out Hessian | **`+0.3345`** (`5/5 > 0`) | — | — | **`-0.1816`** (`5/5 < 0`) |

---

## Section 5 — Verification Receipts & Complete File Manifest

- **4-Mode Isolation Parity (`-I -B`, `-I -O -B`, `-I -B -W error`, `PYTHONHASHSEED=42 -B`):** `--standalone --json` bit-identical to `0o600` disk certificate across all `4/4` modes; `--check` `valid=true` across `4/4`; `--verify-receipts .` `13/13` checks `true` across `4/4`; `test_rtx3080_siliquant_a16_r5.py` `6/6 OK` across `4/4`.
- **Live Mutation Testing (exhaustive mutation testing):** `100.0%` kill rate (`0` survivors, `truncated=False`) on both `rtx3080_siliquant_a16_oracle.py` and `rtx3080_siliquant_a16_audit.py`.
- **Whole-File AST `BoolOp` Sweep (`OP_SWAP + SUBCOND_DEL`):** `100.0%` kill rate (`0` survivors) on both `oracle.py` and `audit.py`.
- **4-Surface Adversarial Certificate Fuzzer (`CLI`, `VCD_FR1`, `VCD_FR0`, `SEC_DIRECT`):** `82/82 PASS` (`25/25 CLI + 19/19 VCD_FR1 + 19/19 VCD_FR0 + 19/19 SEC_DIRECT`; `bypasses=0, crashes=0`).
- **Hygiene & Safety (Static hygiene & safety scan):** `PASS` (`0` findings, `0` local-path leaks, `0` private tokens, `0` CRLF, `0` bare `assert`s, `0` AST floats in oracle/audit/test, `0` `secret*` vars, scratch `os.mkfifo` non-regular probes only, `PYC=0`).

### File Manifest

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
