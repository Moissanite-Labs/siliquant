# Wave 2 Round 7 Arm R7-1 (`AUDIT-068R`): Deep Technical & Trust-Boundary Audit of `magnitudedev/magnitude`

- **Audit Target (Strictly Read-Only, `0` external writes):** `magnitudedev/magnitude` @ `d89dfffcbb2f27251e99163934c04770b453339c` (`2026-11-21`, `Merge pull request #259 from magnitudedev/cli-only-install`)
- **Cross-Pollination Targets:** `Moissanite-Labs/warp-drive` (PR `#15` / `#13` / `#12`), `Moissanite-Labs/siliquant#24`, `Moissanite-Labs/hillclimb#4`
- **Novelty Classification:** **`N1`** (engineering synthesis, exact pure-$\mathbb{Q}$ transfer of existing Ville supermartingale & IEEE `RoundingCells::width` admission rules, and formal trust-boundary bridge; zero unverified `N2` claims)
- **Local Executable Artifacts:**
  - Study Overview & Quickstart: [README.md](./README.md)
  - Oracle & Cross-Pollination Bridge: [magnitude_autotune_bridge_audit.py](./magnitude_autotune_bridge_audit.py)
  - Adversarial R5 Test Suite: [test_magnitude_bridge_r5.py](./test_magnitude_bridge_r5.py)
  - Zero-Float Canonical Certificate: [magnitude_bridge_certificate.json](./magnitude_bridge_certificate.json)

---

## 1. Law-4 Prior-Art Quarantine (`N0`)

All upstream components belong strictly to their respective authors and are quarantined as `N0`:

| ID | Prior Art Component | Upstream Source / Citation |
|---|---|---|
| `R7-1-N0-01` | Anytime-valid uniform-Beta mixture paired sign test (Ville's inequality) | `magnitudedev/magnitude` `inference/seismic/measure/src/statistics.rs:27-53, 105-116` (classical Ville 1939 / Robbins 1970 mixture martingale) |
| `R7-1-N0-02` | Paired round randomization (`xorshift64*`), sustained-clock GPU governor raising, out-of-process `InferenceWorker` measurement isolation | `magnitudedev/magnitude` `inference/seismic/measure/src/{statistics.rs:277-306, clock.rs:1-74}`, `inference/service/server/src/assessment/measurement.rs:1-27` |
| `R7-1-N0-03` | IEEE-754 `RoundingCells::width` (`0.5 * below + 0.5 * above`), RMS comparison & `GoverningStep` admission framework | `magnitudedev/magnitude` `inference/seismic/lang/src/lattice.rs:448-462`, `inference/seismic/runtime/src/native/{comparison.rs:14-80, admission.rs:47-188}` |
| `R7-1-N0-04` | Typed multi-bank hybrid state (`ConvWindow`, `DeltaState`, `DeltaTape`, `SsmState`, `SsmTape`, `ZERO_SEED_BANK = 0`) & exact `(tokens, coordinates, conditioning)` prefix cache | `magnitudedev/magnitude` `inference/engine/state/src/{bank.rs, domain.rs, lib.rs}`, `inference/engine/scheduler/src/prefix_cache.rs` |
| `R7-1-N0-05` | Stage 1–4 runtime flag pinning, 4-layer SHA-256 `CheckpointFingerprint`, `BankFlushReceipt`, loopback/auth hardening | `Moissanite-Labs/warp-drive` (`#12`, `#13`, `#15`) |
| `R7-1-N0-06` | Behavioral Hessian certified outlier-split $E_8^{\mathrm{Had}}$ & Leech24 fixed-rate quantization (`+5.48 dB` heavy-tail gain, `+1.22..+3.97 dB` 2.0 bpw gain) | `Moissanite-Labs/siliquant#24` |
| `R7-1-N0-07` | Bounded optimization ratchet & held-out outside-anchor Goodhart defense | `Moissanite-Labs/hillclimb#4` & Aletheia Portal-Archive |

---

## 2. Law-2 Steelman: Genuine Engineering Strengths in `magnitudedev/magnitude`

Before cataloging defects, four concrete engineering patterns in `magnitudedev/magnitude` are genuinely high-quality and directly valuable to `Moissanite-Labs`:

1. **Distribution-Free Anytime-Valid Paired Sign Testing (`inference/seismic/measure/src/statistics.rs:27-53, 105-116`; `inference/seismic/runtime/src/native/sampling.rs:1-75`):**
   - Evaluates paired in-session ratios $d_r = \sum_p w_p I_p (c_{p,r}/i_{p,r} - 1)$ under sustained device clock (`clock.rs:1-74`) and randomized AB/BA order (`Shuffle`, `statistics.rs:277-306`), using the exact uniform-Beta mixture supermartingale $M_n(s) = \frac{1}{(n+1)\binom{n}{s}}\sum_{j=0}^{s}\binom{n+1}{j}$ which resolves unanimously in $n=7$ rounds at one-sided $\alpha=0.05$ ($M_7(7) = 255/8 \ge 20 > 127/7 = M_6(6)$).
2. **Typed Hybrid Multi-Bank State Geometry (`inference/engine/state/src/{bank.rs, domain.rs, lib.rs}`):**
   - Explicitly separates attention history (`HistoryDomainKind::{Token, Window, Block, Shared}`) from recurrent banks (`ConvWindow`, `DeltaState`, `DeltaTape`, `SsmState`, `SsmTape`) with an immutable `ZERO_SEED_BANK = 0` (`state/src/lib.rs:1019`) supporting Qwen3.5 (Gated DeltaNet), Nemotron-H (Mamba-2), LFM2, DeepSeek41, GLM5Next, Gemma4, and Llama.
3. **Existing Cryptographic & Memory-Safety Primitives:**
   - Constant-time XOR-fold Bearer verification (`inference/service/api/src/lib.rs:407-427`),
   - Strict package path & SHA-256 verification (`inference/service/models/src/validation.rs:84-203`) and content-addressed skeleton GGUF verification (`inference/engine/artifacts/src/gguf.rs:259-296`),
   - `seismic_io::Store` already implements `Privacy::Owner` (`0700`/`0600` + `O_NOFOLLOW`, `inference/seismic/io/src/store.rs:203-220`),
   - Explicit W^X sealing & `aarch64` icache invalidation (`CoherentCode`, `inference/seismic/backends/cpu/src/command.rs:21-96`).

---

## 3. Verbatim Findings Matrix (`MAG-01` .. `MAG-08`)

Every finding below is pinned to verbatim `file:line` anchors at commit `d89dfffcbb2f27251e99163934c04770b453339c` (`19/19` anchors live-verified in `AUDIT-068R`) and reproduced in exact zero-float rational arithmetic matching Rust `RoundingCells::width` (`inference/seismic/lang/src/lattice.rs:448-462`) by [magnitude_autotune_bridge_audit.py](./magnitude_autotune_bridge_audit.py):

| ID | Severity | Subsystem | Verbatim `file:line` Anchors | Finding & Exact Zero-Float Proof |
|---|---|---|---|---|
| **`MAG-01`** | **HIGH** | Kernel Autotuning Numerical Admission | `inference/seismic/runtime/src/native/comparison.rs:61, 146-166, 203-224, 271, 508-512`; `inference/seismic/lang/src/lattice.rs:448-462`; `inference/seismic/runtime/src/native/admission.rs:155-188`; `inference/engine/executor/src/native/tuning/precision.rs:31` (`STANDARD_ALPHA = 8.0`) | **RMS-Only Gate Blind to Localized $L_\infty$ Spikes & Heavy-Tail Outlier Masking.** `comparison.rs:61` records `maximum_absolute_error`, yet `admission.rs:155-188` never checks it. Under exact Rust `RoundingCells::width` + `Enumerated::containing` (round-to-nearest-even into the governing dtype's finite cells, returning `None`/zero step power at or beyond `max_finite + ulp_max/2`, `(3/4) * ULP` at normal powers of two $2^e, e > e_{\min}$, and full `ULP` elsewhere including `max_finite` per `lattice.rs:458-461` & `comparison.rs:508-512`): **(a) Localized spike:** On $N=4096$ BF16 values at $3/2$ ($w_{\text{BF16}}(3/2) = 1/128$), corrupting 1 element by $+147$ rounding cells ($+147/128 = +76.5625\%$) gives exact $\text{ratio}^2 = 64827/1024 < 64 = 8^2$ — **ADMITTED**. **(b) Heavy-tail masking:** With 1 non-power-of-two outlier at $6144$ ($w_{\text{BF16}}(6144) = 32$) and $4095$ normal channels at $1$ ($w_{\text{BF16}}(1) = 3/512$), **zeroing all 4,095 normal channels completely** yields exact $\text{ratio}^2 = 12881756160 / 268472311 \approx 47.9817 < 64$ — **ADMITTED** by Magnitude's actual Rust formula (while `hardened_dual_norm_admission` rejects with exact bulk $\text{ratio}^2 = 1048576 / 3$). |
| **`MAG-02`** | **HIGH** | Recurrent / Accumulator State Precision | `inference/seismic/runtime/src/native/admission.rs:60-68, 140, 196, 385-396`; `inference/engine/state/src/bank.rs:27-45`; `inference/engine/executor/src/native/tuning/precision.rs:59-68` | **`GOVERNING_STEP = GoverningStep::Activation` Relaxes `f32` Recurrent States by $65{,}536\times$.** `admission.rs:60-68` hardcodes `GoverningStep::Activation` (`activation.unwrap_or(stored)`), judging `F32` recurrent tensors (`DeltaState`, `SsmState`) against `BF16` rounding cells ($w_{\text{BF16}}(1)/w_{\text{F32}}(1) = (3/2^9)/(3/2^{25}) = 65{,}536\times$). An `F32` state at $1.0$ corrupted by **`16,384` `F32` ULPs** (`+1/512`) on every element yields exact $\text{ratio}^2 = 4/3 < 64$ under `BF16` Activation (**ADMITTED**, corroborated by `admission.rs:390-394`) vs exact `17179869184/3` under `F32` Subject (**REJECTED**). Cross-format nearest-cell (`1 - 2^{-24} \to w_{\text{BF16}}(1) = 3/512`) and absent-cell (`f32::MAX` beyond `BF16`'s largest cell $\to$ zero step power, infinite ratio, rejected) semantics are also matched verbatim. |
| **`MAG-03`** | **HIGH** | Statistical Autotune Search (`Seismic`) | `inference/seismic/measure/src/statistics.rs:27-53, 105-116, 156, 185-210`; `inference/seismic/runtime/src/native/sampling.rs:41-75`; `inference/seismic/runtime/src/native/search.rs:53-127` | **Two-Sided $2\alpha$ Leak, Uncorrected $K$-Candidate Multiplicity, Float Evaluation, & No Outside Anchor.** `median_interval` (`statistics.rs:156`) splits `rate / 2.`, but `Tally::verdict` (`statistics.rs:202-210`) tests both `Leads` and `Trails` at unsplit `rate`, inflating two-sided anytime error to $2\alpha$ (`10%` at $\alpha=0.05$; true two-sided $\alpha=0.05$ requires $M_n \ge 40$, minimum $n=8$ unanimous rounds: $M_8(8)=511/9 \ge 40 > 255/8=M_7(7)$). Every observation increments `rounds = ahead + behind + ties` (`Tally::observe`, `statistics.rs:185-192`), so a single tie weakens `7` ahead rounds to $M_8(7)=251/36 < 20$ (`undecided`) and `8` ahead rounds to $M_9(8)=1013/90 < 40$ (`undecided`, `promote=False`). Without multiplicity correction across $K$ null challengers, the dependence-robust union bound only bounds FWER by $\le \min(1, K\alpha)$ (vacuous $1$ at $K=20, \alpha=1/20$); as an illustrative reference under the extra assumptions of $K=20$ mutually independent null challengers each calibrated at exact rejection probability $\alpha=1/20$ (neither independence nor exact calibration follows from Ville's inequality alone), $1 - (19/20)^{20} = 67267626542454041806644399 / 104857600000000000000000000 \approx 64.15\%$. `hardened_supermartingale_gate`'s two-sided Bonferroni threshold $2K/\alpha$ bounds two-sided FWER $\le \alpha$ under arbitrary dependence, while also requiring minimum practical speedup $\delta_{\min} > 0$ and held-out outside-anchor ratification (`hillclimb#4`). |
| **`MAG-04`** | **HIGH** | Native Program & Kernel Cache Integrity / JIT | `inference/seismic/runtime/src/artifacts/directory.rs:110`; `inference/seismic/measure/src/store.rs:54`; `inference/seismic/io/src/store.rs:203-220`; `inference/seismic/runtime/src/formation.rs:542-559`; `inference/seismic/backends/cpu/src/{library.rs:20, toolchain.rs:280-385, command.rs:30-46}`; `inference/engine/executor/src/kernel_cache.rs:374-423` | **`Privacy::Shared` Cache + Unauthenticated JIT Pointer Trust + Unverified 64-bit Evidence Digests.** Although `seismic_io::Store` implements `Privacy::Owner` (`0700`/`0600` + `O_NOFOLLOW`), both `DirectoryStore::open` (`directory.rs:110`) and `JsonFileStore` (`store.rs:54`) use `Privacy::Shared`. `inference/seismic/runtime/src/formation.rs:542-559` checks cached `CpuArtifact` bytes ONLY against an unauthenticated co-located `Pointer` file in the same `Privacy::Shared` directory before `toolchain.rs:351-385` maps machine code into executable `CoherentCode` pages, resolves arbitrary `Target::Library(name)` via `libc::dlsym(RTLD_DEFAULT, name)` (`library.rs:20`), and `transmute`s to `LaunchEntry` (`toolchain.rs:384`). Separately, `KernelCache::store_tuning` (`kernel_cache.rs:421`) truncates evidence SHA-256 to 8 bytes (64 bits) and `KernelCache::tuning` (`kernel_cache.rs:391-401`) never re-hashes loaded evidence bytes against `entry.evidence`. |
| **`MAG-05`** | **HIGH** | HTTP API Auth, Sampling Determinism, & Adapter Surface | `inference/service/server/src/main.rs:70-80, 345-351`; `inference/service/api/src/lib.rs:380-427`; `inference/engine/serving/src/sampling.rs:8-11, 79-84` | **Unauthenticated Non-Loopback Bind Allowed + Random Default Sampling + Zero LoRA Governance.** `main.rs:70-80, 345-351` permits `--bind 0.0.0.0:<port>` with `auth_token: None`, exposing management routes (HF credentials, local model root config, deletion) unauthenticated. Default sampling (`sampling.rs:8-11, 79-84`) uses `temperature=1.0, top_p=0.95, top_k=20` with a random `RandomState` seed per omitted-seed request (incompatible with deterministic Stage 1/3 verifier execution). Zero dynamic LoRA runtime or Stage 1–4 verifier surface exists (`lora` appears only as MLA `q_lora_rank`/`kv_lora_rank` metadata). |
| **`MAG-06`** | **MEDIUM** | Model Weight Lineage | `inference/service/models/src/weight_lineage.rs:31-39, 66, 124-149, 169-176` | **Heuristic 64-Float Signed Pearson `WeightSample` Probes ONLY 1D `*norm.weight` Tensors in `blk.0`/`blk.1` + `output_norm`.** `weight_lineage.rs:34-35, 66, 169-176` filters strictly for `tensor.shape.len() == 1` and `rest.ends_with("norm")` in `blk.0.`/`blk.1.` plus `output_norm.weight` — NEVER probing ANY 2D projection weight matrix (`attn_q`, `ffn_gate`, etc.) even in layers 0–1, nor any tensor in `blk.2..blk.31` (exact `ml-os#6` / `AUDIT-064` last-layer edit vector). Moreover, signed `1.0 - correlation(values, other)` (`weight_lineage.rs:133`, `LINEAGE_DIVERGENCE = 2.5e-4`) is invariant under positive affine rescaling ($w' = a w + b, a > 0 \implies r = 1, \text{divergence} = 0$, while rejecting negative rescaling $a < 0$). Must never replace `warp-drive`'s 4-layer SHA-256 `CheckpointFingerprint`. |
| **`MAG-07`** | **MEDIUM** | Quantization & Backend Dispatch | `inference/ggml/src/lib.rs:22-56` (`Mxfp4 = 39`); `inference/seismic/native-cpu/src/quant.rs:47-55, 250-289` | **Scalar/Block Quantization Ceiling vs Lattice & Outlier-Split Kernels.** Supports GGML scalar/block formats (`Q4_0..Q8_K`, `IQ1..IQ4`, `MXFP4`, `NVFP4`, `Q1_0`, `Q2_0`) plus 32-element `Q8Block` activation quantization, with zero structured lattice codebooks ($E_8^{\mathrm{Had}}$ / Leech24) and zero Hessian-certified outlier splitting (`siliquant#24`). |
| **`MAG-08`** | **MEDIUM** | Prefix Cache & Multi-Bank State Isolation | `inference/engine/scheduler/src/prefix_cache.rs:10, 85-107`; `inference/engine/state/src/lib.rs:653-673, 1039-1093, 1614, 1629` | **Unscoped Global `PrefixKey` & Non-Zeroed Bank Slot Reuse.** `PrefixKey` (`prefix_cache.rs:85`) keys retained KV + recurrent states solely by `(tokens, coordinates, conditioning)` with zero tenant/zone isolation across callers. `StateStore::new` zeroes slabs only at initial `add_slab(SlabFill::Zeroed)` (`lib.rs:1614, 1629`); `Arena::release` (`lib.rs:653-673`) and `Drop for BankClaim` (`lib.rs:1039`) return slots to the free list without zeroing backing memory or emitting a `BankFlushReceipt`. |

---

## 4. Three-Way Cross-Pollination Architecture (`N1` Synthesis)

1. **`magnitudedev/magnitude` $\leftrightarrow$ `Moissanite-Labs/hillclimb#4`:**
   - **Adopt from Magnitude:** Anytime-valid paired sign supermartingale $M_n(s)$ with in-session AB/BA randomization and sustained clock raising (`statistics.rs`, `clock.rs`).
   - **Harden via `hillclimb#4` & Exact $\mathbb{Q}$:** Evaluate $M_n(s)$ in exact pure-$\mathbb{Q}$ rational arithmetic (`ville_mixture_ratio`), enforce two-sided Bonferroni threshold $2K/\alpha$, require minimum practical speedup $\delta_{\min} \in \mathbb{Q}_{>0}$, and require held-out outside-anchor ratification before promoting any candidate kernel (`hardened_supermartingale_gate`).
2. **`magnitudedev/magnitude` $\leftrightarrow$ `Moissanite-Labs/siliquant#24`:**
   - **Bridge Defect (`MAG-01` + `MAG-02` + `MAG-07`):** Magnitude's global RMS admission breaks catastrophically on heavy-tailed transformer layers (one outlier at `6144` hides 100% erasure of `4,095` normal channels even under exact Rust `RoundingCells::width`) and relaxes `F32` recurrent states by `65,536x`.
   - **Hardened Contract:** `hardened_dual_norm_admission` locks recurrent states (`DeltaState`, `SsmState`) to `GoverningStep::Subject`, enforces an exact per-element $L_\infty$ rounding-cell ceiling, and applies `siliquant#24`-style outlier/bulk channel splitting so bulk RMS and outlier RMS must pass independently—providing a safe admission contract for future $E_8^{\mathrm{Had}}$ / Leech24 / outlier-split Seismic kernels.
3. **`magnitudedev/magnitude` $\leftrightarrow$ `Moissanite-Labs/warp-drive`:**
   - **Bridge Defect (`MAG-04`, `MAG-05`, `MAG-06`, `MAG-08`):** `evaluate_trust_boundary` unifies 9 fail-closed Stage 1–4 invariants with strict type/range checks (zero `float` acceptance, zero `bool`-as-`int` smuggling): loopback/strong-token bind guard (`>= 16` non-whitespace chars), deterministic verifier sampling (`temperature=0, top_p=1`, explicit int seed) + 64-hex `tuning_identity_sha256` pinning, strict `Privacy::Owner` (`0 <= mode <= 0o777`, `(mode & 0o077) == 0`, `O_NOFOLLOW`) cache enforcement, full 64-hex verified evidence SHA-256, authenticated JIT pointers + math-only `dlsym` allowlist, full cryptographic `checkpoint_sha256` pinning (blocking `WeightSample` spoofing), non-empty tenant-scoped `PrefixKey` isolation, and non-empty zero-taint multi-bank `BankFlushReceipt` parity across `ConvWindow`/`DeltaState`/`DeltaTape`/`SsmState`/`KVPages`.

---

## 5. Reproducible Clean-Room Verification Commands

```bash
# 1. Verify standalone JSON output matches committed certificate bit-for-bit
python3 -I -B magnitude_autotune_bridge_audit.py --standalone --json | diff -u magnitude_bridge_certificate.json -

# 2. 4/4 isolated -I R4 bit-identical check (standalone)
for flags in "-I -B" "-I -O -B" "-I -B -W error" "-I -O -B -W error"; do
  python3 $flags magnitude_autotune_bridge_audit.py --standalone --json | sha256sum
done

# 3. Adversarial R5 unit test suite
python3 -I -B -W error test_magnitude_bridge_r5.py -v

# 4. Optional Aletheia v2.0.0 mutation kill (26/26 = 100.0% KILLED) & strict-ancestors audit-check
#    (requires Foundation-0/aletheia on PYTHONPATH)
python3 -B -W error -m aletheia mutate magnitude_autotune_bridge_audit.py \
  --max-mutants 0 --test-cmd "python3 -I -O -B test_magnitude_bridge_r5.py" --add-cwd
python3 -B -W error -m aletheia audit-check . \
  --strict-ancestors --safety-scan --scripts magnitude_autotune_bridge_audit.py test_magnitude_bridge_r5.py
```

### Explicit Falsification Criteria
This audit is falsified if any of the following holds:
1. Any of the 19 verbatim `file:line` anchors fails to match `magnitudedev/magnitude @ d89dfffcbb2f27251e99163934c04770b453339c`.
2. Either `MAG-01` counterexample (`+147` BF16 rounding-cell localized spike at `3/2`, or `6144` non-power-of-two outlier masking `4095` zeroed channels at `1`) disagrees with exact Rust `RoundingCells::width` (`lattice.rs:448-462`), fails to pass `magnitude_rms_admission` at `STANDARD_ALPHA = 8`, OR is admitted by `hardened_dual_norm_admission`.
3. `weight_sample_matches` admits negative affine rescaling (`-x`) or rejects positive affine rescaling (`5x + 3`).
4. Any float primitive appears anywhere in `magnitude_bridge_certificate.json`, or any of the 4 isolated `-I` R4 invocations diverges in SHA-256.
5. Any mutant of `magnitude_autotune_bridge_audit.py` under `aletheia v2.0.0 mutate --max-mutants 0 --add-cwd` survives (`survived > 0`), or `aletheia v2.0.0 audit-check --strict-ancestors --safety-scan` reports any failure.
