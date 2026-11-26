# Preregistration Addendum 2026-11-24 — A_16 ball arms and 16-bit-decoded stock controls on Gemma

This addendum extends `PREREG.md` in this folder under the plan-before-data rule, following
`notes/handoff-addendum-2026-11-24.md` on branch `bayan/charming-gates-jlas14` (commit `2c52ec9`).
It is committed **before** any row of the arms below is scored. Runner: `exp/gemma-a16-port`
(off `ccf045d04ada2170eeafaa37254a764b96552723`, which is `exp/gemma-ball-runner` HEAD); the port
vendors `hamming.py` / `hamming_dp.c` from `experiments/hamming-lattices-real-weights-2026-11-24`
and adds the arms below to `run.py`. All prior rules of `PREREG.md` and HANDOFF.md section 6 apply.

## 1. Why these arms

The 2026-11-24 records (`experiments/hamming-lattices-real-weights-2026-11-24`) found `A16BG128`
(the Construction-A lattice of the extended Hamming code [16,11,4], one ball per 128 weights,
`b560s16` = 4.5 bits) the lowest or tied-lowest arm at 4.5 bits on Qwen2.5, SmolLM2 and Llama-3.2,
and beating integer levels at 3.0 bits on SmolLM2. On Gemma, turning E8 by H8 (`HBG`) did **not**
help, so A_16 on Gemma is an **out-of-family test**: Gemma would be the fourth model family.
The same records found the committed `Q4_K_M` rows were scored on the CPU backend, where llama.cpp
rounds activations to 8 bits when multiplying by `Q4_K` blocks; the ball arms, stored as 16-bit
weights, never pay that. The `@16` controls measure that confound on this machine's backend.

## 2. Pinned inputs (SHA-256, from HANDOFF.md section 2; verified on the local files)

| Input | SHA-256 |
|---|---|
| `gemma-3-4b-it-BF16.gguf` | `7bcfca74279608552011d3c4d11e9298934502ffa410eb21b9047a6276b87ba6` |
| `gemma-3-12b-it-BF16.gguf` | `276920b11d80002d45b9b85291e333d9a7798df92b148ba9ae3d0625dd9f82f2` |
| `gemma-4-E4B-it-BF16.gguf` | `38e0dba6818d18e2b41062c7b3b9083dcfddc44767078c415ff7e13661911ba5` |
| `wiki.test.raw` | `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08` |
| `wiki.train.raw` | `6707892fa3788b5ab9ed78ab5ff37d9fe825f6011a2ad4fcd6a6d467f0e7da57` |
| `base.py` (python text) | `e38975e1c68d98ac1664dfd530616eb35c72294382a4dd873d4746b23f27779f` |
| 4B real imatrix | `19b11624568c4352ef0a43c2aca3599ac4a0af166f7a8102c4db301db8f35b7a` |
| 12B real imatrix | `9aaa78bb7785aaa02971b681983eb0918f92ba76842f4ac9bee1bea5961ebef8` |
| E4B real imatrix | `b67f51e20abb6bfe4243104514c4164bfc1ba58d147d66df1595140676348505` |

llama.cpp binaries: record `llama-cli/llama-perplexity/llama-quantize/llama-imatrix --version`
in every run log (local build: `0.5.0 (build 11146, commit 7fe450e19)`, AppleClang arm64).
Every row records `backend` (`metal` on Darwin with `-ngl 999`, else `cpu`), `ngl`, and `threads`,
as the addendum requires.

## 3. Arms and models

| Arm | Meaning |
|---|---|
| `A16BG128:b560s16` | A_16 in one ball per 128 weights, 16-bit scale per group, no imatrix (4.5 bits/weight by counted bits) |
| `hA16BG128:b560s16` | same with half-strength (power 0.25) real-imatrix column weights applied before rounding |
| `file:Q4_K_M@16` | the `file:Q4_K_M` quantized file's block matrices decoded to 16-bit floats in the 16-bit model |
| `file:Q4_K_M-imat@16` | same from the `file:Q4_K_M-imat` file |

Models and texts:

1. `gemma-3-4b-it` and `gemma-3-12b-it`, texts `wikitext,python`: arms
   `A16BG128:b560s16`, `hA16BG128:b560s16`, `file:Q4_K_M@16`, `file:Q4_K_M-imat@16`,
   alongside the Priority-1a tensor-matched stock reruns (`file:Q4_0`, `file:Q4_K_M`,
   `file:Q4_K_M-imat`, `other_tensors_that_differ == []` required) and Priority-1b/1c
   (`hEBG128:b496s16`, `hEBG128:b560s16`, `hHBG128:b560s16`).
2. `gemma-4-E4B-it` BF16, texts `wikitext,python`: `A16BG128:b560s16` added to the Priority-1d
   closure sweep (out-of-family check on the primary PREREG model; require
   `imatrix_fallback_tensors` to contain only the 36 unread KV-sharing tensors as in 1d).

Bit counts are counted, not stored: `(log2(points) + 16) / 128` with `points` from
`hamming.ball_bound(4, 128, B)` (`B = 560` → 4.4997 bits; `B = 368` → 3.0 bits if run).

## 4. Predictions and falsifiers (pre-registered)

- **A1 (out-of-family transfer):** on `gemma-3-4b-it` and `gemma-3-12b-it` WikiText,
  `A16BG128:b560s16` beats `HBG128:b560s16` (the closest existing arm) by `>= 2.0` combined SE
  in mean KL. **Falsifier:** A_16 is not below HBG by 2.0 combined SE on either model
  → no transfer to Gemma; record as non-transfer, do not claim a Gemma win.
- **A2 (weighted variant):** `hA16BG128:b560s16` beats `A16BG128:b560s16` by `>= 2.0` combined
  SE on WikiText KL on at least one of the two models. **Falsifier:** misses 2.0 SE on both.
- **A3 (activation-rounding confound):** on this backend (Metal, `-ngl 999`), `file:Q4_K_M@16`
  and `file:Q4_K_M-imat@16` have mean KL **at least as low as** their 4-bit-block counterparts
  `file:Q4_K_M` / `file:Q4_K_M-imat` on both texts and both models (16-bit decode cannot lose to
  the same weights multiplied as 4-bit blocks with 8-bit activations). The size of the gap
  (`KL(file:Q4_K_M) - KL(file:Q4_K_M@16)`) is the measured activation-rounding share of the
  committed Q4_K_M rows; report it per model/text with the backend fields. **Falsifier for the
  CPU-record tilt:** if on this backend `@16` is *worse* than the 4-bit row, the earlier CPU
  tilt conclusion does not transfer to Metal and must be re-measured there (do not assume).
- **A4 (provenance):** every `@16` row carries `source_file_sha256` equal to the `file:` arm's
  `file_sha256` in the same run, and `other_tensors_that_differ == []`.

## 5. Controls, nulls, stop rules

- Unchanged-payload and embedding-only negative controls as in PREREG.md P2 gates.
- The `~rand` null exists for the weighted ball arms via `hA16BG128:b560s16~rand` if run.
- Resume rule: rows append to `results.jsonl` keyed by `(model_id, text, variant)`; a rerun never
  overwrites a committed row.
- Stop rule: if the bracket rule of Priority 1b halts (`KL(b560) >= KL(b496)` anomaly), halt the
  A_16 runs for that model too and record the anomaly before any further arm.
- macOS note (from the addendum): the 2026-11-24 run programs assumed Linux `fork`; this runner
  uses `spawn` on macOS. The port was validated with a `--metrics-only` single-arm build on
  `gemma-3-270m-it` (spawn path) before any scored run, and runner commits are switched strictly
  between model jobs, never while `Pool` workers are live.

## 6. Verification gates before committing any row

- 3-mode parity (`python3 -B`, `-O -B`, `-B -W error`) on `test_run.py`, including the new
  `TestAddendumArms` (regex routing, dispatch, bits, ball-bound invariant, C/NumPy decoder parity,
  bit-for-bit scale parity with the 2026-11-24 record's `ball_round`).
- Vendored `hamming.py` tests: `python3 -m unittest .../test_hamming.py shape/test_hammingindex.py
  shape/test_hampack.py` on the `bayan/charming-gates-jlas14` checkout (C built with
  `cc -O2 -ffp-contract=off`).
- `python3 checks/plan_before_data.py` and `python3 checks/claim_ledger.py` in all 3 modes.
- Exact SHA-256 provenance for every input and every `results.jsonl` / `results.out`, plus the
  runner commit SHA, per HANDOFF.md section 6. No weights or datasets committed; no local paths,
  secrets, or internal names in anything pushed.
