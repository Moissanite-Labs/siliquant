# What Moissanite-Labs/warp-drive holds for this repository

A reading on 2026-11-24 of warp-drive's `main` (`1545cc4`), its branches, and its pull requests and issues. It adds to [cross-org-survey-2026-11-24.md](cross-org-survey-2026-11-24.md). Read-only: nothing was written to warp-drive.

## What warp-drive is

The operating-system layer of the siliconcordat stack. `warp-modeld` stands between an agent harness and `llama-server`. It admits a LoRA adapter only after four checks: fingerprints, an exact-integer certificate, the runtime flags, and canary prompts. It can also roll the adapter back. Its target machines are an RTX 3080 with 8 GiB, serving models of 4 to 8 billion weights, and a Mac. Its demo serves `Qwen3.5-4B-Q8_0.gguf`.

## No quantization result to borrow

- **The studies prove facts about made-up matrices of size 3 or 4.** `studies/expedition-weight-iteration-bridge` (WI-5) quantizes a group of 4 numbers to 2 bits. It shows that weighting by a Hessian makes the error 9 times larger, and that setting the outlier apart takes it to 0. Its "siliquant#24" constants are hardcoded and then checked against themselves:
  - 929/1080, the cell of E8 over that of the integers;
  - (8/3)^2 < 8;
  - 1/3 for the Leech lattice.
  The kept columns of `loud-columns-2026-11-18` are the same idea, measured on real models.
- **`studies/expedition-latent-lattice-2611-21608` is not about lattice quantization.** It reads arXiv:2611.21608, *Latent Lattice Language Models*. The "lattice" there is a grid of layers by time, with a recurrent state in each layer, and the paper has no code or weights. Its "siliquant horizontal quantization" theorem (LLLM-SQ-01) applies the rounding-cell rule of `magnitudedev/magnitude` to 256 numbers. It does not use this repository's code.
- Nothing in warp-drive rounds a model's weights or scores a model's output.

## Where it depends on this repository

**The SiliQuant bridge.** Two pull requests carry it, and neither is merged.

- **PR #40** (branch `hackathon/cerebras-siliquant-bridge`) was closed without being merged.
- **PR #41** ("Prepare warp-drive for offline alpha testing", open, head `0d3b551`) replaces it.
  - It loads `shape/check_certificate.py` from a checkout of this repository.
  - It needs the file's SHA-256 to be the one the caller expects. Today that is `2941d412…`, the same on `main` and on this branch.
  - It runs the file's `check`, `identities` and `pi_bounds_hold`.
  - It admits only three quantization names, `siliquant_4_5`, `siliquant_e8` and `siliquant_ballpack`.
  - It admits only five arms, `X`, `Z8:p1`, `Z8:p2`, `E8:p1` and `E8:p2`.

Three things follow.

1. **It certifies the wrong quantity.**
   - A receipt of `shape/certified.py` proves that the arm picked has the smallest *predicted weight error* among the candidates.
   - This repository's records show that weight error does not order the arms by output. On SmolLM2 at 3.0 bits, E8 in a ball has 0.65 dB less weight error than integer levels: -16.452 against -15.798 dB. Its mean KL is 0.664189 against 0.419643, 23.4 errors worse. Those rows are in `experiments/e8-hadamard-real-weights-2026-11-22/results.out`.
   - At 4.5 bits E8 and E8 turned have the same weight error, -25.585 dB. They are 5.4 errors apart on output.
   - So a certified pick can be the worse model by a wide margin. PR #41's own scope note says the receipt does not prove model quality. It is right.
2. **It names arms this repository no longer recommends.** The five arms are cut codebooks over 8 weights, from `shape-formula-held-out-2026-11-17`. The arm the records recommend with no data is not among them: one ball per group of 128 weights, integer levels or E8 turned, with ten columns of each `ffn_down` kept. The arm `X` is the plain cube.
3. **The formats it names cannot be loaded by the server it guards.** `llama-server` reads none of them.
   - `siliquant_ballpack` is the packed integer format of `packed-ball-format-2026-11-18`. It is decoded to 16-bit floats before the model runs.
   - `siliquant_e8` has no file at all.
   - Decoded to 16 bits, a model of 4 billion weights takes about 8 GB. That is more than the `Q8_0` file the demo serves, and more than the 8 GiB card it targets.

## What would make the link useful

- **A receipt of measured output, not predicted error.** Each scored row of this repository's records names:
  - the SHA-256 of the 16-bit model;
  - the arm, its bits, and its weight error;
  - its mean KL with its standard error, on a stated text.

  A bridge could check a receipt like that against the model file it is given. That would bind an admitted model to a measured distance from its source. It is a proposal only. Nothing here is built.
- **A note before changing `shape/check_certificate.py`.** Any edit changes the SHA-256 that PR #41 pins. Renaming or removing `check`, `identities` or `pi_bounds_hold` breaks it.
- **An engine that reads a ball format.** This is the item that gates any use by warp-drive. On an 8 GiB card a model of 4.5 bits a weight is where a better arm would matter. Until `llama-server` or a kernel reads one, only the decoded floats can be served.

## Other open items there, for the record

They do not touch this repository.

- Issue #44 says the Stage 2 certificate accepts invented metrics checked only by its own hash.
- Issue #45 says the library's default mount path never hashes the adapter file.
- PR #43 is an audit with 3 high, 6 medium and 6 low findings.
- PRs #20 and #38 preserve research ledgers: tokenizer, spectral-inertia and replay trials, with no model scored.
