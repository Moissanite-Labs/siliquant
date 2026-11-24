# Cross-org survey of 2026-11-24: what the other repositories and the literature offer this one

The owner asked what the organisations' work since this repository's last record, and the field's, could unlock here, and to look back 72 hours, open pull requests and branches included.

## How it was done

- **Reach.** 175 repositories in Moissanite-Labs, Foundation-0, Proven-Commons, aska-digital, aska-physical, siliconcordat and a few personal accounts.
  - Read:
    - Every pull request updated from 2026-11-21 to 2026-11-24: about 153 in Moissanite-Labs, about 130 in Foundation-0 across some 25 repositories, and a handful elsewhere.
    - Every branch with commits in that window, in 18 repositories near this work.
  - Not reached:
    - The issues of aska-digital/ingestion (not open to this session).
    - Moissanite-Labs/statos.
    - The GitHub API of siliconcordat and of the added Foundation-0 repositories. Those were read through git instead.
- **Who read what.** Five survey agents of this session, read-only, each with one part:
  - this repository's own `studies/` lane;
  - Foundation-0's mathematics;
  - Moissanite-Labs' machine-learning repositories;
  - siliconcordat;
  - the literature since 2025.

  Where an agent ran a check, this note says so. What this session checked itself is said plainly, with the file that holds it.

## The leads, ranked, and what was done with each

**1. Lattices from extended Hamming codes. Being tested now.**
- E8 turned by the Hadamard matrix of order 8 is the arm that leads at 4.5 bits (e8-hadamard-real-weights-2026-11-22). Over the root of 2 it is C_8 + 2Z^8, with C_8 the extended Hamming code [8,4,4]; `studies/audit115r-e8had-simd-ballpack` says so too.
- The longer extended Hamming codes [16,11,4] and [32,26,4] give lattices A_16 and A_32 by the same construction. A_16's cell is 0.12 dB better than E8's. On Laplace data it decodes 97 % of the small inputs to exactly zero, against 89 % for E8 turned.
- The idea came from the agent that read Foundation-0's code tools (Foundation-0/gfq-codes). No repository there builds these lattices.
- This session checked the cell constants by brute force over all codewords (`experiments/hamming-lattices-real-weights-2026-11-24/cell_check.out`).
- The record hamming-lattices-real-weights-2026-11-24 scores them on three models at 4.5 and 3.0 bits. Its plan was committed at `aec439b`, before any model matrix was rounded by them.

**2. The `Q4_K_M` rows pay for something the ball arms do not. Being tested now.**
- llama.cpp rounds the activations to 8 bits before it multiplies by 4-bit blocks. Every ball arm is written as 16-bit weights, so its activations are not rounded that way. The point was found in Moissanite-Labs/ml-os, branch `geminus/2026-11-19-m1-runtime-findings`, `receipts/geminus-2026-11-19/FRONTIER.md` §E2.
- A survey agent decoded the `Q4_K_M` file of Llama-3.2 to 16-bit weights. Mean KL was 0.041195 against 0.043179 for the committed row, so about 4.6 % of `Q4_K_M`'s KL there is the activation rounding. E8 turned stays ahead (0.036038), by about 6 errors in place of about 8.5.
- Part B of the same record scores this on all three models with a committed program. It checks each `Q4_K_M` file against the hash of the row it stands for: this session regenerated the Qwen2.5 and SmolLM2 files byte for byte.

**3. NF4, the organisation's own 4-bit format, has never been compared. Being tested now.**
- Moissanite-Labs/hillclimb trains and serves Qwen3-4B in bitsandbytes NF4.
- A survey agent scored NF4 with its defaults (blocks of 64, 4.5 bits) on Llama-3.2: mean KL 0.110765, three times E8 turned's at the same bits. Part B scores it on all three models.

**4. A packed format and decoder for E8 in a ball over 32 weights exists.** PRs #35, #36 and #39, merged: `studies/expedition-ballpack-tree-unrank-audit112r`, `audit115r-e8had-simd-ballpack`, `audit116r-e8-leaf-stage1-attack`. A survey agent found:
- **Same codebook:** its count of points equals `shape/design.ball_bound("E8", 32, 136)`, and 300 of 300 points it unranks are in `shape/ballindex.BallIndex`.
- **Different numbering:** only 1 rank in 300 agrees, so its files are a new format.
- **Decode speed:** on one core here it decodes 37 to 42 million weights a second, against 24 to 31 for `shape/ballpack.c`; a compact 88-bit variant reaches 62 to 75.
- **What it is not:** it covers blocks of 32 weights, not the groups of 128 behind the best arms. It is not wired into `shape/ballpack.py`, and it has been run on made-up matrices only.
- **Next:** score `EBG32`/`HBG32` on SmolLM2 and Llama-3.2, about one to two hours. Then read a file back bit for bit through it.

**5. A cheaper turn gives the same lattice.**
- PR #42, merged (`studies/expedition-e8had-anisotropy-audit119br`): (I_4 ⊗ H_2)/√2 generates the same lattice as H_8/√8, with 8 additions a block in place of 24.
- A survey agent checked 240 of 240 roots and 2,000 random points. On a real matrix of one million weights both turns gave the same 16-bit weights in 99.998 % of places and the same error.

**6. A decoder exact on every machine needs no special arithmetic.** A survey agent found, starting from siliconcordat/weft's `src/weft/cross_isa.py` and siliconcordat/loom's `docs/04`:
- 3,000 adversarial groups of 128 whole-number coordinates (up to 8) times 16-bit activations all sum exactly, and in any order, in 64-bit floats. In 32-bit floats 2,866 of them depend on the order.
- Every point of E8 times H_8 is whole (25,617 points checked), so the 1/√8 can sit in the group's scale.
- A decoder that is the same bit for bit on every machine is then possible without the Ozaki splitting weft uses.

**7. Gemma 3 at four sizes, measured on an M1 Max.** Branch `exp/gemma-ball-results`, which has no pull request, and PR #75.
- From the rows, as a survey agent read them:
  - E8 in a ball is ahead of integer levels in a ball at all four sizes.
  - Turning E8 makes no difference on Gemma.
  - Keeping ten loud columns does not help.
- So "turning helps" holds on SmolLM2 and Llama-3.2 and not on Gemma. These rows are provisional; see the red flags.

**8. From the literature, not tried here** (from the literature agent; arXiv numbers in its report, not re-read here):
- **For the Voronoi code** (open problem 2): NestQuant's union of four scales per block of 8 (2502.09720), the known fix for overload of an E8 Voronoi code.
- **For numbering E8 points:** RE8's absolute-leader indexing from the AMR-WB+ and EVS speech codecs.
- **For allocation:** per-tensor bits from the "linearity theorem" of HIGGS (2411.17525), with each tensor's sensitivity from noise injection on random tokens.
- **For coding:** entropy-coded lattices (HyperQuant, 2606.23406; EntQuant, 2601.22787), and μ-law companding before the lattice (GLVQ, 2510.20984).
- **For rotation:** per-tensor choice of rotation, after MR-GPTQ's analysis that rotation helps large groups and hurts small ones (2509.23202).
- **Data-free comparisons to add:** HIGGS. **Calibrated:** WaterSIC (2603.04956), which reports Llama-3.2-1B; ik_llama.cpp's IQ4_KS and IQ4_KT.

**9. The importance matrix from random tokens may be capped.**
- siliconcordat/weft, experiment E5, on TinyStories-1M: the random-token activations have an effective rank of 2.4 against 4.0 for real text.
- Recovery of the real matrix's gain stays at 52 to 55 % from 64 to 2,048 sequences; 16 real examples reach 112 %.
- Not checked here, and the model is tiny. It would explain why noise buys 69 to 96 % of the gain here and not all of it.

## Checked here, and negative

- **No figure of the weights alone predicts the importance matrix on these models.** `notes/colproxy.py`, output `notes/colproxy.out`; exploratory, no plan committed first.
  - The literature proposes 1/(column root mean square)² (SINQ, 2509.22944). The other candidate is the RMSNorm gain before a matrix.
  - Their rank correlation with each column's measured mean squared input, median over layers:
    - Column size: −0.70 to −0.60 on Qwen2.5's `attn_k`, `attn_q` and `ffn_gate`, which is the wrong sign. It is weak elsewhere, best +0.65 on Llama-3.2's `ffn_up`.
    - Norm gain: +0.63 on Llama-3.2's attention matrices, near zero or negative on the other two models.
    - The matrix from random tokens: 0.24 to 0.66 everywhere.
  - This repository's weighted arms already scale each column by a power of the matrix before rounding (`column_weights` in model-output-qwen25-2026-11-17), which is what the literature calls reverse waterfilling. The only new part would have been doing it with no data, and that does not hold here.
- **Nothing in siliconcordat is lattice work.** "E8" there is Experiment 8, "Hadamard" the elementwise product, and "Latent Lattice" a language-model architecture.
- **No engine or kernel in the organisations reads a lattice format.** Moissanite-Labs/mlx PR #4 is a Metal template for a plain matrix-vector product, and its own certificate says it never ran on Apple hardware. warp-drive's names for siliquant formats (PR #40) refer to formats nothing loads.
- **Foundation-0's mathematics gives no codebook.**
  - point-charges holds equilibria of fixed charges, not spherical codes.
  - The Niemeier lattices of Foundation-0/ramanujan are worse quantisers than A_16 (a survey agent's Monte Carlo agrees with their stated values).
  - BW16 and the Leech lattice are built and checked there, but they keep fewer small weights at zero: 0.77 and 0.47 in the agent's test, against 0.97 for A_16.
- **The two-stage E8 table of PR #65 loses about 6 dB.** A survey agent measured it against the integer ball at 4.0 bits on a real matrix (whisper-base, squared error only).

## Red flags for the team

Found by the survey agents in their reading. This session did not check them again.

- **The Gemma record is not on main.**
  - `experiments/model-output-gemma-ball-2026-11-23` holds only its plan. Its rows are on branch `exp/gemma-ball-results`, which has no pull request.
  - A survey agent found that the llama.cpp baselines there changed tensors the ball arms left alone: `token_embd` from BF16 to F16 at 4B and 12B, and probably `per_layer_token_embd` at E4B, 37.6 % of that model. The fixed runner (`ccf045d`) has no rows yet.
  - PR #71 labelled this evidence as from an unmerged branch, and PR #73 removes the labels.
- **Stale figures.**
  - `docs/status` and the comments on #23 and #25 give the E8 tables as 70.8 KB and 56.9 KB; the corrected figures in the study are 172.7 KB and 154.0 KB.
  - "E8_Had at 4.47 bits" in `docs/status` and PR #67 is a made-up arm of draft PR #24. The measured arm is 4.5 bits.
- **PR #67's brief.** It labels the importance-matrix figures as results on Python text. Its "2.79 billion weights zeroed at 0 KL" are embedding rows the text never reads.
- **"VERIFIED" in `studies/`** means the producing team's own protocol passed: scripts that reproduce their own hashes, and mutation tests. No study re-runs a model. Foundation-0's outside check (cave-expedition #4) lowered most "N2" labels.
- **Circular evidence.**
  - siliconcordat/loom's notes and weft's `ARCHITECTURE.md` cite this repository's PR #12 figures as "convergences".
  - warp-drive's `studies/expedition-weight-iteration-bridge` hardcodes this repository's constants and then checks them against themselves.
- **A dependency on this repository's code.** The warp-drive branch `hackathon/cerebras-siliquant-bridge` imports `shape/check_certificate.py` and needs `check`, `identities` and `pi_bounds_hold`. Renaming them breaks it.
- **Speed figures.** The "Q4" speed baselines in `studies/` are a custom nibble proxy, not llama.cpp's kernels.

## How the approach has changed since 2026-11-22

- **From counted bits to decoders.** E8 in a ball over 32 weights now has a codec in C, a cheaper turn, and a decode that moves the turn onto the activations. All of it runs on one CPU core and is far behind a 4-bit kernel. The case made for it shifted to exact TPU and edge rooflines, which nobody has measured.
- **Real-model work moved to Gemma** on an M1 Max, with the importance matrix on by default. There the lead arm is the integer ball with the matrix (`hZBG128`), and turning E8 did not help.
- **Most new output is exact-arithmetic studies.** They prove properties of codebooks, and none scores a model.

This note's own contribution is the two tests above and the negative check of data-free column weights. Their results go in the record when the run ends.
