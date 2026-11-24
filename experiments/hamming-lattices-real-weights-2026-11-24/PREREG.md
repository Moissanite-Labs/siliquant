# Plan: lattices from extended Hamming codes, on real weights, and two baselines scored like the ball arms

Committed before any matrix of a model is rounded by the new lattices and before anything of Part A or of Part B on Qwen2.5 or SmolLM2 is scored.

## Where this comes from

A survey on 2026-11-24 of the other organisations' repositories and of the literature since 2025, for anything that could move this repository's open problems. Two of its results are tested here.

1. **Lattices from extended Hamming codes.** The record e8-hadamard-real-weights-2026-11-22 found that E8 turned by the Hadamard matrix of order 8 (E8_Had) is the lowest arm at 4.5 bits on three models. It has E8's squared error and stores far more small weights as zero. E8_Had over the root of 2 is A_8 = C_8 + 2Z^8, with C_8 the extended Hamming code [8,4,4] (AUDIT-115R notes this). The same construction from the longer extended Hamming codes [16,11,4] and [32,26,4] gives lattices A_16 and A_32. In the survey of Foundation-0 (its code tools are in Foundation-0/gfq-codes, but nothing there builds these lattices), an agent computed their cell constants, and this session checked them independently by brute force over all codewords (`cell_check.py`, output `cell_check.out`):
   - Normalised second moment G: A_8 0.07174 ± 0.00008 (E8 is 0.07168), A_16 0.06970 ± 0.00009. That is 0.12 dB better than E8 and 0.78 dB better than the integers.
   - On Laplace data at unit density, the share of inputs under a quarter step decoded to exactly zero: A_8 0.886, A_16 0.969.
   No repository of the organisations and no paper found by the survey uses them for weights.
2. **The baseline is scored differently from the arms.** Found in Moissanite-Labs/ml-os, branch `geminus/2026-11-19-m1-runtime-findings`, `receipts/geminus-2026-11-19/FRONTIER.md` §E2. When llama.cpp multiplies by 4-bit blocks it first rounds the activations to 8 bits. Every ball arm here is written as 16-bit weights, so its activations are not rounded that way, while llama.cpp's `Q4_K_M` rows pay for it. Also from the survey: the organisation trains and serves in bitsandbytes' NF4 (Moissanite-Labs/hillclimb), which no record has compared.

**Seen before this plan, and so not predicted:** a survey agent scored two of Part B's cells on Llama-3.2 with its own scripts. `Q4_K_M` decoded to 16 bits: mean KL 0.041195 ± 0.000573, against 0.043179 for the committed row. NF4 with blocks of 64: 0.110765 ± 0.001523. Those two cells are scored again here by the committed program; they are reported, not predicted. Nothing else of Part A or Part B was seen.

**Pilot, on made-up matrices only** (`pilot.py` on a 512 × 4096 Laplace matrix, seed 0; output `pilot.out`). Pooled weight error at 4.5 bits: E8_Had −25.587 dB, A_16 −25.709, A_32 −25.631. At 3.0 bits: integers −15.889, E8_Had −16.456, A_16 −16.615, A_32 −16.558. Weights under 0.05 of the root mean square stored as zero, at 4.5 bits: E8_Had 87.4 %, A_16 96.7 %, A_32 99.7 %. Build time a weight on one core, at 4.5 bits: E8_Had 1.3 µs, A_16 4.7 µs, A_32 6.1 µs (times vary by a few percent between runs). So a model takes 20 to 35 minutes to build on 4 cores.

## Part A: the arms

`A8BG128`, `A16BG128` and `A32BG128`: A_8, A_16 or A_32 repeated over a group of 128 weights, cut by a ball, with one 16-bit scale per group. The ball is the largest whose count of points is at most 2^B, counted exactly from the weight enumerator of the code (MacWilliams, from its dual RM(1, m)). The bits are (log2 of the count + 16) / 128: B = 560 gives 4.5 bits a weight, B = 368 gives 3.0. Bits are counted, not stored. The scale and the rounding are those of the base program's ball arm (`group_ball_round` of model-output-qwen25-2026-11-17), copied with the nearest-point function, the bound and the margin of the bisection given; a test holds the copy to the base function bit for bit on the integers and on E8. The nearest point of A_n is exact: a dynamic program over the 2^(m+1) partial syndromes, in C, held to a NumPy version bit for bit and to a search over all codewords. `+K10` leaves the ten loud columns of every `ffn_down` whole, as in the earlier records.

The arms they are compared with are the committed rows of e8-hadamard-real-weights-2026-11-22 and of the earlier records: `ZBG128` (integers), `EBG128` (E8), `HBG128` (E8_Had), at the same bits.

**Rows to score** (models, text and score as in the earlier records: wikitext-2 test text, mean KL against the 16-bit model over 10,200 positions, `llama-perplexity` of llama.cpp at `eada89c`):

| Model | 4.5 bits | 3.0 bits |
|---|---|---|
| SmolLM2-1.7B-Instruct | `A8BG128` (the check), `A16BG128`, `A32BG128` | `A16BG128`, `A32BG128` |
| Qwen2.5-1.5B-Instruct | `A16BG128`, `A32BG128` | `A16BG128`, `A32BG128` |
| Llama-3.2-1B-Instruct | `A16BG128`, `A32BG128`, `A16BG128+K10` | |

**Check C1.** `A8BG128:b560s16` on SmolLM2 is E8_Had decoded another way. Its mean KL is within a tenth of a standard error of the committed `HBG128:b560s16` row (0.051522), and its pooled weight error within 0.002 dB of it. If not, the decoder or the copy of the ball arm is wrong and no row of Part A is read.

## Part B: two baselines scored as the arms are

- `Q4_K_M@16`: the `Q4_K_M` file the earlier record scored (checked by SHA-256 against the committed row), every block matrix decoded to 16-bit floats and put in the 16-bit model. The same weights; llama.cpp then multiplies by 16-bit weights, as for every ball arm.
- `NF4B64`: bitsandbytes' NF4 with its defaults, blocks of 64 consecutive weights of a row, the block's largest size in a 32-bit float: 4.5 bits a weight. Block matrices only.

Both on the three models.

## Predictions

A gap is in errors: the difference of two mean KLs over the root of the sum of their squared standard errors. Under 2 errors is a tie. Confidence is stated where it is low.

**Part A, 4.5 bits.**

- **H1.** `A16BG128` is not above `HBG128` by 2 errors on any of the three models.
- **H2.** `A32BG128` is not above `HBG128` by 2 errors on any of the three models.
- **H3.** On SmolLM2, `A16BG128` is under `HBG128` by 2 errors. Low confidence.
- **H4.** On at least two of the three models, the lowest mean KL of the five arms (`ZBG`, `EBG`, `HBG`, `A16`, `A32`) is `A16` or `A32`.
- **H5.** Pooled weight error: `A16` is under `HBG` by at least 0.05 dB, and `A32` by at least 0.02 dB, on each model.
- **H6.** Weights under 0.05 of their matrix's root mean square stored as zero: `A16` is at least 5 points above `HBG`, and `A32` is at least as high as `A16`, on each model.
- **H7.** On Llama-3.2, `A16BG128+K10` is not above `HBG128+K10` (0.025204) by 2 errors.

**Part A, 3.0 bits.**

- **H8.** On SmolLM2, `A16` and `A32` are each under `HBG` (0.629902) by 2 errors.
- **H9.** On SmolLM2, `A32` is not above `ZBG` (0.419643) by 2 errors. Low confidence: at 3.0 bits E8_Had stored 94.4 % of those small weights as zero and still scored far behind the integers, so something other than the smallest weights is at work there.
- **H10.** On Qwen2.5, `A16` is not above `EBG` (0.169338, the lowest at 3.0 bits) by 2 errors.
- **H11.** Pooled weight error at 3.0 bits: `A16` and `A32` are each under `ZBG` by at least 0.4 dB and under `HBG` by at least 0.03 dB, on both models.

**Part B** (Qwen2.5 and SmolLM2; Llama-3.2 was seen).

- **B1.** `Q4_K_M@16` scores under the committed `Q4_K_M` row on both models.
- **B2.** The committed `HBG128:b560s16` row is under `Q4_K_M@16` by 2 errors on both models.
- **B3.** `NF4B64` is above `HBG128:b560s16` by 10 errors on both models.
- **B4.** `NF4B64` is above `Q4_K_M@16` on both models.

## What would change what the repository says

- If H1, H4 and H6 hold, the arm to recommend at 4.5 bits with no data becomes A_16 or A_32 in a ball, and the next work is its numbering and decoder, which the survey found to be simpler than E8's: the count over a codeword depends only on its weight.
- If H8 and H9 hold, the gap at 3.0 bits on SmolLM2 between the integers and the lattices is closed by a lattice.
- If B1 holds, every comparison with llama.cpp's files so far was tilted towards the ball arms by the activation rounding, and the records' gaps to `Q4_K_M` should be read against `Q4_K_M@16`.

## Order of the run

Part B first (three `Q4_K_M@16`, three `NF4B64`), then C1, then `A16` and `A32` at 4.5 bits on the three models, then 3.0 bits, then `A16BG128+K10`. Rows are committed one at a time as they land, by a watcher, before any record is written. If a row fails, the run stops and the failure is recorded.

## Disclosure

Written by an AI agent session under the owner's direction. The lattices are textbook (Construction A, Conway and Sloane, *Sphere Packings, Lattices and Groups*, ch. 5). Their use here was proposed by a survey agent of this session reading Foundation-0's code tools; the activation-rounding point comes from Moissanite-Labs/ml-os, and NF4 from Moissanite-Labs/hillclimb.
