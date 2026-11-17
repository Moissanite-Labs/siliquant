# Deviations from the plan, and things to know

No threshold, variant or definition of PREREG.md was changed. What follows is where the run did the closest thing, or where the plan left a choice.

## Choices the plan left open

- The row's largest entry R in the arms `X32sB` is stored as a 16-bit float rounded up, so that R is never below the true largest entry and nothing is clipped. The plan says "stored in 16 bits" and "nothing is clipped".
- The arms `XRo` and `X32o` still count log2(Q) bits for each kept entry's own slot. The plan says "the bits of the plain arm, plus" 16 and the position bits for each kept entry.
- Standard deviation of a row, for the kept entries, is taken over the whole row with divisor n, as the clip probes of the first record do.
- Bits over all block weights count the weights not changed at 16 bits (how a changed copy stores them). `bits_per_weight_changed` counts the changed weights only. `bits_per_weight` equals the second.
- The weight error of this record is the mean and pooled squared error and the mean by kind. The path-weighted errors of the first record are not recorded.
- C4 compares `XRo:32~8` with `XR:32` from the first record (0.1180, not scored again), because the plan lists no `XR:32` for Part C. `XR:16` is the one scored here in the first step.
- A4, B2, B4, C5, D1 to D3 use the `X32:16` and `X32:8` scored in this record. Where the plan gives a number (B1, C1 to C3) the plan's number is used.
- Part F reads KL of the 26 variants from the first record's `results.jsonl`.

## First step

The three scores were 0.000000, 0.117297 and 0.448857. The limits are under 0.00001, 0.117297 within 0.002 and 0.448262 within 0.006. All passed. Evaluation text and reference outputs were rebuilt. The reference perplexity (7.8871) equals the first record's.

## Gates, and which parts ran under which

- Part A ran under the hand rule: the first step was checked by eye before it was started. `run.py` at that commit (c5ef566) did not check it.
- The check was added to `run.py` in commit 2b543c5 and runs before every part but the first step. Parts B, C, D, E and the joined arm ran under the coded check. Part A's job was already running when the file changed.
- The same commit makes Parts B and G refuse unless their `allocate` output is committed, unchanged and on the remote, and records its commit and commit time in `results.start`. Part B ran at head 059fa99, the commit of `allocate.out`. The joined arm's levels are in `allocate_joined.out` (commit 1a70e93, 05:05 local) and were scored at 05:15 local. Commit times in `git log` against the `started` times in `results.start` show the order.
- `results.start` has one line per invocation of `run.py` (seven).

## Reversed arms

The reversed arms differ from the best arms at both budgets (`allocate.out`). The window of 0.05 bit did not need widening.

## Part F

- `measures.py` draws `flip:3.45` with a seed over the 42 matrices, as the first record's `spectral.py` does. The scored `flip:3.45` of the first record used another draw, over 294 matrices. This is a limit on F4.
- A first run of `measures.py` was started while Part A was scoring and was stopped after one variant. Its output file was deleted. Part F was run again, from the start, after all scoring had ended, on the GPU through `tq` with no scoring job running.
- The weights-alone measures ran at the same time as the measures of variants.

## Files that shape results but are outside the gate's list

At every run these were at the commits below. They were not edited in this record.

- `experiments/model-output-gemma-e4b-2026-11-15/run.py`: 7961368
- `experiments/model-output-gemma-e4b-2026-11-15/spectral.py`: eac9d89
- `experiments/model-output-gemma-e4b-2026-11-15/summary.py` (the rank correlation): 7961368
- `persist/persistlib.py`: 5998731
- `lattice/lattices.py`: 195f06e

The gate lists the plan, `run.py`, `allocate.py`, `measures.py`, `summary.py`, `harness/tq.py`, `harness/ggufedit.py` and `harness/evalkl.py`. Each of the other files above is last changed at a commit before the plan.

## Machine

M1 Max, torch on MPS, llama.cpp at 427291b. A scored variant took about three minutes.
