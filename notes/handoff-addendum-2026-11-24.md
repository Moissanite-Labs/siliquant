# Addendum to the Gemma handoff, 2026-11-24

Read [`experiments/model-output-gemma-ball-2026-11-23/HANDOFF.md`](https://github.com/Moissanite-Labs/siliquant/blob/main/experiments/model-output-gemma-ball-2026-11-23/HANDOFF.md) on `main` first. It remains the operating guide for the M1 Max work: runner commit `ccf045d`, the Gemma closure queue, the 64 GB limits, and the methodology rules.

This note adds what was learned after it was written. That work is on branch `bayan/charming-gates-jlas14` and is not yet on `main`. Every figure below is in a committed output on that branch.

## What changed

1. **A new arm is ahead with no data: A_16 in one ball per 128 weights.**
   - **What it is.** A_16 is the lattice of the extended Hamming code [16, 11, 4]. E8 turned by H8 (`HBG`) is the same construction from [8, 4, 4].
   - **Results.** At 4.5 bits it is the lowest arm, or tied with the lowest, on Qwen2.5, SmolLM2 and Llama-3.2:

     | Model | Mean KL |
     |---|---|
     | Qwen2.5 | 0.018974 |
     | SmolLM2 | 0.047957 |
     | Llama-3.2 | 0.034455 |
     | Llama-3.2, ten columns of each `ffn_down` kept | 0.024053 |

   - **At 3.0 bits** it and A_32 beat integer levels on SmolLM2, where every E8 arm had been far behind.
   - **Where:** `experiments/hamming-lattices-real-weights-2026-11-24`.
   - **Code:**
     - `hamming.py` holds the exact nearest point. `hamming_dp.c` is its C version, held bit for bit to NumPy.
     - `shape/hammingindex.py` numbers the points of the ball.
2. **The `Q4_K_M` comparisons were tilted towards the ball arms, on the CPU backend.**
   - llama.cpp rounds the activations to 8 bits when it multiplies by `Q4_K` blocks. The ball arms are written as 16-bit weights and never pay that.
   - Scored the same way (`Q4_K_M@16`, the same weights decoded to 16 bits), `Q4_K_M` is lower than its committed rows by 1.8, 5.6 and 2.4 errors.
   - Every lead of a ball arm still holds, by 0.8 to 5.4 errors less.
   - Those runs used the CPU backend with no `-ngl`. **The Gemma runner scores on Metal (`-ngl`), whose kernels for `Q4_K` may not round the activations the same way. Measure it, don't assume it.**
3. **A file format with A_16, the same size as the integer file.**
   - `shape/hampack.py` with `shape/hampack.c` uses the 18 bytes per 32 weights of `shape/ballpack.py`, with A_16 × A_16 in place of Z^32. Every matrix reads back exactly.
   - On Qwen2.5 and SmolLM2 its mean KL is 0.83 to 0.89 of the integer file's, 4.6 to 7.3 errors lower, at the same byte count.
   - Its decode is about 2.5 times slower on one core: the table of counts is 30 MB against 0.6 MB.
   - Record: `experiments/packed-hamming-2026-11-24`. The Llama-3.2 rows were still running when this was written.
4. **No single weight figure predicts the output across models** (`notes/analysis-2026-11-24`). Pick arms by measured KL, not by predicted weight error.

## Changes to the queue

- **1a. Add `Q4_K_M@16` and `Q4_K_M-imat@16` rows** to the tensor-matched stock reruns on `gemma-3-4b-it` and `gemma-3-12b-it`.
  - These are the same weights decoded to 16 bits, as `experiments/hamming-lattices-real-weights-2026-11-24/baselines.py` does.
  - Record the backend, `-ngl` and thread count in every row.
  - This shows how much of each Gemma lead over the stock files is the activation rounding, on the backend this machine uses.
- **1b and 1c. Add A_16 next to `hEBG128` and `hHBG128`:**
  - `A16BG128:b560s16` with no data;
  - `hA16BG128:b560s16` with the half-strength importance matrix, if the runner applies column weights before the arm's rounding.

  The Gemma runner (`GROUP_BALL_RE`, line 77 on `exp/gemma-ball-runner`) knows `ZBG`, `EBG` and `HBG` only. A port needs three things from `hamming.py`:
  - `closest(X, 4)` as the nearest point;
  - `ball_bound(4, 128, 560)` for the bound and the count of points;
  - `sqrt(covering_sq(4, 128))` as the bisection's margin, in place of `sqrt(128)/2`.

  Write a preregistration before scoring, as for every arm. On Gemma, turning E8 did not help, so A_16 there is an out-of-family test of the result above. Gemma would be its fourth model family.
- **Priority 3, packed-format benchmark.** Benchmark the A_16 file (`shape/hampack.py`) alongside `shape/ballpack.py`.
  - Measure file bytes, decode speed on one performance core, and resident memory.
  - The M1 Max's large caches may change the 2.5 times gap in decode speed, which comes from the table's size.

## On the M1 Max

- **Run the tests first:**

  ```bash
  python3 -m unittest experiments/hamming-lattices-real-weights-2026-11-24/test_hamming.py shape/test_hammingindex.py shape/test_hampack.py
  ```

  - The C files compile with `cc -O2 -ffp-contract=off`. Apple clang fuses multiply and add by default, which would break the bit-for-bit parity between C and NumPy. The flag is in the build commands, and the tests check the parity.
- **The run programs of the 2026-11-24 records were run on Linux, where `multiprocessing` forks.**
  - They set the model path in the parent process before starting the worker pool. Under macOS's default `spawn`, the workers would not see it.
  - Either port the arms into the Gemma runner, which was written for macOS, or set the start method to `fork` explicitly and run a `--only` test first. Neither has been tried on macOS.
- The machine restarts described in these records were the cloud container's. They do not apply to a local machine.
