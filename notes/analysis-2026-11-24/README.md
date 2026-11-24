# Analysis of 2026-11-24: what the arms say, a numbering for the new lattices, and what to do next

Written while the run of hamming-lattices-real-weights-2026-11-24 was going. The checks are exploratory. They come after that record's plan, are not part of it, and decide none of its predictions. Every figure here is in one of the `.out` files of this folder. Each is printed by the script of the same name.

| File | What it checks | Needs |
|---|---|---|
| `arms_table.py` → `arms_table.out` | The committed rows at 4.5 bits: weight error, error on the smallest weights, and KL. A two-term fit. The gaps of every ball arm and packed file to `Q4_K_M` and to `Q4_K_M@16` | Committed rows only |
| `index_full.py` → `index_full.out` | `shape/hammingindex.py` at the record's sizes | Nothing |
| `group32.py` → `group32.out` | The new lattices in the block of the packed file, 32 weights in 136 bits: weight error and the cost of numbering | The SmolLM2 file |
| `pball.py` → `pball.out` | Integer levels in a p-norm ball over 128 weights | The SmolLM2 file |
| `hampack_speed.py` → `hampack_speed.out` | Both file formats on a made-up matrix: weight error and decode speed | Nothing |

`arms_table.out` was printed with the rows committed when this note was written. It is printed again when the record is written.

## 1. A numbering for balls of A_16, A_32 and E8 turned

`shape/hammingindex.py` numbers the points of A_n repeated over a group, cut by a ball. `rank()` turns a point into a whole number from 0 to N − 1, and `unrank()` turns it back.

- **How it works.** It walks the same trellis as the record's decoder, with the partial syndrome of the extended Hamming code as its state. The order has two levels: the blocks' squared lengths first, then each block's entries.
- **Tests.** `shape/test_hammingindex.py` checks four things:
  - the number and the order against a full enumeration on small cases;
  - the counts of A_8 against `shape/ballindex.py`'s counts for E8 (A_8 is E8 turned by H8/√8, times √2);
  - the counts at 3.0 bits against the record's `hamming.ball_bound`;
  - a round trip at full size.
- **Every arm of the record can be stored in the bits it counts** (`index_full.out`):
  - 560 bits for the point at 4.5 bits a weight, and 368 at 3.0;
  - tables of 0.14 to 9.3 million entries, using at most 0.39 GB;
  - about 1 ms to number or read back a group of 128 in Python.
- **A_8 numbers E8 turned directly, in coordinates where its points are integers.** So the arm that was lowest before this record also has a numbering now.

`shape/hampack.py` with `shape/hampack.c` now uses a one-level form of the same numbering in a file format (section 2).

## 2. The new lattices fit in the packed file's block, and gain 0.77 dB there

The repository's only file format (`shape/ballpack.py`) stores 32 weights in 18 bytes:
- a one-byte scale code;
- 136 bits for the number of a point of Z^32 in a ball.

The same 136 bits can hold a point of A_16 × A_16, A_32 or A_8^4 (E8 turned) in a ball.

**Weight error at the same 136 bits** (`group32.out`). The scale is the arm's 16-bit one, so compare the lattices with one another:

| Lattice | Gain over Z^32 |
|---|---|
| A_16 | 0.767 to 0.778 dB |
| A_32 | about 0.69 dB |
| E8, plain or turned | about 0.65 dB |

The gains are the same on made-up Laplace and Gaussian matrices and on three matrices of SmolLM2. With one ball per group the error does not depend on the data, as the earlier records found.

**Cost of numbering a block** (`group32.out`, 2,000 random points):

| Lattice | Look-ups to read a block back | Table entries |
|---|---|---|
| Z^32 | 264 | 25,872 |
| A_16 | 323 | 660,776 |
| A_32 | 298 | 2,147,824 |

A C reader would need numbers of 136 bits, three 64-bit words.

**Expected effect.** For the ball arms a decibel of weight error has gone with about a decibel of KL. Going from a block of 128 to a block of 32 cost about 0.5 dB of weight error and 5 to 10 % of KL. On that basis A_16 in the file's block should cut the file's KL by 10 to 20 % at the same size. That is a guess from two earlier records. It is the first of the proposals below.

**Built since** (`shape/hampack.py`, `shape/hampack.c`, `shape/test_hampack.py`). It is ballpack's file with A_16, A_32 or A_8 in the same 18-byte block, read back exactly.

On a made-up Laplace matrix of 1024 × 4096 (`hampack_speed.out`), measured with the other cores busy:

| File | Weight error | Decode on one core |
|---|---|---|
| Z^32 (ballpack) | −24.439 dB | 20.8 million weights a second |
| A_16 × A_16 | −25.240 dB | 10.2 million weights a second |

The new file decodes at half the speed because its table of counts is 49 times larger. A layout that keeps a coordinate's two state rows together, and drops the half of the squared lengths whose parity the state rules out, should win some of that back.

The plan to score it on three models is `experiments/packed-hamming-2026-11-24/PREREG.md`, committed before any model was packed.

## 3. A p-norm ball does not help over 128 weights

Over 8 weights a p-norm ball with p = 5/4 or 3/2 beat the ball by 0.2 to 0.3 dB (shape-formula-held-out-2026-11-17). Over 128 weights (`pball.out`):

| Data | Result |
|---|---|
| Laplace (made up) | Smaller p gains up to 0.60 dB (p = 1) |
| Gaussian (made up) | The ball is best |
| `blk.1.ffn_down`, `blk.5.ffn_down` | The ball is best |
| `blk.10.attn_q` | p = 3/2 is 0.045 dB ahead |

Real groups of 128 weights behave like the Gaussian. One scale per group already takes up the tails. The avenue is closed for long groups.

## 4. What decides the output at equal bits

At 4.5 bits every lattice arm has a pooled weight error within 0.12 dB of the others; integer levels are 0.66 dB worse (`arms_table.out`, table 1). What differs between the lattice arms is the error on the smallest weights, those under 0.05 of their matrix's root mean square.

**SmolLM2 follows that error.** The mean KL falls with it:

| Arm | Error on the smallest weights | Mean KL |
|---|---|---|
| E8 | 4.92 dB | 0.060861 |
| E8 turned | 3.75 dB | 0.051522 |
| A_16 | 1.59 dB | 0.047957 |
| A_32 | 0.19 dB | 0.045800 |

**So does Llama-3.2, among the lattice arms.** Its small-weight error falls from 4.64 to 3.48 to 1.45 dB across E8, E8 turned and A_16, and its KL from 0.040161 to 0.036038 to 0.034455.

**Qwen2.5 does not.** E8 turned, A_16 and A_32 score 0.018875, 0.018974 and 0.019032, all within 0.3 errors, while their small-weight error falls from 3.15 to −0.13 dB.

**A fit of KL to the two errors** (table 2), over all the ball arms with integer levels among them:
- It fits SmolLM2 to within 2.4 errors.
- On Llama-3.2 it misses by up to 5.2 errors. Integer levels have the least error on the small weights and the worst KL of the four there, so no one line holds them and the lattices.
- It gives the small weights no part on Qwen2.5.

So no one weight figure predicts the output across models. That is the same lesson as `notes/warp-drive-2026-11-24.md` draws for the certificate that picks by predicted weight error.

**Open question:** on Llama-3.2 the 0.66 dB of pooled error that integer levels give up costs more than their zeros buy: they are behind every lattice arm. On SmolLM2 they are still ahead of plain E8. Why the difference?

## 5. Against the fairer baseline the leads hold, smaller

`Q4_K_M@16` is `Q4_K_M`'s own weights multiplied at 16 bits, with no 8-bit activations, as the ball arms are scored (table 3).

**Every lead of a ball arm or packed file over `Q4_K_M` survives, by a smaller margin:**

| Model | Leads shrink by | Note |
|---|---|---|
| Qwen2.5 | 0.8 to 1.4 errors | |
| SmolLM2 | 4.9 to 5.4 errors | |
| Llama-3.2 | 1.9 to 2.3 errors | |

**On Llama-3.2 three arms change standing:**
- plain E8 becomes a tie, at 1.2 errors;
- integer levels, already behind, fall further behind (7.2 errors);
- the packed file with no kept columns is also further behind (11.3 errors).

**The correction runs both ways.** The ball arms were scored with 16-bit activations. An engine that ran them on 8-bit activations would charge them the same tax that `Q4_K_M` pays in its committed rows. The fair deployment comparison puts both sides on the same activations. Proposal 3 measures the other side.

## Proposals, ranked

1. **A_16 in the packed file's block.**
   - Same 18 bytes, same block of 32, same one-byte scale code.
   - The point becomes one of A_16 × A_16 in a ball of squared radius 1207, in place of Z^32 in a ball of 783.
   - Parts:
     - The numbering exists (`shape/hammingindex.py`).
     - The nearest point exists (the record's `hamming_dp.c`).
     - The reader needs the trellis walk in C with 136-bit counts.
   - Score it on three models with and without ten kept columns, against the files of packed-kept-columns-2026-11-19.
   - Expected: 10 to 20 % lower KL at the same size, if the decibel rule holds. It would be the first file with a lattice in it.
2. **Kept columns with the new lattices on all three models.** H7 of the running record tests A_16 with kept columns on Llama-3.2 only. On SmolLM2 and Qwen2.5 the columns cost nothing for integer levels, but nothing says what they do for A_16 or A_32.
3. **The activation tax on the ball arms.** Store the dequantized weights of `HBG128` and `A16BG128` as `Q8_0`. llama.cpp then rounds the activations to 8 bits, as it does for `Q4_K_M`. Compare with the same arms at 16 bits. The weight rounding of `Q8_0` adds about −45 dB, far below the arms' −25.6. This completes the fairness correction from the other side.
4. **Decoding that favours zero.**
   - Weight each coordinate's squared error in the trellis by 1 + λ when the weight is under a set share of its matrix's root mean square. This is exact, because the dynamic program takes any per-coordinate weight at no cost.
   - λ moves an arm along the line from A_16 toward the integer arm's share of zeros, at a small cost in pooled error.
   - On SmolLM2 the output followed the error of the small weights; on Qwen2.5 it did not.
   - Plan two values of λ on SmolLM2 and Llama-3.2 before running them.
5. **Faster builds** (not on the critical path).
   - A build makes 14 bisection steps and up to 12 shrink steps, each a full decode of the matrix.
   - Bracketing the scale from the covering radius would cut this to about 6 decodes.
   - Running the trellis over its states in vector registers should gain a few times more.
   - Today a model takes 10 to 20 minutes on four cores.

**Closed by the checks above or by the survey:**
- p-norm balls over 128 weights;
- column weights from the weights alone (`notes/cross-org-survey-2026-11-24.md`).
