# What became of the grafts

Date: 2026-11-17
Status: the five grafts ranked first are done or closed. The rest are read and sorted, not run.

[notes/cross-org-grafts.md](cross-org-grafts.md) crossed this repository's methods with mechanisms from other repositories and ranked fifteen crossings, X1 to X15. Nothing in it was run. This note says what was then done with each, in its order of rank. It is the end of that thread unless a row below says otherwise.

## The five ranked first

| | Crossing | Outcome | Where |
|---|---|---|---|
| X2 | A check that each plan was committed before its data | **Done, and it works.** Read from git, for every record. Every record with data has its plan first. It found one gap: the gemma record changed its plan twice after its first data and named neither commit. The record names them now | `checks/plan_before_data.py`, `checks/plan_before_data.out` |
| X3 | A trace of every number in a README to a committed output | **Done as a check after the fact, and it does not do the job.** When first run it found 1,910 of 1,912 numbers, and it found 1,211 of 1,642 numbers changed on purpose. A stricter pass by table row caught 102 of 202 changes. `checks/README.md` has the figures for the records since. The form that would work is the one the crossing names: tables printed by a program and copied. That is a change to how a record is written | `checks/number_trace.py`, `checks/number_trace.out`, `checks/README.md` |
| X1 | A witness for every cell of a closure | **Done.** `closure()` dropped its reasons. `closure_witnessed()` gives each closed cell a witness and each open cell a perfect matching through it, and `verify_closure()` checks both from the board alone. Tested against enumeration and on 15 witnesses made wrong | `perm/permlib.py`, `perm/test_permlib.py` |
| X5 | A gate that bounds the change from the saved baseline | **Does not apply as written.** The gap it guards against is drift over many steps, each measured from the last. The harness has no steps: every variant is rounded once from the reference file, and its weight error against that file is recorded for every matrix. What carries over is a bound known before the run. Two exist: the scale with a proved guarantee in `shape/certified.py`, and the error of a code with one ball per group, which is the same share of every group's size whatever the weights (the record ball-dimension-no-clip-rule-2026-11-17) | no code |
| X4 | A fixed list of kinds for the Basis of each checked claim, and a file on each recomputed row | **Done.** Read from the check notes as they stand: 105 rows in three notes when first run. Eleven rows said recomputed or proved and named no file. Each names one now, among them the 1.26/n row the crossing found | `checks/claim_ledger.py`, `checks/claim_ledger.out` |

## The correction the note asked for

The grafts note says the verdict on claim 8 of the briefs should be looked at again: a reader reported that Laman's count is the right test of rigidity and is coded. Checked: `verify_laman` is in `scripts/verify_note_schema_2d.py` of rigid-realizations at commit `298926c`. The verdict stands, "wrong as stated", and the row of [brief-check.md](brief-check.md) now names the test that is right.

## The other ten

Sorted by what they would give this repository's question, which is how to round weights. None was run.

| | Crossing | Read |
|---|---|---|
| X13 | Bits stored against bits counted | **Partly answered.** For the codebooks cut by a ball, `shape/ballindex.out` gives the whole bits a block's number takes beside log2 of the count. For llama.cpp's files the model-output records give the bits of the file. No record yet stores a rounded model in a packed file of its own |
| X10 | Rank the variants by the largest singular value of the change, not by squared error | **Open, and worth one table.** The gemma record did it: rank correlation 0.889 against 0.603. The run on Qwen2.5 records weight error only. Its record gives the rank of weight error against the output over its own variants |
| X11 | Choose which weights to protect by what relaxing them gains | Published mathematics (Optimal Brain Surgeon). It needs second-order statistics of the inputs, which no arm here uses. It belongs with a study of rounding that reads calibration data, and that study is not planned |
| X12 | Zero the tensors that no output reads and measure no change | A property of one model's graph. Reported for gemma. Not looked for in Qwen2.5 |
| X6 | One file of decoder cases read by both the Python and the GPU code | Tidy, small, and no result depends on it. `harness/test_tq.py` already compares the two on random blocks |
| X7 | A nearest-point rule proved on a box of inputs | Longer than a day. The E8 rule's proof is published. No record rests on an unproved decoder |
| X8 | A certificate for banded rook numbers that holds for every n | About counting matchings, not rounding. Left to `perm/` |
| X9 | Delsarte ceilings for ternary codebooks | A replayable number that is already in print. Left |
| X14 | A bound on the change of a model's output from the size of the rounding error | Needs a toy first. The gemma record found that no formula tried predicts the damage. Left open |
| X15 | A certificate that the best step size is unique | Open in the source too. Left |

## Disclosure

Written by an AI agent session under the owner's direction, on 2026-11-17.
