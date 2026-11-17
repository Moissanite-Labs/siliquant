# Cross-org grafts: a transposition pass

Date: 2026-11-17
Status: idea map. Nothing here has been run, and no code has changed.

Each claim is marked. **Checked** means the source was re-read in this session. **Reported** means it comes from a read-only reader's pass and has not been re-verified here.

## Scope

- **Triage of all five orgs.** About 230 repositories across Moissanite-Labs, aska-digital, aska-physical, Foundation-0 and Proven-Commons were triaged from name, description and topics. GitHub search excludes forks, so forks were taken from the access list. Personal accounts are out of scope.
- **Abstract reads of about 30 repositories**, in five read-only passes. Each repository was reduced to a one-line abstract shape and matched against siliquant's concepts. Weft and weft-loom, the sibling repositories the closure rule comes from, were read together.
- **Not read:** mlx and statos (internal); dabba (private); divan-bus (not cloned); most of the triage list; incentre-lattices and f0 only at index level. One private repository in aska-digital is mentioned here only generically.

## Method: the transposition

Each siliquant concept is crossed with each mechanism from another repository. For each crossing there are two questions. What does the other mechanism do to siliquant's objects (weights, bits, matchings, proofs)? And what does siliquant's method do to the other repository's objects? A crossing is kept only if it gives a first test that can run in about a day, or one clearly labelled as longer.

**Siliquant concepts (rows)**

- S1. Block lattice quantiser (E8, A8\*) with nearest-point decoding.
- S2. Exact counting of matchings that avoid forbidden pairs, and bounds on permanents.
- S3. The table of proved limits, read against every proposal.
- S4. Closure: a path is closed only on a certificate.
- S5. Preregistered study: plan committed before data; null results kept.
- S6. Every number traces to a committed output.
- S7. Row and column balancing before rounding.
- S8. Weight error as the proxy; output error as the target.
- S9. Claim-by-claim source check of the briefs (`notes/brief-check.md`).

**Other mechanisms (columns)**

- W1. Bounded change: accept an update only if ‖W − W₀‖_F ≤ R from a saved baseline, checked at every commit. (weft, `runtime/invariants`) **Checked:** the plasm call passes the baseline (`plasm.py:423`).
- W2. A budget is checked before commit, and a rejected update spends nothing. (weft, `invariants.py`) **Checked.**
- W3. Two candidate updates are scored exactly, and the winner must pass the gate. (weft, `plasm.py:370-440`) **Checked.**
- W4. One vector file of 27 conformance cases, read by independent implementations. (weft, `invariants/vectors.json`) **Checked:** 15 + 8 + 4 cases.
- W5. Dead-weight closure: parts of the computation graph that no output reads can be zeroed with no change in output. (weft-loom, `notes/closure-on-a-real-model.md`) **Reported** on the gemma model.
- W6. An output-change bound from a Lipschitz-type theorem, evaluated by recomputation. (weft-loom, `gate-metric-calibration`, `THEORY.md`) **Reported.**
- W7. A register checker that recomputes every certificate and exits nonzero on failure. (weft-loom, `check_closures.py`) **Checked** that the file exists.
- W8. Damage ranking by spectral norm: rank correlation 0.889, against 0.603 for squared error, over 26 variants. (weft-loom) **Reported.**
- W9. A shortfall certified by a minimum cut. (weft, `synaptic_flow.py`) **Reported.**
- O1. Delsarte LP ceilings on codebook size, replayable from checker code. (gfq-codes) **Reported.**
- O2. A zero-vector recurrence: a banded count holds for all n once a state vector reaches zero. (finite-state-proofs) **Reported.**
- O3. A certified enclosure by Krawczyk boxes with a Lipschitz row-sum term. (rigid-realizations) **Reported.**
- O4. Interval subdivision with a checked index identity. (point-charges) **Reported.**
- O5. Schur-complement relaxation: the stiffness lost when some joints are relaxed. (fibral-rigidity) **Reported.**
- O6. Density-survival ratio: realised saving over nominal saving. (tokenpage charter) **Reported.**
- O7. A canonical registry: a validator rejects unknown codes and non-canonical forms. (abios) **Reported.**
- O8. Approval bound to a hash of the reviewed text, plus rerun-and-diff of committed records. (mission) **Reported.**
- O9. A closed-enum verdict ledger validated by a standard-library script. (formal-claims-audit) **Reported.**
- O10. Schema-fixed append-only records checked by stateless gates. (protean-ops) **Reported.**
- O11. Generate a table from the data, then assert equality. (protean-drafts discipline) **Reported.**
- O12. Certificate-or-nothing: a claim counts only if its witness verifies, with forbid-overrides precedence. (tar, cedar) **Reported.**
- O13. Basis without action: a "recomputed" claim must cite a command and an output file. (cmd-narration-guard) **Reported.**
- O14. Tags: `[F]` read from source, `[I]` inference. (siliconcordat) **Reported.**

## Crossings

| # | Crossing | What it gives | Evidence | First test |
|---|---|---|---|---|
| X1 | S4 × O12, W7 | A witness for each closed cell, re-verified by a checker | **Checked:** `perm/permlib.py` `closure()` returns cells with no witness. With a Hall violator it returns all n·n cells, also with no witness. | Emit a witness per closed cell; a test re-verifies each one. About a day. |
| X2 | S5 × O8 | A preregistration gate: refuse outputs unless the plan's hash matches and the approval commit is an ancestor of the first output commit | **Checked:** the model-output `PREREG.md` changed at `e2ff142` (19:56) and `7961368` (20:19), both after the first results commit `eac9d89` (19:52). The READMEs say only "extended five times" and name no amendment commits or times. | A script over git history, replayed on this study. About a day. |
| X3 | S6 × O11, O8 | A generated number trace: every number in a README is generated from a committed output, or listed with its derivation | **Checked:** experiments/model-output-gemma-e4b-2026-11-15/README.md line 15 gives mean KL 0.075. `results.jsonl` row `VE832:16` has `kl_mean` 0.074728. **Reported:** a naive resolver matches 85 of 90 decimal tokens in the lattice README after rounding. Coincidental matches are possible, and a derivation allowlist is needed. | Generate one README table from `results.jsonl` and diff it. About a day. |
| X4 | S9 × O9, O13, O14 | A closed-enum ledger for `brief-check.md`, with a path or command on each "recomputed" row | **Checked:** claim 10 cites "1.26/n". The literal `1.26/n` appears in `brief-check.md:32` itself and in unrelated `.out` figures, but in no `.py` source. `stein_chen_bound` at `perm/permlib.py:761` computes 2(1 - 1/e)/n = 1.264241/n on the forbidden-diagonal board, so claim 10's constant is reproducible. | Convert the 19 rows to JSON; a validator rejects a "recomputed" row with no output file. |
| X5 | S3 × W1 | A bounded-change gate against the saved baseline, not the last step | **Reported:** `harness/tq.py` has no gate. **Checked:** weft gates against the baseline. **Checked:** weft-eido's gate (`src/research-explainer.js:116`) measures from the current matrix, so its bound is per step and cumulative drift is unbounded. Do not copy that one. | Gate each quantised matrix by ‖Ŵ − W‖_F ≤ R, then check cumulative drift on a toy. |
| X6 | S1 × W4, O7 | One vector file of lattice decoder cases, read by the Python decoder and the GPU path | **Reported:** `harness/test_tq.py` compares the GPU against Python only. | A JSON file with E8 and A8\* cases, read by both paths. |
| X7 | S1 × O4 | A certified decoder: the nearest-point rule proved exact on a box of real inputs, not only tested against brute force | **Reported:** `lattice/` is tested against brute force. Decoder correctness is a claim over every real input in a box, so subdivision is the natural form. | Certify the E8 rule on one cell, then on a box. Longer than a day. |
| X8 | S2 × O2 | A zero-vector certificate for banded rook numbers, valid for all n | **Reported:** the board DP is in `perm/`. The transfer matrix would run over (last-row mask, rook counts). | Width 2 and 3; find the zero vector; check against the DP to n = 40. |
| X9 | S3 × O1 | Replayable Delsarte ceilings for ternary codebooks, beside Singleton and Hamming | **Reported.** Textbook method. A replayable number, not a new one. | Ceilings for n = 5 to 8, d = 2 to 3. |
| X10 | S8 × W8 | Rank quantiser variants by spectral-norm damage, not squared error | **Checked:** `experiments/model-output-gemma-e4b-2026-11-15/spectral.py` exists. Not checked whether it is the selection rule. | Recompute the rank correlation on the 26 variants. |
| X11 | S7 × O5 | Choose the protected set by relaxation gain, not by \|w\| | **Reported:** this is Optimal Brain Surgeon and OBQ mathematics, not new. Siliquant has no Hessian-based selection. | On one linear layer, compare \|w\|, diag(H) and greedy relaxation gain at equal count. |
| X12 | S8 × W5 | Dead-weight closure on the real model: zero the unread tensors and measure KL | **Reported** from weft-loom: KL 0.000000 for unread KV in the last 18 layers and unread token rows. The tied token table gives KL 1.364. | Zero those tensors with `harness/ggufedit.py`; compare with the unchanged model. |
| X13 | S3 × O6 | Density-survival ratio: realised bits per weight (scales, codebook, entropy coding) over nominal | **Reported.** Tables give nominal bits. Scale overhead is 0.04 to 0.06 bits per weight in the lattice study. | Report both for one existing quantiser. |
| X14 | S4 × W6 | An output-change bound with R set to the rounding-error length | **Reported:** no transformer bound exists, so this needs a toy first. | A 3-layer toy: certified bound against measured change. Longer. |
| X15 | S3 × O3 | Root-uniqueness gate: certify a unique optimum of MSE over step size, on a box | **Reported:** rounding is piecewise, so the certificate holds only inside cells. Open. | Uniform-step optima at 3 to 16 levels. Weeks. |

## Ranked grafts

Ranked by whether a first test runs in about a day and whether it closes a gap found above.

1. **X2, a preregistration gate.** Gap confirmed in the history. Replay on the model-output study.
2. **X3, a generated number trace.** Gap confirmed on one line (README line 15 against `VE832:16`).
3. **X1, witnesses for closure.** Gap confirmed in `closure()`. It also tests S4 directly.
4. **X5, a bounded-change gate against the baseline.** Gap confirmed. Use weft's form, not weft-eido's.
5. **X4, a closed-enum ledger for the briefs.** Gap confirmed in row 10.

Next, if the first five land: X6 (shared vectors), X8 (zero-vector recurrence), X10 (spectral ranking), X12 (dead-weight closure). Longer: X7, X14, X15.

## Corrections to our own notes

- **gfq-codes.** The recomputable minimum distance is not built. Its `codes/` directory holds a README only (**checked**). The working checkers certify upper bounds with Delsarte LP (**reported**). Earlier framing treated this repository as a source of certified distances. That is not yet true.
- **rigid-realizations.** The "all 880 real" claim rests on one recurrence, `compute_c2`. The complex count 880 is a hard-coded constant at `scripts/discovery/make_clean_l880_cert.py:212` (**checked**). The verifier does not recompute it (**reported**).
- **brief-check claim 8.** The reader says Laman sparsity is the right combinatorial test and is already coded (**reported**). The verdict should be revisited.
- **brief-check claim 10.** See X4.
- **Inherited from weft-loom.** The closure register evaluates its crossed bounds in double precision, not interval arithmetic. Its README says the bound "covers the MLP only" (**reported**). The rule asks for a formal proof. Siliquant's README states the rule in the same words, so it inherits the gap. The weft-loom Stein-Chen fact is labelled proved, but the cited theorem was not opened; the check is numerical, up to 4 by 4 boards and 8,000 random larger ones (**reported**).

## Defects to avoid copying

- **weft-eido radius gate:** per-step bound, so cumulative drift is unbounded (**checked**, see X5). Its test asserts zero core drift, which holds by construction.
- **fibral-rigidity controls:** the six "must-fail" controls are hard-coded `rejected: True` at `verify.py:376-381` (**checked**). The PASS line prints "Controls (6/6)" regardless (**checked**, `verify.py:413`).
- **point-charges `verify_arb.py`:** prints VERIFIED on the line that also reports the index identity, so the identity is not enforced (**checked**, line 243).
- **cmd-narration-guard:** its hash regex matches any run of seven or more digits (**reported**).
- **protean-drafts:** a substring check lets "0.07" pass against "0.075" (**reported**).
- **abios:** the 4.8x size ratio is not what the repository's own numbers give. Compact JSON of the same task is 311 bytes, so the ratio is 3.1x (**reported**).

## Weft, in the transposition

Weft contributes eleven mechanisms (W1 to W9 above and the two caveats below). Siliquant already cites W1 and uses its name, but no code implements it. The rest are not borrowed.

Two caveats from weft itself, both **checked** in source:

- The frozen-core mask is not passed to the gate call at `plasm.py:423`. The core is protected only by zeroing the delta before the gate.
- The conformance vectors cover the gate, not the A/B loop.

## What this note does not claim

- No code was changed or run.
- Reported claims were not re-verified here.
- Triage covers about 230 repositories at name and description level only.

## Disclosure

Written by an AI agent session under the owner's direction. Five read-only passes were delegated to agents. Their claims are marked above. Spot checks were made against source as listed.
