# How far does the Gemma 270M result transfer?

**Decision: fund the matched Gemma replication and packed-kernel benchmark, not a fleet forecast.** In every available near-rate-matched plain-ball comparison against Q4_0, the ball lowers mean KL: ratios span **0.365–0.440** across Gemma 270M, Qwen2.5 1.5B and SmolLM2 1.7B. This is directionally consistent, not evidence of shrinkage with size: Qwen's ratio is lower than Gemma's on wikitext, SmolLM2's is higher, and size and family vary together. The larger models also support a real-matrix ball advantage over calibrated Q4_K_M at the rates tested, but not the specific Gemma 270M 5.43-bpw baseline or its matched-quality storage savings. This retrospective is **N0, sampled/fragile**, not an independent replication or a new method claim.

## Headline contrasts

Ratios are candidate mean KL / stock mean KL (lower is better). The calibrated ratio includes **candidate / baseline block bpw**; comparisons are not iso-bitrate. Calibration gap is real-text minus random-vocabulary ball KL, always at b560. Dashes mean no comparable measurement located, not zero. Wiki = wikitext-2 test; Py = Python source.

| Model (nominal size) | Text | Plain b560 / Q4_0 | Real matrix / Q4_K_M-imat (bpw) | Real − random ball KL | Sources |
|---|---|---:|---:|---:|---|
| Gemma 3 270M IT | Wiki | 0.388 | 0.399 (b624; 5.0180 / 5.4338) | −0.010941 | [G] |
| Gemma 3 270M IT | Py | 0.372 | 0.631 (b624; 5.0180 / 5.4338) | −0.003719 | [G] |
| Llama 3.2 1B Instruct | Wiki | — | 0.674 (b560; 4.5052 / 4.8023) | −0.004148 | [L] |
| Llama 3.2 1B Instruct | Py | — | 0.635 (b560; 4.5052 / 4.8023) | −0.002307 | [L] |
| Qwen2.5 1.5B Instruct | Wiki | 0.365 | 0.741 (b560; 4.5060 / 4.8120) | −0.001935 | [Q], [R], [M] |
| Qwen2.5 1.5B Instruct | Py | — | 0.790 (b560; 4.5060 / 4.8120) | −0.000798 | [P] |
| SmolLM2 1.7B Instruct | Wiki | 0.440 | 0.746 (b560; 4.5047 / 4.8223) | −0.003155 | [S], [R], [M] |
| SmolLM2 1.7B Instruct | Py | — | 0.711 (b560; 4.5047 / 4.8223) | −0.000979 | [P] |
| Gemma 4 E4B IT (7.5B total) | llama.cpp docs/code | — | — | — | [E], [B] |

![Plain ball to Q4_0 mean KL ratio by nominal model size. Points are shown for Gemma 270M (Wiki and Python), Qwen2.5 1.5B and SmolLM2 1.7B (Wiki only).](KL_RATIO_BY_SIZE.png)

The companion b560 comparison on **Gemma 270M** gives 0.830 on Wiki and **1.141 on Python** against Q4_K_M-imat (4.5178 / 5.4338 bpw; [G]). The b624 win on Python does not carry down to b560. **Llama 1B** has no Q4_0 in the cited run; its plain ball loses to uncalibrated Q4_K_M on both texts, so it is not evidence for a plain-ball win against that stronger stock baseline. Real calibration has lower observed KL than random in all eight cells, but the gaps vary and some are small compared with marginal SEs (e.g. [P]); there is no universal case to pay for real-text calibration. On Qwen Python, both real and random b560 lose to *plain* b560 in mean KL.

## Auditable measurements

Cells are **reported block bpw / mean KL / top-1 kept %**, not whole-file rates. All eight rows here compare against each run's own F16 reference. Source tags are in the same order as the measurements where multiple sources are needed: [Q]/[S] for Q4_0; [M] for real-matrix Q4_K_M (KL/top-1); [R] for four-decimal Q4_K_M rates and ball arms (including plain); [G], [L] or [P] for all cells in their respective rows. The older real ball weights use the historical calibration recipe, not necessarily Gemma's weighting strength or overhead.

| Model / text | Q4_0 | Q4_K_M | Q4_K_M-imat | Plain b560 | Real b560 | Random b560 | Real b624 | Sources |
|---|---|---|---|---|---|---|---|---|
| Gemma 270M Wiki | 4.5000 / 0.501528 / 70.137 | 5.4338 / 0.153572 / 83.049 | 5.4338 / 0.091622 / 86.755 | 4.4998 / 0.194755 / 80.088 | 4.5178 / 0.076017 / 87.922 | 4.5178 / 0.086958 / 87.480 | 5.0180 / 0.036582 / 91.588 | [G] |
| Gemma 270M Py | 4.5000 / 0.565080 / 77.647 | 5.4338 / 0.166596 / 87.899 | 5.4338 / 0.088345 / 91.765 | 4.4998 / 0.209988 / 85.658 | 4.5178 / 0.100806 / 90.868 | 4.5178 / 0.104525 / 90.532 | 5.0180 / 0.055786 / 92.885 | [G] |
| Llama 1B Wiki | — | 4.8023 / 0.043179 / 89.03 | 4.8023 / 0.035923 / 90.02 | 4.4998 / 0.047808 / 88.86 | 4.5052 / 0.024224 / 91.50 | 4.5052 / 0.028372 / 90.49 | — | [L] |
| Llama 1B Py | — | 4.8023 / 0.037139 / 92.54 | 4.8023 / 0.034137 / 92.19 | 4.4998 / 0.045635 / 91.37 | 4.5052 / 0.021682 / 93.96 | 4.5052 / 0.023989 / 93.88 | — | [L] |
| Qwen 1.5B Wiki | 4.500 / 0.062335 / 88.22 | 4.8120 / 0.033348 / 91.11 | 4.8120 / 0.028205 / 91.84 | 4.4998 / 0.022750 / 92.75 | 4.5060 / 0.020898 / 92.64 | 4.5060 / 0.022833 / 92.81 | — | [Q], [M], [R] |
| Qwen 1.5B Py | — | 4.8120 / 0.022319 / 94.52 | 4.8120 / 0.019446 / 94.84 | 4.4998 / 0.013971 / 95.37 | 4.5060 / 0.015366 / 95.48 | 4.5060 / 0.016164 / 95.40 | — | [P] |
| SmolLM2 1.7B Wiki | 4.500 / 0.127149 / 82.35 | 4.8223 / 0.083761 / 85.52 | 4.8223 / 0.037827 / 89.56 | 4.4998 / 0.055962 / 88.25 | 4.5047 / 0.028205 / 91.35 | 4.5047 / 0.031360 / 90.73 | — | [S], [M], [R] |
| SmolLM2 1.7B Py | — | 4.8223 / 0.052163 / 91.18 | 4.8223 / 0.023435 / 93.95 | 4.4998 / 0.036447 / 92.48 | 4.5047 / 0.016664 / 94.89 | 4.5047 / 0.017643 / 94.82 | — | [P] |

For Wiki, the earlier Qwen and SmolLM2 reports also contain Q4_0-imatrix (4.516 / 0.051859 / 88.54 and 4.516 / 0.086033 / 85.80, respectively; [M]). Random-matrix Q4_K_M is a separate stock control in [R], [P] and [L]; the headline ratio uses real-matrix stock only.

| Run / text | EBG128:b560s16 | HBG128:b560s16 | Unrotated VE8Z32:16 | Sources |
|---|---|---|---|---|
| Gemma 270M Wiki | 4.4999 / 0.178904 / 81.608 | 4.4999 / 0.174683 / 81.588 | 4.5000 / 0.394135 / 73.971 | [G] |
| Gemma 270M Py | 4.4999 / 0.186328 / 88.347 | 4.4999 / 0.167353 / 88.011 | 4.5000 / 0.377994 / 82.297 | [G] |
| Qwen 1.5B Wiki | 4.500 / 0.019549 / 93.08 | — | 4.500 / 0.043414 / 89.55 | [Q] |
| SmolLM2 1.7B Wiki | 4.500 / 0.060861 / 87.50 | — | 4.500 / 0.124728 / 82.75 | [S] |

No matching EBG/HBG/VE8Z rows were located for Llama or the older Python extension. **E4B is context only:** the November 15 Gemma 4 E4B record scores on mixed llama.cpp documentation/code against **Q8_0** saved outputs (its F16-stored copy matches that same reference, not an independent F16 model). It modifies block matrices and reports Q4_0 **4.500 / 0.1173 / 89.40**, Q4_K_M **4.803 / 0.0689 / 91.79**, its distinct E8 arm VE832:16 **4.500 / 0.0747 / 91.57**, and D+E8:32 **5.009 / 0.2748 / 85.99** ([E]). The November 17 joined arm **4.484 / 0.0629 / 92.02** also uses that Q8 reference ([B]); it is not a ZBG arm. Neither supplies a matching plain ball, matrix ball or imatrix stock contrast. [W] studies rounding error, not model-output KL/top-1. None fills the 7B-class F16 validation gap.

Different model families, texts or text windows, runner builds, rate overheads, and calibration recipes limit comparisons. These are single reported runs per cell, not multiple calibration seeds or an uncertainty estimate across models. Historical observations support the **direction** of the opportunity where measured, not its production magnitude: retain 25/50/100% as illustrative positive-transfer sensitivities and explicitly allow **0% transfer (zero modeled savings)**; runtime regressions can still make deployment net negative.

## Next experiments, in impact order

1. **Finish and verify Gemma 1B on both texts.** Get raw receipts, common F16 references, and actual block rates for plain b560/Q4_0, real b496/b560/b624/Q4_K_M-imat and real/random b560. This isolates the first within-family scale check and tests the 270M Python rate reversal.
2. **Replicate a compact matched F16 sweep on a 7B-class target.** Run Q4_0, Q4_K_M-imat and enough nearby ball rates to bracket KL parity; prioritize both texts and report packed bytes. The Q8 E4B record cannot answer this. This is the largest remaining evidence gap for a hyperscaler claim.
3. **Benchmark packed decode end to end against stock.** Reuse the first verified quality match. Record full artifact bytes, resident memory, throughput and latency at batch 1 and a realistic batch; benchmark dequantization so a presumed bandwidth gain cannot mask a kernel loss. Start engineering this in parallel if hardware is free.
4. **Run paired downstream evals on baseline and matched candidate.** Prioritize held-out language and code tasks, plus a narrow regression panel; KL, top-1 and PPL are not deployment acceptance criteria.
5. **Repeat real/random calibration across seeds and a held-out domain.** Keep the same arm/rate and quantify paired variation. Promote random calibration only if it preserves quality under this check.
6. **Close targeted historical control gaps.** Add matched Q4_0 to Llama/Python and only rerun Hadamard/E8 controls where they would change the deployment choice. Do not infer missing ratios.
7. **Expand families, hardware, or method variants only after the above.** Then decide whether the result warrants a reproducibility note, a funded serving pilot, or any external announcement.

## Pinned source records

[G]: https://github.com/Moissanite-Labs/siliquant/blob/fcaee366f2c9b9b485bf6df0a6a82e455a678b24/experiments/model-output-gemma-ball-2026-11-23/reports/impact/gemma-3-270m-it/RUNNER_TRANSCRIPTION.md
[L]: https://github.com/Moissanite-Labs/siliquant/blob/15383d5b0c44da1ed1271bd46f54b2b8934d6ba0/experiments/third-model-llama32-2026-11-18/results.out
[Q]: https://github.com/Moissanite-Labs/siliquant/blob/b522426596e9a874f22a84f2427e80238caacc6c/experiments/model-output-qwen25-2026-11-17/results.out
[S]: https://github.com/Moissanite-Labs/siliquant/blob/63dfd5254271d712dabe0c88346ea01fc14a9b31/experiments/model-output-smollm2-2026-11-17/results.out
[M]: https://github.com/Moissanite-Labs/siliquant/blob/10e2677aa7b368cb73fbb79f9af954227c894478/experiments/imatrix-baseline-2026-11-17/results.out
[R]: https://github.com/Moissanite-Labs/siliquant/blob/330979fbf400f42fd39be083d8d5cc04cea74ead/experiments/imatrix-gain-parts-2026-11-18/results.out
[P]: https://github.com/Moissanite-Labs/siliquant/blob/36c09dc6d183b7d77173138df62585d52e9b8579/experiments/second-text-python-2026-11-18/results.out
[E]: https://github.com/Moissanite-Labs/siliquant/blob/79613683bd4770a10abea09448dafe32e6540d9d/experiments/model-output-gemma-e4b-2026-11-15/results.out
[B]: https://github.com/Moissanite-Labs/siliquant/blob/abcf1f6f705b38c278ae88e28908c561ae85c0c6/experiments/output-budget-gemma-e4b-2026-11-17/README.md
[W]: https://github.com/Moissanite-Labs/siliquant/blob/1c3d4c9efc749ea3d947dcc7a385224b862b982d/experiments/lattice-vs-scalar-rounding-2026-11-15/results.out
