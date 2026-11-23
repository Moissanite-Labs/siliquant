# Gemma impact analysis — scope amendment and plan

2026-11-23. **Written after receiving and viewing the runner operator's 270M table;
before computing the impact estimates below.** The original
[pre-data plan](PREREG_ANALYSIS.md) remains an immutable record of the
original assignment. P1–P11 scoring and technical interpretation are outside this scope;
this branch's deliverable is impact analysis and input
validation. Findings from this model are descriptive, not preregistered.

- Score no predictions. Work by model and text. Use the runner's headline
  JSONL with strict typed/provenance validation when available. For the 270M
  handoff, use the separately labelled transcription and retain the claimed,
  unverified receipts; never reconstruct nonexistent provenance. Do not
  pool the Q8 bridge, substitute for E4B, or give a technical explanation.
- Compare each observed 4.5-bpw family arm against `file:Q4_0` (nominal
  comparison; print actual block rate), and all observed 5-bit continuity
  arms against each of `file:Q4_K_M` and `file:Q4_K_M-imat`; explicitly
  label the latter rate mismatches. Report candidate/baseline mean KL ratio,
  baseline-minus-candidate KL, top-1 retained candidate-minus-baseline in
  percentage points, and each PPL's percentage change from the supplied
  F16 PPL (when known).
- For matched-quality estimates, use only the measured per-group plain
  and real-matrix b496/b560/b624 curves. Interpolate linearly in KL versus
  actual block bpw on the adjacent bracket; do not extrapolate. If all
  observed rates already beat a baseline, give a lower bound on savings
  from the lowest observed rate; if no arm reaches the target, report no
  estimate. Report baseline-minus-matched bpw and percentage of baseline
  quantified block weight storage saved. Whole-model percentage is
  **unavailable** from block rates alone; needs unchanged non-block bytes,
  per-format actual full-file sizes and packing/accounting.
- Calibration contrasts: real matrix versus random-vocabulary matrix and
  both versus zero-calibration HBG and VE8Z at nominal 4.5 bits, separately
  by text. Report only rate/metric deltas; comparisons across different
  arm families are descriptive, not a controlled calibration-only effect.
- For KL uncertainty, report the gain divided by the quadrature of the two
  marginal KL SEs (describe absolute distance of at least 2 SE as distinct
  under this heuristic; not paired inference or multiplicity control).
  No supplied SEs for top-1 or PPL. Rank stock comparisons by absolute
  relative KL effect within each text; identify whether each meets the
  descriptive uncertainty screen and report effect sizes rather than
  inventing an application-level utility threshold. Compute arithmetic
  on serialized decimal values with rational numbers; use floats only for
  display, approximate SE multiples, and plotting.
- Plot both texts side by side: KL versus actual block bpw for the two
  measured per-group families and stock points, no extrapolated curves.
  State evidence limitations: rounded transcription, one model, one
  calibration seed, two texts, missing raw verification and full-file
  accounting. Keep the model-specific reports under `reports/impact/`.

## Decision and scenario model

Added after receiving the table, before impact calculations. Triage each
candidate against full-text prior art and the repository's prior studies;
distinguish a new measured Gemma data point from a new method or patent
claim. Recommend publication, announcement, IP review, or a replication
budget separately; disclose the limits of one model and one calibration seed.

Use the smaller of the estimated real-matrix savings at Q4_K_M-imat quality
across the two texts as a scenario anchor. Hypothetical low/mid/high inputs
respectively: 7/27/70 billion weights, 70/85/95% covered weights, 25/50/100%
transfer of that block-rate improvement, 100/1,000/10,000 baseline accelerator
equivalents, 25/50/75% eligible serving load, and USD 2/3/5 per accelerator-hour.
These are assumptions, not estimates for any particular serving operator. Fix uncovered
weights at 16 bpw, add 20 GB for resident KV/workspace, and assume that
non-weight traffic per decode step is 20% of baseline weight bytes.
Use published H200 peak bandwidth and memory capacity as an illustrative
hardware specification, not a statement about any operator's hardware. Compute
resident size and bandwidth roofline at batch one; assume fixed demand and
ideal fractional right-sizing of eligible serving capacity for annual gross
cost opportunity (8,760 hours). Ignore dequantization, capex constraints,
prefill, communication, and any mix/traffic effects not explicitly included.
These scenario outputs are neither realized savings nor guarantees.
