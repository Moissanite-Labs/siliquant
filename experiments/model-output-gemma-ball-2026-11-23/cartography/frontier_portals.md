# Gemma impact follow-up portals

These questions arose from the rounded 270M table. They are not claims
about accuracy, speed or novelty and do not revise the impact plan.

## Portal 1 — Match fidelity at the observed stored rate

The [impact tables](../reports/impact/gemma-3-270m-it/IMPACT_TABLES.md) place
stock Q4_K_M and Q4_K_M-imatrix at 5.4338 block bpw. Their KL targets can
be used to bracket measured ball-family rates, but intermediate points
are interpolations, not loaded artifacts. Re-enter the raw runner record
to verify byte counts, select a held-out intermediate encoding at the
target rate, and check the same comparison at full-file size on the two
texts and downstream tasks before recommending a public claim.

## Portal 2 — Test the serving opportunity against deployment constraints

The hypothetical bandwidth roofline and cost model in the
[decision brief](../reports/impact/gemma-3-270m-it/README.md) separate weight
storage from resident KV and workspace. This prevents an interpolated
block savings from becoming an empirical accelerator-cost number.
To revisit the opportunity, measure the exact same baseline and candidate
on independent hardened serving kernels, record effective bandwidth and
TTFT/TPOT under matched traffic, and rerun the scenario using those
measurements rather than the peak bandwidth assumption.

## Portal 3 — Test transfer without changing the reference

The [cross-model comparison](../reports/impact/CROSS_MODEL_TRANSFER.md)
finds no full matched 16-bit-reference sweep at 7B scale. A Q8-reference
E4B comparison cannot close that gap. The verified 4B run, with reported
BF16 reference, does not repeat the smaller models' Python reversal.
Re-enter the in-flight bridge and then run a full 7B-class comparison:
keep the texts and reference fixed across arms, bracket calibrated stock
KL, and retain receipts and packed bytes. Bench an actual 4B packed arm
near 4.38 bpw before treating the block-rate gap as a deployment benefit.
