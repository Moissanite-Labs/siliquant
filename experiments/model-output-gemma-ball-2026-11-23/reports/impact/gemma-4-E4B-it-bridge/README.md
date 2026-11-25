# Gemma 4 E4B IT: Q8 bridge impact brief

The bridge compares two E8 arms at **4.5000 block bpw** against the
reported `gemma-4-E4B-it-Q8_0.gguf` reference on `eval.txt` (40 chunks;
reference PPL **7.595148**). This is separate from the 16-bit-reference
sweeps. Both arms were active on **258 of 294** block matrices and
**3,890,216,960 of 3,945,267,200** block weights (**98.60%**). These
counts exclude weights outside the evaluated block matrices.

| Arm | Block bpw | Mean KL ± SE | Top-1 kept | PPL |
|---|---:|---:|---:|---:|
| `VE832:16` | 4.5000 | 0.074682 ± 0.002264 | 91.559% | 7.902921 |
| `VE8Z32:16` | 4.5000 | 0.077252 ± 0.002314 | 91.608% | 7.709733 |

`VE8Z32:16` minus `VE832:16` is **+0.002570 KL**, with a marginal
quadrature SE of **0.003237** (**0.79 SE**). This does not clear the
descriptive 2-SE screen, and the rows do not provide paired covariance.
Top-1 keep goes up **0.049 percentage points** and PPL goes down **0.193188**
even as KL increases. Lower PPL alone does not establish better fidelity.
**There is no stock baseline in this bridge, so it yields no stock parity
or savings estimate.**

## What transfers

- The [2026-11-15 E4B record][historical] reports the same `VE832:16` arm
  and reference model hash, with the same documented `eval.txt` protocol
  (40 chunks; no matching historical text hash verified here). Its row
  gives 4.5000 bpw, KL **0.074728 ± 0.002268**, and **91.569%** top-1.
  The bridge differs by **−0.000046 KL** and **−0.010 percentage points**
  top-1. This is close agreement for a matched arm and Q8 reference, not
  independent replication or a new comparison against historical stock
  rows. That record has no matching `VE8Z32:16` row.
- Gemma 3 4B has the same `VE8Z32:16` arm, but a different model, reported
  BF16 reference, and Wikitext/Python texts. Its `EBG128:b560s16` is neither
  bridge arm. Neither supplies a like-for-like contrast.
- No change to the impact priorities: benchmark the packed 4B candidate
  and finish the E4B BF16 sweep in parallel. The reported E4B BF16 run
  stays on runner `ca10cda` to avoid switching mid-run; affected stock
  `token_embd.weight` rows are planned for rerun on `6a9d871` with the 4B
  follow-ups. The 12B BF16 run switches to `6a9d871`. Track runner version
  before pooling or comparing stock rows.

## Verified receipts

Both blobs were fetched by full commit SHA and hashed. The strict parser
accepted exactly two expected bridge rows; both row keys and all 16
displayed numeric fields agree with the printed summary at display
precision. The embedded model and text digests match the supplied
identifiers; the underlying GGUF and text bytes were not independently
hashed here. The 40-chunk count is reported, not encoded in the rows.

Source: [`86bb1b2a2178bd8b68912c1e7496426f53e9ffa8`][source].

| Artifact or row field | SHA-256 |
|---|---|
| `results.jsonl` | `98f2e3e10de7e732b12476a32a47a6e5c81df16b4b0fddaba082b72b8b857e78` |
| `results.out` | `f86e87d6990c298ca6cd9c59cb16d0e62bcfdb772f439d138153f8712dc5a8fa` |
| `model_sha256` | `f8854aa4480df62585a279e7ca0a881554fc18a41c59c4f62642d16a2ae47012` |
| `text_sha256` | `df5748784801c869b7d1ce54b169af665bfa4c0d3c9839c88d097ffbd33df761` |

[source]: https://github.com/Moissanite-Labs/siliquant/tree/86bb1b2a2178bd8b68912c1e7496426f53e9ffa8/experiments/model-output-gemma-ball-2026-11-23/out/gemma-4-E4B-it-bridge
[historical]: https://github.com/Moissanite-Labs/siliquant/tree/79613683bd4770a10abea09448dafe32e6540d9d/experiments/model-output-gemma-e4b-2026-11-15
