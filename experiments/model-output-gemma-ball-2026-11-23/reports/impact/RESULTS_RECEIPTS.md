# Gemma runner output receipts

270M and 1B are pinned to results commit `90a3247dc9bb70ca3afa0f13627ec4b569c73317`;
4B is pinned to `dc10d4bbca8cc049f288aa22e42172aebb493363`.
SHA-256 was computed over the Git blob bytes, with no newline normalization.

From the experiment directory in a clone with the repository as `origin`,
fetch the pinned commit and extract the JSONL blobs for replay:

```sh
pin=90a3247dc9bb70ca3afa0f13627ec4b569c73317
git fetch origin "$pin" &&
git show "$pin:experiments/model-output-gemma-ball-2026-11-23/out/gemma-3-270m-it/results.jsonl" > 270m-results.jsonl &&
git show "$pin:experiments/model-output-gemma-ball-2026-11-23/out/gemma-3-1b-it/results.jsonl" > 1b-results.jsonl
```

For 4B:

```sh
pin=dc10d4bbca8cc049f288aa22e42172aebb493363
git fetch origin "$pin" &&
git show "$pin:experiments/model-output-gemma-ball-2026-11-23/out/gemma-3-4b-it/results.jsonl" > 4b-results.jsonl
```

| Model | Pinned result | SHA-256 | Check |
|---|---|---|---|
| 270M | [results.jsonl](https://github.com/Moissanite-Labs/siliquant/blob/90a3247dc9bb70ca3afa0f13627ec4b569c73317/experiments/model-output-gemma-ball-2026-11-23/out/gemma-3-270m-it/results.jsonl) | `dafe50f93262be5628b97a83b915a0f65aa4eaa52f09b1b9bfd36381c4b34a83` | Matches forwarded receipt |
| 270M | [results.out](https://github.com/Moissanite-Labs/siliquant/blob/90a3247dc9bb70ca3afa0f13627ec4b569c73317/experiments/model-output-gemma-ball-2026-11-23/out/gemma-3-270m-it/results.out) | `4c31bb3b85082db734887b4899bc3e5781c4ec835a20f89ba15cc919c22520b5` | Matches forwarded receipt |
| 1B | [results.jsonl](https://github.com/Moissanite-Labs/siliquant/blob/90a3247dc9bb70ca3afa0f13627ec4b569c73317/experiments/model-output-gemma-ball-2026-11-23/out/gemma-3-1b-it/results.jsonl) | `040717b2d9c24d47e64fcc3b41156a375d651f180cd7b9f558ec603c60f7733f` | Computed from pinned blob; no separately supplied digest |
| 1B | [results.out](https://github.com/Moissanite-Labs/siliquant/blob/90a3247dc9bb70ca3afa0f13627ec4b569c73317/experiments/model-output-gemma-ball-2026-11-23/out/gemma-3-1b-it/results.out) | `aeea9b43446c4f62bb924aaf706f555d47ba498855cc1372546c64de1468429b` | Computed from pinned blob; no separately supplied digest |
| 4B | [results.jsonl](https://github.com/Moissanite-Labs/siliquant/blob/dc10d4bbca8cc049f288aa22e42172aebb493363/experiments/model-output-gemma-ball-2026-11-23/out/gemma-3-4b-it/results.jsonl) | `5895d850aff01a154a3b57e1c47468c90121e34316fa6e850ae866f6ec4dc23e` | Matches forwarded receipt |
| 4B | [results.out](https://github.com/Moissanite-Labs/siliquant/blob/dc10d4bbca8cc049f288aa22e42172aebb493363/experiments/model-output-gemma-ball-2026-11-23/out/gemma-3-4b-it/results.out) | `6ef67975a707ce8a5d93214885fa5b04ccba5bf929a98e32c5f0e0982ff48594` | Matches forwarded receipt |

All three JSONLs passed the strict parser: 34 unique headline rows each, 17
arms on each of two texts, no reported errors. The 270M transcription's
34 cell keys, mean KL, KL SE and top-1 are exact matches; 30 block-bpw
values and all 34 PPL values differ only within their rounding bounds.
The raw F16 PPL is 50.594102 / 8.38385 (270M) and 27.164816 / 6.467454
(1B), wikitext / Python; the latter round to 27.165 / 6.467 as handed off.
Each `results.out` also agrees with its JSONL at display precision on
cell keys, both rate columns, KL, KL SE, and top-1.

For 4B all 272 printed numeric cells (both rates, mean weight error, KL,
KL SE, top-1 and both timings) agree at displayed precision. The 4B
`ppl_base` is 15.325197 / 4.991511. The run was reported as BF16, unlike
270M/1B F16; the JSONL does not encode the reference precision or stock
fallback counts. Both 4B stock K entries report 4.8151 block bpw (all and
active) on both texts; that does not verify per-tensor format assignments.

| Recorded input identifier | SHA-256 in JSONL |
|---|---|
| 270M model (matches prior forwarded F16 identifier) | `140cb395a3a3dcdb3de66e44c60d7265286c9d49f951c1899ec0d8c1f16b2feb` |
| 1B model | `05bd381a5f45611ce53f4fdcc6641cf6cec68c3091d74e8a32ea591f062d3fc5` |
| 4B model (matches forwarded BF16 identifier) | `7bcfca74279608552011d3c4d11e9298934502ffa410eb21b9047a6276b87ba6` |
| Wikitext, all three models (matches forwarded identifier) | `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08` |
| Python, all three models (matches forwarded identifier) | `e38975e1c68d98ac1664dfd530616eb35c72294382a4dd873d4746b23f27779f` |

These are consistent row identifiers, not independently hashed GGUF or
text bytes. The runner operator reported a byte-for-byte match with the original outputs; the check here
is against the committed bytes and available receipts, not against that
machine. The blob pins do not verify calibration or cache files.
