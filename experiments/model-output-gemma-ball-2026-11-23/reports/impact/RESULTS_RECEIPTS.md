# Gemma runner output receipts

Pinned to results commit `964b5f4ac988cac176ee4a5825ad1575430a56ab`.
SHA-256 was computed over the Git blob bytes, with no newline normalization.

| Model | Pinned result | SHA-256 | Check |
|---|---|---|---|
| 270M | [results.jsonl](https://github.com/Moissanite-Labs/siliquant/blob/964b5f4ac988cac176ee4a5825ad1575430a56ab/experiments/model-output-gemma-ball-2026-11-23/out/gemma-3-270m-it/results.jsonl) | `dafe50f93262be5628b97a83b915a0f65aa4eaa52f09b1b9bfd36381c4b34a83` | Matches forwarded receipt |
| 270M | [results.out](https://github.com/Moissanite-Labs/siliquant/blob/964b5f4ac988cac176ee4a5825ad1575430a56ab/experiments/model-output-gemma-ball-2026-11-23/out/gemma-3-270m-it/results.out) | `4c31bb3b85082db734887b4899bc3e5781c4ec835a20f89ba15cc919c22520b5` | Matches forwarded receipt |
| 1B | [results.jsonl](https://github.com/Moissanite-Labs/siliquant/blob/964b5f4ac988cac176ee4a5825ad1575430a56ab/experiments/model-output-gemma-ball-2026-11-23/out/gemma-3-1b-it/results.jsonl) | `040717b2d9c24d47e64fcc3b41156a375d651f180cd7b9f558ec603c60f7733f` | Computed from pinned blob; no separately supplied digest |
| 1B | [results.out](https://github.com/Moissanite-Labs/siliquant/blob/964b5f4ac988cac176ee4a5825ad1575430a56ab/experiments/model-output-gemma-ball-2026-11-23/out/gemma-3-1b-it/results.out) | `aeea9b43446c4f62bb924aaf706f555d47ba498855cc1372546c64de1468429b` | Computed from pinned blob; no separately supplied digest |

Both JSONLs passed the strict parser: 34 unique headline rows each, 17
arms on each of two texts, no reported errors. The 270M transcription's
34 cell keys, mean KL, KL SE and top-1 are exact matches; 30 block-bpw
values and all 34 PPL values differ only within their rounding bounds.
The raw F16 PPL is 50.594102 / 8.38385 (270M) and 27.164816 / 6.467454
(1B), wikitext / Python; the latter round to 27.165 / 6.467 as handed off.
Each `results.out` also agrees with its JSONL at display precision on
cell keys, both rate columns, KL, KL SE, and top-1.

| Recorded input identifier | SHA-256 in JSONL |
|---|---|
| 270M model (matches prior forwarded F16 identifier) | `140cb395a3a3dcdb3de66e44c60d7265286c9d49f951c1899ec0d8c1f16b2feb` |
| 1B model | `05bd381a5f45611ce53f4fdcc6641cf6cec68c3091d74e8a32ea591f062d3fc5` |
| Wikitext, both models (matches prior forwarded identifier) | `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08` |
| Python, both models | `e38975e1c68d98ac1664dfd530616eb35c72294382a4dd873d4746b23f27779f` |

These are consistent row identifiers, not independently hashed GGUF or
text bytes. The runner operator reported a byte-for-byte match with the original outputs; the check here
is against the committed bytes and available receipts, not against that
machine. The blob pins do not verify calibration or cache files.
