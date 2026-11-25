# 27B Q8_0-reference stratum (`gemma-3-27b-it-q8ref`) — runner `15b0964` line

Priority 1e of `../HANDOFF.md` §5: `gemma-3-27b-it` with a `Q8_0` reference only (never BF16
on a 64 GB host). Plan: `../PREREG-1E-2026-11-24.md`, first committed as `38e3f45` on this
branch (runner pin `15b0964` in `b17d28e`-line merges; ngl amendments in `d046d5e`-line and
`b17d28e` merges — see git log for the exact hashes), merged from `exp/gemma-a16-port`.

Contents:

- `gemma-3-27b-it-q8ref/results.jsonl` + `results.out` — raw rows (18 planned: 9 arms x
  wikitext/python), appended live and committed at progress checkpoints.
- `gemma-3-27b-it-q8ref/pilot-gemma-3-27b-it-q8ref.json` — Step 2/3 pilot (zero-codeword
  gate passed; `model_path` redacted to `<MODELS_DIR>/...`).
- `cache/` is NOT committed (baselines, imatrix, kld; ~50 GB local). Only files listed
  above are tracked.

Every row carries `model_id=gemma-3-27b-it-q8ref` (the stratum label), `model_sha256`,
`imatrix_sha256`, `text_sha256`, `file_sha256`, `backend`, `ngl` (=20 for the whole
stratum, per PREREG-1E amendment 3), `threads`, `stream_arms: true`. These rows are a
separate quantized-reference stratum: never pool them with the 16-bit-reference rows in
`../ccf045d/` or the P7 bridge table.

Run record: `notes/experimenter-m1-15b0964-2026-11-24.md`.
