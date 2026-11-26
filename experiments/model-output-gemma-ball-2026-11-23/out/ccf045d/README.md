# Gemma closure runs on runner ccf045d + 2026-11-24 addendum arms (M1 Max)

All rows in this tree were produced on 2026-11-24 on an M1 Max (MacBookPro18,2, 64 GB, 10 cores)
with llama.cpp `0.5.0 (build 11146, commit 7fe450e19)` (AppleClang arm64), `-ngl 999` (Metal),
`-t 4`, `--workers 4`, `--chunks 40 --ctx 512 --batch 512 --imatrix-chunks 100`.
Of the 79 rows in this tree, the 61 base rows from runner `ccf045d` record `ngl` only
(Metal `-ngl 999`, `-t 4` per the environment above); the 18 addendum rows from runner
`31bbd3c+` record `backend`, `ngl` and `threads` (as do the 6 rows of the 27B pilot in
`../15b0964/`).

Runner commits:
- Base Priority-1 rows (stock-matched reruns, hEBG/hHBG arms, E4B full sweep + bridge):
  `ccf045d04ada2170eeafaa37254a764b96552723` (`exp/gemma-ball-runner` HEAD).
- Addendum arms (`A16BG128:b560s16`, `hA16BG128:b560s16`, `file:Q4_K_M@16`, `file:Q4_K_M-imat@16`):
  `exp/gemma-a16-port` @ `31bbd3c` (off `ccf045d`; adds the vendored `hamming.py` A_n arms and the
  `@16` stock-decode controls per `PREREG-ADDENDUM-2026-11-24.md`, committed before these rows).

Inputs (SHA-256 in each row and matching HANDOFF.md section 2): `gemma-3-4b-it-BF16.gguf`
`7bcfca74…`, `gemma-3-12b-it-BF16.gguf` `276920b1…`, `gemma-4-E4B-it-BF16.gguf` `38e0dba6…`,
`gemma-4-E4B-it-Q8_0.gguf` `f8854aa4…`, `wiki.test.raw` `173c87a5…`, `wiki.train.raw` `6707892f…`,
`base.py` `e38975e1…`, `eval.txt` `df574878…`; per-model imatrix SHA-256 as in HANDOFF.md.
Hash coverage per row (counted over all 85 rows under `out/`): `model_sha256` and
`text_sha256` are in every row; `file_sha256` of the produced GGUF only in the 32 `file:`
rows (26 in this tree, 6 in `../15b0964/`), and `source_file_sha256` only in the 8 `@16`
control rows. The 53 ball-arm rows carry `model_sha256`, `text_sha256` and, on `h*` arms,
`imatrix_sha256`, but no `file_sha256` for the produced GGUF.

Directories:
- `gemma-3-4b-it-stock-matched/`, `gemma-3-12b-it-stock-matched/`: Priority 1a tensor-matched
  stock reruns (`file:Q4_0`, `file:Q4_K_M`, `file:Q4_K_M-imat`, `other_tensors_that_differ == []`)
  plus the addendum's `file:Q4_K_M@16` / `file:Q4_K_M-imat@16` rows.
- `gemma-3-4b-it-hebg/`, `gemma-3-12b-it-hebg/`: Priority 1b/1c (`hEBG128:b496s16`,
  `hEBG128:b560s16`, `hHBG128:b560s16`) plus the addendum's `A16BG128:b560s16` / `hA16BG128:b560s16`.
- `gemma-4-E4B-it/`: Priority 1d full 17-arm BF16 sweep (34 rows, `imatrix_fallback_tensors` =
  the 36 unread KV-sharing tensors on `h*` arms) plus `A16BG128:b560s16`.
- `gemma-4-E4B-it-bridge/`: P7 bridge on the Q8_0 file (`VE832:16`, `VE8Z32:16`, `EBG128:b560s16`),
  kept in its own stratum.
