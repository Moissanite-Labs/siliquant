# Gemma closure runs on runner ccf045d + 2026-11-24 addendum arms (M1 Max)

All rows in this tree were produced on 2026-11-24 on an M1 Max (MacBookPro18,2, 64 GB, 10 cores)
with llama.cpp `0.5.0 (build 11146, commit 7fe450e19)` (AppleClang arm64), `-ngl 999` (Metal),
`-t 4`, `--workers 4`, `--chunks 40 --ctx 512 --batch 512 --imatrix-chunks 100`.
Every row records `backend`, `ngl` and `threads`.

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
