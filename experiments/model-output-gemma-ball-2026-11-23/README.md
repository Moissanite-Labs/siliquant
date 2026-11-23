# Gemma impact analysis

This branch covers deployment and publication impact, not prediction scoring.
The dated [scope amendment and impact plan](IMPACT_PLAN.md) supersedes that
part of the historical [pre-data scoring plan](PREREG_ANALYSIS.md).
The impact plan was written **after** the first runner table arrived.

- [Gemma 3 270M: impact, decision brief and provenance](reports/impact/gemma-3-270m-it/README.md)

## Runner contract

Pinned to runner commit `ca10cdacd9b1251b989d5fb66578bfe7e8f2554e`. Each
JSONL object keys on `model_id`, `text` (`wikitext`, `python`, `bridge`)
and `variant`; `bridge` is also a Boolean. Headline models use the
runner-declared 16-bit reference, not the independent E4B Q8_0 bridge.
References and texts are identified by `model_sha256` and `text_sha256`.
Matrix arms carry `imatrix_sha256`; `file:` arms also carry `file_sha256`
and `other_tensors_that_differ`. The parser rejects repeated cells,
conflicting hash labels, partial KL/SE pairs, wrong types, and nonfinite
scores. Rows carry `timestamp_utc`, `ngl`, and timing/weight metadata.

| Measurement | Runner fields |
|---|---|
| Output metrics | `kl_mean`, `kl_mean_se`, optional `same_top` / `same_top_se` (pp), `ppl` / `ppl_se` and other parsed estimates |
| Block / active weights | `matrices`, `weights`, `bits_per_weight` and their `_active` counterparts; not whole-file size |
| Weight diagnostics | `weight_err_db_mean`, `weight_err_db_pooled` and their active counterparts, `weight_err_db_by_kind`, `share_out_of_range`, `kept_columns`; dB fields may have runner `-Infinity` for zero error |

The runner's `results.out` is its rounded display, not another source.
Reference and text hashes do not prove the exact KLD cache was regenerated:
the runner can reuse caches by name. The rows do not record actual scored
positions, eval settings, reference dtype, or per-passage losses. No MMLU
or packed-kernel speed results are implied. A metrics-only row has no output
metrics and is not a zero-quality result. The Markdown message reader is a
separate, explicitly weaker input mode; it does not invent absent hashes or
promote a transcription to verified runner data.

## Replay

From this experiment directory, use `python3 -I -B analyze.py results.jsonl
--model MODEL`. Pass one or more nonoverlapping runner files; use
`--f16-wikitext` and `--f16-python` to supply known baseline PPL. See the
[270M report](reports/impact/gemma-3-270m-it/README.md) for its exact replay.
Add later models as separate `reports/impact/<model>` records. Run
`test_analyze.py` under `-I -B`, `-I -O -B`, and `-I -B -W error`.
Rendering the PNG separately uses Matplotlib.
