# Plan: the E8 result on a second model and a second text

Written 2026-11-17, before any variant is scored. The run is `run.py` in this folder at the commit that adds this file. This plan covers issue #3 as written and nothing more. The arms of the later records are in a second plan, [PREREG-arms.md](PREREG-arms.md).

## Question

The second record rounded `gemma-4-E4B-it` and scored its output on one text. With one scale per group of weights set from the largest entry, the E8 Voronoi code gave a closer output than scalar rounding at equal bits, at three bit points. One model and one text.

Does that win hold, shrink or disappear on a second model and a second text?

## The second model and the second text

**Model: `Qwen2.5-1.5B-Instruct`.** A different family from gemma and a fifth of its size: 28 layers, 196 block matrices holding 1.31 billion weights, rows of 1,536 or 8,960. The file is the publisher's 16-bit float GGUF, `qwen2.5-1.5b-instruct-fp16.gguf`, SHA-256 `fc89e330deb3fd8fa560f1c0f35a1e2b8da96d59e13445559ed190307a6f5649`. So the reference is the weights as released. In the gemma record the reference was an 8-bit file.

It was chosen because it is not a near-twin: another publisher, another tokenizer, a tied token table stored twice, no gain vectors after the blocks, and attention matrices a sixth the size of the feed-forward ones. And because it is small enough to score many variants on the machine at hand.

**Text: the wikitext-2 test set**, `wiki.test.raw`, SHA-256 `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`, from the archive llama.cpp's own script fetches (`wikitext-2-raw-v1.zip`, SHA-256 `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`). Encyclopedia prose. The first text was documentation and source code.

40 passages of 512 tokens, the second half of each scored, as in the gemma record: 10,200 positions.

## How it is scored

As in the gemma record: `llama-perplexity --kl-divergence` against the reference model's saved outputs. The mean KL divergence, and the share of positions where the top token stays the same.

Rounded values are stored as 16-bit floats, and bits are counted from the codebook: log2 of the levels per weight and 16 bits per scale. For llama.cpp's own quantisers the bits are those of the file. With one scale per row the bits are 4.008 and 5.008 here, where gemma's longer rows gave 4.005 and 5.005.

## What differs from the gemma run

- **The backend.** llama.cpp at commit `eada89c`, built for a CPU with 4 threads. The gemma run used Metal on an M1 Max. The tool is the same.
- **The program.** `run.py` here calls the same harness functions with the same arguments: `tq.maxscale_round`, `tq.maxscale_voronoi`, `ggufedit.write_copy`, `evalkl.score`. It rounds the matrices in 4 processes, since there is no GPU. It leaves out the two path-weighted measures of the gemma program, which are written for gemma's layers.
- **The harness is not changed.** So the gemma baseline is not re-scored. The gemma file is not on this machine.

## Variants

| Variant | What | Bits |
|---|---|---|
| `unchanged` | the reference file against its own saved outputs | 16 |
| `identity` | the block matrices written again as 16-bit floats | 16 |
| `X32:16` | this repository's code for the `Q4_0` rule | 4.5 |
| `VE832:16` | the E8 Voronoi code, one scale per 32 weights | 4.5 |
| `XR:16`, `VE8R:16` | the same two with one scale per row, 16 levels | 4.008 |
| `XR:32`, `VE8R:32` | the same with 32 levels | 5.008 |
| llama.cpp `Q4_0` | its own file | 4.500 |
| llama.cpp `Q4_K_M` | its own file | 4.812 |
| llama.cpp `Q3_K_M` | its own file | 3.824 |

llama.cpp's files were made before this plan, with only the block matrices changed:

```bash
llama-quantize --tensor-type '^token_embd\.weight$=f16' --tensor-type '^output\.weight$=f16' qwen2.5-1.5b-instruct-fp16.gguf OUT.gguf Q4_0 4
```

A check found that no other tensor of the three files differs from the reference.

## What is known before the run

- **The gemma record.** Mean KL of E8 against scalar: 0.075 against 0.117 at 4.5 bits, 0.215 against 0.448 at 4.005, 0.059 against 0.118 at 5.005. That is a KL gain of 1.96, 3.20 and 2.99 dB, each 1.17 to 1.29 times the gain in mean weight error. `X32:16` scored as llama.cpp's `Q4_0`: 0.1173 for both. `Q4_K_M` at 4.80 bits scored 0.069.
- **Weight error on this model.** On 21 of its matrices, cut to their first rows, E8 gains 1.80 dB over the scalar rule with groups of 32 and 3.08 and 3.26 dB per row (the record ball-dimension-no-clip-rule-2026-11-17). On gemma the gains were 1.51, 2.52 and 2.55.
- **Seen before this plan.** The reference pass of the model on the text, which gives its perplexity and nothing about any variant. A reference pass of `Qwen2.5-0.5B-Instruct` on the same text, to time the tool. A build of eight small matrices of that smaller model for nine kinds of arm, to test the program, with its weight errors printed and nothing scored. The bits and tensor types of llama.cpp's three files. No variant of this model has been scored.

## Predictions

| | Prediction | It fails if |
|---|---|---|
| P1 | `identity` scores as `unchanged`: mean KL 0.000000 and the same top token on at least 99.99 percent of positions | either is not so |
| P2 | `X32:16` scores as llama.cpp's `Q4_0` file: mean KL within 3 percent, top token within 0.2 points | either is outside |
| P3 | E8 has a lower mean KL than the scalar rule in all three pairs | it is higher in any pair |
| P4 | In each pair its KL gain in dB is 0.5 to 1.5 times its gain in mean weight error | it is outside in any pair |
| P5 | At q = 16 the finer group has the lower KL for both rules: `X32:16` under `XR:16`, `VE832:16` under `VE8R:16` | either is not so |
| P6 | `Q4_K_M` has a lower KL than `Q4_0`, and `VE832:16` at 4.5 bits has no more than 1.5 times the KL of `Q4_K_M` at 4.81 | either is not so |

## The verdict the issue asks for

One line per bit point. The gap is VE8's mean KL against X's, and its error is the root of the sum of the squares of the two standard errors the tool prints. Those treat positions as independent, so they are too small, as the gemma record says.

- **Holds:** E8 is lower by at least 2 such errors, and its KL gain in dB is at least half of what it was on gemma at that bit point: 0.98 dB with groups of 32, 1.60 per row at 16 levels, 1.50 per row at 32 levels.
- **Shrinks:** lower by at least 2 errors, with less than half the gemma gain.
- **Disappears:** not lower by 2 errors.

The prediction is that it holds at all three.

## The frame of 18 predictions

Five of the 18 predictions of the gemma record have a counterpart here: B1 is P2, C1 is P3, C2 is P4, B3 is P5, and R5 is the first half of P6. The other 13 are about arms this plan does not run: the closures, one scale per matrix, balancing, the symmetries, the clip probes and the measures. They do not apply.

## What this run cannot show

- Anything about a third model or a third text. Two of each is two.
- Anything about a packed format, a decoder or speed. No arm here has one.
- Whether a difference between this run and the gemma run comes from the model, the text, the reference file or the backend. All four changed.
- The order of near ties. The standard errors are understated.
