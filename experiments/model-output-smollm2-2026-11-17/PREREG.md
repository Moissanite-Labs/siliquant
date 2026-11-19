# Plan: the ball arms and the zero codeword on a third model

Written 2026-11-17, before any variant of this model is scored and before its reference pass was run. The run is `run.py` in this folder at the commit that adds this file. It is the program of the record model-output-qwen25-2026-11-17 with this model's file added to the files it accepts.

## Question

The run on `Qwen2.5-1.5B-Instruct` found two things on one model and one text.

- A code with one ball per group of weights is far ahead of llama.cpp's `Q4_0` at equal bits, and at 4.5 bits ahead of its `Q4_K_M` at 4.81 bits. Its gain in KL is about its gain in weight error.
- The E8 Voronoi code of the harness has no codeword at zero. With one scale per row that broke it on the output. With a codeword at zero it is ahead of the scalar rule again.

Do both hold on a model of another family?

## The model and the text

**Model: `SmolLM2-1.7B-Instruct`.** A third publisher and a third family: the Llama layout, 24 layers, 168 block matrices holding 1.61 billion weights, rows of 2,048 or 8,192, one token table used at both ends, no separate output matrix. The file is `SmolLM2-1.7B-Instruct-f16.gguf` from `bartowski/SmolLM2-1.7B-Instruct-GGUF`, a 16-bit float conversion of the publisher's weights, SHA-256 `cf3f0569a9d0b56d5d3b5b7944768c71bd2fef789f8b36b1b85a05d390850852`. The publisher's own GGUF repository holds only a 4-bit file.

**Text: the wikitext-2 test set**, the file of the Qwen2.5 run, SHA-256 `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`. 40 passages of 512 tokens, the second half of each scored. So this is a third model and not a third text.

## How it is scored

As in the Qwen2.5 record: `llama-perplexity --kl-divergence` at llama.cpp commit `eada89c` on 4 CPU threads, against the reference model's saved outputs. Mean KL divergence and the share of positions where the first choice stays the same. Rounded values are stored as 16-bit floats and bits are counted from the codebook, with the scale.

llama.cpp's own files were made before this plan with only the block matrices changed, as for Qwen2.5:

```bash
llama-quantize --tensor-type '^token_embd\.weight$=f16' --tensor-type '^output\.weight$=f16' SmolLM2-1.7B-Instruct-f16.gguf OUT.gguf Q4_0 4
```

`Q4_0` is 4.500 bits per block weight (SHA-256 `ee1c87054df98b36e4b32539fe99fba817ae8e9704080c0165d327f6340f081a`). `Q4_K_M` is 4.822 (SHA-256 `25e0bcfb12f106a5556863f7a94a88aaa0bcdcf1ae76c8034c6c690fabfccdf9`): 144 matrices in `Q4_K` and 24 in `Q6_K`. No importance matrix was used.

## Variants, in the order they are scored

| Variant | What | Bits |
|---|---|---|
| `unchanged` | the reference file against its own saved outputs | 16 |
| `X32:16` | this repository's code for the `Q4_0` rule | 4.5 |
| llama.cpp `Q4_0` | its own file | 4.500 |
| llama.cpp `Q4_K_M` | its own file | 4.822 |
| `EBG128:b560s16` | E8 in one ball per group of 128: 560 bits for the point, 16 for the scale | 4.500 |
| `EB32@32:16` | E8 in one ball per group of 32, 16 levels | 4.498 |
| `ZBG128:b560s16` | integer levels in one ball per group of 128, 560 and 16 bits | 4.500 |
| `VE832:16` | the E8 Voronoi code of the harness, groups of 32 | 4.5 |
| `XR:16` | the scalar rule, one scale per row | 4.006 |
| `VE8R:16` | the E8 Voronoi code of the harness, one scale per row | 4.006 |
| `VE8ZR:16` | the same with a codeword at zero | 4.006 |
| `EBG128:b504s8` | E8 in one ball per group of 128: 504 bits for the point, 8 for the scale | 4.000 |
| `VE8Z32:16` | the E8 Voronoi code with a codeword at zero, groups of 32 | 4.5 |

If the run is cut short, the record says which were not reached.

## What is known before the run

- **The Qwen2.5 run, as far as it had got**: 17 of its variants. Mean KL there: `X32:16` 0.0620, `Q4_0` 0.0623, `Q4_K_M` 0.0333, `VE832:16` 0.0551, `EB32@32:16` 0.0339, `EBG128:b560s16` 0.0195, `XR:16` 0.632, `VE8R:16` 3.658, `VE8ZR:16` 0.209. The gain in KL of the ball arms was 0.91 to 1.02 times their gain in weight error. `ZBG128:b560s16`, `EBG128:b504s8` and `VE8Z32:16` were not scored there yet.
- **The weight error of a ball arm does not depend on the weights.** -25.6 dB for `EBG128:b560s16`, -23.5 for `EB32@32:16`, -24.9 for `ZBG128:b560s16`, -22.8 for `EBG128:b504s8`.
- **Seen of this model.** The names, shapes and types of its tensors. The bits and types of llama.cpp's two files. A build of four of its matrices for eight arms, to test the program, with weight errors printed and nothing scored: `X32:16` -21.1 dB, `VE832:16` -22.6, `VE8Z32:16` -22.6, `XR:16` -16.3, and the ball arms at the figures above.
- **Not seen.** Any score. The reference pass had not been run: a first try was stopped by the machine for lack of memory while the Qwen2.5 run was scoring.

## Predictions

The error of a gap is the root of the sum of the squares of the two standard errors the tool prints, as in the Qwen2.5 record.

| | Prediction | It fails if |
|---|---|---|
| S1 | `unchanged` scores a mean KL of 0.000000, and `X32:16` scores as llama.cpp's `Q4_0`: mean KL within 3 percent, first choice within 0.2 points | any is not so |
| S2 | `EBG128:b560s16` at 4.5 bits has a lower mean KL than `Q4_K_M` at 4.82 bits, by at least 2 errors | it is not |
| S3 | Its KL gain over `X32:16` in dB is 0.75 to 1.25 times its gain in mean weight error | it is outside |
| S4 | `EB32@32:16` has a lower mean KL than `X32:16` and than `VE832:16`, each by at least 2 errors | either is not so |
| S5 | `ZBG128:b560s16`, integer levels with no lattice, has a lower mean KL than `Q4_K_M`, by at least 2 errors | it is not |
| S6 | `VE8ZR:16` has a lower mean KL than `XR:16`, by at least 2 errors | it is not |
| S7 | `VE8ZR:16` is not above `VE8R:16`: it is lower, or within 2 errors | it is higher by 2 errors or more |
| S8 | `EBG128:b504s8` at 4.0 bits has a lower mean KL than `Q4_0` at 4.5 bits, by at least 2 errors | it is not |

Reported without a prediction: whether `VE8R:16` is above `XR:16` here, as it was on Qwen2.5. That depends on whether this model has rows set by one large weight, which was not looked at. `VE8Z32:16` against `VE832:16`. `EBG128:b560s16` against `EB32@32:16`.

## What it would mean

- If S2 and S3 hold, the lead of one ball per group over llama.cpp's files is not a property of one model. Issue #3 puts a packed format and a measure of decoding speed after the check of generality. This would be that check for the ball arms, on three families for the E8 Voronoi code and two for the ball.
- If S2 fails, the Qwen2.5 result stands for one model, and the record says so.
- S6 and S7 say whether a codeword at zero is a safe default for the E8 Voronoi code.

## What this run cannot show

- Anything about a packed format or speed. No arm has one.
- Anything about a second text for this model.
- Whether llama.cpp's files would do better with an importance matrix. They would. None was used here or in the two earlier records.
- The order of near ties. The standard errors treat positions as independent and are understated.
