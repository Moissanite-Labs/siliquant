# The loud columns of `ffn_down`

## Question

Clipping the large entries of one `ffn_down` wrecks SmolLM2 from two of its 24 matrices, and hurts Qwen2.5 from four of its 28. A look at the weights found most of what the clip removes in about ten columns of each. A column of `ffn_down` is one hidden unit of the feed-forward block.

**A. Is the clip's damage in those ten columns?** On gemma the same idea was tested and failed: the ten units with the most change carried 0.05 percent of the damage.

**B. Does it matter to rounding?** Does a ball arm at 4.5 bits gain by leaving those ten columns of each `ffn_down` as they are?

The loud columns of a matrix are the ten in which the clip at 5 standard deviations of a row removes the most squared weight. They are found from the weights alone.

## Status

`complete`, 2026-11-18. 17 variants on two models: 14 of the plan and 3 of its extension. 9 of 12 predictions hold. The tables of two programs were not planned and are marked.

## Preregistration

[PREREG.md](PREREG.md), committed and pushed at `566c318` with `run.py`, `summary.py`, the two `picks` tables and `weights_only.jsonl`, before any variant was scored. Extension 1 was committed at `8f24288`, before any of its three variants was scored and with one row of the plan in, which it states.

## Result in short

1. **On SmolLM2 the clip's damage is the damage to a handful of weights.** In layer 1 the clip in the ten loud columns touches 41 entries and costs a mean KL of 1.449. The clip everywhere else in that matrix touches 346 entries and costs 0.0003. In layer 7 it is 37 entries and 1.232, against 496 entries and 0.0001.
2. **In layer 1 it is two weights.** One hidden unit holds 98.19 percent of the matrix's input activity. Clipping its two large weights, of 1.7 billion in the model, costs a mean KL of 1.175, and the model keeps its first choice 53 percent of the time. In layer 7 the one unit carries less: 0.347 of the ten columns' 1.232.
3. **On Qwen2.5 it is the ten columns in layer 1 and not in the last layer.** In layer 1 the ten cost 0.0477 of the whole clip's 0.0478. In layer 27 the ten cost 17 percent of the whole and everything else 60 percent.
4. **It hardly matters to rounding.** With the ten loud columns of every `ffn_down` left whole, the ball arm on SmolLM2 scores 0.0528 where it scored 0.0560: 5.6 percent, 2.2 errors. I predicted at least 20 percent. With the two matrices left whole altogether it scores 0.0531. So 5 percent is all those two matrices are worth to this arm.
5. **The weights find the columns as well as calibration data does**: 0.0528 with the weights' pick, 0.0526 with the importance matrix's pick. And one column in each of the two layers does what ten do: 0.0537 against 0.0536.
6. **Added to the arm that uses the importance matrix, the kept columns do nothing**: 0.0279 against 0.0282. On Qwen2.5 they do nothing that shows either: 0.0222 against 0.0227, 0.6 errors.

So the units that a clip must not touch are not the units that rounding must protect. The importance matrix halves this arm's KL on SmolLM2, and these columns are a ninth of that. Where the rest is, the plan imatrix-gain-parts-2026-11-18 asks.

## What was run

**The models, the text and the score** are those of the model-output records: `SmolLM2-1.7B-Instruct` and `Qwen2.5-1.5B-Instruct`, 16-bit files, the wikitext-2 test text, 10,200 positions, mean KL against the reference's saved outputs, with `llama-perplexity --kl-divergence` of llama.cpp at `eada89c`.

**Part A.** The clip of one `ffn_down` at 5 standard deviations of a row, split in two: in its ten loud columns only, and everywhere but in them. SmolLM2 layers 1 and 7, Qwen2.5 layers 1 and 27. The whole clip of each is in the clip-by-layer records.

**Part B.** `ZBG128:b560s16`: integer levels in one ball per group of 128 weights, 4.5 bits, no data. A kept column is set to zero before the rest of its matrix is rounded and put back after. It is counted at 16 bits an entry and 16 bits for its number. `+K10` keeps the ten loud columns of every `ffn_down`. `+K10#1,7` keeps them in layers 1 and 7 only. `+S#1,7` leaves those two matrices whole: it is not an arm at equal bits, it is the most that anything done to those two matrices can gain. `+A10` keeps the ten most active columns by llama.cpp's importance matrix in place of the loud ones. `hZBG128:b560s16+K10` adds the kept columns to the arm whose columns are weighted by that matrix.

**Extension 1.** The clip in one column only, the most active one, in layers 1 and 7 of SmolLM2: two entries each. And `+A1#1,7`: that one column kept in each of the two layers.

**The program** is `run.py`. It calls the code of the earlier records for the clip and for the ball arm. On single matrices, a matrix that keeps no column comes out bit for bit as from the earlier records' program.

## Results

Every number is in `results.out`, printed by `summary.py` from `results.jsonl` and the rows of five earlier records, or in the `picks`, `units` and `inputs` tables.

### Part A: where the clip's damage is

| Model | Layer | Where the clip is applied | Entries clipped | Weight error, dB | Mean KL | Over the whole clip | First choice kept |
|---|---|---|---|---|---|---|---|
| SmolLM2 | 1 | everywhere, from the clip record | 387 | -29.75 | 1.7406 | | 44.5% |
| SmolLM2 | 1 | the ten loud columns only | 41 | -31.41 | 1.4492 | 0.833 | 48.9% |
| SmolLM2 | 1 | everywhere but the ten | 346 | -34.71 | 0.0003 | 0.0001 | 99.3% |
| SmolLM2 | 1 | extension: the most active column only | 2 | -36.86 | 1.1752 | 0.675 | 52.9% |
| SmolLM2 | 7 | everywhere, from the clip record | 533 | -28.45 | 1.2519 | | 50.0% |
| SmolLM2 | 7 | the ten loud columns only | 37 | -29.32 | 1.2319 | 0.984 | 50.3% |
| SmolLM2 | 7 | everywhere but the ten | 496 | -35.90 | 0.0001 | 0.0001 | 99.4% |
| SmolLM2 | 7 | extension: the most active column only | 2 | -37.63 | 0.3472 | 0.277 | 72.7% |
| Qwen2.5 | 1 | everywhere, from the clip record | 21,171 | -3.19 | 0.0478 | | 90.8% |
| Qwen2.5 | 1 | the ten loud columns only | 5,097 | -3.21 | 0.0477 | 0.998 | 90.8% |
| Qwen2.5 | 1 | everywhere but the ten | 16,074 | -26.40 | 0.0002 | 0.004 | 99.3% |
| Qwen2.5 | 27 | everywhere, from the clip record | 4,861 | -14.36 | 0.0351 | | 90.7% |
| Qwen2.5 | 27 | the ten loud columns only | 448 | -15.90 | 0.0061 | 0.173 | 97.2% |
| Qwen2.5 | 27 | everywhere but the ten | 4,413 | -19.62 | 0.0212 | 0.603 | 91.3% |

In layer 1 of SmolLM2 the whole clip costs more than its two parts together: 1.741 against 1.449 and 0.0003. The entries outside the ten columns cost nothing alone and something once the ten are clipped. With the first choice kept half the time, these ratios are not shares of anything.

In layer 27 of Qwen2.5 the two parts sum to 0.0272, under the whole's 0.0351, and neither part is small. So "about ten hidden units", as the Qwen2.5 clip record described its four matrices, is right for layer 1 and wrong for the last layer. That record marked it as a description and not a test. This is the test.

In layer 1 of SmolLM2 the one unit's clip is 0.811 of the ten columns' clip. In layer 7 it is 0.282.

### Part B: does keeping them help rounding

| Model | Arm | Reads calibration data | Bits a weight | Columns kept | Mean KL | Under the plain arm | In errors | First choice kept |
|---|---|---|---|---|---|---|---|---|
| SmolLM2 | `ZBG128:b560s16`, earlier record | no | 4.4998 | 0 | 0.0560 | | | 88.3% |
| SmolLM2 | `+K10` | no | 4.5047 | 240 | 0.0528 | 5.6% | 2.2 | 88.5% |
| SmolLM2 | `+K10#1,7` | no | 4.5002 | 20 | 0.0536 | 4.1% | 1.6 | 88.4% |
| SmolLM2 | `+S#1,7`, two matrices whole | no | 4.7394 | | 0.0531 | 5.1% | 2.0 | 88.3% |
| SmolLM2 | `+A10` | yes | 4.5047 | 240 | 0.0526 | 6.0% | 2.3 | 88.0% |
| SmolLM2 | extension: `+A1#1,7` | yes | 4.4998 | 2 | 0.0537 | 4.0% | 1.5 | 88.4% |
| SmolLM2 | `hZBG128:b560s16`, earlier record | yes | 4.5047 | 0 | 0.0282 | | | 91.4% |
| SmolLM2 | `hZBG128:b560s16+K10` | yes | 4.5096 | 240 | 0.0279 | 1.1% | 0.4 | 91.4% |
| SmolLM2 | llama.cpp `Q4_K_M` with the matrix, earlier record | yes | 4.8223 | | 0.0378 | | | 89.6% |
| Qwen2.5 | `ZBG128:b560s16`, earlier record | no | 4.4998 | 0 | 0.0227 | | | 92.8% |
| Qwen2.5 | `+K10` | no | 4.5051 | 280 | 0.0222 | 2.2% | 0.6 | 92.8% |

The five SmolLM2 arms with kept columns lie between 0.0526 and 0.0537, and a standard error is 0.0010. They are one result: about 5 percent, at the edge of what the run can see.

`+K10#1,7` gains 0.742 of what `+K10` gains, and is 0.4 errors from the two matrices left whole.

The arm with no data and kept columns is 12.2 errors behind `Q4_K_M` with the matrix, and 21.3 behind the ball arm with the matrix. So keeping these columns does not stand in for calibration data.

### Not planned: what the columns are

`units.py` and `inputs.py`. Nothing is scored in either. The first was made after the plan and before any score, and the extension was written from it. The second was made while the variants were being scored.

**One unit in each of two layers of SmolLM2** (`units-smollm2.out`). In layer 1, column 4987 holds 98.19 percent of the matrix's input activity, and is 8,052 times as active as the mean column. The clip touches two of its 2,048 entries: its weights into dimensions 749 and 161 of the stream, at 26.7 and 17.5 standard deviations of their rows. In layer 7, column 7758 holds 95.68 percent. The same few dimensions of the stream come up as where the large entries of the loud columns point: 227, 749, 161, 209, 737, 277 and 423.

**Qwen2.5 has nothing like it** (`units-qwen25.out`). Its most active column of layer 1 holds 3.31 percent. Its loud columns there are large columns, up to 61 times the mean column's size, with hundreds of entries beyond the limit each. They hold 58.1 percent of the matrix's squared weight and 2.2 percent of its input activity.

**On SmolLM2 those units are marked by their own input weights** (`inputs-smollm2.out`). A hidden unit reads the stream through one row of `ffn_gate` and one of `ffn_up`. Take the product of the two rows' sizes, over the layer's median. Three layers of 24 have a unit over 10: layer 1 (unit 4987, at 20.5), layer 7 (units 7758 and 2153, at 18.0 and 11.5) and layer 22 (units 3997 and 7833, at 57.7 and 32.1). All five are loud columns. Layers 1 and 7 are the two the clip wrecks, and layer 22 is sixth by cost. On Qwen2.5 one unit of the model is over 10, in layer 27, and it holds none of the activity (`inputs-qwen25.out`).

That is a figure of the weights that marks SmolLM2's two matrices, where the clip's weight error did not. It was found after the result, on one model, with the threshold chosen by eye. It is a lead and not a rule.

### The predictions

| | Prediction | Result | Holds |
|---|---|---|---|
| A1 | SmolLM2, layers 1 and 7: the clip everywhere but in the ten costs under a hundredth of the whole clip | 0.0001 of it in both | yes |
| A2 | SmolLM2, layers 1 and 7: the clip in the ten alone costs at least 0.8 of the whole | 0.833 and 0.984 | yes |
| A3 | Qwen2.5 layer 1: the clip everywhere but in the ten costs under a tenth of the whole | 0.004 | yes |
| A4 | Qwen2.5 layer 27: the clip everywhere but in the ten costs more than a tenth of the whole | 0.603 | yes |
| B1 | SmolLM2: `+K10` is under the plain arm by 2 errors and by at least 20 percent | 2.2 errors, 5.6 percent | **no** |
| B2 | `+K10#1,7` gains at least half of what `+K10` gains | 0.742 | yes |
| B3 | `+K10#1,7` is not above `+S#1,7` by 2 errors | above by 0.4 | yes |
| B4 | `+A10` and `+K10` are within 2 errors | 0.2 | yes |
| B5 | SmolLM2: `hZBG128:b560s16+K10` is under `hZBG128:b560s16` by 2 errors | 0.4 errors | **no** |
| B6 | Qwen2.5: `+K10` is within 2 errors of the plain arm | 0.6 | yes |
| X1 | The clip in the one column costs at least 0.8 of the clip in the ten in layer 1, and at least 0.5 in layer 7 | 0.811 and 0.282 | **no** |
| X2 | `+A1#1,7` is within 2 errors of `+K10#1,7` | 0.1 | yes |

What I got wrong.

- **B1.** The plan's own table said the ten columns hold 98.5 percent of the input of that matrix, and I took the matrix's output error to be the model's. Everything about the matrix was right: ten columns, even one, do what leaving it whole does. What was wrong is that the matrix matters. Whole, the two are worth 5 percent of this arm's KL. The clip cuts a unit's two large weights to a fifth and to under a third of their size, and that wrecks the model. Rounding leaves that matrix with a weight error of -24.9 dB, and taking even that away changes the KL by 5 percent.
- **B5** follows from the same mistake.
- **X1 in layer 7.** I gave it low confidence for the reason it failed: four other loud columns there are 30 to 106 times as active as the mean and point into the same dimensions.

## Deviations from the plan

- **Extension 1 was added after the plan**, before its variants were scored. `run.py` and `summary.py` were extended for it. The planned variants come out bit for bit as before on the single-matrix test.
- **`units.py` and `inputs.py` were not planned.** Their tables are marked above.
- **Another program ran on the machine while `+K10#1,7` was scored**: the forecast of the next plan. That variant took 488 seconds to score where the others took 285 to 414. A score does not depend on how long it takes.
- Nothing else. No row was lost and no variant was scored twice.

## Limits

- **Two models, and gemma differs.** There the ten units with the most change carried 0.05 percent of the damage. Part A's answer is yes on SmolLM2, yes and no on Qwen2.5, no on gemma.
- **Ten columns, one limit, one kind of matrix, one arm, one bit point.** Nothing here is about keeping columns of the matrices that read the stream, or about other arms.
- **A KL above 1 is not a quantity that splits.** The shares of Part A for SmolLM2 say "nearly all" and "nearly none". They do not say 83 percent.
- **The standard errors are the tool's** and treat positions as independent. Part B's gains of 4 to 6 percent are 1.5 to 2.3 errors each. That five arms agree is the evidence that there is a small gain and not none.
- **Activity is from the wikitext-2 training text**, the source of the test text.
- **A kept column is counted, not stored.** `shape/ballpack.py` has no place for one.
- **Why SmolLM2 has such units.** Not asked. Nothing here measures an activation directly: activity is the importance matrix's mean square.

## Read against the closure rule

Nothing is closed by a certificate. These are 17 measurements on two models.

**What changes in the repository's account.**

1. The clip records left open whether the damage is in a few hidden units. It is, on SmolLM2 and in layer 1 of Qwen2.5. It is not in the last layer of Qwen2.5, and it was not on gemma. So the Qwen2.5 clip record's description of its four matrices holds for one of the two it was tested on.
2. The clip records' limit, "this bears on rules that clip", is now measured. Sparing these columns is worth 5 percent to the ball arm on SmolLM2 and nothing that shows on Qwen2.5. The three clip records are not a map of where rounding hurts.
3. The importance matrix's gain on SmolLM2 is not in these columns. They are a ninth of it, and nothing on top of it.
4. The SmolLM2 clip record said the weights do not say which matrix. By the clip's weight error they do not. By the size of a unit's input weights they would have, on that model. That is a description made afterwards.

**What follows.**

1. The plan imatrix-gain-parts-2026-11-18 asks where the importance matrix's gain is, and whether it needs a text with a meaning.
2. A rule that clips could spare the columns of units whose input weights stand far above their layer's. Not tested. The test is the clip on every layer but those columns, on a model the threshold was not chosen on.
3. For the seat that holds gemma: `run.py --picks`, `units.py` and `inputs.py` run on any model given its file and importance matrix, once its file is added to the list `run.py` accepts.

## How to rerun

Needs NumPy, torch, llama.cpp's `gguf` package and `llama-perplexity`, a model file, its reference outputs on the test text and its importance matrix.

```bash
python3 experiments/loud-columns-2026-11-18/run.py --model SmolLM2-1.7B-Instruct-f16.gguf --imatrix smollm2.imatrix.gguf --kld base.kld --text test --work DIR --out results.jsonl --bin llama-perplexity 'clip:5@ffn_down#1/only10' 'ZBG128:b560s16+K10'
```

```bash
python3 experiments/loud-columns-2026-11-18/run.py --model SmolLM2-1.7B-Instruct-f16.gguf --imatrix smollm2.imatrix.gguf --picks > picks-smollm2.out
```

```bash
python3 experiments/loud-columns-2026-11-18/units.py --model SmolLM2-1.7B-Instruct-f16.gguf --imatrix smollm2.imatrix.gguf --layers 0,1,7 > units-smollm2.out
```

```bash
python3 experiments/loud-columns-2026-11-18/inputs.py --model SmolLM2-1.7B-Instruct-f16.gguf --imatrix smollm2.imatrix.gguf > inputs-smollm2.out
```

```bash
python3 experiments/loud-columns-2026-11-18/summary.py > results.out
```

A clip variant takes about five and a half minutes on 4 CPU cores, a ball variant about seven.

## Disclosure

Written by an AI agent session under the owner's direction, on 2026-11-18. The plan, the programs, the run and this record are the session's. The clip and its limit are those of the record output-budget-gemma-e4b-2026-11-17.
