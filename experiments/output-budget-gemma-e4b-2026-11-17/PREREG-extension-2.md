# Extension 2: one layer carries the clip's damage. Where in it, and what follows for the scale rule

Written on 2026-11-17 after Parts H and K of [PREREG-extension-1.md](PREREG-extension-1.md) were scored, and committed before anything below was built, computed or scored. The results it was written with are in `results.out` at commit `0fbdf85`.

## What was seen

Part H clipped the entries of `ffn_down` beyond 5 standard deviations of their row in each of the 42 layers alone.

| | KL |
|---|---|
| Layer 10 alone | 0.155993 |
| The next largest, layer 21 | 0.001990 |
| The 42 layers added up | 0.174156 |
| All 42 at once, first record | 0.163763 |

One layer carries 90 percent of the sum. The weight error of its clip, -27.41 dB, is not the largest: layer 21's is -26.24 dB and its KL is 78 times smaller.

The first record found that pulling back a few large weights does more damage than rounding every weight, and that the output followed how the scale is set. Both may be facts about one matrix. This extension asks where in that matrix the damage sits, and how much of the gap between two scale rules it explains.

Not known when this was written: anything about the entries of layer 10 beyond its one score and its weight error, and any score of a rule applied to all matrices but one.

## Part L: the large entries by layer, weights only

No scoring. For the `ffn_down` matrix of each layer: the number of entries beyond 5 standard deviations of their row, the largest entry in standard deviations of its row, and the squared size of the change the clip makes, over the squared size of the matrix.

For layer 10 also: the same change split by hidden unit, which is the input side of the matrix, and by dimension of the stream, which is its output side. For each split, how many units or dimensions hold half of the squared change, and how many hold nine tenths.

Nothing is predicted. The figures say which units Part M clips.

## Part M: parts of layer 10, and one matrix under another rule

All in layer 10's `ffn_down` unless said. "The top units" are the hidden units with the largest squared change under the clip at 5, as Part L ranks them. "The top dimensions" are the same for dimensions of the stream.

Six scored variants.

1. The clip at 5 on the entries of the top hidden unit only.
2. The clip at 5 on the entries of the top ten hidden units only.
3. The clip at 5 on every entry it touches except those of the top ten hidden units.
4. The clip at 5 on the entries of the top ten dimensions of the stream only.
5. `R+S:32` on all 294 matrices, except that layer 10's `ffn_down` is rounded by `XR:32`.
6. `XR:32` on all 294 matrices, except that layer 10's `ffn_down` is rounded by `R+S:32`.

`R+S:32` is the first record's arm with one searched step per row, which clips. It scored 0.3395. `XR:32` sets each row's step from its largest entry and clips nothing. It scored 0.1180. Both are at 5.005 bits.

| | Prediction | Fails if |
|---|---|---|
| M1 | Ten hidden units carry it: variant 2 scores at least 0.8 of layer 10's 0.1560, which is 0.1248. | it scores less |
| M2 | The rest carries little: variant 3 scores at most 0.2 of it, which is 0.0312. | it scores more |
| M3 | One hidden unit carries half: variant 1 scores at least 0.0780. | it scores less |
| M4 | Most of the gap between the two rules is this one matrix: variant 5 scores at most 0.2288, the midpoint of 0.1180 and 0.3395. | it scores more |
| M5 | And the other way round: variant 6 scores at least 0.2288. | it scores less |

Variant 4 has no prediction. It is reported beside variant 2, to say whether the damage is better described by hidden units or by dimensions of the stream.

M4 and M5 together would say that "the output followed how the scale is set" is mostly "do not clip one matrix". If M4 fails, the gap between the rules is spread over the model, and layer 10 explains the clip probes only.

## Order of work

1. This file, committed and pushed. The programs refuse to compute Part L or score Part M unless it is.
2. Part L. Its output is committed and pushed before any variant of Part M is scored, because it fixes which units Part M clips.
3. Part M. Results committed and pushed.
4. `summary.py` scores M1 to M5 with the thresholds above.

## Limits known before the run

- One model, one text, one runtime, one kind of matrix and one limit of the clip.
- The standard errors treat positions as independent, which understates them.
- Variants 5 and 6 change the rule of one matrix. They do not say what a searched step does to that matrix entry by entry.
- A reading that fits what was seen: published work reports a few activations in a language model that are far larger than the rest and arise in one early layer. Nothing here measures an activation, and this extension does not test that reading.
