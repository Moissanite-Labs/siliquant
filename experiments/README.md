# Experiment records

One folder per study. Each record states its question, how it was run, what came out and its limits. A record is evidence for what it measured and nothing more.

Statuses are those of [weft-loom's index](https://github.com/Moissanite-Labs/weft-loom/blob/main/experiments/README.md): `complete`, `partial`, `pilot`, `exploratory`, `design`, `withdrawn`, `superseded`.

| Record | Question | Status | Plan before data | Result in one line |
|---|---|---|---|---|
| [lattice-vs-scalar-rounding-2026-11-15](lattice-vs-scalar-rounding-2026-11-15/README.md) | At equal bits per weight, does rounding blocks of 8 weights to the lattice A8\* or E8 give lower weight error than rounding each weight alone, with and without balancing rows and columns? | `complete` | yes, commit `195f06e` | E8 beats scalar rounding on every matrix. In the mean of each family A8\* gains less than half as much. The brief's claim for balanced A8\* holds on 32 real matrices (0.39 dB) and misses on the synthetic family (0.16 dB). Weight error only. |
| [model-output-gemma-e4b-2026-11-15](model-output-gemma-e4b-2026-11-15/README.md) | Does a smaller weight error give a closer output on a real language model, and what does the output follow if not? | `exploratory` | yes, commit `60ec5aa`, extended four times, each before its variants were scored | No. The output follows how the scale is set: one scale per row or per 32 weights, from the largest entry. With that rule E8 gives a mean KL of 0.075 at 4.5 bits against 0.117 for llama.cpp's `Q4_0`. Pulling back the largest 0.17 percent of weights does more damage than rounding all of them. Three closures hold exactly. One model, one text. |
