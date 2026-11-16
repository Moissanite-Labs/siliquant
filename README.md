# siliquant

Research on rounding model weights to a few bits each (quantisation), and on the mathematics around it. It sits beside [weft](https://github.com/Moissanite-Labs/weft), the software of ML Weft, and [weft-loom](https://github.com/Moissanite-Labs/weft-loom), its research records.

The repository follows one rule from weft-loom: a path is closed only on a proof. So it keeps a table of proved limits, and a proposal is read against that table before anything is run.

## What is here

| Folder | What it holds |
|---|---|
| `bounds/` | Proved floors and ceilings for rounding weights, computed from their formulas: Shannon's floor, the best scalar quantiser, what the shape of a lattice cell is worth. Also lattice constants, covering thickness and codebook counts. Start with `bounds/bounds.out`. |
| `perm/` | Exact counting of matchings that avoid forbidden pairs: derangements, rook and hit polynomials, permanents, bounds on permanents, and the closure of a set of forbidden cells. Counts are tested against full enumeration on small cases, and bounds against exact values. `perm/demo.out` holds the figures the notes quote. |
| `lattice/` | Nearest-point decoders for the lattices Z^n, D_n, E_8, A_n and A_n\*, Voronoi codes, scalar reference quantisers, and row and column balancing. Tested against brute force and against known constants. |
| `harness/` | Tools for experiments on real model files: read weights out of a GGUF file and write a changed copy, round on a GPU, and score a changed model against the original with llama.cpp's own tool. `harness/test_tq.py` checks the GPU code against `lattice/`. |
| `experiments/` | One folder per study, each with its plan, program, output and limits. Index in [experiments/README.md](experiments/README.md). |
| `notes/` | [concept-map.md](notes/concept-map.md): the combinatorics the project owner listed, mapped to code, to the closure rule and to rounding. [brief-check.md](notes/brief-check.md): a source check of the two briefs the repository started from. |

## How to check it

The counting code needs only Python 3. The rest needs NumPy. The harness also needs torch.

```bash
python3 -m unittest perm/test_permlib.py
```

```bash
python3 -m unittest lattice/test_lattices.py
```

```bash
python3 bounds/bounds.py
```

```bash
python3 harness/test_tq.py
```

## How records are kept

The conventions are those of [weft-loom's CONTRIBUTING.md](https://github.com/Moissanite-Labs/weft-loom/blob/main/CONTRIBUTING.md). In short:

- A study has a plan committed before its data, or says "not preregistered".
- Every number in a README can be traced to a committed output file.
- A null or negative result is recorded the same way as a positive one.
- No model weights and no datasets are committed. A record names the file it used by its SHA-256.

## Status

Two studies so far.

The first measured weight error only: at 2 bits per weight the lattice E8 beats scalar rounding on every matrix tried, and the lattice the starting briefs proposed, A8\*, gains less than half as much in the mean of each family of data.

The second measured the output of a real model, `gemma-4-E4B-it`. Weight error turned out not to say what the output does. In these runs the output followed how the scale is set more than the size of the error.

With one scale per 32 weights, set from the largest entry, E8 gives a closer output than llama.cpp's `Q4_0` at the same 4.5 bits. With one scale per matrix and no balancing, E8 is worse than scalar rounding at 4 and 5 bits. One model and one text so far.

## Licence

No licence file yet. The owner has not chosen one.

## Disclosure

Most of the code and text here was written by AI agents under the owner's direction, and checked by a second agent before merging.
