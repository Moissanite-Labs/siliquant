# Check of the two briefs this repository started from

Date: 2026-11-15
Status: done for the claims listed. Sources opened on 2026-11-15 unless marked.

Two briefs were written for the project owner on 2026-11-15. They propose to join three existing mathematics tracks (incentre-lattices, rigid-realizations, gfq-codes) and ML Weft into a pipeline for rounding model weights to about 2 bits. The second brief adds a line of classical combinatorics: inclusion-exclusion, rook and hit polynomials, permanents, Sinkhorn scaling.

An AI agent checked each factual claim from sources and reported what it found. Where a claim could be recomputed, it was recomputed in this repository. The last column says which.

## Result

- **The formulas tested are right.** The identities for derangements, rencontres numbers, rook and hit polynomials, Ryser's formula and the ménage problem pass the tests in `perm/`. The briefs' formulas were not gone through one by one.
- **The links to the existing tracks are proposals.** A read of the three tracks' summaries found no work on rounding weights. The briefs call them quantisation tracks.
- **Most of the headline pipeline exists already.** NestQuant rounds weights to E8 with no table. QuIP# uses an E8 codebook. SINQ balances rows and columns before scalar rounding. A search found no use of A8\* for weights and no combination of balancing with a lattice.
- **A8\* is the weaker lattice for this job.** At high rates it is worth 0.40 dB against 0.65 dB for E8, and E8 also has a nearest-point rule with no table.
- **Measured here.** At 2 bits A8\* gains 0.16 to 0.53 dB over scalar rounding in the mean of each family of data, balanced or not, and E8 gains 0.85 to 1.67 dB. The briefs' own prediction held on the real matrices and missed on the synthetic family built for it. See the first experiment.
- **Several statements are wrong or unsupported.** The verdict column says which.

## Claims

| # | Claim in the briefs | Verdict | What was found | Basis |
|---|---|---|---|---|
| 1 | SINQ balances rows and columns by Sinkhorn scaling and halves the perplexity gap | Roughly right | SINQ exists. It equalises the standard deviations of rows and columns with a damped loop, 8 rounds by default in its code and 16 in one routine. The gain of over 50 percent is against plain uniform rounding without calibration. | source |
| 2 | QuIP# and QTIP both use E8 and need costly tables or slow search | Wrong in part | QuIP# uses an E8 codebook with a small table, 256 entries as NestQuant's paper describes it. QTIP uses trellis codes and no lattice. | source |
| 3 | Sorting-based decoding of A8\* avoids the lookup tables E8 needs | Wrong premise | E8 has had a nearest-point rule with no table since 1982: round in two cosets, fix the parity, keep the nearer. NestQuant uses it. `lattice/lattices.py` implements and tests it. | source, and recomputed |
| 4 | Conway and Sloane's decoder for A_n\* runs in n log n | Attribution off | Their 1982 method is slower. The n log n and linear methods are later, by Clarkson and by McKilliam and co-authors. | source |
| 5 | The Voronoi cell of A_n\* is the permutohedron | Confirmed | Standard. | source |
| 6 | Lattice cells beat scalar rounding | True, with a known size | At high rates: 0.40 dB for A8\*, 0.65 dB for E8, at most 0.73 dB for any cells in 8 dimensions and 1.53 dB in any dimension. See `bounds/bounds.out`. At 2 bits see the first experiment. | recomputed |
| 7 | Inclusion-exclusion is needed to hit an exact bit budget | Not needed | The counts are right (3,139 sum-zero vectors with entries in -1..1 among 9). A Voronoi code has exactly q^n codewords with no counting. | recomputed |
| 8 | Generic rigidity of a graph can be verified by one permanent being positive | Wrong as stated | The matching tests for rigidity apply to a family of graphs, one per edge. A positive permanent of a 0/1 matrix is only a perfect matching test. | source |
| 9 | Sinkhorn scaling drives dead-end entries to zero automatically | True only in the limit | Measured here on one 4 by 4 example: the entry falls like 1/(2k) after k rounds (`perm/demo.out`). An exact rule with a certificate exists and is in `perm/` (`dead_cells`). | recomputed, one example |
| 10 | The Stein-Chen method gives a total variation error of order 1/n! for fixed points | Wrong | The true distance to the Poisson law is at most 2^n/(n+1)!, from the exact formula. Stein-Chen gives order 1/n: 1.26/n by the bound in `perm/`. | source, and recomputed |
| 11 | The Bethe approximation of a permanent costs n^2 | Unsupported | One sweep costs n^2. No bound of that order on the number of sweeps was found. | source |
| 12 | Scaling gives the van der Waerden lower bound on a permanent | True at exact scaling | After a finite number of rounds the matrix is not exactly doubly stochastic. The bound held in 2,400 sampled cases here (`perm/demo.out`), which proves nothing. The upper bound holds after any number of rounds. | recomputed |
| 13 | A randomised scheme approximates permanents of non-negative matrices in about n^7 steps | Confirmed | Jerrum, Sinclair and Vigoda, with the later speed-up. It gives an estimate with a probability, not a certificate. | source |
| 14 | Hit polynomials give exact bit-flip error laws for lattice codes | No source, no construction | Nothing published was found, and the briefs do not say what the board would be. Not evaluated. | search |
| 15 | The rigidity matrix can pick out weights to protect | Proposal only | No construction is given. The rigid-realizations track counts realisations of small graphs. Not evaluated. | read of the track |
| 16 | Packing five ternary weights in a byte wastes about 5 percent | Right in states | 243 of 256 states are used. In bits the byte is 99.06 percent used. | recomputed |
| 17 | The 16,240 exact certificate coefficients | Real, unrelated | The figure belongs to a packing motion in incentre-lattices. It says nothing about rounding. The read of the tracks found no machine-checked statement about scaling or lattice radii. | source |
| 18 | An optimal-transport distance can certify that an adapted cell stays within a verified radius | Tested, and closed for one-radius gates on 12 test hosts | On those hosts no radius admits a passing fine-tune and rejects a breaking move, for any distance between the singular-value distance and the Frobenius distance. That range includes the 2-Wasserstein distance between neuron sets with equal weights, suitably scaled. See the weft-loom closure register, CL-5. | recomputed in weft-loom |
| 19 | BosonSampling as a source of quantisation methods | Nothing found | Its output probabilities are squared sizes of permanents of complex matrices. The word there refers to quantised light. No published link to rounding weights was found. | search |

## Sources

- SINQ: Müller and others, arXiv 2509.22944, and its code repository.
- QuIP#: Tseng and others, arXiv 2402.04396. QTIP: Tseng and others, arXiv 2406.11235.
- NestQuant: Savkin, Porat, Ordentlich, Polyanskiy, arXiv 2502.09720.
- Leech lattice for weights: van der Ouderaa and others, arXiv 2603.11021. Pyramid vector quantisation for weights: arXiv 2410.16926.
- Lattice decoders: Conway and Sloane, IEEE Transactions on Information Theory 28 (1982). McKilliam, Clarkson, Smith, Quinn, arXiv 0903.0673.
- Lattice quantiser constants: Agrell and Allen, arXiv 2202.09605.
- Coverings: Schürmann and Vallentin, arXiv math/0403272.
- Fixed points and the Poisson law: DasGupta, Journal of Statistical Planning and Inference 130 (2005).
- Permanents: Bezáková, Štefankovič, Vazirani, Vigoda (2006). Anari and Rezaei, arXiv 1811.02933.
- Sinkhorn scaling: Knight, "The Sinkhorn-Knopp algorithm: convergence and applications" (2008).

Not opened: Barbour, Holst and Janson, "Poisson Approximation" (1992), and Conway and Sloane's book. Statements that would rest on them are recomputed here or marked.

The source check was run by an AI agent. Its report was read against the code and the numbers in this repository before this table was written.
