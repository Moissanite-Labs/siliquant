# Check of the persistence document and of two replies on flow and drag

Date: 2026-11-17
Status: done for the claims listed. One study is planned from it and has its own record: `experiments/output-budget-gemma-e4b-2026-11-17`.

Three texts reached the project owner on 2026-11-17, beside a proposal of the owner's own: that data should shape models as flows shape the earth. The proposal, and what of it is mathematics, is in weft-loom, [notes/flow-shapes-form.md](https://github.com/Moissanite-Labs/weft-loom/blob/main/notes/flow-shapes-form.md). This note checks the three texts claim by claim.

- **The persistence document.** A "technical specification" of early stopping by neural persistence, 27,165 bytes, SHA-256 `b6d6dcc1d1326e2a9a885875bd4a9807e9a965b310b24a579bb34088b420c53e`. It restates one paper: Rieck and others, ICLR 2019. It is not in this repository.
- **The first reply.** An assistant's answer to the proposal, in four parts.
- **The second reply.** An assistant's answer to the owner's question about drag, in six mappings from fluid mechanics to models.

The note reads them the way the repository's rule asks: against the table of proved limits and the records, before anything is run.

**Checked** means the number was computed by the session that wrote this note, or the file was read by it. **Reported** means it comes from a read-only agent's pass over a source and was not re-opened here. **Search** means that agent saw a search result's text and did not open the source. **Derived** means it follows from the definitions by a short argument given in the row.

## Result

- **The persistence document restates the paper correctly in the main.** Three of its table cells break off mid-sentence. One of them is the cell headed as being about the count in the paper's bound, and the document's own algorithm writes that count as a product, which is not the paper's reading. On a random 650 by 650 layer the two readings give 0.37 and 0.02.
- **The score reads each unit's largest weight.** The largest link at every unit is in the tree when it is unique. On random matrices 72 to 85 percent of the tree is a row's or a column's largest entry, and those maxima alone give 97 to 98 percent of the score. The rule that did best in the gemma record, an exploratory one, sets a row's step from the same entry.
- **It cannot be a measure of behaviour.** Flipping signs, or reordering one layer's rows alone, leaves it equal to six digits and changes the outputs by more than their own size. Rescaling ReLU units leaves the outputs equal to 1e-15 and moves it from 0.45 to 0.89.
- **The paper's bound needs two things its main text leaves open.** It holds with each layer's own largest weight and fails with the network's, which the paper divides by: 9.4 against a cap of 1 on one layer of the check. Its own example meets it exactly under one way of counting the piece that never joins, falls short under a second and passes it under a third (row 4).
- **A published critique exists and the document does not have it.** Girrbach and others (2023) find the variance of the weights and the concentration of large weights to be the main factors of the score, and in later layers of deep networks the variance alone.
- **Its closing claim is larger than its evidence.** "Production readiness" rests on small fully connected networks on four data sets. On CIFAR-10 the paper's own table has the method 1.13 points of accuracy under stopping by validation loss.
- **Of the ten claims in the two replies, three are wrong as stated or contradicted by a record** (rows 18, 21 and 23). One is closed at its end point, with its schedule not run. One was measured on Gaussian data only, where it gains nothing, and not on the heavy tails it speaks of. Two hold in a narrower form, one is right for another problem, one has no source and one cannot be checked.
- **What is left is a budget.** Read as a budget with parts, the gap to Shannon's floor has two parts no record has scored on a model's output: where the bits go among the kinds of matrix, and what a scale costs to store. The plan named above scores both.

All checks of the score ran on random matrices and one random two-layer network, with one seed. No model was read for this note.

## What the score is

A layer with a rows and b columns is read as a graph: one vertex for each row, one for each column, one link for each weight, carrying its size. A level is lowered from the largest weight. A link appears when the level reaches its size, and the levels at which separate pieces join are recorded. Those levels are the sizes of the a + b - 1 links of a maximum spanning tree.

The score of a layer is the p-norm of one minus those sizes, the sizes having been divided by the largest weight. It is divided by a bound to lie between 0 and 1, and averaged over layers. The stopping rule ends training when the score has not risen for a set number of checks.

`persist/persistlib.py` computes it. `persist/test_persist.py` checks the tree against every spanning tree of four small graphs, against a second method, and against the paper's example for its bound. `persist/demo.out` holds the figures below.

## Claims of the persistence document

| # | Claim | Verdict | What was found | Basis |
|---|---|---|---|---|
| 1 | The recorded levels are those of a maximum spanning tree, or of single-linkage clustering | Confirmed | Standard. The tree's sorted sizes are at least those of every other spanning tree, place by place, on 80 random cases with full enumeration | recomputed in `persist/test_persist.py` |
| 2 | Weights are divided by the largest absolute weight of the whole network, which makes the score ignore scale | Confirmed as the paper's definition | Section 3.1 of the paper: the largest is taken over "the set of weights" of one training step. It ignores one scale for the whole network, and nothing finer. See rows 11 and 12 | reported, recomputed in `persist/demo.out` |
| 3 | Theorem 1: the score of a layer is at most (largest - smallest) times (N - 1)^(1/p) | Confirmed, with the count misread | The paper says N is "the number of neurons in the layer". The document's cell on the count breaks off, and its Algorithm 1, line 10, writes a product of the two layer sizes. With neurons a random 650 by 650 layer scores 0.3685. With weights it scores 0.0204 | reported, recomputed in `persist/demo.out` |
| 4 | The bound is reached when one weight is large and the rest are equal | True under one convention | A connected graph ends as one piece, which has no joining level. Counted with the layer's smallest weight, the example reaches the bound exactly. Left out, it falls short: 2.884 against 2.993. Counted with level 0, as the paper's Algorithm 3 does for convolutions, it passes the bound: 3.053 | recomputed in `persist/demo.out`, reported |
| 5 | The bound normalises every layer to at most 1 | False with the network's largest weight | A 64 by 64 layer with weights ten times smaller than the network's largest scores 9.413. With its own largest weight it scores 0.422 | recomputed in `persist/demo.out` |
| 6 | Theorem 2: the score lies between the norms built on the n largest and the n smallest weights | True, and it says little | Any n - 1 links lie between the n - 1 largest and the n - 1 smallest weights. It holds for every spanning tree | derived |
| 7 | The cost is that of a sort and a union-find | Confirmed for the method named | Standard. This repository uses another method, on the dense matrix, which took about half a second for a 2560 by 10240 layer on the machine that ran the demo | derived, recomputed in `persist/demo.out` |
| 8 | Against stopping by validation loss, as pairs of (epochs, points of accuracy): (-0.53, -0.08) on Fashion-MNIST, (+0.17, -0.06) on MNIST, (-1.33, -1.13) on CIFAR-10, (-1.68, +0.07) on IMDB | Confirmed as the paper's table | The four pairs match the paper's. A negative first number is an earlier stop and a negative second a lower accuracy. So accuracy is within 0.1 point on three data sets and 1.13 points lower on CIFAR-10. The document pairs network sizes with data sets in a way the paper's tables do not show one for one | reported |
| 9 | With permuted labels the score's rule stays stable where validation loss does not. With little data it frees the validation split | Confirmed as the paper's appendix | Appendix A.7, on MNIST and Fashion-MNIST with one network of 500, 500 and 200 units | reported |
| 10 | The norm of all weights fails as a stopping rule because it never triggers | Confirmed as the paper's statement | | search |
| 11 | Not in the document: what the score ignores | The sign of every weight, and the order of rows and of columns of each layer on its own | Flipping the sign of a layer: score equal to six digits, outputs changed by 1.54 times their size. Reordering the rows of one layer alone: equal, 1.26 times | recomputed in `persist/demo.out` |
| 12 | Not in the document: what the score does not ignore | A rescaling that leaves the function as it is | For a ReLU unit, incoming weights times s and outgoing weights over s. Outputs equal to 1.8e-15. The mean score of two layers goes 0.445, 0.557, 0.787, 0.885 as the spread of s grows | recomputed in `persist/demo.out` |
| 13 | Not in the document: a published critique | Girrbach and others, TMLR 2023 | "variance of network weights and spatial concentration of large weights are the main factors". In later layers of deep networks they find no relevant spatial structure | reported |
| 14 | "Production readiness" | Larger than the evidence | Fully connected networks on four data sets. The paper itself says convolutional layers need more work | reported |

The document is not in this repository, so rows 3, 4 and 14 cannot be checked from here. What this note says of its cut-off cells rests on the quotations below, with the escape characters of the file removed.

A cell of the table in its section 1.4 is headed as the one about the count, and breaks off in the middle of a formula:

```text
Cross-Product Space & Vertex Cardinality:  While the bipartite layer graph  $G_k = (V_k \sqcup V_{k+1}, E_k)$  contains $
```

What the cell went on to say is not known. Line 10 of the document's Algorithm 1 writes the count as a product:

```text
NP_k_plus <- (w_max_k - w_min_k) * (|V_k * V_{k+1}| - 1)^(1/p)
```

Two more cells break off the same way, one in the same table and one in the table of section 2.4:

```text
The persistent span (distance to diagonal) is $\text{pers}(1, x) =
```

```text
Incremental edge addition using a Disjoint-Set Union-Find data structure with path compression and union-by-rank.,$O(n \cdot \alpha(n))$,$O(
```

Rows 11 and 12 together close one path under the closure rule, by counterexample: a score that is constant on moves that break the function, and varies on moves that keep it, is not a measure of behaviour. It may still say where the large weights sit.

## What the score reads

The largest link at any vertex is in the maximum spanning tree when it is unique. Cut that vertex from the rest: the largest link across the cut is the vertex's own largest, and a spanning tree without it is improved by a swap. `test_persist.py` checks it.

| Matrix | Gaussian | Student t, 3 degrees |
|---|---|---|
| 64 by 64, share of the tree that is a row's or column's largest entry | 0.756 | 0.717 |
| 256 by 256 | 0.765 | 0.736 |
| 128 by 512 | 0.854 | 0.834 |
| 256 by 1024 | 0.837 | 0.835 |
| 256 by 1024, score from the maxima alone over the true score | 0.9715 | 0.9832 |

So the score is close to a statistic of the largest entry of each row and each column, set against the largest of all. The `XR` rule of the gemma record sets a row's step from the row's largest entry, so the two read the same entries of the rows. The rule does not use the columns. Whether the score, or the plain spread of the rows, says which kinds of matrix are sensitive is predictions F7 and F8 of the plan.

## Claims of the two replies

| # | Claim | Verdict | What was found | Basis |
|---|---|---|---|---|
| 15 | A metric built from the expected product of the Jacobian with itself makes distance in weight space equal change of function | True for small moves on the inputs it was built from | It is the matrix G of weft-loom's gate calibration record. Its distance matched the probes within 1.1 on 12 of 12 hosts, was blind to moves the probes cannot see, and was off by more than 3 on a large natural move | reported from weft-loom |
| 16 | Point-vortex equilibria give codebooks that fit heavy tails | Not evaluated on heavy tails. No gain on Gaussian data | [pr6-proposal-check.md](pr6-proposal-check.md), row 17: on Gaussian data a vortex codebook is about as good as the best scalar quantiser. Its row 19: the changed force law meant for heavy tails was not evaluated | checked there |
| 17 | The fold of F_3 as a smooth path to a dense packing avoids the shock of rounding | End point closed by a ceiling, schedule not run | The same note, row 8: the end point is worth 0.25 dB, and no cell in 3 dimensions is worth more than 0.35. In 3 dimensions the lattice A_3\* gives a lower error than the end point, 0.257 dB against 0.247, and it is not the densest packing (`bounds/bounds.out`, sections 1 and 9). Issue #9 asks for the schedule | checked there, recomputed in `budget.out` |
| 18 | Symmetries of the weights are waste, and removing them removes resistance | Contradicted for rounding on one model | Balancing through the exact symmetries of gemma and rounding to 5 bits: KL 5.507. Rounding the weights as trained: 2.184. With the gain vectors moved into the matrices: 17.846. The record did not examine why | checked, `budget.out` |
| 19 | Aligning permutations joins separately trained models with no loss barrier | Right for another problem | Git Re-Basin reports it for ResNets on CIFAR-10. It is about merging two models. For a gate, weft-loom's CL-5 measured distance after the best relabelling and found no usable radius on 12 hosts | reported |
| 20 | "mLoRA-Asym": freezing some entries of an adapter breaks its symmetry | No source | A search for the name found no paper. The nearest work fixes a chosen number of entries in each row of a network's weights (Lim and others, 2024) | reported |
| 21 | A layer-wise loss penalises harmless rotations, and distances along a sparse rigid graph fix that | Wrong as stated | The same note, row 15: the proposed loss is zero for any orthogonal change of the output, including one that reverses every output | checked there |
| 22 | Lookup-free decoding of incentre codes removes the cost of search | Partly right | Memory read per token is a real cost. E8 already needs no table ([brief-check.md](brief-check.md), row 3). As proposed the incentre codes do not beat a scalar mix at equal bits, and F_8 loses to the scalar rule on every matrix. No record has measured the speed of any decoder | checked there |
| 23 | Neural persistence filters "topological noise": cycles in a Vietoris-Rips complex of the weights, pruned by lifetime | Wrong | The paper uses the weight graph of each layer and pieces joining, not cycles and not a Vietoris-Rips complex. It prunes nothing. Persistent homology dimension is from other work (Birdal and others, 2021), whose abstract does not say what the reply says of it | reported |
| 24 | With these in place, optimisation "becomes natural self-organization" | Not a claim that can be checked | | |

## What is left: a budget

The replies name devices. What carries over from drag reduction is the habit behind the devices: split the loss into parts with floors, and work on the largest part. For rounding the parts can be put in bits per weight. `notes/budget.out` holds the arithmetic.

| Part | Size | Scored on a model's output? |
|---|---|---|
| The shape of a cell | At most 0.255 bit at high rates in any dimension. E8: 0.108 | E8 against scalar, in the gemma record |
| The shape of a code's region | With one scale per row on Gaussian data, 1.64 dB of E8's gain is from its region: 0.272 bit ([e8-gain-largest-entry-rule-2026-11-17](../experiments/e8-gain-largest-entry-rule-2026-11-17/README.md)) | No. Weight error only |
| The storage of scales | 0.500 bit at 16 bits for every 32 weights. 0.250 at 8 bits. 0.188 at 6 | No |
| Bits spread evenly over kinds of matrix | Unknown. At equal bits one kind costs 52.5 times the KL per weight of another, under arm S | No |
| The few large entries | Keeping the 188,362 entries of `ffn_down` beyond 5 standard deviations apart costs 0.005 bit per weight of that kind. Pulling them back costs a KL of 0.164 | The clip only. Not the keeping |

llama.cpp's own K formats already act on two of these rows: they store scales in 6 bits, and `Q4_K_M` gives some `attn_v` and `ffn_down` tensors a wider type by a fixed rule (reported, from the project's pull request 1684 and its source). No record here has measured what either is worth at equal bits.

If one more bit divides a kind's KL by four and the parts add, the split with equal gain per bit moves the modelled sum from 3.375 to 1.306 at the same total. Both assumptions are untested, and the table it starts from is from the arm that breaks the model. It is the arithmetic of a prediction.

## What follows

One study, planned before any run: `experiments/output-budget-gemma-e4b-2026-11-17/PREREG.md`. Until its own pull request is merged it is on the branch `record/output-budget-gemma-e4b-20261117`. It scores the last three rows of the budget on gemma's output, asks whether the damage of a clip adds up over layers as rounding's does, and computes three more measures of a weight change, the tree shift among them, with the failures predicted above.

Not proposed: early stopping by this score. Nothing in this repository trains a model.

## Sources

- Rieck, Togninalli, Bock, Moor, Horn, Gumbsch, Borgwardt, "Neural Persistence: A Complexity Measure for Deep Neural Networks Using Algebraic Topology", ICLR 2019, arXiv 1812.09764. Main text and appendix. **Reported.** The appendix was read through a tool's summary.
- Girrbach, Christensen, Winther, Akata, Koepke, "Addressing caveats of neural persistence with deep graph persistence", TMLR 2023, arXiv 2307.10865. **Reported.**
- Ainsworth, Hayase, Srinivasa, "Git Re-Basin", ICLR 2023, arXiv 2209.04836. **Reported.**
- Lim, Putterman, Walters, Maron, Jegelka, "The Empirical Impact of Neural Parameter Symmetries, or Lack Thereof", NeurIPS 2024, arXiv 2405.20231. **Reported.**
- Dettmers and others, "SpQR", arXiv 2306.03078: outlier weights kept at higher precision. **Reported.**
- Birdal, Lou, Guibas, Simsekli, arXiv 2111.13171. **Reported**, abstract only.
- llama.cpp pull request 1684, "k-quants", and `src/llama-quant.cpp` on its main branch on 2026-11-17, which may differ from commit `427291b`. **Reported.**
- The limit of 1.53 dB for any cell: Zador and Gersho. **Search.** The figure is recomputed in `notes/budget.out`.
- weft-loom: `experiments/gate-metric-calibration-2026-11-15` and `notes/closure-counted.md`. **Checked.**

The source checks were run by AI agents. Their reports were read against the code and the numbers in this repository before the tables were written.

## Disclosure

The note, the toolkit and the checks were written by an AI agent session under the owner's direction.
