# The owner's list of concepts, mapped

Date: 2026-11-15
Status: map. Each line says what is proved, what is tested here and what is only a reading.

The project owner listed 22 concepts on 2026-11-15, beside a rule proposed for ML Weft: close a path, or a whole sector of paths, only on a formal proof that nothing valid lies in it. The rule and its first register are in weft-loom, `notes/closure-by-certificate.md` and `notes/closure-counted.md`.

## What the list is about

Most of the list counts, bounds or estimates one number: **how many one-to-one matchings avoid a set of forbidden pairs?** Gray code ordering is a device for computing it. The last three items, the Birkhoff polytope, differentiable permutohedra and BosonSampling, are neighbours of the question and not answers to it.

That number is the closure rule's own question when the paths are assignments. A forbidden pair is a closed cell. The count of what is left is what the closure is worth. Reading: the list is the arithmetic of the rule.

Rounding weights meets the rule in three places, each taken up below: a grid makes every sector a finite set, closing a region with balls is a covering problem, and proved limits close parts of the search for a better rounding scheme.

## The concepts

Code is in `perm/permlib.py`, with tests in `perm/test_permlib.py`. "Exact" means whole-number arithmetic checked against full enumeration on small cases. Figures quoted below that the tests do not print are in `perm/demo.out`.

### Counting matchings that avoid forbidden pairs

| Concept | What it gives | Code and check | Where it meets the closure rule |
|---|---|---|---|
| Inclusion-exclusion | An exact count of objects that avoid every one of a set of conditions | `hit_numbers`, `count_zero_sum_box`. Exact | The number of assignments that avoid every closed cell |
| Derangements | The count when one cell per row is closed | `derangements`, three ways, equal up to n = 40 | One closure per row is worth n!/D_n, which tends to e |
| Recursive derangement formulae | D_n = (n-1)(D_(n-1) + D_(n-2)) and D_n = n D_(n-1) + (-1)^n | both tested against the sum | A count updated as cells close, without starting again |
| Rencontres numbers | How many permutations fix exactly k points | `rencontres`. Exact to n = 7, sums to n! | The law of how many closed cells a random proposal uses |
| Montmort's matching problem | The chance of no match tends to 1/e, with error under 1/(n+1)! | tested to n = 15 | The share of proposals that survive one closure per row |
| Rook polynomials | r_k, the ways to place k rooks on the forbidden cells with none attacking | `rook_numbers`, three independent ways, with brute force to 6 rows | The input to every count above for an arbitrary closed set |
| Hit polynomials | h_j, the permutations that use exactly j forbidden cells | `hit_numbers`. Exact | h_0 is the number of valid assignments left |
| Ménage problem | The count when two cyclic diagonals are closed | `menage`. Matches M_3 to M_10 | Two closures per row are worth 8.25 at n = 10, and tend to e squared, 7.39, as n grows |
| Stein-Chen method | A bound on the distance between the law of hits and a Poisson law. Its proof is in weft-loom, `notes/closure-counted.md`, from a standard theorem that was not opened | `stein_chen_bound`. Holds on every board up to 4 by 4 and 8,000 random larger ones | A bracket for the share removed when n is too large to count. It is loose unless n is large |

**Deletion and contraction, and whether it can be made recursive.** It is recursive as it stands. For a cell c of the board, R(B) = R(B without c) + x R(B without the row and column of c). `rook_numbers_deletion_contraction` runs it with two additions: results are remembered, and a board that splits into parts sharing no row or column is the product of its parts.

It stays exponential in the worst case. No fast method is expected, because the count is a permanent and counting permanents is #P-complete. It is fast when the closed cells fall into small separate blocks. Reading: sectors that share no row or column multiply, so their worths compound exactly.

### Permanents, exact

| Concept | What it gives | Code and check | Where it meets the closure rule |
|---|---|---|---|
| Permanent | The number of valid assignments is the permanent of the 0/1 matrix of open cells | `permanent_bruteforce`, `permanent_ryser`, `permanent_glynn`. All agree | The exact worth of a closed set. With weights it handles a proposer that is not uniform |
| Ryser's formula | The permanent in n 2^n steps in place of n! | perm(J) = n! and perm(J - I) = D_n to n = 16 | Exact counts. Tested to 16 rows. About 25 is an estimate of what is affordable |
| Gray code ordering | Each step of Ryser's sum changes one column | built into `permanent_ryser`, with no test of its own | None of its own. It is what makes the exact count affordable |

### Permanents, bounded or estimated

| Concept | What it gives | Code and check | Under the closure rule |
|---|---|---|---|
| Bregman-Minc bound | A proved ceiling for 0/1 matrices | `bregman_minc_upper`. Holds on 500 random matrices | A ceiling on what is left. If it reaches zero the sector is closed |
| Van der Waerden bound | A proved floor of n!/n^n for doubly stochastic matrices | `van_der_waerden_lower` | A floor on what is left, valid only at exact scaling |
| Sinkhorn scaling | Rows and columns scaled in turn to sum 1 | `sinkhorn`, `sinkhorn_permanent_bounds` | The ceiling it gives is proved after any number of rounds. The floor is not |
| Godsil-Gutman determinant | An unbiased estimate from a random signed determinant | `godsil_gutman`. Mean within 5 standard errors | An estimate. Its spread grows with n (1.6 to 3.4 times the permanent for n = 4 to 10 on the all-ones matrix) |
| Jerrum-Sinclair-Vigoda, and "FPRAS for matrices" | A randomised scheme within any factor in polynomial time, for non-negative matrices | not implemented | It can rank sectors by size. It cannot close one, because its guarantee is a probability |
| Approximate permanent algorithms in general | See the three rows above | | Estimates steer effort. Only the proved bounds close anything |

### Scaling, polytopes and what lies beyond

| Concept | What it is | Where it meets the rule or rounding |
|---|---|---|
| Sinkhorn scaling, again | As a way to find cells that no valid assignment uses | It finds them only in the limit. On one 4 by 4 example the entry falls like 1/(2k). The exact rule with a counting certificate is `dead_cells`, tested against enumeration |
| Birkhoff polytope | The doubly stochastic matrices. Its corners are the assignments | A proposer over assignments is a point of it. The best relabelling of hidden units, used in the weft-loom register (CL-5), is a corner |
| Differentiable permutohedra | Sorting made smooth by projecting onto the permutohedron | The permutohedron is the cell of the lattice A_n\*. It could pass gradients through rounding to that lattice. No published use for lattice rounding was found, and nothing is tested here |
| BosonSampling | Sampling whose probabilities are squared sizes of permanents of complex matrices | No link to rounding weights was found. Its lesson for this list: permanents with signs or phases cannot be approximated the way non-negative ones can, so keep to 0/1 and non-negative matrices |

## Where rounding meets the closure rule

1. **A grid makes sectors countable.** In a continuous weight space a closed ball can hold a vanishing share of what a proposer tries. On a grid a sector is a finite set, and the counts above apply. On the 12 test hosts in weft-loom the proved ball is still smaller than one grid step from 5 to 7 bits (`notes/closure-counted.md` there, CL-7).
2. **Closing a region is covering it.** The lattice A_n\* is the thinnest lattice covering up to 5 dimensions. In 8 it needs about 18 times fewer balls than the cubic grid. As a quantiser in 8 dimensions it is behind E8. See `bounds/bounds.out`, sections 2 and 4.
3. **Proved limits close parts of the search for better rounding.** No code beats Shannon's floor. No cell shape in 8 dimensions is worth more than 0.73 dB at high rates. A proposal is read against `bounds/bounds.out` before anything is run.
4. **The gate and the quantiser want the same map.** Both need to know what a change of weights does to outputs. On the test hosts, rounding to 5 bits moves the weights 5 to 23 times farther than a move that breaks the model, and does no harm. Distance in weight space does not rank them. (weft-loom, `notes/closure-counted.md`, CL-6.)

## Not done

- Nothing here measures a real proposer. The worth of a closure is computed for uniform proposals only.
- The Bethe approximation of the permanent was tried and left out. The iteration written for it did not converge reliably on 3 by 3 matrices. That is a fault of that code and says nothing about the method.
- The randomised scheme of Jerrum, Sinclair and Vigoda is not implemented.
