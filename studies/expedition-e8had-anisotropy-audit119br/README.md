# Study (`AUDIT-119BR`): Exact-$\mathbb{Q}$ Coordinate-Axis Outlier Anisotropy, Low-Bitrate Shell-Gap/Steiner Crossover, `Block-H2 == WHT_8` Identity & Gate-8 Pre-Registration for $E_8^{\mathrm{Had}}$

- **Scope Boundary:** Strictly additive, self-contained standalone research study under `studies/expedition-e8had-anisotropy-audit119br/`. No files under `shape/`, `bounds/`, `experiments/`, `tests/`, or `.github/` are modified; this package is a standalone exact-arithmetic analysis and pre-registered Gate-8 verifier, not wired into the core `shape/` runtime.
- **Oracle Class:** `exact+empirical-corpus` (pure Python 3 standard library over exact $\mathbb{Z}/\mathbb{Q}$; zero floating-point literals or operations on trust/certificate paths; zero bare `assert` statements; zero third-party dependencies).
- **Novelty Class:** `N1` — exact-$\mathbb{Q}$ mechanistic proofs explaining three empirical real-weight regimes, global $O(8)$ minimax optimality and uniqueness up to $S_8$ via doubly-even self-dual `[8,4,4]` codes, the infinite-lattice `Block-H2 == WHT_8` `3x` butterfly-cut identity, and a cryptographically anchored Gate-8 pre-registration (`G8-P1`..`G8-P4`).
- **Attribution & Empirical Snapshot Anchor:**
  - $E_8^{\mathrm{Had}} = U E_8$ ($U = H_8 / \sqrt{8}$) construction credited to `Moissanite-Labs/siliquant` PR `#24` (`AUDIT-028`).
  - Live real-weight evaluation runs (`Llama-3.2-1B`, `SmolLM2-1.7B`, `Qwen2.5-1.5B`) credited to `Moissanite-Labs/siliquant` at `51235fa753ecd32a15c9af0251560156e3bf440f` (`2026-11-22T23:06:57Z`), `experiments/e8-hadamard-real-weights-2026-11-22/results.jsonl` (`13/15` scored rows landed, SHA-256 `e4023053b6f17cd7a0e820cc591a328d0b4ad6738d10c40d8a5f9e2a9920aa5c`; `Qwen2.5 EBG128:b368s16` and `Qwen2.5 HBG128:b368s16` strictly unlanded at this snapshot).

---

## Manifest

| File | Lines | Bytes | SHA-256 | Role |
|---|---:|---:|---|---|
| [`E8HAD_ANISOTROPY_GATE8_REPORT.md`](./E8HAD_ANISOTROPY_GATE8_REPORT.md) | `103` | `15,884` | `91b56a6855c6e548d5ee22208076dc7f8509f7b0de7a8666c44aa75849fba65f` | Full mathematical derivations, empirical decomposition (`51235fa7`), Gate-8 pre-registration table, and Law-4 scope boundaries |
| [`e8had_anisotropy_oracle.py`](./e8had_anisotropy_oracle.py) | `1,039` | `48,286` | `dc13922753737ec127cd59817479680d7615b2a326d2664b56d0016fb146dc72` | Pure-$\mathbb{Z}/\mathbb{Q}$ exact lattice oracle (`k`-sparse Voronoi error, 8 orthogonal coset families, `Block-H2 == WHT_8` proof, `51235fa7` decomposition, deterministic `evaluate_gate8_rows` scorer) |
| [`e8had_anisotropy_audit.py`](./e8had_anisotropy_audit.py) | `141` | `5,865` | `574f39a049287706026adbd072d78d10b463ae29f8b65ee337d3201fc238b561` | Standalone read-only verifier (`--check`), canonical JSON parser, and Zone-B I/O boundary checks |
| [`e8had_anisotropy_certificate.json`](./e8had_anisotropy_certificate.json) | `1` | `10,252` | `444fb585f18f79f4073b21edacc54991bfad90c51d2cc3fe5275047eb38d4f6f` | Sealed canonical JSON certificate (`payload_sha256: a7453fbf...`, `certificate_sha256: d7002dbc...`) |
| [`test_e8had_anisotropy_r5.py`](./test_e8had_anisotropy_r5.py) | `628` | `32,824` | `bd4051d8ef553d46c174acb6c345bc367e4067d6f7a3a52fe002be9821fbcb14` | Adversarial R5 & Zone-B regression suite (`6` test classes covering mathematical identities, must-fail certificate tampers, and public-API strict type hardening) |

---

## Quick Reproduction (Zero External Dependencies)

Run from this directory (`studies/expedition-e8had-anisotropy-audit119br/`):

```bash
# 1. Four-mode isolated parity check on certificate generation (all 4 must match 84e79bb9...)
python3 -I -B e8had_anisotropy_oracle.py --standalone --json | sha256sum
python3 -I -O -B e8had_anisotropy_oracle.py --standalone --json | sha256sum
python3 -I -B -W error e8had_anisotropy_oracle.py --standalone --json | sha256sum
PYTHONHASHSEED=42 python3 -I -B e8had_anisotropy_oracle.py --standalone --json | sha256sum
# Expected SHA-256 (all 4 modes): 444fb585f18f79f4073b21edacc54991bfad90c51d2cc3fe5275047eb38d4f6f

# 2. Standalone read-only certificate audit across all 3 isolated interpreter modes
python3 -I -B e8had_anisotropy_audit.py --check e8had_anisotropy_certificate.json
python3 -I -O -B e8had_anisotropy_audit.py --check e8had_anisotropy_certificate.json
python3 -I -B -W error e8had_anisotropy_audit.py --check e8had_anisotropy_certificate.json

# 3. Full adversarial R5 + Zone-B unit test suite across all 3 isolated interpreter modes
python3 -I -B test_e8had_anisotropy_r5.py
python3 -I -O -B test_e8had_anisotropy_r5.py
python3 -I -B -W error test_e8had_anisotropy_r5.py
```

---

## Summary of Certified Results

1. **Exact-$\mathbb{Q}$ $k$-Sparse Coordinate-Outlier Anisotropy (`Law 1`):** Rotating $E_8 \to E_8^{\mathrm{Had}}$ eliminates all `128` weight-8 roots (`128 -> 0`) and places `16` weight-1 roots $\pm\sqrt{2}\,e_i$ directly on the coordinate axes, halving 1-sparse Voronoi error ($\max d^2: 1 \to 1/2$, $\mathbb{E}[d^2]: 1/3 \to 1/6$) and raising the 1-sparse neighbor-bleed threshold from $\varepsilon_{\mathrm{crit}} = 0$ to $\varepsilon_{\mathrm{crit}}^2 = 1/2$. On `Llama-3.2-1B @ 51235fa7`, this explains why `HBG128` cuts loud-column KL damage by `26.71%` (`3949/14783`), accounting for `95.78%` (`3949/4123`) of the full `HBG128` gain and why `+K10` collapses the gap (`10.27% -> 0.69%`).
2. **Global $O(8)$ Minimax Optimality & Uniqueness (`Law 2`):** Across all $R \in O(8)$, $\sup_t d^2(t e_i, R E_8) \ge 1/2$, with simultaneous equality across all 8 axes iff $\sqrt{2}\,R E_8 / 2\mathbb{Z}^8$ is a doubly-even self-dual `[8,4,4]` code (uniquely the extended Hamming code $\mathcal{H}_8$ up to $S_8$ coordinate permutation).
3. **Voronoi-Gauge Reversal (`Law 3`):** Because $g^2(e_i; E_8^{\mathrm{Had}}) = 2$ vs $g^2(e_i; E_8) = 1$, `maxscale_voronoi` (`VH8Z32`) inflates group scales by up to $\sqrt{2}\times$ on 1-sparse outliers, explaining the `+7.03%` Qwen2.5 KL degradation (`H11` falsified at `51235fa7`) while spherical `HBG128` remains $O(8)$-invariant.
4. **Low-Bitrate Shell-Gap & Steiner-Quadruple Forced-Jump Crossover (`Law 4`):** Both $E_8$ and $E_8^{\mathrm{Had}}$ have zero nonzero shell in $(0, \sqrt{2})$ (unlike $\mathbb{Z}^8$'s unit shell at $1$), and $\mathcal{H}_8$'s `3-(8,4,1)` Steiner quadruple structure forces a 4th tiny coordinate to jump by squared error `1/2` (`2x` larger than $E_8$'s `1/4`, vs `0` in $\mathbb{Z}^8$), matching SmolLM2 3.0-bit `HBG/EBG` small-weight energy ratios (`-4.55 dB` predicted vs `-4.67 dB` observed on `<0.05 RMS`; `-0.07 dB` vs `+0.025 dB` on `<0.2 RMS`).
5. **`Block-H2 == WHT_8` Exact Infinite-Lattice Identity (`Law 5`):** $A = U_{\mathrm{BH2}}^{-1} U_{\mathrm{WHT8}} = \frac{1}{2}(H_4 \otimes I_2) \in W(E_8)$ satisfies $A^\top A = I_8$ and maps all `240/240` roots of $E_8$ into $E_8$, so the 1-stage pairwise transform $U_{\mathrm{BH2}} = \frac{1}{\sqrt{2}}(I_4 \otimes H_2)$ (`8` add/subs per 8-block) generates the **exact same infinite lattice** $E_8^{\mathrm{Had}}$ as 3-stage `WHT_8` (`24` add/subs) at **`3x` lower butterfly cost**.
6. **Pinned Gate-8 Pre-Registration (`G8-P1`..`G8-P4`):** Deterministic zero-float scorer `e8had_anisotropy_oracle.evaluate_gate8_rows(rows)` anchored strictly to the `51235fa7` snapshot (`13/15` landed rows), covering the two unlanded Qwen2.5 3.0-bit rows (`G8-P1`), unrun Llama `VH8Z32` `+/-K10` (`G8-P2`), unrun Qwen2.5 4.5-bit `+K10` (`G8-P3`), and drop-in `Block-H2` vs `WHT_8` equivalence (`G8-P4`).

---

## Scope Limits (Law 4)

- Lattice theorems (`Laws 1–5`) are exact mathematical statements in $\mathbb{R}^8$ proved over $\mathbb{Z}/\mathbb{Q}$ independent of any neural network.
- Empirical decomposition is scoped strictly to the `13/15` committed `wikitext-2` rows across `Llama-3.2-1B`, `SmolLM2-1.7B`, and `Qwen2.5-1.5B` at `51235fa7`; no claims are extrapolated to larger parameter counts, alternate calibration/imatrix schemes, or non-`ffn_down` column-keeping heuristics beyond the explicitly falsifiable criteria in `G8-P1`..`G8-P4`.
