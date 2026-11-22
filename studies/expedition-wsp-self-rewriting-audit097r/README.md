# Weight-Edit Minimization, Self-Rewriting Governance & `siliquant` Quantization-Survival Study (`AUDIT-097R`)

Self-contained, pure Python 3 stdlib (`int` / `fractions.Fraction`, zero floats on trust paths or JSON, zero bare `assert`s) study independently `VERIFIED` across two model families (`Muse Code` + `verifier`, `185/185` `aletheia v2.0.0 mutate` killed, `156/156` AST `BoolOp` killed, `0` survivors).

## Scope & Direct Bridges to `siliquant`

- **Exact Subspace-Leakage & Safety-Orthogonality Dichotomy (`Exp 8` `Qwen3.5-0.8B` `down_proj`):** Column pruning preserves exact rank-1 (`0/1` leakage, $S^\top \Delta = 0$), row pruning obeys $\|(I-P_u)\Delta_{\mathrm{row}}\|_F^2 = E_R(1-E_R)\|\Delta_0\|_F^2$ (`6,113.75×` leakage on `ER384_L15`), and entry-wise pruning shatters rank-1 (`23,696.11×` leakage on `D4k_entries`).
- **Live `ffn_down` Symmetric-INT4 (`qmax = 7`) Quantization-Survival Double Bind (Part L, `#10`, `#12`, `#21`, `#26`):**
  - Small dense rank-1 edits (`Exp 8` `L15+L19`, `G = 128`): **`57,060 / 130,188` (`43.83%`)** completely erased, **`79,380 / 130,188` (`60.97%`)** below `(3/4)·ULP` (`siliquant#26`), and **`87,619` unedited weights flipped collaterally** (`1.20×` surviving edits).
  - Sparse $L_0$-spike edits (`car-ramp` `C2-min`, `greedy12`): group scale inflates `60.979×`–`657.963×`, zeroing **`100%`** of non-zero unedited neighbors (`29/29` at `G=32`, `80/80` at `G=128` in `C2-min`; `24/24` and `90/90` in `greedy12`).
- **Paired Dual-Metric Screening & 8-Gate Self-Rewriting Governance (`arXiv:2611.21885v2`):** Exact pure-$\mathbb{Q}$ McNemar `cut_at_32k` catches hidden loop-breaking regressions (`D4k_entries`: `p = 0.008147 <= 0.01`), and the 8-gate fail-closed governance oracle audits self-editing session protocols (`G1`–`G8`).
- **Honest Novelty Standing (Law 4):** `N1` (exact pure-$\mathbb{Q}$ executable operator bridge, bit-level live recount, and fail-closed verification gates) over `N0` primary artifacts (`arXiv:2611.21885v2`, `alphaXiv/weight-edit-minimization @ 768d7340`).

## Offline Reproduction (4-Mode Isolated `-I` Parity)

The committed [`wsp_self_rewriting_certificate.json`](wsp_self_rewriting_certificate.json) (`sha256: 212611b157019453b9de2f2b31b96c883f26cc9f5e4773c4524e74cd0b1b70f7`) was generated in `--live` mode against pinned read-only upstream intake files. In this standalone repository snapshot, `--standalone` replays all pure-$\mathbb{Q}$ theorems, fixtures, and certificate validators offline with zero external files (bit-identical `28a5d1ce2e80ceb9375a793332e8e27ea5b4d6033b44983f5cdf0b2b09f1ad32` across all 4 isolated modes):

```bash
python3 -I wsp_self_rewriting_audit.py --standalone --json | sha256sum
python3 -I -B wsp_self_rewriting_audit.py --standalone --json | sha256sum
python3 -I -O -B wsp_self_rewriting_audit.py --standalone --json | sha256sum
python3 -I -O -B -W error wsp_self_rewriting_audit.py --standalone --json | sha256sum
python3 -I -O -B -W error test_wsp_self_rewriting_r5.py -v
```

See [`WSP_SELF_REWRITING_AUDIT_REPORT.md`](WSP_SELF_REWRITING_AUDIT_REPORT.md) for the full audit tables and falsification criteria.
