# Exact Pure-$\mathbb{Q}$ SSD-Streaming Roofline, `DSpark`-vs-`Uno` Budget Crossover, and Sparse `MoVA` Water-Filling Study

Self-contained, zero-float pure-$\mathbb{Z}/\mathbb{Q}$ (`fractions.Fraction`) study analyzing three interacting inference-budget operators relevant to low-bit quantization and local execution:

1. **SSD-Streaming Roofline & Phase Transition (`exact` arithmetic over external `anchor` throughput reports):** Rational Zipf-1 hot-expert harmonic model ($H_{16} = 2436559/720720$) reproducing the sharp `9 GiB -> 10 GiB` SSD-dominance transition and live `20 -> 8 -> 20 GiB` resize ratio (`48760831/21929031`), parameterized against reported Apple M4 Max Inkling 2 Mini anchor figures (`109.53 tok/s` vs `33.11 tok/s`, ratio `10953/3311 ≈ 3.3080x`).
2. **Exact Fixed-Memory `DSpark` vs. Zero-Draft `Uno` Crossover (`exact` linear solve):** Renewal-reward external drafter speedup $S_{\mathrm{ext}} = 3843/1250 = 3.0744\times$ (requiring `4 GiB` resident draft memory, feasible domain $B \ge 4\text{ GiB}$) vs. zero-draft adapter speedup $S_{\mathrm{uno}} = 11/5 = 2.2\times$ (`0 GiB` extra memory) yields the exact unique crossover memory budget:
   $$B^* = \frac{3625767011}{304898880}\text{ GiB} \approx 11.8917\text{ GiB}$$
   (zero residual): for $4 \le B < B^*$, zero-draft `Uno` strictly dominates external drafting, whereas for $B > B^*$ external drafting wins.
3. **Sparse `MoVA` Fisher Water-Filling vs. Dense Uniform Allocation & Outlier-Expert Task-Break (`exact`):** Exact rational Lagrange water-filling demonstrates `263/16 = 16.4375x` KL gain on sparse `MoVA` vs. `5/4 = 1.25x` on dense layers (`263/20 = 13.15x` leverage, `+1/512` strict convexity excess) plus a `CS-20` outlier-expert logit-margin flip (`+1 -> -1/2 < 0` under uniform bpw vs. `+7/8 > 0` under Fisher-protected allocation).
4. **Greedy Parity & Scope Ledger (`exact` SHA-256 + honest `anchor` attribution):** Certifies `64/64` token lossless greedy parity (`b7fd068c...`) vs. single-token drift (`713ac840...`) and records external benchmark numbers honestly as `anchor` references (no live model weights or hardware kernels are executed in this pure-stdlib study).

## Reproduction (3-Mode R4 Parity + `PYTHONHASHSEED=42`)

```bash
cd studies/expedition-atomic-mova-uno-roofline
for flags in "-I -B" "-I -O -B" "-I -B -W error"; do
  python3 $flags atomic_mova_uno_audit.py --standalone --json | sha256sum
  python3 $flags atomic_mova_uno_audit.py --check atomic_mova_uno_certificate.json --json
  python3 $flags test_atomic_mova_uno_r5.py
done
PYTHONHASHSEED=42 python3 -B atomic_mova_uno_audit.py --standalone --json | sha256sum
```

Expected certificate SHA-256 across all modes: `651338833ae035bd0cb91709fab1222ca3976e8f87d14bfcc475998410019270`.

## Shipped File Manifest

| File | SHA-256 |
| :--- | :--- |
| `atomic_mova_uno_oracle.py` | `afb2afeea4183f49164f2fff5d6a4457174d79b20d1d1ee07fce295f6a8a6b89` |
| `atomic_mova_uno_audit.py` | `d0a3c3b1b1198030c831781b8a126dddeb75b8a967a9e27349362739665e3ffa` |
| `test_atomic_mova_uno_r5.py` | `1c6a009506eca88b9ced29b3e926e2c87fd81c908fe4c7c8d24eae28209417fc` |
| `atomic_mova_uno_certificate.json` | `651338833ae035bd0cb91709fab1222ca3976e8f87d14bfcc475998410019270` |
| `ATOMIC_MOVA_UNO_ROOFLINE_REPORT.md` | `b2417a39cd39c3cf17b5ae0ab9dd8dbb567b3893223b2caedf9211234a989115` |
