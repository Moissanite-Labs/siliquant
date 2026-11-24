# Exact Pure-Q Fisher Water-Filling, Task-Break Logit-Margin Bifurcation, P99 Order-Statistic Bounds, and Latent-Memory Airlock Study

Self-contained, zero-float pure-Z/Q (`fractions.Fraction`) analytical study (`5` verified files + this README, zero external dependencies, zero bare `assert`s; no live model weights downloaded or executed):

1. **AM-GM Fisher Water-Filling Rate-Distortion Law (`exact`):** On a heavy-tailed 8-row sensitivity spectrum at zero extra average bits-per-weight (`extra_average_bpw = 0`), exact Lagrange water-filling achieves `263/16 = 16.4375x` lower distortion than uniform allocation (`K_Fisher = 1/32`, `3/3` strict convexity perturbations `+1/512`, `+9/1024`, `+11/1024`), vs. `389/128 = 3.0390x` on a mild skew and `1x` on a uniform spectrum. **Scope caveat (`#77` `notes/colproxy.out`):** Real-weight tests in `#77` showed that data-free weight-only proxies (`1/(column RMS)^2`, RMSNorm gain) do not predict the importance matrix; realizing water-filling gains on real models therefore requires calibration-measured activation/importance-matrix statistics, not data-free weight norms.
2. **Task-Break Logit-Margin Inversion vs. Smooth Average KL (`exact`):** Certifies an explicit 4-token / 4-vocab witness on BOTH exact quadratic Fisher KL (`1499/176 = 8.5170x` higher average KL on the passing arm) AND exact dyadic Shannon KL (`21/4 = 5.25x` higher average KL on the passing arm) where uniform allocation lowers average KL while flipping the critical-token margin negative (`-1/2` / `-1/4`, task FAIL) and targeted protection preserves positive margin (`+1/4`, task PASS).
3. **Exact Binomial `P99` Order-Statistic Sample-Size Law (`exact`):** Certifies exact `(99^n, 100^n)` bounds across `n in {100, 458, 459, 1099}`: sharp `99%` upper-tail threshold (`n=458` FAIL vs. `n=459` PASS) and exact `n=1,099` two-sided coverage `[X_(1081), X_(1096)] <=> 4 <= E <= 18` (`978,156..978,157 ppm = 97.8156% > 95%`; narrower `[X_(1081), X_(1095)] <=> 5 <= E <= 18` gives `968,111..968,112 ppm = 96.8111%`).
4. **Fixed-Rank State Capacity Obstruction & Channel-2B Latent-Memory Poisoning (`exact`):** Proves rank-4 `GDN-2` state recovers `L=4` orthogonal keys with zero error (`0`) while failing `L=5` with universal Cauchy-Schwarz lower bound `106/5 > 0` (Parseval exact `212/5`), and certifies a persistent rank-1 latent-memory poisoning witness flipping target margin `+2 -> -83/25` under frozen weights (`Delta W = 0`), with pinned canonical state SHA-256 checks.

## Reproduction (3-Mode R4 Parity + `PYTHONHASHSEED=42`)

```bash
cd studies/expedition-fisher-kintsugi-memt
for flags in "-I -B" "-I -O -B" "-I -B -W error"; do
  python3 $flags fisher_kintsugi_memt_audit.py --standalone --json | sha256sum
  python3 $flags fisher_kintsugi_memt_audit.py --check fisher_kintsugi_memt_certificate.json --json
  python3 $flags test_fisher_kintsugi_memt_r5.py
done
PYTHONHASHSEED=42 python3 -B fisher_kintsugi_memt_audit.py --standalone --json | sha256sum
```

Expected certificate SHA-256 across all modes: `ba44be55ca16873671b41f7e0cac32cf5c43361a07cff7c4ff7b04fcc917cee7`.

## Shipped File Manifest

| File | SHA-256 |
| :--- | :--- |
| `fisher_kintsugi_memt_oracle.py` | `a2b458d3e8e78b5807ec26228acc5ed89491a12fb1fe14498875315c7b046169` |
| `fisher_kintsugi_memt_audit.py` | `68104af983094015d7eb1a50045e7c9e7b040fdfeeeae3033098699484abfacb` |
| `test_fisher_kintsugi_memt_r5.py` | `30cfe8c4ef4a5245060f2d8966ce54188c89feb4d017e9ec85557fa626058aee` |
| `fisher_kintsugi_memt_certificate.json` | `ba44be55ca16873671b41f7e0cac32cf5c43361a07cff7c4ff7b04fcc917cee7` |
| `FISHER_KINTSUGI_MEMT_REPORT.md` | `8785f82c6c1ae4f902d0b3756ed92070bd18b74f99c569761b766c2af3a8874e` |
