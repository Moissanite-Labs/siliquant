# Extended Hamming Construction-A Lattices ($A_8, A_{16}, A_{32}$), Exact Syndrome Decoding & Sharp `f64` Exact-Decode Threshold

Standalone, zero-float pure-$\mathbb{Z}/\mathbb{Q}$ certification study accompanying `Moissanite-Labs/siliquant#77` (zero external dependencies; standard library only).

## Highlights

1. **Construction-A Lattices $A_8, A_{16}, A_{32}$ (`[8,4,4]`, `[16,11,4]`, `[32,26,4]`):**
   - Exact $H_n G_n^\top \equiv 0 \pmod 2$, exhaustive $C_8$/$C_{16}$ plus MacWilliams polynomial expansion for $C_{32}$ (`67,108,864` codewords), weight-4 counts $A_4 = (14, 140, 1240)$, covolumes $(1, 1/8, 1/1024)$, Gram determinants $(1, 1/64, 1/1048576)$, kissing numbers $\tau = (240, 2272, 19904)$ with exact `240/240` bijection $\frac{1}{2}H_8^{\mathrm{Syl}} E_8 \leftrightarrow A_8$, and scale-invariant Hermite ordering $\gamma^{32}: A_8 (2^{32}) < A_{16} (2^{44}) < \Lambda_{16} (2^{48}) < A_{32} (2^{52})$.
2. **Naive-Wagner Suboptimality Counterexamples & Exact Syndrome Branch-and-Bound:**
   - Three explicit fractional counterexamples where naive 1-/2-flip Wagner syndrome decoding is strictly suboptimal (`643/400` vs. `363/400` on $A_8$ and $A_{32}$; `191/100` vs. `121/100` on $A_{16}$), verified against exhaustive brute force on $A_8$/$A_{16}$ and an exact analytical $\ge 3$-flip global lower bound on $A_{32}$.
3. **Voronoi Zero-Retention Lift & Sharp `f64` (`53`-bit) Exact-Decode Theorem:**
   - Exact discrete-Laplace unit-covolume zero-retention lift `152/256 < 222/256 < 240/256` (`59.375% -> 86.719% -> 93.750%`) plus `INT8`/`INT16`/`INT32`/`WHT` accumulator bounds.
   - Unconditional `f64` (`53`-bit significand) exactness for `128` integer weights $\times$ arbitrary finite `f16` activations holds **iff $W_{\max} \le 64$** (`9,002,801,208,229,888 <= 2^{53}`) and **fails sharply at $W_{\max} = 65$** (`9,072,036,617,912,321 > 2^{53}` odd witness; for `INT8` $W_{\max} = 128$, exact whenever $|a_i| \le 32,752$).
4. **Data-Free Column-Proxy Obstruction:**
   - Exact `16,384x` calibration-Hessian vs. weight-RMS proxy inversion with `measured_live_model_weights = false` honestly flagged.

## Reproduce

```bash
python3 -I -B studies/hamming-construction-a-lattices/hamming_construction_a_audit.py --check studies/hamming-construction-a-lattices/hamming_construction_a_certificate.json --json
python3 -I -B studies/hamming-construction-a-lattices/test_hamming_construction_a_r5.py
```

## File Manifest

| File | Lines | Bytes | SHA-256 |
|---|---:|---:|---|
| `hamming_construction_a_oracle.py` | 1,063 | 45,189 | `591cb773152c68b631b597f3d236e635b3b9eab235bbf3767685defe300de3d0` |
| `hamming_construction_a_audit.py` | 977 | 39,865 | `dd6eb32c9dc906e93616e1e0c71ba5d7825df20c331cd879891efb1614b780ee` |
| `test_hamming_construction_a_r5.py` | 792 | 47,341 | `3d31647c400b180a4d309968ab219e641e833df49a398c0dbabccbb6870473fa` |
| `hamming_construction_a_certificate.json` | 1 | 7,541 | `51e2c9e2ec5528d6a76c2d840b5b4956acd393ec8565c1810ac8f0db3dcd719e` |
| `HAMMING_CONSTRUCTION_A_REPORT.md` | 117 | 15,171 | `5212b2a1760ddf6ea5edd7aa5f35bb2236652be6547e336461bd05d24db80c95` |
