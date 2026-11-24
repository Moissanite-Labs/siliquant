# Zero-Unpack `WHT_8` Shell-Radius Frobenius & `WHT_8`-Bypassed Bareiss PSD Stage-2 Certifier

- **Oracle Class:** `exact` (pure Python 3 stdlib $\mathbb{Z}/\mathbb{Q}$; `0` AST floats, `0` `float()` calls, `0` bare `assert`s, `0` `.isdigit()`/`.isnumeric()`/`.isdecimal()`, `0` mathematical RNG; atomic `dir_fd` Zone-B I/O hardening)
- **Novelty Class:** `N1` (exact algebraic & cryptographic Stage-2 bridge uniting `Block-H2 == WHT_8` isometry and `E_8^4` shell-convolution coding with byte-derived exact Frobenius and Bareiss PSD spectral certification)
- **Sealed Certificate:** `wht8_frobenius_bridge_certificate.json` (`sha256: d30b2e5ce907d2cd80ae364e98af65d05627078741b56e3aed2dab9e9aca74c5`, `payload_sha256: 280c038cd55c5db44dc798aa803a702fb7f7ec0f2d442c900e20bfde1ee81a1c`, `certificate_sha256: 4baf940e96e0fcc983bec5df0bfb117b4cc54884f4fcb2881e52717fb5efed82`)

---

## 1. Problem Statement

Three quantization and adapter-certification mechanisms share a single exact algebraic invariant:

1. **Unkeyed self-declared norm trust gap:** An unkeyed checksum over self-declared integer norms can be bypassed by relabeling an inflated adapter (`||BA||_F^2 = 6 > 5`) from `radix_scale_exp2 = 32` to `48`, dividing the declared norm by $2^{16}$ (`6 -> 3/32768`).
2. **Exact byte-derived Gram & Bareiss PSD certification:** Computing exact integer Gram matrices $G_B = B^\top B$, $G_A = A A^\top$, exact $\|BA\|_F^2 = \operatorname{Tr}(G_B G_A)$, exact unit-scaling (`alpha == rank`), and exact fraction-free Bareiss PSD spectral verification $\tau G_A - G_A G_B G_A \succeq 0$ directly from raw adapter bytes closes self-declared norm forgery.
3. **Packed `E_8^Had` (`Block-H2 == WHT_8`) shell-ordered encoding:** Packed `E_8^Had` blocks store an exact dyadic scale $s_b = \sigma_b \cdot 2^{-q_s}$ and a shell-ordered enumerative index $k_b$, reconstructing $w_b = s_b U_8^\top v_b$ with $U_8 = H_8 / \sqrt{8}$ and $v_b \in E_8$.

**This study proves and implements a two-tier pure-$\mathbb{Z}/\mathbb{Q}$ Stage-2 certifier (`SQE8HAD1`) that certifies exact Frobenius and spectral bounds directly from packed `E_8^Had` bitstreams with ZERO `WHT_8` butterflies, ZERO floating-point operations, and (in Tier 1) ZERO coordinate unranking.**

---

## 2. Exact Theorems & Verified Rational Witnesses

### Theorem 1 — Exact `WHT_8` (`Block-H2`) Isometry & Shell-Radius Invariance
- Verified entrywise in $\mathbb{Z}$ across all $64$ entries: $H_8 = H_8^\top$ and $H_8^\top H_8 = 8 I_8$.
- Verified across all $8$ basis vectors and all $240$ $E_8$ roots: the 3-stage 24-op integer butterfly `apply_h8_butterfly(z)` equals $H_8 z$ and satisfies $\|H_8^\top z\|_2^2 = 8 \|z\|_2^2$ identically.
- Therefore, for every $v_b \in E_8$ on shell $m_b$ ($\|v_b\|_2^2 = 2 m_b$):
  $$\|w_b\|_2^2 = s_b^2 \|U_8^\top v_b\|_2^2 = s_b^2 \|v_b\|_2^2 = 2 s_b^2 m_b, \qquad \langle U_8^\top u, U_8^\top v \rangle = \langle u, v \rangle \in \mathbb{Z}.$$

### Theorem 2 — Exact Eisenstein Shell Tables & Zero-Unpack Complexity (`E_8` & `E_8^4`)
- Exact shell-ordered DP over doubled coordinates $z = 2v \in 2E_8$ matches the Eisenstein $E_4$ theta series $r_{E_8}(m) = 240 \sigma_3(m)$ identically for $m = 0..8$:
  `[1, 240, 2160, 6720, 17520, 30240, 60480, 82560, 140400]`.
- Exact convolution $r_{E_8^2}(m)$ matches the modular-form identity $E_4(q)^2 = E_8(q) = 1 + 480 \sum_{m \ge 1} \sigma_7(m) q^m$ identically for $m = 1..12$.
- For 32-weight `E_8^4` super-blocks, ONE $O(\log M_4)$ binary search on the cumulative table $T_{E_8^4}$ reads the exact total squared norm $2m_g = 2(m_1+m_2+m_3+m_4)$ of all 32 weights at once without unranking any coordinate or executing `WHT_8`:

| Packed `E_8^4` Format (`32` weights/block + `8`-bit per-superblock scale) | Exact Max Shell $M_4$ | Exact Codebook Size $T_{E_8^4}[M_4]$ | Binary-Search Cmp | Zero-Unpack Ops | Full Unpack + `WHT_8` + Frob Ops | Exact Op Ratio |
|---|---:|---:|---:|---:|---:|---:|
| `88`-bit index (`(88+8)/32 = 3.000` bpw) | `48` | `263,197,133,594,409,826,368,003,841` | `<= 6` | `9` | `191` | **`191/9` (`21.22x`)** |
| `136`-bit index (`(136+8)/32 = 4.500` bpw) | `391` | `85,889,107,931,845,736,051,215,402,649,708,868,585,601` | `<= 9` | `12` | `191` | **`191/12` (`15.92x`)** |

### Theorems 3 & 4 — Two-Tier Zero-Unpack & `WHT_8`-Bypassed Stage-2 Verification Parity
1. **Tier 1 (`TIER1_ZERO_UNPACK_EXACT_RANK1` / `TIER1_ZERO_UNPACK_FAST_ACCEPT`):** Zero coordinate unranks (`coordinates_unranked = 0`), zero `WHT_8` butterflies (`wht8_butterflies_run = 0`). Computes exact column/row squared norms $\|b_k\|_2^2$, $\|a_k\|_2^2$ and exact rational Cauchy-Schwarz upper bound
   $$U_{\text{zero-unpack}}(B, A) = \min\!\left(\|B\|_F^2 \|A\|_F^2,\; r \sum_{k=1}^r \|b_k\|_2^2 \|a_k\|_2^2\right).$$
   Exact (sound and complete) for all rank-1 adapters and all dense matrices; sound fast-accept for rank $r \ge 2$ whenever $U_{\text{zero-unpack}} \le \min(\tau_{\text{frob}}, \tau_{\text{spec}})$ ($\tau_{\text{spec}} \le \tau_{\text{frob}}$ enforced up front).
2. **Tier 2 (`TIER2_WHT8_BYPASSED_GRAM_BAREISS`):** When $r \ge 2$ and $U_{\text{zero-unpack}}$ exceeds the tight spectral/Frobenius threshold, unranks doubled $E_8$ coordinates directly in the pre-`WHT_8` domain (`wht8_butterflies_run = 0`, since $\langle U_8^\top u, U_8^\top v \rangle = \langle u, v \rangle$), computes exact integer $\tilde{G}_B, \tilde{G}_A$, checks exact $\|BA\|_F^2 = \operatorname{Tr}(G_B G_A)$, and decides $\sigma_{\max}(BA)^2 \le \tau$ via exact fraction-free symmetric Bareiss integer PSD elimination on $(S_{\text{spec}} 2^{4 q_s}) \tilde{G}_A - 2^E (\tilde{G}_A \tilde{G}_B \tilde{G}_A) \succeq 0$.
3. **Brute-Force Full-`WHT_8` Parity Witnesses:**
   - Honest single-$E_8$ rank-1 adapter: Tier 1 (`0` unranks, `0` `WHT_8`) gives exact `3/2`, matching full-unpack + `WHT_8` dense reference `3/2` bit-for-bit.
   - `E_8^4` 32-weight super-block adapter: Tier 1 (`2` shell lookups, `0` unranks, `0` `WHT_8`) gives exact `3`, matching full binary-tree unrank + `WHT_8` dense reference `3` bit-for-bit.
   - Rank-2 non-orthogonal adapter ($G_B = \begin{pmatrix}2&1\\1&2\end{pmatrix}$, $G_A = \frac{1}{2}I_2$, exact squared singular values `3/2` and `1/2`, $\|BA\|_F^2 = 2$, $U_{\text{zero-unpack}} = 4$): Tier 2 (`0` `WHT_8` butterflies) matches full-unpack + `WHT_8` dense reference `2` bit-for-bit, verifies exact Gram-diagonal identity against Tier 1, admits at exact $\tau = 3/2$, and fast-accepts via Tier 1 (`0` unranks) when declared bound is `4`.

### Theorem 5 — Five Live-Proved Attack Vectors (`6` Test Variants)

| Attack Vector | Forged / Relabelled Claim | Exact True Value | Fail-Closed Rejection Code |
|---|---|---|---|
| `radix_scale_exp2` relabel (`32 -> 48`) | `3/32768 <= 5` | `6 > 5` | **`FROBENIUS_EXCEEDS_CERTIFIED`** |
| `b-times-8` (`64x` scale forgery, stale SHA) | `3/2` | `96` | **`ADAPTER_SHA_MISMATCH`** |
| `b-times-8` (`64x` scale forgery, re-signed SHA) | `3/2` | `96` | **`FROBENIUS_EXCEEDS_CERTIFIED`** |
| `alpha-32` metadata inflation (`alpha/r = 8`) | `alpha = 32, r = 4` | `64x` update energy | **`NON_UNIT_ALPHA_SCALE`** |
| Shell-index bump (`m = 1 -> m = 3`, re-signed SHA) | `3/2` | `5/2` | **`FROBENIUS_EXCEEDS_CERTIFIED`** |
| Rank-2 1-ULP Bareiss spectral understatement | `6442450943/4294967296` | `3/2` | **`SPECTRAL_EXCEEDS_CERTIFIED`** |

---

## 3. Scope Limits, Runtime vs. Analytical Caps, & Trust Boundaries

1. **Exact Packed Semantics vs. Post-`WHT_8` `float16` Rounding:** The zero-unpack identity $\|s_b U_8^\top v_b\|_2^2 = 2 s_b^2 m_b$ holds identically for exact packed `SQE8HAD1` semantics ($w_b = s_b U_8^\top v_b$ with exact dyadic scale $s_b = \sigma_b 2^{-q_s}$). If a downstream runtime casts intermediate values to `float16` before or after `WHT_8`, finite-precision rounding perturbs norms at the $\pm \frac{1}{2}\text{ulp}_{16}$ scale.
2. **`SQE8HAD1` Runtime Caps vs. Analytical `E_8^4` Table Scope:** The standalone `SQE8HAD1` reference container enforces bounded-memory runtime caps (`MAX_CERT_SHELL = 32`, `index_bytes <= 16` = 128-bit unsigned index per block, `MAX_JSON_BYTES = 2_000_000` bytes, `MAX_INT_BITS = 2048`), whereas Theorem 2's analytical `E_8^4` shell-convolution witness tables cover the full `88`-bit index (`(88+8)/32 = 3.000` bpw with an `8`-bit per-superblock scale, `M_4 = 48`, 11-byte index) and `136`-bit index (`(136+8)/32 = 4.500` bpw with an `8`-bit per-superblock scale, `M_4 = 391`, 17-byte index) formats. Wiring the `136`-bit `E_8^4` codebook into a production loader requires **both** raising the packed-index byte cap from `16` to `>= 17` bytes **and** raising `MAX_CERT_SHELL` from `32` to `391` (plus precomputing the `M_4 = 391` cumulative shell table). Likewise, `verify_packed_e8had_stage2` checks exact byte-derived Frobenius and Bareiss PSD spectral bounds against caller-supplied `(declared_frob_scaled, declared_spec_scaled, radix_scale_exp2)` policy arguments.
3. **Stand-Alone & Training-Subspace Scope:** `SQE8HAD1` is a self-contained Stage-2 reference container; wiring it into production loaders requires a separate integration PR. Enforcing unit $\alpha/r = 1$ prevents metadata amplification of subspace leakage metrics, while verifying training-covariance or safety-subspace projections themselves requires those matrices (not present in the adapter bitstream alone).
4. **Honest `--check` Deterministic-Replay Scope:** `wht8_frobenius_bridge_audit.py --check` performs full pure-$\mathbb{Z}/\mathbb{Q}$ deterministic replay (`generate_certificate()`) plus schema-safe structural validation (`_validate_payload_structure`) and secondary algebraic cross-checking (`secondary_audit_payload`) across all 5 theorems and 5 attack closures (enforced under both `full_replay=True` and `full_replay=False`). Because `wht8_frobenius_bridge_audit.py` imports `generate_certificate` and strict parsers from `wht8_frobenius_bridge_oracle.py`, `--check` provides deterministic replay and secondary algebraic cross-checking rather than a separate zero-shared-code implementation.
5. **Zone-B I/O & Same-UID POSIX Boundary (`write_regular_file_bytes` / `read_regular_bytes_capped`):**
   - Reads use atomic `dir_fd` `O_RDONLY|O_NOFOLLOW|O_NONBLOCK` + `os.fstat(fd)` `S_ISREG` + `65,536`-byte streaming `os.read` capped at `max_bytes` (`<= 64 MiB`, closing `st_size == 0` procfs bypasses such as `/proc/kallsyms`), with `OSError` wrapped into `Wht8BridgeError` (`FILE_READ_ERROR`).
   - Writes use atomic `dir_fd` traversal (`O_RDONLY|O_DIRECTORY|O_NOFOLLOW`), `0o600` exclusive temp creation (`O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_NONBLOCK`) with `threading.Lock()` + monotonic sequence + `secrets.token_hex(8)` + bounded `32`-attempt `EEXIST` retry, `os.fsync(tmp_fd)`, pre-replace `os.fstat(tmp_fd).st_nlink == 1`, target `st_nlink <= 1` check, `os.replace(..., src_dir_fd=dfd, dst_dir_fd=dfd)`, and directory `os.fsync(dfd)`.
   - Under the standard POSIX security model, a concurrent *same-UID* process with write permission on the parent directory retains kernel authority to rename or hardlink paths between `lstat`/`fstat` and `os.replace`; isolating against malicious same-UID writers requires OS-level directory permissions or separate UIDs.

---

## 4. Reproducibility & Committed File Manifest

### Verification Commands (Run From This Directory)
- Standalone exact certificate: `python3 -I -B wht8_frobenius_bridge_oracle.py --standalone --json`
- Fail-closed certificate auditor: `python3 -I -B wht8_frobenius_bridge_audit.py --check wht8_frobenius_bridge_certificate.json`
- Atomic `0o600` certificate writer: `python3 -I -B wht8_frobenius_bridge_audit.py --write wht8_frobenius_bridge_certificate.json`
- Full adversarial, concurrency, kernel-race & boundary test suite (passes across `-I -B`, `-I -O -B`, `-I -B -W error`, `-I -O -B -W error`, and `PYTHONHASHSEED=42 python3 -B`):
  `python3 -I -B -W error test_wht8_frobenius_bridge_r5.py`

### Committed Non-Report File Manifest (`wc -lc` & SHA-256)

| File | Lines | Bytes | SHA-256 |
|---|---:|---:|---|
| `wht8_frobenius_bridge_oracle.py` | `1363` | `56692` | `eefab8eccc1c67f5840c00440db6d0fe8a748d4affcb1922e3654734a89c90ce` |
| `wht8_frobenius_bridge_audit.py` | `853` | `32825` | `df7622df3b6ec9ea16f4994841acc7ba7f497cc28839ee3141b3f6e3b82e79d6` |
| `test_wht8_frobenius_bridge_r5.py` | `1101` | `61202` | `e3ed588c30c8daedc6c3ea49f0c94b4f185c8fcf0bfb53f7434b07183f233963` |
| `wht8_frobenius_bridge_certificate.json` | `1` | `3654` | `d30b2e5ce907d2cd80ae364e98af65d05627078741b56e3aed2dab9e9aca74c5` |
