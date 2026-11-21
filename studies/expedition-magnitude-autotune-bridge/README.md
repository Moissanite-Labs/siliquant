# Expedition Study: `magnitudedev/magnitude` Autotuning, Quantization, & Trust-Boundary Bridge (`AUDIT-068R`)

> **Provenance & Verification Status:** Independently audited and cleared under Aletheia v1.4.0 (`AUDIT-068R VERIFIED`, `exact+anchor`, novelty class `N1`, `0` defects, `0` red-team bypasses, `0` external writes). Target snapshot audited strictly read-only at `magnitudedev/magnitude @ d89dfffcbb2f27251e99163934c04770b453339c` (`19/19` verbatim `file:line` anchors verified).

This standalone, zero-float pure Python 3 stdlib study bridges `magnitudedev/magnitude` (Seismic autotuning, IEEE `RoundingCells::width` admission, GGML/Q8 quantization, hybrid multi-bank KV/SSM/DeltaNet state, and `DirectoryStore`/`KernelCache` execution) with:
- **`Moissanite-Labs/siliquant` (`#24`):** Exposes two exact pure-$\mathbb{Q}$ failure modes of RMS-only kernel admission (`MAG-01`, `MAG-02`) and defines `hardened_dual_norm_admission` ($L_\infty$ rounding-cell cap + certified outlier/bulk channel split + `GoverningStep::Subject` lock for `F32` recurrent states) as a safe admission contract for future $E_8^{\mathrm{Had}}$ / Leech24 / outlier-split kernels (`MAG-07`).
- **`Moissanite-Labs/hillclimb#4`:** Transfers Magnitude's anytime-valid uniform-Beta mixture paired sign supermartingale $M_n(s)$ into exact rational arithmetic (`ville_mixture_ratio`) while closing its two-sided $2\alpha$ leak, uncorrected $K$-candidate multiplicity (`FWER > 64.15%` at $K=20, \alpha=1/20$), zero-margin churn, and isolated-kernel Goodhart blind spot (`MAG-03`).
- **`Moissanite-Labs/warp-drive` (`#12`/`#13`/`#15`):** Unifies 9 fail-closed Stage 1–4 execution, cache, JIT, weight-lineage, tenant-scoped prefix-cache, and multi-bank flush invariants (`MAG-04`, `MAG-05`, `MAG-06`, `MAG-08`).

## Files

| File | Role |
|---|---|
| [magnitude_autotune_bridge_audit.py](./magnitude_autotune_bridge_audit.py) | Pure Python 3 stdlib zero-float oracle (`Fraction`/`int` only, `0` floats, `0` bare `assert`s) |
| [test_magnitude_bridge_r5.py](./test_magnitude_bridge_r5.py) | Isolated R5 adversarial, boundary, & `bool`-smuggling unit test suite (`14/14` tests, `19/19` `100.0%` R5 mutant kill) |
| [magnitude_bridge_certificate.json](./magnitude_bridge_certificate.json) | Canonical zero-float standalone JSON certificate (`sha256: 9a371c9401f2c7239d99a6155501aa19f8daae4f457e3367d60911859d04b051`) |
| [MAGNITUDE_AUTOTUNE_AUDIT_REPORT.md](./MAGNITUDE_AUTOTUNE_AUDIT_REPORT.md) | Complete verbatim `file:line` report (`MAG-01`..`MAG-08`), Law-4 `N0` quarantine, Law-2 steelman, & 3-way synthesis |

## Quickstart (100% Standalone — Zero External Clones or Packages Required)

Run from inside `studies/expedition-magnitude-autotune-bridge/`:

```bash
# 1. Verify standalone JSON output matches committed certificate bit-for-bit
python3 -I -B magnitude_autotune_bridge_audit.py --standalone --json | diff -u magnitude_bridge_certificate.json -

# 2. Verify 4/4 isolated -I R4 bit-identical reproducibility
for flags in "-I -B" "-I -O -B" "-I -B -W error" "-I -O -B -W error"; do
  python3 $flags magnitude_autotune_bridge_audit.py --standalone --json | sha256sum
done

# 3. Run the isolated R5 adversarial & boundary test suite
python3 -I -B -W error test_magnitude_bridge_r5.py -v
```

### Optional Live-Anchor Replay (Against Local Read-Only `magnitudedev/magnitude` Checkout)

If you have a local read-only checkout of `magnitudedev/magnitude` pinned to commit `d89dfffcbb2f27251e99163934c04770b453339c`:

```bash
python3 -I -B magnitude_autotune_bridge_audit.py --repo /path/to/magnitude --json
```

In live-anchored mode against `d89dfffcbb2f27251e99163934c04770b453339c`, all `19/19` verbatim `file:line` anchors match (`anchors_verified: 19, all_anchors_matched: true`, live JSON SHA-256 `22ee71b24c0a64790a931022532da6802c4d4460433ce62a684180c8704c3f30`, `canonical_sha256: 577e5645b3c35c6ae80dba05724fcdfec52e2ecce78c47fb038260293cf63317`).
