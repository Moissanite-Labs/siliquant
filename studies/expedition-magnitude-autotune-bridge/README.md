# Expedition Study: `magnitudedev/magnitude` Autotuning, Quantization, & Trust-Boundary Bridge (`AUDIT-068R`)

> **Provenance & Verification Status:** Audited under Aletheia (`AUDIT-068R`, `exact+anchor`, novelty class `N1`, `0` external writes). Target snapshot audited strictly read-only at `magnitudedev/magnitude @ d89dfffcbb2f27251e99163934c04770b453339c` (`19/19` verbatim `file:line` anchors verified).

This standalone, zero-float pure Python 3 stdlib study bridges `magnitudedev/magnitude` (Seismic autotuning, IEEE `RoundingCells::width` admission, GGML/Q8 quantization, hybrid multi-bank KV/SSM/DeltaNet state, and `DirectoryStore`/`KernelCache` execution) with:
- **`Moissanite-Labs/siliquant` (`#24`):** Exposes two exact pure-$\mathbb{Q}$ failure modes of RMS-only kernel admission (`MAG-01`, `MAG-02`) and defines `hardened_dual_norm_admission` ($L_\infty$ rounding-cell cap + certified outlier/bulk channel split + `GoverningStep::Subject` lock for `F32` recurrent states) as a safe admission contract for future $E_8^{\mathrm{Had}}$ / Leech24 / outlier-split kernels (`MAG-07`).
- **`Moissanite-Labs/hillclimb#4`:** Transfers Magnitude's anytime-valid uniform-Beta mixture paired sign supermartingale $M_n(s)$ (including tied observations in `rounds = ahead + behind + ties`) into exact rational arithmetic (`ville_mixture_ratio`) while closing its two-sided $2\alpha$ leak, uncorrected $K$-candidate multiplicity (dependence-robust Bonferroni threshold $2K/\alpha$ bounding two-sided FWER $\le \alpha$ without assuming independence; under the extra illustrative assumptions of $K=20$ mutually independent null challengers each calibrated at exact rejection probability $\alpha=1/20$, $1-(19/20)^{20} \approx 64.15\%$), zero-margin churn, and isolated-kernel Goodhart blind spot (`MAG-03`).
- **`Moissanite-Labs/warp-drive` (`#12`/`#13`/`#15`):** Unifies 9 fail-closed Stage 1–4 execution, cache, JIT, weight-lineage, tenant-scoped prefix-cache, and multi-bank flush invariants (`MAG-04`, `MAG-05`, `MAG-06`, `MAG-08`), including `>= 16` non-whitespace auth-token characters on non-loopback binds.

## Files

| File | Role |
|---|---|
| [magnitude_autotune_bridge_audit.py](./magnitude_autotune_bridge_audit.py) | Pure Python 3 stdlib zero-float oracle (`Fraction`/`int` only, `0` floats, `0` bare `assert`s) |
| [test_magnitude_bridge_r5.py](./test_magnitude_bridge_r5.py) | Isolated R5 adversarial, boundary, & `bool`-smuggling unit test suite (`14/14` tests, `26/26` `100.0%` live mutant & `44/44` `BoolOp` kill) |
| [magnitude_bridge_certificate.json](./magnitude_bridge_certificate.json) | Canonical zero-float standalone JSON certificate (`sha256: 662484a43a7953ea124fd3851ee76bac76a1682707a3aecafbad1825818375e6`) |
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

In live-anchored mode against `d89dfffcbb2f27251e99163934c04770b453339c`, all `19/19` verbatim `file:line` anchors match (`anchors_verified: 19, all_anchors_matched: true`, live JSON SHA-256 `f93311340f3d96e56770c7443bf8e42d6516344ae836b20b9a9a151ede4b8b62`, `canonical_sha256: fe0652cd3f3e28351ecfc8470669345291ef30466567b0268e6a8a68c2165c51`).
