# Expedition Study: `loom-fun/unsloth-Inkling-2-Mini-loom` Quantized Speculative Decoding, Lockfile Binding & Compiler-IR Tomography (`AUDIT-074R`)

> **Provenance & Verification Status:** Independently audited and cleared under `Foundation-0/aletheia v1.4.0` (`AUDIT-074R VERIFIED`, `exact+anchor`, novelty class **`N1`**, `0` unresolved defects, `0` red-team bypasses across `17` attacks, `98/98` `100.0%` R5 mutation killed, `0` external writes). Target snapshot audited strictly read-only at `https://huggingface.co/loom-fun/unsloth-Inkling-2-Mini-loom` @ commit `277f099b5003b725e099d48caa1a2c0db1f9b650` (`14/14` lightweight metadata/config/`.loom` files verified byte-for-byte; multi-GB GGUF weights and `12.24 MB` LFS `tokenizer.json` excluded).

This standalone, zero-float pure Python 3 stdlib (`int` / `fractions.Fraction`) study audits AMD Loom's `Inkling-2-Mini (UD-Q4_K_XL)` + `DFlash`/`DSpark` speculative decoding package and bridges directly to:
- **`Moissanite-Labs/siliquant` (`#24`/`#26`, `AUDIT-068R`):** Exact pure-$\mathbb{Q}$ analysis of `head_weights.loom` / `@g_prep_q8_0_q4_0` (`llm.loom:34454-34521`) load-time `Q8_0 -> Q4_0` requantization of `output.weight` (`14,404,176` blocks, `489,741,984 B -> 259,275,168 B`, compression **`9/17`**, bandwidth saved **`8/17 = 230,466,816 B`**), proving an exact **`1/8 = 12.5%`** asymmetric clipping error on `-M` and an exact rational argmax-flip witness (`z_A = 8 > z_B = 15/2` flipping to `z_A_q4 = 7 < z_B_q4 = 15/2`) that collapses greedy draft acceptance from **`1/1 -> 0/1`**.
- **`Moissanite-Labs/scry#4` (`AUDIT-032`):** Exact pure-$\mathbb{Q}$ tail-extinction proof (`TV = 35/64 = 54.69%`) for `llm.loom:17998-18078`'s silent `top_k = 64` hard cap on `vocab = 200,058` (`32/100029`) plus live compiler-IR bug **`LOOM-IR-01`** (`control.loom:314,348-367` bit-30 `"logits": true` flag unmasked in `@g_head_select`, silently corrupting EVERY `top_k in 1..63` to `64`) and **`LOOM-IR-02`** (`@g_logits_export` omitted from `@llm_verify`).
- **`Moissanite-Labs/weft-loom#42` (`AUDIT-023`):** Exact `9/9` metric audit (catching IEEE-754 half-tie diff `-1/20` on `prefill_2k`: `24287/20 = 1214.35 -> 1214.3`), apples-to-oranges decode decomposition (`427/136 = 3.1397x` drafted headline vs `53375/27234 = 1.9599x` per verify pass vs `423/190 = 2.2263x` drafting-off `streams4`), and exact MoE coupon-collector (`E = 128, k = 8`) occupancy theorem (`E[U(1..8)] = 8, 31/2, 14911/512, 1732076671/33554432`; break-even acceptance `40.02%` at 1 stream and `56.43%` at 4 streams before drafter cost; independent-routing `streams4` ceiling `185.24 tok/s`, raising Loom's true bandwidth efficiency from `22%` to **`45.67%`**).
- **`Moissanite-Labs/warp-drive` (`#12`/`#13`/`#15`, `AUDIT-064`) & `siliconcordat/loom#10`:** Exact looped-core (`n_phys = 24, n_passes = 3 -> 48` logical = `40` SWA + `8` full-attn `16` KV planes) single-byte memory geometry (`174,227,456 B` recurrent slot, `163,577,856 B` capture buffer, `8,404,992 B` DSpark carry), speculative shortconv replay isolation + **`LOOM-IR-05`** (`row >= 16` silent replay drop), strict `DSpark` Markov-RNN contraction, and fail-closed Ring-0 lockfile binding (`verify_loom_lockfile_bundle` closing `LOOM-LOCK-01`/`LOOM-PROV-01`, `16` defects on published bundle).

## Files

| File | SHA-256 / Role |
|---|---|
| [inkling_loom_tomography.py](inkling_loom_tomography.py) | `0e5395e67141edcd1439adb715b2bdea8513d76c94594f388094db1e10b47fbc` — Pure Python 3 stdlib zero-float tomography engine, Ring-0 lockfile governor, speculative coupling oracle, `Q8_0 -> Q4_0` block quantizer, and MoE roofline/memory-geometry verifier |
| [test_inkling_loom_r5.py](test_inkling_loom_r5.py) | `4273e50980a464fbb2ebb03ace6f0bebfee322e0e86cc9c97411f895fb2632f5` — Isolated R1–R5 adversarial, boundary, & synthetic-mirror test suite (`5/5` test groups, `98/98` `100.0%` R5 mutation kill) |
| [inkling_loom_certificate.json](inkling_loom_certificate.json) | `60ce03085ababde8fbce9b4886f02ed1cfb288d461b28eae6618a4b5d633497d` — Sealed `AUDIT-074R` live-anchored zero-float JSON certificate (`canonical_sha256 = 69e92b234d0881c97c0a45eae3e9815bcc9fad47cb6e3ca92e3b0129de17cae6`) |
| [INKLING_LOOM_AUDIT_REPORT.md](INKLING_LOOM_AUDIT_REPORT.md) | Complete verbatim `file:line` report (`LOOM-LOCK-01..02`, `LOOM-PROV-01`, `LOOM-IR-01..05`, `Theorem S1–S2`, `Q8_0 -> Q4_0`, MoE roofline, memory geometry), Law-2 steelman, Law-4 `N0` quarantine, and 5-repo bridge |

## Quickstart (100% Standalone — Zero External Clones, Weights, or Packages Required)

Run from inside `studies/expedition-inkling-loom-tomography/`:

```bash
# 1. Verify 4/4 isolated -I R4 bit-identical standalone JSON output
#    Expected SHA-256 across all 4 modes:
#      fb6e32e4165719bdbc85d3dfa55e3845f12b288dff877094055f4a1678e4f232
#    (canonical_sha256 = 958a985d9c5feda0...; sealed live-anchored cert is 60ce0308...)
for flags in "-I -B" "-I -O -B" "-I -B -W error" "-I -O -B -W error"; do
  PYTHONDONTWRITEBYTECODE=1 python3 $flags inkling_loom_tomography.py --json | sha256sum
done

# 2. Run the isolated R5 adversarial, boundary & synthetic-mirror test suite
PYTHONDONTWRITEBYTECODE=1 python3 -I -O -B -W error test_inkling_loom_r5.py -v
```

With [`Foundation-0/aletheia`](https://github.com/Foundation-0/aletheia) v1.4.0 installed or on `PYTHONPATH`:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -W error -m aletheia audit-check . \
  --strict-ancestors --scripts inkling_loom_tomography.py test_inkling_loom_r5.py --json
PYTHONDONTWRITEBYTECODE=1 python3 -B -W error -m aletheia mutate inkling_loom_tomography.py \
  --work-root . --add-cwd --test-cmd "python3 -I -O -B -W error test_inkling_loom_r5.py" --json
```

### Optional Live-Anchor Replay (Against Local Read-Only Mirror of `277f099b`)

If you have downloaded the `14` lightweight metadata/config/`.loom` files from `loom-fun/unsloth-Inkling-2-Mini-loom @ 277f099b5003b725e099d48caa1a2c0db1f9b650`:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -I -O -B -W error inkling_loom_tomography.py \
  --repo-dir /path/to/inkling-loom --json | diff -u inkling_loom_certificate.json -
```
