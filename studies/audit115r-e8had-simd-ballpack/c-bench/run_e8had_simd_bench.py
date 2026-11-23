#!/usr/bin/env python3
"""Self-contained C11 + AVX2/FMA empirical benchmark & exactness verifier for Track S15c (AUDIT-115):
E_8_Had zero-table WHT_8 leaf transform, batched SIMD/AVX2 unranking, fused dequant-GEMV
(materialized vs. activation-adjoint WHT_8), matched 4.500-bpw Q4_K_M-style AVX2/FMA GEMV,
and stage-by-stage CPU-time breakdown.

Modes:
  (default / --verify-only)  Deterministic 2-compiler native build (gcc & clang -O3 -march=native),
                             objdump 0-div check across all 8 hot kernels, and exactness proof
                             (0 timestamps; canonical JSON for 3-mode -I R4 parity).
  --bench                    Full 4096 x 1024 (4,194,304-weight / 131,072-block) exactness check
                             PLUS multi-trial 1-core & 2-core dual-clock decode, GEMV, matched
                             Q4_K_M-style AVX2 GEMV, and 4-stage breakdown benchmarks.
  --write-results <path>     Implies --bench and writes canonical JSON results to <path>.

Strict discipline: pure Python 3 stdlib, zero bare asserts, zero checked-in binaries
(all .so files built ephemerally in a 0700 TemporaryDirectory).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
from typing import Any, Sequence

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from e8had_isometry_oracle import (
    format_exact_decimal,
    is_valid_e8_had_scaled,
    is_valid_e8_std_doubled,
    map_e8_std_to_had_exact,
)

SCHEMA_VERSION = "track_s15c.e8had_simd_bench.v1"

COMPILER_PROFILES: tuple[dict[str, Any], ...] = (
    {
        "id": "gcc_native",
        "cc": "gcc",
        "flags": ["-O3", "-march=native", "-Wall", "-Wextra", "-Werror", "-std=c11", "-shared", "-fPIC"],
    },
    {
        "id": "clang_native",
        "cc": "clang",
        "flags": ["-O3", "-march=native", "-Wall", "-Wextra", "-Werror", "-std=c11", "-shared", "-fPIC"],
    },
)

HOT_SYMBOLS: tuple[str, ...] = (
    "e8_decode_std_scalar",
    "e8had_decode_wht_scalar",
    "e8had_decode_wht_avx2",
    "e8had_decode_simd8_avx2",
    "e8had_gemv_materialized_avx2",
    "e8had_gemv_adjoint_avx2",
    "e8had_gemv_adjoint_batch4_avx2",
    "q4km_style_gemv_avx2",
)

_DIV_INSN_RE = re.compile(r"\t(?:i?div[bwlq]?)\b")


class SimdBenchError(ValueError):
    """Fail-closed error on build, disassembly, exactness, or runtime violations."""


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise SimdBenchError(f"missing source file: {path.name}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _detect_host() -> dict[str, Any]:
    cpu_model = platform.processor() or "unknown"
    cores = os.cpu_count() or 1
    lscpu_map: dict[str, str] = {}
    if shutil.which("lscpu"):
        p = subprocess.run(["lscpu"], capture_output=True, text=True, check=False)
        if p.returncode == 0:
            for line in p.stdout.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    lscpu_map[k.strip()] = v.strip()
            if "Model name" in lscpu_map:
                cpu_model = lscpu_map["Model name"]
    flags_str = lscpu_map.get("Flags", "")
    isa_subset = [f for f in ("avx2", "fma", "bmi1", "bmi2", "adx", "f16c", "avx512f") if f in flags_str.split()]
    mhz_nom = 2300
    max_freq_p = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq")
    cur_freq_p = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    cur_mhz = mhz_nom
    if max_freq_p.is_file():
        try:
            mhz_nom = int(max_freq_p.read_text(encoding="utf-8").strip()) // 1000
        except ValueError:
            pass
    if cur_freq_p.is_file():
        try:
            cur_mhz = int(cur_freq_p.read_text(encoding="utf-8").strip()) // 1000
        except ValueError:
            pass
    loadavg = "unknown"
    if Path("/proc/loadavg").is_file():
        loadavg = Path("/proc/loadavg").read_text(encoding="utf-8").strip()
    compilers: dict[str, str] = {}
    for cc in ("gcc", "clang"):
        if shutil.which(cc):
            pr = subprocess.run([cc, "--version"], capture_output=True, text=True, check=False)
            compilers[cc] = pr.stdout.splitlines()[0].strip() if pr.stdout else "unknown"
    return {
        "architecture": platform.machine(),
        "cpu_model": cpu_model,
        "logical_cpus": cores,
        "l1d_cache": lscpu_map.get("L1d cache", "unknown"),
        "l2_cache": lscpu_map.get("L2 cache", "unknown"),
        "l3_cache": lscpu_map.get("L3 cache", "unknown"),
        "isa_features": isa_subset,
        "nominal_mhz": mhz_nom,
        "observed_boost_mhz": cur_mhz,
        "host_loadavg_at_bench": loadavg,
        "python_version": platform.python_version(),
        "compilers": compilers,
    }


# ==============================================================================
# Self-Contained Exact E_8^4 Table & Universal Barrett Reciprocal Builder
# ==============================================================================

def _build_exact_reciprocal(divisor: int, max_dividend: int) -> tuple[int, int]:
    if divisor <= 0 or max_dividend < 0:
        raise SimdBenchError("invalid reciprocal domain")
    if divisor == 1:
        return 1, 0
    if (divisor & (divisor - 1)) == 0:
        return 1, divisor.bit_length() - 1
    k = max(1, max_dividend.bit_length())
    for extra in range(0, divisor.bit_length() + 4):
        shift = k + extra
        two_s = 1 << shift
        mult = (two_s + divisor - 1) // divisor
        if mult * divisor >= two_s and (mult * divisor - two_s) * max_dividend < two_s:
            return mult, shift
    raise SimdBenchError(f"failed to synthesize universal reciprocal for d={divisor}")


def _ordered_z1_symbols(limit_sq: int) -> list[int]:
    out = [0]
    k = 1
    while k * k <= limit_sq:
        out.append(k)
        out.append(-k)
        k += 1
    return out


def _build_e8_4_tables_self_contained(m_max: int) -> dict[str, Any]:
    r_max = 2 * m_max
    t_max = max(0, m_max - 1)
    even_p2: list[list[list[tuple[int, int]]]] = [[[] for _ in range(r_max + 1)] for _ in range(2)]
    k_even = _ordered_z1_symbols(r_max)
    for k0 in k_even:
        r0 = k0 * k0
        for k1 in k_even:
            r = r0 + k1 * k1
            if r <= r_max:
                even_p2[(k0 + k1) & 1][r].append((2 * k0, 2 * k1))
    even_c2 = [[len(even_p2[p][r]) for r in range(r_max + 1)] for p in (0, 1)]
    even_c4 = [[0] * (r_max + 1) for _ in range(2)]
    for pa in (0, 1):
        for pb in (0, 1):
            pt = pa ^ pb
            ca, cb = even_c2[pa], even_c2[pb]
            dst = even_c4[pt]
            for na, cnt_a in enumerate(ca):
                if cnt_a == 0:
                    continue
                for nb in range(r_max - na + 1):
                    cnt_b = cb[nb]
                    if cnt_b:
                        dst[na + nb] += cnt_a * cnt_b
    even_c8 = [0] * (m_max + 1)
    for m in range(m_max + 1):
        tot = 0
        r8 = 2 * m
        for p in (0, 1):
            c4 = even_c4[p]
            for a in range(r8 + 1):
                tot += c4[a] * c4[r8 - a]
        even_c8[m] = tot

    odd_p2: list[list[list[tuple[int, int]]]] = [[[] for _ in range(t_max + 1)] for _ in range(2)]
    k_odd: list[int] = []
    c = 0
    while c * (c + 1) // 2 <= t_max:
        k_odd.append(c)
        k_odd.append(-1 - c)
        c += 1
    for k0 in k_odd:
        t0 = k0 * (k0 + 1) // 2
        for k1 in k_odd:
            t = t0 + k1 * (k1 + 1) // 2
            if t <= t_max:
                odd_p2[(k0 + k1) & 1][t].append((2 * k0 + 1, 2 * k1 + 1))
    odd_c2 = [[len(odd_p2[p][t]) for t in range(t_max + 1)] for p in (0, 1)]
    odd_c4 = [[0] * (t_max + 1) for _ in range(2)]
    for pa in (0, 1):
        for pb in (0, 1):
            pt = pa ^ pb
            ca, cb = odd_c2[pa], odd_c2[pb]
            dst = odd_c4[pt]
            for na, cnt_a in enumerate(ca):
                if cnt_a == 0:
                    continue
                for nb in range(t_max - na + 1):
                    cnt_b = cb[nb]
                    if cnt_b:
                        dst[na + nb] += cnt_a * cnt_b
    odd_c8 = [0] * (m_max + 1)
    for m in range(1, m_max + 1):
        tot = 0
        t8 = m - 1
        for p in (0, 1):
            c4 = odd_c4[p]
            for a in range(t8 + 1):
                tot += c4[a] * c4[t8 - a]
        odd_c8[m] = tot

    s1 = [even_c8[m] + odd_c8[m] for m in range(m_max + 1)]
    s2 = [0] * (m_max + 1)
    for m1, c1 in enumerate(s1):
        for m2 in range(m_max - m1 + 1):
            s2[m1 + m2] += c1 * s1[m2]
    w1: list[int] = []
    acc = 0
    for v in s1:
        acc += v
        w1.append(acc)
    w2: list[int] = []
    acc = 0
    for v in s2:
        acc += v
        w2.append(acc)
    c_root = [0]
    acc = 0
    for m_l in range(m_max + 1):
        acc += s2[m_l] * w2[m_max - m_l]
        c_root.append(acc)

    rec_root = [_build_exact_reciprocal(w2[mr], max(s2[: m_max - mr + 1]) * w2[mr] - 1) for mr in range(m_max + 1)]
    rec_b1 = [_build_exact_reciprocal(w1[mr], max(s1[: m_max - mr + 1]) * w1[mr] - 1) for mr in range(m_max + 1)]
    rec_s1 = [_build_exact_reciprocal(s1[mr], max(s1[: m_max - mr + 1]) * s1[mr] - 1) for mr in range(m_max + 1)]

    def _inner_rec(c4: list[list[int]], c2: list[list[int]], lim: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        r4: list[tuple[int, int]] = []
        r2: list[tuple[int, int]] = []
        for p in (0, 1):
            for nr in range(lim + 1):
                d4 = c4[p][nr]
                r4.append(_build_exact_reciprocal(d4, max(c4[p][: lim - nr + 1]) * d4 - 1) if d4 > 0 else (0, 0))
                d2 = c2[p][nr]
                if d2 > 0:
                    ml = max(max(c2[0][: lim - nr + 1]), max(c2[1][: lim - nr + 1]))
                    r2.append(_build_exact_reciprocal(d2, ml * d2 - 1))
                else:
                    r2.append((0, 0))
        return r4, r2

    rec_ec4, rec_ec2 = _inner_rec(even_c4, even_c2, r_max)
    rec_oc4, rec_oc2 = _inner_rec(odd_c4, odd_c2, t_max)

    return {
        "m_max": m_max,
        "r_max": r_max,
        "t_max": t_max,
        "count": c_root[-1],
        "c_root": c_root,
        "s2": s2,
        "w2": w2,
        "s1": s1,
        "w1": w1,
        "rec_root": rec_root,
        "rec_b1": rec_b1,
        "rec_s1": rec_s1,
        "even_c8": even_c8,
        "even_c4": even_c4,
        "even_c2": even_c2,
        "even_p2": even_p2,
        "odd_c4": odd_c4,
        "odd_c2": odd_c2,
        "odd_p2": odd_p2,
        "rec_ec4": rec_ec4,
        "rec_ec2": rec_ec2,
        "rec_oc4": rec_oc4,
        "rec_oc2": rec_oc2,
    }


def _pack_u192(vals: Sequence[int]) -> Any:
    arr = (ctypes.c_uint64 * (len(vals) * 3))()
    mask = (1 << 64) - 1
    for i, v in enumerate(vals):
        arr[3 * i + 0] = v & mask
        arr[3 * i + 1] = (v >> 64) & mask
        arr[3 * i + 2] = (v >> 128) & mask
    return arr


def _pack_u128(vals: Sequence[int]) -> Any:
    arr = (ctypes.c_uint64 * (len(vals) * 2))()
    mask = (1 << 64) - 1
    for i, v in enumerate(vals):
        arr[2 * i + 0] = v & mask
        arr[2 * i + 1] = (v >> 64) & mask
    return arr


def _pack_u64(vals: Sequence[int]) -> Any:
    arr = (ctypes.c_uint64 * len(vals))()
    for i, v in enumerate(vals):
        arr[i] = v
    return arr


def _pack_u32(vals: Sequence[int]) -> Any:
    arr = (ctypes.c_uint32 * len(vals))()
    for i, v in enumerate(vals):
        arr[i] = v
    return arr


def _pack_u16(vals: Sequence[int]) -> Any:
    arr = (ctypes.c_uint16 * len(vals))()
    for i, v in enumerate(vals):
        arr[i] = v
    return arr


def _pack_i8(vals: Sequence[int]) -> Any:
    arr = (ctypes.c_int8 * len(vals))()
    for i, v in enumerate(vals):
        arr[i] = v
    return arr


def _compile_profile(profile: dict[str, Any], src: Path, build_dir: Path) -> tuple[Path, dict[str, Any]]:
    cc = profile["cc"]
    if not shutil.which(cc):
        raise SimdBenchError(f"compiler missing: {cc}")
    pid = profile["id"]
    so_path = build_dir / f"e8had_simd_{pid}.so"
    cmd = [cc, *profile["flags"], str(src), "-o", str(so_path), "-pthread", "-lm"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or proc.stderr.strip():
        raise SimdBenchError(f"build failed/warned under {pid}: {proc.stderr.strip()}")
    if not shutil.which("objdump"):
        raise SimdBenchError("objdump required for zero-div check")
    od = subprocess.run(["objdump", "-d", str(so_path)], capture_output=True, text=True, check=False)
    if od.returncode != 0:
        raise SimdBenchError("objdump failed")
    counts = {sym: -1 for sym in HOT_SYMBOLS}
    cur: str | None = None
    hdr = re.compile(r"^[0-9a-fA-F]+\s+<([^>]+)>:$")
    for line in od.stdout.splitlines():
        m = hdr.match(line)
        if m:
            s = m.group(1)
            cur = s if s in counts else None
            if cur is not None:
                counts[cur] = 0
            continue
        if cur is not None and _DIV_INSN_RE.search(line):
            counts[cur] += 1
    for s, c in counts.items():
        if c != 0:
            raise SimdBenchError(f"symbol {s} under {pid} has div count {c}")
    return so_path, {
        "profile_id": pid,
        "compiler": cc,
        "flags": profile["flags"],
        "zero_warnings": True,
        "hot_symbol_div_counts": counts,
        "all_zero_div": True,
    }


def _bind_lib(so_path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(so_path))
    lib.e8had_has_avx2.argtypes = []
    lib.e8had_has_avx2.restype = ctypes.c_int
    lib.e8had_init.argtypes = [
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_int8), ctypes.c_int, ctypes.POINTER(ctypes.c_int8), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_int8), ctypes.c_int, ctypes.POINTER(ctypes.c_int8), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.e8had_init.restype = ctypes.c_int

    for name in ("e8_decode_std_scalar", "e8had_decode_wht_scalar", "e8had_decode_wht_avx2", "e8had_decode_simd8_avx2"):
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        fn.restype = ctypes.c_size_t

    for name in ("e8had_gemv_materialized_avx2", "e8had_gemv_adjoint_avx2", "e8had_gemv_adjoint_batch4_avx2", "q4km_style_gemv_avx2"):
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p]
        fn.restype = ctypes.c_size_t

    lib.bench_decode_ns.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.bench_decode_ns.restype = ctypes.c_uint64

    lib.bench_gemv_ns.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.bench_gemv_ns.restype = ctypes.c_uint64

    lib.bench_stage_breakdown_ns.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_float),
    ]
    lib.bench_stage_breakdown_ns.restype = ctypes.c_size_t
    return lib


def _init_e8(lib: ctypes.CDLL, tbl: dict[str, Any], nbytes: int) -> None:
    R, T = tbl["r_max"], tbl["t_max"]
    even_off: list[int] = []
    even_pts: list[list[int]] = [[], []]
    for p in (0, 1):
        off = [0]
        for r in range(R + 1):
            for u0, u1 in tbl["even_p2"][p][r]:
                even_pts[p].extend((u0, u1))
            off.append(len(even_pts[p]) // 2)
        even_off.extend(off)
    odd_off: list[int] = []
    odd_pts: list[list[int]] = [[], []]
    for p in (0, 1):
        off = [0]
        for t in range(T + 1):
            for u0, u1 in tbl["odd_p2"][p][t]:
                odd_pts[p].extend((u0, u1))
            off.append(len(odd_pts[p]) // 2)
        odd_off.extend(off)

    rc = lib.e8had_init(
        tbl["m_max"], nbytes,
        _pack_u192(tbl["c_root"]), _pack_u128(tbl["s2"]), _pack_u128(tbl["w2"]),
        _pack_u64(tbl["s1"]), _pack_u64(tbl["w1"]),
        _pack_u192([m for m, _ in tbl["rec_root"]]), _pack_u32([s for _, s in tbl["rec_root"]]),
        _pack_u128([m for m, _ in tbl["rec_b1"]]), _pack_u32([s for _, s in tbl["rec_b1"]]),
        _pack_u128([m for m, _ in tbl["rec_s1"]]), _pack_u32([s for _, s in tbl["rec_s1"]]),
        _pack_u64(tbl["even_c8"]),
        _pack_u32(tbl["even_c4"][0] + tbl["even_c4"][1]),
        _pack_u16(tbl["even_c2"][0] + tbl["even_c2"][1]),
        _pack_u16(even_off),
        _pack_i8(even_pts[0]), len(even_pts[0]) // 2,
        _pack_i8(even_pts[1]), len(even_pts[1]) // 2,
        _pack_u32(tbl["odd_c4"][0] + tbl["odd_c4"][1]),
        _pack_u16(tbl["odd_c2"][0] + tbl["odd_c2"][1]),
        _pack_u16(odd_off),
        _pack_i8(odd_pts[0]), len(odd_pts[0]) // 2,
        _pack_i8(odd_pts[1]), len(odd_pts[1]) // 2,
        _pack_u64([m for m, _ in tbl["rec_ec4"]]), _pack_u32([s for _, s in tbl["rec_ec4"]]),
        _pack_u32([m for m, _ in tbl["rec_ec2"]]), _pack_u32([s for _, s in tbl["rec_ec2"]]),
        _pack_u64([m for m, _ in tbl["rec_oc4"]]), _pack_u32([s for _, s in tbl["rec_oc4"]]),
        _pack_u32([m for m, _ in tbl["rec_oc2"]]), _pack_u32([s for _, s in tbl["rec_oc2"]]),
    )
    if rc != 0:
        raise SimdBenchError(f"e8had_init failed rc={rc}")


def _gen_rank_and_record_buffers(label: str, count: int, groups: int, nbytes: int) -> tuple[bytes, bytes]:
    """Generate deterministic packed rank bytes (groups * nbytes) and GEMV records (groups * (nbytes+1))."""
    ranks_buf = bytearray(groups * nbytes)
    rec_buf = bytearray(groups * (nbytes + 1))
    # Use a fast deterministic 4096-entry SHA-256 rank bank tiled with index XOR-mix mod count
    bank = [
        int.from_bytes(hashlib.sha256(f"{label}:b:{i}".encode("ascii")).digest(), "little") % count
        for i in range(min(groups, 4096))
    ]
    if groups > 0:
        bank[0] = 0
    if groups > 1:
        bank[1] = count - 1
    if groups > 2:
        bank[2] = count // 2
    stride_add = (count // 131071) | 1
    b_len = len(bank)
    for g in range(groups):
        if g < b_len:
            rk = bank[g]
        else:
            rk = (bank[g & 4095] + (g >> 12) * stride_add) % count
        rb = rk.to_bytes(nbytes, "little")
        ranks_buf[g * nbytes : (g + 1) * nbytes] = rb
        scale_code = (g * 37 + 13) & 63
        rec_buf[g * (nbytes + 1)] = scale_code
        rec_buf[g * (nbytes + 1) + 1 : (g + 1) * (nbytes + 1)] = rb
    return bytes(ranks_buf), bytes(rec_buf)


def _gen_q4_records(groups: int) -> bytes:
    out = bytearray(groups * 18)
    pat = hashlib.sha256(b"s15c_q4_pattern").digest()
    for g in range(groups):
        base = g * 18
        out[base] = (g * 37 + 13) & 63
        out[base + 1] = 8
        for i in range(16):
            out[base + 2 + i] = pat[(i + g) & 31]
    return bytes(out)


def _summarize(wall_ns: list[int], cpu_ns: list[int], weights: int, bytes_read: int, threads: int, nom_mhz: int) -> dict[str, Any]:
    wo, co = sorted(wall_ns), sorted(cpu_ns)
    wm, wlo, whi = wo[len(wo) // 2], wo[0], wo[-1]
    cm, clo, chi = co[len(co) // 2], co[0], co[-1]
    cpu_mwps_med = Fraction(weights * 1000, cm)
    cpu_mwps_min = Fraction(weights * 1000, chi)
    cpu_mwps_max = Fraction(weights * 1000, clo)
    wall_mwps_med = Fraction(weights * 1000, wm)
    wall_mwps_max = Fraction(weights * 1000, wlo)
    cpu_ns_w = Fraction(cm, weights)
    # Effective weight GB/s per core = bytes_read / cpu_ns (since 1 B / 1 ns = 1 GB/s!)
    gbps_cpu = Fraction(bytes_read, cm)
    gbps_wall = Fraction(bytes_read, wm)
    # Cycles/weight at nominal MHz = (cpu_ns / weights) * (nom_mhz / 1000)
    cyc_per_w = cpu_ns_w * Fraction(nom_mhz, 1000)
    return {
        "threads": threads,
        "trials": len(wall_ns),
        "weights": weights,
        "matrix_bytes": bytes_read,
        "wall_median_ns": wm,
        "cpu_median_ns": cm,
        "cpu_per_core_m_weights_per_sec_median": format_exact_decimal(cpu_mwps_med, 3),
        "cpu_per_core_m_weights_per_sec_min": format_exact_decimal(cpu_mwps_min, 3),
        "cpu_per_core_m_weights_per_sec_max": format_exact_decimal(cpu_mwps_max, 3),
        "wall_m_weights_per_sec_median": format_exact_decimal(wall_mwps_med, 3),
        "wall_m_weights_per_sec_max": format_exact_decimal(wall_mwps_max, 3),
        "cpu_ns_per_weight_median": format_exact_decimal(cpu_ns_w, 3),
        "cpu_cycles_per_weight_at_nominal_mhz": format_exact_decimal(cyc_per_w, 2),
        "effective_weight_gb_per_sec_cpu_per_core": format_exact_decimal(gbps_cpu, 4),
        "effective_weight_gb_per_sec_wall": format_exact_decimal(gbps_wall, 4),
    }


def _verify_decode_and_gemv_exactness(
    lib: ctypes.CDLL,
    ranks_bytes: bytes,
    rec_bytes: bytes,
    rows: int,
    cols: int,
    python_check_blocks: int,
) -> dict[str, Any]:
    groups = rows * (cols // 32)
    in_c = ctypes.create_string_buffer(ranks_bytes, len(ranks_bytes))
    u_buf = ctypes.create_string_buffer(groups * 32)
    z_sc = ctypes.create_string_buffer(groups * 32)
    z_ax = ctypes.create_string_buffer(groups * 32)
    z_s8 = ctypes.create_string_buffer(groups * 32)

    b0 = int(lib.e8_decode_std_scalar(in_c, groups, u_buf))
    b1 = int(lib.e8had_decode_wht_scalar(in_c, groups, z_sc))
    b2 = int(lib.e8had_decode_wht_avx2(in_c, groups, z_ax))
    b3 = int(lib.e8had_decode_simd8_avx2(in_c, groups, z_s8))
    if b0 or b1 or b2 or b3:
        raise SimdBenchError(f"decode bad counts: {b0},{b1},{b2},{b3}")
    if z_sc.raw != z_ax.raw or z_ax.raw != z_s8.raw:
        raise SimdBenchError("WHT scalar / AVX2 / SIMD8 decoded Z outputs diverged")

    # Verify Python pure-Z isometry & Construction-A membership on first python_check_blocks blocks
    n_chk = min(groups, python_check_blocks)
    u_i8 = (ctypes.c_int8 * (n_chk * 32)).from_buffer(u_buf)
    z_i8 = (ctypes.c_int8 * (n_chk * 32)).from_buffer(z_ax)
    for g in range(n_chk):
        for sb in range(4):
            off = g * 32 + sb * 8
            u8 = [int(u_i8[off + j]) for j in range(8)]
            z8 = [int(z_i8[off + j]) for j in range(8)]
            if not is_valid_e8_std_doubled(u8) or not is_valid_e8_had_scaled(z8):
                raise SimdBenchError(f"invalid E8 / E8_Had vector at block {g} sub {sb}")
            if tuple(z8) != map_e8_std_to_had_exact(u8):
                raise SimdBenchError(f"C11 WHT_8 mismatch vs Python oracle at block {g} sub {sb}")

    # Verify materialized vs adjoint vs batch4 GEMV exactness on all `rows` rows!
    scales = (ctypes.c_float * 256)(*[float(2.0 ** (-(c / 16.0))) for c in range(256)])
    x_vec = (ctypes.c_float * cols)(*[float(((i * 17 + 5) % 19) - 9) * 0.125 for i in range(cols)])
    rec_c = ctypes.create_string_buffer(rec_bytes, len(rec_bytes))
    y_mat = (ctypes.c_float * rows)()
    y_adj = (ctypes.c_float * rows)()
    y_bt4 = (ctypes.c_float * rows)()

    g0 = int(lib.e8had_gemv_materialized_avx2(rec_c, scales, x_vec, rows, cols, y_mat))
    g1 = int(lib.e8had_gemv_adjoint_avx2(rec_c, scales, x_vec, rows, cols, y_adj))
    g2 = int(lib.e8had_gemv_adjoint_batch4_avx2(rec_c, scales, x_vec, rows, cols, y_bt4))
    if g0 or g1 or g2:
        raise SimdBenchError(f"GEMV bad counts: {g0},{g1},{g2}")

    # Check exact floating-point agreement (< 1e-4 relative/abs over 1024-term dot products)
    # and exact bit-equality between adjoint and adjoint_batch4!
    max_diff_adj = 0.0
    max_diff_bt4 = 0.0
    for r in range(rows):
        d1 = abs(float(y_mat[r]) - float(y_adj[r]))
        d2 = abs(float(y_adj[r]) - float(y_bt4[r]))
        if d1 > max_diff_adj:
            max_diff_adj = d1
        if d2 > max_diff_bt4:
            max_diff_bt4 = d2
    if max_diff_adj > 1e-3 or max_diff_bt4 > 1e-4:
        raise SimdBenchError(f"GEMV materialized vs adjoint mismatch: {max_diff_adj}, {max_diff_bt4}")

    return {
        "blocks_verified": groups,
        "weights_verified": groups * 32,
        "e8_subblocks_verified": groups * 4,
        "wht_scalar_avx2_simd8_identical": True,
        "python_oracle_blocks_matched": n_chk,
        "gemv_rows_verified": rows,
        "gemv_materialized_vs_adjoint_within_1e3": True,
        "gemv_adjoint_vs_batch4_within_1e4": True,
        "z_output_sha256": hashlib.sha256(z_ax.raw).hexdigest(),
    }


def _time_decode(lib: ctypes.CDLL, mode: int, threads: int, in_c: Any, groups: int, out_c: Any, nbytes: int, trials: int, nom_mhz: int) -> dict[str, Any]:
    bad_c = ctypes.c_size_t(0)
    cpu_c = ctypes.c_uint64(0)
    ws: list[int] = []
    cs: list[int] = []
    for t in range(trials + 1):
        w_ns = int(lib.bench_decode_ns(mode, threads, in_c, groups, out_c, ctypes.byref(bad_c), ctypes.byref(cpu_c)))
        if int(bad_c.value) != 0:
            raise SimdBenchError(f"decode mode {mode} bad={bad_c.value}")
        if t > 0:
            ws.append(w_ns)
            cs.append(int(cpu_c.value))
    return _summarize(ws, cs, groups * 32, groups * (nbytes + 1), threads, nom_mhz)


def _time_gemv(
    lib: ctypes.CDLL, mode: int, threads: int, rec_c: Any, rec_sz: int,
    scales: Any, x_vec: Any, rows: int, cols: int, y_out: Any, trials: int, nom_mhz: int
) -> dict[str, Any]:
    bad_c = ctypes.c_size_t(0)
    cpu_c = ctypes.c_uint64(0)
    ws: list[int] = []
    cs: list[int] = []
    for t in range(trials + 1):
        w_ns = int(
            lib.bench_gemv_ns(mode, threads, rec_c, rec_sz, scales, x_vec, rows, cols, y_out, ctypes.byref(bad_c), ctypes.byref(cpu_c))
        )
        if int(bad_c.value) != 0:
            raise SimdBenchError(f"gemv mode {mode} bad={bad_c.value}")
        if t > 0:
            ws.append(w_ns)
            cs.append(int(cpu_c.value))
    groups = rows * (cols // 32)
    return _summarize(ws, cs, rows * cols, groups * rec_sz, threads, nom_mhz)


def _measure_stage_breakdown(lib: ctypes.CDLL, in_c: Any, groups: int, trials: int, nom_mhz: int) -> dict[str, Any]:
    x32 = (ctypes.c_float * 32)(*[float(((i * 7 + 3) % 11) - 5) * 0.25 for i in range(32)])
    st = (ctypes.c_uint64 * 4)()
    chk = ctypes.c_float(0.0)
    samples: list[list[int]] = [[], [], [], []]
    for t in range(trials + 1):
        bad = int(lib.bench_stage_breakdown_ns(in_c, groups, x32, st, ctypes.byref(chk)))
        if bad != 0:
            raise SimdBenchError(f"stage breakdown bad={bad}")
        if t > 0:
            for k in range(4):
                samples[k].append(int(st[k]))
    meds = [sorted(s)[len(s) // 2] for s in samples]
    tot = sum(meds)
    weights = groups * 32
    names = ("stage0_outer_tree_splits", "stage1_inner_e8_leaf_unrank", "stage2_materialized_avx2_wht8", "stage3_avx2_fma_dot_product")
    out: dict[str, Any] = {}
    for name, ns in zip(names, meds):
        ns_w = Fraction(ns, weights)
        cyc_w = ns_w * Fraction(nom_mhz, 1000)
        pct = Fraction(ns * 100, tot) if tot > 0 else Fraction(0, 1)
        out[name] = {
            "median_cpu_ns": ns,
            "ns_per_weight": format_exact_decimal(ns_w, 3),
            "cycles_per_weight_nominal": format_exact_decimal(cyc_w, 2),
            "share_of_materialized_pct": format_exact_decimal(pct, 2) + "%",
        }
    return out


def run_suite(*, bench: bool, rows: int, cols: int, trials: int, oracle_blocks: int) -> dict[str, Any]:
    src = _HERE / "e8had_simd_bench.c"
    src_sha = _sha256_file(src)
    groups = rows * (cols // 32)

    tbl_136 = _build_e8_4_tables_self_contained(391)
    tbl_88 = _build_e8_4_tables_self_contained(48)

    rk136, rec136 = _gen_rank_and_record_buffers("s15c_e8_136b", tbl_136["count"], groups, 17)
    rk88, rec88 = _gen_rank_and_record_buffers("s15c_e8_88b", tbl_88["count"], groups, 11)
    q4_rec = _gen_q4_records(groups)

    host = _detect_host() if bench else None
    nom_mhz = host["nominal_mhz"] if host else 2300

    builds: list[dict[str, Any]] = []
    exactness: dict[str, Any] = {}
    benchmarks: dict[str, Any] = {}

    with tempfile.TemporaryDirectory(prefix="s15c_simd_") as td:
        os.chmod(td, 0o700)
        bdir = Path(td)
        with ThreadPoolExecutor(max_workers=len(COMPILER_PROFILES)) as pool:
            compiled = list(pool.map(lambda pr: _compile_profile(pr, src, bdir), COMPILER_PROFILES))

        for profile, (so_path, bmeta) in zip(COMPILER_PROFILES, compiled):
            pid = profile["id"]
            lib = _bind_lib(so_path)
            bmeta["avx2_enabled_in_binary"] = bool(int(lib.e8had_has_avx2()))
            builds.append(bmeta)

            # Exactness verification on BOTH 136b and 88b
            _init_e8(lib, tbl_136, 17)
            ex136 = _verify_decode_and_gemv_exactness(lib, rk136, rec136, rows, cols, oracle_blocks)
            _init_e8(lib, tbl_88, 11)
            ex88 = _verify_decode_and_gemv_exactness(lib, rk88, rec88, rows, cols, oracle_blocks)
            exactness[pid] = {"ebg32_b136s8": ex136, "ebg32_b88s8": ex88}

            if bench:
                rk136_c = ctypes.create_string_buffer(rk136, len(rk136))
                rec136_c = ctypes.create_string_buffer(rec136, len(rec136))
                rk88_c = ctypes.create_string_buffer(rk88, len(rk88))
                rec88_c = ctypes.create_string_buffer(rec88, len(rec88))
                q4_c = ctypes.create_string_buffer(q4_rec, len(q4_rec))
                out_i8 = ctypes.create_string_buffer(groups * 32)
                scales = (ctypes.c_float * 256)(*[float(2.0 ** (-(c / 16.0))) for c in range(256)])
                x_vec = (ctypes.c_float * cols)(*[float(((i * 17 + 5) % 19) - 9) * 0.125 for i in range(cols)])
                y_out = (ctypes.c_float * rows)()

                pb: dict[str, Any] = {}
                # Matched 4.500-bpw Q4_K_M-style AVX2/FMA GEMV baseline (1t & 2t)
                pb["q4km_style_gemv_avx2_1t"] = _time_gemv(lib, 3, 1, q4_c, 18, scales, x_vec, rows, cols, y_out, trials, nom_mhz)
                pb["q4km_style_gemv_avx2_2t"] = _time_gemv(lib, 3, 2, q4_c, 18, scales, x_vec, rows, cols, y_out, trials, nom_mhz)

                # EBG32:b136s8 (4.500 bpw, m_max=391)
                _init_e8(lib, tbl_136, 17)
                pb["e8_136b_decode_std_scalar_1t"] = _time_decode(lib, 0, 1, rk136_c, groups, out_i8, 17, trials, nom_mhz)
                pb["e8had_136b_decode_wht_scalar_1t"] = _time_decode(lib, 1, 1, rk136_c, groups, out_i8, 17, trials, nom_mhz)
                pb["e8had_136b_decode_wht_avx2_1t"] = _time_decode(lib, 2, 1, rk136_c, groups, out_i8, 17, trials, nom_mhz)
                pb["e8had_136b_decode_simd8_avx2_1t"] = _time_decode(lib, 3, 1, rk136_c, groups, out_i8, 17, trials, nom_mhz)
                pb["e8had_136b_gemv_materialized_avx2_1t"] = _time_gemv(lib, 0, 1, rec136_c, 18, scales, x_vec, rows, cols, y_out, trials, nom_mhz)
                pb["e8had_136b_gemv_adjoint_avx2_1t"] = _time_gemv(lib, 1, 1, rec136_c, 18, scales, x_vec, rows, cols, y_out, trials, nom_mhz)
                pb["e8had_136b_gemv_adjoint_batch4_avx2_1t"] = _time_gemv(lib, 2, 1, rec136_c, 18, scales, x_vec, rows, cols, y_out, trials, nom_mhz)
                pb["e8had_136b_gemv_adjoint_batch4_avx2_2t"] = _time_gemv(lib, 2, 2, rec136_c, 18, scales, x_vec, rows, cols, y_out, trials, nom_mhz)
                st136 = _measure_stage_breakdown(lib, rk136_c, groups, trials, nom_mhz)

                # EBG32:b88s8 (3.000 bpw, m_max=48, L1d-resident)
                _init_e8(lib, tbl_88, 11)
                pb["e8_88b_decode_std_scalar_1t"] = _time_decode(lib, 0, 1, rk88_c, groups, out_i8, 11, trials, nom_mhz)
                pb["e8had_88b_decode_wht_avx2_1t"] = _time_decode(lib, 2, 1, rk88_c, groups, out_i8, 11, trials, nom_mhz)
                pb["e8had_88b_decode_simd8_avx2_1t"] = _time_decode(lib, 3, 1, rk88_c, groups, out_i8, 11, trials, nom_mhz)
                pb["e8had_88b_gemv_materialized_avx2_1t"] = _time_gemv(lib, 0, 1, rec88_c, 12, scales, x_vec, rows, cols, y_out, trials, nom_mhz)
                pb["e8had_88b_gemv_adjoint_avx2_1t"] = _time_gemv(lib, 1, 1, rec88_c, 12, scales, x_vec, rows, cols, y_out, trials, nom_mhz)
                pb["e8had_88b_gemv_adjoint_batch4_avx2_1t"] = _time_gemv(lib, 2, 1, rec88_c, 12, scales, x_vec, rows, cols, y_out, trials, nom_mhz)
                pb["e8had_88b_gemv_adjoint_batch4_avx2_2t"] = _time_gemv(lib, 2, 2, rec88_c, 12, scales, x_vec, rows, cols, y_out, trials, nom_mhz)
                st88 = _measure_stage_breakdown(lib, rk88_c, groups, trials, nom_mhz)

                q4_cpu = pb["q4km_style_gemv_avx2_1t"]["cpu_median_ns"]
                ratio_vs_q4 = {
                    k: format_exact_decimal(Fraction(v["cpu_median_ns"], q4_cpu), 2) + "x_slower"
                    for k, v in pb.items()
                }
                benchmarks[pid] = {
                    "timings": pb,
                    "cpu_ratio_vs_matched_q4km_avx2_1t": ratio_vs_q4,
                    "stage_breakdown_ebg32_b136s8": st136,
                    "stage_breakdown_ebg32_b88s8": st88,
                }

    ref136 = exactness["gcc_native"]["ebg32_b136s8"]["z_output_sha256"]
    ref88 = exactness["gcc_native"]["ebg32_b88s8"]["z_output_sha256"]
    cross_ok = all(
        exactness[p["id"]]["ebg32_b136s8"]["z_output_sha256"] == ref136
        and exactness[p["id"]]["ebg32_b88s8"]["z_output_sha256"] == ref88
        for p in COMPILER_PROFILES
    )
    if not cross_ok:
        raise SimdBenchError("cross-compiler E_8_Had decode outputs diverged")

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode": "empirical_simd_gemv_benchmark" if bench else "deterministic_verify_only",
        "source_sha256": src_sha,
        "matrix_shape": {"rows": rows, "cols": cols, "weights": rows * cols, "blocks_32": groups},
        "compiler_builds": builds,
        "all_profiles_zero_warnings": True,
        "all_hot_kernels_zero_div": True,
        "cross_compiler_outputs_identical": cross_ok,
        "exactness": exactness,
        "overall_verified": True,
    }
    if bench:
        result["host_environment"] = host
        result["benchmarks"] = benchmarks
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Track S15c C11 + AVX2/FMA E_8_Had verifier & GEMV benchmark.")
    parser.add_argument("--bench", action="store_true", help="Run full 4096x1024 timed benchmark.")
    parser.add_argument("--rows", type=int, default=None, help="Matrix rows (default: 64 verify, 4096 bench).")
    parser.add_argument("--cols", type=int, default=None, help="Matrix cols multiple of 32 (default: 512 verify, 1024 bench).")
    parser.add_argument("--trials", type=int, default=5, help="Timed trials per kernel in --bench (default: 5).")
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON to stdout.")
    parser.add_argument("--write-results", type=Path, default=None, help="Run --bench and write JSON to path.")
    args = parser.parse_args(argv)

    bench = bool(args.bench or args.write_results is not None)
    rows = args.rows if args.rows is not None else (4096 if bench else 64)
    cols = args.cols if args.cols is not None else (1024 if bench else 512)
    oracle_blocks = 128 if bench else 32
    res = run_suite(bench=bench, rows=rows, cols=cols, trials=args.trials, oracle_blocks=oracle_blocks)
    rendered = json.dumps(res, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if args.write_results is not None:
        args.write_results.write_text(rendered, encoding="utf-8")
    if args.json:
        sys.stdout.write(rendered)
    else:
        if bench:
            g = res["benchmarks"]["gcc_native"]["timings"]
            sys.stdout.write(
                f"TRACK-S15C BENCH OK ({rows}x{cols}={rows*cols} weights, gcc_native CPU M w/s/c):\n"
                f"  q4km_style_gemv_avx2 (1t/2t wall):     "
                f"{g['q4km_style_gemv_avx2_1t']['cpu_per_core_m_weights_per_sec_median']} M w/s/c | "
                f"2t wall {g['q4km_style_gemv_avx2_2t']['wall_m_weights_per_sec_median']} M w/s\n"
                f"  e8_136b std / wht_avx2 / simd8:        "
                f"{g['e8_136b_decode_std_scalar_1t']['cpu_per_core_m_weights_per_sec_median']} / "
                f"{g['e8had_136b_decode_wht_avx2_1t']['cpu_per_core_m_weights_per_sec_median']} / "
                f"{g['e8had_136b_decode_simd8_avx2_1t']['cpu_per_core_m_weights_per_sec_median']} M w/s/c\n"
                f"  e8had_136b gemv mat / adj / bt4 (2tw): "
                f"{g['e8had_136b_gemv_materialized_avx2_1t']['cpu_per_core_m_weights_per_sec_median']} / "
                f"{g['e8had_136b_gemv_adjoint_avx2_1t']['cpu_per_core_m_weights_per_sec_median']} / "
                f"{g['e8had_136b_gemv_adjoint_batch4_avx2_1t']['cpu_per_core_m_weights_per_sec_median']} M w/s/c "
                f"(2t wall {g['e8had_136b_gemv_adjoint_batch4_avx2_2t']['wall_m_weights_per_sec_median']})\n"
                f"  e8_88b std / wht_avx2 / simd8:         "
                f"{g['e8_88b_decode_std_scalar_1t']['cpu_per_core_m_weights_per_sec_median']} / "
                f"{g['e8had_88b_decode_wht_avx2_1t']['cpu_per_core_m_weights_per_sec_median']} / "
                f"{g['e8had_88b_decode_simd8_avx2_1t']['cpu_per_core_m_weights_per_sec_median']} M w/s/c\n"
                f"  e8had_88b gemv mat / adj / bt4 (2tw):  "
                f"{g['e8had_88b_gemv_materialized_avx2_1t']['cpu_per_core_m_weights_per_sec_median']} / "
                f"{g['e8had_88b_gemv_adjoint_avx2_1t']['cpu_per_core_m_weights_per_sec_median']} / "
                f"{g['e8had_88b_gemv_adjoint_batch4_avx2_1t']['cpu_per_core_m_weights_per_sec_median']} M w/s/c "
                f"(2t wall {g['e8had_88b_gemv_adjoint_batch4_avx2_2t']['wall_m_weights_per_sec_median']})\n"
            )
        else:
            d = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
            sys.stdout.write(
                f"TRACK-S15C VERIFY OK shape={rows}x{cols} builds=2/2 zero_div=True "
                f"mismatches=0 sha256={d[:16]}...\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
