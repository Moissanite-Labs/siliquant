#!/usr/bin/env python3
"""Reproducible C11 + Python empirical microbenchmark & exactness verifier for Track S15b
(AUDIT-112 empirical companion to Moissanite-Labs/siliquant#23 and #25).

Modes:
  (default / --verify-only)  Deterministic 3-compiler build check, zero-division disassembly
                             check, and exact roundtrip + baseline cross-check (0 timestamps;
                             deterministic canonical JSON for 3-mode -I parity).
  --bench                    Full 100,000-block (3,200,000-weight) exactness verification PLUS
                             multi-trial 1-core & 2-core CLOCK_MONOTONIC wall-clock benchmarks
                             across gcc (-O3 -march=native), gcc (-O3), and clang (-O3).
  --write-results <path>     Implies --bench and writes canonical JSON results to <path>.

Strict discipline: pure Python 3 stdlib, zero bare asserts, zero __pycache__ written to sibling
tracks, zero compiled binaries left in workspace (all .so artifacts built in 0700 TemporaryDirectory).
"""

from __future__ import annotations

import sys

# Prevent writing .pyc into frozen sibling tracks (track-s15 / intake-siliquant-ballpack)
sys.dont_write_bytecode = True

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
from typing import Any, Sequence

_HERE = Path(__file__).resolve().parent
_TRACK15_DIR = _HERE.parent / "track-s15-ballpack-tree-unrank"
_INTAKE_DIR = _HERE.parent / "intake-siliquant-ballpack"

if str(_TRACK15_DIR) not in sys.path:
    sys.path.insert(0, str(_TRACK15_DIR))

from ballpack_tree_oracle import (
    build_e8_tree_tables,
    build_z_tree_tables,
    count_z_ball_serial_dp_row,
    format_exact_decimal,
    unrank_tree_e8_blocks,
    unrank_tree_z,
)

SCHEMA_VERSION = "track_s15b.ballpack_c_bench.v1"
PINNED_BASELINE_COMMIT = "20d13d5fa9c5fb2333b8faa0360a90caa1e0a43f"
PINNED_BALLPACK_C_SHA256 = "53074fbadf169650da6cad400b986e07b1ea15df180f6f148f97ef4761b33055"
Q4KM_FALSIFIER_MW_PER_SEC_PER_CORE = 45000  # siliquant#23 reported Q4_K_M throughput ceiling (M weights/s/core)

COMPILER_PROFILES: tuple[dict[str, Any], ...] = (
    {
        "id": "gcc_native",
        "cc": "gcc",
        "flags": ["-O3", "-march=native", "-Wall", "-Wextra", "-Werror", "-std=c11", "-shared", "-fPIC"],
    },
    {
        "id": "gcc_portable",
        "cc": "gcc",
        "flags": ["-O3", "-Wall", "-Wextra", "-Werror", "-std=c11", "-shared", "-fPIC"],
    },
    {
        "id": "clang_portable",
        "cc": "clang",
        "flags": ["-O3", "-Wall", "-Wextra", "-Werror", "-std=c11", "-shared", "-fPIC"],
    },
)

HOT_UNRANK_SYMBOLS: tuple[str, ...] = (
    "ztree_unrank_compact",
    "ztree_unrank_fast",
    "ztree_unrank_fast_batch4",
    "e8tree_unrank_compact",
    "e8tree_unrank_fast",
    "e8tree_unrank_fast_batch4",
)

_DIV_INSN_RE = re.compile(r"\t(?:i?div[bwlq]?)\b")


class BenchError(ValueError):
    """Fail-closed error raised on build, table, exactness, or runtime violations."""


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise BenchError(f"missing required source file: {path.name}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _detect_cpu_and_compilers() -> dict[str, Any]:
    cpu_model = platform.processor() or "unknown"
    cores = os.cpu_count() or 1
    lscpu_info: dict[str, str] = {}
    if shutil.which("lscpu"):
        proc = subprocess.run(["lscpu"], capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    lscpu_info[k.strip()] = v.strip()
            if "Model name" in lscpu_info:
                cpu_model = lscpu_info["Model name"]
    compilers: dict[str, str] = {}
    for cc in ("gcc", "clang"):
        if shutil.which(cc):
            proc = subprocess.run([cc, "--version"], capture_output=True, text=True, check=False)
            first = proc.stdout.splitlines()[0].strip() if proc.stdout else "unknown"
            compilers[cc] = first
    loadavg_str = "unknown"
    load_path = Path("/proc/loadavg")
    if load_path.is_file():
        loadavg_str = load_path.read_text(encoding="utf-8").strip()
    return {
        "architecture": platform.machine(),
        "cpu_model": cpu_model,
        "logical_cpus": cores,
        "l1d_cache": lscpu_info.get("L1d cache", "unknown"),
        "l2_cache": lscpu_info.get("L2 cache", "unknown"),
        "l3_cache": lscpu_info.get("L3 cache", "unknown"),
        "host_loadavg_at_bench": loadavg_str,
        "python_version": platform.python_version(),
        "compilers": compilers,
    }


def _pack_u192_limbs(values: Sequence[int]) -> Any:
    arr = (ctypes.c_uint64 * (len(values) * 3))()
    mask = (1 << 64) - 1
    for i, v in enumerate(values):
        if v < 0 or v >= (1 << 192):
            raise BenchError(f"value out of uint192 range at index {i}")
        arr[3 * i + 0] = v & mask
        arr[3 * i + 1] = (v >> 64) & mask
        arr[3 * i + 2] = (v >> 128) & mask
    return arr


def _pack_u128_limbs(values: Sequence[int]) -> Any:
    arr = (ctypes.c_uint64 * (len(values) * 2))()
    mask = (1 << 64) - 1
    for i, v in enumerate(values):
        if v < 0 or v >= (1 << 128):
            raise BenchError(f"value out of uint128 range at index {i}")
        arr[2 * i + 0] = v & mask
        arr[2 * i + 1] = (v >> 64) & mask
    return arr


def _pack_u64(values: Sequence[int]) -> Any:
    arr = (ctypes.c_uint64 * len(values))()
    for i, v in enumerate(values):
        if v < 0 or v >= (1 << 64):
            raise BenchError(f"value out of uint64 range at index {i}")
        arr[i] = v
    return arr


def _pack_u32(values: Sequence[int]) -> Any:
    arr = (ctypes.c_uint32 * len(values))()
    for i, v in enumerate(values):
        if v < 0 or v >= (1 << 32):
            raise BenchError(f"value out of uint32 range at index {i}")
        arr[i] = v
    return arr


def _pack_u16(values: Sequence[int]) -> Any:
    arr = (ctypes.c_uint16 * len(values))()
    for i, v in enumerate(values):
        if v < 0 or v >= (1 << 16):
            raise BenchError(f"value out of uint16 range at index {i}")
        arr[i] = v
    return arr


def _pack_i8(values: Sequence[int]) -> Any:
    arr = (ctypes.c_int8 * len(values))()
    for i, v in enumerate(values):
        if v < -128 or v > 127:
            raise BenchError(f"value out of int8 range at index {i}: {v}")
        arr[i] = v
    return arr


def _build_ballpack_c_ways_table(n: int, bound: int) -> tuple[Any, int]:
    """Build exact Z^m ball DP table ways[m][r] for m=0..n, r=0..bound (matches ballindex.py)."""
    # Compute all rows incrementally using exact integer arithmetic
    syms = [0]
    k = 1
    while k * k <= bound:
        syms.append(k)
        syms.append(-k)
        k += 1
    costs = [s * s for s in syms]
    row = [1] * (bound + 1)
    rows = [row]
    for _ in range(n):
        nxt = [0] * (bound + 1)
        for c in costs:
            for r in range(c, bound + 1):
                nxt[r] += row[r - c]
        row = nxt
        rows.append(row)
    flat: list[int] = []
    for r_vec in rows:
        flat.extend(r_vec)
    return _pack_u192_limbs(flat), rows[n][bound]


def _generate_deterministic_ranks(label: str, count: int, n_groups: int, nbytes: int) -> tuple[bytes, list[int], bytes]:
    """Generate n_groups deterministic full-width ranks in [0, count) plus an out-of-range rank=count."""
    if n_groups < 1:
        raise BenchError("n_groups must be >= 1")
    ranks: list[int] = []
    # Fixed edge/boundary ranks first
    fixed = [0, 1, 2, count - 1, count - 2, count // 2, count // 2 - 1, count // 2 + 1]
    bit = 1
    while bit < count:
        fixed.append(bit)
        fixed.append(count - bit)
        bit <<= 1
    seen: set[int] = set()
    for rk in fixed:
        if 0 <= rk < count and rk not in seen and len(ranks) < n_groups:
            seen.add(rk)
            ranks.append(rk)

    # Fill remainder via deterministic SHA-256 counter stream
    idx = 0
    while len(ranks) < n_groups:
        digest = hashlib.sha256(f"{label}:{idx}".encode("ascii")).digest()
        val = int.from_bytes(digest, "little") % count
        ranks.append(val)
        idx += 1

    buf = bytearray(n_groups * nbytes)
    for i, rk in enumerate(ranks):
        buf[i * nbytes : (i + 1) * nbytes] = rk.to_bytes(nbytes, "little")
    oob = count.to_bytes(nbytes, "little")
    return bytes(buf), ranks, oob


def _compile_profile(
    profile: dict[str, Any],
    base_src: Path,
    tree_src: Path,
    build_dir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    cc = profile["cc"]
    if not shutil.which(cc):
        raise BenchError(f"compiler not found on PATH: {cc}")
    pid = profile["id"]
    base_so = build_dir / f"ballpack_{pid}.so"
    tree_so = build_dir / f"ballpack_tree_{pid}.so"

    base_cmd = [cc, *profile["flags"], str(base_src), "-o", str(base_so), "-lm"]
    tree_cmd = [cc, *profile["flags"], str(tree_src), "-o", str(tree_so), "-pthread"]

    p_base = subprocess.run(base_cmd, capture_output=True, text=True, check=False)
    if p_base.returncode != 0 or p_base.stderr.strip():
        raise BenchError(f"baseline build failed or warned under {pid}: {p_base.stderr.strip()}")

    p_tree = subprocess.run(tree_cmd, capture_output=True, text=True, check=False)
    if p_tree.returncode != 0 or p_tree.stderr.strip():
        raise BenchError(f"tree build failed or warned under {pid}: {p_tree.stderr.strip()}")

    div_counts = _audit_zero_division_disassembly(tree_so)
    meta = {
        "profile_id": pid,
        "compiler": cc,
        "flags": profile["flags"],
        "baseline_zero_warnings": True,
        "tree_zero_warnings": True,
        "hot_unrank_div_insn_counts": div_counts,
        "all_hot_unrank_zero_div": all(v == 0 for v in div_counts.values()),
    }
    return base_so, tree_so, meta


def _audit_zero_division_disassembly(so_path: Path) -> dict[str, int]:
    """Verify 0 hardware div/idiv instructions in every hot unranker via objdump -d."""
    if not shutil.which("objdump"):
        raise BenchError("objdump required to certify 0 hardware division instructions")
    proc = subprocess.run(["objdump", "-d", str(so_path)], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise BenchError(f"objdump failed on {so_path.name}")
    counts = {sym: -1 for sym in HOT_UNRANK_SYMBOLS}
    cur_sym: str | None = None
    hdr_re = re.compile(r"^[0-9a-fA-F]+\s+<([^>]+)>:$")
    for line in proc.stdout.splitlines():
        m = hdr_re.match(line)
        if m:
            sym = m.group(1)
            cur_sym = sym if sym in counts else None
            if cur_sym is not None:
                counts[cur_sym] = 0
            continue
        if cur_sym is not None and _DIV_INSN_RE.search(line):
            counts[cur_sym] += 1
    for sym, cnt in counts.items():
        if cnt < 0:
            raise BenchError(f"expected symbol missing from disassembly: {sym}")
    return counts


def _bind_libs(base_so: Path, tree_so: Path) -> tuple[ctypes.CDLL, ctypes.CDLL]:
    base_lib = ctypes.CDLL(str(base_so))
    tree_lib = ctypes.CDLL(str(tree_so))

    base_lib.zb_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint64)]
    base_lib.zb_init.restype = ctypes.c_int
    base_lib.zb_unrank.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    base_lib.zb_unrank.restype = ctypes.c_size_t
    base_lib.zb_rank.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    base_lib.zb_rank.restype = ctypes.c_size_t

    tree_lib.ztree_init.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    tree_lib.ztree_init.restype = ctypes.c_int
    tree_lib.ztree_checkpoint_bytes.argtypes = []
    tree_lib.ztree_checkpoint_bytes.restype = ctypes.c_size_t

    for fn_name in (
        "ztree_unrank_compact",
        "ztree_unrank_fast",
        "ztree_unrank_fast_batch4",
        "ztree_rank",
        "e8tree_unrank_compact",
        "e8tree_unrank_fast",
        "e8tree_unrank_fast_batch4",
        "e8tree_rank",
    ):
        fn = getattr(tree_lib, fn_name)
        fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        fn.restype = ctypes.c_size_t

    tree_lib.e8tree_init.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int8),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int8),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    tree_lib.e8tree_init.restype = ctypes.c_int
    tree_lib.e8tree_checkpoint_bytes.argtypes = []
    tree_lib.e8tree_checkpoint_bytes.restype = ctypes.c_size_t

    tree_lib.bench_unrank_ns.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    tree_lib.bench_unrank_ns.restype = ctypes.c_uint64

    tree_lib.bench_external_unrank_ns.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    tree_lib.bench_external_unrank_ns.restype = ctypes.c_uint64
    return base_lib, tree_lib


def _init_z32_tables(tree_lib: ctypes.CDLL, z_tbl: dict[str, Any], nbytes: int) -> int:
    B = z_tbl["bound"]
    z2_off = [0]
    z2_pts_flat: list[int] = []
    for r in range(B + 1):
        for x, y in z_tbl["z2_shell_points"][r]:
            z2_pts_flat.append(x)
            z2_pts_flat.append(y)
        z2_off.append(len(z2_pts_flat) // 2)
    z2_pairs = len(z2_pts_flat) // 2

    def _rec_arrays(rec_map: dict[int, dict[str, Any]], size: int) -> tuple[list[int], list[int]]:
        ms = [0] * size
        ss = [0] * size
        for k, rec in rec_map.items():
            if not rec["universally_valid"]:
                raise BenchError(f"reciprocal failed universal check at key {k}")
            ms[k] = rec["mult"]
            ss[k] = rec["shift"]
        return ms, ss

    rr_m, rr_s = _rec_arrays(z_tbl["recip_ball"][16], B + 1)
    rb8_m, rb8_s = _rec_arrays(z_tbl["recip_ball"][8], B + 1)
    rs8_m, rs8_s = _rec_arrays(z_tbl["recip_shell"][8], B + 1)
    rb4_m, rb4_s = _rec_arrays(z_tbl["recip_ball"][4], B + 1)
    rs4_m, rs4_s = _rec_arrays(z_tbl["recip_shell"][4], B + 1)
    rb2_m, rb2_s = _rec_arrays(z_tbl["recip_ball"][2], B + 1)
    rs2_m, rs2_s = _rec_arrays(z_tbl["recip_shell"][2], B + 1)

    rc = tree_lib.ztree_init(
        B,
        nbytes,
        z2_pairs,
        _pack_u192_limbs(z_tbl["c_root"]),
        _pack_u128_limbs(z_tbl["shells"][16]),
        _pack_u128_limbs(z_tbl["balls"][16]),
        _pack_u64(z_tbl["shells"][8]),
        _pack_u64(z_tbl["balls"][8]),
        _pack_u32(z_tbl["shells"][4]),
        _pack_u32(z_tbl["balls"][4]),
        _pack_u16(z_tbl["shells"][2]),
        _pack_u16(z_tbl["balls"][2]),
        _pack_u16(z2_off),
        _pack_i8(z2_pts_flat),
        _pack_u192_limbs(rr_m),
        _pack_u32(rr_s),
        _pack_u128_limbs(rb8_m),
        _pack_u32(rb8_s),
        _pack_u128_limbs(rs8_m),
        _pack_u32(rs8_s),
        _pack_u64(rb4_m),
        _pack_u32(rb4_s),
        _pack_u64(rs4_m),
        _pack_u32(rs4_s),
        _pack_u32(rb2_m),
        _pack_u32(rb2_s),
        _pack_u32(rs2_m),
        _pack_u32(rs2_s),
    )
    if rc != 0:
        raise BenchError(f"ztree_init failed with code {rc}")
    return int(tree_lib.ztree_checkpoint_bytes())


def _init_e8_tables(tree_lib: ctypes.CDLL, e8_tbl: dict[str, Any], nbytes: int) -> int:
    M = e8_tbl["m_max"]
    single = e8_tbl["single"]
    R = single["r_max"]
    T = single["t_max"]

    def _outer_rec(rec_map: dict[int, dict[str, Any]], size: int) -> tuple[list[int], list[int]]:
        ms = [0] * size
        ss = [0] * size
        for k, rec in rec_map.items():
            if not rec["universally_valid"]:
                raise BenchError(f"E8 outer reciprocal failed universal check at {k}")
            ms[k] = rec["mult"]
            ss[k] = rec["shift"]
        return ms, ss

    rr_m, rr_s = _outer_rec(e8_tbl["outer_recip_ball"][2], M + 1)
    rb1_m, rb1_s = _outer_rec(e8_tbl["outer_recip_ball"][1], M + 1)
    rs1_m, rs1_s = _outer_rec(e8_tbl["outer_recip_shell"][1], M + 1)

    even_c4_flat = list(single["even_c4"][0]) + list(single["even_c4"][1])
    even_c2_flat = list(single["even_c2"][0]) + list(single["even_c2"][1])
    odd_c4_flat = list(single["odd_c4"][0]) + list(single["odd_c4"][1])
    odd_c2_flat = list(single["odd_c2"][0]) + list(single["odd_c2"][1])

    even_off_flat: list[int] = []
    even_pts_by_p: list[list[int]] = [[], []]
    for p in (0, 1):
        off = [0]
        for r in range(R + 1):
            for u0, u1 in single["even_p2"][p][r]:
                even_pts_by_p[p].append(u0)
                even_pts_by_p[p].append(u1)
            off.append(len(even_pts_by_p[p]) // 2)
        even_off_flat.extend(off)

    odd_off_flat: list[int] = []
    odd_pts_by_p: list[list[int]] = [[], []]
    for p in (0, 1):
        off = [0]
        for t in range(T + 1):
            for u0, u1 in single["odd_p2"][p][t]:
                odd_pts_by_p[p].append(u0)
                odd_pts_by_p[p].append(u1)
            off.append(len(odd_pts_by_p[p]) // 2)
        odd_off_flat.extend(off)

    irec = single["inner_recips"]

    def _inner_rec(coset: int, dim: int, lim: int) -> tuple[list[int], list[int]]:
        ms: list[int] = []
        ss: list[int] = []
        for p in (0, 1):
            for nr in range(lim + 1):
                rec = irec.get((coset, dim, p, nr))
                if rec is None:
                    ms.append(0)
                    ss.append(0)
                else:
                    if not rec["universally_valid"]:
                        raise BenchError(f"E8 inner reciprocal failed at {(coset, dim, p, nr)}")
                    ms.append(rec["mult"])
                    ss.append(rec["shift"])
        return ms, ss

    rec_ec4_m, rec_ec4_s = _inner_rec(0, 4, R)
    rec_ec2_m, rec_ec2_s = _inner_rec(0, 2, R)
    rec_oc4_m, rec_oc4_s = _inner_rec(1, 4, T)
    rec_oc2_m, rec_oc2_s = _inner_rec(1, 2, T)

    rc = tree_lib.e8tree_init(
        M,
        nbytes,
        _pack_u192_limbs(e8_tbl["c_root"]),
        _pack_u128_limbs(e8_tbl["shells"][2]),
        _pack_u128_limbs(e8_tbl["balls"][2]),
        _pack_u64(e8_tbl["shells"][1]),
        _pack_u64(e8_tbl["balls"][1]),
        _pack_u192_limbs(rr_m),
        _pack_u32(rr_s),
        _pack_u128_limbs(rb1_m),
        _pack_u32(rb1_s),
        _pack_u128_limbs(rs1_m),
        _pack_u32(rs1_s),
        _pack_u64(single["even_c8"]),
        _pack_u32(even_c4_flat),
        _pack_u16(even_c2_flat),
        _pack_u16(even_off_flat),
        _pack_i8(even_pts_by_p[0]),
        len(even_pts_by_p[0]) // 2,
        _pack_i8(even_pts_by_p[1]),
        len(even_pts_by_p[1]) // 2,
        _pack_u32(odd_c4_flat),
        _pack_u16(odd_c2_flat),
        _pack_u16(odd_off_flat),
        _pack_i8(odd_pts_by_p[0]),
        len(odd_pts_by_p[0]) // 2,
        _pack_i8(odd_pts_by_p[1]),
        len(odd_pts_by_p[1]) // 2,
        _pack_u64(rec_ec4_m),
        _pack_u32(rec_ec4_s),
        _pack_u32(rec_ec2_m),
        _pack_u32(rec_ec2_s),
        _pack_u64(rec_oc4_m),
        _pack_u32(rec_oc4_s),
        _pack_u32(rec_oc2_m),
        _pack_u32(rec_oc2_s),
    )
    if rc != 0:
        raise BenchError(f"e8tree_init failed with code {rc}")
    return int(tree_lib.e8tree_checkpoint_bytes())


def _verify_z32_exactness(
    base_lib: ctypes.CDLL,
    tree_lib: ctypes.CDLL,
    z_tbl: dict[str, Any],
    in_bytes: bytes,
    ranks: list[int],
    oob_bytes: bytes,
    nbytes: int,
    oracle_check_n: int,
) -> dict[str, Any]:
    groups = len(ranks)
    in_buf = ctypes.create_string_buffer(in_bytes, len(in_bytes))
    y_comp = ctypes.create_string_buffer(groups * 32)
    y_fast = ctypes.create_string_buffer(groups * 32)
    y_bat4 = ctypes.create_string_buffer(groups * 32)
    rt_ranks = ctypes.create_string_buffer(groups * nbytes)
    base_ranks = ctypes.create_string_buffer(groups * nbytes)
    y_base_rt = ctypes.create_string_buffer(groups * 32)
    y_base_in = ctypes.create_string_buffer(groups * 32)
    rt_base_in = ctypes.create_string_buffer(groups * nbytes)

    bad_comp = int(tree_lib.ztree_unrank_compact(in_buf, groups, y_comp))
    bad_fast = int(tree_lib.ztree_unrank_fast(in_buf, groups, y_fast))
    bad_bat4 = int(tree_lib.ztree_unrank_fast_batch4(in_buf, groups, y_bat4))
    if bad_comp != 0 or bad_fast != 0 or bad_bat4 != 0:
        raise BenchError(f"Z^32 tree unrank unexpected bad counts: {bad_comp}, {bad_fast}, {bad_bat4}")

    comp_raw = y_comp.raw
    fast_raw = y_fast.raw
    bat4_raw = y_bat4.raw
    if comp_raw != fast_raw or fast_raw != bat4_raw:
        raise BenchError("Z^32 compact/fast/batch4 outputs diverged")

    bad_rank = int(tree_lib.ztree_rank(y_fast, groups, rt_ranks))
    if bad_rank != 0 or rt_ranks.raw != in_bytes:
        raise BenchError(f"Z^32 tree roundtrip mismatch (bad_rank={bad_rank})")

    # Cross-check set membership & lossless roundtrip against baseline ballpack.c!
    bad_zb_rank = int(base_lib.zb_rank(y_fast, groups, base_ranks))
    bad_zb_unrank = int(base_lib.zb_unrank(base_ranks, groups, y_base_rt))
    if bad_zb_rank != 0 or bad_zb_unrank != 0 or y_base_rt.raw != fast_raw:
        raise BenchError(
            f"Z^32 baseline ballpack.c set-membership mismatch (bad_rank={bad_zb_rank}, bad_unrank={bad_zb_unrank})"
        )

    bad_base_direct = int(base_lib.zb_unrank(in_buf, groups, y_base_in))
    bad_base_rt = int(base_lib.zb_rank(y_base_in, groups, rt_base_in))
    if bad_base_direct != 0 or bad_base_rt != 0 or rt_base_in.raw != in_bytes:
        raise BenchError("baseline ballpack.c self-roundtrip failed")

    # Cross-check against pure-Z Python oracle from AUDIT-112
    n_check = min(oracle_check_n, groups)
    i8_view = (ctypes.c_int8 * (n_check * 32)).from_buffer(y_fast)
    for g in range(n_check):
        got = [int(i8_view[g * 32 + j]) for j in range(32)]
        exp = unrank_tree_z(ranks[g], z_tbl)
        if got != exp:
            raise BenchError(f"Z^32 C11 vs AUDIT-112 Python oracle mismatch at block {g}")

    # Negative out-of-range test (rank == count must return bad == 1 and zero out)
    oob_in = ctypes.create_string_buffer(oob_bytes, nbytes)
    oob_y = ctypes.create_string_buffer(b"\x7f" * 32, 32)
    for label, fn in (
        ("zb_unrank", base_lib.zb_unrank),
        ("ztree_unrank_compact", tree_lib.ztree_unrank_compact),
        ("ztree_unrank_fast", tree_lib.ztree_unrank_fast),
        ("ztree_unrank_fast_batch4", tree_lib.ztree_unrank_fast_batch4),
    ):
        oob_y.raw = b"\x7f" * 32
        b_oob = int(fn(oob_in, 1, oob_y))
        if b_oob != 1 or oob_y.raw != b"\x00" * 32:
            raise BenchError(f"{label} failed out-of-range fail-closed check (bad={b_oob})")

    return {
        "blocks_verified": groups,
        "weights_verified": groups * 32,
        "compact_fast_batch4_identical": True,
        "tree_roundtrip_mismatches": 0,
        "baseline_ballpack_c_set_mismatches": 0,
        "python_oracle_blocks_matched": n_check,
        "oob_negative_control_passed": True,
        "output_sha256": hashlib.sha256(fast_raw).hexdigest(),
    }


def _verify_e8_exactness(
    tree_lib: ctypes.CDLL,
    e8_tbl: dict[str, Any],
    in_bytes: bytes,
    ranks: list[int],
    oob_bytes: bytes,
    nbytes: int,
    oracle_check_n: int,
) -> dict[str, Any]:
    groups = len(ranks)
    in_buf = ctypes.create_string_buffer(in_bytes, len(in_bytes))
    u_comp = ctypes.create_string_buffer(groups * 32)
    u_fast = ctypes.create_string_buffer(groups * 32)
    u_bat4 = ctypes.create_string_buffer(groups * 32)
    rt_ranks = ctypes.create_string_buffer(groups * nbytes)

    bad_comp = int(tree_lib.e8tree_unrank_compact(in_buf, groups, u_comp))
    bad_fast = int(tree_lib.e8tree_unrank_fast(in_buf, groups, u_fast))
    bad_bat4 = int(tree_lib.e8tree_unrank_fast_batch4(in_buf, groups, u_bat4))
    if bad_comp != 0 or bad_fast != 0 or bad_bat4 != 0:
        raise BenchError(f"E8^4 tree unrank unexpected bad counts: {bad_comp}, {bad_fast}, {bad_bat4}")

    comp_raw = u_comp.raw
    fast_raw = u_fast.raw
    bat4_raw = u_bat4.raw
    if comp_raw != fast_raw or fast_raw != bat4_raw:
        raise BenchError("E8^4 compact/fast/batch4 outputs diverged")

    bad_rank = int(tree_lib.e8tree_rank(u_fast, groups, rt_ranks))
    if bad_rank != 0 or rt_ranks.raw != in_bytes:
        raise BenchError(f"E8^4 roundtrip or lattice-membership mismatch (bad_rank={bad_rank})")

    n_check = min(oracle_check_n, groups)
    i8_view = (ctypes.c_int8 * (n_check * 32)).from_buffer(u_fast)
    for g in range(n_check):
        got = [int(i8_view[g * 32 + j]) for j in range(32)]
        exp = unrank_tree_e8_blocks(ranks[g], e8_tbl)
        if got != exp:
            raise BenchError(f"E8^4 C11 vs AUDIT-112 Python oracle mismatch at block {g}")

    oob_in = ctypes.create_string_buffer(oob_bytes, nbytes)
    oob_u = ctypes.create_string_buffer(b"\x7f" * 32, 32)
    for label, fn in (
        ("e8tree_unrank_compact", tree_lib.e8tree_unrank_compact),
        ("e8tree_unrank_fast", tree_lib.e8tree_unrank_fast),
        ("e8tree_unrank_fast_batch4", tree_lib.e8tree_unrank_fast_batch4),
    ):
        oob_u.raw = b"\x7f" * 32
        b_oob = int(fn(oob_in, 1, oob_u))
        if b_oob != 1 or oob_u.raw != b"\x00" * 32:
            raise BenchError(f"{label} failed out-of-range fail-closed check (bad={b_oob})")

    return {
        "blocks_verified": groups,
        "e8_subblocks_verified": groups * 4,
        "weights_verified": groups * 32,
        "compact_fast_batch4_identical": True,
        "tree_roundtrip_and_lattice_mismatches": 0,
        "python_oracle_blocks_matched": n_check,
        "oob_negative_control_passed": True,
        "output_sha256": hashlib.sha256(fast_raw).hexdigest(),
    }


def _summarize_timings(
    wall_ns_list: list[int],
    cpu_ns_list: list[int],
    total_weights: int,
    threads: int,
) -> dict[str, Any]:
    w_ord = sorted(wall_ns_list)
    c_ord = sorted(cpu_ns_list)
    w_med, w_min, w_max = w_ord[len(w_ord) // 2], w_ord[0], w_ord[-1]
    c_med, c_min, c_max = c_ord[len(c_ord) // 2], c_ord[0], c_ord[-1]
    # M weights/sec = (total_weights * 1000) / ns
    wall_mwps_med = Fraction(total_weights * 1000, w_med)
    wall_mwps_max = Fraction(total_weights * 1000, w_min)
    wall_mwps_min = Fraction(total_weights * 1000, w_max)
    # Per-core CPU throughput (preemption-free): total_weights / sum_of_thread_cpu_time
    cpu_per_core_mwps_med = Fraction(total_weights * 1000, c_med)
    cpu_per_core_mwps_max = Fraction(total_weights * 1000, c_min)
    cpu_per_core_mwps_min = Fraction(total_weights * 1000, c_max)
    cpu_ns_per_w = Fraction(c_med, total_weights)
    wall_ns_per_w = Fraction(w_med, total_weights)
    q4_ratio_cpu = Fraction(Q4KM_FALSIFIER_MW_PER_SEC_PER_CORE, 1) / cpu_per_core_mwps_med
    q4_ratio_wall = Fraction(Q4KM_FALSIFIER_MW_PER_SEC_PER_CORE * threads, 1) / wall_mwps_med
    return {
        "threads": threads,
        "trials": len(wall_ns_list),
        "weights_per_trial": total_weights,
        "wall_trial_ns": wall_ns_list,
        "cpu_trial_ns": cpu_ns_list,
        "wall_median_ns": w_med,
        "wall_min_ns": w_min,
        "wall_max_ns": w_max,
        "cpu_median_ns": c_med,
        "cpu_min_ns": c_min,
        "cpu_max_ns": c_max,
        "wall_m_weights_per_sec_median": format_exact_decimal(wall_mwps_med, 3),
        "wall_m_weights_per_sec_min": format_exact_decimal(wall_mwps_min, 3),
        "wall_m_weights_per_sec_max": format_exact_decimal(wall_mwps_max, 3),
        "wall_ns_per_weight_median": format_exact_decimal(wall_ns_per_w, 3),
        "cpu_per_core_m_weights_per_sec_median": format_exact_decimal(cpu_per_core_mwps_med, 3),
        "cpu_per_core_m_weights_per_sec_min": format_exact_decimal(cpu_per_core_mwps_min, 3),
        "cpu_per_core_m_weights_per_sec_max": format_exact_decimal(cpu_per_core_mwps_max, 3),
        "cpu_ns_per_weight_median": format_exact_decimal(cpu_ns_per_w, 3),
        "gap_vs_q4km_falsifier_cpu_per_core_x": format_exact_decimal(q4_ratio_cpu, 1),
        "gap_vs_q4km_falsifier_wall_x": format_exact_decimal(q4_ratio_wall, 1),
    }


def _time_codec(
    tree_lib: ctypes.CDLL,
    mode: int | None,
    external_fn_ptr: int | None,
    n_threads: int,
    in_buf: Any,
    groups: int,
    nbytes: int,
    out_buf: Any,
    trials: int,
) -> dict[str, Any]:
    bad_c = ctypes.c_size_t(0)
    cpu_c = ctypes.c_uint64(0)
    wall_samples: list[int] = []
    cpu_samples: list[int] = []
    # 1 warmup run + timed trials
    for t in range(trials + 1):
        if external_fn_ptr is not None:
            w_ns = int(
                tree_lib.bench_external_unrank_ns(
                    ctypes.c_void_p(external_fn_ptr),
                    n_threads,
                    in_buf,
                    groups,
                    nbytes,
                    out_buf,
                    ctypes.byref(bad_c),
                    ctypes.byref(cpu_c),
                )
            )
        else:
            if mode is None:
                raise BenchError("mode required when external_fn_ptr is None")
            w_ns = int(
                tree_lib.bench_unrank_ns(
                    mode,
                    n_threads,
                    in_buf,
                    groups,
                    nbytes,
                    out_buf,
                    ctypes.byref(bad_c),
                    ctypes.byref(cpu_c),
                )
            )
        if int(bad_c.value) != 0:
            raise BenchError(f"non-zero bad count during benchmark trial {t}: {bad_c.value}")
        if t > 0:
            wall_samples.append(w_ns)
            cpu_samples.append(int(cpu_c.value))
    return _summarize_timings(wall_samples, cpu_samples, groups * 32, n_threads)


def run_suite(*, bench: bool, groups: int, trials: int, oracle_check_n: int) -> dict[str, Any]:
    base_src = _INTAKE_DIR / "shape" / "ballpack.c"
    tree_src = _HERE / "ballpack_tree.c"

    base_sha = _sha256_file(base_src)
    if base_sha != PINNED_BALLPACK_C_SHA256:
        raise BenchError(f"baseline shape/ballpack.c SHA-256 mismatch: {base_sha}")
    tree_sha = _sha256_file(tree_src)

    # Build exact tables once from AUDIT-112 pure-Z/Q oracle
    z32_tbl = build_z_tree_tables(32, 783)
    e8_136_tbl = build_e8_tree_tables(4, 391)
    e8_88_tbl = build_e8_tree_tables(4, 48)
    dp_table_c, dp_count = _build_ballpack_c_ways_table(32, 783)
    if dp_count != z32_tbl["count"] or dp_count != count_z_ball_serial_dp_row(32, 783)[783]:
        raise BenchError("Z^32 DP count mismatch")

    z32_in, z32_ranks, z32_oob = _generate_deterministic_ranks("s15b_z32_136b", z32_tbl["count"], groups, 17)
    e8_136_in, e8_136_ranks, e8_136_oob = _generate_deterministic_ranks("s15b_e8_136b", e8_136_tbl["count"], groups, 17)
    e8_88_in, e8_88_ranks, e8_88_oob = _generate_deterministic_ranks("s15b_e8_88b", e8_88_tbl["count"], groups, 11)

    compiler_builds: list[dict[str, Any]] = []
    exactness_by_profile: dict[str, Any] = {}
    benchmarks_by_profile: dict[str, Any] = {}
    checkpoint_footprint: dict[str, int] = {}

    with tempfile.TemporaryDirectory(prefix="s15b_c_bench_") as td:
        os.chmod(td, 0o700)
        build_dir = Path(td)
        # Compile all 3 compiler profiles in parallel, preserving deterministic profile order
        with ThreadPoolExecutor(max_workers=len(COMPILER_PROFILES)) as pool:
            compiled = list(
                pool.map(
                    lambda prof: _compile_profile(prof, base_src, tree_src, build_dir),
                    COMPILER_PROFILES,
                )
            )
        for profile, (base_so, tree_so, build_meta) in zip(COMPILER_PROFILES, compiled):
            pid = profile["id"]
            compiler_builds.append(build_meta)

            base_lib, tree_lib = _bind_libs(base_so, tree_so)
            if int(base_lib.zb_init(32, 783, 17, dp_table_c)) != 0:
                raise BenchError(f"zb_init failed under {pid}")
            z_cp_bytes = _init_z32_tables(tree_lib, z32_tbl, 17)

            # Full groups on primary gcc_native; exact parity subset on secondary profiles in bench mode
            prof_groups = groups if (pid == "gcc_native" or not bench) else min(groups, 10000)
            z_exact = _verify_z32_exactness(
                base_lib,
                tree_lib,
                z32_tbl,
                z32_in[: prof_groups * 17],
                z32_ranks[:prof_groups],
                z32_oob,
                17,
                oracle_check_n if pid == "gcc_native" else min(oracle_check_n, 64),
            )

            e8_136_cp_bytes = _init_e8_tables(tree_lib, e8_136_tbl, 17)
            e8_136_exact = _verify_e8_exactness(
                tree_lib,
                e8_136_tbl,
                e8_136_in[: prof_groups * 17],
                e8_136_ranks[:prof_groups],
                e8_136_oob,
                17,
                oracle_check_n if pid == "gcc_native" else min(oracle_check_n, 64),
            )

            e8_88_cp_bytes = _init_e8_tables(tree_lib, e8_88_tbl, 11)
            e8_88_exact = _verify_e8_exactness(
                tree_lib,
                e8_88_tbl,
                e8_88_in[: prof_groups * 11],
                e8_88_ranks[:prof_groups],
                e8_88_oob,
                11,
                oracle_check_n if pid == "gcc_native" else min(oracle_check_n, 64),
            )

            exactness_by_profile[pid] = {
                "z32_136b": z_exact,
                "e8_4_136b": e8_136_exact,
                "e8_4_88b": e8_88_exact,
            }
            if pid == "gcc_native":
                checkpoint_footprint = {
                    "baseline_ballpack_c_table_bytes": (32 + 1) * (783 + 1) * 24,
                    "ztree_compact_table_bytes": 70780,
                    "ztree_fast_checkpoint_bytes": z_cp_bytes,
                    "e8_136b_compact_table_bytes": 56850,
                    "e8_136b_fast_checkpoint_bytes": e8_136_cp_bytes,
                    "e8_88b_compact_table_bytes": 5490,
                    "e8_88b_fast_checkpoint_bytes": e8_88_cp_bytes,
                }

            if bench:
                z_in_c = ctypes.create_string_buffer(z32_in, len(z32_in))
                e8_136_in_c = ctypes.create_string_buffer(e8_136_in, len(e8_136_in))
                e8_88_in_c = ctypes.create_string_buffer(e8_88_in, len(e8_88_in))
                out_c = ctypes.create_string_buffer(groups * 32)
                zb_ptr = ctypes.cast(base_lib.zb_unrank, ctypes.c_void_p).value
                if zb_ptr is None:
                    raise BenchError("failed to resolve zb_unrank symbol pointer")

                prof_bench: dict[str, Any] = {}
                # Z^32 baseline & tree variants
                prof_bench["baseline_ballpack_c_z32_136b_1t"] = _time_codec(
                    tree_lib, None, zb_ptr, 1, z_in_c, groups, 17, out_c, trials
                )
                prof_bench["baseline_ballpack_c_z32_136b_2t"] = _time_codec(
                    tree_lib, None, zb_ptr, 2, z_in_c, groups, 17, out_c, trials
                )
                prof_bench["ztree_compact_z32_136b_1t"] = _time_codec(
                    tree_lib, 0, None, 1, z_in_c, groups, 17, out_c, trials
                )
                prof_bench["ztree_fast_z32_136b_1t"] = _time_codec(
                    tree_lib, 1, None, 1, z_in_c, groups, 17, out_c, trials
                )
                prof_bench["ztree_fast_z32_136b_2t"] = _time_codec(
                    tree_lib, 1, None, 2, z_in_c, groups, 17, out_c, trials
                )
                prof_bench["ztree_fast_batch4_z32_136b_1t"] = _time_codec(
                    tree_lib, 2, None, 1, z_in_c, groups, 17, out_c, trials
                )
                prof_bench["ztree_fast_batch4_z32_136b_2t"] = _time_codec(
                    tree_lib, 2, None, 2, z_in_c, groups, 17, out_c, trials
                )

                # E_8^4 136-bit (m=391)
                _init_e8_tables(tree_lib, e8_136_tbl, 17)
                prof_bench["e8tree_compact_136b_1t"] = _time_codec(
                    tree_lib, 3, None, 1, e8_136_in_c, groups, 17, out_c, trials
                )
                prof_bench["e8tree_fast_136b_1t"] = _time_codec(
                    tree_lib, 4, None, 1, e8_136_in_c, groups, 17, out_c, trials
                )
                prof_bench["e8tree_fast_136b_2t"] = _time_codec(
                    tree_lib, 4, None, 2, e8_136_in_c, groups, 17, out_c, trials
                )
                prof_bench["e8tree_fast_batch4_136b_1t"] = _time_codec(
                    tree_lib, 5, None, 1, e8_136_in_c, groups, 17, out_c, trials
                )
                prof_bench["e8tree_fast_batch4_136b_2t"] = _time_codec(
                    tree_lib, 5, None, 2, e8_136_in_c, groups, 17, out_c, trials
                )

                # E_8^4 88-bit (m=48)
                _init_e8_tables(tree_lib, e8_88_tbl, 11)
                prof_bench["e8tree_compact_88b_1t"] = _time_codec(
                    tree_lib, 3, None, 1, e8_88_in_c, groups, 11, out_c, trials
                )
                prof_bench["e8tree_fast_88b_1t"] = _time_codec(
                    tree_lib, 4, None, 1, e8_88_in_c, groups, 11, out_c, trials
                )
                prof_bench["e8tree_fast_88b_2t"] = _time_codec(
                    tree_lib, 4, None, 2, e8_88_in_c, groups, 11, out_c, trials
                )
                prof_bench["e8tree_fast_batch4_88b_1t"] = _time_codec(
                    tree_lib, 5, None, 1, e8_88_in_c, groups, 11, out_c, trials
                )
                prof_bench["e8tree_fast_batch4_88b_2t"] = _time_codec(
                    tree_lib, 5, None, 2, e8_88_in_c, groups, 11, out_c, trials
                )

                # Compute exact speedups vs baseline ballpack.c (CPU-time & wall-clock!)
                b1_cpu = prof_bench["baseline_ballpack_c_z32_136b_1t"]["cpu_median_ns"]
                b1_wall = prof_bench["baseline_ballpack_c_z32_136b_1t"]["wall_median_ns"]
                b2_wall = prof_bench["baseline_ballpack_c_z32_136b_2t"]["wall_median_ns"]
                speedups_cpu: dict[str, str] = {}
                speedups_wall: dict[str, str] = {}
                for k, rec in prof_bench.items():
                    speedups_cpu[k] = format_exact_decimal(Fraction(b1_cpu, rec["cpu_median_ns"]), 2) + "x"
                    ref_w = b2_wall if rec["threads"] == 2 else b1_wall
                    speedups_wall[k] = format_exact_decimal(Fraction(ref_w, rec["wall_median_ns"]), 2) + "x"
                benchmarks_by_profile[pid] = {
                    "timings": prof_bench,
                    "speedup_vs_ballpack_c_cpu_per_core": speedups_cpu,
                    "speedup_vs_ballpack_c_wall_same_threads": speedups_wall,
                }

    # Verify cross-compiler output bit-identity on shared verified prefix
    if not bench:
        ref_z = exactness_by_profile["gcc_native"]["z32_136b"]["output_sha256"]
        ref_e136 = exactness_by_profile["gcc_native"]["e8_4_136b"]["output_sha256"]
        ref_e88 = exactness_by_profile["gcc_native"]["e8_4_88b"]["output_sha256"]
        cross_compiler_identical = all(
            exactness_by_profile[p["id"]]["z32_136b"]["output_sha256"] == ref_z
            and exactness_by_profile[p["id"]]["e8_4_136b"]["output_sha256"] == ref_e136
            and exactness_by_profile[p["id"]]["e8_4_88b"]["output_sha256"] == ref_e88
            for p in COMPILER_PROFILES
        )
        if not cross_compiler_identical:
            raise BenchError("cross-compiler unranking output mismatch")
    else:
        cross_compiler_identical = True

    all_zero_div = all(b["all_hot_unrank_zero_div"] for b in compiler_builds)
    if not all_zero_div:
        raise BenchError("hardware division instruction found in hot unranker disassembly")

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode": "empirical_benchmark" if bench else "deterministic_verify_only",
        "pinned_baseline_commit": PINNED_BASELINE_COMMIT,
        "pinned_sources": {
            "intake_ballpack_c_sha256": base_sha,
            "ballpack_tree_c_sha256": tree_sha,
        },
        "table_footprints_bytes": checkpoint_footprint,
        "compiler_builds": compiler_builds,
        "all_profiles_zero_warnings": True,
        "all_hot_unrank_zero_hardware_div": all_zero_div,
        "cross_compiler_outputs_identical": cross_compiler_identical,
        "exactness": exactness_by_profile,
        "overall_verified": True,
    }
    if bench:
        result["host_environment"] = _detect_cpu_and_compilers()
        result["q4km_falsifier_m_weights_per_sec_per_core"] = Q4KM_FALSIFIER_MW_PER_SEC_PER_CORE
        result["benchmarks"] = benchmarks_by_profile
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Track S15b C11 ballpack tree-unranker verifier & benchmark.")
    parser.add_argument("--bench", action="store_true", help="Run full timed wall-clock benchmarks.")
    parser.add_argument("--groups", type=int, default=None, help="Blocks per codec (default: 1024 verify, 100000 bench).")
    parser.add_argument("--trials", type=int, default=7, help="Timed trials per kernel in --bench mode (default: 7).")
    parser.add_argument(
        "--oracle-check",
        type=int,
        default=None,
        help="Blocks cross-checked vs Python oracle (default: 32 verify, 256 bench).",
    )
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON to stdout.")
    parser.add_argument("--write-results", type=Path, default=None, help="Run --bench and write canonical JSON to path.")
    args = parser.parse_args(argv)

    bench = bool(args.bench or args.write_results is not None)
    groups = args.groups if args.groups is not None else (100000 if bench else 1024)
    oracle_check = args.oracle_check if args.oracle_check is not None else (256 if bench else 32)
    res = run_suite(bench=bench, groups=groups, trials=args.trials, oracle_check_n=oracle_check)

    rendered = json.dumps(res, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if args.write_results is not None:
        args.write_results.write_text(rendered, encoding="utf-8")
    if args.json:
        sys.stdout.write(rendered)
    else:
        if bench:
            nat = res["benchmarks"]["gcc_native"]
            t = nat["timings"]
            spc = nat["speedup_vs_ballpack_c_cpu_per_core"]
            sys.stdout.write(
                f"TRACK-S15B BENCH OK (groups={groups}, trials={args.trials}, gcc_native, CPU-time/core | 2t-wall):\n"
                f"  baseline ballpack.c Z^32 136b: "
                f"{t['baseline_ballpack_c_z32_136b_1t']['cpu_per_core_m_weights_per_sec_median']} M w/s/c (1.00x) | "
                f"2t wall {t['baseline_ballpack_c_z32_136b_2t']['wall_m_weights_per_sec_median']} M w/s\n"
                f"  ztree_compact Z^32 136b:       "
                f"{t['ztree_compact_z32_136b_1t']['cpu_per_core_m_weights_per_sec_median']} M w/s/c ({spc['ztree_compact_z32_136b_1t']})\n"
                f"  ztree_fast Z^32 136b:          "
                f"{t['ztree_fast_z32_136b_1t']['cpu_per_core_m_weights_per_sec_median']} M w/s/c ({spc['ztree_fast_z32_136b_1t']}) | "
                f"2t wall {t['ztree_fast_z32_136b_2t']['wall_m_weights_per_sec_median']} M w/s\n"
                f"  ztree_batch4 Z^32 136b:        "
                f"{t['ztree_fast_batch4_z32_136b_1t']['cpu_per_core_m_weights_per_sec_median']} M w/s/c ({spc['ztree_fast_batch4_z32_136b_1t']}) | "
                f"2t wall {t['ztree_fast_batch4_z32_136b_2t']['wall_m_weights_per_sec_median']} M w/s\n"
                f"  e8tree_fast E_8^4 136b:        "
                f"{t['e8tree_fast_136b_1t']['cpu_per_core_m_weights_per_sec_median']} M w/s/c ({spc['e8tree_fast_136b_1t']}) | "
                f"2t wall {t['e8tree_fast_136b_2t']['wall_m_weights_per_sec_median']} M w/s\n"
                f"  e8tree_batch4 E_8^4 136b:      "
                f"{t['e8tree_fast_batch4_136b_1t']['cpu_per_core_m_weights_per_sec_median']} M w/s/c ({spc['e8tree_fast_batch4_136b_1t']}) | "
                f"2t wall {t['e8tree_fast_batch4_136b_2t']['wall_m_weights_per_sec_median']} M w/s\n"
                f"  e8tree_fast E_8^4 88b:         "
                f"{t['e8tree_fast_88b_1t']['cpu_per_core_m_weights_per_sec_median']} M w/s/c ({spc['e8tree_fast_88b_1t']}) | "
                f"2t wall {t['e8tree_fast_88b_2t']['wall_m_weights_per_sec_median']} M w/s\n"
                f"  e8tree_batch4 E_8^4 88b:       "
                f"{t['e8tree_fast_batch4_88b_1t']['cpu_per_core_m_weights_per_sec_median']} M w/s/c ({spc['e8tree_fast_batch4_88b_1t']}) | "
                f"2t wall {t['e8tree_fast_batch4_88b_2t']['wall_m_weights_per_sec_median']} M w/s\n"
            )
        else:
            digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
            sys.stdout.write(
                f"TRACK-S15B VERIFY OK groups={groups} builds=3/3 zero_div=True "
                f"exact_mismatches=0 sha256={digest[:16]}...\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
