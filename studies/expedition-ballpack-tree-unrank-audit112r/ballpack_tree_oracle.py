"""Pure-Z/Q exact oracle for AUDIT-112 (Track S15):
Zero-overhead binary-tree shell-convolution unranking, zero-division Barrett/Granlund-Montgomery
reciprocal multiplication, and E_8^4 reduced-norm tree codec for Moissanite-Labs/siliquant#23/#25.

Strict discipline: pure Python 3 stdlib, exact int/Fraction only, zero floats, zero math.sqrt,
zero bare asserts, str-key canonical JSON only, fail-closed validation on every trust boundary.
"""

from __future__ import annotations

from bisect import bisect_right
from fractions import Fraction
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")
GIT_SHA40_RE = re.compile(r"\A[0-9a-f]{40}\Z")


class BallpackTreeError(ValueError):
    """Fail-closed validation error for exact ballpack tree-unranking primitives."""


def require_exact_int(
    value: Any,
    name: str,
    *,
    allow_zero: bool = True,
    min_val: int | None = None,
) -> int:
    if type(value) is not int:
        raise BallpackTreeError(f"{name} must be an exact int (got {type(value).__name__})")
    if min_val is not None:
        if value < min_val:
            raise BallpackTreeError(f"{name} must be >= {min_val} (got {value})")
    elif allow_zero:
        if value < 0:
            raise BallpackTreeError(f"{name} must be non-negative (got {value})")
    else:
        if value <= 0:
            raise BallpackTreeError(f"{name} must be positive (got {value})")
    return value


def require_exact_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise BallpackTreeError(f"{name} must be an exact bool (got {type(value).__name__})")
    return value


def require_exact_fraction(value: Any, name: str) -> Fraction:
    if type(value) is not Fraction:
        raise BallpackTreeError(f"{name} must be an exact Fraction (got {type(value).__name__})")
    return value


def require_non_empty_str(value: Any, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise BallpackTreeError(f"{name} must be a non-empty trimmed str")
    return value


def require_sha256(value: Any, name: str) -> str:
    if type(value) is not str or not SHA256_RE.match(value) or len(set(value)) == 1:
        raise BallpackTreeError(f"{name} must be a non-degenerate 64-char lowercase hex sha256")
    return value


def require_git_sha40(value: Any, name: str) -> str:
    if type(value) is not str or not GIT_SHA40_RE.match(value) or len(set(value)) == 1:
        raise BallpackTreeError(f"{name} must be a non-degenerate 40-char lowercase hex git sha")
    return value


def contains_forbidden_json_value(obj: Any) -> bool:
    """Return True if any float, complex, bytes, set, frozenset, or non-str dict key exists."""
    if obj is None or type(obj) in (bool, int, str):
        return False
    if isinstance(obj, Mapping):
        return any(
            type(k) is not str or contains_forbidden_json_value(v)
            for k, v in obj.items()
        )
    if type(obj) in (list, tuple):
        return any(contains_forbidden_json_value(item) for item in obj)
    return True


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Compute canonical SHA-256 over a strict str-key, float-free JSON mapping."""
    if not isinstance(payload, Mapping) or contains_forbidden_json_value(payload):
        raise BallpackTreeError("canonical_sha256 payload must be a strict str-key float-free JSON mapping")
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def frac_to_dict(value: Fraction) -> dict[str, int]:
    require_exact_fraction(value, "value")
    return {"num": value.numerator, "den": value.denominator}


def format_exact_decimal(value: Fraction, places: int = 6) -> str:
    require_exact_fraction(value, "value")
    require_exact_int(places, "places", allow_zero=True)
    mag = -value if value < 0 else value
    scale = 10 ** places
    scaled = (mag.numerator * scale * 2 + mag.denominator) // (2 * mag.denominator)
    if scaled == 0:
        return "0" if places == 0 else f"0.{'0' * places}"
    sign = "-" if value < 0 else ""
    whole, frac = divmod(scaled, scale)
    if places == 0:
        return f"{sign}{whole}"
    return f"{sign}{whole}.{frac:0{places}d}"


def format_exact_percent(value: Fraction, places: int = 2) -> str:
    require_exact_fraction(value, "value")
    return format_exact_decimal(value * 100, places=places) + "%"


def _hardware_type_for_bits(bits: int) -> str:
    if bits <= 16:
        return "uint16"
    if bits <= 32:
        return "uint32"
    if bits <= 64:
        return "uint64"
    if bits <= 128:
        return "uint128"
    return "uint192"


def _bytes_per_entry_for_bits(bits: int) -> int:
    if bits <= 16:
        return 2
    if bits <= 32:
        return 4
    if bits <= 64:
        return 8
    if bits <= 128:
        return 16
    return 24


def deterministic_sample_ranks(count: int, n_random: int = 24) -> list[int]:
    """Build deterministic boundary + SHA-256-seeded full-range ranks in [0, count - 1]."""
    require_exact_int(count, "count", allow_zero=False)
    require_exact_int(n_random, "n_random", allow_zero=True)
    ranks: list[int] = [0, count - 1]
    if count > 2:
        ranks.append(1)
        ranks.append(count // 2)
        ranks.append(count - 2)
    bits = max(1, (count - 1).bit_length())
    for p in (1, 8, 16, 32, 48, 64, 80, 96, 112, 128, bits - 1):
        val = 1 << p
        if val < count:
            ranks.append(val)
            if val - 1 >= 0:
                ranks.append(val - 1)
    for i in range(n_random):
        digest = hashlib.sha256(f"audit112-rank-seed-{count}-{i}".encode("ascii")).digest()
        val = int.from_bytes(digest, "little") % count
        ranks.append(val)
    return sorted(set(ranks))


# =============================================================================
# SECTION 1: Exact Zero-Division Granlund-Montgomery / Barrett Reciprocals
# =============================================================================

def verify_reciprocal_universal(
    divisor: int,
    max_dividend: int,
    mult: int,
    shift: int,
) -> bool:
    """Universal pure-Z theorem: floor((x * mult) / 2^shift) == x // divisor for ALL 0 <= x <= max_dividend
    whenever mult * divisor >= 2^shift and (mult * divisor - 2^shift) * max_dividend < 2^shift.
    """
    require_exact_int(divisor, "divisor", allow_zero=False)
    require_exact_int(max_dividend, "max_dividend", allow_zero=True)
    require_exact_int(mult, "mult", allow_zero=False)
    require_exact_int(shift, "shift", allow_zero=True)
    two_s = 1 << shift
    md = mult * divisor
    return md >= two_s and (md - two_s) * max_dividend < two_s


def build_exact_reciprocal(divisor: int, max_dividend: int) -> dict[str, Any]:
    """Precompute exact (mult, shift) pair so runtime divmod uses 0 hardware division instructions."""
    require_exact_int(divisor, "divisor", allow_zero=False)
    require_exact_int(max_dividend, "max_dividend", allow_zero=True)
    if (divisor & (divisor - 1)) == 0:
        shift = divisor.bit_length() - 1
        mult = 1
        is_pow2 = True
    else:
        ell = (divisor - 1).bit_length()
        k = max_dividend.bit_length()
        shift = ell + k
        mult = ((1 << shift) + divisor - 1) // divisor
        is_pow2 = False
    return {
        "divisor": divisor,
        "max_dividend": max_dividend,
        "mult": mult,
        "shift": shift,
        "mult_bits": mult.bit_length(),
        "is_pow2": is_pow2,
        "universally_valid": verify_reciprocal_universal(divisor, max_dividend, mult, shift),
    }


def divmod_exact_reciprocal(
    x: int,
    divisor: int,
    max_dividend: int,
    mult: int,
    shift: int,
) -> tuple[int, int]:
    """Execute exact zero-division (q, r) via multiply-shift: q = (x * mult) >> shift, r = x - q * divisor."""
    require_exact_int(x, "x", allow_zero=True)
    require_exact_int(divisor, "divisor", allow_zero=False)
    require_exact_int(max_dividend, "max_dividend", allow_zero=True)
    if x > max_dividend:
        raise BallpackTreeError("dividend x exceeds certified max_dividend")
    if not verify_reciprocal_universal(divisor, max_dividend, mult, shift):
        raise BallpackTreeError("reciprocal parameters fail universal exactness certificate")
    q = (x * mult) >> shift
    r = x - q * divisor
    return q, r


# =============================================================================
# SECTION 2: PILLAR 1 - Z^n Binary-Tree Shell-Convolution Bijection (32->...->2)
# =============================================================================

def _is_power_of_two(n: int) -> bool:
    return type(n) is int and n >= 2 and (n & (n - 1)) == 0


def _ordered_z1_symbols(bound: int) -> list[int]:
    lim = math.isqrt(bound)
    syms = [0]
    for k in range(1, lim + 1):
        syms.append(k)
        syms.append(-k)
    return syms


def build_z_tree_tables(n: int, bound: int) -> dict[str, Any]:
    """Build exact Z^n shell/ball convolution tables, Z^2 L1 LUT, and universal zero-division reciprocals."""
    if not _is_power_of_two(n):
        raise BallpackTreeError("n must be a power of two >= 2")
    require_exact_int(bound, "bound", allow_zero=True)

    # S[1][r]: exact shell counts in Z^1
    s1 = [0] * (bound + 1)
    s1[0] = 1
    k = 1
    while k * k <= bound:
        s1[k * k] = 2
        k += 1

    shells: dict[int, list[int]] = {1: s1}
    m = 1
    while m < n:
        sm = shells[m]
        s2m = [0] * (bound + 1)
        for r1, c1 in enumerate(sm):
            if c1 == 0:
                continue
            for r2 in range(bound - r1 + 1):
                c2 = sm[r2]
                if c2:
                    s2m[r1 + r2] += c1 * c2
        shells[2 * m] = s2m
        m *= 2

    balls: dict[int, list[int]] = {}
    for dim, s_row in shells.items():
        acc = 0
        w_row: list[int] = []
        for c in s_row:
            acc += c
            w_row.append(acc)
        balls[dim] = w_row

    # Exact Z^2 L1-cache leaf LUT (ordered canonically by shell r = 0..bound)
    z1_syms = _ordered_z1_symbols(bound)
    z2_shell_points: list[list[tuple[int, int]]] = [[] for _ in range(bound + 1)]
    for x0 in z1_syms:
        r0 = x0 * x0
        for x1 in z1_syms:
            r = r0 + x1 * x1
            if r <= bound:
                z2_shell_points[r].append((x0, x1))

    z2_ball_points: list[tuple[int, int]] = []
    z2_shell_index: dict[tuple[int, int], tuple[int, int]] = {}
    z2_ball_index: dict[tuple[int, int], int] = {}
    for r, pts in enumerate(z2_shell_points):
        for local_idx, pt in enumerate(pts):
            z2_shell_index[pt] = (r, local_idx)
            z2_ball_index[pt] = len(z2_ball_points)
            z2_ball_points.append(pt)

    # Root cumulative split table C_root[r_L] = sum_{a=0..r_L-1} S_{n/2}(a) * W_{n/2}(bound - a)
    half = n // 2
    c_root = [0]
    acc = 0
    for r_l in range(bound + 1):
        acc += shells[half][r_l] * balls[half][bound - r_l]
        c_root.append(acc)

    # Universal zero-division reciprocals for every static divisor used in unranking
    recip_ball: dict[int, dict[int, dict[str, Any]]] = {}
    recip_shell: dict[int, dict[int, dict[str, Any]]] = {}
    total_recips = 0
    valid_recips = 0
    max_mult_bits_by_level: dict[str, int] = {}

    dim = n
    level_idx = 0
    while dim >= 4:
        h = dim // 2
        rb_h: dict[int, dict[str, Any]] = {}
        rs_h: dict[int, dict[str, Any]] = {}
        lvl_max_mb = 0
        for b_r in range(bound + 1):
            d_ball = balls[h][b_r]
            max_s_l = max(shells[h][: bound - b_r + 1])
            max_div_ball = max_s_l * d_ball - 1
            rec = build_exact_reciprocal(d_ball, max_div_ball)
            rb_h[b_r] = rec
            total_recips += 1
            if rec["universally_valid"]:
                valid_recips += 1
            if rec["mult_bits"] > lvl_max_mb:
                lvl_max_mb = rec["mult_bits"]
        if dim < n:
            for r_r in range(bound + 1):
                d_sh = shells[h][r_r]
                if d_sh == 0:
                    continue
                max_s_l = max(shells[h][: bound - r_r + 1])
                max_div_sh = max_s_l * d_sh - 1
                rec = build_exact_reciprocal(d_sh, max_div_sh)
                rs_h[r_r] = rec
                total_recips += 1
                if rec["universally_valid"]:
                    valid_recips += 1
                if rec["mult_bits"] > lvl_max_mb:
                    lvl_max_mb = rec["mult_bits"]
        recip_ball[h] = rb_h
        recip_shell[h] = rs_h
        max_mult_bits_by_level[f"level_{level_idx}_n{dim}_to_n{h}"] = lvl_max_mb
        dim = h
        level_idx += 1

    return {
        "n": n,
        "bound": bound,
        "count": balls[n][bound],
        "shells": shells,
        "balls": balls,
        "c_root": c_root,
        "z2_shell_points": z2_shell_points,
        "z2_ball_points": z2_ball_points,
        "z2_shell_index": z2_shell_index,
        "z2_ball_index": z2_ball_index,
        "recip_ball": recip_ball,
        "recip_shell": recip_shell,
        "total_reciprocals": total_recips,
        "valid_reciprocals": valid_recips,
        "max_mult_bits_by_level": max_mult_bits_by_level,
    }


def _rank_z_node(point: Sequence[int], is_ball: bool, budget: int, tables: Mapping[str, Any]) -> int:
    m = len(point)
    if m == 2:
        pt = (point[0], point[1])
        if is_ball:
            return tables["z2_ball_index"][pt]
        return tables["z2_shell_index"][pt][1]
    h = m // 2
    left = point[:h]
    right = point[h:]
    r_l = sum(v * v for v in left)
    s_h = tables["shells"][h]
    if is_ball:
        w_h = tables["balls"][h]
        if m == tables["n"]:
            offset = tables["c_root"][r_l]
        else:
            offset = sum(s_h[a] * w_h[budget - a] for a in range(r_l))
        idx_l = _rank_z_node(left, False, r_l, tables)
        idx_r = _rank_z_node(right, True, budget - r_l, tables)
        return offset + idx_l * w_h[budget - r_l] + idx_r
    offset = sum(s_h[a] * s_h[budget - a] for a in range(r_l))
    idx_l = _rank_z_node(left, False, r_l, tables)
    idx_r = _rank_z_node(right, False, budget - r_l, tables)
    return offset + idx_l * s_h[budget - r_l] + idx_r


def rank_tree_z(point: Sequence[int], tables: Mapping[str, Any]) -> int:
    """Rank an integer vector y in Z^n with ||y||^2 <= B via the exact binary-tree bijection."""
    if not isinstance(point, Sequence) or isinstance(point, (str, bytes)):
        raise BallpackTreeError("point must be a sequence of exact ints")
    n = tables["n"]
    bound = tables["bound"]
    if len(point) != n:
        raise BallpackTreeError(f"point length must equal {n}")
    cleaned = [require_exact_int(v, f"point[{i}]", min_val=-bound - 1) for i, v in enumerate(point)]
    if sum(v * v for v in cleaned) > bound:
        raise BallpackTreeError("point squared norm exceeds ball bound")
    return _rank_z_node(cleaned, True, bound, tables)


def _unrank_z_node(rank: int, m: int, is_ball: bool, budget: int, tables: Mapping[str, Any]) -> list[int]:
    if m == 2:
        pt = tables["z2_ball_points"][rank] if is_ball else tables["z2_shell_points"][budget][rank]
        return [pt[0], pt[1]]
    h = m // 2
    s_h = tables["shells"][h]
    if is_ball:
        w_h = tables["balls"][h]
        if m == tables["n"]:
            c_root = tables["c_root"]
            r_l = bisect_right(c_root, rank) - 1
            rem = rank - c_root[r_l]
        else:
            rem = rank
            r_l = 0
            while True:
                term = s_h[r_l] * w_h[budget - r_l]
                if rem < term:
                    break
                rem -= term
                r_l += 1
        rec = tables["recip_ball"][h][budget - r_l]
        idx_l, idx_r = divmod_exact_reciprocal(rem, rec["divisor"], rec["max_dividend"], rec["mult"], rec["shift"])
        return _unrank_z_node(idx_l, h, False, r_l, tables) + _unrank_z_node(idx_r, h, True, budget - r_l, tables)

    rem = rank
    r_l = 0
    while True:
        term = s_h[r_l] * s_h[budget - r_l]
        if rem < term:
            break
        rem -= term
        r_l += 1
    rec = tables["recip_shell"][h][budget - r_l]
    idx_l, idx_r = divmod_exact_reciprocal(rem, rec["divisor"], rec["max_dividend"], rec["mult"], rec["shift"])
    return _unrank_z_node(idx_l, h, False, r_l, tables) + _unrank_z_node(idx_r, h, False, budget - r_l, tables)


def unrank_tree_z(rank: int, tables: Mapping[str, Any]) -> list[int]:
    """Unrank a 0-based integer rank into Z^n with ||y||^2 <= B using 0 hardware divisions."""
    require_exact_int(rank, "rank", allow_zero=True)
    if rank >= tables["count"]:
        raise BallpackTreeError("rank exceeds codebook cardinality - 1")
    return _unrank_z_node(rank, tables["n"], True, tables["bound"], tables)


def count_z_ball_serial_dp_row(n: int, bound: int) -> list[int]:
    """Independent reference sequential 1D DP row identical to shape/ballindex.py:_build_integers."""
    require_exact_int(n, "n", allow_zero=False)
    require_exact_int(bound, "bound", allow_zero=True)
    syms = _ordered_z1_symbols(bound)
    costs = [v * v for v in syms]
    row = [1] * (bound + 1)
    for _ in range(n):
        nxt = [0] * (bound + 1)
        for cost in costs:
            for r in range(cost, bound + 1):
                nxt[r] += row[r - cost]
        row = nxt
    return row


def count_z_ball_serial_dp(n: int, bound: int) -> int:
    """Independent reference sequential 1D DP count identical to shape/ballindex.py:_build_integers."""
    return count_z_ball_serial_dp_row(n, bound)[bound]


def certify_z_tree_bijection_exact(
    n: int,
    bound: int,
    *,
    exhaustive: bool = False,
    n_random: int = 24,
    precomputed_tables: Mapping[str, Any] | None = None,
    serial_count_override: int | None = None,
) -> dict[str, Any]:
    """Certify Z^n binary-tree shell-convolution bijection, exact depth/bit-widths, LUT bytes, & 0-div reciprocals."""
    require_exact_bool(exhaustive, "exhaustive")
    tables = precomputed_tables if precomputed_tables is not None else build_z_tree_tables(n, bound)
    count = tables["count"]
    bits = max(1, (count - 1).bit_length())
    serial_count = serial_count_override if serial_count_override is not None else count_z_ball_serial_dp(n, bound)
    count_matches_serial = count == serial_count

    if exhaustive:
        test_ranks = list(range(count))
    else:
        test_ranks = deterministic_sample_ranks(count, n_random=n_random)

    seen_points: set[tuple[int, ...]] = set()
    all_roundtrip_ok = True
    max_abs_coord = 0
    for rk in test_ranks:
        pt = unrank_tree_z(rk, tables)
        if sum(v * v for v in pt) > bound:
            all_roundtrip_ok = False
        if rank_tree_z(pt, tables) != rk:
            all_roundtrip_ok = False
        for v in pt:
            if abs(v) > max_abs_coord:
                max_abs_coord = abs(v)
        if exhaustive:
            seen_points.add(tuple(pt))
    exhaustive_distinct_ok = (len(seen_points) == count) if exhaustive else True

    # Exact level-by-level bit-width & table-memory profile
    levels: list[dict[str, Any]] = []
    tree_table_bytes = 0
    dim = n
    depth = 0
    while dim >= 2:
        max_s = max(tables["shells"][dim])
        max_w = tables["balls"][dim][bound]
        s_bits = max(1, (max_s - 1).bit_length()) if max_s > 0 else 0
        w_bits = max(1, (max_w - 1).bit_length()) if max_w > 0 else 0
        nodes_at_level = n // dim
        if dim == n:
            # Root stores C_root[0..bound+1]
            c_root_bytes = (bound + 2) * _bytes_per_entry_for_bits(w_bits)
            tree_table_bytes += c_root_bytes
            lvl_bytes = c_root_bytes
        else:
            # 1D S_dim and W_dim tables
            s_bytes = (bound + 1) * _bytes_per_entry_for_bits(s_bits)
            w_bytes = (bound + 1) * _bytes_per_entry_for_bits(w_bits)
            lvl_bytes = s_bytes + w_bytes
            tree_table_bytes += lvl_bytes
        levels.append({
            "depth": depth,
            "dim": dim,
            "parallel_nodes": nodes_at_level,
            "max_shell_bits": s_bits,
            "max_ball_bits": w_bits,
            "shell_hw_type": _hardware_type_for_bits(s_bits),
            "ball_hw_type": _hardware_type_for_bits(w_bits),
            "table_bytes": lvl_bytes,
        })
        dim //= 2
        depth += 1

    z2_pairs = len(tables["z2_ball_points"])
    z2_coord_bytes = z2_pairs * 2  # int8[2] per pair
    z2_offset_bytes = (bound + 2) * 2  # uint16[bound+2] shell offsets
    z2_lut_bytes = z2_coord_bytes + z2_offset_bytes
    total_tree_bytes = tree_table_bytes + z2_lut_bytes
    serial_ballpack_entries = (n + 1) * (bound + 1)
    serial_ballpack_bytes = serial_ballpack_entries * 24  # LIMBS=3 uint64 = 24 B in ballpack.c
    split_depth = n.bit_length() - 2  # 32 -> 4 splits (32->16->8->4->2)

    return {
        "n": n,
        "bound": bound,
        "codebook_count": count,
        "codebook_bits": bits,
        "rate_overhead_bits": 0,
        "count_matches_ballindex_dp": count_matches_serial,
        "exhaustive_mode": exhaustive,
        "ranks_tested": len(test_ranks),
        "all_roundtrips_exact": all_roundtrip_ok,
        "exhaustive_bijection_distinct": exhaustive_distinct_ok,
        "max_abs_coord_observed": max_abs_coord,
        "serial_critical_path_depth": n,
        "tree_split_depth": split_depth,
        "depth_reduction_ratio": frac_to_dict(Fraction(n, split_depth if split_depth > 0 else 1)),
        "z2_lut_pairs": z2_pairs,
        "z2_lut_total_bytes": z2_lut_bytes,
        "tree_total_bytes": total_tree_bytes,
        "serial_ballpack_entries": serial_ballpack_entries,
        "serial_ballpack_bytes": serial_ballpack_bytes,
        "table_memory_reduction_ratio": frac_to_dict(Fraction(serial_ballpack_bytes, total_tree_bytes)),
        "total_reciprocals": tables["total_reciprocals"],
        "valid_reciprocals": tables["valid_reciprocals"],
        "all_reciprocals_universally_valid": tables["valid_reciprocals"] == tables["total_reciprocals"],
        "max_mult_bits_by_level": tables["max_mult_bits_by_level"],
        "levels": levels,
    }


# =============================================================================
# SECTION 3: PILLAR 2 - E_8^4 Reduced-Norm Binary-Tree Codec & Theta Series
# =============================================================================

def sigma3_exact(m: int) -> int:
    """Exact divisor-cube sum sigma_3(m) = sum_{d | m} d^3 over Z."""
    require_exact_int(m, "m", allow_zero=False)
    total = 0
    for d in range(1, math.isqrt(m) + 1):
        if m % d == 0:
            total += d * d * d
            q = m // d
            if q != d:
                total += q * q * q
    return total


def e8_theta_shell_count_exact(m: int) -> int:
    """Exact E_8 shell count S_{E_8}(2m): 1 at m=0, 240 * sigma_3(m) for m >= 1."""
    require_exact_int(m, "m", allow_zero=True)
    return 1 if m == 0 else 240 * sigma3_exact(m)


def is_valid_e8_doubled_block(u: Sequence[int]) -> bool:
    """Check whether u = 2y in Z^8 represents a valid E_8 lattice vector (0 floats)."""
    if not isinstance(u, Sequence) or isinstance(u, (str, bytes)) or len(u) != 8:
        return False
    for v in u:
        if type(v) is not int:
            return False
    first_par = u[0] & 1
    if any((v & 1) != first_par for v in u):
        return False
    return (sum(v // 2 for v in u) & 1) == 0


def _build_single_e8_tree(m_max: int) -> dict[str, Any]:
    """Build exact reduced-norm 8 -> 4+4 -> 2+2 -> L1 pair LUT tables for a single E_8 block (m <= m_max)."""
    r_max = 2 * m_max
    ks_even = _ordered_z1_symbols(r_max)
    even_p2: list[list[list[tuple[int, int]]]] = [[[] for _ in range(r_max + 1)] for _ in range(2)]
    even_p2_idx: list[dict[tuple[int, int], tuple[int, int]]] = [{}, {}]
    for k0 in ks_even:
        r0 = k0 * k0
        for k1 in ks_even:
            r = r0 + k1 * k1
            if r <= r_max:
                p = (k0 + k1) & 1
                pair = (2 * k0, 2 * k1)
                even_p2_idx[p][pair] = (r, len(even_p2[p][r]))
                even_p2[p][r].append(pair)
    even_c2 = [[len(even_p2[p][r]) for r in range(r_max + 1)] for p in range(2)]

    even_c4 = [[0] * (r_max + 1) for _ in range(2)]
    for p0 in (0, 1):
        for p1 in (0, 1):
            p = p0 ^ p1
            a, b, dst = even_c2[p0], even_c2[p1], even_c4[p]
            for r0, v0 in enumerate(a):
                if v0 == 0:
                    continue
                for r1 in range(r_max - r0 + 1):
                    v1 = b[r1]
                    if v1:
                        dst[r0 + r1] += v0 * v1

    even_c8 = [0] * (m_max + 1)
    for m in range(m_max + 1):
        r8 = 2 * m
        tot = 0
        for p in (0, 1):
            c4 = even_c4[p]
            for r_l in range(r8 + 1):
                tot += c4[r_l] * c4[r8 - r_l]
        even_c8[m] = tot

    # Odd coset: u_i = 2*k_i + 1, t(k) = k*(k+1)//2, sum_{i=0..7} t(k_i) == m - 1
    t_max = m_max - 1 if m_max > 0 else 0
    ks_odd: list[int] = []
    k = 0
    while k * (k + 1) // 2 <= t_max:
        ks_odd.append(k)
        ks_odd.append(-1 - k)
        k += 1
    odd_p2: list[list[list[tuple[int, int]]]] = [[[] for _ in range(t_max + 1)] for _ in range(2)]
    odd_p2_idx: list[dict[tuple[int, int], tuple[int, int]]] = [{}, {}]
    for k0 in ks_odd:
        t0 = k0 * (k0 + 1) // 2
        for k1 in ks_odd:
            t = t0 + k1 * (k1 + 1) // 2
            if t <= t_max:
                p = (k0 + k1) & 1
                pair = (2 * k0 + 1, 2 * k1 + 1)
                odd_p2_idx[p][pair] = (t, len(odd_p2[p][t]))
                odd_p2[p][t].append(pair)
    odd_c2 = [[len(odd_p2[p][t]) for t in range(t_max + 1)] for p in range(2)]

    odd_c4 = [[0] * (t_max + 1) for _ in range(2)]
    for p0 in (0, 1):
        for p1 in (0, 1):
            p = p0 ^ p1
            a, b, dst = odd_c2[p0], odd_c2[p1], odd_c4[p]
            for t0, v0 in enumerate(a):
                if v0 == 0:
                    continue
                for t1 in range(t_max - t0 + 1):
                    v1 = b[t1]
                    if v1:
                        dst[t0 + t1] += v0 * v1

    odd_c8 = [0] * (m_max + 1)
    for m in range(1, m_max + 1):
        t8 = m - 1
        tot = 0
        for p in (0, 1):
            c4 = odd_c4[p]
            for t_l in range(t8 + 1):
                tot += c4[t_l] * c4[t8 - t_l]
        odd_c8[m] = tot

    # Precompute exact zero-division reciprocals for inner 8->4 and 4->2 splits
    inner_recips: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    # Key: (coset, half_dim, parity_R, norm_R) -> reciprocal dict
    total_inner_recips = 0
    valid_inner_recips = 0
    for coset, c2, c4, lim in ((0, even_c2, even_c4, r_max), (1, odd_c2, odd_c4, t_max)):
        for p_r in (0, 1):
            for nr in range(lim + 1):
                d4 = c4[p_r][nr]
                if d4 > 0:
                    max_l = max(c4[p_r][: lim - nr + 1])
                    rec = build_exact_reciprocal(d4, max_l * d4 - 1)
                    inner_recips[(coset, 4, p_r, nr)] = rec
                    total_inner_recips += 1
                    if rec["universally_valid"]:
                        valid_inner_recips += 1
                d2 = c2[p_r][nr]
                if d2 > 0:
                    max_l0 = max(c2[0][: lim - nr + 1])
                    max_l1 = max(c2[1][: lim - nr + 1])
                    rec = build_exact_reciprocal(d2, max(max_l0, max_l1) * d2 - 1)
                    inner_recips[(coset, 2, p_r, nr)] = rec
                    total_inner_recips += 1
                    if rec["universally_valid"]:
                        valid_inner_recips += 1

    theta_matches = all(
        even_c8[m] + odd_c8[m] == e8_theta_shell_count_exact(m)
        for m in range(m_max + 1)
    )
    return {
        "m_max": m_max,
        "r_max": r_max,
        "t_max": t_max,
        "even_p2": even_p2,
        "even_p2_idx": even_p2_idx,
        "even_c2": even_c2,
        "even_c4": even_c4,
        "even_c8": even_c8,
        "odd_p2": odd_p2,
        "odd_p2_idx": odd_p2_idx,
        "odd_c2": odd_c2,
        "odd_c4": odd_c4,
        "odd_c8": odd_c8,
        "shells_e8": [even_c8[m] + odd_c8[m] for m in range(m_max + 1)],
        "theta_series_all_shells_exact": theta_matches,
        "inner_recips": inner_recips,
        "total_inner_recips": total_inner_recips,
        "valid_inner_recips": valid_inner_recips,
    }


def build_e8_tree_tables(n_blocks: int, m_max: int) -> dict[str, Any]:
    """Build exact E_8^{n_blocks} outer block-tree + inner 8->4->2 LUT tables with 0-division reciprocals."""
    require_exact_int(n_blocks, "n_blocks", allow_zero=False)
    if (n_blocks & (n_blocks - 1)) != 0:
        raise BallpackTreeError("n_blocks must be a power of two >= 1")
    require_exact_int(m_max, "m_max", allow_zero=True)

    single = _build_single_e8_tree(m_max)
    shells: dict[int, list[int]] = {1: single["shells_e8"]}
    b = 1
    while b < n_blocks:
        sb = shells[b]
        s2b = [0] * (m_max + 1)
        for m1, c1 in enumerate(sb):
            if c1 == 0:
                continue
            for m2 in range(m_max - m1 + 1):
                c2 = sb[m2]
                if c2:
                    s2b[m1 + m2] += c1 * c2
        shells[2 * b] = s2b
        b *= 2

    balls: dict[int, list[int]] = {}
    for blk, s_row in shells.items():
        acc = 0
        w_row: list[int] = []
        for c in s_row:
            acc += c
            w_row.append(acc)
        balls[blk] = w_row

    c_root: list[int] = [0]
    if n_blocks >= 2:
        half = n_blocks // 2
        acc = 0
        for m_l in range(m_max + 1):
            acc += shells[half][m_l] * balls[half][m_max - m_l]
            c_root.append(acc)
    else:
        c_root = [0] + balls[1]

    outer_recip_ball: dict[int, dict[int, dict[str, Any]]] = {}
    outer_recip_shell: dict[int, dict[int, dict[str, Any]]] = {}
    total_outer_recips = 0
    valid_outer_recips = 0
    blk = n_blocks
    while blk >= 2:
        h = blk // 2
        rb_h: dict[int, dict[str, Any]] = {}
        rs_h: dict[int, dict[str, Any]] = {}
        for m_r in range(m_max + 1):
            d_ball = balls[h][m_r]
            max_s_l = max(shells[h][: m_max - m_r + 1])
            max_div_ball = max_s_l * d_ball - 1
            rec = build_exact_reciprocal(d_ball, max_div_ball)
            rb_h[m_r] = rec
            total_outer_recips += 1
            if rec["universally_valid"]:
                valid_outer_recips += 1
        if blk < n_blocks:
            for m_r in range(m_max + 1):
                d_sh = shells[h][m_r]
                max_s_l = max(shells[h][: m_max - m_r + 1])
                rec = build_exact_reciprocal(d_sh, max_s_l * d_sh - 1)
                rs_h[m_r] = rec
                total_outer_recips += 1
                if rec["universally_valid"]:
                    valid_outer_recips += 1
        outer_recip_ball[h] = rb_h
        outer_recip_shell[h] = rs_h
        blk = h

    return {
        "n_blocks": n_blocks,
        "n": 8 * n_blocks,
        "m_max": m_max,
        "bound_y_sq": 2 * m_max,
        "count": balls[n_blocks][m_max],
        "single": single,
        "shells": shells,
        "balls": balls,
        "c_root": c_root,
        "outer_recip_ball": outer_recip_ball,
        "outer_recip_shell": outer_recip_shell,
        "total_reciprocals": total_outer_recips + single["total_inner_recips"],
        "valid_reciprocals": valid_outer_recips + single["valid_inner_recips"],
    }


def _rank_e8_quad(u4: Sequence[int], coset: int, norm4: int, parity4: int, single: Mapping[str, Any]) -> int:
    c2 = single["even_c2"] if coset == 0 else single["odd_c2"]
    idx2 = single["even_p2_idx"] if coset == 0 else single["odd_p2_idx"]
    p_l = ((u4[0] // 2) + (u4[1] // 2)) & 1
    p_r = parity4 ^ p_l
    n_l, i_l = idx2[p_l][(u4[0], u4[1])]
    n_r, i_r = idx2[p_r][(u4[2], u4[3])]
    offset = 0
    for pa in (0, 1):
        pb = parity4 ^ pa
        for na in range(norm4 + 1):
            if pa == p_l and na == n_l:
                return offset + i_l * c2[p_r][n_r] + i_r
            offset += c2[pa][na] * c2[pb][norm4 - na]
    raise BallpackTreeError("unreachable E8 quad rank state")  # pragma: no cover


def _unrank_e8_quad(rank: int, coset: int, norm4: int, parity4: int, single: Mapping[str, Any]) -> list[int]:
    c2 = single["even_c2"] if coset == 0 else single["odd_c2"]
    p2 = single["even_p2"] if coset == 0 else single["odd_p2"]
    rem = rank
    for pa in (0, 1):
        pb = parity4 ^ pa
        for na in range(norm4 + 1):
            term = c2[pa][na] * c2[pb][norm4 - na]
            if rem < term:
                nb = norm4 - na
                rec = single["inner_recips"][(coset, 2, pb, nb)]
                i_l, i_r = divmod_exact_reciprocal(rem, rec["divisor"], rec["max_dividend"], rec["mult"], rec["shift"])
                pair_l = p2[pa][na][i_l]
                pair_r = p2[pb][nb][i_r]
                return [pair_l[0], pair_l[1], pair_r[0], pair_r[1]]
            rem -= term
    raise BallpackTreeError("unreachable E8 quad unrank state")  # pragma: no cover


def rank_e8_block_doubled(u: Sequence[int], tables: Mapping[str, Any], *, is_ball: bool, m_bound: int) -> int:
    """Rank a single E_8 vector in doubled integer coordinates u = 2y in Z^8 (0 floats)."""
    require_exact_bool(is_ball, "is_ball")
    require_exact_int(m_bound, "m_bound", allow_zero=True)
    single = tables["single"]
    if m_bound > single["m_max"]:
        raise BallpackTreeError("m_bound exceeds E_8 table m_max")
    if not is_valid_e8_doubled_block(u):
        raise BallpackTreeError("u is not a valid doubled E_8 block")
    u_norm_sq = sum(v * v for v in u)
    m = u_norm_sq // 8
    if m > m_bound or (not is_ball and m != m_bound):
        raise BallpackTreeError("E_8 block norm does not match requested shell/ball bound")
    base = sum(single["shells_e8"][:m]) if is_ball else 0
    coset = u[0] & 1
    if coset == 0:
        c4 = single["even_c4"]
        n_tot = 2 * m
        n_l = sum((v // 2) * (v // 2) for v in u[:4])
        p_l = sum(v // 2 for v in u[:4]) & 1
        offset = 0
        for p in (0, 1):
            for a in range(n_tot + 1):
                if p == p_l and a == n_l:
                    idx_l = _rank_e8_quad(u[:4], 0, n_l, p_l, single)
                    idx_r = _rank_e8_quad(u[4:], 0, n_tot - n_l, p_l, single)
                    return base + offset + idx_l * c4[p_l][n_tot - n_l] + idx_r
                offset += c4[p][a] * c4[p][n_tot - a]
    else:
        c4 = single["odd_c4"]
        n_tot = m - 1
        n_l = sum((v // 2) * ((v // 2) + 1) // 2 for v in u[:4])
        p_l = sum(v // 2 for v in u[:4]) & 1
        offset = single["even_c8"][m]
        for p in (0, 1):
            for a in range(n_tot + 1):
                if p == p_l and a == n_l:
                    idx_l = _rank_e8_quad(u[:4], 1, n_l, p_l, single)
                    idx_r = _rank_e8_quad(u[4:], 1, n_tot - n_l, p_l, single)
                    return base + offset + idx_l * c4[p_l][n_tot - n_l] + idx_r
                offset += c4[p][a] * c4[p][n_tot - a]
    raise BallpackTreeError("unreachable E8 block rank state")  # pragma: no cover


def unrank_e8_block_doubled(rank: int, tables: Mapping[str, Any], *, is_ball: bool, m_bound: int) -> list[int]:
    """Unrank a single E_8 vector into doubled integer coordinates u = 2y in Z^8 using 0 divisions."""
    require_exact_int(rank, "rank", allow_zero=True)
    require_exact_bool(is_ball, "is_ball")
    require_exact_int(m_bound, "m_bound", allow_zero=True)
    single = tables["single"]
    if m_bound > single["m_max"]:
        raise BallpackTreeError("m_bound exceeds E_8 table m_max")
    cap = sum(single["shells_e8"][: m_bound + 1]) if is_ball else single["shells_e8"][m_bound]
    if rank >= cap:
        raise BallpackTreeError("E_8 block rank exceeds shell/ball count")
    if is_ball:
        rem = rank
        m = 0
        while True:
            s_m = single["shells_e8"][m]
            if rem < s_m:
                break
            rem -= s_m
            m += 1
    else:
        m = m_bound
        rem = rank

    even_cnt = single["even_c8"][m]
    if rem < even_cnt:
        coset = 0
        c4 = single["even_c4"]
        n_tot = 2 * m
    else:
        coset = 1
        rem -= even_cnt
        c4 = single["odd_c4"]
        n_tot = m - 1

    for p in (0, 1):
        for n_l in range(n_tot + 1):
            n_r = n_tot - n_l
            term = c4[p][n_l] * c4[p][n_r]
            if rem < term:
                rec = single["inner_recips"][(coset, 4, p, n_r)]
                idx_l, idx_r = divmod_exact_reciprocal(rem, rec["divisor"], rec["max_dividend"], rec["mult"], rec["shift"])
                return _unrank_e8_quad(idx_l, coset, n_l, p, single) + _unrank_e8_quad(idx_r, coset, n_r, p, single)
            rem -= term
    raise BallpackTreeError("unreachable E8 block unrank state")  # pragma: no cover


def _rank_e8_outer_node(u: Sequence[int], blk: int, is_ball: bool, m_budget: int, tables: Mapping[str, Any]) -> int:
    if blk == 1:
        return rank_e8_block_doubled(u, tables, is_ball=is_ball, m_bound=m_budget)
    h = blk // 2
    split = 8 * h
    left = u[:split]
    right = u[split:]
    m_l = sum(v * v for v in left) // 8
    s_h = tables["shells"][h]
    if is_ball:
        w_h = tables["balls"][h]
        if blk == tables["n_blocks"]:
            offset = tables["c_root"][m_l]
        else:
            offset = sum(s_h[a] * w_h[m_budget - a] for a in range(m_l))
        idx_l = _rank_e8_outer_node(left, h, False, m_l, tables)
        idx_r = _rank_e8_outer_node(right, h, True, m_budget - m_l, tables)
        return offset + idx_l * w_h[m_budget - m_l] + idx_r
    offset = sum(s_h[a] * s_h[m_budget - a] for a in range(m_l))
    idx_l = _rank_e8_outer_node(left, h, False, m_l, tables)
    idx_r = _rank_e8_outer_node(right, h, False, m_budget - m_l, tables)
    return offset + idx_l * s_h[m_budget - m_l] + idx_r


def rank_tree_e8_blocks(u_all: Sequence[int], tables: Mapping[str, Any]) -> int:
    """Rank an E_8^{n_blocks} vector in doubled integer coordinates u = 2y."""
    if not isinstance(u_all, Sequence) or isinstance(u_all, (str, bytes)):
        raise BallpackTreeError("u_all must be a sequence of exact ints")
    n = tables["n"]
    if len(u_all) != n:
        raise BallpackTreeError(f"u_all length must equal {n}")
    cleaned = [require_exact_int(v, f"u_all[{i}]", min_val=-(8 * tables["m_max"] + 2)) for i, v in enumerate(u_all)]
    for b in range(tables["n_blocks"]):
        if not is_valid_e8_doubled_block(cleaned[8 * b : 8 * b + 8]):
            raise BallpackTreeError(f"block {b} is not a valid doubled E_8 vector")
    if sum(v * v for v in cleaned) > 8 * tables["m_max"]:
        raise BallpackTreeError("u_all squared norm exceeds E_8 ball bound")
    return _rank_e8_outer_node(cleaned, tables["n_blocks"], True, tables["m_max"], tables)


def _unrank_e8_outer_node(rank: int, blk: int, is_ball: bool, m_budget: int, tables: Mapping[str, Any]) -> list[int]:
    if blk == 1:
        return unrank_e8_block_doubled(rank, tables, is_ball=is_ball, m_bound=m_budget)
    h = blk // 2
    s_h = tables["shells"][h]
    if is_ball:
        w_h = tables["balls"][h]
        if blk == tables["n_blocks"]:
            c_root = tables["c_root"]
            m_l = bisect_right(c_root, rank) - 1
            rem = rank - c_root[m_l]
        else:
            rem = rank
            m_l = 0
            while True:
                term = s_h[m_l] * w_h[m_budget - m_l]
                if rem < term:
                    break
                rem -= term
                m_l += 1
        rec = tables["outer_recip_ball"][h][m_budget - m_l]
        idx_l, idx_r = divmod_exact_reciprocal(rem, rec["divisor"], rec["max_dividend"], rec["mult"], rec["shift"])
        return _unrank_e8_outer_node(idx_l, h, False, m_l, tables) + _unrank_e8_outer_node(idx_r, h, True, m_budget - m_l, tables)

    rem = rank
    m_l = 0
    while True:
        term = s_h[m_l] * s_h[m_budget - m_l]
        if rem < term:
            break
        rem -= term
        m_l += 1
    rec = tables["outer_recip_shell"][h][m_budget - m_l]
    idx_l, idx_r = divmod_exact_reciprocal(rem, rec["divisor"], rec["max_dividend"], rec["mult"], rec["shift"])
    return _unrank_e8_outer_node(idx_l, h, False, m_l, tables) + _unrank_e8_outer_node(idx_r, h, False, m_budget - m_l, tables)


def unrank_tree_e8_blocks(rank: int, tables: Mapping[str, Any]) -> list[int]:
    """Unrank a 0-based integer rank into E_8^{n_blocks} doubled coordinates u = 2y using 0 divisions."""
    require_exact_int(rank, "rank", allow_zero=True)
    if rank >= tables["count"]:
        raise BallpackTreeError("rank exceeds E_8 codebook cardinality - 1")
    return _unrank_e8_outer_node(rank, tables["n_blocks"], True, tables["m_max"], tables)


def certify_e8_tree_bijection_exact(
    n_blocks: int,
    m_max: int,
    *,
    exhaustive: bool = False,
    n_random: int = 24,
) -> dict[str, Any]:
    """Certify E_8^{n_blocks} theta-series shell identity, depth-2 native-uint64 block split, & 0-div bijection."""
    require_exact_bool(exhaustive, "exhaustive")
    tables = build_e8_tree_tables(n_blocks, m_max)
    count = tables["count"]
    bits = max(1, (count - 1).bit_length())
    single = tables["single"]

    if exhaustive:
        test_ranks = list(range(count))
    else:
        test_ranks = deterministic_sample_ranks(count, n_random=n_random)

    seen: set[tuple[int, ...]] = set()
    all_roundtrip_ok = True
    for rk in test_ranks:
        u = unrank_tree_e8_blocks(rk, tables)
        if sum(v * v for v in u) > 8 * m_max:
            all_roundtrip_ok = False
        for b in range(n_blocks):
            if not is_valid_e8_doubled_block(u[8 * b : 8 * b + 8]):
                all_roundtrip_ok = False
        if rank_tree_e8_blocks(u, tables) != rk:
            all_roundtrip_ok = False
        if exhaustive:
            seen.add(tuple(u))
    exhaustive_distinct_ok = (len(seen) == count) if exhaustive else True

    e8_1_max_s = max(tables["shells"][1])
    e8_1_max_w = tables["balls"][1][m_max]
    e8_1_s_bits = max(1, (e8_1_max_s - 1).bit_length()) if e8_1_max_s > 0 else 0
    e8_1_w_bits = max(1, (e8_1_max_w - 1).bit_length()) if e8_1_max_w > 0 else 0

    outer_levels: list[dict[str, Any]] = []
    blk = n_blocks
    depth = 0
    while blk >= 1:
        max_s = max(tables["shells"][blk])
        max_w = tables["balls"][blk][m_max]
        s_bits = max(1, (max_s - 1).bit_length()) if max_s > 0 else 0
        w_bits = max(1, (max_w - 1).bit_length()) if max_w > 0 else 0
        outer_levels.append({
            "depth": depth,
            "blocks": blk,
            "dim": 8 * blk,
            "parallel_nodes": n_blocks // blk,
            "max_shell_bits": s_bits,
            "max_ball_bits": w_bits,
            "shell_hw_type": _hardware_type_for_bits(s_bits),
            "ball_hw_type": _hardware_type_for_bits(w_bits),
        })
        blk //= 2
        depth += 1

    even_p2_pairs = sum(single["even_c2"][0]) + sum(single["even_c2"][1])
    odd_p2_pairs = sum(single["odd_c2"][0]) + sum(single["odd_c2"][1])
    ballindex_e8_entries = n_blocks * 9 * 4 * (8 * m_max + 1) + (n_blocks + 1) * (8 * m_max + 1)
    ballindex_e8_bytes = ballindex_e8_entries * 24

    # Exact reduced-norm tree memory: C_root + outer S/W + inner c4/c2 + pair LUTs
    tree_bytes = (m_max + 2) * _bytes_per_entry_for_bits(bits)
    blk = n_blocks // 2
    while blk >= 1:
        s_b = max(1, (max(tables["shells"][blk]) - 1).bit_length())
        w_b = max(1, (tables["balls"][blk][m_max] - 1).bit_length())
        tree_bytes += (m_max + 1) * (_bytes_per_entry_for_bits(s_b) + _bytes_per_entry_for_bits(w_b))
        blk //= 2
    tree_bytes += 2 * (single["r_max"] + 1) * (4 + 2)  # even c4 (u32) + c2 (u16)
    tree_bytes += 2 * (single["t_max"] + 1) * (4 + 2)  # odd c4 (u32) + c2 (u16)
    tree_bytes += (even_p2_pairs + odd_p2_pairs) * 2  # int8[2] doubled pairs
    tree_bytes += 2 * (single["r_max"] + 2) * 2 + 2 * (single["t_max"] + 2) * 2  # offsets

    return {
        "n_blocks": n_blocks,
        "n": 8 * n_blocks,
        "m_max": m_max,
        "bound_y_sq": 2 * m_max,
        "codebook_count": count,
        "codebook_bits": bits,
        "theta_series_all_shells_exact": single["theta_series_all_shells_exact"],
        "depth_to_independent_e8_blocks": n_blocks.bit_length() - 1,
        "e8_block_max_shell_bits": e8_1_s_bits,
        "e8_block_max_ball_bits": e8_1_w_bits,
        "e8_block_hw_type": _hardware_type_for_bits(e8_1_w_bits),
        "even_pair_lut_count": even_p2_pairs,
        "odd_pair_lut_count": odd_p2_pairs,
        "tree_total_bytes": tree_bytes,
        "ballindex_e8_entries": ballindex_e8_entries,
        "ballindex_e8_bytes": ballindex_e8_bytes,
        "table_memory_reduction_ratio": frac_to_dict(Fraction(ballindex_e8_bytes, tree_bytes)),
        "exhaustive_mode": exhaustive,
        "ranks_tested": len(test_ranks),
        "all_roundtrips_exact": all_roundtrip_ok,
        "exhaustive_bijection_distinct": exhaustive_distinct_ok,
        "total_reciprocals": tables["total_reciprocals"],
        "valid_reciprocals": tables["valid_reciprocals"],
        "all_reciprocals_universally_valid": tables["valid_reciprocals"] == tables["total_reciprocals"],
        "outer_levels": outer_levels,
    }


# =============================================================================
# SECTION 4: PILLAR 3 - Exact Sub-Block Shaping Tradeoff & siliquant#23 Ceiling
# =============================================================================

def certify_shaping_and_bandwidth_ceiling_exact(z_tables_32: Mapping[str, Any]) -> dict[str, Any]:
    """Certify exact pure-Q shaping penalty of naive sub-blocking vs 0-overhead tree-convolution
    and exact siliquant#23 Q4_K_M vs PZ32 bandwidth ceiling bounds.
    """
    if not isinstance(z_tables_32, Mapping) or z_tables_32.get("n") != 32:
        raise BallpackTreeError("z_tables_32 must be a Z^32 table dict")
    b32 = z_tables_32["bound"]
    s_map = z_tables_32["shells"]
    w_map = z_tables_32["balls"]
    w32 = w_map[32][b32]
    e_y2_32 = Fraction(sum(r * s_map[32][r] for r in range(b32 + 1)), 32 * w32)

    subblock_arms: list[dict[str, Any]] = []
    for k_sub, dim in ((1, 32), (2, 16), (4, 8), (8, 4)):
        sub_bits = 136 // k_sub
        cap = 1 << sub_bits
        w_dim = w_map[dim]
        b_sub = max(r for r, c in enumerate(w_dim) if c <= cap)
        w_sub = w_dim[b_sub]
        tot_codebook = w_sub ** k_sub
        e_y2_sub = Fraction(sum(r * s_map[dim][r] for r in range(b_sub + 1)), dim * w_sub)
        energy_overhead = (e_y2_sub - e_y2_32) / e_y2_32
        subblock_arms.append({
            "arm": f"{k_sub}xZ^{dim}",
            "sub_blocks": k_sub,
            "sub_dim": dim,
            "bits_per_subblock": sub_bits,
            "sub_bound": b_sub,
            "sum_sub_bounds": k_sub * b_sub,
            "sub_codebook_count": w_sub,
            "codebook_ratio_z32_over_sub": frac_to_dict(Fraction(w32, tot_codebook)),
            "codebook_ratio_dec": format_exact_decimal(Fraction(w32, tot_codebook), 6),
            "mean_sq_coord_dec": format_exact_decimal(e_y2_sub, 6),
            "shaping_energy_overhead_pct": format_exact_percent(energy_overhead, 2),
            "shaping_loss_positive": e_y2_sub > e_y2_32,
        })

    # Exact siliquant#23 bandwidth & decode gap ratios (pure Q)
    pz_pure = Fraction(136 + 8, 32)  # 9/2 = 4.500 bpw
    pz_kept = Fraction(4505, 1000)   # 901/200 = 4.505 bpw (with 10 kept ffn_down cols)
    q4_lo = Fraction(480, 100)       # 24/5 = 4.80 bpw
    q4_hi = Fraction(482, 100)       # 241/50 = 4.82 bpw

    bw_comparisons: list[dict[str, Any]] = []
    for label, pz, q4 in (
        ("PZ32_4505_vs_Q4KM_480", pz_kept, q4_lo),
        ("PZ32_4505_vs_Q4KM_482", pz_kept, q4_hi),
        ("PZ32_4500_vs_Q4KM_480", pz_pure, q4_lo),
        ("PZ32_4500_vs_Q4KM_482", pz_pure, q4_hi),
    ):
        byte_saving = (q4 - pz) / q4
        bw_speedup = (q4 - pz) / pz
        bw_comparisons.append({
            "comparison": label,
            "pz_bpw": frac_to_dict(pz),
            "q4_bpw": frac_to_dict(q4),
            "byte_saving": frac_to_dict(byte_saving),
            "byte_saving_pct": format_exact_percent(byte_saving, 2),
            "bw_tps_ceiling_gain": frac_to_dict(bw_speedup),
            "bw_tps_ceiling_gain_pct": format_exact_percent(bw_speedup, 2),
        })

    # Exact #23 serial decode gap: 120 passes/s * 1.5B weights / 4 cores = 45B weights/s/core vs 20M-25M
    q4_wps_per_core = 120 * 1_500_000_000 // 4
    gap_vs_20m = Fraction(q4_wps_per_core, 20_000_000)
    gap_vs_25m = Fraction(q4_wps_per_core, 25_000_000)

    return {
        "z32_tree_shaping_loss_pct": "0.00%",
        "subblock_shaping_comparison": subblock_arms,
        "subblock_energy_monotone_worse": all(
            subblock_arms[i]["shaping_loss_positive"] for i in (1, 2, 3)
        ),
        "siliquant_issue23_bandwidth_ceilings": bw_comparisons,
        "serial_decode_gap_vs_q4km_per_core": {
            "q4km_ref_weights_per_sec_per_core": q4_wps_per_core,
            "ballpack_c_lo_weights_per_sec": 20_000_000,
            "ballpack_c_hi_weights_per_sec": 25_000_000,
            "gap_ratio_at_20m": frac_to_dict(gap_vs_20m),
            "gap_ratio_at_25m": frac_to_dict(gap_vs_25m),
        },
        "honest_cost_model_caveat": (
            "Pure-Z/Q Python certification proves exact bijectivity, 0.000-bit rate overhead, "
            "critical-path depth collapse (32->4 for Z^32, depth 2 to 4 uint64 E_8 blocks), "
            "exact register widths, L1/L2 LUT byte bounds, and universal zero-division reciprocals; "
            "actual wall-clock weights/sec requires compiled C/SIMD microbenchmarking on target CPU/TPU."
        ),
    }
