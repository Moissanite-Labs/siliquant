#!/usr/bin/env python3
"""Track S163 EXP-Q42a x Model Carving (arXiv 2611.27962) g = 0 lockout duality verifier.

Pure exact integer and rational verification. Standard library only.

Pillar 1: [16, 11, 4] A_16 codeword complement c -> c XOR 0xFFFF (exhaustive, 65,536 vectors).
Pillar 2: Block-H4 Parseval, scale sign-flip s -> -s, multi-round signs, ternary {-1, 0, +1}.
Pillar 3: 16 B Weft spinor checkpoint ("<4i") bijection and check-before-assign rollback.
Pillar 4: Chantless 0/1 up_row_zero collapse and Model Carving Gates.step g = 0 lockout.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
import subprocess
import tempfile

ALLOWED_AXIOMS = frozenset({"propext", "Quot.sound"})
EXPECTED_THEOREMS = 36
MAX_BYTES = 1048576
MAX_DEPTH = 32
MAX_INT_DIGITS = 256
CERT_NAME = "s163_ternary_a16_carving_certificate.json"

MASKS = (0xFFFF, 0xAAAA, 0xCCCC, 0xF0F0, 0xFF00)
MASK_WEIGHTS = (16, 8, 8, 8, 8)
FLIP = 0xFFFF
H4_ROWS = ((1, 1, 1, 1), (1, -1, 1, -1), (1, 1, -1, -1), (1, -1, -1, 1))
H4_GRAM = ((4, 0, 0, 0), (0, 4, 0, 0), (0, 0, 4, 0), (0, 0, 0, 4))
CANON_SCALE = Fraction(3, 8)
INT32_LO = -2147483648
INT32_HI = 2147483648
CANON_SPINOR = (805306368, -268435456, 536870912, -671088640)
CANON_D = 1207959552
CANON_SEQUENCES = (
    (1,),
    (-1,),
    (-1, -1),
    (-1, 1, -1, -1),
    (1, -1, -1, -1, 1, -1),
    (-1, -1, -1, -1, -1, -1, -1),
    (-1, -1, -1, -1, -1, -1, -1, -1),
    tuple(-1 if (t * t + t) % 3 == 0 else 1 for t in range(64)),
)
CANON_OPS = (
    ("switch", 0, -1),
    ("switch", 1, 0),
    ("switch", 2, 2),
    ("assign", 3, "tamper_lsb"),
    ("switch", 3, -1),
    ("switch", 4, -1),
    ("switch", 4, 1),
    ("assign", 5, "truncate"),
    ("switch", 5, 0),
    ("switch", 6, -2),
    ("tamper_ckpt", 0, "flip_byte"),
    ("switch", 7, -1),
    ("switch", 0, 0),
    ("switch", 0, -1),
    ("tamper_ckpt", 0, "truncate"),
    ("switch", 2, -1),
    ("tamper_ckpt", 0, "nonunit"),
    ("switch", 1, -1),
    ("switch", 6, 0),
)
RETENTION = ((9921, 8333), (9396, 8322), (9558, 8729))
RETENTION_DEN = 10000
INKLING = (6, 46)
INKLING_REDUCED = (3, 23)
CARVING_REMOVED = 193
CARVING_KEPT = 671
CARVING_HALF = (184, 141)
CARVING_TENTH = (35, 33)
HOLD_OPEN = Fraction(1, 1000000)
REOPEN_GRAD = Fraction(-1, 3)
CANON_PRESSURES = tuple(Fraction((t * 7) % 5, 10000000) for t in range(48))

_FORBIDDEN_LEAN_RE = re.compile(
    r"\b(?:sorry|admit|axiom|unsafe|native_decide|sorryAx|Classical|classical|ofReduceBool|ofReduceNat"
    r"|trustCompiler|implemented_by|extern|initialize|builtin_initialize|run_cmd|run_meta|run_tac"
    r"|syntax|declare_syntax_cat|macro|macro_rules|elab|elab_rules|notation|dbg_trace|dbgTrace"
    r"|dbgTraceVal|dbgTraceIfShared|dbgSleep|dbgStackTrace|panicWithPosWithDecl|throwError|csimp"
    r"|opaque|mkSorry|bv_decide|set_option|noncomputable|partial|prelude)\b"
    r"|\b(?:panic|unreachable)!"
    r"|#\s*(?:eval|reduce|exit|check(?:_failure)?|synth|guard(?:_msgs)?)\b"
    r"|\bIO\s*\."
    r"|\bdebug\s*\.\s*(?:skipKernelTC|byAsSorry|proofAsSorry|sorryAsError)\b"
    r"|\bwarn\s*\.\s*sorry\b"
)
_ANY_IMPORT_RE = re.compile(r"(?:^|\n)[ \t]*(?:(?:public|private|meta)[ \t]+)*import[ \t]+([A-Za-z0-9_.]+)")
_VALID_IMPORT_RE = re.compile(r"TernaryA16Carving(?:\.[A-Za-z0-9_]+)*")
_PUB_THM_RE = re.compile(r"(?:^|\n)[ \t]*theorem[ \t]+([A-Za-z0-9_.']+)")
_PRINT_RE = re.compile(r"(?:^|\n)[ \t]*#print[ \t]+axioms[ \t]+TernaryA16Carving\.([A-Za-z0-9_.']+)")


class VerifyError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(code + ": " + message)


def require_ok(cond, code, message):
    if cond is not True:
        raise VerifyError(code, message)


def frac_str(value):
    value = Fraction(value)
    return str(value.numerator) + "/" + str(value.denominator)


def normsq(vec):
    total = 0
    for x in vec:
        total = total + x * x
    return total


# ---------------------------------------------------------------------------
# Pillar 1: [16, 11, 4] A_16 codeword complement
# ---------------------------------------------------------------------------

def syndrome16(m, masks=MASKS):
    out = 0
    shift = 0
    for mask in masks:
        out = out | (((m & mask).bit_count() & 1) << shift)
        shift = shift + 1
    return out


def complement16(m, flip=FLIP):
    return (m ^ flip) & 0xFFFF


_BYTE_BIPOLAR = tuple(tuple(1 - 2 * ((v >> j) & 1) for j in range(8)) for v in range(256))


def bipolar16(m):
    return list(_BYTE_BIPOLAR[m & 0xFF] + _BYTE_BIPOLAR[(m >> 8) & 0xFF])


def zero_syndrome_codewords(masks=MASKS):
    return [m for m in range(65536) if syndrome16(m, masks) == 0]


def verify_pillar1(masks=MASKS, flip=FLIP):
    weights = tuple(mask.bit_count() for mask in masks)
    require_ok(weights == MASK_WEIGHTS, "P1_WEIGHTS", "parity-check row weights must be 16,8,8,8,8")
    codewords = 0
    closed = 0
    checked = 0
    for m in range(65536):
        checked = checked + 1
        c = complement16(m, flip)
        require_ok(complement16(c, flip) == m, "P1_INVOLUTION", "complement must be an involution")
        syn_m = syndrome16(m, masks)
        syn_c = syndrome16(c, masks)
        require_ok(syn_c == syn_m, "P1_SYN_INVARIANT", "complement must preserve the syndrome")
        phi = bipolar16(m)
        phi_c = bipolar16(c)
        require_ok(phi_c == [-x for x in phi], "P1_BIPOLAR_NEG", "complement must negate the bipolar vector")
        require_ok(normsq(phi_c) == 16, "P1_BIPOLAR_NORM", "bipolar L2^2 norm must be 16")
        codewords = codewords + (1 if syn_m == 0 else 0)
        closed = closed + (1 if syn_m == 0 else 0) * (1 if syn_c == 0 else 0)
    require_ok(codewords == 2048, "P1_CODEWORDS", "zero-syndrome codeword count must be 2048")
    return {
        "masks": ["0x%04X" % mask for mask in masks],
        "mask_weights": list(weights),
        "flip": "0x%04X" % flip,
        "vectors_checked": checked,
        "codewords": codewords,
        "codewords_closed_under_flip": closed,
        "syndrome_0x0000": syndrome16(0, masks),
        "syndrome_0xFFFF": syndrome16(0xFFFF, masks),
        "syndrome_0x0001": syndrome16(1, masks),
        "bipolar_norm_sq": normsq(bipolar16(0xFFFF)),
        "bipolar_0xFFFF_head": bipolar16(0xFFFF)[:4],
    }


# ---------------------------------------------------------------------------
# Pillar 2: Block-H4 Parseval, scale sign-flip, multi-round, ternary switches
# ---------------------------------------------------------------------------

def lsb_mask(c):
    out = 0
    for j in range(len(c)):
        out = out | ((c[j] & 1) << j)
    return out


def make_block(mask, k):
    out = []
    for j in range(16):
        base = ((5 * j + 3 * k) % 16) - 8
        bit = (mask >> j) & 1
        out.append(base if (base & 1) == bit else base ^ 1)
    return out


def pack_nibbles(c):
    w64 = 0
    for j in range(len(c)):
        w64 = w64 | ((c[j] & 0xF) << (4 * j))
    return w64.to_bytes(8, "little")


def h4_apply(group, rows=H4_ROWS):
    return [sum(rows[i][k] * group[k] for k in range(4)) for i in range(4)]


def h4_transform(c, rows=H4_ROWS):
    out = []
    for g in range(4):
        out.extend(h4_apply(c[4 * g:4 * g + 4], rows))
    return out


def block_weights(h, s):
    half = Fraction(1, 2)
    return [half * s * x for x in h]


def verify_block(c, s, rows=H4_ROWS):
    require_ok(len(c) == 16, "P2_BLOCK_LEN", "block must have 16 nibbles")
    require_ok(all(-8 <= x <= 7 for x in c), "P2_NIBBLE_RANGE", "nibbles must lie in [-8, 7]")
    require_ok(syndrome16(lsb_mask(c)) == 0, "P2_BLOCK_SYN", "block LSB mask must have zero syndrome")
    h = h4_transform(c, rows)
    require_ok(normsq(h) == 4 * normsq(c), "P2_PARSEVAL", "Block-H4 Parseval identity failed")
    w = block_weights(h, s)
    require_ok(4 * normsq(w) == (2 * s) * (2 * s) * normsq(c), "P2_WEIGHT_NORM", "cleared weight norm failed")
    w_neg = block_weights(h, -s)
    require_ok(w_neg == [-x for x in w], "P2_FLIP_NEG", "scale sign-flip must negate weights")
    return h, w, w_neg


def apply_round(s, sigma):
    return sigma * s


def verify_pillar2(scale=CANON_SCALE, rows=H4_ROWS, sequences=CANON_SEQUENCES):
    require_ok(scale != 0, "P2_SCALE_ZERO", "scale must be nonzero")
    gram = tuple(tuple(sum(rows[k][i] * rows[k][j] for k in range(4)) for j in range(4)) for i in range(4))
    require_ok(gram == H4_GRAM, "P2_H4_GRAM", "H4^T H4 must equal 4 I")
    codewords = zero_syndrome_codewords()
    blocks = [make_block(m, k) for k, m in enumerate(codewords)]
    sum_c = 0
    sum_h = 0
    sum_w = Fraction(0)
    flip_drift = Fraction(0)
    bytes_changed = 0
    for c in blocks:
        before_bytes = pack_nibbles(c)
        h, w, w_neg = verify_block(c, scale, rows)
        sum_c = sum_c + normsq(c)
        sum_h = sum_h + normsq(h)
        sum_w = sum_w + normsq(w)
        flip_drift = flip_drift + (normsq(w_neg) - normsq(w))
        bytes_changed = bytes_changed + sum(1 for a, b in zip(before_bytes, pack_nibbles(c)) if a != b)
    require_ok(bytes_changed == 0, "P2_PAYLOAD_MUTATED", "verification and scale flip must not change nibble bytes")
    probe = blocks[5]
    h_probe = h4_transform(probe, rows)
    w_probe = block_weights(h_probe, scale)
    round_drift = Fraction(0)
    identity_returns = 0
    negations = 0
    for seq in sequences:
        s_t = scale
        flips = 0
        for sigma in seq:
            require_ok(sigma in (-1, 1), "P2_SIGMA", "active round sign must be -1 or +1")
            s_t = apply_round(s_t, sigma)
            flips = flips + (1 if sigma == -1 else 0)
        w_t = block_weights(h_probe, s_t)
        expected = w_probe if flips % 2 == 0 else [-x for x in w_probe]
        require_ok(w_t == expected, "P2_ROUND_NET", "multi-round net sign mismatch")
        round_drift = round_drift + (normsq(w_t) - normsq(w_probe))
        identity_returns = identity_returns + (1 if flips % 2 == 0 else 0)
        negations = negations + (1 if flips % 2 == 1 else 0)
    return {
        "scale": frac_str(scale),
        "h4_gram_diag": [gram[i][i] for i in range(4)],
        "blocks": len(blocks),
        "sum_c_sq": sum_c,
        "sum_h4_sq": sum_h,
        "sum_w_sq": frac_str(sum_w),
        "flip_norm_drift": frac_str(flip_drift),
        "flip_payload_bytes_changed": bytes_changed,
        "sequences": len(sequences),
        "round_lengths": [len(seq) for seq in sequences],
        "round_norm_drift": frac_str(round_drift),
        "even_identity_returns": identity_returns,
        "odd_negations": negations,
        "probe_block_hex": pack_nibbles(probe).hex(),
    }


def ternary_view(t, c, s):
    require_ok(t in (-1, 0, 1), "T_STATE", "ternary state must be -1, 0 or +1")
    if t == 0:
        return 0, [0] * 16, s
    if t == 1:
        return lsb_mask(c), list(c), s
    return complement16(lsb_mask(c)), list(c), -s


def ternary_weights(t, c, s):
    view = ternary_view(t, c, s)
    return view[0], block_weights(h4_transform(view[1]), view[2])


def switch_state(cur, req):
    if req in (-1, 0, 1):
        return req
    return cur


def verify_ternary(blocks, scale=CANON_SCALE):
    syndromes_checked = 0
    restored = 0
    for c in blocks:
        base = ternary_weights(1, c, scale)[1]
        for t in (1, 0, -1):
            mask_t, w_t = ternary_weights(t, c, scale)
            require_ok(syndrome16(mask_t) == 0, "T_SYNDROME", "ternary state must keep zero syndrome")
            syndromes_checked = syndromes_checked + 1
        for t in (0, -1):
            back = switch_state(t, 1)
            require_ok(ternary_weights(back, c, scale)[1] == base, "T_REVERSE", "switch back to +1 must restore base")
            restored = restored + 1
    zero_w = ternary_weights(0, blocks[0], scale)[1]
    neg_w = ternary_weights(-1, blocks[0], scale)[1]
    base_w = ternary_weights(1, blocks[0], scale)[1]
    return {
        "blocks": len(blocks),
        "state_syndromes_zero": syndromes_checked,
        "exact_restores": restored,
        "zero_state_norm_sq": frac_str(normsq(zero_w)),
        "neg_state_is_negation": neg_w == [-x for x in base_w],
        "bounced_invalid_requests": sum(1 for r in (2, -2, 3) if switch_state(1, r) == 1),
    }


# ---------------------------------------------------------------------------
# Pillar 3: 16 B Weft spinor checkpoint and rollback
# ---------------------------------------------------------------------------

def pack16B(q):
    return struct.pack("<4i", q[0], q[1], q[2], q[3])


def unpack16B(raw):
    return tuple(struct.unpack("<4i", raw))


def spinor_norm(q):
    return q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]


def flip_spinor(q):
    return (-q[0], -q[1], -q[2], -q[3])


def sign_of(t):
    return -1 if t == -1 else 1


def assign_base(cur_c, cand):
    if len(cand) != 16:
        return list(cur_c), 1
    if syndrome16(lsb_mask(cand)) != 0:
        return list(cur_c), 1
    return list(cand), 0


def tamper_candidate(c, kind):
    if kind == "tamper_lsb":
        return [c[0] ^ 1] + list(c[1:])
    return list(c[:15])


def rollback(state, ckpt, digest, ckpt_switches, d_sq):
    if len(ckpt) != 16:
        return state, 1
    if hashlib.sha256(ckpt).hexdigest() != digest:
        return state, 1
    cand = unpack16B(ckpt)
    if spinor_norm(cand) != d_sq:
        return state, 1
    return {"switches": list(ckpt_switches), "spinor": cand}, 0


def tampered_checkpoint(ckpt, digest, kind):
    if kind == "flip_byte":
        return bytes([ckpt[0] ^ 1]) + ckpt[1:], digest
    if kind == "truncate":
        return ckpt[:15], digest
    raw = pack16B((1, 2, 3, 4))
    return raw, hashlib.sha256(raw).hexdigest()


def verify_pillar3(spinor=CANON_SPINOR, d=CANON_D, ops=CANON_OPS, scale=CANON_SCALE):
    q0 = tuple(spinor)
    require_ok(all(INT32_LO <= x < INT32_HI for x in q0), "W_INT32_RANGE", "spinor must fit int32")
    d_sq = d * d
    require_ok(spinor_norm(q0) == d_sq, "W_UNIT_NORM", "spinor must satisfy qw^2+qx^2+qy^2+qz^2 = D^2")
    ckpt = pack16B(q0)
    require_ok(len(ckpt) == 16, "W_PACK_LEN", "checkpoint must be exactly 16 bytes")
    probes = (q0, flip_spinor(q0), (INT32_LO, INT32_HI - 1, 0, -1), (1, -1, 2147483647, -2147483647))
    for q in probes:
        require_ok(unpack16B(pack16B(q)) == q, "W_ROUNDTRIP", "unpack16B(pack16B(q)) must equal q")
    byte_probes = 0
    for i in range(256):
        raw = bytes((i * 37 + j * 11) % 256 for j in range(16))
        require_ok(pack16B(unpack16B(raw)) == raw, "W_BYTES_ROUNDTRIP", "pack16B(unpack16B(b)) must equal b")
        byte_probes = byte_probes + 1
    digest = hashlib.sha256(ckpt).hexdigest()
    codewords = zero_syndrome_codewords()
    blocks = [make_block(codewords[(k * 293 + 7) % 2048], k) for k in range(8)]
    base_weights = [ternary_weights(1, c, scale)[1] for c in blocks]
    ckpt_switches = [1] * len(blocks)
    state = {"switches": list(ckpt_switches), "spinor": q0}
    bounced_switch = 0
    bounced_assign = 0
    bounced_ckpt = 0
    flips = 0
    for op in ops:
        kind = op[0]
        require_ok(kind in ("switch", "assign", "tamper_ckpt"), "W_OP_KIND", "unknown composite op")
        idx = op[1]
        if kind == "switch":
            old = state["switches"][idx]
            new = switch_state(old, op[2])
            bounced_switch = bounced_switch + (0 if op[2] in (-1, 0, 1) else 1)
            if sign_of(old) != sign_of(new):
                state["spinor"] = flip_spinor(state["spinor"])
                flips = flips + 1
            state["switches"][idx] = new
        if kind == "assign":
            cand = tamper_candidate(blocks[idx], op[2])
            kept_c, bounced = assign_base(blocks[idx], cand)
            blocks[idx] = kept_c
            bounced_assign = bounced_assign + bounced
        if kind == "tamper_ckpt":
            bad, bad_digest = tampered_checkpoint(ckpt, digest, op[2])
            state, bounced = rollback(state, bad, bad_digest, ckpt_switches, d_sq)
            bounced_ckpt = bounced_ckpt + bounced
        require_ok(spinor_norm(state["spinor"]) == d_sq, "W_ROUND_NORM", "spinor norm must stay D^2 every round")
        for b in range(len(blocks)):
            mask_t = ternary_weights(state["switches"][b], blocks[b], scale)[0]
            require_ok(syndrome16(mask_t) == 0, "W_ROUND_SYN", "every block must keep zero syndrome every round")
    pre_spinor = state["spinor"]
    pre_switches = list(state["switches"])
    state, bounced = rollback(state, ckpt, digest, ckpt_switches, d_sq)
    require_ok(bounced == 0, "W_ROLLBACK_BOUNCED", "trusted checkpoint must be accepted")
    require_ok(pack16B(state["spinor"]) == ckpt, "W_RESTORE_BYTES", "rollback must restore the 16 B checkpoint")
    restored = [ternary_weights(state["switches"][b], blocks[b], scale)[1] for b in range(len(blocks))]
    require_ok(restored == base_weights, "W_RESTORE_WEIGHTS", "rollback must restore every block weight")
    weight_drift = Fraction(0)
    for b in range(len(blocks)):
        weight_drift = weight_drift + (normsq(restored[b]) - normsq(base_weights[b]))
    return {
        "spinor": list(q0),
        "d": d,
        "d_sq": d_sq,
        "checkpoint_hex": ckpt.hex(),
        "checkpoint_sha256": digest,
        "checkpoint_bytes": len(ckpt),
        "roundtrip_probes": len(probes),
        "byte_roundtrip_probes": byte_probes,
        "ops": len(ops),
        "spinor_flips": flips,
        "pre_rollback_spinor": list(pre_spinor),
        "pre_rollback_switches": pre_switches,
        "bounced_switch_requests": bounced_switch,
        "bounced_assignments": bounced_assign,
        "bounced_tampered_checkpoints": bounced_ckpt,
        "post_rollback_switches": list(state["switches"]),
        "spinor_norm_drift": frac_str(spinor_norm(state["spinor"]) - d_sq),
        "weight_norm_drift": frac_str(weight_drift),
    }


# ---------------------------------------------------------------------------
# Pillar 4: Chantless up_row_zero and Model Carving g = 0 lockout duality
# ---------------------------------------------------------------------------

def up_row_zero(w):
    return [0 for _ in w]


def verify_chantless(retention=RETENTION, den=RETENTION_DEN, inkling=INKLING, reduced=INKLING_REDUCED, blocks=None):
    drops = []
    for pair in retention:
        require_ok(pair[1] < pair[0], "C_RETENTION_DROP", "retention must strictly drop under up_row_zero")
        require_ok(pair[0] <= den, "C_RETENTION_RANGE", "retention numerator must not exceed denominator")
        drops.append(frac_str(Fraction(pair[0] - pair[1], den)))
    require_ok(Fraction(inkling[0], inkling[1]) == Fraction(reduced[0], reduced[1]), "C_INKLING", "Inkling-2-Mini ratio mismatch")
    if blocks is None:
        blocks = [h4_transform(make_block(m, k)) for k, m in list(enumerate(zero_syndrome_codewords()))[1:65]]
    require_ok(len(set(tuple(w) for w in blocks)) == len(blocks), "C_DISTINCT", "collapse witnesses must be distinct")
    collapsed = 0
    images = set()
    for w in blocks:
        require_ok(normsq(w) > 0, "C_NONZERO", "collapse witness must be a nonzero block")
        z = up_row_zero(w)
        require_ok(normsq(z) == 0, "C_ZERO_NORM", "up_row_zero must zero the L2^2 norm")
        collapsed = collapsed + (1 if normsq(z) < normsq(w) else 0)
        images.add(tuple(z))
    require_ok(len(images) == 1, "C_COLLISION", "up_row_zero must collapse all witnesses to one image")
    return {
        "retention_pairs": [[pair[0], pair[1]] for pair in retention],
        "retention_den": den,
        "retention_drops": drops,
        "inkling": frac_str(Fraction(inkling[0], inkling[1])),
        "witness_blocks": len(blocks),
        "strict_collapses": collapsed,
        "distinct_images": len(images),
    }


def carving_step(g, u, p):
    return max(g - u - p, Fraction(0))


def gate_grad(g, raw):
    return Fraction(0) if g == 0 else raw


def hold_open_step(g, u, p, eps):
    n = carving_step(g, u, p)
    return eps if n <= 0 else n


def verify_carving(removed=CARVING_REMOVED, kept=CARVING_KEPT, half=CARVING_HALF, tenth=CARVING_TENTH,
                   pressures=CANON_PRESSURES, reopen=REOPEN_GRAD, eps=HOLD_OPEN):
    for p in pressures:
        require_ok(p >= 0, "K_PRESSURE_NEG", "gate pressure pi must be nonnegative")
    require_ok(reopen < 0, "K_REOPEN_GRAD", "reopening gradient move must be negative")
    require_ok(half[1] <= half[0] <= kept, "K_KEPT_HALF", "half-peak reopen counts must be nested")
    require_ok(tenth[1] <= tenth[0] <= half[0], "K_KEPT_TENTH", "tenth-peak reopen counts must be nested")
    locked_reopened = 0
    held_reopened = 0
    for i in range(removed):
        g = Fraction(0)
        g_hold = Fraction(0)
        raw = reopen * (1 + i % 3)
        for p in pressures:
            g = carving_step(g, gate_grad(g, raw), p)
            g_hold = hold_open_step(g_hold, raw, p, eps)
        locked_reopened = locked_reopened + (1 if g > 0 else 0)
        held_reopened = held_reopened + (1 if g_hold > eps else 0)
    require_ok(locked_reopened == 0, "K_LOCKOUT", "zeroed gates must never reopen")
    require_ok(held_reopened == removed, "K_HOLD_OPEN", "hold_open gates must all reopen")
    g_pos = carving_step(Fraction(1, 2), reopen, Fraction(1, 10))
    return {
        "removed": removed,
        "kept": kept,
        "locked_reopened": str(locked_reopened) + "/" + str(removed),
        "locked_reopened_count": locked_reopened,
        "hold_open_eps": frac_str(eps),
        "hold_open_reopened_count": held_reopened,
        "kept_half_reopen": str(half[1]) + "/" + str(half[0]),
        "kept_tenth_reopen": str(tenth[1]) + "/" + str(tenth[0]),
        "horizon": len(pressures),
        "positive_gate_after_step": frac_str(g_pos),
    }


def verify_duality(n=CARVING_REMOVED, scale=CANON_SCALE):
    codewords = zero_syndrome_codewords()
    recovered = 0
    for k in range(n):
        c = make_block(codewords[(k * 97 + 3) % 2048], k)
        base = ternary_weights(1, c, scale)[1]
        t = switch_state(1, 0)
        t = switch_state(t, 1)
        recovered = recovered + (1 if ternary_weights(t, c, scale)[1] == base else 0)
    require_ok(recovered == n, "D_RECOVERY", "every soft-gated A_16 block must recover exactly")
    return {
        "ternary_recovered": str(recovered) + "/" + str(n),
        "ternary_lockout": str(n - recovered) + "/" + str(n),
        "carving_lockout": str(n) + "/" + str(n),
    }


# ---------------------------------------------------------------------------
# Strict Zone-B I/O and JSON
# ---------------------------------------------------------------------------

def track_base():
    here = Path(__file__).parent
    if here.is_absolute():
        return here
    return Path(os.getcwd()).joinpath(here)


def _unnormalized_parts(path):
    raw = Path(str(path))
    if raw.is_absolute():
        return raw.parts
    return Path.cwd().joinpath(raw).parts


def check_no_symlink_ancestors(path):
    parts = _unnormalized_parts(path)
    cur = Path("/")
    for seg in parts[1:]:
        cur = cur.joinpath(seg)
        require_ok(cur.is_symlink() is False, "SYMLINK_ANCESTOR", "symlink in path ancestors")


def _open_parent_nofollow(path, code="READ_OPEN"):
    parts = _unnormalized_parts(path)
    require_ok(len(parts) >= 2, code, "cannot open file")
    dfd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        for seg in parts[1:-1]:
            nxt = os.open(seg, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dfd)
            os.close(dfd)
            dfd = nxt
    except OSError:
        os.close(dfd)
        raise VerifyError(code, "cannot open file")
    return dfd, parts[-1]


def read_strict_bytes(path):
    check_no_symlink_ancestors(path)
    dfd, leaf = _open_parent_nofollow(path)
    try:
        try:
            fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dfd)
        except OSError:
            raise VerifyError("READ_OPEN", "cannot open file")
    finally:
        os.close(dfd)
    chunks = []
    try:
        info = os.fstat(fd)
        require_ok(stat.S_ISREG(info.st_mode), "NOT_REGULAR", "not a regular file")
        total = 0
        while True:
            chunk = os.read(fd, 65536)
            if chunk == b"":
                break
            total = total + len(chunk)
            require_ok(total <= MAX_BYTES, "TOO_LARGE", "file exceeds size cap")
            chunks.append(chunk)
    finally:
        os.close(fd)
    return b"".join(chunks)


def decode_strict_text(raw):
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise VerifyError("BAD_UTF8", "invalid utf-8")
    require_ok(text.startswith("\ufeff") is False, "BAD_BOM", "byte order mark rejected")
    require_ok("\r" not in text, "BAD_LF", "carriage return rejected")
    require_ok(text.endswith("\n"), "BAD_FINAL_LF", "missing final newline")
    return text


def read_strict_text(path):
    return decode_strict_text(read_strict_bytes(path))


def scan_json_text(text):
    depth = 0
    in_str = False
    escaped = False
    digit_run = 0
    saw_float = False
    prev_digit = False
    for ch in text:
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            digit_run = 0
            prev_digit = False
            continue
        if ch == '"':
            in_str = True
            digit_run = 0
            prev_digit = False
            continue
        if ch == "{":
            depth = depth + 1
        if ch == "[":
            depth = depth + 1
        if ch == "}":
            depth = depth - 1
        if ch == "]":
            depth = depth - 1
        require_ok(depth <= MAX_DEPTH, "DEPTH", "nesting depth exceeded")
        require_ok(depth >= 0, "UNBALANCED", "nesting depth went negative")
        if ch.isdigit():
            digit_run = digit_run + 1
            require_ok(digit_run <= MAX_INT_DIGITS, "INT_DIGITS", "integer digit bound exceeded")
            prev_digit = True
        else:
            if ch == ".":
                if prev_digit:
                    saw_float = True
            if ch == "e":
                if prev_digit:
                    saw_float = True
            if ch == "E":
                if prev_digit:
                    saw_float = True
            digit_run = 0
            prev_digit = False
    require_ok(depth == 0, "UNBALANCED_FINAL", "unbalanced nesting")
    require_ok(saw_float is False, "FLOAT_TOKEN", "floating token rejected")
    require_ok(in_str is False, "UNTERMINATED_JSON_STRING", "unterminated json string")


def dup_free_object(pairs):
    out = {}
    for pair in pairs:
        require_ok(pair[0] not in out, "DUP_KEY", "duplicate json key")
        out[pair[0]] = pair[1]
    return out


def reject_constant(_value):
    raise VerifyError("BAD_CONSTANT", "non-json constant rejected")


def parse_strict_json(text):
    scan_json_text(text)
    try:
        data = json.loads(text, object_pairs_hook=dup_free_object, parse_constant=reject_constant)
    except VerifyError:
        raise
    except (json.JSONDecodeError, ValueError):
        raise VerifyError("BAD_JSON", "malformed json")
    require_ok(isinstance(data, dict), "NOT_OBJECT", "top level must be object")
    return data


def read_strict_json(path):
    return parse_strict_json(read_strict_text(path))


def _json_exact_equal(lhs, rhs):
    """Type-strict JSON equality: `type(lhs) is type(rhs)` at every node, so True != 1 and False != 0."""
    if type(lhs) is not type(rhs):
        return False
    if type(lhs) is dict:
        if _json_exact_equal(sorted(lhs), sorted(rhs)) is False:
            return False
        for key in lhs:
            if _json_exact_equal(lhs[key], rhs[key]) is False:
                return False
        return True
    if type(lhs) is list:
        if len(lhs) != len(rhs):
            return False
        for idx in range(len(lhs)):
            if _json_exact_equal(lhs[idx], rhs[idx]) is False:
                return False
        return True
    return lhs == rhs


# ---------------------------------------------------------------------------
# Lean package
# ---------------------------------------------------------------------------

def strip_lean_comments_and_strings(src):
    out = []
    i = 0
    n = len(src)
    depth = 0
    in_str = False
    while i < n:
        ch = src[i]
        nxt = ""
        if i + 1 < n:
            nxt = src[i + 1]
        if depth > 0:
            if ch == "/":
                if nxt == "-":
                    depth = depth + 1
                    out.append("  ")
                    i = i + 2
                    continue
            if ch == "-":
                if nxt == "/":
                    depth = depth - 1
                    out.append("  ")
                    i = i + 2
                    continue
            if ch == "\n":
                out.append("\n")
            else:
                out.append(" ")
            i = i + 1
            continue
        if in_str:
            if ch == "\\":
                if i + 1 < n:
                    out.append("  ")
                    i = i + 2
                    continue
            if ch == '"':
                in_str = False
                out.append(" ")
                i = i + 1
                continue
            if ch == "\n":
                out.append("\n")
            else:
                out.append(" ")
            i = i + 1
            continue
        if ch == "-":
            if nxt == "-":
                while i < n:
                    if src[i] == "\n":
                        break
                    out.append(" ")
                    i = i + 1
                continue
        if ch == "/":
            if nxt == "-":
                depth = 1
                out.append("  ")
                i = i + 2
                continue
        if ch == '"':
            in_str = True
            out.append(" ")
            i = i + 1
            continue
        out.append(ch)
        i = i + 1
    require_ok(depth == 0, "UNTERMINATED_BLOCK_COMMENT", "unterminated block comment")
    require_ok(in_str is False, "UNTERMINATED_STRING_LITERAL", "unterminated string literal")
    return "".join(out)


def validate_axiom_lines(lines, expected_count=EXPECTED_THEOREMS):
    non_empty = []
    for ln in lines:
        stripped = ln.strip()
        if stripped != "":
            non_empty.append(stripped)
    require_ok(len(non_empty) == expected_count, "AXIOM_LINE_COUNT", "axiom line count mismatch")
    zero_count = 0
    listed_count = 0
    for ln in non_empty:
        if "does not depend on any axioms" in ln:
            zero_count = zero_count + 1
            continue
        require_ok("depends on axioms:" in ln, "AXIOM_LINE_FORMAT", "unrecognized axiom line")
        rhs = ln.split("depends on axioms:", 1)[1].strip()
        require_ok(rhs.startswith("["), "AXIOM_LBRACKET", "missing opening bracket")
        require_ok(rhs.endswith("]"), "AXIOM_RBRACKET", "missing closing bracket")
        names = []
        for part in rhs[1:-1].split(","):
            item = part.strip()
            if item != "":
                names.append(item)
        for name in names:
            require_ok(name in ALLOWED_AXIOMS, "FORBIDDEN_AXIOM", "axiom not allowlisted")
        listed_count = listed_count + 1
    return {"zero_axiom": zero_count, "listed": listed_count}


LEAN_PACKAGE_FILES = (
    ("lean-toolchain",),
    ("lakefile.toml",),
    ("lake-manifest.json",),
    ("TernaryA16Carving.lean",),
    ("AxiomAudit.lean",),
    ("TernaryA16Carving", "Basic.lean"),
)


def stage_lean_package(lean_dir, work):
    """Stage exactly the pinned Lean package files into a fresh writable tree.

    Every source is read through read_strict_bytes (symlink-free ancestors,
    O_NOFOLLOW, O_NONBLOCK, S_ISREG, MAX_BYTES) before anything is created, so a
    planted symlink/FIFO/oversized file aborts staging; read-only source modes
    (e.g. a chmod -R a-w snapshot) are never propagated to the staged tree.
    """
    staged = []
    for parts in LEAN_PACKAGE_FILES:
        staged.append((parts, read_strict_bytes(Path(lean_dir).joinpath(*parts))))
    os.mkdir(str(work))
    os.chmod(str(work), 0o755)
    sub = work.joinpath("TernaryA16Carving")
    os.mkdir(str(sub))
    os.chmod(str(sub), 0o755)
    for parts, raw in staged:
        dest = work.joinpath(*parts)
        dest.write_bytes(raw)
        os.chmod(str(dest), 0o644)
    return len(staged)


def default_live_lean_runner(lean_dir):
    home = os.path.expanduser("~")
    env = {}
    env["PATH"] = str(Path(home).joinpath(".elan", "bin")) + ":/usr/bin:/bin"
    env["HOME"] = home
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp).joinpath("lean")
        stage_lean_package(lean_dir, work)
        version = subprocess.run(["lean", "--version"], cwd=str(work), env=env, capture_output=True, text=True, timeout=60, check=False)
        build = subprocess.run(["lake", "build"], cwd=str(work), env=env, capture_output=True, text=True, timeout=300, check=False)
        audit = subprocess.run(["lake", "env", "lean", "AxiomAudit.lean"], cwd=str(work), env=env, capture_output=True, text=True, timeout=180, check=False)
    return {
        "version_rc": version.returncode,
        "version_output": version.stdout + version.stderr,
        "build_rc": build.returncode,
        "build_output": build.stdout + build.stderr,
        "axiom_rc": audit.returncode,
        "axiom_stdout": audit.stdout,
    }


def verify_lean_package(base, run_live_lean=False, lean_runner=None):
    lean_dir = Path(base).joinpath("lean")
    require_ok(lean_dir.is_dir(), "MISSING_LEAN_DIR", "lean directory missing")
    toolchain = read_strict_text(lean_dir.joinpath("lean-toolchain")).strip()
    require_ok(toolchain == "leanprover/lean4:v4.36.0", "BAD_TOOLCHAIN", "toolchain mismatch")
    manifest = read_strict_json(lean_dir.joinpath("lake-manifest.json"))
    require_ok(manifest.get("packages") == [], "NONEMPTY_PACKAGES", "lake packages must be empty")
    lakefile = read_strict_text(lean_dir.joinpath("lakefile.toml"))
    require_ok('name = "TernaryA16Carving"' in lakefile, "BAD_LAKEFILE", "lakefile package name mismatch")
    lean_files = sorted(p for p in lean_dir.rglob("*.lean") if ".lake" not in p.parts)
    require_ok(len(lean_files) == 3, "LEAN_FILE_COUNT", "lean file count mismatch")
    total_loc = 0
    pub_thms = []
    audited = []
    for lf in lean_files:
        src = read_strict_text(lf)
        total_loc = total_loc + len(src.splitlines())
        cleaned = strip_lean_comments_and_strings(src)
        require_ok(_FORBIDDEN_LEAN_RE.search(cleaned) is None, "LEAN_ESCAPE", "forbidden lean construct")
        for match in _ANY_IMPORT_RE.finditer(cleaned):
            mod = match.group(1).strip()
            require_ok(_VALID_IMPORT_RE.fullmatch(mod) is not None, "BAD_IMPORT", "forbidden lean import")
        if lf.name == "Basic.lean":
            pub_thms = _PUB_THM_RE.findall(cleaned)
        if lf.name == "AxiomAudit.lean":
            audited = _PRINT_RE.findall(cleaned)
    require_ok(len(pub_thms) == EXPECTED_THEOREMS, "THM_COUNT", "public theorem count mismatch")
    require_ok(len(audited) == EXPECTED_THEOREMS, "AUDIT_COUNT", "audited theorem count mismatch")
    require_ok(sorted(pub_thms) == sorted(audited), "AUDIT_SET", "audited set must be 1:1 with public theorems")
    info = {}
    info["toolchain"] = toolchain
    info["lean_files"] = len(lean_files)
    info["lean_loc"] = total_loc
    info["public_theorems"] = len(pub_thms)
    info["theorem_names"] = list(pub_thms)
    runner = lean_runner
    if runner is None:
        if run_live_lean:
            runner = default_live_lean_runner
    if runner is not None:
        result = runner(lean_dir)
        require_ok(result["version_rc"] == 0, "LEAN_VERSION_RC", "lean version command failed")
        require_ok("4.36.0" in result["version_output"], "BAD_LEAN_VERSION", "lean version mismatch")
        require_ok(result["build_rc"] == 0, "LAKE_BUILD_FAIL", "lake build failed")
        require_ok("warning:" not in result["build_output"].lower(), "LAKE_BUILD_WARNING", "lake build warning")
        require_ok(result["axiom_rc"] == 0, "AXIOM_RC", "axiom audit command failed")
        validate_axiom_lines(result["axiom_stdout"].splitlines(), EXPECTED_THEOREMS)
    return info


# ---------------------------------------------------------------------------
# Payload and CLI
# ---------------------------------------------------------------------------

def build_payload(base=None, run_live_lean=False, lean_runner=None):
    if base is None:
        base = track_base()
    codewords = zero_syndrome_codewords()
    payload = {}
    payload["schema"] = "s163.ternary_a16_carving.v1"
    payload["track"] = "S163"
    payload["queue_id"] = "EXP-Q42a"
    payload["contrast"] = "Model Carving arXiv 2611.27962 g = 0 lockout"
    payload["pillar1_a16_complement"] = verify_pillar1()
    payload["pillar2_blockh4_scale_flip"] = verify_pillar2()
    payload["pillar2_ternary"] = verify_ternary([make_block(codewords[(k * 61 + 1) % 2048], k) for k in range(32)])
    payload["pillar3_weft_rollback"] = verify_pillar3()
    payload["pillar4_chantless"] = verify_chantless()
    payload["pillar4_carving"] = verify_carving()
    payload["pillar4_duality"] = verify_duality()
    payload["lean"] = verify_lean_package(base, run_live_lean=run_live_lean, lean_runner=lean_runner)
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    payload["payload_sha256"] = hashlib.sha256(canon).hexdigest()
    return payload


def canonical_cert_bytes(payload):
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    return text.encode("utf-8")


def _temp_token():
    return os.urandom(8).hex()


def _check_write_leaf(dfd, leaf):
    try:
        info = os.stat(leaf, dir_fd=dfd, follow_symlinks=False)
    except FileNotFoundError:
        return 0
    except OSError:
        raise VerifyError("WRITE_STAT", "cannot stat certificate target")
    require_ok(stat.S_ISREG(info.st_mode), "WRITE_NOT_REGULAR", "certificate target must be a regular file")
    require_ok(info.st_nlink == 1, "WRITE_HARDLINK", "certificate target must not be hard-linked")
    return 1


def _unlink_quiet(dfd, name):
    try:
        os.unlink(name, dir_fd=dfd)
    except OSError:
        return 0
    return 1


def _write_temp_at(dfd, leaf, raw):
    name = "." + leaf + "." + _temp_token() + ".tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        fd = os.open(name, flags, 0o600, dir_fd=dfd)
    except OSError:
        raise VerifyError("WRITE_TEMP", "cannot create temporary certificate file")
    done = False
    try:
        view = memoryview(raw)
        while len(view) > 0:
            n = os.write(fd, view)
            require_ok(n in range(1, len(view) + 1), "WRITE_SHORT", "certificate write made no progress")
            view = view[n:]
        os.fchmod(fd, 0o644)
        os.fsync(fd)
        done = True
    finally:
        os.close(fd)
        if done is False:
            _unlink_quiet(dfd, name)
    return name


def write_cert(path, payload):
    raw = canonical_cert_bytes(payload)
    check_no_symlink_ancestors(path)
    dfd, leaf = _open_parent_nofollow(path, "WRITE_OPEN")
    try:
        _check_write_leaf(dfd, leaf)
        tmp = _write_temp_at(dfd, leaf, raw)
        try:
            os.replace(tmp, leaf, src_dir_fd=dfd, dst_dir_fd=dfd)
        except OSError:
            _unlink_quiet(dfd, tmp)
            raise VerifyError("WRITE_REPLACE", "atomic certificate replace failed")
        os.fsync(dfd)
    finally:
        os.close(dfd)


def emit_error(code, message):
    err = {}
    err["status"] = "FAIL"
    err["code"] = code
    err["message"] = message
    line = json.dumps(err, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    os.write(2, line.encode("utf-8"))


def resolve_cert_path(base, given):
    if given is None:
        return Path(base).joinpath(CERT_NAME)
    cand = Path(given)
    if cand.is_absolute():
        return cand
    return Path(base).joinpath(cand)


def run_cli(argv=None):
    parser = argparse.ArgumentParser(description="Track S163 EXP-Q42a ternary A_16 vs Model Carving lockout verifier")
    parser.add_argument("--check", nargs="?", const=CERT_NAME, default=None)
    parser.add_argument("--write", nargs="?", const=CERT_NAME, default=None)
    parser.add_argument("--standalone", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_out")
    parser.add_argument("--live-lean", action="store_true")
    args = parser.parse_args(argv)
    modes = 0
    if args.check is not None:
        modes = modes + 1
    if args.write is not None:
        modes = modes + 1
    if args.standalone:
        modes = modes + 1
    require_ok(modes <= 1, "MODE_COUNT", "at most one mode allowed")
    check_given = args.check
    if modes == 0:
        check_given = CERT_NAME
    base = track_base()
    payload = build_payload(base=base, run_live_lean=args.live_lean)
    if args.write is not None:
        write_cert(resolve_cert_path(base, args.write), payload)
    if check_given is not None:
        cert = read_strict_json(resolve_cert_path(base, check_given))
        require_ok(_json_exact_equal(cert, payload), "CERT_MISMATCH", "certificate mismatch")
    if args.json_out:
        line = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        os.write(1, line.encode("utf-8"))
    else:
        os.write(1, b"PASS s163.ternary_a16_carving.v1\n")
    return 0


def main(argv=None):
    try:
        return run_cli(argv)
    except VerifyError as exc:
        emit_error(exc.code, str(exc))
        return 2
    except RecursionError:
        emit_error("RECURSION", "recursion limit exceeded")
        return 2
    except OSError as exc:
        emit_error("OS_ERROR", type(exc).__name__)
        return 2
    except subprocess.SubprocessError as exc:
        emit_error("SUBPROCESS_ERROR", type(exc).__name__)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
