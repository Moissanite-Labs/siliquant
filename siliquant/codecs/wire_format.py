"""Wire header codecs (.slqi/.slgb/.slqw), IEEE CRC32, 64-bit [16,11,4] SECDED, and IRIS canary contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
import struct
from typing import Sequence

from ._validation import require

SLQI_MAGIC = 0x534C5149  # "SLQI"
SLQI_VERSION = 1

SLGB_MAGIC = 0x534C4734  # "SLG4"
SLGB_VERSION = 1

SLQW_MAGIC = 0x534C5157  # "SLQW"
SLQW_VERSION = 1

IRIS_WEYL_MIX = 0x9E3779B9
MAX_SILQI_VRAM_CAP_BYTES = 6_657_199_308

_SLQI_HEADER_FMT = "<8I10H5I6Q2I"
_SLGB_HEADER_FMT = "<7I6H2fQ2I"
_SLQW_HEADER_FMT = "<4I8H3IfQ2I"
_CANARY64_FMT = "<16I"


class WatchdogAction(IntEnum):
    """Fail-closed runtime integrity watchdog actions."""

    CONTINUE = 0
    RAMP_SLEW_CAPPED = 1
    SAFE_HALT_DED_ROLLBACK = 2
    SAFE_HALT_DEADLINE_OVERRUN = 3
    SAFE_HALT_BOS_CONTRACT = 4
    SAFE_HALT_RICCATI_INSTABILITY = 5
    REFUSAL_WORK_EXHAUSTED = 6


def crc32_ieee(data: bytes) -> int:
    """Compute canonical IEEE 802.3 CRC32 (0xEDB88320 reflected polynomial)."""
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            mask = -(crc & 1) & 0xFFFFFFFF
            crc = (crc >> 1) ^ (0xEDB88320 & mask)
    return (crc ^ 0xFFFFFFFF) & 0xFFFFFFFF


def secded_syndrome_u64(w64: int) -> int:
    """Compute 5-bit Extended Hamming [16, 11, 4] syndrome on the 16 nibble LSBs of a uint64."""
    w = int(w64) & 0xFFFFFFFFFFFFFFFF
    w0 = w & 0xFFFFFFFF
    w1 = (w >> 32) & 0xFFFFFFFF
    px = (w0 ^ w1) & 0x11111111
    p0 = px.bit_count() & 1
    s0 = (px & 0x10101010).bit_count() & 1
    s1 = (px & 0x11001100).bit_count() & 1
    s2 = (px & 0x11110000).bit_count() & 1
    s3 = (w1 & 0x11111111).bit_count() & 1
    return p0 | (s0 << 1) | (s1 << 2) | (s2 << 3) | (s3 << 4)


def secded_encode_u64(u11: int, high_nibbles_16: Sequence[int] | None = None) -> int:
    """Encode 11 information bits + optional upper 3-bit nibble payloads into a 0-syndrome uint64."""
    info = int(u11) & 0x7FF
    b = [0] * 16
    info_pos = (3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15)
    for j, pos in enumerate(info_pos):
        b[pos] = (info >> j) & 1
    b[1] = b[3] ^ b[5] ^ b[7] ^ b[9] ^ b[11] ^ b[13] ^ b[15]
    b[2] = b[3] ^ b[6] ^ b[7] ^ b[10] ^ b[11] ^ b[14] ^ b[15]
    b[4] = b[5] ^ b[6] ^ b[7] ^ b[12] ^ b[13] ^ b[14] ^ b[15]
    b[8] = b[9] ^ b[10] ^ b[11] ^ b[12] ^ b[13] ^ b[14] ^ b[15]
    b[0] = sum(b[1:]) & 1

    hi_nibs = list(high_nibbles_16) if high_nibbles_16 is not None else [0] * 16
    require(len(hi_nibs) == 16, "SECDED_NIBS", "expected 16 nibble upper halves")
    w64 = 0
    for i in range(16):
        nib = ((int(hi_nibs[i]) & 0xE) | (b[i] & 1)) & 0xF
        w64 |= nib << (4 * i)
    require(secded_syndrome_u64(w64) == 0, "SECDED_ENC", "encoded uint64 failed zero-syndrome invariant")
    return w64


def secded_heal_u64(w64: int) -> tuple[int, int, int]:
    """Verify/heal 16-nibble LSB Extended Hamming codeword: returns `(healed_u64, sec_cnt, ded_cnt)`."""
    w = int(w64) & 0xFFFFFFFFFFFFFFFF
    syn = secded_syndrome_u64(w)
    if syn == 0:
        return w, 0, 0
    if (syn & 1) != 0:
        pos = (syn >> 1) & 15
        healed = w ^ (1 << (pos * 4))
        require(secded_syndrome_u64(healed) == 0, "SECDED_HEAL", "SEC flip did not clear syndrome")
        return healed, 1, 0
    return w, 0, 1


def iris_mix_word_u32(state: int, word: int) -> int:
    """Apply IRIS golden-ratio Weyl + Murmur3 fmix32 avalanche step matching sm_86 device primitive."""
    s = int(state) & 0xFFFFFFFF
    w = int(word) & 0xFFFFFFFF
    x = (s ^ ((w + IRIS_WEYL_MIX + ((s << 6) & 0xFFFFFFFF) + (s >> 2)) & 0xFFFFFFFF)) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & 0xFFFFFFFF
    x ^= x >> 16
    return x & 0xFFFFFFFF


def iris_finalize_words_u32(words: Sequence[int], seed: int = 0x5185A17B) -> int:
    """Fold a stream of uint32 words into a 32-bit IRIS runtime signature."""
    h = int(seed) & 0xFFFFFFFF
    for w in words:
        h = iris_mix_word_u32(h, int(w))
    return h


@dataclass(frozen=True)
class RazorCanary64:
    """64-byte cacheline-aligned telemetry canary state."""

    sec_count_u32: int = 0
    ded_count_u32: int = 0
    epoch_u32: int = 0
    rollback_token_u32: int = 0
    non_lsb_canary_u32: int = 0
    last_syndrome_u32: int = 0
    ewma_ber_q16_u32: int = 0
    clean_streak_u32: int = 0
    hold_remaining_u32: int = 0
    ded_latch_remaining_u32: int = 0
    iris_runtime_crc_u32: int = 0
    iris_mismatch_count_u32: int = 0
    poisoned_u32: int = 0
    adaptive_k_halted_u32: int = 0
    refusal_work_count_u32: int = 0
    reserved_u32: int = 0

    def pack(self) -> bytes:
        """Pack into exact 64 little-endian bytes."""
        blob = struct.pack(
            _CANARY64_FMT,
            self.sec_count_u32,
            self.ded_count_u32,
            self.epoch_u32,
            self.rollback_token_u32,
            self.non_lsb_canary_u32,
            self.last_syndrome_u32,
            self.ewma_ber_q16_u32,
            self.clean_streak_u32,
            self.hold_remaining_u32,
            self.ded_latch_remaining_u32,
            self.iris_runtime_crc_u32,
            self.iris_mismatch_count_u32,
            self.poisoned_u32,
            self.adaptive_k_halted_u32,
            self.refusal_work_count_u32,
            self.reserved_u32,
        )
        require(len(blob) == 64, "CANARY_LEN", "canary must be 64 bytes")
        return blob

    @classmethod
    def unpack(cls, raw: bytes) -> "RazorCanary64":
        """Unpack 64 bytes into RazorCanary64."""
        require(len(raw) == 64, "CANARY_LEN", f"expected 64 bytes, got {len(raw)}")
        return cls(*struct.unpack(_CANARY64_FMT, raw))


def evaluate_razor_watchdog(
    canary: RazorCanary64,
    expected_iris_crc: int,
    *,
    strict_iris: bool = True,
) -> WatchdogAction:
    """Fail-closed watchdog check trapping DED, poison flags, IRIS CRC mismatch, or refusal-work exhaustion."""
    if canary.poisoned_u32 != 0 or canary.ded_count_u32 != 0 or canary.non_lsb_canary_u32 != 0:
        return WatchdogAction.SAFE_HALT_DED_ROLLBACK
    if canary.iris_mismatch_count_u32 != 0:
        return WatchdogAction.SAFE_HALT_DED_ROLLBACK
    if strict_iris and expected_iris_crc != 0 and (canary.iris_runtime_crc_u32 & 0xFFFFFFFF) != (expected_iris_crc & 0xFFFFFFFF):
        return WatchdogAction.SAFE_HALT_DED_ROLLBACK
    if canary.refusal_work_count_u32 != 0:
        return WatchdogAction.REFUSAL_WORK_EXHAUSTED
    return WatchdogAction.CONTINUE


@dataclass(frozen=True)
class SlqiHeader:
    """128-byte `.slqi` wire header descriptor."""

    magic: int = SLQI_MAGIC
    version: int = SLQI_VERSION
    eos_token_id: int = 200006
    bos_token_id: int = 0
    bos_fail_closed: int = 1
    pad_token_id: int = 0
    unpadded_vocab_size: int = 200058
    padded_vocab_size: int = 201024
    hidden_size: int = 2304
    num_layers: int = 24
    loop_start: int = 6
    loop_end: int = 18
    loop_passes_base: int = 3
    max_loop_passes: int = 5
    num_experts: int = 128
    top_k_experts: int = 8
    moe_inter: int = 1152
    loop_168_bf16_kept: int = 168
    wire_variant: int = 0
    num_linears: int = 145
    num_expert_slots: int = 3072
    hot_expert_slots: int = 1850
    cold_expert_slots: int = 1222
    compute_wire_bytes: int = 10_759_099_296
    text_wire_bytes: int = 11_685_422_496
    shared_backbone_bytes: int = 821_744_544
    hot_bank_vram_bytes: int = 5_524_070_400
    cold_bank_ram_bytes: int = 3_648_872_448
    small_backbone_bytes: int = 78_076_416
    flags: int = 0
    header_crc32: int = 0


def encode_slqi_header(hdr: SlqiHeader) -> bytes:
    """Serialize a 128-byte `.slqi` header with trailing IEEE CRC32 over bytes [0..124)."""
    prefix = struct.pack(
        "<8I10H5I6QI",
        hdr.magic,
        hdr.version,
        hdr.eos_token_id,
        hdr.bos_token_id,
        hdr.bos_fail_closed,
        hdr.pad_token_id,
        hdr.unpadded_vocab_size,
        hdr.padded_vocab_size,
        hdr.hidden_size,
        hdr.num_layers,
        hdr.loop_start,
        hdr.loop_end,
        hdr.loop_passes_base,
        hdr.max_loop_passes,
        hdr.num_experts,
        hdr.top_k_experts,
        hdr.moe_inter,
        hdr.loop_168_bf16_kept,
        hdr.wire_variant,
        hdr.num_linears,
        hdr.num_expert_slots,
        hdr.hot_expert_slots,
        hdr.cold_expert_slots,
        hdr.compute_wire_bytes,
        hdr.text_wire_bytes,
        hdr.shared_backbone_bytes,
        hdr.hot_bank_vram_bytes,
        hdr.cold_bank_ram_bytes,
        hdr.small_backbone_bytes,
        hdr.flags,
    )
    crc = crc32_ieee(prefix)
    out = prefix + struct.pack("<I", crc)
    require(len(out) == 128, "SLQI_LEN", f"expected 128 bytes, got {len(out)}")
    return out


def decode_slqi_header(raw: bytes) -> SlqiHeader:
    """Deserialize a 128-byte `.slqi` header."""
    require(len(raw) == 128, "SLQI_LEN", f"expected 128 bytes, got {len(raw)}")
    return SlqiHeader(*struct.unpack(_SLQI_HEADER_FMT, raw))


def validate_slqi_header(
    raw_or_hdr: bytes | SlqiHeader,
    tokens: Sequence[int] | None = None,
    *,
    verify_crc: bool = True,
) -> SlqiHeader:
    """Fail-closed validation of `.slqi` header invariants, CRC32, VRAM cap, and optional token BOS/vocab."""
    if isinstance(raw_or_hdr, (bytes, bytearray, memoryview)):
        raw = bytes(raw_or_hdr)
        hdr = decode_slqi_header(raw)
        if verify_crc:
            expected_crc = crc32_ieee(raw[:124])
            require(hdr.header_crc32 == expected_crc, "SLQI_CRC32", "CRC32 mismatch on .slqi header")
    else:
        hdr = raw_or_hdr
    require(hdr.magic == SLQI_MAGIC, "SLQI_MAGIC", f"bad magic {hdr.magic:#x}")
    require(hdr.version == SLQI_VERSION, "SLQI_VER", f"bad version {hdr.version}")
    require(hdr.unpadded_vocab_size > 0 and hdr.padded_vocab_size >= hdr.unpadded_vocab_size, "SLQI_VOCAB", "invalid vocab")
    require(hdr.num_layers > 0 and hdr.loop_start < hdr.loop_end <= hdr.num_layers, "SLQI_LOOP", "invalid loop bounds")
    require(hdr.hot_expert_slots + hdr.cold_expert_slots == hdr.num_expert_slots, "SLQI_SLOTS", "slot sum mismatch")
    require(
        hdr.shared_backbone_bytes + hdr.hot_bank_vram_bytes <= MAX_SILQI_VRAM_CAP_BYTES,
        "SLQI_VRAM",
        "VRAM budget exceeds 6.2 GiB cap",
    )
    if tokens is not None:
        require(len(tokens) > 0, "SLQI_TOK_EMPTY", "token sequence empty")
        if hdr.bos_fail_closed:
            require(int(tokens[0]) == hdr.bos_token_id, "SLQI_BOS", f"expected BOS {hdr.bos_token_id}, got {tokens[0]}")
        for tok in tokens:
            require(0 <= int(tok) < hdr.unpadded_vocab_size, "SLQI_TOK_OOB", f"token {tok} >= {hdr.unpadded_vocab_size}")
    return hdr


@dataclass(frozen=True)
class SlgbHeader:
    """64-byte `.slgb` wire header descriptor."""

    magic: int = SLGB_MAGIC
    version: int = SLGB_VERSION
    bos_token_id: int = 2
    eos_token_id_0: int = 1
    eos_token_id_1: int = 106
    pad_token_id: int = 0
    vocab_size: int = 262144
    hidden_size: int = 1536
    num_layers: int = 35
    num_kv_shared_layers: int = 20
    group_size: int = 32
    num_live_linears: int = 276
    num_dead_kv_omitted: int = 60
    final_logit_softcapping: float = 30.0
    rms_norm_eps: float = 1.0e-6
    total_wire_bytes: int = 1_055_711_232
    flags: int = 3
    header_crc32: int = 0


def encode_slgb_header(hdr: SlgbHeader) -> bytes:
    """Serialize a 64-byte `.slgb` header with trailing IEEE CRC32 over bytes [0..60)."""
    require(math.isfinite(hdr.final_logit_softcapping) and math.isfinite(hdr.rms_norm_eps), "SLGB_FLOAT", "non-finite header float")
    prefix = struct.pack(
        "<7I6H2fQI",
        hdr.magic,
        hdr.version,
        hdr.bos_token_id,
        hdr.eos_token_id_0,
        hdr.eos_token_id_1,
        hdr.pad_token_id,
        hdr.vocab_size,
        hdr.hidden_size,
        hdr.num_layers,
        hdr.num_kv_shared_layers,
        hdr.group_size,
        hdr.num_live_linears,
        hdr.num_dead_kv_omitted,
        hdr.final_logit_softcapping,
        hdr.rms_norm_eps,
        hdr.total_wire_bytes,
        hdr.flags,
    )
    crc = crc32_ieee(prefix)
    out = prefix + struct.pack("<I", crc)
    require(len(out) == 64, "SLGB_LEN", f"expected 64 bytes, got {len(out)}")
    return out


def decode_slgb_header(raw: bytes) -> SlgbHeader:
    """Deserialize a 64-byte `.slgb` header."""
    require(len(raw) == 64, "SLGB_LEN", f"expected 64 bytes, got {len(raw)}")
    return SlgbHeader(*struct.unpack(_SLGB_HEADER_FMT, raw))


def validate_slgb_header(
    raw_or_hdr: bytes | SlgbHeader,
    tokens: Sequence[int] | None = None,
    *,
    verify_crc: bool = True,
) -> SlgbHeader:
    """Fail-closed validation of `.slgb` header invariants, CRC32, BOS=2 contract, and dead-KV omission."""
    if isinstance(raw_or_hdr, (bytes, bytearray, memoryview)):
        raw = bytes(raw_or_hdr)
        hdr = decode_slgb_header(raw)
        if verify_crc:
            require(hdr.header_crc32 == crc32_ieee(raw[:60]), "SLGB_CRC32", "CRC32 mismatch on .slgb header")
    else:
        hdr = raw_or_hdr
    require(hdr.magic == SLGB_MAGIC, "SLGB_MAGIC", f"bad magic {hdr.magic:#x}")
    require(hdr.version == SLGB_VERSION, "SLGB_VER", f"bad version {hdr.version}")
    require(hdr.bos_token_id == 2, "SLGB_BOS_ID", f"expected bos_token_id=2, got {hdr.bos_token_id}")
    require(hdr.group_size in (32, 128), "SLGB_GRP", f"bad group_size={hdr.group_size}")
    require(hdr.num_kv_shared_layers == 20 and hdr.num_dead_kv_omitted == 60, "SLGB_KV", "invalid shared KV topology")
    require(hdr.total_wire_bytes > 0 and math.isfinite(hdr.rms_norm_eps) and hdr.rms_norm_eps > 0.0, "SLGB_PARAM", "invalid wire/eps")
    if tokens is not None:
        require(len(tokens) > 0 and int(tokens[0]) == 2, "SLGB_BOS_SEQ", "sequence must start with bos=2")
        for tok in tokens:
            require(0 <= int(tok) < hdr.vocab_size, "SLGB_TOK_OOB", f"token {tok} >= {hdr.vocab_size}")
    return hdr


@dataclass(frozen=True)
class SlqwHeader:
    """64-byte `.slqw` wire header descriptor for dense/gauge-equilibrated Qwen-family models."""

    magic: int = SLQW_MAGIC
    version: int = SLQW_VERSION
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    hidden_size: int = 1536
    num_layers: int = 28
    num_q_heads: int = 12
    num_kv_heads: int = 2
    head_dim: int = 128
    intermediate_size: int = 8960
    group_size: int = 128
    tie_word_embeddings: int = 1
    vocab_size: int = 151936
    num_a16_linears: int = 182
    num_mxfp8_linears: int = 14
    rms_norm_eps: float = 1.0e-6
    total_wire_bytes: int = 1_359_683_413
    flags: int = 0
    header_crc32: int = 0


def encode_slqw_header(hdr: SlqwHeader) -> bytes:
    """Serialize a 64-byte `.slqw` header with trailing IEEE CRC32 over bytes [0..60)."""
    require(math.isfinite(hdr.rms_norm_eps) and hdr.rms_norm_eps > 0.0, "SLQW_EPS", "non-positive/non-finite eps")
    prefix = struct.pack(
        "<4I8H3IfQI",
        hdr.magic,
        hdr.version,
        hdr.bos_token_id,
        hdr.eos_token_id,
        hdr.hidden_size,
        hdr.num_layers,
        hdr.num_q_heads,
        hdr.num_kv_heads,
        hdr.head_dim,
        hdr.intermediate_size,
        hdr.group_size,
        hdr.tie_word_embeddings,
        hdr.vocab_size,
        hdr.num_a16_linears,
        hdr.num_mxfp8_linears,
        hdr.rms_norm_eps,
        hdr.total_wire_bytes,
        hdr.flags,
    )
    crc = crc32_ieee(prefix)
    out = prefix + struct.pack("<I", crc)
    require(len(out) == 64, "SLQW_LEN", f"expected 64 bytes, got {len(out)}")
    return out


def decode_slqw_header(raw: bytes) -> SlqwHeader:
    """Deserialize a 64-byte `.slqw` header."""
    require(len(raw) == 64, "SLQW_LEN", f"expected 64 bytes, got {len(raw)}")
    return SlqwHeader(*struct.unpack(_SLQW_HEADER_FMT, raw))


def validate_slqw_header(
    raw_or_hdr: bytes | SlqwHeader,
    tokens: Sequence[int] | None = None,
    *,
    verify_crc: bool = True,
) -> SlqwHeader:
    """Fail-closed validation of `.slqw` header invariants, CRC32, GQA geometry, and vocab bounds."""
    if isinstance(raw_or_hdr, (bytes, bytearray, memoryview)):
        raw = bytes(raw_or_hdr)
        hdr = decode_slqw_header(raw)
        if verify_crc:
            require(hdr.header_crc32 == crc32_ieee(raw[:60]), "SLQW_CRC32", "CRC32 mismatch on .slqw header")
    else:
        hdr = raw_or_hdr
    require(hdr.magic == SLQW_MAGIC, "SLQW_MAGIC", f"bad magic {hdr.magic:#x}")
    require(hdr.version == SLQW_VERSION, "SLQW_VER", f"bad version {hdr.version}")
    require(hdr.group_size in (32, 128), "SLQW_GRP", f"bad group_size={hdr.group_size}")
    require(hdr.num_kv_heads > 0 and (hdr.num_q_heads % hdr.num_kv_heads) == 0, "SLQW_GQA", "invalid GQA ratio")
    require(hdr.num_a16_linears + hdr.num_mxfp8_linears == hdr.num_layers * 7, "SLQW_LIN", "layer linear tally mismatch")
    require(hdr.total_wire_bytes > 0 and hdr.vocab_size > 0, "SLQW_WIRE", "invalid wire/vocab size")
    if tokens is not None:
        require(len(tokens) > 0, "SLQW_TOK_EMPTY", "token sequence empty")
        for tok in tokens:
            require(0 <= int(tok) < hdr.vocab_size, "SLQW_TOK_OOB", f"token {tok} >= {hdr.vocab_size}")
    return hdr
