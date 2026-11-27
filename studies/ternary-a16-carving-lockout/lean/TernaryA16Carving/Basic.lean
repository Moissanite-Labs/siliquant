/-
  Track S163 EXP-Q42a x Model Carving (arXiv 2611.27962) g = 0 Lockout Duality.
  Pure Init, exact Int / Nat / Bool arithmetic, zero floats.

  Pillar 1: [16, 11, 4] A_16 codeword complement c -> c XOR 0xFFFF.
  Pillar 2: Block-H4 Parseval, scale sign-flip s -> -s, multi-round signs, ternary switch.
  Pillar 3: 16 B Weft spinor checkpoint pack/unpack bijection and rollback.
  Pillar 4: Chantless up_row_zero collapse and Model Carving g = 0 absorbing lockout.
-/

namespace TernaryA16Carving

/-! ## Pillar 1: concrete 16-bit masks -/

/-- Bit `i` of the natural number `n`. -/
def bitAt (n i : Nat) : Bool := (n / 2 ^ i) % 2 == 1

/-- Parity of `n AND mask` over the 16 bit positions. -/
def parMask (n mask : Nat) : Bool :=
  (List.range 16).foldl (fun acc i => acc ^^ (bitAt mask i && bitAt n i)) false

/-- Hamming weight of a 16-bit mask. -/
def weight16 (mask : Nat) : Nat :=
  (List.range 16).foldl (fun acc i => acc + (if bitAt mask i then 1 else 0)) 0

/-- 5-bit syndrome over (0xFFFF, 0xAAAA, 0xCCCC, 0xF0F0, 0xFF00), bit order as `a16_syndrome_u16`. -/
def syndromeNat (n : Nat) : Nat :=
  (if parMask n 0xFFFF then 1 else 0) + (if parMask n 0xAAAA then 2 else 0) +
  (if parMask n 0xCCCC then 4 else 0) + (if parMask n 0xF0F0 then 8 else 0) +
  (if parMask n 0xFF00 then 16 else 0)

theorem a16_masks_and_anchor_codewords :
    weight16 0xFFFF = 16 ∧ weight16 0xAAAA = 8 ∧ weight16 0xCCCC = 8 ∧
    weight16 0xF0F0 = 8 ∧ weight16 0xFF00 = 8 ∧
    syndromeNat 0x0000 = 0 ∧ syndromeNat 0xFFFF = 0 ∧
    syndromeNat 0x0FF0 = 0 ∧ syndromeNat (0x0FF0 ^^^ 0xFFFF) = 0 ∧
    (0x0FF0 ^^^ 0xFFFF) ^^^ 0xFFFF = (0x0FF0 : Nat) ∧
    syndromeNat 0x0001 = 1 ∧ syndromeNat 0x0002 = 3 := by
  decide

/-! ## Pillar 1: universal structural form over 16 Boolean coordinates -/

/-- A 16-bit word as 16 Boolean coordinates (bit `i` is `bi`). -/
structure W16 where
  b0 : Bool
  b1 : Bool
  b2 : Bool
  b3 : Bool
  b4 : Bool
  b5 : Bool
  b6 : Bool
  b7 : Bool
  b8 : Bool
  b9 : Bool
  b10 : Bool
  b11 : Bool
  b12 : Bool
  b13 : Bool
  b14 : Bool
  b15 : Bool

/-- Bitwise complement `c XOR 0xFFFF`. -/
def complement (w : W16) : W16 :=
  ⟨!w.b0, !w.b1, !w.b2, !w.b3, !w.b4, !w.b5, !w.b6, !w.b7,
   !w.b8, !w.b9, !w.b10, !w.b11, !w.b12, !w.b13, !w.b14, !w.b15⟩

/-- The all-zero codeword `0x0000`. -/
def zeroW : W16 :=
  ⟨false, false, false, false, false, false, false, false,
   false, false, false, false, false, false, false, false⟩

/-- Parity of eight bits. -/
def par8 (a b c d e f g h : Bool) : Bool := a ^^ b ^^ c ^^ d ^^ e ^^ f ^^ g ^^ h

/-- Five syndrome bits. -/
structure Syn where
  s0 : Bool
  s1 : Bool
  s2 : Bool
  s3 : Bool
  s4 : Bool

/-- The zero syndrome. -/
def synZero : Syn := ⟨false, false, false, false, false⟩

/-- Syndrome rows: 0xFFFF (all), 0xAAAA (odd), 0xCCCC, 0xF0F0, 0xFF00. -/
def syn (w : W16) : Syn :=
  ⟨par8 w.b0 w.b1 w.b2 w.b3 w.b4 w.b5 w.b6 w.b7 ^^
     par8 w.b8 w.b9 w.b10 w.b11 w.b12 w.b13 w.b14 w.b15,
   par8 w.b1 w.b3 w.b5 w.b7 w.b9 w.b11 w.b13 w.b15,
   par8 w.b2 w.b3 w.b6 w.b7 w.b10 w.b11 w.b14 w.b15,
   par8 w.b4 w.b5 w.b6 w.b7 w.b12 w.b13 w.b14 w.b15,
   par8 w.b8 w.b9 w.b10 w.b11 w.b12 w.b13 w.b14 w.b15⟩

theorem par8_complement (a b c d e f g h : Bool) :
    par8 (!a) (!b) (!c) (!d) (!e) (!f) (!g) (!h) = par8 a b c d e f g h := by
  revert a b c d e f g h
  decide

theorem syn_complement (w : W16) : syn (complement w) = syn w := by
  simp only [syn, complement, par8_complement]

theorem syn_complement_codeword (w : W16) (h : syn w = synZero) :
    syn (complement w) = synZero := by
  rw [syn_complement]
  exact h

theorem complement_involution (w : W16) : complement (complement w) = w := by
  cases w
  simp only [complement, Bool.not_not]

/-- Bipolar embedding `phi(b) = 1 - 2 b`. -/
def phi (b : Bool) : Int := if b then -1 else 1

/-- Bipolar vector of a word. -/
def bipolar (w : W16) : List Int :=
  [phi w.b0, phi w.b1, phi w.b2, phi w.b3, phi w.b4, phi w.b5, phi w.b6, phi w.b7,
   phi w.b8, phi w.b9, phi w.b10, phi w.b11, phi w.b12, phi w.b13, phi w.b14, phi w.b15]

/-- Exact squared L2 norm. -/
def normSq : List Int → Int
  | [] => 0
  | x :: xs => x * x + normSq xs

theorem bipolar_complement_neg (w : W16) :
    bipolar (complement w) = (bipolar w).map (fun x => -x) := by
  have hphi : ∀ b : Bool, phi (!b) = -phi b := by
    intro b
    cases b <;> rfl
  simp only [bipolar, complement, hphi, List.map]

theorem bipolar_normSq_sixteen (w : W16) :
    normSq (bipolar w) = 16 ∧ normSq (bipolar (complement w)) = 16 := by
  have hsq : ∀ b : Bool, phi b * phi b = 1 := by
    intro b
    cases b <;> rfl
  constructor
  · simp only [bipolar, normSq, hsq]
    rfl
  · simp only [bipolar, normSq, hsq]
    rfl

theorem normSq_neg (l : List Int) : normSq (l.map (fun x => -x)) = normSq l := by
  induction l with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.map, normSq, ih, Int.neg_mul_neg]

/-! ## Pillar 2: Block-H4 Parseval and scale sign-flip -/

/-- Sylvester H_4 butterfly on one nibble group. -/
def h4 (c0 c1 c2 c3 : Int) : List Int :=
  [c0 + c1 + c2 + c3, c0 - c1 + c2 - c3, c0 + c1 - c2 - c3, c0 - c1 - c2 + c3]

theorem h4_parseval (c0 c1 c2 c3 : Int) :
    normSq (h4 c0 c1 c2 c3) = 4 * normSq [c0, c1, c2, c3] := by
  simp only [h4, normSq, Int.add_mul, Int.mul_add, Int.mul_sub, Int.mul_comm]
  omega

/-- Cleared weights `2 * w_j = s * (H_4 c)_j`. -/
def wNum (s c0 c1 c2 c3 : Int) : List Int := (h4 c0 c1 c2 c3).map (fun h => s * h)

theorem normSq_scale (s : Int) (l : List Int) :
    normSq (l.map (fun x => s * x)) = s * s * normSq l := by
  induction l with
  | nil => simp only [List.map, normSq, Int.mul_zero]
  | cons x xs ih =>
    simp only [List.map, normSq, ih, Int.mul_add, Int.mul_assoc]
    rw [Int.mul_left_comm x s x]

theorem blockh4_weight_normSq (s c0 c1 c2 c3 : Int) :
    normSq (wNum s c0 c1 c2 c3) = (2 * s) * (2 * s) * normSq [c0, c1, c2, c3] := by
  simp only [wNum]
  rw [normSq_scale, h4_parseval]
  simp only [Int.mul_assoc, Int.mul_comm, Int.mul_left_comm]
  rw [show (2 : Int) * (2 * normSq [c0, c1, c2, c3]) = 4 * normSq [c0, c1, c2, c3] by omega]

theorem scale_flip_negates_preserves_norm (s c0 c1 c2 c3 : Int) :
    wNum (-s) c0 c1 c2 c3 = (wNum s c0 c1 c2 c3).map (fun x => -x) ∧
    normSq (wNum (-s) c0 c1 c2 c3) = normSq (wNum s c0 c1 c2 c3) := by
  have hflip : wNum (-s) c0 c1 c2 c3 = (wNum s c0 c1 c2 c3).map (fun x => -x) := by
    simp only [wNum, h4, List.map, Int.neg_mul]
  constructor
  · exact hflip
  · rw [hflip, normSq_neg]

/-! ## Pillar 2: arbitrary multi-round sign composition -/

/-- Apply a round sequence of switches; `true` is an active flip `sigma = -1`. -/
def applySigns : List Bool → Int → Int
  | [], x => x
  | b :: bs, x => applySigns bs (if b then -x else x)

/-- Net flip parity: `true` iff the product of all `sigma_t` is `-1`. -/
def flipParity : List Bool → Bool
  | [] => false
  | b :: bs => b ^^ flipParity bs

theorem applySigns_parity (bs : List Bool) (x : Int) :
    applySigns bs x = if flipParity bs then -x else x := by
  induction bs generalizing x with
  | nil => rfl
  | cons b bs ih =>
    cases b <;> cases hp : flipParity bs <;>
      simp only [applySigns, flipParity, ih, hp, Bool.true_xor, Bool.false_xor, Bool.not_true,
        Bool.not_false, Bool.false_eq_true, ↓reduceIte, Int.neg_neg]

theorem applySigns_normSq (bs : List Bool) (x : Int) :
    applySigns bs x * applySigns bs x = x * x := by
  rw [applySigns_parity]
  cases flipParity bs
  · rfl
  · exact Int.neg_mul_neg x x

theorem flipParity_append (bs cs : List Bool) :
    flipParity (bs ++ cs) = (flipParity bs ^^ flipParity cs) := by
  induction bs with
  | nil => simp only [List.nil_append, flipParity, Bool.false_xor]
  | cons b bs ih => simp only [List.cons_append, flipParity, ih, Bool.xor_assoc]

theorem applySigns_self_append (bs : List Bool) (x : Int) :
    applySigns (bs ++ bs) x = x := by
  rw [applySigns_parity, flipParity_append, Bool.xor_self]
  rfl

theorem multi_round_scale_zero_drift (bs : List Bool) (s c0 c1 c2 c3 : Int) :
    normSq (wNum (applySigns bs s) c0 c1 c2 c3) = normSq (wNum s c0 c1 c2 c3) := by
  rw [blockh4_weight_normSq, blockh4_weight_normSq]
  have h := applySigns_normSq bs s
  simp only [Int.mul_assoc, Int.mul_left_comm (applySigns bs s), h]
  simp only [Int.mul_left_comm s]

/-! ## Pillar 2: ternary switches {-1, 0, +1} -/

/-- Effective sign codeword of a ternary state: `+1` keeps `m`, `-1` complements, `0` gates to `0x0000`. -/
def ternaryMask (t : Int) (w : W16) : W16 :=
  if t = 0 then zeroW else if t = 1 then w else complement w

theorem ternary_syndrome_all_states (t : Int) (w : W16) (h : syn w = synZero) :
    syn (ternaryMask t w) = synZero := by
  unfold ternaryMask
  split
  · rfl
  · split
    · exact h
    · rw [syn_complement]
      exact h

/-- A valid ternary state. -/
def validT (t : Int) : Prop := t = -1 ∨ t = 0 ∨ t = 1

/-- Check-before-assign switch: invalid requests are rejected and leave the state unchanged. -/
def switchReq (cur req : Int) : Int :=
  if req = -1 ∨ req = 0 ∨ req = 1 then req else cur

/-- Run an arbitrary request sequence. -/
def runReqs (cur : Int) : List Int → Int
  | [] => cur
  | r :: rs => runReqs (switchReq cur r) rs

/-- Effective scale of the frozen base under ternary state `t`. -/
def ternaryEff (t s : Int) : Int := t * s

theorem ternary_switch_closed (cur req : Int) (rs : List Int) (h : validT cur) :
    validT (runReqs cur rs) ∧ (¬ validT req → switchReq cur req = cur) := by
  constructor
  · induction rs generalizing cur with
    | nil => exact h
    | cons r rs ih =>
      apply ih
      unfold switchReq
      split
      · assumption
      · exact h
  · intro hr
    unfold switchReq
    split
    · exact absurd (by assumption) hr
    · rfl

theorem ternary_reversible_from_any (cur s : Int) (rs : List Int) :
    ternaryEff (switchReq (runReqs cur rs) 1) s = s ∧
    ternaryEff 0 s = 0 ∧ ternaryEff (-1) s = -s := by
  refine ⟨?_, ?_, ?_⟩
  · simp only [switchReq, ternaryEff]
    simp
  · simp only [ternaryEff, Int.zero_mul]
  · simp only [ternaryEff, Int.neg_mul, Int.one_mul]

/-! ## Pillar 3: 16 B Weft spinor checkpoint -/

/-- Signed int32 range. -/
def inI32 (z : Int) : Prop := -2147483648 ≤ z ∧ z < 2147483648

/-- Little-endian two's-complement int32 decode of four bytes. -/
def decI32 (b0 b1 b2 b3 : Int) : Int :=
  if b0 + 256 * b1 + 65536 * b2 + 16777216 * b3 < 2147483648 then
    b0 + 256 * b1 + 65536 * b2 + 16777216 * b3
  else
    b0 + 256 * b1 + 65536 * b2 + 16777216 * b3 - 4294967296

/-- Little-endian two's-complement int32 encode to four bytes. -/
def encI32 (z : Int) : List Int :=
  [z % 4294967296 % 256, z % 4294967296 / 256 % 256,
   z % 4294967296 / 65536 % 256, z % 4294967296 / 16777216 % 256]

theorem int32_roundtrip (z : Int) (h : inI32 z) :
    decI32 (z % 4294967296 % 256) (z % 4294967296 / 256 % 256)
      (z % 4294967296 / 65536 % 256) (z % 4294967296 / 16777216 % 256) = z := by
  unfold inI32 at h
  unfold decI32
  split <;> omega

/-- An unsigned byte `0 ≤ b < 256`. -/
def inByte (b : Int) : Prop := 0 ≤ b ∧ b < 256

/-- Byte-level inverse: decoding four in-range bytes and re-encoding returns the same bytes. -/
theorem int32_byte_roundtrip (b0 b1 b2 b3 : Int) (h0 : inByte b0) (h1 : inByte b1)
    (h2 : inByte b2) (h3 : inByte b3) :
    encI32 (decI32 b0 b1 b2 b3) = [b0, b1, b2, b3] := by
  unfold inByte at h0 h1 h2 h3
  unfold encI32 decI32
  split <;> simp only [List.cons.injEq, and_true] <;> refine ⟨?_, ?_, ?_, ?_⟩ <;> omega

/-- Cleared unit spinor `(qw, qx, qy, qz)` with `qw^2 + qx^2 + qy^2 + qz^2 = D^2`. -/
structure Spinor where
  qw : Int
  qx : Int
  qy : Int
  qz : Int

def spinorNorm (q : Spinor) : Int := q.qw * q.qw + q.qx * q.qx + q.qy * q.qy + q.qz * q.qz

/-- `struct.pack("<4i", qw, qx, qy, qz)`. -/
def pack16B (q : Spinor) : List Int := encI32 q.qw ++ encI32 q.qx ++ encI32 q.qy ++ encI32 q.qz

/-- `struct.unpack("<4i", b)`; any length other than 16 is rejected. -/
def unpack16B : List Int → Option Spinor
  | [a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d3] =>
    some ⟨decI32 a0 a1 a2 a3, decI32 b0 b1 b2 b3, decI32 c0 c1 c2 c3, decI32 d0 d1 d2 d3⟩
  | _ => none

theorem pack16B_roundtrip (q : Spinor) (hw : inI32 q.qw) (hx : inI32 q.qx)
    (hy : inI32 q.qy) (hz : inI32 q.qz) :
    unpack16B (pack16B q) = some q ∧ (pack16B q).length = 16 := by
  constructor
  · simp only [pack16B, encI32, List.cons_append, List.nil_append, unpack16B]
    rw [int32_roundtrip q.qw hw, int32_roundtrip q.qx hx, int32_roundtrip q.qy hy,
      int32_roundtrip q.qz hz]
  · simp only [pack16B, encI32, List.cons_append, List.nil_append, List.length]

/-- Two-sided bijection, byte side: any 16 in-range bytes unpack and re-pack to themselves. -/
theorem pack16B_unpack16B (a0 a1 a2 a3 b0 b1 b2 b3 c0 c1 c2 c3 d0 d1 d2 d3 : Int)
    (hb : ∀ x ∈ [a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d3], inByte x) :
    (unpack16B [a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d3]).map pack16B =
      some [a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d3] := by
  simp only [unpack16B, Option.map, pack16B]
  rw [int32_byte_roundtrip a0 a1 a2 a3 (hb a0 (by simp)) (hb a1 (by simp)) (hb a2 (by simp))
      (hb a3 (by simp)),
    int32_byte_roundtrip b0 b1 b2 b3 (hb b0 (by simp)) (hb b1 (by simp)) (hb b2 (by simp))
      (hb b3 (by simp)),
    int32_byte_roundtrip c0 c1 c2 c3 (hb c0 (by simp)) (hb c1 (by simp)) (hb c2 (by simp))
      (hb c3 (by simp)),
    int32_byte_roundtrip d0 d1 d2 d3 (hb d0 (by simp)) (hb d1 (by simp)) (hb d2 (by simp))
      (hb d3 (by simp))]
  rfl

/-- Apply the round flips to all four spinor coordinates (spin double-cover sign). -/
def runFlips (bs : List Bool) (q : Spinor) : Spinor :=
  ⟨applySigns bs q.qw, applySigns bs q.qx, applySigns bs q.qy, applySigns bs q.qz⟩

theorem spinor_flip_runs_preserve_norm (bs : List Bool) (q : Spinor) :
    spinorNorm (runFlips bs q) = spinorNorm q ∧ runFlips (bs ++ bs) q = q := by
  constructor
  · simp only [spinorNorm, runFlips, applySigns_normSq]
  · cases q
    simp only [runFlips, applySigns_self_append]

/-- Switch states plus spinor. -/
structure WeftState where
  switches : List Int
  spin : Spinor

/-- Check-before-assign rollback: unpack, verify `D^2`, then assign; otherwise keep `cur`. -/
def rollback (cur : WeftState) (ckpt : List Int) (ckptSwitches : List Int) (dsq : Int) : WeftState :=
  match unpack16B ckpt with
  | some q => if spinorNorm q = dsq then ⟨ckptSwitches, q⟩ else cur
  | none => cur

theorem weft_rollback_exact (cur st0 : WeftState) (dsq : Int)
    (hw : inI32 st0.spin.qw) (hx : inI32 st0.spin.qx) (hy : inI32 st0.spin.qy)
    (hz : inI32 st0.spin.qz) (hn : spinorNorm st0.spin = dsq) :
    rollback cur (pack16B st0.spin) st0.switches dsq = st0 ∧
    spinorNorm (rollback cur (pack16B st0.spin) st0.switches dsq).spin - dsq = 0 := by
  have hu := (pack16B_roundtrip st0.spin hw hx hy hz).1
  have hr : rollback cur (pack16B st0.spin) st0.switches dsq = st0 := by
    unfold rollback
    rw [hu]
    simp only [hn, ↓reduceIte]
  constructor
  · exact hr
  · rw [hr, hn, Int.sub_self]

theorem unpack16B_length (l : List Int) (hl : l.length ≠ 16) : unpack16B l = none := by
  unfold unpack16B
  split
  · simp only [List.length] at hl
    exact absurd rfl hl
  · rfl

theorem weft_rollback_rejects_tampered (cur : WeftState) (ckpt sw : List Int) (dsq : Int) :
    (ckpt.length ≠ 16 → rollback cur ckpt sw dsq = cur) ∧
    (∀ q : Spinor, unpack16B ckpt = some q → spinorNorm q ≠ dsq →
      rollback cur ckpt sw dsq = cur) := by
  constructor
  · intro hl
    unfold rollback
    rw [unpack16B_length ckpt hl]
  · intro q hq hne
    unfold rollback
    rw [hq]
    simp only [hne, ↓reduceIte]

/-! ## Pillar 4: Chantless 0/1 `up_row_zero` -/

/-- Destructive row zeroing `d = 0`. -/
def upRowZero (w : List Int) : List Int := w.map (fun _ => 0)

theorem up_row_zero_collapse (w : List Int) (h : 0 < normSq w) :
    normSq (upRowZero w) = 0 ∧ normSq (upRowZero w) < normSq w := by
  have hall : ∀ l : List Int, normSq (upRowZero l) = 0 := by
    intro l
    unfold upRowZero
    induction l with
    | nil => rfl
    | cons x xs ih =>
      simp only [List.map, normSq, Int.mul_zero, Int.zero_add]
      exact ih
  have hz : normSq (upRowZero w) = 0 := hall w
  constructor
  · exact hz
  · rw [hz]
    exact h

theorem up_row_zero_noninjective (w1 w2 : List Int) (hl : w1.length = w2.length) :
    upRowZero w1 = upRowZero w2 ∧ upRowZero [1, 2, 3, 4] = upRowZero [4, 3, 2, 1] ∧
    ([1, 2, 3, 4] : List Int) ≠ [4, 3, 2, 1] := by
  refine ⟨?_, ?_, ?_⟩
  · unfold upRowZero
    induction w1 generalizing w2 with
    | nil =>
      cases w2 with
      | nil => rfl
      | cons y ys => simp only [List.length] at hl; omega
    | cons x xs ih =>
      cases w2 with
      | nil => simp only [List.length] at hl; omega
      | cons y ys =>
        simp only [List.length] at hl
        simp only [List.map]
        rw [ih ys (by omega)]
  · rfl
  · decide

theorem chantless_retention_witness :
    (9921 : Int) - 8333 = 1588 ∧ (9396 : Int) - 8322 = 1074 ∧ (9558 : Int) - 8729 = 829 ∧
    1588 * 2500 = 397 * (10000 : Int) ∧ 1074 * 5000 = 537 * (10000 : Int) ∧
    (6 : Int) * 23 = 3 * 46 ∧ (8333 : Int) < 9921 ∧ (8322 : Int) < 9396 ∧ (8729 : Int) < 9558 := by
  decide

/-! ## Pillar 4: Model Carving `Gates.step` clip-at-zero lockout -/

/-- Cleared-integer `Gates.step`: subtract the gradient move `u` and pressure move `p`, clip at `0`. -/
def carvingStep (g u p : Int) : Int := max (g - u - p) 0

/-- Iterate the step; the gradient is a function of the gate (`grad 0 = 0` once zeroed/removed). -/
def carvingRun (grad : Int → Int) : Int → List Int → Int
  | g, [] => g
  | g, p :: ps => carvingRun grad (carvingStep g (grad g) p) ps

theorem carving_zero_absorbing (grad : Int → Int) (hg : grad 0 = 0) (ps : List Int)
    (hp : ∀ p ∈ ps, 0 ≤ p) : carvingRun grad 0 ps = 0 := by
  induction ps with
  | nil => rfl
  | cons p ps ih =>
    have hp0 : 0 ≤ p := hp p (List.mem_cons_self ..)
    have hstep : carvingStep 0 (grad 0) p = 0 := by
      rw [hg]
      unfold carvingStep
      omega
    simp only [carvingRun, hstep]
    exact ih (fun q hq => hp q (List.mem_cons_of_mem p hq))

/-- `hold_open`: a gate that would clip to `0` is held at `eps` (`1e-6` cleared to `eps = 1`). -/
def holdOpenStep (eps g u p : Int) : Int :=
  if carvingStep g u p ≤ 0 then eps else carvingStep g u p

theorem hold_open_positive_and_reopenable (eps g u p : Int) (he : 0 < eps) (hu : u < 0) :
    0 < holdOpenStep eps g u p ∧ eps < holdOpenStep eps eps u 0 := by
  constructor
  · unfold holdOpenStep
    split
    · exact he
    · omega
  · unfold holdOpenStep carvingStep
    split <;> omega

theorem carving_positive_gate_reopens (g u p : Int) (hg : 0 < g) (hp : 0 ≤ p) (hup : u + p < 0) :
    g < carvingStep g u p ∧ carvingStep 0 0 p = 0 := by
  unfold carvingStep
  constructor <;> omega

theorem carving_lockout_witness :
    (193 : Nat) - 0 = 193 ∧ (0 : Nat) * 193 = 0 ∧ (141 : Nat) ≤ 184 ∧ (184 : Nat) ≤ 671 ∧
    (33 : Nat) ≤ 35 ∧ (35 : Nat) ≤ 184 ∧ 671 + 193 = (864 : Nat) := by
  decide

theorem duality_lockout_vs_recovery (grad : Int → Int) (hg : grad 0 = 0) (ps : List Int)
    (hp : ∀ p ∈ ps, 0 ≤ p) (cur s : Int) (rs : List Int) :
    carvingRun grad 0 ps = 0 ∧ ternaryEff (switchReq (runReqs cur rs) 1) s = s := by
  exact ⟨carving_zero_absorbing grad hg ps hp, (ternary_reversible_from_any cur s rs).1⟩

end TernaryA16Carving
