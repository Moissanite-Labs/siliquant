/-
Track S167 (EXP-Q51a/b/c, PREREG-W22-ES-SCALE)
Pure-Init Lean 4.36.0 Formalization of Evolution Strategies Operators on
SiliQuant (A_16 Block-H4 + +0 B Folded Scales) and HoloSphere-V4 (Cayley-SU(2) / SO(3)).

Zero imports, zero Mathlib, zero sorry, zero user-declared axioms, zero Classical.choice
(Lean Init kernel propext/Quot.sound only).
-/

namespace EvolutionOperators

-- ============================================================================
-- Helper definitions (pure Init, totally computable, constructive)
-- ============================================================================

def intAbs (x : Int) : Int :=
  if x >= 0 then x else -x

def listSum : List Int -> Int
  | [] => 0
  | x :: xs => x + listSum xs

def zipNeg : List Int -> List Int
  | [] => []
  | x :: xs => (-x) :: zipNeg xs

def dotProd : List Int -> List Int -> Int
  | [], _ => 0
  | _, [] => 0
  | x :: xs, y :: ys => x * y + dotProd xs ys

def sumSq : List Int -> Int
  | [] => 0
  | x :: xs => x * x + sumSq xs

def poly4 (c0 c1 c2 c3 c4 x : Int) : Int :=
  c0 + c1 * x + c2 * (x * x) + c3 * (x * x * x) + c4 * (x * x * x * x)

def h4Row : Nat -> List Int
  | 0 => [1, 1, 1, 1]
  | 1 => [1, -1, 1, -1]
  | 2 => [1, 1, -1, -1]
  | _ => [1, -1, -1, 1]

def h4Apply (a b c d : Int) : List Int :=
  [a + b + c + d, a - b + c - d, a + b - c - d, a - b - c + d]

def randH4Row (s0 s1 s2 s3 : Int) : Nat -> List Int
  | 0 => [s0, s1, s2, s3]
  | 1 => [s0, -s1, s2, -s3]
  | 2 => [s0, s1, -s2, -s3]
  | _ => [s0, -s1, -s2, s3]

def vecCrossSum : List (Int × Int × List Int × List Int) -> Int
  | [] => 0
  | (a, b, u, v) :: rest => a * b * dotProd u v + vecCrossSum rest

def allVecPairsOrthogonal : List (Int × Int × List Int × List Int) -> Prop
  | [] => True
  | (_, _, u, v) :: rest => dotProd u v = 0 ∧ allVecPairsOrthogonal rest

/-- Cleared-denominator Choromanski (2018) Thm 3.2 i.i.d. Gaussian ES MSE:
    N * MSE_iid = (d + 1) * ||g||^2. -/
def choromanskiMseIidScaled (d g2 : Nat) : Nat :=
  (d + 1) * g2

/-- Cleared-denominator Choromanski (2018) Thm 3.2 Orthogonal Gaussian ES MSE (N <= d):
    N * MSE_orth = (d + 2 - N) * ||g||^2 = N * MSE_iid - (N - 1) * ||g||^2. -/
def choromanskiMseOrthScaled (d n g2 : Nat) : Nat :=
  (d + 2 - n) * g2

def binom : Nat -> Nat -> Nat
  | _, 0 => 1
  | 0, _ + 1 => 0
  | n + 1, k + 1 => binom n k + binom n (k + 1)

def binomRowSumAux (n : Nat) : Nat -> Nat
  | 0 => binom n 0
  | k + 1 => binomRowSumAux n k + binom n (k + 1)

def binomRowSum (n : Nat) : Nat :=
  binomRowSumAux n n

def powNat (b : Nat) : Nat -> Nat
  | 0 => 1
  | n + 1 => b * powNat b n

/-- Nearest-grid rounding onto exact integer multiples of `ulp` (half-up tie rule):
    every output is a genuine grid multiple `ulp * k'`, with no hardcoded stall branch. -/
def bf16GridRound (ulp x : Int) : Int :=
  ulp * ((2 * x + ulp) / (2 * ulp))

def iterateBf16GridStep (ulp delta : Int) : Nat -> Int -> Int
  | 0, w => w
  | n + 1, w => iterateBf16GridStep ulp delta n (bf16GridRound ulp (w + delta))

/-- Centered Al-Kashi FixedRadix (SOP-16) error-feedback step with even quantum `q = 2*h`:
    matches symmetric half-up grid rounding in `s167_es_live_bench.py`. -/
def alkashiCenteredStep (h : Int) (st : Int × Int) (g : Int) : Int × Int :=
  let q := 2 * h
  let u := g + st.2
  let dq := (u + h) / q
  let eNew := u - dq * q
  (st.1 + dq, eNew)

def alkashiCenteredRun (h : Int) (st : Int × Int) : List Int -> Int × Int
  | [] => st
  | g :: gs => alkashiCenteredRun h (alkashiCenteredStep h st g) gs

/-- General projected/clamped/reverted Al-Kashi EF single step (covers A_16 lattice
    projection, [-qmax, qmax] grid clamping, and OOB ball-bound revert `realized = 0`). -/
def alkashiProjectedStep (q : Int) (st : Int × Int) (g realized : Int) : Int × Int :=
  let u := g + st.2
  (st.1 + realized, u - realized * q)

def sumScaledDot : List (Int × Int × Int × Int) -> Int
  | [] => 0
  | (w, x, p, q) :: rest => (w * p) * (x * q) + sumScaledDot rest

def sumFactoredDot : List (Int × Int × Int × Int) -> Int
  | [] => 0
  | (w, x, p, q) :: rest => (p * q) * (w * x) + sumFactoredDot rest

def sumUnscaledDot : List (Int × Int × Int × Int) -> Int
  | [] => 0
  | (w, x, _, _) :: rest => w * x + sumUnscaledDot rest

/-- Cleared-denominator reciprocal scale fold condition: (p/d) * (q/d) = 1 iff p * q = d * d. -/
def allReciprocalScales (d : Int) : List (Int × Int × Int × Int) -> Prop
  | [] => True
  | (_, _, p, q) :: rest => p * q = d * d ∧ allReciprocalScales d rest

structure Q4 where
  w : Int
  x : Int
  y : Int
  z : Int

def qNormSq (q : Q4) : Int :=
  q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z

def qMul (a b : Q4) : Q4 :=
  ⟨ a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
    a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
    a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w ⟩

def cayleyQ (d a b c : Int) : Q4 :=
  ⟨d * d - (a * a + b * b + c * c), 2 * d * a, 2 * d * b, 2 * d * c⟩

structure Mat3 where
  m00 : Int
  m01 : Int
  m02 : Int
  m10 : Int
  m11 : Int
  m12 : Int
  m20 : Int
  m21 : Int
  m22 : Int

def cayleySO3 (d x y z : Int) : Mat3 :=
  ⟨ d * d + x * x - y * y - z * z, 2 * (x * y - d * z), 2 * (x * z + d * y),
    2 * (x * y + d * z), d * d - x * x + y * y - z * z, 2 * (y * z - d * x),
    2 * (x * z - d * y), 2 * (y * z + d * x), d * d - x * x - y * y + z * z ⟩

def mat3TransposeMul (m : Mat3) : Mat3 :=
  ⟨ m.m00 * m.m00 + m.m10 * m.m10 + m.m20 * m.m20,
    m.m00 * m.m01 + m.m10 * m.m11 + m.m20 * m.m21,
    m.m00 * m.m02 + m.m10 * m.m12 + m.m20 * m.m22,
    m.m01 * m.m00 + m.m11 * m.m10 + m.m21 * m.m20,
    m.m01 * m.m01 + m.m11 * m.m11 + m.m21 * m.m21,
    m.m01 * m.m02 + m.m11 * m.m12 + m.m21 * m.m22,
    m.m02 * m.m00 + m.m12 * m.m10 + m.m22 * m.m20,
    m.m02 * m.m01 + m.m12 * m.m11 + m.m22 * m.m21,
    m.m02 * m.m02 + m.m12 * m.m12 + m.m22 * m.m22 ⟩

def mat3Det (m : Mat3) : Int :=
  m.m00 * (m.m11 * m.m22 - m.m12 * m.m21)
  - m.m01 * (m.m10 * m.m22 - m.m12 * m.m20)
  + m.m02 * (m.m10 * m.m21 - m.m11 * m.m20)

-- ============================================================================
-- Private arithmetic lemmas (pure Init, zero Classical.choice)
-- ============================================================================

private theorem int_sq_nonneg (x : Int) : 0 <= x * x := by
  if h : 0 <= x then
    exact Int.mul_nonneg h h
  else
    have hneg : 0 <= -x := by omega
    have hmul : 0 <= (-x) * (-x) := Int.mul_nonneg hneg hneg
    rw [Int.neg_mul_neg] at hmul
    exact hmul

private theorem int_sq_pos_of_ne_zero (x : Int) (hx : x ≠ 0) : 0 < x * x := by
  if h : 0 < x then
    exact Int.mul_pos h h
  else
    have hneg : 0 < -x := by omega
    have hmul : 0 < (-x) * (-x) := Int.mul_pos hneg hneg
    rw [Int.neg_mul_neg] at hmul
    exact hmul

private theorem int_mul_pair_swap (a b c d : Int) :
    (a * b) * (c * d) = (a * c) * (b * d) := by
  rw [Int.mul_assoc a b (c * d), ← Int.mul_assoc b c d, Int.mul_comm b c,
      Int.mul_assoc c b d, ← Int.mul_assoc a c (b * d)]

private theorem int_sq_add_sub_cancel (x y : Int) :
    (x + y) * (x + y) + (x - y) * (x - y) = 2 * (x * x + y * y) := by
  simp only [Int.add_mul, Int.sub_mul, Int.mul_add, Int.mul_sub]
  rw [Int.mul_comm y x]
  omega

private theorem prod_sq_split (a b : Int) : (a * b) * (a * b) = (a * a) * (b * b) :=
  int_mul_pair_swap a b a b

private theorem int_ediv_unique_of_rem (b q r x : Int) (hb : 0 < b)
    (heq : b * q + r = x) (hr_lo : 0 <= r) (hr_hi : r < b) :
    x / b = q := by
  have hdiv : b * (x / b) + x % b = x := Int.mul_ediv_add_emod x b
  have hlo : 0 <= x % b := Int.emod_nonneg x (by omega)
  have hhi : x % b < b := Int.emod_lt_of_pos x hb
  if hlt : x / b < q then
    have hle : x / b + 1 <= q := by omega
    have hmul : b * (x / b + 1) <= b * q := Int.mul_le_mul_of_nonneg_left hle (by omega)
    rw [Int.mul_add, Int.mul_one] at hmul
    omega
  else if hgt : q < x / b then
    have hle : q + 1 <= x / b := by omega
    have hmul : b * (q + 1) <= b * (x / b) := Int.mul_le_mul_of_nonneg_left hle (by omega)
    rw [Int.mul_add, Int.mul_one] at hmul
    omega
  else
    omega

private theorem bf16GridRound_fixed_of_small (ulp k delta : Int)
    (hulp : 0 < ulp) (hsmall : 2 * intAbs delta < ulp) :
    bf16GridRound ulp (k * ulp + delta) = k * ulp := by
  simp only [bf16GridRound]
  have habs_lo : -ulp < 2 * delta := by
    unfold intAbs at hsmall
    split at hsmall <;> omega
  have habs_hi : 2 * delta < ulp := by
    unfold intAbs at hsmall
    split at hsmall <;> omega
  have hnum : (2 * ulp) * k + (2 * delta + ulp) = 2 * (k * ulp + delta) + ulp := by
    rw [Int.mul_add, ← Int.mul_assoc 2 k ulp, Int.mul_comm 2 k, Int.mul_assoc k 2 ulp,
        Int.mul_comm k (2 * ulp)]
    omega
  have hq : (2 * (k * ulp + delta) + ulp) / (2 * ulp) = k :=
    int_ediv_unique_of_rem (2 * ulp) k (2 * delta + ulp) (2 * (k * ulp + delta) + ulp)
      (by omega) hnum (by omega) (by omega)
  rw [hq, Int.mul_comm]

private theorem dotProd_zipNeg (g e : List Int) :
    dotProd g (zipNeg e) = - dotProd g e := by
  induction g generalizing e with
  | nil => simp [dotProd]
  | cons gh gt ih =>
    cases e with
    | nil => simp [zipNeg, dotProd]
    | cons eh et =>
      simp only [zipNeg, dotProd, ih et, Int.mul_neg]
      omega

private theorem dotProd_self_eq_sumSq (e : List Int) :
    dotProd e e = sumSq e := by
  induction e with
  | nil => rfl
  | cons x xs ih =>
    simp only [dotProd, sumSq, ih]

private theorem sumSq_nonneg (e : List Int) : 0 <= sumSq e := by
  induction e with
  | nil => exact Int.le_refl 0
  | cons x xs ih =>
    simp only [sumSq]
    exact Int.add_nonneg (int_sq_nonneg x) ih

private theorem sumSq_pos_of_mem_ne_zero (e : List Int) (x : Int) (hmem : x ∈ e) (hx : x ≠ 0) :
    0 < sumSq e := by
  induction e with
  | nil => cases hmem
  | cons hd tl ih =>
    simp only [sumSq]
    cases hmem with
    | head =>
      have h1 := int_sq_pos_of_ne_zero x hx
      have h2 := sumSq_nonneg tl
      omega
    | tail _ htl =>
      have h1 := int_sq_nonneg hd
      have h2 := ih htl
      omega

private theorem nat_pow_lt_pow_of_lt {a b : Nat} (h : a < b) :
    ∀ r : Nat, 1 <= r -> powNat a r < powNat b r := by
  intro r hr
  induction r with
  | zero => omega
  | succ k ih =>
    cases k with
    | zero =>
      simp only [powNat, Nat.mul_one]
      exact h
    | succ m =>
      have hprev : powNat a (m + 1) < powNat b (m + 1) := ih (by omega)
      simp only [powNat]
      have h1 : a * powNat a (m + 1) <= b * powNat a (m + 1) := Nat.mul_le_mul_right _ (Nat.le_of_lt h)
      have hbpos : 0 < b := by omega
      have h2 : b * powNat a (m + 1) < b * powNat b (m + 1) := Nat.mul_lt_mul_of_pos_left hprev hbpos
      exact Nat.lt_of_le_of_lt h1 h2

private theorem nat_pow_mul (a b : Nat) :
    ∀ r : Nat, powNat (a * b) r = powNat a r * powNat b r := by
  intro r
  induction r with
  | zero => rfl
  | succ k ih =>
    simp only [powNat, ih]
    rw [Nat.mul_assoc a b (powNat a k * powNat b k),
        ← Nat.mul_assoc b (powNat a k) (powNat b k),
        Nat.mul_comm b (powNat a k),
        Nat.mul_assoc (powNat a k) b (powNat b k),
        ← Nat.mul_assoc a (powNat a k) (b * powNat b k)]

private theorem binom_gt (n k : Nat) (h : n < k) : binom n k = 0 := by
  induction n generalizing k with
  | zero =>
    cases k with
    | zero => omega
    | succ m => rfl
  | succ m ih =>
    cases k with
    | zero => omega
    | succ j =>
      simp only [binom]
      have h1 : binom m j = 0 := ih j (by omega)
      have h2 : binom m (j + 1) = 0 := ih (j + 1) (by omega)
      omega

private theorem binom_self (n : Nat) : binom n n = 1 := by
  induction n with
  | zero => rfl
  | succ m ih =>
    simp only [binom, ih, binom_gt m (m + 1) (by omega)]

private theorem binomRowSumAux_step (n k : Nat) :
    binomRowSumAux (n + 1) (k + 1)
    = binomRowSumAux n k + binomRowSumAux n (k + 1) := by
  induction k with
  | zero =>
    simp only [binomRowSumAux, binom]
  | succ j ih =>
    have hL : binomRowSumAux (n + 1) (j + 2)
        = binomRowSumAux (n + 1) (j + 1) + binom (n + 1) (j + 2) := rfl
    have hB : binom (n + 1) (j + 2) = binom n (j + 1) + binom n (j + 2) := rfl
    have hR1 : binomRowSumAux n (j + 1) = binomRowSumAux n j + binom n (j + 1) := rfl
    have hR2 : binomRowSumAux n (j + 2) = binomRowSumAux n (j + 1) + binom n (j + 2) := rfl
    rw [hL, ih, hB, hR1, hR2]
    omega

private theorem binom_absorption (n k : Nat) :
    (k + 1) * binom (n + 1) (k + 1) = (n + 1) * binom n k := by
  induction n generalizing k with
  | zero =>
    cases k with
    | zero => rfl
    | succ j =>
      have h0 : binom 1 (j + 2) = 0 := binom_gt 1 (j + 2) (by omega)
      have h1 : binom 0 (j + 1) = 0 := rfl
      rw [h0, h1]
      omega
  | succ m ih =>
    cases k with
    | zero =>
      have hprev := ih 0
      simp only [binom] at hprev ⊢
      omega
    | succ j =>
      have h1 := ih j
      have h2 := ih (j + 1)
      have hpas_top : binom (m + 2) (j + 2) = binom (m + 1) (j + 1) + binom (m + 1) (j + 2) := rfl
      have hpas_bot : binom (m + 1) (j + 1) = binom m j + binom m (j + 1) := rfl
      rw [hpas_top, Nat.mul_add]
      have hsplit : (j + 2) * binom (m + 1) (j + 1)
          = (j + 1) * binom (m + 1) (j + 1) + binom (m + 1) (j + 1) := by
        have : j + 2 = (j + 1) + 1 := rfl
        rw [this, Nat.add_mul, Nat.one_mul]
      rw [hsplit, h1, h2]
      have hR : (m + 2) * binom (m + 1) (j + 1)
          = (m + 1) * (binom m j + binom m (j + 1)) + binom (m + 1) (j + 1) := by
        have : m + 2 = (m + 1) + 1 := rfl
        rw [this, Nat.add_mul, Nat.one_mul, ← hpas_bot]
      rw [hR, Nat.mul_add]
      omega

private theorem binom_pos_le (n k : Nat) (h : k <= n) : 0 < binom n k := by
  induction n generalizing k with
  | zero =>
    have hk : k = 0 := by omega
    rw [hk]
    decide
  | succ m ih =>
    cases k with
    | zero =>
      simp only [binom]
      decide
    | succ j =>
      simp only [binom]
      have h1 := ih j (by omega)
      omega

private theorem alkashiCenteredStep_invariants (h : Int) (hh : 0 < h) (w e g : Int) :
    let st' := alkashiCenteredStep h (w, e) g
    (2 * h) * st'.1 + st'.2 = (2 * h) * w + e + g
    ∧ -h <= st'.2
    ∧ st'.2 < h := by
  simp only [alkashiCenteredStep]
  have hq : 0 < 2 * h := by omega
  have hdiv : (2 * h) * ((g + e + h) / (2 * h)) + (g + e + h) % (2 * h) = g + e + h :=
    Int.mul_ediv_add_emod (g + e + h) (2 * h)
  have hnonneg : 0 <= (g + e + h) % (2 * h) := Int.emod_nonneg (g + e + h) (by omega)
  have hlt : (g + e + h) % (2 * h) < 2 * h := Int.emod_lt_of_pos (g + e + h) hq
  have hcomm : ((g + e + h) / (2 * h)) * (2 * h) = (2 * h) * ((g + e + h) / (2 * h)) :=
    Int.mul_comm _ _
  refine ⟨?_, ?_, ?_⟩
  · rw [Int.mul_add, hcomm]
    omega
  · rw [hcomm]
    omega
  · rw [hcomm]
    omega

-- ============================================================================
-- Family 1: Exact Rational Antithetic-ES Odd/Even-Term Cancellation (Thms 1-4)
-- ============================================================================

/-- Theorem 1: Universal exact rational Antithetic-ES even-term cancellation on
    degree-4 Taylor expansions: f(sigma*e) - f(-sigma*e) eliminates c0, c2, c4
    identically, isolating 2*sigma*e*(c1 + c3*(sigma*e)^2). -/
theorem antithetic_poly4_even_cancellation_universal
    (c0 c1 c2 c3 c4 sigma e : Int) :
    poly4 c0 c1 c2 c3 c4 (sigma * e) - poly4 c0 c1 c2 c3 c4 (-(sigma * e))
    = 2 * (sigma * e) * (c1 + c3 * ((sigma * e) * (sigma * e))) := by
  generalize sigma * e = t
  simp only [poly4, Int.mul_neg, Int.neg_mul, Int.neg_neg]
  rw [Int.mul_add]
  have h1 : 2 * t * c1 = 2 * (c1 * t) := by
    rw [Int.mul_assoc, Int.mul_comm t c1]
  have h3 : 2 * t * (c3 * (t * t)) = 2 * (c3 * (t * t * t)) := by
    rw [Int.mul_assoc 2 t (c3 * (t * t)), ← Int.mul_assoc t c3 (t * t),
        Int.mul_comm t c3, Int.mul_assoc c3 t (t * t), ← Int.mul_assoc t t t]
  rw [h1, h3]
  omega

/-- Theorem 2: Universal exact rational Antithetic-ES odd-term cancellation:
    f(sigma*e) + f(-sigma*e) eliminates all odd perturbation terms c1, c3
    identically, leaving 2*(c0 + c2*(sigma*e)^2 + c4*(sigma*e)^4). -/
theorem antithetic_poly4_odd_cancellation_universal
    (c0 c1 c2 c3 c4 sigma e : Int) :
    poly4 c0 c1 c2 c3 c4 (sigma * e) + poly4 c0 c1 c2 c3 c4 (-(sigma * e))
    = 2 * (c0 + c2 * ((sigma * e) * (sigma * e))
           + c4 * ((sigma * e) * (sigma * e) * (sigma * e) * (sigma * e))) := by
  simp only [poly4, Int.mul_neg, Int.neg_mul, Int.neg_neg]
  omega

/-- Theorem 3: Universal arbitrary-dimension vector antithetic cancellation:
    for ANY g, e : List Int, baseline c0, even quadratic qEven, and scale sigma,
    the antithetic difference eliminates c0 and qEven while the antithetic sum
    eliminates the linear odd term sigma * <g, e>. -/
theorem antithetic_vec_linear_quadratic_cancellation_universal
    (g e : List Int) (c0 qEven sigma : Int) :
    (c0 + sigma * dotProd g e + qEven) - (c0 + sigma * dotProd g (zipNeg e) + qEven)
      = 2 * sigma * dotProd g e
    ∧ (c0 + sigma * dotProd g e + qEven) + (c0 + sigma * dotProd g (zipNeg e) + qEven)
      = 2 * (c0 + qEven) := by
  rw [dotProd_zipNeg, Int.mul_neg, Int.mul_assoc 2 sigma (dotProd g e)]
  exact ⟨by omega, by omega⟩

/-- Theorem 4: Universal antithetic squared-error dominance over one-sided ES:
    any nonzero baseline/even bias b != 0 adds strictly positive squared error
    2 * b^2 > 0 to one-sided ES, whereas antithetic differencing has zero error. -/
theorem antithetic_variance_strictly_beats_onesided_universal
    (a b : Int) (hb : b ≠ 0) :
    ((b + a) - a) * ((b + a) - a) + ((b - a) - (-a)) * ((b - a) - (-a)) = 2 * (b * b)
    ∧ ((b + a) - (b - a) - 2 * a) * ((b + a) - (b - a) - 2 * a) = 0
    ∧ 0 < 2 * (b * b) := by
  have h1 : (b + a) - a = b := by omega
  have h2 : (b - a) - (-a) = b := by omega
  have h3 : (b + a) - (b - a) - 2 * a = 0 := by omega
  have hpos : 0 < b * b := int_sq_pos_of_ne_zero b hb
  rw [h1, h2, h3]
  exact ⟨by omega, rfl, by omega⟩

-- ============================================================================
-- Family 2: Exact Orthogonal-Hadamard (H_4 / Block-H4) Zero-Cross-Term ES (Thms 5-9)
-- ============================================================================

/-- Theorem 5: Universal H_4 pairwise zero off-diagonal inner products (attached to
    explicit `h4Row` and `h4Apply` definitions) and exact Parseval norm identity
    `sumSq (h4Apply a b c d) = 4 * sumSq [a, b, c, d]`. -/
theorem hadamard_h4_pairwise_orthogonality_and_parseval_universal (a b c d : Int) :
    dotProd (h4Row 0) (h4Row 1) = 0
    ∧ dotProd (h4Row 0) (h4Row 2) = 0
    ∧ dotProd (h4Row 0) (h4Row 3) = 0
    ∧ dotProd (h4Row 1) (h4Row 2) = 0
    ∧ dotProd (h4Row 1) (h4Row 3) = 0
    ∧ dotProd (h4Row 2) (h4Row 3) = 0
    ∧ sumSq (h4Apply a b c d) = 4 * sumSq [a, b, c, d] := by
  refine ⟨by decide, by decide, by decide, by decide, by decide, by decide, ?_⟩
  simp only [h4Apply, sumSq, Int.add_zero]
  have r1 : a + b + c + d = (a + c) + (b + d) := by omega
  have r2 : a - b + c - d = (a + c) - (b + d) := by omega
  have r3 : a + b - c - d = (a - c) + (b - d) := by omega
  have r4 : a - b - c + d = (a - c) - (b - d) := by omega
  rw [r1, r2, r3, r4]
  have e1 := int_sq_add_sub_cancel (a + c) (b + d)
  have e2 := int_sq_add_sub_cancel (a - c) (b - d)
  have e3 := int_sq_add_sub_cancel a c
  have e4 := int_sq_add_sub_cancel b d
  omega

/-- Theorem 6: Universal sign-randomized Block-H4 (`randH4Row s`, `s_i^2 = 1`)
    pairwise orthogonality and exact Parseval norm preservation. -/
theorem randomized_block_h4_orthogonality_and_parseval_universal
    (s0 s1 s2 s3 v0 v1 v2 v3 : Int)
    (hs0 : s0 * s0 = 1) (hs1 : s1 * s1 = 1) (hs2 : s2 * s2 = 1) (hs3 : s3 * s3 = 1) :
    dotProd (randH4Row s0 s1 s2 s3 0) (randH4Row s0 s1 s2 s3 1) = 0
    ∧ dotProd (randH4Row s0 s1 s2 s3 0) (randH4Row s0 s1 s2 s3 2) = 0
    ∧ dotProd (randH4Row s0 s1 s2 s3 0) (randH4Row s0 s1 s2 s3 3) = 0
    ∧ dotProd (randH4Row s0 s1 s2 s3 1) (randH4Row s0 s1 s2 s3 2) = 0
    ∧ dotProd (randH4Row s0 s1 s2 s3 1) (randH4Row s0 s1 s2 s3 3) = 0
    ∧ dotProd (randH4Row s0 s1 s2 s3 2) (randH4Row s0 s1 s2 s3 3) = 0
    ∧ sumSq (h4Apply (s0 * v0) (s1 * v1) (s2 * v2) (s3 * v3)) = 4 * sumSq [v0, v1, v2, v3] := by
  have hp := (hadamard_h4_pairwise_orthogonality_and_parseval_universal
      (s0 * v0) (s1 * v1) (s2 * v2) (s3 * v3)).2.2.2.2.2.2
  have hsq0 : (s0 * v0) * (s0 * v0) = v0 * v0 := by
    rw [prod_sq_split, hs0, Int.one_mul]
  have hsq1 : (s1 * v1) * (s1 * v1) = v1 * v1 := by
    rw [prod_sq_split, hs1, Int.one_mul]
  have hsq2 : (s2 * v2) * (s2 * v2) = v2 * v2 := by
    rw [prod_sq_split, hs2, Int.one_mul]
  have hsq3 : (s3 * v3) * (s3 * v3) = v3 * v3 := by
    rw [prod_sq_split, hs3, Int.one_mul]
  simp only [sumSq, Int.add_zero] at hp ⊢
  rw [hsq0, hsq1, hsq2, hsq3] at hp
  simp only [randH4Row, dotProd, Int.mul_neg, Int.neg_mul, Int.neg_neg, hs0, hs1, hs2, hs3, Int.add_zero]
  exact ⟨by omega, by omega, by omega, by omega, by omega, by omega, hp⟩

/-- Theorem 7: Universal exact gradient recovery (zero MSE) of 4-member
    randomized Orthogonal-BlockH4 ES on Z^4: sum_{k=0}^3 <g, u_k> u_k = 4 * g. -/
theorem orthogonal_h4_es_exact_recovery_zero_mse_universal
    (g0 g1 g2 g3 s0 s1 s2 s3 : Int)
    (hs0 : s0 * s0 = 1) (hs1 : s1 * s1 = 1) (hs2 : s2 * s2 = 1) (hs3 : s3 * s3 = 1) :
    let d0 := dotProd [g0, g1, g2, g3] (randH4Row s0 s1 s2 s3 0)
    let d1 := dotProd [g0, g1, g2, g3] (randH4Row s0 s1 s2 s3 1)
    let d2 := dotProd [g0, g1, g2, g3] (randH4Row s0 s1 s2 s3 2)
    let d3 := dotProd [g0, g1, g2, g3] (randH4Row s0 s1 s2 s3 3)
    (d0 * s0 + d1 * s0 + d2 * s0 + d3 * s0 = 4 * g0)
    ∧ (d0 * s1 - d1 * s1 + d2 * s1 - d3 * s1 = 4 * g1)
    ∧ (d0 * s2 + d1 * s2 - d2 * s2 - d3 * s2 = 4 * g2)
    ∧ (d0 * s3 - d1 * s3 - d2 * s3 + d3 * s3 = 4 * g3) := by
  dsimp only
  simp only [randH4Row, dotProd, Int.mul_neg, Int.add_zero]
  have h0 : (g0 * s0) * s0 = g0 := by rw [Int.mul_assoc, hs0, Int.mul_one]
  have h1 : (g1 * s1) * s1 = g1 := by rw [Int.mul_assoc, hs1, Int.mul_one]
  have h2 : (g2 * s2) * s2 = g2 := by rw [Int.mul_assoc, hs2, Int.mul_one]
  have h3 : (g3 * s3) * s3 = g3 := by rw [Int.mul_assoc, hs3, Int.mul_one]
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp only [Int.add_mul, Int.neg_mul, h0]; omega
  · simp only [Int.add_mul, Int.neg_mul, h1]; omega
  · simp only [Int.add_mul, Int.neg_mul, h2]; omega
  · simp only [Int.add_mul, Int.neg_mul, h3]; omega

/-- Theorem 8: Universal pairwise-orthogonal zero cross-term identity and
    exact Choromanski (2018) Thm 3.2 MSE variance-reduction identity & ratio:
    (a) Any list of pairwise-orthogonal perturbation vector pairs (`dotProd u v = 0`)
        has `vecCrossSum = 0`;
    (b) For `2 <= N <= d` and `||g||^2 > 0`, Gaussian orthogonal ES satisfies
        `N * MSE_orth + (N - 1) * ||g||^2 = N * MSE_iid`, strict reduction
        `MSE_orth < MSE_iid`, and exact cleared-denominator ratio
        `(d + 1) * (N * MSE_orth) = (d + 2 - N) * (N * MSE_iid)` (i.e. `(d + 2 - N)/(d + 1)`). -/
theorem orthogonal_es_zero_crossterm_mse_reduction_universal
    (pairs : List (Int × Int × List Int × List Int)) (horth : allVecPairsOrthogonal pairs)
    (d n g2 : Nat) (hn : 2 <= n) (hnd : n <= d) (hg2 : 0 < g2) :
    vecCrossSum pairs = 0
    ∧ choromanskiMseOrthScaled d n g2 + (n - 1) * g2 = choromanskiMseIidScaled d g2
    ∧ choromanskiMseOrthScaled d n g2 < choromanskiMseIidScaled d g2
    ∧ (d + 1) * choromanskiMseOrthScaled d n g2 = (d + 2 - n) * choromanskiMseIidScaled d g2 := by
  have hcross : vecCrossSum pairs = 0 := by
    induction pairs with
    | nil => rfl
    | cons hd tl ih =>
      rcases hd with ⟨a, b, u, v⟩
      rcases horth with ⟨hip, hrest⟩
      simp only [vecCrossSum, hip, Int.mul_zero, Int.zero_add, ih hrest]
  unfold choromanskiMseOrthScaled choromanskiMseIidScaled
  have hsum : (d + 2 - n) + (n - 1) = d + 1 := by omega
  have hgap : (d + 2 - n) * g2 + (n - 1) * g2 = (d + 1) * g2 := by
    rw [← Nat.add_mul, hsum]
  have hpos : 0 < (n - 1) * g2 := Nat.mul_pos (by omega) hg2
  have hratio : (d + 1) * ((d + 2 - n) * g2) = (d + 2 - n) * ((d + 1) * g2) := by
    rw [← Nat.mul_assoc (d + 1) (d + 2 - n) g2,
        Nat.mul_comm (d + 1) (d + 2 - n),
        Nat.mul_assoc (d + 2 - n) (d + 1) g2]
  exact ⟨hcross, hgap, by omega, hratio⟩

/-- Theorem 9: Universal cross-layer seed-reuse correlation obstruction
    (diagnosing VsonicV/es-at-scale WorkerExtension.py shared-seed defect):
    reusing identical noise `e` across layers yields `dotProd e e = sumSq e > 0`
    for any nonzero perturbation, whereas distinct sign-randomized Block-H4 rows
    have universal zero cross-layer dot product. -/
theorem shared_seed_cross_layer_correlation_obstruction_universal
    (e : List Int) (x : Int) (hmem : x ∈ e) (hx : x ≠ 0)
    (s0 s1 s2 s3 : Int)
    (hs0 : s0 * s0 = 1) (hs1 : s1 * s1 = 1) (hs2 : s2 * s2 = 1) (hs3 : s3 * s3 = 1) :
    dotProd e e = sumSq e
    ∧ 0 < dotProd e e
    ∧ dotProd (randH4Row s0 s1 s2 s3 0) (randH4Row s0 s1 s2 s3 1) = 0 := by
  have heq := dotProd_self_eq_sumSq e
  have hpos := sumSq_pos_of_mem_ne_zero e x hmem hx
  have horth := (randomized_block_h4_orthogonality_and_parseval_universal
      s0 s1 s2 s3 0 0 0 0 hs0 hs1 hs2 hs3).1
  exact ⟨heq, by rw [heq]; exact hpos, horth⟩

-- ============================================================================
-- Family 3: Exact Heat-Kernel / Discrete-Gaussian Goodhart Spike Attenuation (Thms 10-12)
-- ============================================================================

/-- Theorem 10: Universal cleared-denominator rank-r heat-kernel Goodhart spike
    attenuation theorem: for narrow spike width w < broad plateau width W
    and smoothing scale sigma > 0, (w^2 / (w^2 + sigma^2))^r < (W^2 / (W^2 + sigma^2))^r
    for every active dimension rank r >= 1. -/
theorem heat_kernel_goodhart_spike_attenuation_universal
    (w W sigma r : Nat) (hlt : w < W) (hsig : 0 < sigma) (hr : 1 <= r) :
    powNat (w * w) r * powNat (W * W + sigma * sigma) r
    < powNat (W * W) r * powNat (w * w + sigma * sigma) r := by
  have hw2 : w * w < W * W := Nat.mul_lt_mul_of_lt_of_lt hlt hlt
  have hsig2 : 0 < sigma * sigma := Nat.mul_pos hsig hsig
  have hbase : (w * w) * (W * W + sigma * sigma) < (W * W) * (w * w + sigma * sigma) := by
    rw [Nat.mul_add, Nat.mul_add, Nat.mul_comm (W * W) (w * w)]
    have hmul : (w * w) * (sigma * sigma) < (W * W) * (sigma * sigma) :=
      Nat.mul_lt_mul_of_pos_right hw2 hsig2
    omega
  have hpow := nat_pow_lt_pow_of_lt hbase r hr
  rw [nat_pow_mul (w * w) (W * W + sigma * sigma) r,
      nat_pow_mul (W * W) (w * w + sigma * sigma) r] at hpow
  exact hpow

/-- Theorem 11: Universal discrete Pascal/binomial heat-kernel plateau conservation:
    sum_{j=0}^k C(k, j) = 2^k, so flat high-reward plateaus suffer zero attenuation
    under k-step discrete Gaussian diffusion. -/
theorem discrete_pascal_heat_kernel_plateau_conservation_universal (k : Nat) :
    binomRowSum k = powNat 2 k := by
  induction k with
  | zero => rfl
  | succ m ih =>
    unfold binomRowSum at ih ⊢
    rw [binomRowSumAux_step m m]
    have htop : binomRowSumAux m (m + 1) = binomRowSumAux m m := by
      simp only [binomRowSumAux, binom_gt m (m + 1) (by omega), Nat.add_zero]
    rw [htop, ih]
    simp only [powNat]
    omega

/-- Theorem 12: Universal central binomial Goodhart spike recurrence and strict
    per-step contraction under discrete Gaussian heat-kernel smoothing:
    C(2(m+1), m+1) / 4^(m+1) = ((2m+1)/(2m+2)) * C(2m, m) / 4^m < C(2m, m) / 4^m. -/
theorem discrete_gaussian_center_spike_strict_contraction_universal (m : Nat) :
    (2 * m + 2) * binom (2 * (m + 1)) (m + 1) = 4 * (2 * m + 1) * binom (2 * m) m
    ∧ binom (2 * (m + 1)) (m + 1) < 4 * binom (2 * m) m := by
  have h1 : (m + 1) * binom (2 * m + 2) (m + 1) = (2 * m + 2) * binom (2 * m + 1) m :=
    binom_absorption (2 * m + 1) m
  have hsym_aux : ∀ n k : Nat, k <= n -> binom n k = binom n (n - k) := by
    intro n
    induction n with
    | zero =>
      intro k hk
      have : k = 0 := by omega
      rw [this]
    | succ a iha =>
      intro k hk
      cases k with
      | zero =>
        simp only [Nat.sub_zero, binom_self]
        rfl
      | succ j =>
        match Nat.lt_or_eq_of_le hk with
        | Or.inr heq =>
          rw [heq, Nat.sub_self, binom_self]
          rfl
        | Or.inl hlt =>
          have hle1 : j <= a := Nat.le_of_lt (Nat.lt_of_succ_lt_succ hlt)
          have hle2 : j + 1 <= a := Nat.le_of_lt_succ hlt
          have hsucc : a + 1 - (j + 1) = a - j := Nat.succ_sub_succ a j
          rw [hsucc]
          have hpos_sub : 0 < a - j := Nat.sub_pos_of_lt (Nat.lt_of_succ_lt_succ hlt)
          match hsub : a - j with
          | 0 => exact False.elim (Nat.lt_irrefl 0 (hsub ▸ hpos_sub))
          | r + 1 =>
            have hL : binom (a + 1) (j + 1) = binom a j + binom a (j + 1) := rfl
            have hR : binom (a + 1) (r + 1) = binom a r + binom a (r + 1) := rfl
            rw [hL, hR]
            have hr1 : binom a j = binom a (r + 1) := by
              have h := iha j hle1
              rw [hsub] at h
              exact h
            have hr2 : binom a (j + 1) = binom a r := by
              have h := iha (j + 1) hle2
              have he1 : a - (j + 1) = r := by
                rw [Nat.sub_succ, hsub]
                rfl
              rw [he1] at h
              exact h
            rw [hr1, hr2, Nat.add_comm]
  have h2 : (m + 1) * binom (2 * m + 1) m = (2 * m + 1) * binom (2 * m) m := by
    have habs := binom_absorption (2 * m) m
    have hs := hsym_aux (2 * m + 1) m (by omega)
    have heq : 2 * m + 1 - m = m + 1 := by
      have h2m : 2 * m + 1 = m + (m + 1) := by omega
      rw [h2m, Nat.add_sub_cancel_left]
    rw [heq] at hs
    rw [hs]
    exact habs
  have hrec : (2 * m + 2) * binom (2 * (m + 1)) (m + 1) = 4 * (2 * m + 1) * binom (2 * m) m := by
    have h2m : 2 * (m + 1) = 2 * m + 2 := by omega
    rw [h2m]
    have hmul2 : (2 * m + 2) * binom (2 * m + 2) (m + 1)
        = 2 * ((m + 1) * binom (2 * m + 2) (m + 1)) := by
      have : 2 * m + 2 = 2 * (m + 1) := by omega
      rw [this, Nat.mul_assoc]
    rw [hmul2, h1]
    have hmul4 : 2 * ((2 * m + 2) * binom (2 * m + 1) m)
        = 4 * ((m + 1) * binom (2 * m + 1) m) := by
      have : 2 * m + 2 = 2 * (m + 1) := by omega
      rw [this, ← Nat.mul_assoc, ← Nat.mul_assoc]
      have h4 : 2 * 2 = 4 := rfl
      rw [h4, Nat.mul_assoc]
    rw [hmul4, h2, ← Nat.mul_assoc]
  have hpos : 0 < binom (2 * m) m := binom_pos_le (2 * m) m (by omega)
  have hstrict : (2 * m + 2) * binom (2 * (m + 1)) (m + 1) < (2 * m + 2) * (4 * binom (2 * m) m) := by
    rw [hrec]
    have hdiff : 4 * (2 * m + 1) < (2 * m + 2) * 4 := by omega
    have hmul := Nat.mul_lt_mul_of_pos_right hdiff hpos
    rw [Nat.mul_assoc (2 * m + 2) 4 (binom (2 * m) m)] at hmul
    exact hmul
  have hlt : binom (2 * (m + 1)) (m + 1) < 4 * binom (2 * m) m :=
    match Nat.lt_or_ge (binom (2 * (m + 1)) (m + 1)) (4 * binom (2 * m) m) with
    | Or.inl h => h
    | Or.inr hge =>
      False.elim (Nat.not_lt_of_ge (Nat.mul_le_mul_left (2 * m + 2) hge) hstrict)
  exact ⟨hrec, hlt⟩

-- ============================================================================
-- Family 4: BF16 ULP Stall vs Al-Kashi FixedRadix EF & CS-08 Parity (Thms 13-17)
-- ============================================================================

/-- Theorem 13: Universal BF16 nearest-grid ULP half-threshold stall obstruction:
    using genuine nearest-grid rounding `bf16GridRound` (no hardcoded stall branch),
    whenever `0 < ulp` and `2 * |delta| < ulp`, iterating T consecutive updates from
    any grid point `k * ulp` produces identically zero weight movement for all T : Nat. -/
theorem bf16_ulp_half_threshold_stall_universal
    (ulp k delta : Int) (t : Nat) (hulp : 0 < ulp) (hstall : 2 * intAbs delta < ulp) :
    iterateBf16GridStep ulp delta t (k * ulp) = k * ulp := by
  induction t with
  | zero => rfl
  | succ m ih =>
    simp only [iterateBf16GridStep]
    rw [bf16GridRound_fixed_of_small ulp k delta hulp hstall]
    exact ih

/-- Theorem 14: Coupled canonical VsonicV/es-at-scale BF16 stall & sharp advance
    threshold witness at `|w| in [1, 2)` (common denominator `256000`):
    `ULP_BF16 = 1/128 = 2000/256000`, `alpha = 5e-4 = 128/256000`.
    For EVERY grid coordinate `k`, EVERY step count `t`, and EVERY bounded gradient
    factor `|gHat| <= 7`, `iterateBf16GridStep` stalls completely (`= k * 2000`),
    whereas at the sharp threshold `gHat = 8` (`128 * 8 = 1024 >= 1000 = half-ULP`),
    `bf16GridRound` advances to `(k + 1) * 2000`. -/
theorem bf16_canonical_es_ulp_stall_witness
    (k gHat : Int) (t : Nat) (hg : intAbs gHat <= 7) :
    iterateBf16GridStep 2000 (128 * gHat) t (k * 2000) = k * 2000
    ∧ bf16GridRound 2000 (k * 2000 + 128 * 8) = (k + 1) * 2000
    ∧ 2 * 128 < 2000 := by
  have hsmall : 2 * intAbs (128 * gHat) < 2000 := by
    unfold intAbs at hg ⊢
    split at hg <;> split <;> omega
  have hstall := bf16_ulp_half_threshold_stall_universal 2000 k (128 * gHat) t (by omega) hsmall
  have hadv : bf16GridRound 2000 (k * 2000 + 128 * 8) = (k + 1) * 2000 := by
    simp only [bf16GridRound]
    have hnum : (2 * 2000) * (k + 1) + 48 = 2 * (k * 2000 + 128 * 8) + 2000 := by
      rw [Int.mul_add, Int.mul_one, Int.mul_add, ← Int.mul_assoc 2 k 2000,
          Int.mul_comm 2 k, Int.mul_assoc k 2 2000, Int.mul_comm k (2 * 2000)]
      omega
    have hq : (2 * (k * 2000 + 128 * 8) + 2000) / (2 * 2000) = k + 1 :=
      int_ediv_unique_of_rem (2 * 2000) (k + 1) 48 (2 * (k * 2000 + 128 * 8) + 2000)
        (by omega) hnum (by omega) (by omega)
    rw [hq, Int.mul_comm]
  exact ⟨hstall, hadv, by decide⟩

/-- Theorem 15: Universal Al-Kashi FixedRadix (SOP-16) centered error-feedback
    conservation, symmetric residual bound, zero-stall advance, AND arbitrary
    projection/clamping/OOB-revert single-step conservation law:
    for ANY half-quantum `h > 0` (`Q = 2 * h`), ANY update stream `gs : List Int`,
    and ANY realized discrete jump `realized` (lattice projection, grid clamp, or
    OOB revert `realized = 0`), total fixed-radix mass `Q * w + e` is conserved
    identically, with `-h <= eFinal < h` and guaranteed advance `wFinal >= w0 + 1`
    whenever `listSum gs >= 2 * h`. -/
theorem alkashi_fixedradix_ef_exact_conservation_universal
    (h w0 e0 w e g realized : Int) (gs : List Int)
    (hh : 0 < h) (he0_lo : -h <= e0) (he0_hi : e0 < h) :
    let q := 2 * h
    let fin := alkashiCenteredRun h (w0, e0) gs
    let proj := alkashiProjectedStep q (w, e) g realized
    q * fin.1 + fin.2 = q * w0 + e0 + listSum gs
    ∧ -h <= fin.2
    ∧ fin.2 < h
    ∧ (q <= listSum gs -> w0 + 1 <= fin.1)
    ∧ q * proj.1 + proj.2 = q * w + e + g := by
  dsimp only
  have hmain : ∀ (st : Int × Int) (xs : List Int),
      -h <= st.2 -> st.2 < h ->
      let res := alkashiCenteredRun h st xs
      (2 * h) * res.1 + res.2 = (2 * h) * st.1 + st.2 + listSum xs
      ∧ -h <= res.2
      ∧ res.2 < h := by
    intro st xs
    induction xs generalizing st with
    | nil =>
      intro hlo hhi
      simp only [alkashiCenteredRun, listSum, Int.add_zero]
      exact ⟨trivial, hlo, hhi⟩
    | cons x rest ih =>
      intro _ _
      simp only [alkashiCenteredRun, listSum]
      have hstep := alkashiCenteredStep_invariants h hh st.1 st.2 x
      rcases hstep with ⟨hcons, hlo', hhi'⟩
      have hrec := ih (alkashiCenteredStep h st x) hlo' hhi'
      rcases hrec with ⟨hcons_r, hlo_r, hhi_r⟩
      refine ⟨?_, hlo_r, hhi_r⟩
      rw [hcons_r, hcons]
      omega
  have hres := hmain (w0, e0) gs he0_lo he0_hi
  dsimp only at hres
  rcases hres with ⟨hcons, hlo, hhi⟩
  have hproj : (2 * h) * (alkashiProjectedStep (2 * h) (w, e) g realized).1
      + (alkashiProjectedStep (2 * h) (w, e) g realized).2 = (2 * h) * w + e + g := by
    simp only [alkashiProjectedStep]
    rw [Int.mul_add, Int.mul_comm realized (2 * h)]
    omega
  refine ⟨hcons, hlo, hhi, ?_, hproj⟩
  intro hbig
  have hgt : (2 * h) * w0 < (2 * h) * (alkashiCenteredRun h (w0, e0) gs).1 := by omega
  if hle : (alkashiCenteredRun h (w0, e0) gs).1 <= w0 then
    have hmul : (2 * h) * (alkashiCenteredRun h (w0, e0) gs).1 <= (2 * h) * w0 :=
      Int.mul_le_mul_of_nonneg_left hle (by omega)
    omega
  else
    omega

/-- Theorem 16: Universal CS-08 compute-parity inequalities:
    (a) Forward-only ES uses 2PL FLOPs vs 6PL (Dr.GRPO, 3x) and 8PL (GRPO, 4x);
    (b) On a single GPU, N >= 2 unbatched weight perturbations incur 2*N*P > 3*P
        HBM weight traffic vs shared-weight batched GRPO. -/
theorem cs08_es_vs_grpo_flop_and_hbm_crossover_universal
    (p l n : Nat) (hp : 0 < p) (hl : 0 < l) (hn : 2 <= n) :
    2 * p * l < 6 * p * l
    ∧ 6 * p * l < 8 * p * l
    ∧ 3 * (2 * p * l) = 6 * p * l
    ∧ 4 * (2 * p * l) = 8 * p * l
    ∧ 3 * p < 2 * n * p := by
  have hpl : 0 < p * l := Nat.mul_pos hp hl
  have h1 : 2 * p * l = 2 * (p * l) := Nat.mul_assoc 2 p l
  have h2 : 6 * p * l = 6 * (p * l) := Nat.mul_assoc 6 p l
  have h3 : 8 * p * l = 8 * (p * l) := Nat.mul_assoc 8 p l
  have h2n : 3 < 2 * n := by omega
  have hhbm : 3 * p < (2 * n) * p := Nat.mul_lt_mul_of_pos_right h2n hp
  refine ⟨?_, ?_, ?_, ?_, hhbm⟩
  · rw [h1, h2]; exact Nat.mul_lt_mul_of_pos_right (by decide) hpl
  · rw [h2, h3]; exact Nat.mul_lt_mul_of_pos_right (by decide) hpl
  · rw [h1, h2, ← Nat.mul_assoc]
  · rw [h1, h3, ← Nat.mul_assoc]

/-- Theorem 17: Universal +0 B reciprocal folded-scale ES invariance across arbitrary
    channel lists:
    (a) General factoring `sumScaledDot items = sumFactoredDot items`;
    (b) Reciprocal cancellation: whenever every channel scale pair `(p, q)` satisfies
        cleared-denominator reciprocity `p * q = d * d` (i.e. `(p/d)*(q/d) = 1`),
        folding `p/d` into weights and `q/d` into adjacent RMSNorm activations preserves
        exact inner products `sumScaledDot items = (d * d) * sumUnscaledDot items`
        with zero extra runtime parameters. -/
theorem folded_scale_zero_byte_es_invariance_universal
    (d : Int) (items : List (Int × Int × Int × Int)) :
    sumScaledDot items = sumFactoredDot items
    ∧ (allReciprocalScales d items -> sumScaledDot items = (d * d) * sumUnscaledDot items) := by
  have hfac : sumScaledDot items = sumFactoredDot items := by
    induction items with
    | nil => rfl
    | cons hd tl ih =>
      rcases hd with ⟨w, x, p, q⟩
      simp only [sumScaledDot, sumFactoredDot, ih]
      rw [int_mul_pair_swap w p x q, Int.mul_comm (w * x) (p * q)]
  have hrec : allReciprocalScales d items -> sumScaledDot items = (d * d) * sumUnscaledDot items := by
    clear hfac
    induction items with
    | nil =>
      intro _
      simp only [sumScaledDot, sumUnscaledDot, Int.mul_zero]
    | cons hd tl ih =>
      intro hinv
      rcases hd with ⟨w, x, p, q⟩
      rcases hinv with ⟨hpq, htl⟩
      simp only [sumScaledDot, sumUnscaledDot]
      rw [ih htl, int_mul_pair_swap w p x q, Int.mul_comm (w * x) (p * q), hpq, Int.mul_add]
  exact ⟨hfac, hrec⟩

-- ============================================================================
-- Family 5: Exact Rational Cayley-SU(2) / SO(3) Tangent-Space ES Manifold (Thms 18-21)
-- ============================================================================

/-- Theorem 18: Universal rational Cayley-SU(2) tangent-space ES unit-sphere
    preservation: ||cayleyQ(d, a, b, c)||^2 = (d^2 + a^2 + b^2 + c^2)^2 identically,
    with strictly positive denominator whenever d != 0. -/
theorem cayley_su2_unit_sphere_universal (d a b c : Int) :
    qNormSq (cayleyQ d a b c) = (d * d + a * a + b * b + c * c) * (d * d + a * a + b * b + c * c)
    ∧ (d ≠ 0 -> 0 < d * d + a * a + b * b + c * c) := by
  refine ⟨?_, ?_⟩
  · simp only [qNormSq, cayleyQ]
    have h1 : (2 * d * a) * (2 * d * a) = 4 * (d * d) * (a * a) := by
      have : 2 * d * a = (2 * d) * a := rfl
      rw [this, prod_sq_split (2 * d) a, prod_sq_split 2 d]
      rfl
    have h2 : (2 * d * b) * (2 * d * b) = 4 * (d * d) * (b * b) := by
      have : 2 * d * b = (2 * d) * b := rfl
      rw [this, prod_sq_split (2 * d) b, prod_sq_split 2 d]
      rfl
    have h3 : (2 * d * c) * (2 * d * c) = 4 * (d * d) * (c * c) := by
      have : 2 * d * c = (2 * d) * c := rfl
      rw [this, prod_sq_split (2 * d) c, prod_sq_split 2 d]
      rfl
    rw [h1, h2, h3]
    generalize d * d = D
    generalize a * a = A
    generalize b * b = B
    generalize c * c = C
    simp only [Int.add_mul, Int.sub_mul, Int.mul_add, Int.mul_sub, Int.mul_assoc]
    rw [Int.mul_comm A D, Int.mul_comm B D, Int.mul_comm C D,
        Int.mul_comm B A, Int.mul_comm C A, Int.mul_comm C B]
    omega
  · intro hd
    have hd2 := int_sq_pos_of_ne_zero d hd
    have ha2 := int_sq_nonneg a
    have hb2 := int_sq_nonneg b
    have hc2 := int_sq_nonneg c
    omega

/-- Theorem 19: Universal Euler four-square quaternion composition closure:
    ||p * q||^2 = ||p||^2 * ||q||^2, certifying that arbitrary chains of
    Cayley-SU(2) ES tangent steps and holonomy prefix scans stay on S^3. -/
theorem cayley_su2_composition_norm_closure_universal (p q : Q4) :
    qNormSq (qMul p q) = qNormSq p * qNormSq q := by
  rcases p with ⟨pw, px, py, pz⟩
  rcases q with ⟨qw, qx, qy, qz⟩
  simp only [qNormSq, qMul]
  simp only [Int.add_mul, Int.sub_mul, Int.mul_add, Int.mul_sub]
  simp only [int_mul_pair_swap]
  have cp_xw : px * pw = pw * px := Int.mul_comm px pw
  have cp_yw : py * pw = pw * py := Int.mul_comm py pw
  have cp_zw : pz * pw = pw * pz := Int.mul_comm pz pw
  have cp_yx : py * px = px * py := Int.mul_comm py px
  have cp_zx : pz * px = px * pz := Int.mul_comm pz px
  have cp_zy : pz * py = py * pz := Int.mul_comm pz py
  have cq_xw : qx * qw = qw * qx := Int.mul_comm qx qw
  have cq_yw : qy * qw = qw * qy := Int.mul_comm qy qw
  have cq_zw : qz * qw = qw * qz := Int.mul_comm qz qw
  have cq_yx : qy * qx = qx * qy := Int.mul_comm qy qx
  have cq_zx : qz * qx = qx * qz := Int.mul_comm qz qx
  have cq_zy : qz * qy = qy * qz := Int.mul_comm qz qy
  rw [cp_xw, cp_yw, cp_zw, cp_yx, cp_zx, cp_zy,
      cq_xw, cq_yw, cq_zw, cq_yx, cq_zx, cq_zy]
  omega

/-- Theorem 20: Universal rational Cayley-SO(3) tangent-space ES manifold
    preservation: for ANY scaled skew-symmetric tangent coordinates `(x, y, z)`
    over denominator `d`, `M(d, x, y, z)^T * M(d, x, y, z) = D^2 * I_3` identically
    (all 9 entries), `det(M(d, x, y, z)) = D^3`, AND whenever `d != 0`, `0 < D^3`
    (so `(1/D)*M` lies strictly in `SO(3)` with `det = +1`). -/
theorem cayley_so3_tangent_orthogonality_and_det_universal (d x y z : Int) :
    let D := d * d + x * x + y * y + z * z
    let M := cayleySO3 d x y z
    mat3TransposeMul M = ⟨D * D, 0, 0, 0, D * D, 0, 0, 0, D * D⟩
    ∧ mat3Det M = D * D * D
    ∧ (d ≠ 0 -> 0 < D * D * D) := by
  dsimp only
  have hnorm_sym : ∀ u v a b : Int,
      (u * u + v * v - a * a - b * b) * (u * u + v * v - a * a - b * b)
      + (2 * (v * a + u * b)) * (2 * (v * a + u * b))
      + (2 * (v * b - u * a)) * (2 * (v * b - u * a))
      = (u * u + v * v + a * a + b * b) * (u * u + v * v + a * a + b * b) := by
    intro u v a b
    have h4 : ∀ p q : Int, (2 * p) * (2 * q) = 4 * (p * q) := by
      intro p q; rw [int_mul_pair_swap 2 p 2 q]; rfl
    rw [h4, h4]
    simp only [Int.add_mul, Int.sub_mul, Int.mul_add, Int.mul_sub]
    have s_vava : (v * a) * (v * a) = (v * v) * (a * a) := prod_sq_split v a
    have s_ubub : (u * b) * (u * b) = (u * u) * (b * b) := prod_sq_split u b
    have s_vbvb : (v * b) * (v * b) = (v * v) * (b * b) := prod_sq_split v b
    have s_uaua : (u * a) * (u * a) = (u * u) * (a * a) := prod_sq_split u a
    have s_vaub : (v * a) * (u * b) = (u * v) * (a * b) := by
      rw [int_mul_pair_swap v a u b, Int.mul_comm v u]
    have s_ubva : (u * b) * (v * a) = (u * v) * (a * b) := by
      rw [Int.mul_comm (u * b) (v * a), s_vaub]
    have s_vbua : (v * b) * (u * a) = (u * v) * (a * b) := by
      rw [int_mul_pair_swap v b u a, Int.mul_comm v u, Int.mul_comm b a]
    have s_uavb : (u * a) * (v * b) = (u * v) * (a * b) := by
      rw [Int.mul_comm (u * a) (v * b), s_vbua]
    rw [s_vava, s_ubub, s_vbvb, s_uaua, s_vaub, s_ubva, s_vbua, s_uavb]
    have d1 : (v * v) * (u * u) = (u * u) * (v * v) := Int.mul_comm _ _
    have d2 : (a * a) * (u * u) = (u * u) * (a * a) := Int.mul_comm _ _
    have d3 : (b * b) * (u * u) = (u * u) * (b * b) := Int.mul_comm _ _
    have d4 : (a * a) * (v * v) = (v * v) * (a * a) := Int.mul_comm _ _
    have d5 : (b * b) * (v * v) = (v * v) * (b * b) := Int.mul_comm _ _
    have d6 : (b * b) * (a * a) = (a * a) * (b * b) := Int.mul_comm _ _
    rw [d1, d2, d3, d4, d5, d6]
    omega
  have hoff_sym : ∀ u v a b : Int,
      (u * u + v * v - a * a - b * b) * (2 * (v * a - u * b))
      + (2 * (v * a + u * b)) * (u * u - v * v + a * a - b * b)
      + (2 * (v * b - u * a)) * (2 * (a * b + u * v)) = 0 := by
    intro u v a b
    have hpull1 : ∀ p q : Int, p * (2 * q) = 2 * (p * q) := by
      intro p q; rw [Int.mul_comm p (2 * q), Int.mul_assoc, Int.mul_comm q p]
    have hpull2 : ∀ p q : Int, (2 * p) * q = 2 * (p * q) := by
      intro p q; rw [Int.mul_assoc]
    have hpull4 : ∀ p q : Int, (2 * p) * (2 * q) = 4 * (p * q) := by
      intro p q; rw [int_mul_pair_swap 2 p 2 q]; rfl
    rw [hpull1, hpull2, hpull4]
    simp only [Int.add_mul, Int.sub_mul, Int.mul_add, Int.mul_sub]
    have m_va_uu : (v * a) * (u * u) = (u * u) * (v * a) := Int.mul_comm _ _
    have m_va_vv : (v * a) * (v * v) = (v * v) * (v * a) := Int.mul_comm _ _
    have m_va_aa : (v * a) * (a * a) = (a * a) * (v * a) := Int.mul_comm _ _
    have m_va_bb : (v * a) * (b * b) = (b * b) * (v * a) := Int.mul_comm _ _
    have m_ub_uu : (u * b) * (u * u) = (u * u) * (u * b) := Int.mul_comm _ _
    have m_ub_vv : (u * b) * (v * v) = (v * v) * (u * b) := Int.mul_comm _ _
    have m_ub_aa : (u * b) * (a * a) = (a * a) * (u * b) := Int.mul_comm _ _
    have m_ub_bb : (u * b) * (b * b) = (b * b) * (u * b) := Int.mul_comm _ _
    have m_vb_ab : (v * b) * (a * b) = (b * b) * (v * a) := by
      rw [int_mul_pair_swap v b a b, Int.mul_comm (v * a) (b * b)]
    have m_vb_uv : (v * b) * (u * v) = (v * v) * (u * b) := by
      rw [int_mul_pair_swap v b u v, Int.mul_comm v u, Int.mul_comm b v,
          int_mul_pair_swap u v v b, Int.mul_comm u v,
          int_mul_pair_swap v u v b]
    have m_ua_ab : (u * a) * (a * b) = (a * a) * (u * b) := by
      rw [int_mul_pair_swap u a a b, Int.mul_comm u a,
          int_mul_pair_swap a u a b]
    have m_ua_uv : (u * a) * (u * v) = (u * u) * (v * a) := by
      rw [int_mul_pair_swap u a u v, Int.mul_comm a v]
    rw [m_va_uu, m_va_vv, m_va_aa, m_va_bb, m_ub_uu, m_ub_vv, m_ub_aa, m_ub_bb,
        m_vb_ab, m_vb_uv, m_ua_ab, m_ua_uv]
    omega
  refine ⟨?_, ?_, ?_⟩
  · simp only [mat3TransposeMul, cayleySO3, Mat3.mk.injEq]
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · exact hnorm_sym d x y z
    · exact hoff_sym d x y z
    · have h := hoff_sym d x z (-y)
      simp only [Int.mul_neg, Int.neg_mul, Int.neg_neg] at h
      have h1 : d * d + x * x - z * z - y * y = d * d + x * x - y * y - z * z := by omega
      have h2 : d * d - x * x + z * z - y * y = d * d - x * x - y * y + z * z := by omega
      rw [h1, h2] at h
      have hs1 : 2 * (x * z + d * y) = 2 * (x * z - -(d * y)) := by omega
      have hs2 : 2 * (x * z - d * y) = 2 * (x * z + -(d * y)) := by omega
      have hs3 : (2 * (x * y + d * z)) * (2 * (y * z - d * x))
          = (2 * (-(x * y) - d * z)) * (2 * (-(z * y) + d * x)) := by
        rw [Int.mul_comm z y]
        have ha : -(x * y) - d * z = -(x * y + d * z) := by omega
        have hb : -(y * z) + d * x = -(y * z - d * x) := by omega
        rw [ha, hb, Int.mul_neg, Int.mul_neg, Int.neg_mul_neg]
      rw [hs1, hs2, hs3]
      have hord : ∀ p1 p2 p3 : Int, p1 + p3 + p2 = p1 + p2 + p3 := by intros; omega
      rw [hord]
      exact h
    · have h := hoff_sym d x y z
      rw [Int.mul_comm (2 * (x * y - d * z)),
          Int.mul_comm (d * d - x * x + y * y - z * z),
          Int.mul_comm (2 * (y * z + d * x))]
      exact h
    · have h := hnorm_sym d y z x
      have hD : d * d + y * y + z * z + x * x = d * d + x * x + y * y + z * z := by omega
      have h1 : d * d + y * y - z * z - x * x = d * d - x * x + y * y - z * z := by omega
      have h2 : (2 * (x * y - d * z)) * (2 * (x * y - d * z))
          = (2 * (y * x - d * z)) * (2 * (y * x - d * z)) := by rw [Int.mul_comm x y]
      rw [hD, h1] at h
      rw [h2]
      have hr : ∀ a b c : Int, a + b + c = b + c + a := by intros; omega
      rw [hr]
      exact h
    · have h := hoff_sym d y z x
      have h1 : d * d + y * y - z * z - x * x = d * d - x * x + y * y - z * z := by omega
      have h2 : d * d - y * y + z * z - x * x = d * d - x * x - y * y + z * z := by omega
      rw [h1, h2] at h
      have hc1 : (2 * (x * y - d * z)) * (2 * (x * z + d * y))
          = (2 * (y * x - d * z)) * (2 * (z * x + d * y)) := by
        rw [Int.mul_comm x y, Int.mul_comm x z]
      rw [hc1]
      have hr : ∀ a b c : Int, c + a + b = a + b + c := by intros; omega
      rw [hr]
      exact h
    · have h := hoff_sym d x z (-y)
      simp only [Int.mul_neg, Int.neg_mul, Int.neg_neg] at h
      have h1 : d * d + x * x - z * z - y * y = d * d + x * x - y * y - z * z := by omega
      have h2 : d * d - x * x + z * z - y * y = d * d - x * x - y * y + z * z := by omega
      rw [h1, h2] at h
      rw [Int.mul_comm (2 * (x * z + d * y)),
          Int.mul_comm (2 * (y * z - d * x)),
          Int.mul_comm (d * d - x * x - y * y + z * z)]
      have hs1 : 2 * (x * z + d * y) = 2 * (x * z - -(d * y)) := by omega
      have hs2 : 2 * (x * z - d * y) = 2 * (x * z + -(d * y)) := by omega
      have hs3 : (2 * (x * y + d * z)) * (2 * (y * z - d * x))
          = (2 * (-(x * y) - d * z)) * (2 * (-(z * y) + d * x)) := by
        rw [Int.mul_comm z y]
        have ha : -(x * y) - d * z = -(x * y + d * z) := by omega
        have hb : -(y * z) + d * x = -(y * z - d * x) := by omega
        rw [ha, hb, Int.mul_neg, Int.mul_neg, Int.neg_mul_neg]
      rw [hs1, hs2, hs3]
      have hord : ∀ p1 p2 p3 : Int, p1 + p3 + p2 = p1 + p2 + p3 := by intros; omega
      rw [hord]
      exact h
    · have h := hoff_sym d y z x
      have h1 : d * d + y * y - z * z - x * x = d * d - x * x + y * y - z * z := by omega
      have h2 : d * d - y * y + z * z - x * x = d * d - x * x - y * y + z * z := by omega
      rw [h1, h2] at h
      rw [Int.mul_comm (2 * (x * z + d * y)),
          Int.mul_comm (2 * (y * z - d * x)),
          Int.mul_comm (d * d - x * x - y * y + z * z)]
      have hc1 : (2 * (x * y - d * z)) * (2 * (x * z + d * y))
          = (2 * (y * x - d * z)) * (2 * (z * x + d * y)) := by
        rw [Int.mul_comm x y, Int.mul_comm x z]
      rw [hc1]
      have hr : ∀ a b c : Int, c + a + b = a + b + c := by intros; omega
      rw [hr]
      exact h
    · have h := hnorm_sym d z x y
      have hD : d * d + z * z + x * x + y * y = d * d + x * x + y * y + z * z := by omega
      have h1 : d * d + z * z - x * x - y * y = d * d - x * x - y * y + z * z := by omega
      have hc1 : (2 * (x * z + d * y)) * (2 * (x * z + d * y))
          = (2 * (z * x + d * y)) * (2 * (z * x + d * y)) := by rw [Int.mul_comm x z]
      have hc2 : (2 * (y * z - d * x)) * (2 * (y * z - d * x))
          = (2 * (z * y - d * x)) * (2 * (z * y - d * x)) := by rw [Int.mul_comm y z]
      rw [hD, h1] at h
      rw [hc1, hc2]
      have hr : ∀ a b c : Int, a + b + c = c + a + b := by intros; omega
      rw [hr]
      exact h
  · simp only [mat3Det, cayleySO3]
    have hcof : ∀ u v a b : Int,
        (u * u - v * v + a * a - b * b) * (u * u - v * v - a * a + b * b)
        - (2 * (a * b - u * v)) * (2 * (a * b + u * v))
        = (u * u + v * v + a * a + b * b) * (u * u + v * v - a * a - b * b) := by
      intro u v a b
      have h4 : ∀ p q : Int, (2 * p) * (2 * q) = 4 * (p * q) := by
        intro p q; rw [int_mul_pair_swap 2 p 2 q]; rfl
      rw [h4]
      simp only [Int.add_mul, Int.sub_mul, Int.mul_add, Int.mul_sub]
      have s1 : (a * b) * (a * b) = (a * a) * (b * b) := prod_sq_split a b
      have s2 : (u * v) * (u * v) = (u * u) * (v * v) := prod_sq_split u v
      have s3 : (u * v) * (a * b) = (a * b) * (u * v) := Int.mul_comm _ _
      rw [s1, s2, s3]
      have d1 : (v * v) * (u * u) = (u * u) * (v * v) := Int.mul_comm _ _
      have d2 : (a * a) * (u * u) = (u * u) * (a * a) := Int.mul_comm _ _
      have d3 : (b * b) * (u * u) = (u * u) * (b * b) := Int.mul_comm _ _
      have d4 : (a * a) * (v * v) = (v * v) * (a * a) := Int.mul_comm _ _
      have d5 : (b * b) * (v * v) = (v * v) * (b * b) := Int.mul_comm _ _
      have d6 : (b * b) * (a * a) = (a * a) * (b * b) := Int.mul_comm _ _
      rw [d1, d2, d3, d4, d5, d6]
      omega
    have hcof_off : ∀ u v a b : Int,
        (2 * (v * a + u * b)) * (u * u - v * v - a * a + b * b)
        - (2 * (a * b - u * v)) * (2 * (v * b - u * a))
        = (u * u + v * v + a * a + b * b) * (2 * (u * b - v * a)) := by
      intro u v a b
      have hp1 : ∀ p q : Int, (2 * p) * q = 2 * (p * q) := by intro p q; rw [Int.mul_assoc]
      have hp2 : ∀ p q : Int, p * (2 * q) = 2 * (p * q) := by
        intro p q; rw [Int.mul_comm p (2 * q), Int.mul_assoc, Int.mul_comm q p]
      have hp4 : ∀ p q : Int, (2 * p) * (2 * q) = 4 * (p * q) := by
        intro p q; rw [int_mul_pair_swap 2 p 2 q]; rfl
      rw [hp4, hp1, hp2]
      simp only [Int.add_mul, Int.sub_mul, Int.mul_add, Int.mul_sub]
      have m1 : (v * a) * (u * u) = (u * u) * (v * a) := Int.mul_comm _ _
      have m2 : (v * a) * (v * v) = (v * v) * (v * a) := Int.mul_comm _ _
      have m3 : (v * a) * (a * a) = (a * a) * (v * a) := Int.mul_comm _ _
      have m4 : (v * a) * (b * b) = (b * b) * (v * a) := Int.mul_comm _ _
      have m5 : (u * b) * (u * u) = (u * u) * (u * b) := Int.mul_comm _ _
      have m6 : (u * b) * (v * v) = (v * v) * (u * b) := Int.mul_comm _ _
      have m7 : (u * b) * (a * a) = (a * a) * (u * b) := Int.mul_comm _ _
      have m8 : (u * b) * (b * b) = (b * b) * (u * b) := Int.mul_comm _ _
      have n1 : (a * b) * (v * b) = (b * b) * (v * a) := by
        rw [int_mul_pair_swap a b v b, Int.mul_comm a v, Int.mul_comm (v * a) (b * b)]
      have n2 : (a * b) * (u * a) = (a * a) * (u * b) := by
        rw [int_mul_pair_swap a b u a, Int.mul_comm a u, Int.mul_comm b a,
            int_mul_pair_swap u a a b, Int.mul_comm u a,
            int_mul_pair_swap a u a b]
      have n3 : (u * v) * (v * b) = (v * v) * (u * b) := by
        rw [int_mul_pair_swap u v v b, Int.mul_comm u v,
            int_mul_pair_swap v u v b]
      have n4 : (u * v) * (u * a) = (u * u) * (v * a) := by
        rw [int_mul_pair_swap u v u a]
      rw [m1, m2, m3, m4, m5, m6, m7, m8, n1, n2, n3, n4]
      omega
    have c00 : (d * d - x * x + y * y - z * z) * (d * d - x * x - y * y + z * z)
        - (2 * (y * z - d * x)) * (2 * (y * z + d * x))
        = (d * d + x * x + y * y + z * z) * (d * d + x * x - y * y - z * z) := hcof d x y z
    have c01 : (2 * (x * y + d * z)) * (d * d - x * x - y * y + z * z)
        - (2 * (y * z - d * x)) * (2 * (x * z - d * y))
        = -((d * d + x * x + y * y + z * z) * (2 * (x * y - d * z))) := by
      have h := hcof_off d x y z
      have hneg : 2 * (d * z - x * y) = -(2 * (x * y - d * z)) := by omega
      rw [hneg, Int.mul_neg] at h
      exact h
    have c02 : (2 * (x * y + d * z)) * (2 * (y * z + d * x))
        - (d * d - x * x + y * y - z * z) * (2 * (x * z - d * y))
        = (d * d + x * x + y * y + z * z) * (2 * (x * z + d * y)) := by
      have h := hcof_off d x z (-y)
      simp only [Int.mul_neg, Int.neg_mul, Int.neg_neg] at h
      have hD' : d * d + x * x + z * z + y * y = d * d + x * x + y * y + z * z := by omega
      have hmid : d * d - x * x - z * z + y * y = d * d - x * x + y * y - z * z := by omega
      rw [hD', hmid] at h
      have hp1 : (2 * (x * z - d * y)) * (d * d - x * x + y * y - z * z)
          = (d * d - x * x + y * y - z * z) * (2 * (x * z - d * y)) := Int.mul_comm _ _
      have hs1 : 2 * (x * z + -(d * y)) = 2 * (x * z - d * y) := by omega
      have hs2 : 2 * (-(d * y) - x * z) = -(2 * (x * z + d * y)) := by omega
      have hs3 : (2 * (-(z * y) - d * x)) * (2 * (-(x * y) - d * z))
          = (2 * (x * y + d * z)) * (2 * (y * z + d * x)) := by
        rw [Int.mul_comm z y, Int.mul_comm (2 * (-(y * z) - d * x))]
        have ha : -(x * y) - d * z = -(x * y + d * z) := by omega
        have hb : -(y * z) - d * x = -(y * z + d * x) := by omega
        rw [ha, hb, Int.mul_neg, Int.mul_neg, Int.neg_mul_neg]
      rw [hs1, hp1, hs3, hs2, Int.mul_neg] at h
      omega
    rw [c00, c01, c02, Int.mul_neg, Int.sub_neg]
    have hrow : ∀ D : Int,
        (d * d + x * x - y * y - z * z) * (D * (d * d + x * x - y * y - z * z))
        + (2 * (x * y - d * z)) * (D * (2 * (x * y - d * z)))
        + (2 * (x * z + d * y)) * (D * (2 * (x * z + d * y)))
        = D * ((d * d + x * x - y * y - z * z) * (d * d + x * x - y * y - z * z)
               + (2 * (x * y - d * z)) * (2 * (x * y - d * z))
               + (2 * (x * z + d * y)) * (2 * (x * z + d * y))) := by
      intro D
      have hp : ∀ a : Int, a * (D * a) = D * (a * a) := by
        intro a; rw [← Int.mul_assoc, Int.mul_comm a D, Int.mul_assoc]
      rw [hp, hp, hp, ← Int.mul_add, ← Int.mul_add]
    rw [hrow (d * d + x * x + y * y + z * z)]
    have hrow_norm := hnorm_sym d x (-y) z
    simp only [Int.mul_neg, Int.neg_mul, Int.neg_neg] at hrow_norm
    have ha1 : (2 * (-(x * y) + d * z)) * (2 * (-(x * y) + d * z))
        = (2 * (x * y - d * z)) * (2 * (x * y - d * z)) := by
      have : -(x * y) + d * z = -(x * y - d * z) := by omega
      rw [this, Int.mul_neg, Int.neg_mul_neg]
    have ha2 : 2 * (x * z - -(d * y)) = 2 * (x * z + d * y) := by omega
    rw [ha1, ha2] at hrow_norm
    rw [hrow_norm, ← Int.mul_assoc]
  · intro hd
    have hD : 0 < d * d + x * x + y * y + z * z := (cayley_su2_unit_sphere_universal d x y z).2 hd
    exact Int.mul_pos (Int.mul_pos hD hD) hD

/-- Theorem 21: Universal off-manifold obstruction for naive Euclidean additive ES
    on S^3: adding any nonzero orthogonal tangent sigma * v (<q, v> = 0, sigma != 0,
    ||v||^2 > 0) strictly inflates ||q + sigma * v||^2 = ||q||^2 + sigma^2 * ||v||^2 > ||q||^2. -/
theorem euclidean_additive_es_off_manifold_obstruction_universal
    (q v : Q4) (sigma : Int)
    (horth : q.w * v.w + q.x * v.x + q.y * v.y + q.z * v.z = 0)
    (hsig : sigma ≠ 0) (hv : 0 < qNormSq v) :
    qNormSq ⟨q.w + sigma * v.w, q.x + sigma * v.x, q.y + sigma * v.y, q.z + sigma * v.z⟩
      = qNormSq q + (sigma * sigma) * qNormSq v
    ∧ qNormSq q < qNormSq ⟨q.w + sigma * v.w, q.x + sigma * v.x, q.y + sigma * v.y, q.z + sigma * v.z⟩ := by
  have heq : qNormSq ⟨q.w + sigma * v.w, q.x + sigma * v.x, q.y + sigma * v.y, q.z + sigma * v.z⟩
      = qNormSq q + (sigma * sigma) * qNormSq v := by
    simp only [qNormSq]
    have hcomp : ∀ a b : Int, (a + sigma * b) * (a + sigma * b)
        = a * a + 2 * sigma * (a * b) + (sigma * sigma) * (b * b) := by
      intro a b
      simp only [Int.add_mul, Int.mul_add]
      rw [prod_sq_split sigma b]
      have h1 : a * (sigma * b) = sigma * (a * b) := by
        rw [← Int.mul_assoc, Int.mul_comm a sigma, Int.mul_assoc]
      have h2 : (sigma * b) * a = sigma * (a * b) := by
        rw [Int.mul_assoc, Int.mul_comm b a]
      rw [h1, h2, Int.mul_assoc 2 sigma (a * b)]
      omega
    rw [hcomp, hcomp, hcomp, hcomp]
    have hzero : 2 * sigma * (q.w * v.w) + 2 * sigma * (q.x * v.x)
        + 2 * sigma * (q.y * v.y) + 2 * sigma * (q.z * v.z) = 0 := by
      rw [← Int.mul_add, ← Int.mul_add, ← Int.mul_add, horth, Int.mul_zero]
    simp only [Int.mul_add]
    omega
  have hsig2 : 0 < sigma * sigma := int_sq_pos_of_ne_zero sigma hsig
  have hpos : 0 < (sigma * sigma) * qNormSq v := Int.mul_pos hsig2 hv
  exact ⟨heq, by omega⟩

end EvolutionOperators

#print axioms EvolutionOperators.antithetic_poly4_even_cancellation_universal
#print axioms EvolutionOperators.antithetic_poly4_odd_cancellation_universal
#print axioms EvolutionOperators.antithetic_vec_linear_quadratic_cancellation_universal
#print axioms EvolutionOperators.antithetic_variance_strictly_beats_onesided_universal
#print axioms EvolutionOperators.hadamard_h4_pairwise_orthogonality_and_parseval_universal
#print axioms EvolutionOperators.randomized_block_h4_orthogonality_and_parseval_universal
#print axioms EvolutionOperators.orthogonal_h4_es_exact_recovery_zero_mse_universal
#print axioms EvolutionOperators.orthogonal_es_zero_crossterm_mse_reduction_universal
#print axioms EvolutionOperators.shared_seed_cross_layer_correlation_obstruction_universal
#print axioms EvolutionOperators.heat_kernel_goodhart_spike_attenuation_universal
#print axioms EvolutionOperators.discrete_pascal_heat_kernel_plateau_conservation_universal
#print axioms EvolutionOperators.discrete_gaussian_center_spike_strict_contraction_universal
#print axioms EvolutionOperators.bf16_ulp_half_threshold_stall_universal
#print axioms EvolutionOperators.bf16_canonical_es_ulp_stall_witness
#print axioms EvolutionOperators.alkashi_fixedradix_ef_exact_conservation_universal
#print axioms EvolutionOperators.cs08_es_vs_grpo_flop_and_hbm_crossover_universal
#print axioms EvolutionOperators.folded_scale_zero_byte_es_invariance_universal
#print axioms EvolutionOperators.cayley_su2_unit_sphere_universal
#print axioms EvolutionOperators.cayley_su2_composition_norm_closure_universal
#print axioms EvolutionOperators.cayley_so3_tangent_orthogonality_and_det_universal
#print axioms EvolutionOperators.euclidean_additive_es_off_manifold_obstruction_universal
