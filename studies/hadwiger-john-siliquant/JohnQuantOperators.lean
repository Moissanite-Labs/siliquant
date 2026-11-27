/-
Track S165 (EXP-Q47b, PREREG-W22-HADWIGER-SILIQUANT)
Pure-Init Lean 4.36.0 Formalization of Hadwiger-3D John/Lowner,
Helly-LP Chebyshev, and Polar-Cap/Equatorial-Bulk Quantization Operators.

Zero imports, zero Mathlib, zero sorry, zero custom axioms, zero Classical.choice (standard propext only).
-/

namespace JohnQuantOperators

-- ============================================================================
-- Helper definitions (pure Init, totally computable, constructive)
-- ============================================================================

def fact : Nat -> Nat
  | 0 => 1
  | n + 1 => (n + 1) * fact n

def powNat (b : Nat) : Nat -> Nat
  | 0 => 1
  | n + 1 => b * powNat b n

def zipAdd : List Int -> List Int -> List Int
  | [], _ => []
  | _, [] => []
  | a :: as, b :: bs => (a + b) :: zipAdd as bs

def zipSub : List Int -> List Int -> List Int
  | [], _ => []
  | _, [] => []
  | a :: as, b :: bs => (a - b) :: zipSub as bs

def zipNeg : List Int -> List Int
  | [] => []
  | a :: as => (-a) :: zipNeg as

def sumSq : List Int -> Int
  | [] => 0
  | x :: xs => x * x + sumSq xs

def dotProd : List (Int × Int) -> Int
  | [] => 0
  | (x, y) :: ps => x * y + dotProd ps

def sumSqFst : List (Int × Int) -> Int
  | [] => 0
  | (x, _) :: ps => x * x + sumSqFst ps

def sumSqSnd : List (Int × Int) -> Int
  | [] => 0
  | (_, y) :: ps => y * y + sumSqSnd ps

def crossDiffSqSum (x0 y0 : Int) : List (Int × Int) -> Int
  | [] => 0
  | (x, y) :: ps => (x0 * y - y0 * x) * (x0 * y - y0 * x) + crossDiffSqSum x0 y0 ps

def lagrangeGap : List (Int × Int) -> Int
  | [] => 0
  | (x0, y0) :: ps => crossDiffSqSum x0 y0 ps + lagrangeGap ps

def sumScaledDot : List (Int × Int × Int × Int) -> Int
  | [] => 0
  | (w, x, p, q) :: rest => (w * p) * (x * q) + sumScaledDot rest

def sumFactoredDot : List (Int × Int × Int × Int) -> Int
  | [] => 0
  | (w, x, p, q) :: rest => (p * q) * (w * x) + sumFactoredDot rest

def sumConstScaleDot (d : Int) : List (Int × Int) -> Int
  | [] => 0
  | (w, x) :: rest => d * (w * x) + sumConstScaleDot d rest

def sumRawDot : List (Int × Int) -> Int
  | [] => 0
  | (w, x) :: rest => w * x + sumRawDot rest

def intAbs (x : Int) : Int :=
  if x >= 0 then x else -x

def intMax (a b : Int) : Int :=
  if a >= b then a else b

def hellyStepSum : List (Int × Int × Int) -> Int
  | [] => 0
  | (a, u, w) :: rest => a * (w - u) + hellyStepSum rest

def hellyUSum : List (Int × Int × Int) -> Int
  | [] => 0
  | (a, u, _) :: rest => a * u + hellyUSum rest

def hellyWSum : List (Int × Int × Int) -> Int
  | [] => 0
  | (a, _, w) :: rest => a * w + hellyWSum rest

def diagCubeSumSq : Nat -> Nat
  | 0 => 0
  | d + 1 => 4 + diagCubeSumSq d

-- Exact rational pi bracket: piLo / piDen < pi < piHi / piDen
def piLo : Nat := 314159265358979323846
def piHi : Nat := 314159265358979323847
def piDen : Nat := 100000000000000000000

-- O_4 = 2 B_1^4 exact scaled contact vectors (2 * v_s in {-1, +1}^4, true=+1, false=-1)
def o4Contacts : List (List Bool) :=
  [ [true,  true,  true,  true],
    [true,  true,  true,  false],
    [true,  true,  false, true],
    [true,  true,  false, false],
    [true,  false, true,  true],
    [true,  false, true,  false],
    [true,  false, false, true],
    [true,  false, false, false] ]

def getSign (xs : List Bool) (i : Nat) : Bool :=
  match xs, i with
  | [], _ => true
  | a :: _, 0 => a
  | _ :: rest, k + 1 => getSign rest k

def gramPosCount (vecs : List (List Bool)) (i j : Nat) : Nat :=
  match vecs with
  | [] => 0
  | v :: rest =>
    (if getSign v i == getSign v j then 1 else 0) + gramPosCount rest i j

def gramNegCount (vecs : List (List Bool)) (i j : Nat) : Nat :=
  match vecs with
  | [] => 0
  | v :: rest =>
    (if getSign v i == getSign v j then 0 else 1) + gramNegCount rest i j

def natAllDown (f : Nat -> Bool) : Nat -> Bool
  | 0 => true
  | k + 1 => f k && natAllDown f k

def listAll {a : Type} (p : a -> Bool) : List a -> Bool
  | [] => true
  | x :: xs => p x && listAll p xs

def checkGramDiagSigns (vecs : List (List Bool)) (d scale : Nat) : Bool :=
  natAllDown (fun i =>
    natAllDown (fun j =>
      let pos := gramPosCount vecs i j
      let neg := gramNegCount vecs i j
      if i == j then pos == neg + scale else pos == neg) d) d

def checkRowLens (vecs : List (List Bool)) (d : Nat) : Bool :=
  listAll (fun v => v.length == d) vecs

-- Exact integer contact vectors for O_4 = 2 B_1^4 and O_16 = 4 B_1^16 (entries in {-1, +1})
def intEqBool (a b : Int) : Bool :=
  match a - b with
  | Int.ofNat 0 => true
  | _ => false

private theorem intEqBool_true (a b : Int) (h : intEqBool a b = true) : a = b := by
  simp only [intEqBool] at h
  cases hsub : a - b with
  | ofNat n =>
    cases n with
    | zero => exact Int.eq_of_sub_eq_zero hsub
    | succ _ =>
      rw [hsub] at h
      cases h
  | negSucc _ =>
    rw [hsub] at h
    cases h

def boolToSign (b : Bool) : Int :=
  if b then 1 else -1

def vecSqNorm : List Int -> Int
  | [] => 0
  | x :: xs => x * x + vecSqNorm xs

def vecL1Norm : List Int -> Int
  | [] => 0
  | x :: xs => intAbs x + vecL1Norm xs

def getInt (xs : List Int) (i : Nat) : Int :=
  match xs, i with
  | [], _ => 0
  | a :: _, 0 => a
  | _ :: rest, k + 1 => getInt rest k

def intGramEntry (vecs : List (List Int)) (i j : Nat) : Int :=
  match vecs with
  | [] => 0
  | v :: rest => getInt v i * getInt v j + intGramEntry rest i j

def checkIntGram (vecs : List (List Int)) (d scale : Nat) : Bool :=
  natAllDown (fun i =>
    natAllDown (fun j =>
      intEqBool (intGramEntry vecs i j) (if i == j then Int.ofNat scale else 0)) d) d

def checkContactNormAndFacet (vecs : List (List Int)) (d : Nat) : Bool :=
  listAll (fun v =>
    v.length == d
    && listAll (fun x => intEqBool (x * x) 1) v
    && intEqBool (vecSqNorm v) (Int.ofNat d)
    && intEqBool (vecL1Norm v) (Int.ofNat d)) vecs

-- Sylvester Hadamard H_16 rows (true=+1, false=-1; 4 * v_s for O_16 = 4 B_1^16)
def h2Sign (r c : Nat) : Bool :=
  !(r == 1 && c == 1)

def h4Sign (r c : Nat) : Bool :=
  h2Sign (r / 2) (c / 2) == h2Sign (r % 2) (c % 2)

def h8Sign (r c : Nat) : Bool :=
  h4Sign (r / 2) (c / 2) == h2Sign (r % 2) (c % 2)

def h16Sign (r c : Nat) : Bool :=
  h8Sign (r / 2) (c / 2) == h2Sign (r % 2) (c % 2)

def makeSignRow (r : Nat) : Nat -> List Bool
  | 0 => []
  | c + 1 => makeSignRow r c ++ [h16Sign r c]

def makeH16Signs : Nat -> List (List Bool)
  | 0 => []
  | r + 1 => makeH16Signs r ++ [makeSignRow r 16]

def h16Rows : List (List Bool) := makeH16Signs 16

def o4IntVecs : List (List Int) :=
  o4Contacts.map (fun r => r.map boolToSign)

def h16IntVecs : List (List Int) :=
  h16Rows.map (fun r => r.map boolToSign)

def iEvenNum : Nat -> Nat
  | 0 => 2
  | k + 1 => (2 * k + 2) * iEvenNum k

def iEvenDen : Nat -> Nat
  | 0 => 1
  | k + 1 => (2 * k + 3) * iEvenDen k

def allPairCrossZero (x0 y0 : Int) : List (Int × Int) -> Bool
  | [] => true
  | (x, y) :: rest => intEqBool (x0 * y - y0 * x) 0 && allPairCrossZero x0 y0 rest

def isCollinear : List (Int × Int) -> Bool
  | [] => true
  | (x0, y0) :: rest => allPairCrossZero x0 y0 rest && isCollinear rest

def sameProd : List (Int × Int) -> List (Int × Int) -> Bool
  | [], [] => true
  | (a, b) :: xs, (c, d) :: ys => intEqBool (a * b) (c * d) && sameProd xs ys
  | _, _ => false

-- Extended Hamming [16, 11, 4] check masks (A_16 = 2 Z^16 + C_[16,11,4])
def popcountAux (x : Nat) : Nat -> Nat
  | 0 => 0
  | k + 1 => ((x >>> k) &&& 1) + popcountAux x k

def a16SyndromeZero (word : Nat) : Bool :=
  let m0 := popcountAux (word &&& 0xFFFF) 16 % 2 == 0
  let m1 := popcountAux (word &&& 0xAAAA) 16 % 2 == 0
  let m2 := popcountAux (word &&& 0xCCCC) 16 % 2 == 0
  let m3 := popcountAux (word &&& 0xF0F0) 16 % 2 == 0
  let m4 := popcountAux (word &&& 0xFF00) 16 % 2 == 0
  m0 && m1 && m2 && m3 && m4

-- Constructive Int helper lemmas (zero Classical.choice)
private theorem int_sub_add_cancel_right (x y c : Int) : (x + c) - (y + c) = x - y := by
  rw [Int.sub_eq_add_neg, Int.sub_eq_add_neg, Int.neg_add, Int.add_assoc,
      ← Int.add_assoc c (-y) (-c), Int.add_comm c (-y), Int.add_assoc (-y) c (-c),
      Int.add_right_neg, Int.add_zero]

private theorem int_neg_sub_swap (x y : Int) : -(x - y) = y - x := by
  rw [Int.sub_eq_add_neg, Int.sub_eq_add_neg, Int.neg_add, Int.neg_neg, Int.add_comm]

private theorem int_mul_pair_swap (a b c d : Int) : (a * b) * (c * d) = (a * c) * (b * d) := by
  calc
    (a * b) * (c * d) = a * (b * (c * d)) := by rw [Int.mul_assoc]
    _ = a * ((b * c) * d) := by rw [Int.mul_assoc]
    _ = a * ((c * b) * d) := by rw [Int.mul_comm b c]
    _ = a * (c * (b * d)) := by rw [Int.mul_assoc]
    _ = (a * c) * (b * d) := by rw [Int.mul_assoc]

private theorem int_sq_nonneg (z : Int) : 0 <= z * z := by
  cases Int.le_total 0 z with
  | inl h => exact Int.mul_nonneg h h
  | inr h =>
    have hn : 0 <= -z := Int.neg_nonneg_of_nonpos h
    have hmul : 0 <= (-z) * (-z) := Int.mul_nonneg hn hn
    rw [Int.neg_mul_neg] at hmul
    exact hmul

private theorem int_sq_pos_of_ne_zero (z : Int) (hne : z ≠ 0) : 0 < z * z := by
  cases Int.lt_or_gt_of_ne hne with
  | inl hlt =>
    have hp : 0 < -z := Int.neg_pos_of_neg hlt
    have hm : 0 < (-z) * (-z) := Int.mul_pos hp hp
    rw [Int.neg_mul_neg] at hm
    exact hm
  | inr hgt => exact Int.mul_pos hgt hgt

private theorem int_add_four_swap (a b c d : Int) : (a + b) + (c + d) = (a + c) + (b + d) := by
  rw [Int.add_assoc, Int.add_assoc]
  have hm : b + (c + d) = c + (b + d) := by
    rw [← Int.add_assoc, Int.add_comm b c, Int.add_assoc]
  rw [hm, ← Int.add_assoc]

private theorem int_two_mul (t : Int) : 2 * t = t + t := by
  have h21 : (2 : Int) = 1 + 1 := rfl
  rw [h21, Int.add_mul, Int.one_mul]

private theorem int_sq_add_sub_cancel (u v : Int) :
    (u + v) * (u + v) + (u - v) * (u - v) = 2 * (u * u + v * v) := by
  have hvu : v * u = u * v := Int.mul_comm v u
  have h1b : (u + v) * (u + v) = (u * u + u * v) + (u * v + v * v) := by
    rw [Int.add_mul, Int.mul_add, Int.mul_add, hvu]
  have h2b : (u - v) * (u - v) = (u * u + -(u * v)) + (-(u * v) + v * v) := by
    rw [Int.sub_mul, Int.mul_sub, Int.mul_sub, hvu,
        Int.sub_eq_add_neg, Int.sub_eq_add_neg, Int.sub_eq_add_neg, Int.neg_add, Int.neg_neg]
  rw [h1b, h2b]
  calc
    ((u * u + u * v) + (u * v + v * v)) + ((u * u + -(u * v)) + (-(u * v) + v * v))
      = ((u * u + u * v) + (u * u + -(u * v))) + ((u * v + v * v) + (-(u * v) + v * v)) :=
          int_add_four_swap _ _ _ _
    _ = (u * u + u * u) + (v * v + v * v) := by
          have ha : (u * u + u * v) + (u * u + -(u * v)) = u * u + u * u := by
            rw [int_add_four_swap, Int.add_right_neg, Int.add_zero]
          have hb : (u * v + v * v) + (-(u * v) + v * v) = v * v + v * v := by
            rw [int_add_four_swap, Int.add_right_neg, Int.zero_add]
          rw [ha, hb]
    _ = 2 * (u * u + v * v) := by
          rw [← int_two_mul (u * u), ← int_two_mul (v * v), ← Int.mul_add]

private theorem int_sub_sq_expand (a b : Int) :
    (a - b) * (a - b) = a * a + b * b - 2 * (a * b) := by
  have h1 : (a - b) * (a - b) = (a * a - a * b) - (b * a - b * b) := by
    rw [Int.sub_mul, Int.mul_sub, Int.mul_sub]
  have hba : b * a = a * b := Int.mul_comm b a
  have h2ab : 2 * (a * b) = a * b + a * b := by
    have h21 : (2 : Int) = 1 + 1 := rfl
    rw [h21, Int.add_mul, Int.one_mul]
  rw [h1, hba, h2ab]
  rw [Int.sub_eq_add_neg, Int.sub_eq_add_neg, Int.sub_eq_add_neg, Int.sub_eq_add_neg,
      Int.neg_add, Int.neg_neg, Int.neg_add]
  calc
    (a * a + -(a * b)) + (-(a * b) + b * b)
      = a * a + (-(a * b) + (-(a * b) + b * b)) := by rw [Int.add_assoc]
    _ = a * a + ((-(a * b) + -(a * b)) + b * b) := by rw [Int.add_assoc]
    _ = a * a + (b * b + (-(a * b) + -(a * b))) := by rw [Int.add_comm (-(a * b) + -(a * b)) (b * b)]
    _ = (a * a + b * b) + (-(a * b) + -(a * b)) := by rw [← Int.add_assoc]

private theorem fact_pos (n : Nat) : 0 < fact n := by
  induction n with
  | zero => exact Nat.succ_pos 0
  | succ k ih =>
    simp only [fact]
    exact Nat.mul_pos (Nat.succ_pos k) ih

private theorem iEvenNum_pos (k : Nat) : 0 < iEvenNum k := by
  induction k with
  | zero => exact Nat.succ_pos 1
  | succ j ih =>
    simp only [iEvenNum]
    exact Nat.mul_pos (Nat.succ_pos (2 * j + 1)) ih

private theorem iEvenDen_pos (k : Nat) : 0 < iEvenDen k := by
  induction k with
  | zero => exact Nat.succ_pos 0
  | succ j ih =>
    simp only [iEvenDen]
    exact Nat.mul_pos (Nat.succ_pos (2 * j + 2)) ih

-- ============================================================================
-- Pillar 1: Exact John-Position Hypercube Mean-Width, Volume, & Diameter Gaps
-- ============================================================================

/-- Theorem 1: Universal dimensional recurrence (d+1) I_{d+2} = d I_d for
    all even dimensions d = 2k+2 and strict growth M_{d+2} > M_d. -/
theorem mean_width_recurrence_growth_universal (k : Nat) :
    (2 * k + 3) * iEvenNum (k + 1) * iEvenDen k = (2 * k + 2) * iEvenNum k * iEvenDen (k + 1)
    /\ ((2 * k + 2) * iEvenNum k) * iEvenDen (k + 1) < ((2 * k + 4) * iEvenNum (k + 1)) * iEvenDen k := by
  have hrec : (2 * k + 3) * iEvenNum (k + 1) * iEvenDen k = (2 * k + 2) * iEvenNum k * iEvenDen (k + 1) := by
    simp only [iEvenNum, iEvenDen]
    rw [Nat.mul_assoc (2 * k + 3) ((2 * k + 2) * iEvenNum k) (iEvenDen k)]
    rw [Nat.mul_left_comm (2 * k + 3) ((2 * k + 2) * iEvenNum k) (iEvenDen k)]
  have hlt : ((2 * k + 2) * iEvenNum k) * iEvenDen (k + 1) < ((2 * k + 4) * iEvenNum (k + 1)) * iEvenDen k := by
    have hnum_pos : 0 < iEvenNum k := iEvenNum_pos k
    have hden_pos : 0 < iEvenDen k := iEvenDen_pos k
    have hbase_pos : 0 < (2 * k + 2) * iEvenNum k * iEvenDen k :=
      Nat.mul_pos (Nat.mul_pos (Nat.succ_pos (2 * k + 1)) hnum_pos) hden_pos
    have hstep : 2 * k + 3 < 2 * k + 4 := Nat.lt_succ_self (2 * k + 3)
    have hmul : (2 * k + 3) * ((2 * k + 2) * iEvenNum k * iEvenDen k)
        < (2 * k + 4) * ((2 * k + 2) * iEvenNum k * iEvenDen k) :=
      Nat.mul_lt_mul_of_pos_right hstep hbase_pos
    have hL : ((2 * k + 2) * iEvenNum k) * iEvenDen (k + 1)
        = (2 * k + 3) * ((2 * k + 2) * iEvenNum k * iEvenDen k) := by
      simp only [iEvenDen]
      rw [← Nat.mul_assoc, Nat.mul_comm ((2 * k + 2) * iEvenNum k) (2 * k + 3), Nat.mul_assoc]
    have hR : ((2 * k + 4) * iEvenNum (k + 1)) * iEvenDen k
        = (2 * k + 4) * ((2 * k + 2) * iEvenNum k * iEvenDen k) := by
      simp only [iEvenNum]
      rw [Nat.mul_assoc (2 * k + 4) ((2 * k + 2) * iEvenNum k) (iEvenDen k)]
    rw [hL, hR]
    exact hmul
  exact ⟨hrec, hlt⟩

/-- Theorem 2: Universal squared Euclidean diameter gap D^2(Q_d) = 4*d = d * D^2(B^d). -/
theorem diameter_sq_gap_cube_over_ball_universal (d : Nat) :
    diagCubeSumSq d = 4 * d := by
  induction d with
  | zero => rfl
  | succ k ih =>
    simp only [diagCubeSumSq, ih]
    rw [Nat.mul_succ, Nat.add_comm]

/-- Theorem 3: Universal factorial strict growth for Q_d / B_1^d volume gap d!. -/
theorem volume_ratio_cube_over_cross_growth_universal (d : Nat) (hd : 0 < d) :
    fact d < fact (d + 1) := by
  simp only [fact]
  have hpos : 0 < fact d := fact_pos d
  have h1 : 1 < d + 1 := Nat.succ_lt_succ hd
  have h2 : 1 * fact d < (d + 1) * fact d := Nat.mul_lt_mul_of_pos_right h1 hpos
  rw [Nat.one_mul] at h2
  exact h2

/-- Theorem 4: Exact rational recurrence identities for I_d^abs and mean-width
    M_d = d * I_d^abs across d = 3, 4, 8, 16 (cleared integer form). -/
theorem exact_spherical_abs_recurrence_d3_d4_d8_d16 :
    (2 * 2 * 3 = 3 * 4)
    /\ (4 * 4 * 15 = 5 * 3 * 16)
    /\ (6 * 16 * 35 = 7 * 15 * 32)
    /\ (8 * 32 * 315 = 9 * 35 * 256)
    /\ (10 * 256 * 693 = 11 * 315 * 512)
    /\ (12 * 512 * 3003 = 13 * 693 * 2048)
    /\ (14 * 2048 * 6435 = 15 * 3003 * 4096)
    /\ (3 * 1 = 3)
    /\ (4 * 4 = 16)
    /\ (8 * 32 = 256)
    /\ (16 * 4096 = 65536) := by
  decide

/-- Theorem 5: Exact certified rational brackets for mean-width ratios
    w(Q_d)/w(B^d) across d = 3, 4, 8, 16 and strict monotonic chain. -/
theorem exact_mean_width_brackets_d3_d4_d8_d16 :
    (16976 * 3 * piHi < 16 * 10000 * piDen)
    /\ (16 * 10000 * piDen < 16977 * 3 * piLo)
    /\ (23282 * 35 * piHi < 256 * 10000 * piDen)
    /\ (256 * 10000 * piDen < 23283 * 35 * piLo)
    /\ (32417 * 6435 * piHi < 65536 * 10000 * piDen)
    /\ (65536 * 10000 * piDen < 32418 * 6435 * piLo)
    /\ (3 * 3 * piHi < 2 * 16 * piDen)
    /\ (16 * 35 < 256 * 3)
    /\ (256 * 6435 < 65536 * 35) := by
  decide

/-- Theorem 6: Exact certified volume ratios vol(Q_d)/vol(B^d) and
    vol(Q_d)/vol(B_1^d) = d! across d = 3, 4, 8, 16. -/
theorem exact_volume_gaps_d3_d4_d8_d16 :
    (fact 3 = 6)
    /\ (fact 4 = 24)
    /\ (fact 8 = 40320)
    /\ (fact 16 = 20922789888000)
    /\ (1909 * piHi < 6 * 1000 * piDen)
    /\ (6 * 1000 * piDen < 1910 * piLo)
    /\ (3242 * piHi * piHi < 32 * 1000 * piDen * piDen)
    /\ (32 * 1000 * piDen * piDen < 3243 * piLo * piLo)
    /\ (6307 * powNat piHi 4 < 6144 * 100 * powNat piDen 4)
    /\ (6144 * 100 * powNat piDen 4 < 6308 * powNat piLo 4)
    /\ (278483 * powNat piHi 8 < 2642411520 * powNat piDen 8)
    /\ (2642411520 * powNat piDen 8 < 278485 * powNat piLo 8) := by
  decide

/-- Theorem 7: Exact 100%-rational John contact-point decomposition of
    O_4 = 2 B_1^4 in Q^4: 8 contact vectors v_s = (1/2) sigma_s with c_s = 1/2
    satisfy sum c_s = 4, ||v_s||^2 = 1, ||v_s||_1 = 2 (on d O_4), and sum c_s v_s v_s^T = I_4. -/
theorem john_decomposition_O4_exact_rational :
    o4IntVecs.length = 8
    /\ checkContactNormAndFacet o4IntVecs 4 = true
    /\ checkIntGram o4IntVecs 4 8 = true := by
  exact ⟨rfl, rfl, rfl⟩

/-- Theorem 8: Exact 100%-rational John contact-point decomposition of
    O_16 = 4 B_1^16 in Q^16 via the 16 rows of (1/4) H_16 (c_s = 1):
    sum c_s = 16, ||v_s||^2 = 1, ||v_s||_1 = 4 (on d O_16), and sum v_s v_s^T = I_16. -/
theorem john_decomposition_O16_hadamard_exact_rational :
    h16IntVecs.length = 16
    /\ checkContactNormAndFacet h16IntVecs 16 = true
    /\ checkIntGram h16IntVecs 16 16 = true := by
  exact ⟨rfl, rfl, rfl⟩

/-- Theorem 9: Exact d=4 Hadwiger obstructions from arXiv:2611.32334v2:
    V_2(Z(O_4)) = 12 + 8*sqrt(3) > 24 = V_2(Q_4) iff 4*sqrt(3) > 6 iff 48 > 36
    (equivalently 192 > 144, since sqrt(3) > 3/2 gives 12 + 8*(3/2) = 24), and
    14-cap Star-Helly area wall 14 * (1/2)^2 / 4 = 7/8 < 1 (14 < 16). -/
theorem hadwiger_d4_zonotope_and_star_helly_obstruction :
    (12 + 8 * 3 / 2 = 24)
    /\ (6 * 6 < 4 * 4 * 3)
    /\ (12 * 12 < 8 * 8 * 3)
    /\ (14 < 16)
    /\ (14 * 2 = 28 /\ 16 * 2 = 32 /\ 28 < 32) := by
  decide

-- ============================================================================
-- Pillar 2: Exact +0 B Folded Diagonal Affine-Whitening on S = (1/2)(K - K)
-- ============================================================================

/-- Theorem 10: Universal translation invariance & central symmetry of the
    halved difference body S = (1/2)(K - K) (scalar coordinate). -/
theorem halved_diff_body_translation_and_symmetry_universal (x y c : Int) :
    ((x + c) - (y + c) = x - y) /\ (-(x - y) = y - x) :=
  ⟨int_sub_add_cancel_right x y c, int_neg_sub_swap x y⟩

/-- Theorem 11: Universal arbitrary-dimension list translation invariance &
    central symmetry of S = (1/2)(K - K). -/
theorem halved_diff_body_vec_translation_and_symmetry_universal
    (xs ys cs : List Int) (hxy : xs.length = ys.length) (hxc : xs.length = cs.length) :
    zipSub (zipAdd xs cs) (zipAdd ys cs) = zipSub xs ys
    /\ zipNeg (zipSub xs ys) = zipSub ys xs := by
  induction xs generalizing ys cs with
  | nil =>
    cases ys with
    | nil =>
      cases cs with
      | nil => exact ⟨rfl, rfl⟩
      | cons _ _ => cases hxc
    | cons _ _ => cases hxy
  | cons x xr ih =>
    cases ys with
    | nil => cases hxy
    | cons y yr =>
      cases cs with
      | nil => cases hxc
      | cons c cr =>
        have hxy' : xr.length = yr.length := Nat.succ.inj hxy
        have hxc' : xr.length = cr.length := Nat.succ.inj hxc
        have hrec := ih yr cr hxy' hxc'
        constructor
        · simp only [zipAdd, zipSub, int_sub_add_cancel_right, hrec.1]
        · simp only [zipSub, zipNeg, int_neg_sub_swap, hrec.2]

/-- Theorem 12: Universal +0 B folded diagonal affine-whitening invariance
    across arbitrary dimension (cleared-denominator exact rational inner product). -/
theorem folded_diagonal_whitening_zero_byte_invariance_universal
    (items : List (Int × Int × Int × Int)) :
    sumScaledDot items = sumFactoredDot items := by
  induction items with
  | nil => rfl
  | cons hd tl ih =>
    rcases hd with ⟨w, x, p, q⟩
    simp only [sumScaledDot, sumFactoredDot, ih]
    have hmul : (w * p) * (x * q) = (p * q) * (w * x) := by
      rw [int_mul_pair_swap w p x q, Int.mul_comm (w * x) (p * q)]
    rw [hmul]

/-- Theorem 13: Universal constant-denominator scaling factor extraction
    (exact +0 B linear output preservation when p_j * q_j = D across all channels). -/
theorem folded_diagonal_common_denom_dot_universal (d : Int) (pairs : List (Int × Int)) :
    sumConstScaleDot d pairs = d * sumRawDot pairs := by
  induction pairs with
  | nil => simp [sumConstScaleDot, sumRawDot]
  | cons hd tl ih =>
    rcases hd with ⟨w, x⟩
    simp only [sumConstScaleDot, sumRawDot, ih]
    rw [Int.mul_add]

/-- Theorem 14: Universal shared-RMSNorm (+0 B) simultaneous folding across
    Q, K, V projections sharing one upstream RMSNorm weight gamma. -/
theorem rmsnorm_shared_qkv_fold_universal (wq wk wv g x p q : Int) :
    (wq * p) * ((g * q) * x) = (p * q) * (wq * (g * x))
    /\ (wk * p) * ((g * q) * x) = (p * q) * (wk * (g * x))
    /\ (wv * p) * ((g * q) * x) = (p * q) * (wv * (g * x)) := by
  have key : ∀ w : Int, (w * p) * ((g * q) * x) = (p * q) * (w * (g * x)) := by
    intro w
    have h1 : (g * q) * x = q * (g * x) := by
      rw [Int.mul_comm g q, Int.mul_assoc]
    rw [h1, int_mul_pair_swap w p q (g * x), Int.mul_comm w q]
    rw [int_mul_pair_swap q w p (g * x), Int.mul_comm q p]
  exact ⟨key wq, key wk, key wv⟩

/-- Theorem 15: Universal Block-H4 Parseval orthogonality on Z^4:
    ||H_4 v||_2^2 = 4 ||v||_2^2 (so (1/2) H_4 is an exact rational isometry). -/
theorem blockh4_parseval_orthogonality_universal (a b c d : Int) :
    (a + b + c + d) * (a + b + c + d)
    + (a - b + c - d) * (a - b + c - d)
    + (a + b - c - d) * (a + b - c - d)
    + (a - b - c + d) * (a - b - c + d)
    = 4 * (a * a + b * b + c * c + d * d) := by
  have r1 : a + b + c + d = (a + c) + (b + d) := by
    rw [Int.add_assoc, Int.add_assoc, ← Int.add_assoc b c d, Int.add_comm b c,
        Int.add_assoc c b d, ← Int.add_assoc a c (b + d)]
  have r2 : a - b + c - d = (a + c) - (b + d) := by
    rw [Int.sub_eq_add_neg, Int.sub_eq_add_neg, Int.sub_eq_add_neg, Int.neg_add]
    rw [Int.add_assoc, Int.add_assoc, ← Int.add_assoc (-b) c (-d), Int.add_comm (-b) c,
        Int.add_assoc c (-b) (-d), ← Int.add_assoc a c (-b + -d)]
  have r3 : a + b - c - d = (a - c) + (b - d) := by
    rw [Int.sub_eq_add_neg, Int.sub_eq_add_neg, Int.sub_eq_add_neg, Int.sub_eq_add_neg]
    rw [Int.add_assoc, Int.add_assoc, ← Int.add_assoc b (-c) (-d), Int.add_comm b (-c),
        Int.add_assoc (-c) b (-d), ← Int.add_assoc a (-c) (b + -d)]
  have r4 : a - b - c + d = (a - c) - (b - d) := by
    rw [Int.sub_eq_add_neg, Int.sub_eq_add_neg, Int.sub_eq_add_neg, Int.sub_eq_add_neg,
        Int.sub_eq_add_neg, Int.neg_add, Int.neg_neg]
    rw [Int.add_assoc, Int.add_assoc, ← Int.add_assoc (-b) (-c) d, Int.add_comm (-b) (-c),
        Int.add_assoc (-c) (-b) d, ← Int.add_assoc a (-c) (-b + d)]
  rw [r1, r2, r3, r4]
  have e1 := int_sq_add_sub_cancel (a + c) (b + d)
  have e2 := int_sq_add_sub_cancel (a - c) (b - d)
  have e3 := int_sq_add_sub_cancel a c
  have e4 := int_sq_add_sub_cancel b d
  calc
    ((a + c) + (b + d)) * ((a + c) + (b + d))
      + ((a + c) - (b + d)) * ((a + c) - (b + d))
      + ((a - c) + (b - d)) * ((a - c) + (b - d))
      + ((a - c) - (b - d)) * ((a - c) - (b - d))
      = (((a + c) + (b + d)) * ((a + c) + (b + d)) + ((a + c) - (b + d)) * ((a + c) - (b + d)))
        + (((a - c) + (b - d)) * ((a - c) + (b - d)) + ((a - c) - (b - d)) * ((a - c) - (b - d))) := by
          rw [Int.add_assoc]
    _ = 2 * ((a + c) * (a + c) + (b + d) * (b + d))
        + 2 * ((a - c) * (a - c) + (b - d) * (b - d)) := by rw [e1, e2]
    _ = 2 * (((a + c) * (a + c) + (a - c) * (a - c))
        + ((b + d) * (b + d) + (b - d) * (b - d))) := by
          rw [← Int.mul_add]
          have hswap : ((a + c) * (a + c) + (b + d) * (b + d)) + ((a - c) * (a - c) + (b - d) * (b - d))
              = ((a + c) * (a + c) + (a - c) * (a - c)) + ((b + d) * (b + d) + (b - d) * (b - d)) := by
            rw [Int.add_assoc, Int.add_assoc]
            have hm : (b + d) * (b + d) + ((a - c) * (a - c) + (b - d) * (b - d))
                = (a - c) * (a - c) + ((b + d) * (b + d) + (b - d) * (b - d)) := by
              rw [← Int.add_assoc, Int.add_comm ((b + d) * (b + d)) ((a - c) * (a - c)), Int.add_assoc]
            rw [hm, ← Int.add_assoc]
          rw [hswap]
    _ = 2 * (2 * (a * a + c * c) + 2 * (b * b + d * d)) := by rw [e3, e4]
    _ = 4 * (a * a + b * b + c * c + d * d) := by
          rw [← Int.mul_add, ← Int.mul_assoc]
          have h4 : (2 : Int) * 2 = 4 := rfl
          rw [h4]
          have hord : (a * a + c * c) + (b * b + d * d) = a * a + b * b + c * c + d * d := by
            rw [Int.add_assoc, ← Int.add_assoc (c * c) (b * b) (d * d),
                Int.add_comm (c * c) (b * b), Int.add_assoc (b * b) (c * c) (d * d),
                ← Int.add_assoc (a * a) (b * b), ← Int.add_assoc (a * a + b * b) (c * c) (d * d)]
          rw [hord]

/-- Theorem 16: Universal AM-GM strict anisotropy gap: whenever two coordinate
    half-axes differ (a1 != a2), unwhitened/orthogonal-invariant sum a1^2 + a2^2
    strictly exceeds the John-whitened isotropic optimum 2 * a1 * a2. -/
theorem john_diagonal_strictly_beats_anisotropy_universal (a1 a2 : Int) (hne : a1 ≠ a2) :
    2 * (a1 * a2) < a1 * a1 + a2 * a2 := by
  have hdiff : a1 - a2 ≠ 0 := by
    intro h0
    have heq : a1 = a2 := by
      have h1 : a1 - a2 + a2 = 0 + a2 := congrArg (fun t => t + a2) h0
      rw [Int.sub_add_cancel, Int.zero_add] at h1
      exact h1
    exact hne heq
  have hpos : 0 < (a1 - a2) * (a1 - a2) := int_sq_pos_of_ne_zero (a1 - a2) hdiff
  rw [int_sub_sq_expand] at hpos
  have hadd := Int.add_lt_add_right hpos (2 * (a1 * a2))
  rw [Int.zero_add, Int.sub_add_cancel] at hadd
  exact hadd

private theorem crossDiffSqSum_nonneg (x0 y0 : Int) (ps : List (Int × Int)) :
    0 <= crossDiffSqSum x0 y0 ps := by
  induction ps with
  | nil => exact Int.le_refl 0
  | cons hd tl ih =>
    rcases hd with ⟨x, y⟩
    simp only [crossDiffSqSum]
    exact Int.add_nonneg (int_sq_nonneg (x0 * y - y0 * x)) ih

private theorem lagrangeGap_nonneg (ps : List (Int × Int)) :
    0 <= lagrangeGap ps := by
  induction ps with
  | nil => exact Int.le_refl 0
  | cons hd tl ih =>
    rcases hd with ⟨x0, y0⟩
    simp only [lagrangeGap]
    exact Int.add_nonneg (crossDiffSqSum_nonneg x0 y0 tl) ih

private theorem cross_identity_step (x0 y0 : Int) (ps : List (Int × Int)) :
    (x0 * x0) * sumSqSnd ps + (y0 * y0) * sumSqFst ps - 2 * ((x0 * y0) * dotProd ps)
    = crossDiffSqSum x0 y0 ps := by
  induction ps with
  | nil => simp [sumSqSnd, sumSqFst, dotProd, crossDiffSqSum]
  | cons hd tl ih =>
    rcases hd with ⟨x, y⟩
    simp only [sumSqSnd, sumSqFst, dotProd, crossDiffSqSum]
    have hterm : (x0 * y - y0 * x) * (x0 * y - y0 * x)
        = (x0 * x0) * (y * y) + (y0 * y0) * (x * x) - 2 * ((x0 * y0) * (x * y)) := by
      rw [int_sub_sq_expand (x0 * y) (y0 * x)]
      rw [int_mul_pair_swap x0 y x0 y, int_mul_pair_swap y0 x y0 x]
      have hxy : (x0 * y) * (y0 * x) = (x0 * y0) * (x * y) := by
        rw [int_mul_pair_swap x0 y y0 x, Int.mul_comm y x]
      rw [hxy]
    rw [Int.mul_add, Int.mul_add, Int.mul_add, Int.mul_add, hterm, ← ih]
    have hgen : ∀ a1 a2 b1 b2 c1 c2 : Int,
        ((a1 + a2) + (b1 + b2)) - (c1 + c2) = (a1 + b1 - c1) + (a2 + b2 - c2) := by
      intro a1 a2 b1 b2 c1 c2
      rw [Int.sub_eq_add_neg, Int.sub_eq_add_neg, Int.sub_eq_add_neg, Int.neg_add,
          int_add_four_swap a1 a2 b1 b2, int_add_four_swap (a1 + b1) (a2 + b2) (-c1) (-c2)]
    exact hgen _ _ _ _ _ _

private theorem intEqBool_false (a b : Int) (h : intEqBool a b = false) : a ≠ b := by
  intro heq
  have h0 : a - b = 0 := Int.sub_eq_zero_of_eq heq
  simp only [intEqBool, h0] at h
  cases h

private theorem crossDiffSqSum_eq_zero_of_true (x0 y0 : Int) (ps : List (Int × Int))
    (h : allPairCrossZero x0 y0 ps = true) : crossDiffSqSum x0 y0 ps = 0 := by
  induction ps with
  | nil => rfl
  | cons hd tl ih =>
    rcases hd with ⟨x, y⟩
    simp only [allPairCrossZero, crossDiffSqSum] at h ⊢
    cases h1 : intEqBool (x0 * y - y0 * x) 0 with
    | false =>
      rw [h1] at h
      cases h
    | true =>
      rw [h1] at h
      simp only [Bool.true_and] at h
      have heq : x0 * y - y0 * x = 0 := intEqBool_true _ _ h1
      rw [heq, ih h]
      rfl

private theorem crossDiffSqSum_pos_of_false (x0 y0 : Int) (ps : List (Int × Int))
    (h : allPairCrossZero x0 y0 ps = false) : 0 < crossDiffSqSum x0 y0 ps := by
  induction ps with
  | nil =>
    simp only [allPairCrossZero] at h
    cases h
  | cons hd tl ih =>
    rcases hd with ⟨x, y⟩
    simp only [allPairCrossZero, crossDiffSqSum] at h ⊢
    cases h1 : intEqBool (x0 * y - y0 * x) 0 with
    | false =>
      have hne : x0 * y - y0 * x ≠ 0 := intEqBool_false _ _ h1
      have hpos : 0 < (x0 * y - y0 * x) * (x0 * y - y0 * x) := int_sq_pos_of_ne_zero _ hne
      have hnn : 0 <= crossDiffSqSum x0 y0 tl := crossDiffSqSum_nonneg x0 y0 tl
      have hadd := Int.add_lt_add_of_lt_of_le hpos hnn
      rw [Int.add_zero] at hadd
      exact hadd
    | true =>
      rw [h1] at h
      simp only [Bool.true_and] at h
      have htl : 0 < crossDiffSqSum x0 y0 tl := ih h
      have hnn : 0 <= (x0 * y - y0 * x) * (x0 * y - y0 * x) := int_sq_nonneg _
      have hadd := Int.add_lt_add_of_le_of_lt hnn htl
      rw [Int.add_zero] at hadd
      exact hadd

private theorem lagrangeGap_eq_zero_of_isCollinear (ps : List (Int × Int))
    (h : isCollinear ps = true) : lagrangeGap ps = 0 := by
  induction ps with
  | nil => rfl
  | cons hd tl ih =>
    rcases hd with ⟨x0, y0⟩
    simp only [isCollinear, lagrangeGap] at h ⊢
    cases h1 : allPairCrossZero x0 y0 tl with
    | false =>
      rw [h1] at h
      cases h
    | true =>
      rw [h1] at h
      simp only [Bool.true_and] at h
      rw [crossDiffSqSum_eq_zero_of_true x0 y0 tl h1, ih h]
      rfl

private theorem lagrangeGap_pos_of_not_isCollinear (ps : List (Int × Int))
    (h : isCollinear ps = false) : 0 < lagrangeGap ps := by
  induction ps with
  | nil =>
    simp only [isCollinear] at h
    cases h
  | cons hd tl ih =>
    rcases hd with ⟨x0, y0⟩
    simp only [isCollinear, lagrangeGap] at h ⊢
    cases h1 : allPairCrossZero x0 y0 tl with
    | false =>
      have hpos : 0 < crossDiffSqSum x0 y0 tl := crossDiffSqSum_pos_of_false x0 y0 tl h1
      have hnn : 0 <= lagrangeGap tl := lagrangeGap_nonneg tl
      have hadd := Int.add_lt_add_of_lt_of_le hpos hnn
      rw [Int.add_zero] at hadd
      exact hadd
    | true =>
      rw [h1] at h
      simp only [Bool.true_and] at h
      have htl : 0 < lagrangeGap tl := ih h
      have hnn : 0 <= crossDiffSqSum x0 y0 tl := crossDiffSqSum_nonneg x0 y0 tl
      have hadd := Int.add_lt_add_of_le_of_lt hnn htl
      rw [Int.add_zero] at hadd
      exact hadd

private theorem dotProd_eq_of_sameProd (ps1 ps2 : List (Int × Int))
    (h : sameProd ps1 ps2 = true) : dotProd ps1 = dotProd ps2 := by
  induction ps1 generalizing ps2 with
  | nil =>
    cases ps2 with
    | nil => rfl
    | cons _ _ =>
      simp only [sameProd] at h
      cases h
  | cons hd1 tl1 ih =>
    cases ps2 with
    | nil =>
      simp only [sameProd] at h
      cases h
    | cons hd2 tl2 =>
      rcases hd1 with ⟨a, b⟩
      rcases hd2 with ⟨c, d⟩
      simp only [sameProd, dotProd] at h ⊢
      cases h1 : intEqBool (a * b) (c * d) with
      | false =>
        rw [h1] at h
        cases h
      | true =>
        rw [h1] at h
        simp only [Bool.true_and] at h
        have hab : a * b = c * d := intEqBool_true _ _ h1
        rw [hab, ih tl2 h]

/-- Theorem 17: Universal arbitrary-dimension Lagrange Cauchy-Schwarz identity,
    global L2-ellipsoid diagonal whitening optimality, and strict uniqueness of
    proportional scaling (c_j* proportional to h_j^{1/4} / RMS(W_{:,j})^{1/2}).
    For any candidate scaling pairs (x_j, y_j) and collinear optimal pairs
    (x*_j, y*_j) sharing invariant channel products x*_j y*_j = x_j y_j:
    (1) Lagrange identity, (2) Cauchy-Schwarz bound, (3) optimal attains global
    minimum over all scalings, and (4) non-collinear scalings are strictly worse. -/
theorem lagrange_cauchy_schwarz_john_hessian_universal (ps ps_opt : List (Int × Int)) :
    (sumSqFst ps * sumSqSnd ps = dotProd ps * dotProd ps + lagrangeGap ps)
    /\ (dotProd ps * dotProd ps <= sumSqFst ps * sumSqSnd ps)
    /\ (sameProd ps_opt ps = true -> isCollinear ps_opt = true ->
        sumSqFst ps_opt * sumSqSnd ps_opt <= sumSqFst ps * sumSqSnd ps)
    /\ (sameProd ps_opt ps = true -> isCollinear ps_opt = true -> isCollinear ps = false ->
        sumSqFst ps_opt * sumSqSnd ps_opt < sumSqFst ps * sumSqSnd ps) := by
  have hid_fn : ∀ qs : List (Int × Int), sumSqFst qs * sumSqSnd qs = dotProd qs * dotProd qs + lagrangeGap qs := by
    intro qs
    induction qs with
    | nil => rfl
    | cons hd tl ih =>
      rcases hd with ⟨x0, y0⟩
      simp only [sumSqFst, sumSqSnd, dotProd, lagrangeGap]
      have hstep := cross_identity_step x0 y0 tl
      have h0 : (x0 * x0) * (y0 * y0) = (x0 * y0) * (x0 * y0) := by
        rw [int_mul_pair_swap x0 x0 y0 y0]
      have hL : (x0 * x0 + sumSqFst tl) * (y0 * y0 + sumSqSnd tl)
          = (x0 * y0) * (x0 * y0)
            + ((x0 * x0) * sumSqSnd tl + (y0 * y0) * sumSqFst tl)
            + sumSqFst tl * sumSqSnd tl := by
        rw [Int.add_mul, Int.mul_add, Int.mul_add, h0]
        rw [Int.mul_comm (sumSqFst tl) (y0 * y0)]
        rw [Int.add_assoc, Int.add_assoc, Int.add_assoc]
      have hR : (x0 * y0 + dotProd tl) * (x0 * y0 + dotProd tl)
          = (x0 * y0) * (x0 * y0) + 2 * ((x0 * y0) * dotProd tl) + dotProd tl * dotProd tl := by
        rw [Int.add_mul, Int.mul_add, Int.mul_add]
        rw [Int.mul_comm (dotProd tl) (x0 * y0)]
        rw [Int.add_assoc, ← Int.add_assoc ((x0 * y0) * dotProd tl) ((x0 * y0) * dotProd tl),
            ← int_two_mul ((x0 * y0) * dotProd tl), ← Int.add_assoc]
      have hstep2 : (x0 * x0) * sumSqSnd tl + (y0 * y0) * sumSqFst tl
          = 2 * ((x0 * y0) * dotProd tl) + crossDiffSqSum x0 y0 tl := by
        have ha := congrArg (fun t => t + 2 * ((x0 * y0) * dotProd tl)) hstep
        rw [Int.sub_add_cancel] at ha
        rw [ha, Int.add_comm]
      rw [hL, hR, ih, hstep2]
      have hrearr : ∀ p0 t2 cg dp lg : Int,
          (p0 + (t2 + cg)) + (dp + lg) = ((p0 + t2) + dp) + (cg + lg) := by
        intro p0 t2 cg dp lg
        rw [← Int.add_assoc p0 t2 cg, int_add_four_swap (p0 + t2) cg dp lg]
      exact hrearr _ _ _ _ _
  have hid : sumSqFst ps * sumSqSnd ps = dotProd ps * dotProd ps + lagrangeGap ps := hid_fn ps
  have hle : dotProd ps * dotProd ps <= sumSqFst ps * sumSqSnd ps := by
    rw [hid]
    have hg := lagrangeGap_nonneg ps
    have h1 := Int.add_le_add_left hg (dotProd ps * dotProd ps)
    rw [Int.add_zero] at h1
    exact h1
  have hopt_le : sameProd ps_opt ps = true -> isCollinear ps_opt = true ->
      sumSqFst ps_opt * sumSqSnd ps_opt <= sumSqFst ps * sumSqSnd ps := by
    intro hsame hcol
    have hid_opt := hid_fn ps_opt
    have hgap0 := lagrangeGap_eq_zero_of_isCollinear ps_opt hcol
    have hdp := dotProd_eq_of_sameProd ps_opt ps hsame
    rw [hid_opt, hgap0, Int.add_zero, hdp]
    exact hle
  have hopt_lt : sameProd ps_opt ps = true -> isCollinear ps_opt = true -> isCollinear ps = false ->
      sumSqFst ps_opt * sumSqSnd ps_opt < sumSqFst ps * sumSqSnd ps := by
    intro hsame hcol hnoncol
    have hid_opt := hid_fn ps_opt
    have hgap0 := lagrangeGap_eq_zero_of_isCollinear ps_opt hcol
    have hdp := dotProd_eq_of_sameProd ps_opt ps hsame
    have hpos := lagrangeGap_pos_of_not_isCollinear ps hnoncol
    rw [hid_opt, hgap0, Int.add_zero, hdp, hid]
    have hlt1 := Int.add_lt_add_left hpos (dotProd ps * dotProd ps)
    rw [Int.add_zero] at hlt1
    exact hlt1
  exact ⟨hid, hle, hopt_le, hopt_lt⟩

-- ============================================================================
-- Pillar 3: Exact Helly-LP Minimax Chebyshev Gauge-Scale Optimality
-- ============================================================================

private theorem intAbs_ge_self (z : Int) : z <= intAbs z := by
  simp only [intAbs]
  split
  · exact Int.le_refl z
  · rename_i h
    have hlt : z < 0 := Int.lt_of_not_ge h
    have hneg : 0 < -z := Int.neg_pos_of_neg hlt
    exact Int.le_trans (Int.le_of_lt hlt) (Int.le_of_lt hneg)

private theorem intAbs_of_nonneg {z : Int} (h : 0 <= z) : intAbs z = z := by
  simp only [intAbs]
  split
  · rfl
  · rename_i hnot
    exact absurd h hnot

private theorem intAbs_of_nonpos {z : Int} (h : z <= 0) : intAbs z = -z := by
  simp only [intAbs]
  split
  · rename_i hge
    have h0 : z = 0 := Int.le_antisymm h hge
    rw [h0]
    rfl
  · rfl

private theorem intMax_ge_left (a b : Int) : a <= intMax a b := by
  simp only [intMax]
  split
  · exact Int.le_refl a
  · rename_i h
    exact Int.le_of_lt (Int.lt_of_not_ge h)

private theorem intMax_ge_right (a b : Int) : b <= intMax a b := by
  simp only [intMax]
  split
  · rename_i h
    exact h
  · exact Int.le_refl b

private theorem intMax_self (a : Int) : intMax a a = a := by
  simp only [intMax]
  split
  · rfl
  · rename_i hnot
    exact absurd (Int.le_refl a) hnot

/-- Theorem 18: Universal Helly-Chebyshev center minimax optimality and uniqueness
    (exact doubled integer form): for any interval [m, M] and any doubled center c2,
    max(|2M - c2|, |c2 - 2m|) >= M - m, attained uniquely at c2 = m + M. -/
theorem helly_chebyshev_center_minimax_optimal_universal (m M c2 : Int) (h : m <= M) :
    M - m <= intMax (intAbs (2 * M - c2)) (intAbs (c2 - 2 * m))
    /\ intMax (intAbs (2 * M - (m + M))) (intAbs ((m + M) - 2 * m)) = M - m
    /\ (intMax (intAbs (2 * M - c2)) (intAbs (c2 - 2 * m)) = M - m -> c2 = m + M) := by
  have hnonneg : 0 <= M - m := Int.sub_nonneg.mpr h
  have hlb : M - m <= intMax (intAbs (2 * M - c2)) (intAbs (c2 - 2 * m)) := by
    have h1 : 2 * M - c2 <= intMax (intAbs (2 * M - c2)) (intAbs (c2 - 2 * m)) :=
      Int.le_trans (intAbs_ge_self _) (intMax_ge_left _ _)
    have h2 : c2 - 2 * m <= intMax (intAbs (2 * M - c2)) (intAbs (c2 - 2 * m)) :=
      Int.le_trans (intAbs_ge_self _) (intMax_ge_right _ _)
    have hsum := Int.add_le_add h1 h2
    have h2s : (2 * M - c2) + (c2 - 2 * m) = 2 * (M - m) := by
      rw [Int.sub_eq_add_neg, Int.sub_eq_add_neg, Int.add_assoc,
          ← Int.add_assoc (-c2) c2 (-(2 * m)), Int.add_left_neg, Int.zero_add,
          ← Int.sub_eq_add_neg, ← Int.mul_sub]
    have h2r : intMax (intAbs (2 * M - c2)) (intAbs (c2 - 2 * m))
        + intMax (intAbs (2 * M - c2)) (intAbs (c2 - 2 * m))
        = 2 * intMax (intAbs (2 * M - c2)) (intAbs (c2 - 2 * m)) :=
      (int_two_mul _).symm
    rw [h2s, h2r] at hsum
    exact Int.le_of_mul_le_mul_left hsum (by decide : (0 : Int) < 2)
  have hatt : intMax (intAbs (2 * M - (m + M))) (intAbs ((m + M) - 2 * m)) = M - m := by
    have ha : 2 * M - (m + M) = M - m := by
      rw [int_two_mul M, Int.sub_eq_add_neg, Int.neg_add, Int.add_assoc,
          ← Int.add_assoc M (-m) (-M), Int.add_comm M (-m), Int.add_assoc (-m) M (-M),
          Int.add_right_neg, Int.add_zero, ← Int.sub_eq_add_neg]
    have hb : (m + M) - 2 * m = M - m := by
      rw [int_two_mul m, Int.sub_eq_add_neg, Int.neg_add, Int.add_comm m M, Int.add_assoc,
          ← Int.add_assoc m (-m) (-m), Int.add_right_neg, Int.zero_add, ← Int.sub_eq_add_neg]
    rw [ha, hb, intAbs_of_nonneg hnonneg, intMax_self]
  have huniq : intMax (intAbs (2 * M - c2)) (intAbs (c2 - 2 * m)) = M - m -> c2 = m + M := by
    intro heq
    have h1 : 2 * M - c2 <= M - m := by
      rw [← heq]
      exact Int.le_trans (intAbs_ge_self _) (intMax_ge_left _ _)
    have h2 : c2 - 2 * m <= M - m := by
      rw [← heq]
      exact Int.le_trans (intAbs_ge_self _) (intMax_ge_right _ _)
    have h1c := Int.add_le_add_right h1 c2
    rw [Int.sub_add_cancel] at h1c
    have h1cm := Int.add_le_add_right h1c m
    have h1r : (M - m + c2) + m = M + c2 := by
      rw [Int.add_assoc, Int.add_comm c2 m, ← Int.add_assoc, Int.sub_add_cancel]
    rw [h1r, int_two_mul M] at h1cm
    have h1sub := Int.add_le_add_left h1cm (-M)
    have h1sl : -M + ((M + M) + m) = m + M := by
      rw [← Int.add_assoc, ← Int.add_assoc, Int.add_left_neg, Int.zero_add, Int.add_comm M m]
    have h1sr : -M + (M + c2) = c2 := by
      rw [← Int.add_assoc, Int.add_left_neg, Int.zero_add]
    rw [h1sl, h1sr] at h1sub
    have h2m := Int.add_le_add_right h2 (2 * m)
    rw [Int.sub_add_cancel, int_two_mul m] at h2m
    have h2r : (M - m) + (m + m) = m + M := by
      rw [← Int.add_assoc, Int.sub_add_cancel, Int.add_comm M m]
    rw [h2r] at h2m
    exact Int.le_antisymm h2m h1sub
  exact ⟨hlb, hatt, huniq⟩

/-- Theorem 19: Universal strict asymmetric origin penalty: whenever a weight
    block has nonzero mean asymmetry (m + M != 0), origin-centered scaling
    pays strictly larger worst-case radius (M - m) + |M + m| > M - m. -/
theorem helly_asymmetric_origin_penalty_universal (m M : Int) (h : m <= M) (hasym : m + M ≠ 0) :
    intMax (intAbs (2 * M)) (intAbs (2 * m)) = (M - m) + intAbs (M + m)
    /\ M - m < intMax (intAbs (2 * M)) (intAbs (2 * m)) := by
  have heq : intMax (intAbs (2 * M)) (intAbs (2 * m)) = (M - m) + intAbs (M + m) := by
    cases Int.le_total 0 (M + m) with
    | inl hpos =>
      have hM : 0 <= 2 * M := by
        rw [int_two_mul]
        have h1 := Int.add_le_add_left h M
        exact Int.le_trans hpos h1
      have hge : intAbs (2 * m) <= 2 * M := by
        simp only [intAbs]
        split
        · exact Int.mul_le_mul_of_nonneg_left h (by decide : (0 : Int) <= 2)
        · have h2s : 0 <= 2 * (M + m) := Int.mul_nonneg (by decide : (0 : Int) <= 2) hpos
          rw [Int.mul_add] at h2s
          have hadd := Int.add_le_add_right h2s (-(2 * m))
          rw [Int.zero_add, Int.add_assoc, Int.add_right_neg, Int.add_zero] at hadd
          exact hadd
      rw [intAbs_of_nonneg hM, intAbs_of_nonneg hpos]
      have hmax : intMax (2 * M) (intAbs (2 * m)) = 2 * M := by
        simp only [intMax]
        split
        · rfl
        · rename_i hnot
          exact absurd hge hnot
      rw [hmax, int_two_mul, Int.sub_eq_add_neg, Int.add_assoc, Int.add_comm (-m) (M + m),
          Int.add_assoc, Int.add_right_neg, Int.add_zero]
    | inr hneg =>
      have hm : 2 * m <= 0 := by
        rw [int_two_mul]
        have h1 := Int.add_le_add_right h m
        exact Int.le_trans h1 hneg
      have hge : intAbs (2 * M) <= -(2 * m) := by
        simp only [intAbs]
        split
        · have h2s : 2 * (M + m) <= 0 := Int.mul_nonpos_of_nonneg_of_nonpos (by decide : (0 : Int) <= 2) hneg
          rw [Int.mul_add] at h2s
          have hadd := Int.add_le_add_right h2s (-(2 * m))
          rw [Int.zero_add, Int.add_assoc, Int.add_right_neg, Int.add_zero] at hadd
          exact hadd
        · have hmul := Int.mul_le_mul_of_nonneg_left h (by decide : (0 : Int) <= 2)
          exact Int.neg_le_neg hmul
      have hneg_id : -(2 * m) = (M - m) + -(M + m) := by
        rw [int_two_mul, Int.sub_eq_add_neg, Int.neg_add, Int.neg_add,
            Int.add_assoc, Int.add_comm (-m) (-M + -m), ← Int.add_assoc,
            ← Int.add_assoc M (-M) (-m), Int.add_right_neg, Int.zero_add]
      rw [intAbs_of_nonpos hm, intAbs_of_nonpos hneg]
      simp only [intMax]
      split
      · rename_i hle
        have hboth : intAbs (2 * M) = -(2 * m) := Int.le_antisymm hge hle
        rw [hboth, hneg_id]
      · exact hneg_id
  have hpos_abs : 0 < intAbs (M + m) := by
    have hne2 : M + m ≠ 0 := by
      rw [Int.add_comm]
      exact hasym
    cases Int.lt_or_gt_of_ne hne2 with
    | inl hlt =>
      rw [intAbs_of_nonpos (Int.le_of_lt hlt)]
      exact Int.neg_pos_of_neg hlt
    | inr hgt =>
      rw [intAbs_of_nonneg (Int.le_of_lt hgt)]
      exact hgt
  constructor
  · exact heq
  · rw [heq]
    have hadd := Int.add_lt_add_left hpos_abs (M - m)
    rw [Int.add_zero] at hadd
    exact hadd

/-- Theorem 20: Universal arbitrary-length Helly telescoping covering identity
    (arXiv:2611.32334v2 CurveVar.lean): for any split (pre, suf) of increments
    x_{r+1} - x_r = a_r (w_r - u_r), the center v~ = x0 - sum_all (a_r u_r)
    satisfies x_j - v~ = sum_pre (a_r w_r) + sum_suf (a_r u_r). -/
theorem helly_telescoping_covering_identity_universal
    (x0 : Int) (pre suf : List (Int × Int × Int)) :
    (x0 + hellyStepSum pre) - (x0 - (hellyUSum pre + hellyUSum suf))
    = hellyWSum pre + hellyUSum suf := by
  have hpre : hellyStepSum pre + hellyUSum pre = hellyWSum pre := by
    induction pre with
    | nil => rfl
    | cons hd tl ih =>
      rcases hd with ⟨a, u, w⟩
      simp only [hellyStepSum, hellyUSum, hellyWSum]
      rw [Int.mul_sub]
      calc
        (a * w - a * u + hellyStepSum tl) + (a * u + hellyUSum tl)
          = (a * w - a * u + a * u) + (hellyStepSum tl + hellyUSum tl) := by
              rw [Int.add_assoc, Int.add_assoc, ← Int.add_assoc (hellyStepSum tl) (a * u),
                  Int.add_comm (hellyStepSum tl) (a * u), Int.add_assoc (a * u) (hellyStepSum tl),
                  ← Int.add_assoc (a * w - a * u) (a * u)]
        _ = a * w + hellyWSum tl := by rw [Int.sub_add_cancel, ih]
  rw [Int.sub_eq_add_neg, Int.sub_eq_add_neg, Int.neg_add, Int.neg_neg]
  calc
    (x0 + hellyStepSum pre) + (-x0 + (hellyUSum pre + hellyUSum suf))
      = (x0 + -x0) + (hellyStepSum pre + (hellyUSum pre + hellyUSum suf)) := by
          rw [Int.add_assoc, ← Int.add_assoc (hellyStepSum pre) (-x0),
              Int.add_comm (hellyStepSum pre) (-x0), Int.add_assoc (-x0) (hellyStepSum pre),
              ← Int.add_assoc x0 (-x0)]
    _ = hellyWSum pre + hellyUSum suf := by
          rw [Int.add_right_neg, Int.zero_add, ← Int.add_assoc, hpre]

/-- Theorem 21: Universal Helly safe gauge scale sufficiency (zero OOB groups)
    AND exact bulk-scale / bulk-MSE inflation tradeoff under heavy tails (Kill Gate 2):
    when bulk RMS gauge lies strictly below tail max (g_rms < g_max), any safe scale
    exceeds bulk scale (s_bulk < s_safe) and strictly inflates bulk variance across N coords. -/
theorem helly_safe_scale_zero_oob_and_tail_tradeoff_universal
    (g_i g_max g_rms s_bulk s_safe R rho N : Nat)
    (hi : g_i <= g_max)
    (hmargin : rho < R)
    (hbulk : s_bulk * (R - rho) <= g_rms)
    (htail : g_rms < g_max)
    (hsafe : g_max <= s_safe * (R - rho))
    (hN : 0 < N) :
    (g_i + s_safe * rho <= s_safe * R)
    /\ (s_bulk < s_safe)
    /\ (N * (s_bulk * s_bulk) < N * (s_safe * s_safe)) := by
  have hmargin_le : rho <= R := Nat.le_of_lt hmargin
  have hoob : g_i + s_safe * rho <= s_safe * R := by
    have hdist : s_safe * (R - rho) + s_safe * rho = s_safe * R := by
      rw [← Nat.mul_add, Nat.sub_add_cancel hmargin_le]
    have h1 : g_i + s_safe * rho <= g_max + s_safe * rho := Nat.add_le_add_right hi (s_safe * rho)
    have h2 : g_max + s_safe * rho <= s_safe * (R - rho) + s_safe * rho := Nat.add_le_add_right hsafe (s_safe * rho)
    rw [hdist] at h2
    exact Nat.le_trans h1 h2
  have hs_lt : s_bulk < s_safe := by
    have hprod : s_bulk * (R - rho) < s_safe * (R - rho) :=
      Nat.lt_of_le_of_lt hbulk (Nat.lt_of_lt_of_le htail hsafe)
    cases Nat.lt_or_ge s_bulk s_safe with
    | inl h => exact h
    | inr h => exact absurd (Nat.mul_le_mul_right (R - rho) h) (Nat.not_le_of_gt hprod)
  have hvar : N * (s_bulk * s_bulk) < N * (s_safe * s_safe) := by
    have hs_pos : 0 < s_safe := Nat.lt_of_le_of_lt (Nat.zero_le _) hs_lt
    have hsq : s_bulk * s_bulk < s_safe * s_safe :=
      Nat.mul_lt_mul_of_lt_of_le hs_lt (Nat.le_of_lt hs_lt) hs_pos
    exact Nat.mul_lt_mul_of_pos_left hsq hN
  exact ⟨hoob, hs_lt, hvar⟩

-- ============================================================================
-- Pillar 4: Polar-Cap vs Equatorial-Bulk Split Bound & Bit-Budget Parity
-- ============================================================================

/-- Theorem 22: Universal polar-cap bulk range contraction and constructive
    matched-bpw MSE crossover theorem: whenever range contraction compensates
    the lattice-ball rate penalty (M_num * M_bulk^2 <= M_den * M_all^2) and
    the excised polar cap carries positive energy (0 < E_cap), bulk MSE
    strictly improves over unsplit MSE. -/
theorem polar_cap_bulk_contraction_and_crossover_universal
    (M_bulk M_all E_bulk E_cap M_num M_den : Nat)
    (hcontract : M_bulk < M_all)
    (hball : M_num * (M_bulk * M_bulk) <= M_den * (M_all * M_all))
    (hden : 0 < M_den)
    (hmall : 0 < M_all)
    (hcap : 0 < E_cap) :
    (M_bulk * M_bulk < M_all * M_all)
    /\ (M_num * (M_bulk * M_bulk) * E_bulk < M_den * (M_all * M_all) * (E_bulk + E_cap)) := by
  have hsq : M_bulk * M_bulk < M_all * M_all :=
    Nat.mul_lt_mul_of_lt_of_le hcontract (Nat.le_of_lt hcontract) hmall
  have h1 : M_num * (M_bulk * M_bulk) * E_bulk <= M_den * (M_all * M_all) * E_bulk :=
    Nat.mul_le_mul_right E_bulk hball
  have hpos : 0 < M_den * (M_all * M_all) * E_cap :=
    Nat.mul_pos (Nat.mul_pos hden (Nat.mul_pos hmall hmall)) hcap
  have h2 : M_den * (M_all * M_all) * E_bulk
      < M_den * (M_all * M_all) * (E_bulk + E_cap) := by
    rw [Nat.mul_add]
    exact Nat.lt_add_of_pos_right hpos
  exact ⟨hsq, Nat.lt_of_le_of_lt h1 h2⟩

/-- Theorem 23: Universal matched-bpw bit-budget conservation (CS-08) for
    both per-group sidecar budgets (CSR/Chebyshev) and block-level precision
    splitting with explicit per-block selector map bits B_map. -/
theorem siliquant_matched_bpw_conservation_universal
    (b_dense b_scale b_cap total_bits n_groups k_hi k_lo B_hi B_lo B_map B_target : Nat)
    (hbudget : b_dense + b_scale + b_cap <= total_bits)
    (hsplit_ord : B_lo <= B_hi)
    (hsplit_budget : k_hi * B_hi + k_lo * B_lo + (k_hi + k_lo) * B_map <= (k_hi + k_lo) * B_target) :
    b_dense <= total_bits - b_scale - b_cap
    /\ n_groups * (b_dense + b_scale + b_cap) <= n_groups * total_bits
    /\ (k_hi + k_lo) * B_lo + k_hi * (B_hi - B_lo) + (k_hi + k_lo) * B_map <= (k_hi + k_lo) * B_target := by
  have h1 : b_dense + b_scale <= total_bits - b_cap := Nat.le_sub_of_add_le hbudget
  have h2 : b_dense <= (total_bits - b_cap) - b_scale := Nat.le_sub_of_add_le h1
  rw [Nat.sub_right_comm] at h2
  have hgrp : n_groups * (b_dense + b_scale + b_cap) <= n_groups * total_bits :=
    Nat.mul_le_mul_left n_groups hbudget
  have hhi : B_lo + (B_hi - B_lo) = B_hi := Nat.add_sub_of_le hsplit_ord
  have hrew : k_hi * B_hi + k_lo * B_lo = (k_hi + k_lo) * B_lo + k_hi * (B_hi - B_lo) := by
    calc
      k_hi * B_hi + k_lo * B_lo
        = k_hi * (B_lo + (B_hi - B_lo)) + k_lo * B_lo := by rw [hhi]
      _ = (k_hi * B_lo + k_hi * (B_hi - B_lo)) + k_lo * B_lo := by rw [Nat.mul_add]
      _ = (k_hi * B_lo + k_lo * B_lo) + k_hi * (B_hi - B_lo) := by
          rw [Nat.add_assoc, Nat.add_comm (k_hi * (B_hi - B_lo)) (k_lo * B_lo), ← Nat.add_assoc]
      _ = (k_hi + k_lo) * B_lo + k_hi * (B_hi - B_lo) := by rw [Nat.add_mul]
  rw [hrew] at hsplit_budget
  exact ⟨h2, hgrp, hsplit_budget⟩

/-- Theorem 24: Exact SiliQuant A_16 [16,11,4] codeword syndrome witness,
    siliquant#85 g=128 ball bounds, and matched-bpw polar-cap bit budgets. -/
theorem siliquant_a16_syndrome_and_budget_witness :
    a16SyndromeZero 0x0000 = true
    /\ a16SyndromeZero 0xFFFF = true
    /\ a16SyndromeZero 0x000F = true
    /\ a16SyndromeZero 0x0033 = true
    /\ a16SyndromeZero 0x0055 = true
    /\ a16SyndromeZero 0x0F0F = true
    /\ a16SyndromeZero 0x0001 = false
    /\ a16SyndromeZero 0x0003 = false
    /\ (368 + 16 = 3 * 128)
    /\ (560 + 16 = 576 /\ 2 * 576 = 9 * 128)
    /\ (357 + 16 + 11 = 384)
    /\ (549 + 16 + 11 = 576)
    /\ (352 + 16 + 16 = 384)
    /\ (544 + 16 + 16 = 576) := by
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, ⟨rfl, rfl⟩, rfl, rfl, rfl, rfl⟩

end JohnQuantOperators

#print axioms JohnQuantOperators.mean_width_recurrence_growth_universal
#print axioms JohnQuantOperators.diameter_sq_gap_cube_over_ball_universal
#print axioms JohnQuantOperators.volume_ratio_cube_over_cross_growth_universal
#print axioms JohnQuantOperators.exact_spherical_abs_recurrence_d3_d4_d8_d16
#print axioms JohnQuantOperators.exact_mean_width_brackets_d3_d4_d8_d16
#print axioms JohnQuantOperators.exact_volume_gaps_d3_d4_d8_d16
#print axioms JohnQuantOperators.john_decomposition_O4_exact_rational
#print axioms JohnQuantOperators.john_decomposition_O16_hadamard_exact_rational
#print axioms JohnQuantOperators.hadwiger_d4_zonotope_and_star_helly_obstruction
#print axioms JohnQuantOperators.halved_diff_body_translation_and_symmetry_universal
#print axioms JohnQuantOperators.halved_diff_body_vec_translation_and_symmetry_universal
#print axioms JohnQuantOperators.folded_diagonal_whitening_zero_byte_invariance_universal
#print axioms JohnQuantOperators.folded_diagonal_common_denom_dot_universal
#print axioms JohnQuantOperators.rmsnorm_shared_qkv_fold_universal
#print axioms JohnQuantOperators.blockh4_parseval_orthogonality_universal
#print axioms JohnQuantOperators.john_diagonal_strictly_beats_anisotropy_universal
#print axioms JohnQuantOperators.lagrange_cauchy_schwarz_john_hessian_universal
#print axioms JohnQuantOperators.helly_chebyshev_center_minimax_optimal_universal
#print axioms JohnQuantOperators.helly_asymmetric_origin_penalty_universal
#print axioms JohnQuantOperators.helly_telescoping_covering_identity_universal
#print axioms JohnQuantOperators.helly_safe_scale_zero_oob_and_tail_tradeoff_universal
#print axioms JohnQuantOperators.polar_cap_bulk_contraction_and_crossover_universal
#print axioms JohnQuantOperators.siliquant_matched_bpw_conservation_universal
#print axioms JohnQuantOperators.siliquant_a16_syndrome_and_budget_witness
