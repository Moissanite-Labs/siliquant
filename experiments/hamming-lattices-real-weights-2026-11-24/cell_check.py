"""Independent check of the survey's claim: Construction A lattices of extended Hamming codes.

L(C) = C + 2Z^n for a binary code C of length n, dimension k. Volume of a cell 2^(n-k).
Nearest point by brute force over all 2^k codewords (exact). G = E|x-Q(x)|^2 / (n V^(2/n)),
x uniform on [0,2)^n, which is a fundamental region of 2Z^n, a sublattice, so it samples the cell uniformly mod L.
Also: share of small inputs (|x_i| < step/4 at unit density) decoded to exactly 0.
"""
import itertools, numpy as np

def ext_hamming(m):
    n = 2 ** m
    # parity-check matrix: columns are all m-bit vectors, plus an all-ones row
    cols = np.array([[(i >> b) & 1 for b in range(m)] for i in range(n)]).T   # m x n
    Hm = np.vstack([cols, np.ones(n, dtype=int)])                                  # (m+1) x n
    # codewords: all x in GF(2)^n with Hm x = 0 (enumerate via null space)
    # Gaussian elimination for a generator
    A = Hm.copy() % 2
    r, piv = 0, []
    A = A.astype(np.uint8)
    for c in range(n):
        p = next((i for i in range(r, A.shape[0]) if A[i, c]), None)
        if p is None:
            continue
        A[[r, p]] = A[[p, r]]
        for i in range(A.shape[0]):
            if i != r and A[i, c]:
                A[i] ^= A[r]
        piv.append(c); r += 1
        if r == A.shape[0]:
            break
    free = [c for c in range(n) if c not in piv]
    gens = []
    for f in free:
        v = np.zeros(n, dtype=np.uint8); v[f] = 1
        for i, pc in enumerate(piv):
            v[pc] = A[i, f]
        gens.append(v)
    G = np.array(gens)
    k = len(gens)
    words = np.array([(np.array(bits, dtype=np.uint8) @ G) % 2 for bits in itertools.product([0, 1], repeat=k)], dtype=np.float64)
    assert np.all((Hm @ words.T) % 2 == 0)
    return words

def nearest(X, words):
    best = np.full(len(X), np.inf); out = np.zeros_like(X)
    for c in words:
        P = c + 2.0 * np.round((X - c) / 2.0)
        d = ((X - P) ** 2).sum(1)
        m = d < best
        best[m] = d[m]; out[m] = P[m]
    return out, best

rng = np.random.default_rng(1)
for m in (3, 4):
    words = ext_hamming(m)
    n, k = 2 ** m, int(np.log2(len(words)))
    V = 2.0 ** (n - k)
    N = 40000 if m == 3 else 12000
    X = rng.uniform(0, 2, size=(N, n))
    _, d = nearest(X, words)
    Gv = d.mean() / (n * V ** (2.0 / n)); se = d.std() / np.sqrt(N) / (n * V ** (2.0 / n))
    # equal density: scale so the volume per point is 1; small inputs |x| < 1/4 of unit step
    s = V ** (1.0 / n)                         # lattice coordinates = x * s
    Y = rng.laplace(0, 0.5, size=(N, n))       # step 1 at unit density, Laplace b = 0.5
    Q, _ = nearest(Y * s, words)
    small = np.abs(Y) < 0.25
    zshare = float((Q[small] == 0).mean())
    mse = float(((Q / s - Y) ** 2).mean())
    mseZ = float(((np.round(Y) - Y) ** 2).mean())
    print(f"[{n},{k},4]: G = {Gv:.5f} +- {se:.5f}  (E8 0.07168, Z 0.08333);  Laplace b=0.5 at unit density: mse {mse:.4f} vs Z {mseZ:.4f}, zero share of |y|<1/4: {zshare:.3f}")
