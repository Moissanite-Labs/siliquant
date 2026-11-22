"""The matrix that turns E8, and nothing else. NumPy only.

H is the Hadamard matrix of order 8 by Sylvester's doubling, and U is H over the root of 8. U is symmetric
and is its own inverse, so turning twice gives a matrix back.

E8 turned by U is the lattice that draft pull request 24 calls E8_Had. Its points times the root of 2 have
whole coordinates, and the remainders of those coordinates on division by 2 form a word of the extended
Hamming code of length 8. test_turn.py checks both statements.
"""
from __future__ import annotations

import math

import numpy as np

H = np.array([[1.0]])
for _ in range(3):
    H = np.block([[H, H], [H, -H]])
U = H / math.sqrt(8.0)


def turn(M):
    """Every block of 8 consecutive weights of a row, turned by U. The width of M is a multiple of 8."""
    M = np.asarray(M, dtype=np.float64)
    return (M.reshape(-1, 8) @ U).reshape(M.shape)
