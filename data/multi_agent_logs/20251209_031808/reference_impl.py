import numpy as np
from scipy.linalg import lu

def solve(problem: dict) -> dict:
    """Reference solution for LU factorization.
    Expects problem = {"matrix": [[...], ...]}
    Returns {"LU": {"P": ..., "L": ..., "U": ...}}
    """
    A = np.asarray(problem["matrix"], dtype=float)
    # Use scipy.linalg.lu which returns P, L, U such that A = P @ L @ U
    P, L, U = lu(A)
    return {"LU": {"P": P.tolist(), "L": L.tolist(), "U": U.tolist()}}
