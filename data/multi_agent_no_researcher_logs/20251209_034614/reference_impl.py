import numpy as np
from scipy.linalg import lu

def solve(problem: dict) -> dict:
    """Reference LU factorization using scipy.linalg.lu.
    Returns a dict with key 'LU' mapping to {'P': ..., 'L': ..., 'U': ...}.
    """
    A = np.asarray(problem["matrix"], dtype=float)
    P, L, U = lu(A)
    return {"LU": {"P": P.tolist(), "L": L.tolist(), "U": U.tolist()}}
