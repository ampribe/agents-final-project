"""Reference implementation for Cholesky factorization.
Provides a solve(problem) function returning the factor L as a list of lists.
"""
import numpy as np

def solve(problem):
    """Compute the Cholesky decomposition of a symmetric positive definite matrix.

    Parameters
    ----------
    problem : dict
        Dictionary with key "matrix" containing a 2D list or numpy array.

    Returns
    -------
    dict
        Dictionary with structure {"Cholesky": {"L": L}}
        where L is a list of lists (converted from numpy array).
    """
    A = np.array(problem["matrix"], dtype=float)
    L = np.linalg.cholesky(A)
    return {"Cholesky": {"L": L.tolist()}}
