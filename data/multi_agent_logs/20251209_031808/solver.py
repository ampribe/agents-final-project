import numpy as np
from scipy.linalg import lu

def solve(problem):
    """Solve the LU factorization problem.

    Parameters
    ----------
    problem : dict
        Dictionary containing the key ``"matrix"`` which represents a square matrix
        (list of lists or a NumPy array).

    Returns
    -------
    dict
        A dictionary with a single key ``"LU"`` mapping to another dictionary that
        contains the permutation matrix ``P``, lower‑triangular matrix ``L`` (unit
        diagonal), and upper‑triangular matrix ``U``. All matrices are returned as
        plain Python ``list`` objects (list of rows) so they are JSON serialisable.
    """
    # Validate input existence
    if not isinstance(problem, dict):
        raise TypeError("Problem must be a dictionary.")
    if "matrix" not in problem:
        raise KeyError("Problem dictionary must contain the key 'matrix'.")

    # Convert the provided matrix to a NumPy array of floats
    A = np.asarray(problem["matrix"], dtype=float)

    # Ensure the matrix is square
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Input matrix must be two‑dimensional and square.")

    # Compute the LU decomposition with partial pivoting (SciPy returns P, L, U
    # such that A = P @ L @ U, where P is a permutation matrix).
    P, L, U = lu(A)

    # Convert the results back to native Python lists for JSON compatibility
    result = {
        "LU": {
            "P": P.tolist(),
            "L": L.tolist(),
            "U": U.tolist()
        }
    }
    return result
