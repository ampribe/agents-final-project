import numpy as np
from typing import List, Dict, Any


def solve(problem: Dict[str, Any]) -> Dict[str, Dict[str, List[List[float]]]]:
    """Solve the QR factorization problem.

    The function expects ``problem`` to be a dictionary containing the key ``"matrix"``
    which maps to a two‑dimensional array (list of lists) representing a matrix ``A`` of
    shape *(n, n+1)*.  It computes the reduced QR decomposition ``A = Q @ R`` using
    ``numpy.linalg.qr`` and returns a dictionary of the form::

        {"QR": {"Q": <list of lists>, "R": <list of lists>}}

    ``Q`` has shape *(n, n)`` with orthonormal columns and ``R`` has shape *(n, n+1)``
    and is upper‑triangular in its leading ``n×n`` block.

    Parameters
    ----------
    problem: dict
        Dictionary with key ``"matrix"`` containing the input matrix.

    Returns
    -------
    dict
        Dictionary with the QR factorisation suitable for the validator.
    """
    # Extract and validate the input matrix
    if "matrix" not in problem:
        raise KeyError("Input problem must contain a 'matrix' key.")

    A = np.array(problem["matrix"], dtype=float)
    if A.ndim != 2:
        raise ValueError("Input matrix must be two‑dimensional.")

    # Compute the reduced QR decomposition. For an (n, n+1) matrix this yields
    # Q of shape (n, n) and R of shape (n, n+1).
    Q, R = np.linalg.qr(A, mode="reduced")

    # Convert results back to plain Python lists for JSON‑compatible output.
    solution = {"QR": {"Q": Q.tolist(), "R": R.tolist()}}
    return solution
