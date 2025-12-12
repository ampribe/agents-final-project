"""LU Factorization Solver

This module provides a ``solve`` function that computes the LU decomposition
with partial pivoting of a given square matrix ``A``.  The decomposition is
expressed as:

    A = P @ L @ U

where ``P`` is a permutation matrix, ``L`` is lower‑triangular with unit
 diagonal, and ``U`` is upper‑triangular.  The implementation mirrors the
reference solution by delegating the heavy lifting to ``scipy.linalg.lu``.

The function follows the exact signature expected by the validation script:

    def solve(problem: dict) -> dict:

The ``problem`` dictionary must contain a key ``"matrix"`` holding the matrix
as a list of lists, a NumPy array, or any array‑like structure.  The returned
dictionary contains a single key ``"LU"`` that maps to another dictionary with
the three factor matrices converted to plain Python ``list`` objects.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
from scipy.linalg import lu


def solve(problem: Dict[str, Any]) -> Dict[str, Dict[str, list[list[float]]]]:
    """Compute the LU factorisation of the input matrix.

    Parameters
    ----------
    problem: dict
        A dictionary with a single entry ``"matrix"`` representing the square
        matrix ``A`` to factorise.  The value can be a list‑of‑lists, a NumPy
        array, or any array‑like object that ``np.asarray`` can interpret.

    Returns
    -------
    dict
        ``{"LU": {"P": ..., "L": ..., "U": ...}}`` where the three inner
        matrices are plain Python nested lists (``list[list[float]]``) suitable
        for JSON serialisation and for the validation routine.
    """

    # Extract and ensure the matrix is a NumPy array of type float.
    A = np.asarray(problem["matrix"], dtype=float)

    # Compute the permutation, lower‑triangular and upper‑triangular factors.
    # ``scipy.linalg.lu`` returns P, L, U such that A = P @ L @ U.
    P, L, U = lu(A)

    # Convert the NumPy arrays to nested Python lists for the expected output.
    solution = {
        "LU": {
            "P": P.tolist(),
            "L": L.tolist(),
            "U": U.tolist(),
        }
    }
    return solution


# Optional simple test when running the file directly.
if __name__ == "__main__":
    example = {"matrix": [[2.0, 3.0], [5.0, 4.0]]}
    print(solve(example))
