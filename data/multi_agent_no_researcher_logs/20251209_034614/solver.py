'''solver.py
LU Factorization Solver

This module provides a single public function ``solve`` that computes the LU
factorisation with partial pivoting of a square matrix provided in the input
dictionary.

The expected input format is::

    {
        "matrix": [[...], [...], ...]
    }

The function returns a dictionary with the key ``"LU"`` mapping to another
dictionary containing the permutation matrix ``P``, the unit‑lower‑triangular
matrix ``L`` and the upper‑triangular matrix ``U`` in plain Python list form:

    {
        "LU": {
            "P": [[...], ...],
            "L": [[...], ...],
            "U": [[...], ...]
        }
    }

The implementation relies on :func:`scipy.linalg.lu`, which performs an LU
decomposition with partial pivoting and returns ``P, L, U`` such that ``A = P @
L @ U``.  All numeric values are converted to Python ``float`` before being
included in the output to match the expected validation format.
''' 

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
from scipy.linalg import lu

logger = logging.getLogger(__name__)


def _validate_input(problem: Dict[str, Any]) -> np.ndarray:
    """Extract and validate the matrix from ``problem``.

    Parameters
    ----------
    problem: dict
        Expected to contain the key ``"matrix"`` whose value is a square
        two‑dimensional array‑like structure.

    Returns
    -------
    np.ndarray
        The matrix as a ``float`` ``ndarray``.

    Raises
    ------
    ValueError
        If the key is missing, the matrix is not two‑dimensional, or it is not
        square.
    """
    if "matrix" not in problem:
        raise ValueError("Input dictionary must contain a 'matrix' key.")

    # Convert to NumPy array of floats – this also copies nested Python lists.
    A = np.asarray(problem["matrix"], dtype=float)

    if A.ndim != 2:
        raise ValueError("The provided matrix must be two‑dimensional.")
    rows, cols = A.shape
    if rows != cols:
        raise ValueError("LU factorisation requires a square matrix (n×n).")
    if rows == 0:
        raise ValueError("Empty matrix provided.")

    return A


def solve(problem: Dict[str, Any]) -> Dict[str, Dict[str, List[List[float]]]]:
    """Compute the LU factorisation of a square matrix.

    The function follows the contract required by the evaluation harness:

    * Input is a dictionary containing the key ``"matrix"``.
    * Output is a dictionary with the key ``"LU"`` mapping to another dictionary
      with the keys ``"P"``, ``"L"`` and ``"U"``.
    * All matrices in the output are plain Python ``list`` objects containing
      ``float`` entries.

    Parameters
    ----------
    problem: dict
        ``{"matrix": [[...], [...], ...]}``

    Returns
    -------
    dict
        ``{"LU": {"P": ..., "L": ..., "U": ...}}``
    """
    try:
        A = _validate_input(problem)
    except Exception as exc:
        logger.error("Invalid input: %s", exc)
        raise

    # Perform LU decomposition with partial pivoting.
    # scipy.linalg.lu returns permutation matrix P, lower‑triangular L with unit
    # diagonal, and upper‑triangular U such that A = P @ L @ U.
    P, L, U = lu(A)

    # Convert NumPy arrays to nested Python lists of plain floats.
    # ``tolist()`` already yields Python scalars, but they may be NumPy float64 –
    # casting ensures they are built‑in ``float`` for maximum compatibility.
    def _to_float_lists(mat: np.ndarray) -> List[List[float]]:
        return [[float(item) for item in row] for row in mat.tolist()]

    solution = {
        "LU": {
            "P": _to_float_lists(P),
            "L": _to_float_lists(L),
            "U": _to_float_lists(U),
        }
    }
    return solution

# When the module is imported, expose ``solve`` at the top level for the test
# harness.  No additional objects are required.
