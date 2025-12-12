"""Toeplitz linear system solver using Levinson-Durbin algorithm.

The problem dictionary contains:
  - \"c\": first column of the Toeplitz matrix (list of floats)
  - \"r\": first row of the Toeplitz matrix (list of floats)
  - \"b\": right‑hand side vector (list of floats)

The function returns the solution vector $x$ as a Python list.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_toeplitz


def solve(problem: dict[str, list[float]]) -> list[float]:
    """Solve the Toeplitz system ``Tx = b``.

    Parameters
    ----------
    problem:
        Dictionary with keys ``"c"``, ``"r"`` and ``"b"``.
        ``c`` and ``r`` define the Toeplitz matrix ``T``; ``b`` is the RHS.

    Returns
    -------
    list[float]
        Solution vector ``x`` such that ``T @ x = b``.
    """
    # Convert inputs to NumPy arrays – ``solve_toeplitz`` expects 1‑D arrays.
    c = np.asarray(problem["c"], dtype=float)
    r = np.asarray(problem["r"], dtype=float)
    b = np.asarray(problem["b"], dtype=float)

    # ``solve_toeplitz`` implements the Levinson‑Durbin recursion and is O(n^2).
    x = solve_toeplitz((c, r), b)

    # Return a plain Python list for compatibility with the validation harness.
    return x.tolist()
