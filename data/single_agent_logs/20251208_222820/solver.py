"""QR Factorization Solver

This module provides a single ``solve`` function that computes the QR
factorization of a given matrix ``A``.  The function follows the contract
required by the validation script:

* Input is a dictionary with a key ``"matrix"`` containing a 2‑D array (or a
  list‑of‑lists) representing ``A``.  In the problem instances used for this
  challenge the matrix has shape ``(n, n+1)`` – i.e. more columns than rows.
* The function returns a dictionary with a single key ``"QR"`` that maps to a
  dictionary with keys ``"Q"`` and ``"R"``.  Both ``Q`` and ``R`` are
  converted to plain Python nested lists via ``tolist()`` so that the output
  is JSON‑serialisable.

Implementation details
----------------------
We rely on ``numpy.linalg.qr`` with ``mode='reduced'``.  For an ``(n, m)``
matrix with ``m >= n`` the reduced QR decomposition yields:

* ``Q`` – an ``(n, n)`` matrix with orthonormal columns.
* ``R`` – an ``(n, m)`` upper‑triangular matrix (the upper‑triangular property
  applies to the leading ``n × n`` block).

These shapes satisfy the expectations of the validation logic.  The function
converts the input to a ``numpy.ndarray`` (ensuring a floating‑point dtype) and
then performs the decomposition.  Finally the result is transformed back to
the required list format.
"""

from __future__ import annotations

import numpy as np


def solve(problem: dict) -> dict:
    """Compute the QR factorization of the provided matrix.

    Parameters
    ----------
    problem:
        A dictionary containing the key ``"matrix"`` whose value is a
        2‑dimensional array (list‑of‑lists, ``np.ndarray``, etc.).

    Returns
    -------
    dict
        A dictionary of the form ``{"QR": {"Q": ..., "R": ...}}`` where
        ``Q`` and ``R`` are plain Python nested lists.
    """

    # Extract the matrix and ensure it is a NumPy array of type float.
    A = np.array(problem["matrix"], dtype=float)

    # Compute the reduced QR decomposition.  For an (n, n+1) matrix this yields
    # Q.shape == (n, n) and R.shape == (n, n+1).
    Q, R = np.linalg.qr(A, mode="reduced")

    # Convert the results back to plain lists for JSON compatibility.
    return {"QR": {"Q": Q.tolist(), "R": R.tolist()}}
