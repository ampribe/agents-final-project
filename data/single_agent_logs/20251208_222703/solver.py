"""Cholesky Factorization Solver

This module provides a single function ``solve`` that computes the Cholesky
decomposition of a symmetric positive‑definite matrix ``A`` supplied in the
problem dictionary.  The expected problem format (as described in the task
specification) is::

    {
        "matrix": [[...], [...], ...]
    }

The solver returns a dictionary matching the validation expectations::

    {
        "Cholesky": {"L": [[...], [...], ...]}
    }

The implementation mirrors the reference solution – it converts the input to a
NumPy array, uses ``numpy.linalg.cholesky`` to compute the lower‑triangular
factor ``L`` and finally converts the result back to a plain Python list of
lists so that the validation routine (which expects a JSON‑serialisable
structure) can handle it without issues.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np


def solve(problem: Dict[str, Any]) -> Dict[str, Dict[str, list[list[float]]]]:
    """Compute the Cholesky factorization of a positive‑definite matrix.

    Parameters
    ----------
    problem:
        A dictionary containing the key ``"matrix"`` with the matrix ``A`` as a
        nested list (or any sequence convertible to a NumPy ``ndarray``).

    Returns
    -------
    dict
        A dictionary with the structure ``{"Cholesky": {"L": <lower‑triangular
        matrix as list of lists>}}``.
    """

    # Extract the matrix from the problem definition.  The input may already be
    # a NumPy array or a plain Python nested list – ``np.array`` handles both.
    A = np.array(problem["matrix"], dtype=float)

    # NumPy's ``linalg.cholesky`` returns a lower‑triangular matrix L such that
    # A = L @ L.T.  The function will raise a LinAlgError if A is not positive
    # definite; we let that propagate because the validation step will catch an
    # incorrect solution anyway.
    L = np.linalg.cholesky(A)

    # Convert the result to a plain‑Python structure for JSON compatibility.
    L_list: list[list[float]] = L.tolist()

    # Assemble the output dictionary exactly as required by the validation.
    solution: Dict[str, Dict[str, list[list[float]]]] = {"Cholesky": {"L": L_list}}
    return solution


# The module is deliberately minimal – no I/O or command‑line handling – because
# the surrounding harness imports ``solve`` directly.
