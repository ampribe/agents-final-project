"""Correlate 1D Solver

This module provides a `solve` function that computes the one-dimensional
correlation (convolution) of a pair of input arrays. The expected input is a
tuple ``(a, b)`` where each element is array-like (list, NumPy array, etc.).
The function returns a NumPy ``ndarray`` containing the correlation result
using ``scipy.signal.convolve`` with mode ``"full"`` – matching the reference
implementation used by the validation script.

The validation routine will compare the returned array against the reference
calculated with the same mode, allowing a tolerance of 1e-6.
"""

from __future__ import annotations

from typing import Tuple, Any

import numpy as np
from scipy import signal


def solve(problem: Tuple[Any, Any]) -> np.ndarray:
    """Compute the one-dimensional correlation of two arrays.

    Parameters
    ----------
    problem: tuple
        A pair ``(a, b)`` where each entry is array-like.

    Returns
    -------
    np.ndarray
        The correlation (full convolution) of ``a`` and ``b``.
    """

    a, b = problem
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    # Perform full-mode convolution which corresponds to correlation for 1-D signals.
    result = signal.convolve(a_arr, b_arr, mode="full")
    return result


# Simple sanity check when run directly.
if __name__ == "__main__":
    example = ([0.5, -0.2, 0.3, 0.7], [1.0, 0.8])
    print(solve(example))
