# -*- coding: utf-8 -*-
"""FFT Convolution Solver

This module provides a ``solve`` function that computes the convolution of two
signals using the Fast Fourier Transform (FFT) approach.  It mirrors the
reference implementation and conforms to the validation expectations.

The ``solve`` function expects a dictionary with the following keys:

- ``"signal_x"``: list of numbers (the first signal)
- ``"signal_y"``: list of numbers (the second signal)
- ``"mode"`` (optional): one of ``"full"``, ``"same"``, ``"valid"`` indicating
  the convolution mode.  Defaults to ``"full"``.

It returns a dictionary containing a single key ``"convolution"`` whose value
is a plain Python list (JSON‑serialisable) representing the convolution result.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from scipy import signal


def solve(problem: Dict[str, Any]) -> Dict[str, List[float]]:
    """Compute the convolution of two signals using FFT.

    Parameters
    ----------
    problem: dict
        Dictionary with keys ``signal_x``, ``signal_y`` and optional ``mode``.

    Returns
    -------
    dict
        ``{"convolution": [...]}`` where the list contains the convolution
        result as floats.
    """
    # Extract inputs, converting to NumPy arrays of dtype float for safety.
    x = np.asarray(problem.get("signal_x", []), dtype=float)
    y = np.asarray(problem.get("signal_y", []), dtype=float)
    mode = problem.get("mode", "full")

    # If either input is empty, the convolution result should be an empty list.
    if x.size == 0 or y.size == 0:
        return {"convolution": []}

    # Perform FFT‑based convolution using scipy.signal.fftconvolve.
    conv_result = signal.fftconvolve(x, y, mode=mode)

    # Convert the NumPy array to a plain Python list of floats.
    conv_list = [float(v) for v in conv_result.tolist()]

    return {"convolution": conv_list}


if __name__ == "__main__":
    example = {
        "signal_x": [1.0, 2.0, 3.0, 4.0],
        "signal_y": [5.0, 6.0, 7.0],
        "mode": "full",
    }
    print(solve(example))
