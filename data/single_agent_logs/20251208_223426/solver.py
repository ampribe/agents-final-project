"""Solver for the 2D Image Shift task.

The problem dictionary contains:
  * ``image`` – a 2‑D list (or array) of floats representing the image.
  * ``shift`` – a list ``[shift_row, shift_col]`` specifying the sub‑pixel
    shift. Positive values shift the content downwards/rightwards.

The required output is a dictionary with a single key ``shifted_image`` whose
value is a 2‑D list of floats of the same shape as the input image.  The shift
must be performed using cubic spline interpolation (order = 3) and constant
padding (``mode='constant'``) – i.e. values outside the original image are
treated as zero.

The reference implementation uses ``scipy.ndimage.shift``; we follow the same
approach but provide a robust wrapper that converts the input to a NumPy array,
applies the operation, and returns a plain‑Python list suitable for JSON
serialization.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from scipy.ndimage import shift as nd_shift


def solve(problem: Dict[str, Any]) -> Dict[str, Any]:
    """Shift a 2‑D image by a sub‑pixel amount.

    Parameters
    ----------
    problem:
        Dictionary with keys ``"image"`` (n×n list of floats) and ``"shift"``
        (list of two floats ``[row_shift, col_shift]``).

    Returns
    -------
    dict
        ``{"shifted_image": <list>}`` where the list has the same shape as the
        input image.  If the operation fails an empty list is returned – this
        mirrors the behaviour expected by the validation script.
    """

    # Extract inputs safely; any KeyError will propagate as an exception which
    # we catch below and convert into the empty‑list failure mode.
    try:
        image = problem["image"]
        shift_vec = problem["shift"]
    except Exception:
        # Missing keys – signal failure.
        return {"shifted_image": []}

    # Convert to NumPy array of float64 for stability.
    try:
        img_arr = np.array(image, dtype=float)
    except Exception:
        return {"shifted_image": []}

    # Ensure shift vector has length 2 and consists of numbers.
    if not (isinstance(shift_vec, (list, tuple)) and len(shift_vec) == 2):
        return {"shifted_image": []}

    try:
        shift_vals = (float(shift_vec[0]), float(shift_vec[1]))
    except Exception:
        return {"shifted_image": []}

    # Perform the shift using cubic spline interpolation (order=3) and
    # constant padding (cval=0.0).
    try:
        shifted_arr = nd_shift(
            img_arr,
            shift_vals,
            order=3,
            mode="constant",
            cval=0.0,
        )
    except Exception:
        # Any failure (e.g., invalid shape) results in an empty list as per
        # the validation expectations.
        return {"shifted_image": []}

    # Convert back to plain Python nested lists.
    shifted_list: List[List[float]] = shifted_arr.tolist()
    return {"shifted_image": shifted_list}
