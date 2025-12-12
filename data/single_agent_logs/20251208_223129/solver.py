# -*- coding: utf-8 -*-
"""Solver for the 2D Affine Transform problem.

The task is to apply a 2‑D affine transformation to an input image using
``scipy.ndimage.affine_transform`` with cubic spline interpolation (order=3)
and constant padding (mode='constant').

The function ``solve`` receives a dictionary with two keys:
    - ``"image"``: a square 2‑D list of floats representing the image.
    - ``"matrix"``: a 2×3 list representing the affine transformation matrix.

It must return a dictionary containing the transformed image under the key
``"transformed_image"``.  The transformed image should be a plain Python list of
lists (i.e., ``numpy.ndarray.tolist()``) so that the validation code can easily
convert it back to an array.

If the transformation fails for any reason we follow the convention of the
reference implementation and return ``{"transformed_image": []}`` – an empty
list signals a failure which the validator treats specially.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from scipy import ndimage


def solve(problem: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a 2‑D affine transformation to an image.

    Parameters
    ----------
    problem:
        Dictionary with keys ``"image"`` (n×n list of floats) and ``"matrix"``
        (2×3 list of floats).

    Returns
    -------
    dict
        ``{"transformed_image": <list>}`` where the list is a nested Python
        list representing the transformed image.  If an exception occurs, the
        value is an empty list.
    """

    # Extract input; validation of types/shapes is delegated to the caller
    image_list: List[List[float]] = problem.get("image")
    matrix_list: List[List[float]] = problem.get("matrix")

    # Basic sanity check – if something is missing we return the failure token
    if image_list is None or matrix_list is None:
        return {"transformed_image": []}

    try:
        # Convert inputs to NumPy arrays of type float64 for stability
        image_arr = np.asarray(image_list, dtype=np.float64)
        matrix_arr = np.asarray(matrix_list, dtype=np.float64)

        # scipy.ndimage.affine_transform expects a (ndim, ndim) matrix that
        # maps output coordinates to input coordinates. The provided 2×3 matrix
        # contains the linear part (first two columns) and the offset (third
        # column). We split them accordingly.
        linear = matrix_arr[:, :2]
        offset = matrix_arr[:, 2]

        # Perform the transformation with the required settings.
        transformed = ndimage.affine_transform(
            image_arr,
            linear,
            offset=offset,
            order=3,            # cubic spline interpolation
            mode="constant",   # pad with zeros outside the input domain
            cval=0.0,
        )

        # Convert back to a plain Python nested list for the output format.
        result_list = transformed.tolist()
        return {"transformed_image": result_list}
    except Exception:
        # Any unexpected error results in an empty list, matching the reference.
        return {"transformed_image": []}
