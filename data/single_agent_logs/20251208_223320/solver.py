'''Solver for 2D Image Rotation.

The task is to rotate a square 2‑D array (image) counter‑clockwise by a given
angle (in degrees).  The output must keep the original shape, use cubic spline
interpolation (order=3) and treat points outside the original image as zero
(constant padding).

The validation code compares the returned list with the result of
scipy.ndimage.rotate called with the same parameters, so the implementation
must match that behaviour exactly.
'''

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
from scipy import ndimage


def solve(problem: Dict[str, Any]) -> Dict[str, Any]:
    '''Rotate the input image.

    Parameters
    ----------
    problem: dict
        {"image": [[...]], "angle": float}
        * ``image`` – a square 2‑D list of floats (0‑255).
        * ``angle`` – rotation angle in degrees, counter‑clockwise.

    Returns
    -------
    dict
        {"rotated_image": [[...]]} – the rotated image as a nested Python
        list. If the rotation cannot be performed the function returns an empty
        list for ``rotated_image`` so that the validator can treat it as a
        failure case.
    '''
    # Basic validation – any problem with the input results in the empty‑list
    # fallback used by the reference validator.
    if not isinstance(problem, dict) or "image" not in problem or "angle" not in problem:
        logging.error("Problem dictionary missing required keys.")
        return {"rotated_image": []}

    try:
        # Convert the supplied nested list into a NumPy array of type float.
        image = np.asarray(problem["image"], dtype=float)
        angle = float(problem["angle"])  # ensure a plain float
    except Exception as exc:  # pragma: no cover – defensive programming
        logging.error(f"Unable to parse problem data: {exc}")
        return {"rotated_image": []}

    # The specification (and the validator) expects:
    #   * reshape=False – keep the original (n, n) shape.
    #   * order=3 – cubic spline interpolation.
    #   * mode='constant' with cval=0 – pad outside the image with zeros.
    try:
        rotated = ndimage.rotate(
            image,
            angle,
            reshape=False,
            order=3,
            mode="constant",
            cval=0.0,
        )
    except Exception as exc:  # pragma: no cover – if SciPy fails for any reason
        logging.error(f"scipy.ndimage.rotate failed: {exc}")
        return {"rotated_image": []}

    # Convert back to a plain Python nested list for the JSON‑compatible output.
    rotated_list: List[List[float]] = rotated.tolist()
    return {"rotated_image": rotated_list}
