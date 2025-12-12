import numpy as np
from scipy import ndimage

def solve(problem: dict) -> dict:
    """Apply a 2D affine transformation to an image using scipy.ndimage.

    Parameters
    ----------
    problem : dict
        Must contain:
        - "image": 2D list/array of floats.
        - "matrix": 2x3 list/array representing affine transform.

    Returns
    -------
    dict
        {'transformed_image': <nested list of transformed image>}
    """
    # Convert inputs to numpy arrays
    image = np.asarray(problem["image"], dtype=float)
    matrix = np.asarray(problem["matrix"], dtype=float)

    # Directly use the provided matrix; scipy treats a (2,3) matrix as including offset.
    transformed = ndimage.affine_transform(
        image,
        matrix,
        order=3,
        mode="constant",
        cval=0.0,
    )

    return {"transformed_image": transformed.tolist()}
