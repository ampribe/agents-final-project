import numpy as np
from scipy import ndimage


def solve(task_input: dict) -> dict:
    """Apply a 2D affine transformation to an image.

    Parameters
    ----------
    task_input : dict
        Dictionary with keys:
        - "image": 2D list/array of floats.
        - "matrix": 2x3 list/array representing the affine transformation.

    Returns
    -------
    dict
        Dictionary with key "transformed_image" containing the transformed
        image as a nested list (same shape as input image).
    """
    try:
        # Convert inputs to numpy arrays
        image = np.array(task_input["image"], dtype=float)
        matrix = np.array(task_input["matrix"], dtype=float)

        # Perform the affine transformation using the matrix directly.
        # scipy.ndimage.affine_transform maps output coordinates to input
        # coordinates; providing the same matrix as in the reference implementation.
        transformed = ndimage.affine_transform(
            image,
            matrix,
            order=3,
            mode="constant",
            cval=0.0,
        )
        return {"transformed_image": transformed.tolist()}
    except Exception:
        # Return empty list on failure as per specification.
        return {"transformed_image": []}
