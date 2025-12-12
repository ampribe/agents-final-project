import numpy as np
import scipy.fftpack

def solve(problem: np.ndarray) -> np.ndarray:
    """Compute the N-dimensional DCT Type I of the input array.

    Parameters
    ----------
    problem : np.ndarray
        Real-valued input array of shape (n+1, n+1) (or any dimensionality).

    Returns
    -------
    np.ndarray
        The DCT Type I transformed array of the same shape as the input.
    """
    # Use scipy's dctn with type=1 to compute the DCT along all axes.
    result = scipy.fftpack.dctn(problem, type=1)
    return result
