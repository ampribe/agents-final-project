import numpy as np
import scipy.fftpack

def solve(problem):
    """Compute the 2-D DST Type II of the input array.

    Parameters
    ----------
    problem : array-like
        Real-valued input matrix (n x n).

    Returns
    -------
    ndarray
        The DST Type II transformed matrix of the same shape.
    """
    # Ensure input is a NumPy array
    arr = np.asarray(problem, dtype=float)
    # Use scipy's dstn for N-dimensional DST (type II)
    result = scipy.fftpack.dstn(arr, type=2)
    return result
