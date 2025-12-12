import numpy as np

def solve(problem):
    """Compute the N-dimensional FFT of a real-valued matrix.

    Parameters
    ----------
    problem : array-like
        Real-valued input matrix (n x n).

    Returns
    -------
    numpy.ndarray
        Complex-valued FFT of the input with the same shape.
    """
    # Use NumPy's FFT implementation which matches the validation reference.
    return np.fft.fftn(problem)
