import logging
import numpy as np
import scipy.linalg

def solve(problem: dict) -> dict:
    """Compute the principal matrix square root of a given square matrix.

    Parameters
    ----------
    problem: dict
        A dictionary containing the key "matrix" mapping to a NumPy 2‑D array
        (real or complex) representing the matrix *A*.

    Returns
    -------
    dict
        A dictionary with a single top‑level key ``"sqrtm"`` whose value is a
        dictionary containing the key ``"X"``. ``"X"`` holds the principal
        square‑root matrix as a nested Python list (the result of ``X.tolist()``).
        If the computation fails, ``"X"`` will be an empty list to signal the
        failure – this behaviour matches the reference implementation and is
        recognised by the validator.
    """
    # Extract the matrix from the problem description
    A = problem.get("matrix")
    if A is None:
        logging.error("Problem dictionary does not contain 'matrix' key.")
        return {"sqrtm": {"X": []}}

    # Ensure we are working with a NumPy array of complex dtype – this guards
    # against issues when the input consists of Python complex literals or
    # strings that NumPy can cast.
    try:
        A = np.array(A, dtype=complex)
    except Exception as e:
        logging.error(f"Failed to convert input matrix to NumPy array: {e}")
        return {"sqrtm": {"X": []}}

    # Compute the principal square root using SciPy.  ``disp=False`` suppresses
    # warnings and returns a tuple ``(X, error)``; we ignore the error estimate.
    try:
        X, _ = scipy.linalg.sqrtm(A, disp=False)
    except Exception as e:
        logging.error(f"scipy.linalg.sqrtm failed: {e}")
        return {"sqrtm": {"X": []}}

    # Convert the resulting NumPy matrix to a plain Python nested list so that
    # it can be JSON‑serialised without additional handling.
    return {"sqrtm": {"X": X.tolist()}}
