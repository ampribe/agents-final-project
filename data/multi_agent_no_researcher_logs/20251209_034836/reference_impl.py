import numpy as np
import scipy.linalg
import logging

def solve(problem: dict) -> dict:
    """Reference solution using scipy.linalg.sqrtm.
    Handles SciPy versions where sqrtm returns either a matrix or a tuple (matrix, error).
    Expects problem["matrix"] to be a numpy array (real or complex).
    Returns dict with key "sqrtm" -> {"X": X.tolist()}.
    """
    A = problem.get("matrix")
    if A is None:
        logging.error("Problem missing 'matrix' key")
        return {"sqrtm": {"X": []}}
    try:
        result = scipy.linalg.sqrtm(A, disp=False)
        # SciPy may return either a matrix or (matrix, error)
        if isinstance(result, tuple) and len(result) >= 1:
            X = result[0]
        else:
            X = result
    except Exception as e:
        logging.error(f"scipy.linalg.sqrtm failed: {e}")
        return {"sqrtm": {"X": []}}
    return {"sqrtm": {"X": X.tolist()}}
