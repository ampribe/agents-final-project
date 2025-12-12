import numpy as np
import scipy.linalg
import logging

def solve(problem: dict) -> dict:
    """Reference solve using scipy.linalg.sqrtm."""
    A = np.array(problem["matrix"], dtype=complex)
    try:
        X, _ = scipy.linalg.sqrtm(A, disp=False)
    except Exception as e:
        logging.error(f"scipy.linalg.sqrtm failed: {e}")
        return {"sqrtm": {"X": []}}
    return {"sqrtm": {"X": X.tolist()}}
