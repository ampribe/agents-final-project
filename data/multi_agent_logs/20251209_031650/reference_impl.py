import numpy as np

def solve(problem: dict) -> dict:
    """Reference QR factorization implementation using numpy.linalg.qr.

    Args:
        problem: A dict with key "matrix" containing a 2‑D list or np.ndarray.
    Returns:
        A dict with key "QR" mapping to {'Q': ..., 'R': ...} where Q and R are
        Python lists (converted via .tolist()).
    """
    A = np.array(problem["matrix"], dtype=float)
    Q, R = np.linalg.qr(A, mode="reduced")
    return {"QR": {"Q": Q.tolist(), "R": R.tolist()}}
