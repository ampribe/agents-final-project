import numpy as np

def solve(problem: dict) -> dict:
    """Reference QR factorization using numpy.linalg.qr (reduced mode)."""
    A = np.array(problem["matrix"], dtype=float)
    Q, R = np.linalg.qr(A, mode="reduced")
    return {"QR": {"Q": Q.tolist(), "R": R.tolist()}}
