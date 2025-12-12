import numpy as np

def solve(problem: dict) -> dict:
    """Reference solution using numpy.linalg.cholesky.
    Returns a dict with key 'Cholesky' containing 'L' as a list of lists.
    """
    A = np.array(problem["matrix"], dtype=float)
    L = np.linalg.cholesky(A)
    # Convert to nested list
    L_list = L.tolist()
    return {"Cholesky": {"L": L_list}}
