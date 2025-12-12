import sys
import json
import numpy as np
from scipy.linalg import lu

def solve(problem):
    """Solve the LU factorization problem.

    Parameters
    ----------
    problem : dict
        Dictionary with key "matrix" containing a square matrix (list of lists of numbers).

    Returns
    -------
    dict
        Dictionary with key "LU" mapping to another dict with keys "P", "L", "U".
        Each matrix is represented as a list of lists of floats.
    """
    # Validate input
    if not isinstance(problem, dict) or "matrix" not in problem:
        raise ValueError("Input must be a dict containing the key 'matrix'.")

    A = np.array(problem["matrix"], dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Input matrix must be square.")

    # Compute LU decomposition with partial pivoting: A = P @ L @ U
    P, L, U = lu(A)

    # Convert results to Python lists (ensure standard Python float type)
    P_list = P.tolist()
    L_list = L.tolist()
    U_list = U.tolist()

    return {"LU": {"P": P_list, "L": L_list, "U": U_list}}

if __name__ == "__main__":
    # Read JSON from stdin
    try:
        input_str = sys.stdin.read()
        problem = json.loads(input_str)
    except Exception as e:
        print(json.dumps({"error": f"Failed to read input JSON: {e}"}))
        sys.exit(1)

    try:
        solution = solve(problem)
        print(json.dumps(solution))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
