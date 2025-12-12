import sys
import json
import numpy as np
from scipy.linalg import solve_toeplitz

def solve(problem):
    """Solve Tx = b for a Toeplitz matrix T defined by first column c and first row r.

    Parameters
    ----------
    problem: dict
        Dictionary with keys "c", "r", "b" containing list-like numeric values.

    Returns
    -------
    list of float
        Solution vector x such that T @ x = b.
    """
    # Convert inputs to NumPy arrays of float type
    c = np.asarray(problem["c"], dtype=float)
    r = np.asarray(problem["r"], dtype=float)
    b = np.asarray(problem["b"], dtype=float)

    # Use SciPy's Levinson-Durbin implementation for Toeplitz systems
    x = solve_toeplitz((c, r), b)
    # Return as plain Python list for JSON serialisation
    return x.tolist()

if __name__ == "__main__":
    # Read JSON from stdin, solve, and output solution as JSON
    data = json.load(sys.stdin)
    result = solve(data)
    json.dump(result, sys.stdout)
