import sys
import json
import numpy as np
from scipy.linalg import solve_toeplitz

def solve(problem):
    """Solve Tx = b for a Toeplitz matrix given by first column c and first row r.

    Parameters
    ----------
    problem : dict
        Dictionary with keys:
        - "c": list of numbers, first column of T
        - "r": list of numbers, first row of T
        - "b": list of numbers, right‑hand side vector

    Returns
    -------
    list of float
        Solution vector x such that T @ x = b.
    """
    # Validate presence of keys
    if not all(k in problem for k in ("c", "r", "b")):
        raise KeyError("Problem dictionary must contain 'c', 'r', and 'b' keys.")

    c = np.asarray(problem["c"], dtype=float)
    r = np.asarray(problem["r"], dtype=float)
    b = np.asarray(problem["b"], dtype=float)

    # Basic shape validation
    if c.ndim != 1 or r.ndim != 1 or b.ndim != 1:
        raise ValueError("c, r and b must be one‑dimensional sequences.")
    n = c.shape[0]
    if r.shape[0] != n or b.shape[0] != n:
        raise ValueError("Lengths of c, r, and b must be equal.")

    # Solve using SciPy's Levinson–Durbin implementation
    x = solve_toeplitz((c, r), b)
    return x.tolist()

if __name__ == "__main__":
    # Read JSON from stdin, solve, and output JSON result
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        sys.stderr.write(f"Invalid JSON input: {e}\n")
        sys.exit(1)

    try:
        result = solve(input_data)
    except Exception as e:
        sys.stderr.write(f"Error during solving: {e}\n")
        sys.exit(1)

    json.dump(result, sys.stdout)
    sys.stdout.flush()
