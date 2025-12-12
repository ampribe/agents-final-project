import sys
import json
import numpy as np

def solve(problem):
    """Compute QR factorization of the given matrix.

    Parameters
    ----------
    problem : dict
        Dictionary containing the key "matrix" mapped to a 2‑D list of numbers.

    Returns
    -------
    dict
        Dictionary with key "QR" mapping to another dict containing "Q" and "R"
        as 2‑D Python lists.
    """
    # Retrieve matrix; handle missing or empty gracefully
    A_list = problem.get("matrix", [])
    if not A_list:
        # Empty input – return empty structures matching expected shapes
        return {"QR": {"Q": [], "R": []}}
    A = np.array(A_list, dtype=float)
    # Use reduced QR so Q is (n, n) and R is (n, n+1) for an (n, n+1) matrix
    Q, R = np.linalg.qr(A, mode="reduced")
    # Convert to regular Python lists for JSON serialization
    return {"QR": {"Q": Q.tolist(), "R": R.tolist()}}

if __name__ == "__main__":
    # When run as a script, read JSON from stdin and output the solution JSON.
    try:
        input_data = json.load(sys.stdin)
    except Exception as e:
        print(json.dumps({"error": f"Failed to parse input JSON: {e}"}))
        sys.exit(1)
    result = solve(input_data)
    json.dump(result, sys.stdout)
