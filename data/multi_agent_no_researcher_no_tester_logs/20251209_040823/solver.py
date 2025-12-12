import sys
import json
import numpy as np
from typing import Any, Dict, List

def solve(problem: Dict[str, Any]) -> Dict[str, Any]:
    """Solve the Cholesky factorization problem.

    Parameters
    ----------
    problem : dict
        Dictionary containing the key "matrix" with a 2‑D list (or array‑like) representing
        a symmetric positive‑definite matrix.

    Returns
    -------
    dict
        Dictionary with structure {"Cholesky": {"L": <list of lists>}} where the inner
        list is the lower‑triangular Cholesky factor.
    """
    # Extract matrix and ensure it is a NumPy array of floats
    A_input = problem.get("matrix")
    if A_input is None:
        raise ValueError("Input problem must contain a 'matrix' key.")
    A = np.array(A_input, dtype=float)

    # Compute Cholesky factor (lower‑triangular) using NumPy
    L = np.linalg.cholesky(A)
    # Force exact lower‑triangular by zeroing the upper part (helpful for numeric noise)
    L = np.tril(L)

    # Convert to plain Python nested lists (ensuring JSON‑serialisable floats)
    L_list: List[List[float]] = L.tolist()

    return {"Cholesky": {"L": L_list}}

if __name__ == "__main__":
    # When run as a script, read JSON from stdin and output the solution as JSON.
    try:
        data = json.load(sys.stdin)
    except Exception as exc:
        print(json.dumps({"error": f"Failed to parse JSON input: {exc}"}), file=sys.stderr)
        sys.exit(1)
    result = solve(data)
    json.dump(result, sys.stdout, ensure_ascii=False)
    sys.stdout.flush()
