import sys
import json
import numpy as np
from typing import Any, Dict, List


def solve(problem: Dict[str, Any]) -> Dict[str, Dict[str, List[List[float]]]]:
    """Compute the QR factorization of a given matrix.

    The input ``problem`` must be a dictionary with a key ``"matrix"`` whose value
    is a two‑dimensional list (or any JSON array) representing a numeric matrix.
    The function returns a dictionary of the form::

        {"QR": {"Q": [...], "R": [...]}}

    where ``Q`` and ``R`` are standard Python lists (converted from NumPy arrays)
    that satisfy ``A = Q @ R``. ``Q`` has orthonormal columns and ``R`` is upper
    triangular in its leading square block.
    """
    if "matrix" not in problem:
        raise ValueError("Input problem must contain a 'matrix' key.")
    # Convert the supplied matrix to a NumPy array of type float64.
    A = np.array(problem["matrix"], dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("Provided matrix must be two‑dimensional.")
    # Perform the reduced QR factorization. For an (n, m) matrix with m > n this
    # yields Q of shape (n, n) and R of shape (n, m).
    Q, R = np.linalg.qr(A, mode="reduced")
    # Convert results back to plain Python lists for JSON serialisation.
    return {"QR": {"Q": Q.tolist(), "R": R.tolist()}}


def _main() -> None:
    """Entry point for the script when executed directly.

    Reads a JSON problem description from standard input, runs ``solve`` and
    writes the JSON solution to standard output.
    """
    try:
        raw_input = sys.stdin.read()
        problem = json.loads(raw_input)
        solution = solve(problem)
        json.dump(solution, sys.stdout, separators=(",", ":"))
    except Exception as exc:
        # Emit a minimal error message to stderr and exit with a non‑zero status.
        sys.stderr.write(f"Error: {exc}\n")
        sys.exit(1)


if __name__ == "__main__":
    _main()
