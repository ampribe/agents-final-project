import sys
import json
import numpy as np

def _compute_cholesky(A: np.ndarray) -> np.ndarray:
    """Compute the lower‑triangular Cholesky factor of a symmetric positive‑definite matrix.

    Tries to use :func:`scipy.linalg.cholesky` for potential speed/robustness. If SciPy is not
    available, falls back to :func:`numpy.linalg.cholesky`.
    """
    # Try SciPy first
    try:
        import scipy.linalg
        # SciPy's cholesky returns upper‑triangular by default; we request lower=True
        L = scipy.linalg.cholesky(A, lower=True, overwrite_a=False, check_finite=True)
        return L
    except Exception:  # noqa: BLE001 – we deliberately fall back on any failure (ImportError, etc.)
        # Fallback to NumPy
        return np.linalg.cholesky(A)


def solve(problem: dict) -> dict:
    """Solve the Cholesky factorization problem.

    Parameters
    ----------
    problem: dict
        Expected to contain the key ``"matrix"`` with a 2‑D list/array representing a
        symmetric positive‑definite matrix.

    Returns
    -------
    dict
        ``{"Cholesky": {"L": <list‑of‑lists>}}`` where ``L`` is the lower‑triangular factor.
    """
    if not isinstance(problem, dict) or "matrix" not in problem:
        raise ValueError("Input must be a dict with a 'matrix' key.")

    # Convert input to a NumPy array (ensure float dtype for stability)
    A = np.array(problem["matrix"], dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square (2‑D).")

    # Optional sanity check: symmetry (within tolerance)
    if not np.allclose(A, A.T, atol=1e-12):
        raise ValueError("Matrix is not symmetric.")

    # Compute the factor
    L = _compute_cholesky(A)

    # Ensure we return a pure Python list structure (JSON‑serialisable)
    result = {"Cholesky": {"L": L.tolist()}}
    return result


if __name__ == "__main__":
    # The script is intended to be used both as a module (imported by the grader) and as a
    # stand‑alone executable that reads JSON from stdin and writes JSON to stdout.
    try:
        raw_input = sys.stdin.read()
        problem = json.loads(raw_input) if raw_input else {}
        solution = solve(problem)
        json.dump(solution, sys.stdout, separators=(",", ":"))
    except Exception as exc:
        # In case of unexpected errors we output a JSON object with an 'error' field – this makes
        # debugging easier for the evaluation harness.
        json.dump({"error": str(exc)}, sys.stdout)
        sys.exit(1)
