import numpy as np
from scipy.linalg import solve_toeplitz


def _validate_input(problem):
    """Validate that problem dict contains proper Toeplitz data.

    Raises:
        ValueError: If any validation fails.
    """
    required_keys = {"c", "r", "b"}
    if not isinstance(problem, dict):
        raise ValueError("Input must be a dictionary.")
    if not required_keys.issubset(problem.keys()):
        missing = required_keys - problem.keys()
        raise ValueError(f"Missing keys in problem dict: {missing}")
    c = np.asarray(problem["c"], dtype=complex)
    r = np.asarray(problem["r"], dtype=complex)
    b = np.asarray(problem["b"], dtype=complex)
    if c.shape != r.shape or c.shape != b.shape:
        raise ValueError("Vectors c, r, and b must have the same length.")
    if c.shape[0] == 0:
        raise ValueError("Vectors must be non‑empty.")
    if not np.isclose(c[0], r[0]):
        raise ValueError("First elements of c and r must be equal for a Toeplitz matrix.")
    return c, r, b


def solve(problem):
    """Solve Tx = b for a Toeplitz matrix defined by its first column ``c`` and first row ``r``.

    Parameters
    ----------
    problem : dict
        Dictionary with keys ``c``, ``r``, and ``b`` where each value is a list of numbers.

    Returns
    -------
    list
        Solution vector ``x`` as a Python list (real/complex values are preserved).
    """
    c, r, b = _validate_input(problem)
    # Use Levinson-Durbin algorithm via scipy's solve_toeplitz.
    x = solve_toeplitz((c, r), b)
    # Convert to plain Python list; if the result is complex with zero imag part, cast to real.
    if np.iscomplexobj(x):
        # Keep complex numbers, but for pure real values return real floats.
        x = np.where(np.isclose(x.imag, 0), x.real, x)
    return x.tolist()


if __name__ == "__main__":
    import json, sys
    if len(sys.argv) != 2:
        print("Usage: python solver.py '<json_dict>'")
        sys.exit(1)
    try:
        problem = json.loads(sys.argv[1])
    except json.JSONDecodeError as e:
        print(f"Invalid JSON input: {e}")
        sys.exit(1)
    try:
        solution = solve(problem)
        print(json.dumps(solution))
    except Exception as e:
        print(f"Error solving Toeplitz system: {e}")
        sys.exit(1)
