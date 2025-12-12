import numpy as np

def solve(problem: dict) -> list:
    """Solve Toeplitz system using dense conversion and numpy.linalg.solve.

    Parameters
    ----------
    problem: dict with keys 'c', 'r', 'b'
        'c' and 'r' are first column and first row of Toeplitz matrix (lists of floats).
        'b' is right‑hand side vector.

    Returns
    -------
    list of floats – solution vector x such that Tx = b.
    """
    c = np.array(problem["c"], dtype=float)
    r = np.array(problem["r"], dtype=float)
    b = np.array(problem["b"], dtype=float)
    n = len(b)
    # Build dense Toeplitz matrix
    T = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i >= j:
                T[i, j] = c[i - j]
            else:
                T[i, j] = r[j - i]
    # Solve dense system
    x = np.linalg.solve(T, b)
    return x.tolist()
