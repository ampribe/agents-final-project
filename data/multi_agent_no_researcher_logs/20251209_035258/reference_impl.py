import numpy as np

def solve(problem: dict) -> list:
    """Reference implementation solving Toeplitz system using full matrix solve.
    Supports real and complex numbers.
    """
    c = np.array(problem["c"], dtype=complex)
    r = np.array(problem["r"], dtype=complex)
    b = np.array(problem["b"], dtype=complex)
    n = len(c)
    if len(r) != n or len(b) != n:
        raise ValueError("Inconsistent dimensions")
    # Build Toeplitz matrix
    T = np.empty((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            if i >= j:
                T[i, j] = c[i - j]
            else:
                T[i, j] = r[j - i]
    x = np.linalg.solve(T, b)
    return x.tolist()
