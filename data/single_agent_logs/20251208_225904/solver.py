import numpy as np

def solve(problem):
    """Solve Euclidean projection of y onto probability simplex.

    Parameters
    ----------
    problem : dict
        Dictionary with key "y" containing a list or array of numbers.

    Returns
    -------
    dict
        Dictionary with key "solution" mapping to a NumPy array of the projected vector.
    """
    # Extract y and convert to NumPy array
    y = np.array(problem.get("y", []), dtype=float)
    # Ensure y is a 1‑D vector
    y = y.flatten()
    n = y.size
    if n == 0:
        # Empty input – return empty array
        return {"solution": np.array([])}

    # Sort y in descending order
    sorted_y = np.sort(y)[::-1]
    # Cumulative sum of sorted values minus 1 (the simplex sum)
    cumsum_y = np.cumsum(sorted_y) - 1.0
    # Create indices 1..n
    rho_candidates = np.arange(1, n + 1)
    # Find the largest index where sorted_y > cumsum_y / rho
    condition = sorted_y > (cumsum_y / rho_candidates)
    # The last true index gives rho
    rho = np.where(condition)[0]
    if rho.size == 0:
        # Fallback: all entries are zero after projection
        theta = 0.0
    else:
        rho = rho[-1]
        theta = cumsum_y[rho] / (rho + 1)
    # Compute the projection
    x = np.maximum(y - theta, 0.0)
    return {"solution": x}
