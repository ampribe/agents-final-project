import numpy as np

def solve(problem: dict) -> dict:
    """Project a symmetric matrix onto the PSD cone.

    Args:
        problem: Dictionary with key "A" containing a symmetric matrix (list of lists or np.ndarray).

    Returns:
        Dictionary with key "X" containing the projected PSD matrix.
    """
    # Convert input to NumPy array of floats
    A = np.array(problem["A"], dtype=float)
    # Use eigh for symmetric matrices (more stable than eig)
    eigvals, eigvecs = np.linalg.eigh(A)
    # Zero out negative eigenvalues
    eigvals = np.maximum(eigvals, 0.0)
    # Reconstruct the matrix
    X = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Ensure symmetry (numerical errors)
    X = (X + X.T) / 2.0
    return {"X": X}
