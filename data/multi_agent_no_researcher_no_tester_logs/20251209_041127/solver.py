import logging
import numpy as np
from scipy.linalg import sqrtm


def _compute_sqrtm(A: np.ndarray) -> np.ndarray:
    """Compute the principal matrix square root of A.

    Handles the case where ``scipy.linalg.sqrtm`` returns either a matrix
    or a tuple ``(matrix, info)`` depending on the SciPy version.
    """
    try:
        result = sqrtm(A)
        if isinstance(result, tuple):
            X = result[0]
        else:
            X = result
        return X
    except Exception as e:
        logging.error(f"scipy.linalg.sqrtm failed: {e}")
        raise


def _solve_impl(problem: dict) -> dict:
    """Core implementation for solving the matrix square root problem.

    Parameters
    ----------
    problem: dict
        Must contain the key ``"matrix"`` with a NumPy ``ndarray`` (complex
        allowed) representing a square matrix.

    Returns
    -------
    dict
        ``{"sqrtm": {"X": <list of lists>}}`` where ``X`` is the principal
        square root of ``A``. If computation fails, ``X`` is an empty list.
    """
    if "matrix" not in problem:
        logging.error("Problem dictionary missing 'matrix' key.")
        return {"sqrtm": {"X": []}}

    A = problem["matrix"]
    # Convert to NumPy array of complex dtype if needed
    if not isinstance(A, np.ndarray):
        try:
            A = np.array(A, dtype=complex)
        except Exception as e:
            logging.error(f"Failed to convert input matrix to ndarray: {e}")
            return {"sqrtm": {"X": []}}

    try:
        X = _compute_sqrtm(A)
    except Exception:
        # Signal failure per reference behaviour
        return {"sqrtm": {"X": []}}

    return {"sqrtm": {"X": X.tolist()}}


class Solver:
    """Optional class wrapper for the solver.

    The evaluation harness may expect either a module-level ``solve`` function
    or an instance with a ``solve`` method. This class delegates to the core
    implementation.
    """

    def solve(self, problem: dict) -> dict:
        return _solve_impl(problem)


# Expose a module-level ``solve`` callable as required by the harness.
solve = Solver().solve
