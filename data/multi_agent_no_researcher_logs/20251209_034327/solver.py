import sys
import json
import logging
from typing import Any, Dict, List

import numpy as np

logging.basicConfig(level=logging.INFO)

def _validate_matrix(matrix: Any) -> np.ndarray:
    """Validate that the input is a square numeric matrix and convert to ndarray.

    Raises:
        ValueError: If validation fails.
    """
    if not isinstance(matrix, list):
        raise ValueError("'matrix' must be a list of lists.")
    if not matrix:
        raise ValueError("'matrix' must not be empty.")
    n = len(matrix)
    for row in matrix:
        if not isinstance(row, list):
            raise ValueError("Each row of 'matrix' must be a list.")
        if len(row) != n:
            raise ValueError("'matrix' must be square (same number of rows and columns).")
    try:
        arr = np.array(matrix, dtype=float)
    except Exception as e:
        raise ValueError(f"Failed to convert matrix to numeric array: {e}")
    if arr.shape != (n, n):
        raise ValueError("Converted matrix does not have expected square shape.")
    # Optional symmetry check (within tolerance)
    if not np.allclose(arr, arr.T, atol=1e-8):
        raise ValueError("Matrix is not symmetric.")
    return arr


def solve(problem: Dict[str, Any]) -> Dict[str, Dict[str, List[List[float]]]]:
    """Compute the Cholesky factorization of a symmetric positive‑definite matrix.

    Args:
        problem: Dictionary with key ``"matrix"`` containing the matrix as a list of lists.

    Returns:
        A dictionary ``{"Cholesky": {"L": …}}`` where ``L`` is the lower‑triangular factor
        represented as a plain Python list of lists.
    """
    if not isinstance(problem, dict):
        raise TypeError("Input problem must be a dictionary.")
    if "matrix" not in problem:
        raise KeyError("Input dictionary must contain the key 'matrix'.")

    A = _validate_matrix(problem["matrix"])

    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Matrix is not positive‑definite: {e}")

    # Ensure the result is strictly lower‑triangular (numpy returns lower by default)
    L = np.tril(L)
    return {"Cholesky": {"L": L.tolist()}}


def _cli() -> None:
    """Command‑line interface: read JSON from stdin, write result to stdout."""
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON input: {e}")
        sys.exit(1)
    try:
        result = solve(input_data)
    except Exception as e:
        logging.error(f"Error during computation: {e}")
        sys.exit(1)
    json.dump(result, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    _cli()
