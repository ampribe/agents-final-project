import logging
import numpy as np
import scipy.linalg

def _format_complex(z: complex) -> str:
    """Format a complex number as a string with sufficient precision.

    Uses '.12g' formatting for both real and imaginary parts. The sign of the
    imaginary part is always included (e.g., '1.23+4.56j' or '1.23-4.56j').
    """
    # Ensure real and imag are floats (could be numpy scalars)
    real = float(z.real)
    imag = float(z.imag)
    real_str = format(real, ".12g")
    imag_abs_str = format(abs(imag), ".12g")
    sign = "+" if imag >= 0 else "-"
    return f"{real_str}{sign}{imag_abs_str}j"

def solve(problem: dict) -> dict:
    """Solve the matrix square root problem.

    Parameters
    ----------
    problem : dict
        Dictionary with key "matrix" containing a nested list of complex
        numbers (as strings or native Python complex objects).

    Returns
    -------
    dict
        Dictionary with structure {"sqrtm": {"X": formatted_matrix}} where
        formatted_matrix is a nested list of strings representing the principal
        square root matrix.
    """
    # Extract and convert the input matrix to a NumPy array of complex dtype
    try:
        A = np.array(problem["matrix"], dtype=complex)
    except Exception as e:
        logging.error(f"Failed to parse input matrix: {e}")
        return {"sqrtm": {"X": []}}

    # Compute the principal matrix square root using SciPy
    try:
        X, _ = scipy.linalg.sqrtm(A, disp=False)
    except Exception as e:
        logging.error(f"scipy.linalg.sqrtm failed: {e}")
        return {"sqrtm": {"X": []}}

    # Format each element of the resulting matrix as a string
    try:
        formatted = [[_format_complex(z) for z in row] for row in X]
    except Exception as e:
        logging.error(f"Failed to format result matrix: {e}")
        return {"sqrtm": {"X": []}}

    return {"sqrtm": {"X": formatted}}
