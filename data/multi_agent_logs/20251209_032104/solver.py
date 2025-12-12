import numpy as np
import scipy.linalg
import logging

def _to_complex(val):
    """Convert a JSON value to a Python complex number.
    Handles strings like "1+2j" and numeric types.
    """
    if isinstance(val, str):
        # Remove any surrounding whitespace
        try:
            return complex(val.replace(' ', ''))
        except Exception as e:
            logging.error(f"Failed to parse complex string '{val}': {e}")
            raise
    else:
        # Directly cast numeric types to complex
        return complex(val)

def _format_complex(c: complex) -> str:
    """Format a complex number as "{real:.2f}{+/-}{abs(imag):.2f}j".
    Ensures the sign is explicitly shown.
    """
    real_part = c.real
    imag_part = c.imag
    sign = '+' if imag_part >= 0 else '-'
    return f"{real_part:.2f}{sign}{abs(imag_part):.2f}j"

def solve(problem: dict) -> dict:
    """Compute the principal matrix square root of a square matrix.

    Parameters
    ----------
    problem: dict
        Dictionary with key "matrix" containing a list of lists of complex
        numbers (or strings representing complex numbers).

    Returns
    -------
    dict
        Dictionary of the form {"sqrtm": {"X": [[...], ...]}} where each
        entry is a formatted string of the resulting complex number.
    """
    if not isinstance(problem, dict) or "matrix" not in problem:
        raise ValueError("Problem dictionary must contain a 'matrix' key.")

    raw_matrix = problem["matrix"]
    # Convert to a NumPy array of dtype complex128
    try:
        A = np.array([[_to_complex(v) for v in row] for row in raw_matrix], dtype=np.complex128)
    except Exception as e:
        logging.error(f"Failed to convert input matrix to complex NumPy array: {e}")
        raise

    # Compute the principal matrix square root using SciPy
    try:
        X, _ = scipy.linalg.sqrtm(A, disp=False)
    except Exception as e:
        logging.error(f"scipy.linalg.sqrtm failed: {e}")
        raise

    # Ensure the result is a NumPy array of complex numbers
    X = np.array(X, dtype=np.complex128)

    # Format each element as a string with two decimal places
    formatted_X = [[_format_complex(val) for val in row] for row in X.tolist()]

    return {"sqrtm": {"X": formatted_X}}

# If the module is executed directly, provide a simple demo
if __name__ == "__main__":
    example = {
        "matrix": [
            ["5+1j", "4+0j"],
            ["1+0j", "2+1j"]
        ]
    }
    print(solve(example))
