import json
import logging
import pathlib
import numpy as np
import scipy.linalg

# Configure logging for both solver and harness use
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ----- Solver utilities -----

def _parse_complex(val):
    """Parse a value that may be a string like '1+2j' or a native complex."""
    if isinstance(val, complex):
        return val
    try:
        # Strip spaces and let Python parse the complex literal
        return complex(val.replace(' ', ''))
    except Exception as e:
        raise ValueError(f"Cannot parse complex value '{val}': {e}")


def _list_to_numpy(mat_list):
    """Convert list-of-lists (strings or numbers) to a numpy complex array."""
    return np.array([[_parse_complex(item) for item in row] for row in mat_list], dtype=complex)


def _complex_to_str(c: complex) -> str:
    """Convert a complex number to a compact string 'a+bj' with limited precision."""
    real = round(c.real, 12)
    imag = round(c.imag, 12)
    sign = '+' if imag >= 0 else '-'
    return f"{real}{sign}{abs(imag)}j"


def _numpy_to_list(mat: np.ndarray) -> list[list[str]]:
    """Convert a numpy array of complex numbers to list-of-lists of strings."""
    return [[_complex_to_str(c) for c in row] for row in mat]

# ----- Solver entry point -----

def solve(problem: dict) -> dict:
    """Compute the principal matrix square root.

    Expected input format:
        {"matrix": [["a+bj", ...], ...]}
    Output format:
        {"sqrtm": {"X": [["c+dj", ...], ...]}}
    """
    if not isinstance(problem, dict) or "matrix" not in problem:
        raise ValueError("Input problem must be a dict with a 'matrix' key")

    A = _list_to_numpy(problem["matrix"])

    try:
        X, _ = scipy.linalg.sqrtm(A, disp=False)
    except Exception as e:
        logging.error(f"scipy.linalg.sqrtm failed: {e}")
        return {"sqrtm": {"X": []}}

    return {"sqrtm": {"X": _numpy_to_list(X)}}

# ----- Harness utilities for testing -----

def _matrix_to_str_list(matrix: np.ndarray) -> list[list[str]]:
    """Helper to convert numpy matrix to list-of-strings (used for generating problems)."""
    return _numpy_to_list(matrix)


def generate_random_problem(seed: int | None = None) -> dict:
    """Generate a random square matrix that has a principal square root.
    The matrix is built as A = X @ X where X is random, guaranteeing a valid sqrt.
    Returns the problem dict with matrix represented as strings.
    """
    rng = np.random.default_rng(seed)
    n = rng.integers(2, 5)  # size between 2 and 4 inclusive
    X = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    A = X @ X
    return {"matrix": _matrix_to_str_list(A)}


def _list_to_numpy_complex(mat_list: list[list[str]]) -> np.ndarray:
    """Convert the list-of-string matrix back to a numpy complex array for validation."""
    return _list_to_numpy(mat_list)


def run_single_test(problem: dict) -> dict:
    """Run the solver on a single problem and verify against the reference implementation.
    Returns a report dictionary.
    """
    A = _list_to_numpy_complex(problem["matrix"])
    # Reference solution using scipy.linalg.sqrtm
    X_ref, _ = scipy.linalg.sqrtm(A, disp=False)
    ref_solution = {"sqrtm": {"X": _numpy_to_list(X_ref)}}

    # Participant's solution (uses the solve function defined in this file)
    try:
        solver_solution = solve(problem)
    except Exception as e:
        logging.error(f"Solver raised an exception: {e}")
        solver_solution = {"error": str(e)}

    passed = False
    if isinstance(solver_solution, dict) and "sqrtm" in solver_solution:
        X_sol_list = solver_solution["sqrtm"].get("X")
        if isinstance(X_sol_list, list):
            # Empty list may indicate failure; let validator handle that
            if len(X_sol_list) == 0:
                # Compare with reference failure (unlikely here)
                passed = False
            else:
                try:
                    X_sol = _list_to_numpy_complex(X_sol_list)
                    if X_sol.shape == A.shape and np.all(np.isfinite(X_sol)):
                        recon = X_sol @ X_sol
                        passed = np.allclose(recon, A, rtol=1e-5, atol=1e-8)
                except Exception as e:
                    logging.error(f"Error during validation of solver output: {e}")
    else:
        logging.error("Solver solution missing required 'sqrtm' key.")

    return {
        "problem": problem,
        "reference": ref_solution,
        "solver": solver_solution,
        "passed": passed,
    }


def main():
    results = []
    for seed in range(5):
        prob = generate_random_problem(seed)
        report = run_single_test(prob)
        results.append(report)
        logging.info(f"Test seed {seed}: passed={report['passed']}")

    out_path = pathlib.Path("results.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results written to {out_path.resolve()}")


if __name__ == "__main__":
    main()
