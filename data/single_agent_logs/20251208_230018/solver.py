import numpy as np
import cvxpy as cp

def solve(problem: dict) -> dict:
    """Solve the Chebyshev center problem.

    Parameters
    ----------
    problem: dict
        Dictionary with keys:
        - "a": list of list of floats, shape (m, n)
        - "b": list of floats, length m

    Returns
    -------
    dict
        Dictionary with key "solution" containing a list of n floats representing the
        Chebyshev center (the optimal x_c). The radius is implicit and validated by the
        `is_solution` method in the validation script.
    """
    # Convert inputs to numpy arrays
    a = np.asarray(problem["a"], dtype=float)
    b = np.asarray(problem["b"], dtype=float)

    # Number of variables (dimension of the space)
    if a.ndim != 2:
        raise ValueError("'a' must be a 2‑dimensional array.")
    m, n = a.shape
    if b.shape != (m,):
        raise ValueError("Length of 'b' must match number of rows in 'a'.")

    # Decision variables: center x (n‑dim) and radius r (scalar, non‑negative)
    x = cp.Variable(n)
    r = cp.Variable(nonneg=True)

    # Constraint: a_i^T x + r * ||a_i||_2 <= b_i for each i
    # cp.norm(a, axis=1) returns a vector of ||a_i|| for each row.
    constraints = [a @ x + r * cp.norm(a, axis=1) <= b]

    # Objective: maximize r
    prob = cp.Problem(cp.Maximize(r), constraints)

    # Solve the problem. ECOS is a lightweight open‑source solver available in the sandbox.
    # If ECOS fails, fall back to SCS.
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        # As a last resort, try the commercial Clarabel if it is installed.
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except Exception:
            raise RuntimeError(f"Solver failed to find an optimal solution. Status: {prob.status}")

    # Extract solution
    x_opt = x.value
    if x_opt is None:
        raise RuntimeError("Solver did not return a solution for x.")

    return {"solution": x_opt.tolist()}
