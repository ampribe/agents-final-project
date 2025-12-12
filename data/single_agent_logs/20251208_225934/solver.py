import numpy as np
import cvxpy as cp

def solve(problem):
    """Solve the LP box problem using CVXPY.

    Parameters
    ----------
    problem : dict
        Dictionary with keys "c", "A", "b".

    Returns
    -------
    dict
        Dictionary with key "solution" containing the optimal x as a list.
    """
    c = np.array(problem["c"], dtype=float)
    A = np.array(problem["A"], dtype=float)
    b = np.array(problem["b"], dtype=float)
    n = c.shape[0]
    # Define variable
    x = cp.Variable(n)
    constraints = [A @ x <= b, x >= 0, x <= 1]
    prob = cp.Problem(cp.Minimize(c @ x), constraints)
    # Try Clarabel first, fallback to default solver if unavailable
    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception:
        prob.solve()
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"LP did not solve to optimality, status: {prob.status}")
    return {"solution": x.value.tolist()}
