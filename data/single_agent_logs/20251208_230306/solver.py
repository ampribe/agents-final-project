import logging
import numpy as np
import cvxpy as cp
from typing import Any, Dict, List, Optional

def solve(problem: Dict[str, Any]) -> Optional[Dict[str, List[float]]]:
    """Solve the Markowitz portfolio optimization problem.

    Args:
        problem: Dictionary with keys "μ" (list of expected returns),
                 "Σ" (covariance matrix), and "γ" (risk aversion).

    Returns:
        A dictionary with key "w" mapping to the optimal weight list,
        or ``None`` if the problem could not be solved.
    """
    # Extract data
    try:
        mu = np.asarray(problem["μ"], dtype=float)
        Sigma = np.asarray(problem["Σ"], dtype=float)
        gamma = float(problem["γ"])
    except Exception as e:
        logging.error("Invalid problem input: %s", e)
        return None

    n = mu.size
    # Define variable
    w = cp.Variable(n)
    # Objective: maximize mu^T w - gamma * w^T Sigma w
    # Use psd_wrap to ensure numerical PSD handling
    obj = cp.Maximize(mu @ w - gamma * cp.quad_form(w, cp.psd_wrap(Sigma)))
    # Constraints: sum to 1, non-negative
    constraints = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve()
    except cp.error.SolverError as e:
        logging.error("CVXPY solver error: %s", e)
        return None

    if w.value is None or not np.isfinite(w.value).all():
        logging.warning("No finite solution returned.")
        return None

    # Return as list
    return {"w": w.value.tolist()}
