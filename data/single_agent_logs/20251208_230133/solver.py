# Solver for Water Filling Task
"""
Implements the solve() function to allocate power across channels by
maximizing sum(log(alpha_i + x_i)) subject to sum(x_i)=P_total, x_i>=0.
The implementation mirrors the reference solution.
"""

import logging
from typing import Any, Dict

import cvxpy as cp
import numpy as np


def solve(problem: Dict[str, Any]) -> Dict[str, Any]:
    """Solve the water‑filling convex optimization problem.

    Parameters
    ----------
    problem: dict
        A dictionary with keys:
        - "alpha": list of positive floats (channel parameters)
        - "P_total": positive float (total power budget)

    Returns
    -------
    dict
        A dictionary with keys:
        - "x": list of optimal power allocations
        - "Capacity": the maximized total capacity (sum of logs)
    """
    # ------------------------------------------------------------------
    # Extract data and perform basic validation.  The reference solution
    # returns NaNs when the input is invalid – we follow the same contract.
    # ------------------------------------------------------------------
    alpha = np.asarray(problem.get("alpha", []), dtype=float)
    P_total = float(problem.get("P_total", 0))
    n = alpha.size

    if n == 0 or P_total <= 0 or not np.all(alpha > 0):
        logging.error("Invalid problem data.")
        return {"x": [float("nan")] * n, "Capacity": float("nan")}

    # ------------------------------------------------------------------
    # Formulate the CVXPY problem.  The objective is concave (log) so we
    # maximise it directly.  The equality constraint forces the total power
    # budget, and the variable is constrained to be non‑negative.
    # ------------------------------------------------------------------
    x_var = cp.Variable(n, nonneg=True)
    objective = cp.Maximize(cp.sum(cp.log(alpha + x_var)))
    constraints = [cp.sum(x_var) == P_total]
    prob = cp.Problem(objective, constraints)

    try:
        prob.solve()
    except cp.SolverError as e:
        logging.error(f"CVXPY solver error: {e}")
        return {"x": [float("nan")] * n, "Capacity": float("nan")}

    # ------------------------------------------------------------------
    # Retrieve the solution.  If the solver did not report an optimal status
    # we still try to use whatever values are available – this mirrors the
    # reference implementation which prefers returning a feasible solution.
    # ------------------------------------------------------------------
    if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x_var.value is None:
        logging.warning(f"Solver status: {prob.status}")
        x_val = x_var.value if x_var.value is not None else np.full(n, float("nan"))
        reported_capacity = prob.value if prob.value is not None else float("nan")
        if np.any(np.isnan(x_val)):
            return {"x": [float("nan")] * n, "Capacity": float("nan")}
    else:
        x_val = x_var.value
        reported_capacity = prob.value

    # ------------------------------------------------------------------
    # Rescale the solution to satisfy the power budget exactly.  Numerical
    # errors from the solver can leave the sum slightly off, so we adjust.
    # After scaling we clip tiny negative values that may appear due to
    # rounding, and if clipping changes the total we perform a second
    # scaling.
    # ------------------------------------------------------------------
    current_sum = np.sum(x_val)
    if current_sum > 1e-9:
        scaling_factor = P_total / current_sum
        x_scaled = x_val * scaling_factor
    else:
        x_scaled = x_val

    x_scaled = np.maximum(x_scaled, 0.0)
    final_sum = np.sum(x_scaled)
    if final_sum > 1e-9 and not np.isclose(final_sum, P_total):
        scaling_factor_final = P_total / final_sum
        x_scaled = x_scaled * scaling_factor_final

    # ------------------------------------------------------------------
    # Compute the capacity with the final (scaled) allocations.  If for any
    # reason the log becomes non‑finite (e.g., due to a negative argument
    # caused by numerical noise) we fall back to the capacity reported by the
    # solver.
    # ------------------------------------------------------------------
    safe_x = np.maximum(x_scaled, 0)
    capacity_terms = np.log(alpha + safe_x)
    if not np.all(np.isfinite(capacity_terms)):
        logging.warning(
            "Capacity became non‑finite after scaling x. Using original solver capacity."
        )
        final_capacity = float(reported_capacity) if reported_capacity is not None else float("nan")
    else:
        final_capacity = float(np.sum(capacity_terms))

    return {"x": x_scaled.tolist(), "Capacity": final_capacity}


# Optional: simple manual test when run as a script.
if __name__ == "__main__":
    example = {"alpha": [0.8, 1.0, 1.2], "P_total": 1.0}
    print(solve(example))
