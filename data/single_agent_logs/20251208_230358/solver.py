import numpy as np
import cvxpy as cp

def solve(problem: dict) -> dict:
    """Solve the LP centering problem.

    The problem dict contains:
        - "c": list of length n
        - "A": list of lists (m x n)
        - "b": list of length m
    Returns a dict with key "solution" mapping to list of n floats.
    """
    # Convert inputs to numpy arrays of float type
    c = np.array(problem["c"], dtype=float)
    A = np.array(problem["A"], dtype=float)
    b = np.array(problem["b"], dtype=float)
    n = c.shape[0]

    # Decision variable
    x = cp.Variable(n)
    # Objective: minimize c^T x - sum(log(x))
    objective = cp.Minimize(c @ x - cp.sum(cp.log(x)))
    # Equality constraints
    constraints = [A @ x == b]
    prob = cp.Problem(objective, constraints)

    # Try to solve with Clarabel if available, otherwise fallback to default solver
    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception:
        prob.solve()

    if prob.status != cp.OPTIMAL:
        raise RuntimeError(f"Problem not solved to optimality, status: {prob.status}")

    solution = x.value.tolist()
    return {"solution": solution}
