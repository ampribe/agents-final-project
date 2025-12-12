import importlib.util
import json
import sys
import time
import traceback
import numpy as np
import os

# Load reference implementation
ref_spec = importlib.util.spec_from_file_location("reference_impl", os.path.abspath("reference_impl.py"))
ref_mod = importlib.util.module_from_spec(ref_spec)
ref_spec.loader.exec_module(ref_mod)

# Try to load solver implementation
solver_mod = None
if os.path.exists("solver.py"):
    try:
        spec = importlib.util.spec_from_file_location("solver", os.path.abspath("solver.py"))
        solver_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(solver_mod)
        # Ensure solve function exists
        if not hasattr(solver_mod, "solve"):
            # maybe class Solver with method solve
            if hasattr(solver_mod, "Solver"):
                solver_mod.solve = solver_mod.Solver().solve
    except Exception as e:
        print("Failed to load solver.py:", e)
        solver_mod = None

# Validation logic copied from description

def validate_solution(A: np.ndarray, solution: dict) -> bool:
    if "QR" not in solution:
        return False
    qr = solution["QR"]
    if "Q" not in qr or "R" not in qr:
        return False
    try:
        Q = np.array(qr["Q"], dtype=float)
        R = np.array(qr["R"], dtype=float)
    except Exception:
        return False
    n = A.shape[0]
    if Q.shape != (n, n) or R.shape != (n, n+1):
        return False
    if not np.all(np.isfinite(Q)) or not np.all(np.isfinite(R)):
        return False
    if not np.allclose(Q.T @ Q, np.eye(n), atol=1e-6):
        return False
    # Upper triangular check for main n x n block
    for i in range(n):
        for j in range(i):
            if abs(R[i, j]) > 1e-6:
                return False
    if not np.allclose(A, Q @ R, atol=1e-6):
        return False
    return True

# Test cases generation
np.random.seed(0)

def generate_cases():
    cases = []
    # 1. Simple deterministic case
    A1 = np.array([[1.0, 2.0, 3.0],[4.0,5.0,6.0]])
    cases.append(("simple", A1))
    # 2. Rank-deficient case (zero row)
    A2 = np.zeros((3,4))
    A2[0] = [1,2,3,4]
    A2[1] = [2,4,6,8]
    cases.append(("rank_def", A2))
    # 3. Random medium case 5x6
    A3 = np.random.randn(5,6)
    cases.append(("random_medium", A3))
    # 4. Large case 50x51
    A4 = np.random.randn(50,51)
    cases.append(("large_50", A4))
    return cases

cases = generate_cases()

results = []
for name, A in cases:
    problem = {"matrix": A.tolist()}
    # Reference solution
    start = time.time()
    ref_sol = ref_mod.solve(problem)
    ref_time = time.time() - start
    ref_ok = validate_solution(A, ref_sol)
    # Solver solution (if available)
    solver_ok = None
    solver_time = None
    if solver_mod:
        try:
            start = time.time()
            sol = solver_mod.solve(problem)
            solver_time = time.time() - start
            solver_ok = validate_solution(A, sol)
        except Exception as e:
            solver_ok = False
            solver_time = None
            print(f"Solver exception on {name}: {e}")
            traceback.print_exc()
    results.append({
        "case": name,
        "ref_ok": ref_ok,
        "ref_time": ref_time,
        "solver_ok": solver_ok,
        "solver_time": solver_time,
    })

# Print summary
print(json.dumps(results, indent=2))
