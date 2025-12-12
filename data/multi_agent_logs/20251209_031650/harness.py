import sys
import importlib
import numpy as np
import time
import traceback

# Load solver module
try:
    solver_mod = importlib.import_module('solver')
except Exception as e:
    print('Failed to import solver module:', e)
    sys.exit(1)

# Retrieve solve function
if hasattr(solver_mod, 'solve'):
    solve_func = solver_mod.solve
else:
    # try class with solve method
    solver_class = getattr(solver_mod, 'Solver', None)
    if solver_class is not None:
        solve_func = solver_class().solve
    else:
        print('Solver does not expose a solve function or Solver class')
        sys.exit(1)

# Reference implementation
import reference_impl
reference_solve = reference_impl.solve

# Validation logic (copied from description)
import logging
logging.basicConfig(level=logging.ERROR)

def is_solution(problem: dict, solution: dict) -> bool:
    A = problem.get('matrix')
    if A is None:
        logging.error("Problem does not contain 'matrix'.")
        return False
    if 'QR' not in solution:
        logging.error("Solution does not contain 'QR' key.")
        return False
    qr_solution = solution['QR']
    for key in ['Q', 'R']:
        if key not in qr_solution:
            logging.error(f"Solution QR does not contain '{key}' key.")
            return False
    try:
        Q = np.array(qr_solution['Q'], dtype=float)
        R = np.array(qr_solution['R'], dtype=float)
    except Exception as e:
        logging.error(f"Error converting solution lists to numpy arrays: {e}")
        return False
    n = A.shape[0]
    if Q.shape != (n, n) or R.shape != (n, n + 1):
        logging.error("Dimension mismatch between input matrix and QR factors.")
        return False
    if not np.all(np.isfinite(Q)):
        logging.error("Matrix Q contains non-finite values (inf or NaN).")
        return False
    if not np.all(np.isfinite(R)):
        logging.error("Matrix R contains non-finite values (inf or NaN).")
        return False
    if not np.allclose(Q.T @ Q, np.eye(n), atol=1e-6):
        logging.error("Matrix Q does not have orthonormal columns.")
        return False
    # Upper triangular in main n x n block
    for i in range(n):
        for j in range(i):
            if abs(R[i, j]) > 1e-6:
                logging.error("Matrix R is not upper triangular in its main block.")
                return False
    if not np.allclose(A, Q @ R, atol=1e-6):
        logging.error("Reconstructed matrix does not match the original matrix within tolerance.")
        return False
    return True

# Test case generation

def gen_random_matrix(n, seed=None):
    rng = np.random.default_rng(seed)
    return rng.random((n, n+1)) * 20 - 10  # values in [-10,10]

def gen_zero_row_matrix(n, seed=None):
    A = gen_random_matrix(n, seed)
    A[0] = 0.0
    return A

def gen_singular_matrix(n, seed=None):
    A = gen_random_matrix(n, seed)
    if n > 1:
        A[1] = A[0]  # duplicate row => rank deficiency
    return A

test_cases = []
# small sizes 2..5 with random matrices
for n in range(2, 6):
    test_cases.append({"name": f"random_{n}", "matrix": gen_random_matrix(n, seed=n)})
    test_cases.append({"name": f"zero_row_{n}", "matrix": gen_zero_row_matrix(n, seed=n+10)})
    test_cases.append({"name": f"singular_{n}", "matrix": gen_singular_matrix(n, seed=n+20)})
# larger case
large_n = 100
test_cases.append({"name": f"large_{large_n}", "matrix": gen_random_matrix(large_n, seed=1234)})

# Run tests
results = []
for case in test_cases:
    name = case['name']
    A = case['matrix']
    problem = {'matrix': A}
    # reference solution sanity check
    ref_sol = reference_solve(problem)
    if not is_solution(problem, ref_sol):
        print(f"Reference implementation failed validation for case {name}")
        sys.exit(1)
    # solver solution
    start = time.time()
    try:
        sol = solve_func(problem)
    except Exception as e:
        sol = None
        err = traceback.format_exc()
    elapsed = time.time() - start
    ok = False
    if sol is not None:
        ok = is_solution(problem, sol)
    results.append({"name": name, "passed": ok, "time": elapsed, "error": None if ok else err if sol is None else None})

# Report
passed_all = all(r['passed'] for r in results)
print("=== QR Factorization Test Summary ===")
for r in results:
    status = "PASS" if r['passed'] else "FAIL"
    print(f"{r['name']}: {status} (time {r['time']:.4f}s)")
print("Overall result:", "PASS" if passed_all else "FAIL")

sys.exit(0 if passed_all else 1)
