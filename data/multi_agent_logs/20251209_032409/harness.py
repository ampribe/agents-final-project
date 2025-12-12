import importlib
import json
import math
import sys
import time
import numpy as np

# Import reference implementation
import reference_impl

# Import solver module
try:
    solver_mod = importlib.import_module('solver')
except Exception as e:
    print(f"Failed to import solver module: {e}")
    sys.exit(1)

# Resolve solve function
if hasattr(solver_mod, 'solve'):
    solver_solve = solver_mod.solve
else:
    # Look for a class with a solve method
    solver_solve = None
    for attr in dir(solver_mod):
        obj = getattr(solver_mod, attr)
        if isinstance(obj, type):
            if hasattr(obj, 'solve'):
                try:
                    solver_solve = obj().solve
                    break
                except Exception:
                    continue
    if solver_solve is None:
        print("Solver does not expose a solve function.")
        sys.exit(1)


def build_toeplitz(c, r):
    n = len(c)
    T = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            T[i, j] = c[i - j] if i >= j else r[j - i]
    return T


def matmul_toeplitz(cr, x):
    c, r = cr
    n = len(c)
    # Use convolution via np.dot on shifted indices
    result = np.empty(n, dtype=float)
    for i in range(n):
        # element i: sum_{j=0}^{n-1} T[i,j] * x[j]
        # split based on j <= i and j > i
        s = 0.0
        # j <= i -> use c[i-j]
        if i >= 0:
            idx = np.arange(i + 1)
            s += np.dot(c[i - idx], x[idx])
        # j > i -> use r[j-i]
        if i + 1 < n:
            idx = np.arange(i + 1, n)
            s += np.dot(r[idx - i], x[idx])
        result[i] = s
    return result


def is_solution(problem, sol):
    c = np.array(problem["c"], dtype=float)
    r = np.array(problem["r"], dtype=float)
    b = np.array(problem["b"], dtype=float)
    x = np.array(sol, dtype=float)
    if x.shape != b.shape:
        return False
    if not np.all(np.isfinite(x)):
        return False
    Tx = matmul_toeplitz((c, r), x)
    return np.allclose(Tx, b, atol=1e-6)


def generate_random_problem(n, seed=None):
    rng = np.random.default_rng(seed)
    # generate c and r, ensure Toeplitz matrix is likely invertible
    while True:
        c = rng.normal(size=n).tolist()
        r = rng.normal(size=n).tolist()
        # enforce first element equal
        r[0] = c[0]
        T = build_toeplitz(np.array(c), np.array(r))
        if np.linalg.matrix_rank(T) == n:
            break
    b = rng.normal(size=n).tolist()
    return {"c": c, "r": r, "b": b}


def run_test(problem, test_id):
    # Reference solution
    ref_sol = reference_impl.solve(problem)
    if not is_solution(problem, ref_sol):
        print(f"Test {test_id}: Reference solution invalid.")
        return False
    # Solver solution
    try:
        start = time.perf_counter()
        sol = solver_solve(problem)
        duration = time.perf_counter() - start
    except Exception as e:
        print(f"Test {test_id}: Solver raised exception: {e}")
        return False
    if not is_solution(problem, sol):
        print(f"Test {test_id}: Solver solution incorrect.")
        return False
    print(f"Test {test_id}: PASS (solver time {duration*1e3:.2f} ms)")
    return True


def main():
    np.random.seed(0)
    all_pass = True
    # Basic correctness tests
    sizes = [3, 5, 10]
    for idx, n in enumerate(sizes, 1):
        prob = generate_random_problem(n, seed=idx)
        if not run_test(prob, f"basic-{n}"):
            all_pass = False
    # Performance benchmark
    n = 150
    prob = generate_random_problem(n, seed=999)
    print("Running performance benchmark (n=150)...")
    start_ref = time.perf_counter()
    _ = reference_impl.solve(prob)
    ref_time = time.perf_counter() - start_ref
    start_sol = time.perf_counter()
    try:
        _ = solver_solve(prob)
        sol_time = time.perf_counter() - start_sol
        print(f"Reference solve time: {ref_time:.3f}s, Solver solve time: {sol_time:.3f}s")
        if sol_time > 5 * ref_time:
            print("Warning: Solver slower than 5x reference.")
    except Exception as e:
        print(f"Performance benchmark solver raised exception: {e}")
        all_pass = False
    sys.exit(0 if all_pass else 1)

if __name__ == "__main__":
    main()
