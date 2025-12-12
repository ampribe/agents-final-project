import importlib.util
import sys
import time
import numpy as np
import json
import traceback

# Load solver module (solver.py) which should expose a top-level solve function
solver_spec = importlib.util.spec_from_file_location("solver", "solver.py")
solver_mod = importlib.util.module_from_spec(solver_spec)
solver_spec.loader.exec_module(solver_mod)
# If the module defines a class with solve method but not a function, expose it
if not hasattr(solver_mod, "solve"):
    # try to find a class with a solve method
    for attr in dir(solver_mod):
        obj = getattr(solver_mod, attr)
        if isinstance(obj, type) and hasattr(obj, "solve"):
            solver_mod.solve = obj().solve
            break

# Load reference implementation
ref_spec = importlib.util.spec_from_file_location("reference_impl", "reference_impl.py")
ref_mod = importlib.util.module_from_spec(ref_spec)
ref_spec.loader.exec_module(ref_mod)


def generate_spd(n, seed=None):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    # Make symmetric positive definite
    M = A @ A.T
    # Add n*I to ensure positive definiteness
    M += n * np.eye(n)
    return M.tolist()


def validate_solution(problem, solution):
    import logging
    A = np.array(problem.get("matrix"))
    if A.size == 0:
        logging.error("Problem does not contain 'matrix'.")
        return False
    if "Cholesky" not in solution:
        logging.error("Solution does not contain 'Cholesky' key.")
        return False
    cholesky_solution = solution["Cholesky"]
    if "L" not in cholesky_solution:
        logging.error("Solution Cholesky does not contain 'L' key.")
        return False
    try:
        L = np.array(cholesky_solution["L"], dtype=float)
    except Exception as e:
        logging.error(f"Error converting solution list to numpy array: {e}")
        return False
    n = A.shape[0]
    if L.shape != (n, n):
        logging.error("Dimension mismatch between input matrix and Cholesky factor L.")
        return False
    if not np.all(np.isfinite(L)):
        logging.error("Matrix L contains non-finite values (inf or NaN).")
        return False
    if not np.allclose(L, np.tril(L)):
        logging.error("Matrix L is not lower triangular.")
        return False
    A_reconstructed = L @ L.T
    if not np.allclose(A, A_reconstructed, atol=1e-6):
        logging.error("Reconstructed matrix does not match the original matrix within tolerance.")
        return False
    return True


def run_test(test_id, matrix, timeout_sec=10):
    problem = {"matrix": matrix}
    # Reference solution (should always be valid)
    ref_start = time.time()
    ref_sol = ref_mod.solve(problem)
    ref_time = time.time() - ref_start
    ref_valid = validate_solution(problem, ref_sol)

    # Solver solution with timeout handling
    solver_sol = None
    solver_time = None
    solver_valid = False
    start = time.time()
    try:
        # Simple timeout enforcement via time check after solve
        solver_sol = solver_mod.solve(problem)
        solver_time = time.time() - start
        if solver_time > timeout_sec:
            raise TimeoutError(f"Solver exceeded timeout of {timeout_sec}s")
        solver_valid = validate_solution(problem, solver_sol)
    except Exception as e:
        solver_time = time.time() - start
        print(f"Test {test_id}: Solver raised exception: {e}")
        traceback.print_exc()
        solver_valid = False

    return {
        "id": test_id,
        "size": len(matrix),
        "ref_time": ref_time,
        "ref_valid": ref_valid,
        "solver_time": solver_time,
        "solver_valid": solver_valid,
        "solver_solution": solver_sol,
    }


def main():
    tests = []
    # Typical small test (3x3) from description
    tests.append(("typical-3x3", [
        [6.0, 15.0, 55.0],
        [15.0, 55.0, 225.0],
        [55.0, 225.0, 979.0]
    ]))
    # Edge case 1x1
    tests.append(("edge-1x1", [[4.0]]))
    # Benchmark sizes
    for n, seed in [(10, 0), (100, 1), (200, 2)]:  # 200 to keep runtime reasonable
        mat = generate_spd(n, seed=seed)
        tests.append((f"bench-{n}", mat))

    results = []
    for test_id, mat in tests:
        print(f"Running test {test_id} (size {len(mat)})...")
        res = run_test(test_id, mat, timeout_sec=15)
        results.append(res)
        print(json.dumps({
            "id": res["id"],
            "size": res["size"],
            "solver_valid": res["solver_valid"],
            "solver_time": res["solver_time"],
            "ref_valid": res["ref_valid"],
            "ref_time": res["ref_time"]
        }, indent=2))
        print("---")

    # Summary
    all_pass = all(r["solver_valid"] for r in results)
    print(f"All solver tests passed: {all_pass}")
    if not all_pass:
        failed = [r["id"] for r in results if not r["solver_valid"]]
        print("Failed tests:", failed)
    sys.exit(0 if all_pass else 1)

if __name__ == "__main__":
    main()
