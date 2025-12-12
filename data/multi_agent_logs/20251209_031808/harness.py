import importlib
import numpy as np
import time
import sys
import traceback

# Load reference implementation
ref = importlib.import_module('reference_impl')

# Load solver implementation
try:
    solver_mod = importlib.import_module('solver')
except Exception as e:
    print('Failed to import solver module:', e)
    sys.exit(1)

# Determine solve function
if hasattr(solver_mod, 'solve'):
    solver_solve = solver_mod.solve
else:
    # Look for a class named Solver
    SolverClass = getattr(solver_mod, 'Solver', None)
    if SolverClass is None:
        print('Solver does not expose solve function or Solver class.')
        sys.exit(1)
    solver_solve = SolverClass().solve

# Validation function (mirrors validation logic from description)

def is_valid_solution(matrix, solution):
    import numpy as np
    if 'LU' not in solution:
        return False, 'Missing LU key'
    lu_sol = solution['LU']
    for key in ('P', 'L', 'U'):
        if key not in lu_sol:
            return False, f'Missing {key} in LU'
    try:
        P = np.asarray(lu_sol['P'], dtype=float)
        L = np.asarray(lu_sol['L'], dtype=float)
        U = np.asarray(lu_sol['U'], dtype=float)
    except Exception as e:
        return False, f'Conversion error: {e}'
    n = matrix.shape[0]
    if P.shape != (n, n) or L.shape != (n, n) or U.shape != (n, n):
        return False, 'Shape mismatch'
    if not np.all(np.isfinite(P)) or not np.all(np.isfinite(L)) or not np.all(np.isfinite(U)):
        return False, 'Non-finite entries'
    atol = 1e-8
    rtol = 1e-6
    I = np.eye(n)
    # Permutation matrix checks
    if not np.all(np.isclose(P, 0.0, atol=atol) | np.isclose(P, 1.0, atol=atol)):
        return False, 'P not binary'
    if not (np.allclose(P.sum(axis=1), 1.0, atol=atol) and np.allclose(P.sum(axis=0), 1.0, atol=atol)):
        return False, 'P rows/cols not sum to 1'
    if not (np.allclose(P @ P.T, I, rtol=rtol, atol=atol) and np.allclose(P.T @ P, I, rtol=rtol, atol=atol)):
        return False, 'P not orthogonal'
    # Triangular checks
    if not np.allclose(L, np.tril(L), rtol=rtol, atol=atol):
        return False, 'L not lower-triangular'
    if not np.allclose(U, np.triu(U), rtol=rtol, atol=atol):
        return False, 'U not upper-triangular'
    # Reconstruction
    A_rec = P @ L @ U
    if not np.allclose(matrix, A_rec, rtol=rtol, atol=1e-6):
        return False, 'Reconstruction mismatch'
    return True, ''

# Test case generator

def generate_tests():
    tests = []
    # 1. Simple 2x2
    tests.append({
        'name': '2x2 simple',
        'matrix': np.array([[2.0, 3.0], [5.0, 4.0]])
    })
    # 2. Random 5x5
    rng = np.random.default_rng(0)
    tests.append({
        'name': '5x5 random',
        'matrix': rng.random((5,5)) * 10
    })
    # 3. Singular 4x4 (second row duplicate of first)
    A = rng.random((4,4))
    A[1] = A[0]
    tests.append({
        'name': '4x4 singular',
        'matrix': A
    })
    # 4. Larger 100x100 random
    tests.append({
        'name': '100x100 random',
        'matrix': rng.random((100,100)) * 20
    })
    return tests

def run_test(test):
    matrix = test['matrix']
    prob = {'matrix': matrix.tolist()}
    # Reference solution
    start_ref = time.time()
    ref_sol = ref.solve(prob)
    ref_time = time.time() - start_ref
    ref_ok, ref_msg = is_valid_solution(matrix, ref_sol)
    # Solver solution
    start_sol = time.time()
    try:
        sol_sol = solver_solve(prob)
        sol_time = time.time() - start_sol
        sol_ok, sol_msg = is_valid_solution(matrix, sol_sol)
    except Exception as e:
        sol_time = time.time() - start_sol
        sol_ok = False
        sol_msg = f'Exception: {e}\n{traceback.format_exc()}'
    return {
        'name': test['name'],
        'ref_time': ref_time,
        'sol_time': sol_time,
        'ref_ok': ref_ok,
        'sol_ok': sol_ok,
        'ref_msg': ref_msg,
        'sol_msg': sol_msg,
    }

if __name__ == '__main__':
    results = []
    for test in generate_tests():
        print(f"Running test: {test['name']} ...")
        res = run_test(test)
        results.append(res)
        print(f"  Ref: {'PASS' if res['ref_ok'] else 'FAIL'} (time {res['ref_time']:.4f}s)")
        print(f"  Solver: {'PASS' if res['sol_ok'] else 'FAIL'} (time {res['sol_time']:.4f}s)")
        if not res['sol_ok']:
            print('  Solver error:', res['sol_msg'])
    # Summary
    total = len(results)
    passed = sum(r['sol_ok'] for r in results)
    print('\nSummary:')
    print(f"{passed}/{total} solver tests passed.")
