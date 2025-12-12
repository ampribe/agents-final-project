import importlib
import logging
import time
import traceback
import numpy as np
import scipy.linalg

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Load solver module (solver.py expected at project root)
try:
    solver_mod = importlib.import_module('solver')
    if hasattr(solver_mod, 'solve'):
        solve_solver = solver_mod.solve
    else:
        SolverCls = getattr(solver_mod, 'Solver', None)
        if SolverCls is not None:
            solve_solver = SolverCls().solve
        else:
            raise AttributeError('Solver module does not expose solve function')
except Exception as e:
    logging.error(f'Failed to import solver: {e}')
    raise

# Load reference implementation
ref_mod = importlib.import_module('reference_impl')
solve_reference = ref_mod.solve

def is_solution(problem, solution):
    A = problem.get('matrix')
    if A is None:
        logging.error('Problem missing matrix')
        return False
    n = A.shape[0]
    if A.shape != (n, n):
        logging.error('Matrix not square')
        return False
    if not isinstance(solution, dict) or 'sqrtm' not in solution:
        logging.error('Solution missing sqrtm')
        return False
    sqrtm_dict = solution['sqrtm']
    if not isinstance(sqrtm_dict, dict) or 'X' not in sqrtm_dict:
        logging.error('Solution missing X')
        return False
    Xlist = sqrtm_dict['X']
    # empty list handling
    if isinstance(Xlist, list) and len(Xlist) == 0:
        try:
            scipy.linalg.sqrtm(A, disp=False)
        except Exception:
            return True
        logging.error('Reference succeeded but solution empty')
        return False
    try:
        X = np.array(Xlist, dtype=complex)
    except Exception:
        logging.error('Cannot convert X to np array')
        return False
    if X.shape != (n, n):
        logging.error(f'Shape mismatch: {X.shape} vs {A.shape}')
        return False
    if not np.all(np.isfinite(X)):
        logging.error('Non-finite values in X')
        return False
    try:
        recon = X @ X
    except Exception as e:
        logging.error(f'Error in multiplication: {e}')
        return False
    if not np.allclose(recon, A, rtol=1e-5, atol=1e-8):
        diff = np.abs(recon - A)
        logging.error(f'X@X not close to A, max diff {np.max(diff)}')
        return False
    return True

def generate_tests():
    tests = []
    rng = np.random.default_rng(1)
    # 2x2 real random matrix
    tests.append({'matrix': rng.random((2, 2))})
    # 3x3 complex random matrix
    tests.append({'matrix': rng.random((3, 3)) + 1j * rng.random((3, 3))})
    # 5x5 random matrix (real)
    tests.append({'matrix': rng.random((5, 5))})
    return tests

def run_test(test_id, problem):
    ref_start = time.time()
    ref_sol = solve_reference(problem)
    ref_time = time.time() - ref_start
    if not is_solution(problem, ref_sol):
        logging.error(f'Reference solution failed validation on test {test_id}')
        return False, ref_time, None
    solver_start = time.time()
    try:
        sol = solve_solver(problem)
    except Exception as e:
        logging.error(f'Solver raised exception on test {test_id}: {e}\n{traceback.format_exc()}')
        return False, ref_time, None
    solver_time = time.time() - solver_start
    ok = is_solution(problem, sol)
    if not ok:
        logging.error(f'Solver solution invalid on test {test_id}')
    else:
        logging.info(f'Test {test_id} passed. Ref time {ref_time:.4f}s, Solver time {solver_time:.4f}s')
    return ok, ref_time, solver_time

def main():
    tests = generate_tests()
    total = len(tests)
    passed = 0
    times = []
    for i, prob in enumerate(tests, 1):
        ok, ref_t, sol_t = run_test(i, prob)
        if ok:
            passed += 1
            if sol_t is not None:
                times.append(sol_t)
    logging.info(f'Passed {passed}/{total} tests.')
    if times:
        avg = sum(times) / len(times)
        logging.info(f'Average solver time (s): {avg:.4f}')

if __name__ == '__main__':
    main()
