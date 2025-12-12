'''Test harness for Cholesky factorization.
Runs a suite of test cases against the reference implementation and the candidate solver.
Validates results using the provided validation logic.
'''
import importlib, sys, time, numpy as np, logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Import reference implementation
ref_mod = importlib.import_module('reference_impl')
reference_solve = getattr(ref_mod, 'solve')

# Import solver implementation
try:
    solver_mod = importlib.import_module('solver')
except Exception as e:
    logging.error(f'Failed to import solver module: {e}')
    sys.exit(1)

# Obtain solve function from solver module
if hasattr(solver_mod, 'solve'):
    solver_solve = getattr(solver_mod, 'solve')
elif hasattr(solver_mod, 'Solver'):
    solver_solve = solver_mod.Solver().solve
else:
    logging.error('Solver module does not expose a solve function or Solver class.')
    sys.exit(1)

def validate(problem, solution):
    A = problem.get('matrix')
    if A is None:
        logging.error('Problem does not contain matrix.')
        return False
    if 'Cholesky' not in solution:
        logging.error('Solution does not contain Cholesky key.')
        return False
    ch = solution['Cholesky']
    if 'L' not in ch:
        logging.error('Solution Cholesky does not contain L key.')
        return False
    try:
        L = np.array(ch['L'], dtype=float)
    except Exception as e:
        logging.error(f'Error converting solution list to numpy array: {e}')
        return False
    n = A.shape[0]
    if L.shape != (n, n):
        logging.error('Dimension mismatch between input matrix and Cholesky factor L.')
        return False
    if not np.all(np.isfinite(L)):
        logging.error('Matrix L contains non-finite values (inf or NaN).')
        return False
    if not np.allclose(L, np.tril(L)):
        logging.error('Matrix L is not lower triangular.')
        return False
    if not np.allclose(A, L @ L.T, atol=1e-6):
        logging.error('Reconstructed matrix does not match the original matrix within tolerance.')
        return False
    return True

# Helper to generate a random SPD matrix of size n
def random_spd(n, seed=None):
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, n))
    return M @ M.T + n * np.eye(n)

# Define test cases
test_cases = []
# Small deterministic cases
test_cases.append({'name': '2x2 simple', 'matrix': np.array([[4.0, 2.0], [2.0, 3.0]])})
test_cases.append({'name': '3x3 example', 'matrix': np.array([[6.0, 15.0, 55.0], [15.0, 55.0, 225.0], [55.0, 225.0, 979.0]])})
# Random SPD cases
for n in [5, 10, 30, 50]:
    test_cases.append({'name': f'random_{n}x{n}', 'matrix': random_spd(n, seed=n)})
# Non-SPD case (should be invalid)
test_cases.append({'name': 'non_spd', 'matrix': np.array([[1.0, 2.0], [2.0, 1.0]])})

summary = []
for case in test_cases:
    name = case['name']
    A = case['matrix']
    # Provide matrix as list of lists to accommodate solvers expecting that format
    problem = {'matrix': A.tolist()}
    logging.info(f'\n=== Test case: {name} (size {A.shape[0]}x{A.shape[1]}) ===')
    # Reference solution (handle possible LinAlgError)
    try:
        start = time.time()
        ref_sol = reference_solve(problem)
        ref_time = time.time() - start
        ref_valid = validate({'matrix': A}, ref_sol)
    except Exception as e:
        ref_time = time.time() - start
        ref_valid = False
        logging.error(f'Reference raised exception: {e}')
    logging.info(f'Reference: time {ref_time:.4f}s, valid={ref_valid}')
    # Solver solution
    try:
        start = time.time()
        sol = solver_solve(problem)
        sol_time = time.time() - start
        sol_valid = validate({'matrix': A}, sol)
        logging.info(f'Solver:    time {sol_time:.4f}s, valid={sol_valid}')
    except Exception as e:
        sol_time = None
        sol_valid = False
        logging.error(f'Solver raised exception: {e}')
    summary.append({'name': name, 'ref_valid': ref_valid, 'solver_valid': sol_valid,
                    'ref_time': ref_time, 'solver_time': sol_time})

passed = [s for s in summary if s['solver_valid']]
failed = [s for s in summary if not s['solver_valid']]
logging.info('\n=== Summary ===')
logging.info(f'Total tests: {len(summary)}')
logging.info(f'Passed: {len(passed)}')
logging.info(f'Failed: {len(failed)}')
if failed:
    logging.info('Failed test names: ' + ', '.join([s['name'] for s in failed]))
