import importlib
import sys
import time
import numpy as np
import scipy.sparse
import logging
from reference_impl import solve as reference_solve

# Load solver module (solver.py)
solver_module = importlib.import_module('solver')
# Determine solve function
if hasattr(solver_module, 'solve'):
    solver_solve = solver_module.solve
else:
    # look for a class with a solve method
    solver_solve = None
    for attr_name in dir(solver_module):
        attr = getattr(solver_module, attr_name)
        if isinstance(attr, type):
            if hasattr(attr, 'solve'):
                solver_solve = attr().solve
                break
    if solver_solve is None:
        raise AttributeError('No solve function found in solver module')


def verify_solution(problem, solution):
    """Validate solution against reference using same logic as validator."""
    # Ensure structure
    if not isinstance(solution, dict) or 'laplacian' not in solution:
        return False, 'Missing laplacian key'
    L_sol = solution['laplacian']
    for k in ['data', 'indices', 'indptr', 'shape']:
        if k not in L_sol:
            return False, f'Missing {k} in laplacian'
    # Reconstruct proposed Laplacian
    try:
        L_csr = scipy.sparse.csr_matrix((L_sol['data'], L_sol['indices'], L_sol['indptr']), shape=L_sol['shape'])
    except Exception as e:
        return False, f'Failed to build csr: {e}'
    # Compute reference
    ref = reference_solve(problem)['laplacian']
    L_ref = scipy.sparse.csr_matrix((ref['data'], ref['indices'], ref['indptr']), shape=ref['shape'])
    # Compare shape
    if L_csr.shape != L_ref.shape:
        return False, f'Shape mismatch {L_csr.shape} vs {L_ref.shape}'
    # Ensure canonical form
    L_csr.eliminate_zeros()
    L_ref.eliminate_zeros()
    # Compare structure
    if not np.array_equal(L_csr.indices, L_ref.indices) or not np.array_equal(L_csr.indptr, L_ref.indptr):
        return False, 'CSR structure differs'
    # Compare data
    if not np.allclose(L_csr.data, L_ref.data, rtol=1e-5, atol=1e-8):
        diff = np.max(np.abs(L_csr.data - L_ref.data)) if L_csr.data.size else 0.0
        return False, f'Data values differ, max diff {diff}'
    return True, 'OK'


def run_test(problem, description, timeout=5.0):
    start = time.time()
    try:
        sol = solver_solve(problem)
    except Exception as e:
        return False, f'Exception {e}', time.time() - start
    elapsed = time.time() - start
    if elapsed > timeout:
        return False, f'Timeout ({elapsed:.2f}s > {timeout}s)', elapsed
    ok, msg = verify_solution(problem, sol)
    return ok, msg, elapsed

def generate_random_csr(n, density, seed=0):
    rng = np.random.default_rng(seed)
    size = n * n
    nnz = int(size * density)
    # generate random positions without replacement
    rows = rng.integers(0, n, size=nnz)
    cols = rng.integers(0, n, size=nnz)
    data = rng.random(nnz) + 0.1  # positive weights
    # Build CSR using scipy for convenience
    mat = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(n, n))
    # make symmetric, zero diagonal
    mat = mat + mat.T
    mat.setdiag(0)
    mat = mat.tocsr()
    mat.eliminate_zeros()
    return {
        'data': mat.data.tolist(),
        'indices': mat.indices.tolist(),
        'indptr': mat.indptr.tolist(),
        'shape': list(mat.shape),
        'normed': False,
    }


def main():
    tests = []
    # 1. Simple triangle graph, unnormed
    problem1 = {
        'data': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        'indices': [1, 0, 2, 1, 0, 2],
        'indptr': [0, 2, 4, 6],
        'shape': [3, 3],
        'normed': False,
    }
    tests.append(('simple_unormed', problem1))
    # 2. Same, normed
    problem2 = dict(problem1)
    problem2['normed'] = True
    tests.append(('simple_normed', problem2))
    # 3. Graph with isolated node
    problem3 = {
        'data': [1.0, 1.0],
        'indices': [1, 0],
        'indptr': [0, 2, 2, 2],
        'shape': [3, 3],
        'normed': False,
    }
    tests.append(('isolated_node', problem3))
    # 4. Weighted graph
    problem4 = {
        'data': [2.0, 3.0, 2.0, 3.0, 4.0, 4.0],
        'indices': [1, 0, 2, 1, 0, 2],
        'indptr': [0, 2, 4, 6],
        'shape': [3, 3],
        'normed': False,
    }
    tests.append(('weighted_unormed', problem4))
    # 5. Larger random graph (~1000 nodes, 0.5% density)
    problem5 = generate_random_csr(1000, 0.005, seed=42)
    tests.append(('large_random', problem5))

    total = len(tests)
    passed = 0
    for name, prob in tests:
        ok, msg, t = run_test(prob, name, timeout=10.0)
        status = 'PASS' if ok else 'FAIL'
        print(f'{name:15s}: {status} (time {t:.3f}s) - {msg}')
        if ok:
            passed += 1
    print(f'\nSummary: {passed}/{total} tests passed.')

if __name__ == '__main__':
    main()
