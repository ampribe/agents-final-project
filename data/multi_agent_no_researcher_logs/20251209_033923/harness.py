import importlib.util
import sys
import time
import traceback
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

# Load reference implementation
import reference_impl

# Load solver implementation
def load_solver():
    spec = importlib.util.spec_from_file_location('solver_module', 'solver.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Prefer a top-level solve function
    if hasattr(mod, 'solve'):
        return mod.solve
    # Otherwise look for a class with a solve method
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        if isinstance(attr, type) and hasattr(attr, 'solve'):
            try:
                return attr().solve
            except Exception:
                continue
    raise RuntimeError('Solver does not expose a callable solve function')

solver_solve = load_solver()

# Validation function (mirrors the provided is_solution logic)
def verify_solution(problem, solution):
    if not isinstance(solution, dict) or 'laplacian' not in solution:
        return False, "Missing 'laplacian' key"
    Ldict = solution['laplacian']
    for k in ['data', 'indices', 'indptr', 'shape']:
        if k not in Ldict:
            return False, "Missing CSR component '{}'".format(k)
    data = Ldict['data']
    indptr = Ldict['indptr']
    def _is_empty(x):
        if x is None:
            return True
        if isinstance(x, np.ndarray):
            return x.size == 0
        try:
            return len(x) == 0
        except Exception:
            return False
    # Detect empty sentinel
    if _is_empty(data) and _is_empty(indptr):
        try:
            graph = scipy.sparse.csr_matrix((problem['data'], problem['indices'], problem['indptr']), shape=problem['shape'])
            ref = scipy.sparse.csgraph.laplacian(graph, normed=problem['normed'])
            if ref.nnz == 0:
                return True, 'Both reference and solution empty'
            else:
                return False, 'Solution empty but reference non-empty'
        except Exception:
            return True, 'Reference also failed'
    # Reconstruct proposed Laplacian
    try:
        L_prop = scipy.sparse.csr_matrix((Ldict['data'], Ldict['indices'], Ldict['indptr']), shape=tuple(Ldict['shape']))
    except Exception as e:
        return False, 'Failed to reconstruct proposed Laplacian: {}'.format(e)
    if L_prop.shape != tuple(problem['shape']):
        return False, 'Shape mismatch {} vs {}'.format(L_prop.shape, problem['shape'])
    # Reference Laplacian
    try:
        graph = scipy.sparse.csr_matrix((problem['data'], problem['indices'], problem['indptr']), shape=problem['shape'])
        ref = scipy.sparse.csgraph.laplacian(graph, normed=problem['normed'])
        if not isinstance(ref, scipy.sparse.csr_matrix):
            ref = ref.tocsr()
        ref.eliminate_zeros()
    except Exception as e:
        return False, 'Reference computation failed: {}'.format(e)
    # Compare structure
    if not (np.array_equal(L_prop.indices, ref.indices) and np.array_equal(L_prop.indptr, ref.indptr)):
        return False, 'CSR indices/indptr differ'
    # Compare data values
    if not np.allclose(L_prop.data, ref.data, rtol=1e-5, atol=1e-8):
        diff = np.max(np.abs(L_prop.data - ref.data)) if L_prop.data.size else 0.0
        return False, 'Data values differ, max diff {}'.format(diff)
    return True, 'OK'

# Helper to build CSR dict from dense numpy matrix
def csr_from_dense(mat):
    csr = scipy.sparse.csr_matrix(mat)
    return {
        'data': csr.data.tolist(),
        'indices': csr.indices.tolist(),
        'indptr': csr.indptr.tolist(),
        'shape': list(csr.shape),
    }

# Generate test cases
tests = []
# 1. Triangle graph, standard Laplacian
A1 = np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float)
tests.append({'name':'triangle_standard','normed':False, **csr_from_dense(A1)})
# 2. Square with diagonal, weighted, normalized Laplacian
A2 = np.array([[0,2,0,1],[2,0,3,0],[0,3,0,4],[1,0,4,0]], dtype=float)
tests.append({'name':'square_weighted_normed','normed':True, **csr_from_dense(A2)})
# 3. Random sparse graph n=100 with isolated node 0, standard Laplacian
np.random.seed(42)
size = 100
rand = np.random.rand(size, size)
mask = rand < 0.05
A3 = np.where(mask, rand, 0.0)
A3 = np.triu(A3, 1)
A3 = A3 + A3.T
A3[0,:] = 0
A3[:,0] = 0
tests.append({'name':'random_sparse_standard','normed':False, **csr_from_dense(A3)})
# 4. Same graph normalized
tests.append({'name':'random_sparse_normed','normed':True, **csr_from_dense(A3)})

# Run tests
results = []
for prob in tests:
    name = prob['name']
    problem = {k: prob[k] for k in prob if k not in ('name',)}
    # reference solution (timing)
    t0 = time.perf_counter()
    ref_sol = reference_impl.solve(problem)
    t_ref = time.perf_counter() - t0
    # solver solution (timing)
    try:
        t0 = time.perf_counter()
        sol = solver_solve(problem)
        t_sol = time.perf_counter() - t0
    except Exception:
        sol = None
        t_sol = None
        err = traceback.format_exc()
    ok, msg = verify_solution(problem, sol if sol is not None else {})
    results.append({'name':name,'ok':ok,'msg':msg,'solver_time':t_sol,'ref_time':t_ref,'error':err if sol is None else None})

# Print summary
for r in results:
    print('Test {}: ok={}, solver_time={:.6f}s, ref_time={:.6f}s'.format(r['name'], r['ok'], r['solver_time'] if r['solver_time'] is not None else float('nan'), r['ref_time']))
    if not r['ok']:
        print('  Reason:', r['msg'])
        if r['error']:
            print(r['error'])
