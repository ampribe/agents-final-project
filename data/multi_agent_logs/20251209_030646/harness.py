import time, json, sys, traceback, math
import numpy as np, scipy.sparse

# Import reference implementation
import reference_impl

# Import solver module
try:
    import solver as solver_mod
except Exception as e:
    print('Failed to import solver module:', e)
    sys.exit(1)

# Resolve solve function from solver module
if hasattr(solver_mod, 'solve') and callable(solver_mod.solve):
    solver_solve = solver_mod.solve
else:
    # Attempt to find a class with a solve method
    solver_solve = None
    for name in dir(solver_mod):
        obj = getattr(solver_mod, name)
        if isinstance(obj, type):
            if hasattr(obj, 'solve'):
                try:
                    instance = obj()
                    if callable(instance.solve):
                        solver_solve = instance.solve
                        break
                except Exception:
                    continue
    if solver_solve is None:
        print('Solver does not expose a callable solve function.')
        sys.exit(1)

def compare_matrices(ref, cand, atol=1e-8, rtol=1e-5):
    # Both are list of lists with None for infinity
    if len(ref) != len(cand):
        return False, 'different size'
    n = len(ref)
    for i in range(n):
        if len(ref[i]) != len(cand[i]):
            return False, 'row size mismatch'
        for j in range(n):
            a = ref[i][j]
            b = cand[i][j]
            if a is None and b is None:
                continue
            if (a is None) != (b is None):
                return False, f'Infinity mismatch at ({i},{j})'
            # both numbers
            if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
                return False, f'Value mismatch at ({i},{j}): {a} vs {b}'
    return True, ''

# Test case generators

def test_empty():
    return {"data": [], "indices": [], "indptr": [0], "shape": [0, 0]}

def test_single_node():
    return {"data": [], "indices": [], "indptr": [0,0], "shape": [1,1]}

def test_triangle():
    # 3 nodes fully connected with weight 1
    data = [1,1, 1,1, 1,1]
    indices = [1,2, 0,2, 0,1]
    indptr = [0,2,4,6]
    return {"data": data, "indices": indices, "indptr": indptr, "shape": [3,3]}

def test_disconnected():
    # Two components: 0-1 edge weight 2, 2 isolated
    data = [2,2]
    indices = [1,0]
    indptr = [0,2,2,2]
    return {"data": data, "indices": indices, "indptr": indptr, "shape": [3,3]}

def test_random_sparse(n, density=0.05, seed=None):
    rng = np.random.default_rng(seed)
    # generate random adjacency matrix, make symmetric, zero diagonal
    mat = scipy.sparse.random(n, n, density=density, format='coo', random_state=rng, data_rvs=lambda s: rng.random(s)+0.1)
    mat = mat + mat.T  # make symmetric
    mat.setdiag(0)
    mat = mat.tocsr()
    return {"data": mat.data.tolist(), "indices": mat.indices.tolist(), "indptr": mat.indptr.tolist(), "shape": [n, n]}

tests = [
    ('empty', test_empty()),
    ('single_node', test_single_node()),
    ('triangle', test_triangle()),
    ('disconnected', test_disconnected()),
    ('random_30', test_random_sparse(30, density=0.08, seed=42)),
    ('random_100', test_random_sparse(100, density=0.03, seed=123)),
]

all_pass = True
for name, problem in tests:
    start = time.time()
    try:
        ref_out = reference_impl.solve(problem)
    except Exception as e:
        print(f'[{name}] Reference solve exception:', e)
        traceback.print_exc()
        all_pass = False
        continue
    try:
        cand_out = solver_solve(problem)
    except Exception as e:
        print(f'[{name}] Solver exception:', e)
        traceback.print_exc()
        all_pass = False
        continue
    ok, msg = compare_matrices(ref_out.get('distance_matrix'), cand_out.get('distance_matrix'))
    elapsed = time.time() - start
    if ok:
        print(f'[{name}] PASS ({elapsed:.3f}s)')
    else:
        print(f'[{name}] FAIL ({elapsed:.3f}s): {msg}')
        all_pass = False

if all_pass:
    print('ALL TESTS PASSED')
else:
    print('SOME TESTS FAILED')
