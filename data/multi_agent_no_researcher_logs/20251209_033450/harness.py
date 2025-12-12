import importlib
import json
import sys
import time
import numpy as np

# Load reference implementation
ref_mod = importlib.import_module('reference_impl')
ref_solve = getattr(ref_mod, 'solve')

# Load solver implementation (assumed at solver.py)
solver_mod = importlib.import_module('solver')
# solver may expose solve directly or via a class instance
if hasattr(solver_mod, 'solve'):
    solver_solve = getattr(solver_mod, 'solve')
else:
    # try to find a class with a solve method
    SolverClass = None
    for name in dir(solver_mod):
        obj = getattr(solver_mod, name)
        if isinstance(obj, type) and hasattr(obj, 'solve'):
            SolverClass = obj
            break
    if SolverClass is None:
        print('No solve function found in solver module.', file=sys.stderr)
        sys.exit(1)
    solver_solve = SolverClass().solve

def build_csr_from_edges(num_nodes, edges):
    """edges: list of (i, j, w) with i<j for undirected graph"""
    rows = []
    cols = []
    data = []
    # Since undirected, add both directions
    for i, j, w in edges:
        rows.append(i)
        cols.append(j)
        data.append(w)
        rows.append(j)
        cols.append(i)
        data.append(w)
    # Build CSR using scipy
    import scipy.sparse
    csr = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    # Extract CSR components
    return {
        'data': csr.data.tolist(),
        'indices': csr.indices.tolist(),
        'indptr': csr.indptr.tolist(),
        'shape': [num_nodes, num_nodes]
    }

def test_case(name, problem, expected=None, benchmark=False):
    print(f'=== Test: {name} ===')
    # Run reference
    ref_start = time.time()
    ref_out = ref_solve(problem)
    ref_time = time.time() - ref_start
    # Run solver
    sol_start = time.time()
    sol_out = solver_solve(problem)
    sol_time = time.time() - sol_start
    # Convert distance matrices to numpy arrays (inf for None)
    def to_array(out):
        dm = out.get('distance_matrix')
        if not dm:
            return np.empty((0,0))
        arr = np.array([[np.inf if v is None else v for v in row] for row in dm], dtype=float)
        return arr
    ref_arr = to_array(ref_out)
    sol_arr = to_array(sol_out)
    # Basic shape check (allow empty for empty graph)
    if ref_arr.shape != sol_arr.shape:
        print('FAIL: shape mismatch')
        return False
    # Check diagonal zeros (skip empty)
    if ref_arr.size > 0 and not np.allclose(np.diag(sol_arr), 0):
        print('FAIL: diagonal not zero')
        return False
    # Check symmetry (undirected) for non-empty
    if ref_arr.size > 0 and not np.allclose(sol_arr, sol_arr.T, equal_nan=True):
        print('FAIL: not symmetric')
        return False
    # Compare values (inf treated as equal)
    ok = np.allclose(sol_arr, ref_arr, rtol=1e-5, atol=1e-8, equal_nan=True)
    if not ok:
        mask = np.isfinite(sol_arr) & np.isfinite(ref_arr)
        if mask.any():
            maxdiff = np.max(np.abs(sol_arr[mask] - ref_arr[mask]))
            print(f'FAIL: values differ, max finite diff {maxdiff}')
        else:
            print('FAIL: values differ (inf mismatches)')
        return False
    print('PASS')
    print(f'  Ref time: {ref_time*1000:.2f} ms')
    print(f'  Solver time: {sol_time*1000:.2f} ms')
    if benchmark:
        print(f'  Benchmark (solver): {sol_time:.4f} s')
    return True

def main():
    all_ok = True
    # 1. Simple triangle graph (3 nodes)
    edges1 = [(0,1,1.0), (1,2,1.0), (0,2,2.0)]
    prob1 = build_csr_from_edges(3, edges1)
    all_ok &= test_case('Triangle (3 nodes)', prob1)

    # 2. Disconnected node (4 nodes, node 3 isolated)
    edges2 = [(0,1,2.0), (1,2,3.0)]
    prob2 = build_csr_from_edges(4, edges2)
    all_ok &= test_case('Disconnected node', prob2)

    # 3. Random sparse 10-node graph
    import random
    random.seed(0)
    edges3 = []
    n3 = 10
    for i in range(n3):
        for j in range(i+1, n3):
            if random.random() < 0.2:  # 20% chance edge
                w = round(random.uniform(1, 10), 2)
                edges3.append((i, j, w))
    prob3 = build_csr_from_edges(n3, edges3)
    all_ok &= test_case('Random 10-node', prob3, benchmark=True)

    # 4. Empty graph (0 nodes)
    prob4 = {'data': [], 'indices': [], 'indptr': [0], 'shape': [0, 0]}
    all_ok &= test_case('Empty graph', prob4)

    # 5. Single-node graph
    prob5 = build_csr_from_edges(1, [])
    all_ok &= test_case('Single node', prob5)

    print('\nOverall result:', 'PASS' if all_ok else 'FAIL')

if __name__ == '__main__':
    main()
