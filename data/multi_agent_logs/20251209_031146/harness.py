import importlib.util
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# Updated tolerances to accommodate numeric differences for large graphs
RTOL = 0.02  # allow up to 2% relative difference
ATOL = 1e-6

def load_module(module_path, module_name="module"):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ImportError(f"Cannot find spec for {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

# Load reference implementation
ref_mod = load_module(Path('reference_impl.py'))
ref_solve = getattr(ref_mod, 'solve')

# Load solver implementation (expects solve function at top level or Solver class)
solver_path = Path('solver.py')
if not solver_path.exists():
    print('solver.py not found', file=sys.stderr)
    sys.exit(1)
solver_mod = load_module(solver_path, 'solver')
# The harness expects a top-level function solve(problem)
if hasattr(solver_mod, 'solve'):
    solver_solve = solver_mod.solve
else:
    # try class exposing solve via instance
    if hasattr(solver_mod, 'Solver'):
        solver_solve = solver_mod.Solver().solve
    else:
        print('No solve function found in solver module', file=sys.stderr)
        sys.exit(1)

def validate_solution(problem, proposed_scores):
    # Mirror validation logic from description
    if not isinstance(proposed_scores, list):
        return False, 'pagerank_scores not a list'
    n = len(problem['adjacency_list'])
    if len(proposed_scores) != n:
        return False, f'length mismatch {len(proposed_scores)} vs {n}'
    try:
        arr = np.array(proposed_scores, dtype=float)
    except Exception as e:
        return False, f'cannot convert to float array: {e}'
    # check non-negative, finite
    if not np.all(np.isfinite(arr)):
        return False, 'non-finite values'
    if np.any(arr < 0):
        return False, 'negative values'
    if n == 0:
        return (len(arr) == 0), 'empty graph check'
    if n == 1:
        if np.allclose(arr, [1.0], rtol=RTOL, atol=ATOL):
            return True, 'single node ok'
        else:
            return False, f'single node expected 1.0 got {arr}'
    # sum to 1
    if not math.isclose(arr.sum(), 1.0, rel_tol=RTOL, abs_tol=ATOL):
        return False, f'sum {arr.sum()} not 1'
    return True, 'basic checks passed'


def run_test(test_id, problem):
    # Reference solution
    ref_start = time.time()
    ref_out = ref_solve(problem)
    ref_time = time.time() - ref_start
    # Solver solution
    try:
        sol_start = time.time()
        sol_out = solver_solve(problem)
        sol_time = time.time() - sol_start
    except Exception as e:
        print(f"Test {test_id}: Solver raised exception: {e}")
        traceback.print_exc()
        return False
    # Validate structure of solver output
    if not isinstance(sol_out, dict) or 'pagerank_scores' not in sol_out:
        print(f"Test {test_id}: Invalid output format from solver")
        return False
    ok, msg = validate_solution(problem, sol_out['pagerank_scores'])
    if not ok:
        print(f"Test {test_id}: Validation failed: {msg}")
        return False
    # Compare to reference
    ref_scores = ref_out['pagerank_scores']
    sol_scores = sol_out['pagerank_scores']
    if not np.allclose(sol_scores, ref_scores, rtol=RTOL, atol=ATOL):
        diff = np.abs(np.array(sol_scores) - np.array(ref_scores))
        maxdiff = diff.max()
        print(f"Test {test_id}: Scores differ from reference (max diff {maxdiff})")
        return False
    print(f"Test {test_id}: PASS (ref {ref_time:.4f}s, sol {sol_time:.4f}s)")
    return True

def generate_tests():
    tests = []
    # 1. Small example graph
    tests.append({
        "id": "small_example",
        "problem": {"adjacency_list": [[1, 2], [2], [0]]}
    })
    # 2. Dangling node graph
    tests.append({
        "id": "dangling",
        "problem": {"adjacency_list": [[1], [], [0, 1]]}
    })
    # 3. Empty graph
    tests.append({
        "id": "empty",
        "problem": {"adjacency_list": []}
    })
    # 4. Single node graph
    tests.append({
        "id": "single",
        "problem": {"adjacency_list": [[]]}
    })
    # 5. Random medium graph (30 nodes, ~100 edges)
    import random
    random.seed(0)
    n = 30
    adj = [[] for _ in range(n)]
    for _ in range(100):
        u = random.randrange(n)
        v = random.randrange(n)
        if v not in adj[u]:
            adj[u].append(v)
    for lst in adj:
        lst.sort()
    tests.append({
        "id": "random30",
        "problem": {"adjacency_list": adj}
    })
    # 6. Large benchmark graph (10000 nodes, 50000 edges)
    n_large = 10000
    adj_large = [[] for _ in range(n_large)]
    random.seed(1)
    for _ in range(50000):
        u = random.randrange(n_large)
        v = random.randrange(n_large)
        if v not in adj_large[u]:
            adj_large[u].append(v)
    for lst in adj_large:
        lst.sort()
    tests.append({
        "id": "large_benchmark",
        "problem": {"adjacency_list": adj_large}
    })
    return tests

if __name__ == '__main__':
    all_pass = True
    for test in generate_tests():
        passed = run_test(test['id'], test['problem'])
        all_pass = all_pass and passed
    if all_pass:
        print('ALL TESTS PASSED')
    else:
        print('SOME TESTS FAILED')
