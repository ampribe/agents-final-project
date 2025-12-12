import importlib
import json
import math
import sys
import time
import traceback
import numpy as np

# Import reference implementation
ref_module = importlib.import_module('reference_impl')
reference_solve = getattr(ref_module, 'solve')

# Import solver implementation
try:
    solver_module = importlib.import_module('solver')
except Exception as e:
    print('Failed to import solver module:', e)
    sys.exit(1)

# Determine solver solve function
if hasattr(solver_module, 'solve'):
    solver_solve = getattr(solver_module, 'solve')
else:
    # Look for a class with a .solve method (fallback)
    solver_solve = None
    for attr_name in dir(solver_module):
        attr = getattr(solver_module, attr_name)
        if isinstance(attr, type):  # it's a class
            if hasattr(attr, 'solve'):
                try:
                    instance = attr()
                    solver_solve = instance.solve
                    break
                except Exception:
                    continue
    if solver_solve is None:
        print('Solver module does not expose a solve function or class with solve method.')
        sys.exit(1)

# Validation parameters (same as validator)
RTOL = 1e-6
ATOL = 1e-6

def validate_solution(problem, proposed_scores, reference_scores):
    # Basic structural checks
    if not isinstance(proposed_scores, list):
        return False, 'proposed scores not a list'
    n = len(problem.get('adjacency_list', []))
    if len(proposed_scores) != n:
        return False, f'length mismatch {len(proposed_scores)} vs {n}'
    # type and non-negative
    for i, s in enumerate(proposed_scores):
        if not isinstance(s, (int, float)):
            return False, f'score {i} not numeric'
        if not math.isfinite(float(s)) or float(s) < 0:
            return False, f'score {i} not finite/non-negative'
    # sum to 1
    if n > 0:
        if not math.isclose(sum(proposed_scores), 1.0, rel_tol=RTOL, abs_tol=ATOL):
            return False, f'sum {sum(proposed_scores)} not close to 1'
    # compare to reference
    if not np.allclose(proposed_scores, reference_scores, rtol=RTOL, atol=ATOL):
        return False, 'values differ from reference'
    return True, ''

def run_test(problem, test_id):
    try:
        ref_solution = reference_solve(problem)
        ref_scores = ref_solution['pagerank_scores']
    except Exception as e:
        print(f'Test {test_id}: Reference solution error: {e}')
        traceback.print_exc()
        return False
    try:
        start = time.time()
        sol = solver_solve(problem)
        duration = time.time() - start
    except Exception as e:
        print(f'Test {test_id}: Solver raised exception: {e}')
        traceback.print_exc()
        return False
    # Ensure solution dict structure
    if not isinstance(sol, dict) or 'pagerank_scores' not in sol:
        print(f'Test {test_id}: Solver returned invalid structure.')
        return False
    proposed = sol['pagerank_scores']
    ok, msg = validate_solution(problem, proposed, ref_scores)
    if ok:
        print(f'Test {test_id}: PASS (time {duration:.4f}s)')
    else:
        print(f'Test {test_id}: FAIL ({msg}) (time {duration:.4f}s)')
    return ok

def main():
    tests = []
    # Test 1: simple triangle
    tests.append({
        "adjacency_list": [
            [1, 2],
            [2],
            [0]
        ]
    })
    # Test 2: dangling and isolated nodes
    tests.append({
        "adjacency_list": [
            [1],    # 0 -> 1
            [2],    # 1 -> 2
            [],     # 2 dangling
            []      # 3 isolated (also dangling)
        ]
    })
    # Test 3: random graph with n=100
    import random
    random.seed(42)
    n = 100
    prob = 0.05
    adj = []
    for i in range(n):
        neighbors = [j for j in range(n) if j != i and random.random() < prob]
        neighbors.sort()
        adj.append(neighbors)
    tests.append({"adjacency_list": adj})

    passed = 0
    for idx, prob in enumerate(tests, 1):
        if run_test(prob, idx):
            passed += 1
    print(f'\nSummary: {passed}/{len(tests)} tests passed.')

if __name__ == '__main__':
    main()
