# harness.py
import importlib.util
import json
import os
import sys
import time
import random

# Load reference implementation
ref_spec = importlib.util.spec_from_file_location('reference_impl', 'reference_impl.py')
ref_mod = importlib.util.module_from_spec(ref_spec)
ref_spec.loader.exec_module(ref_mod)
reference_solve = ref_mod.solve

# Load solver if exists
solver_solve = None
if os.path.exists('solver.py'):
    sol_spec = importlib.util.spec_from_file_location('solver', 'solver.py')
    sol_mod = importlib.util.module_from_spec(sol_spec)
    sol_spec.loader.exec_module(sol_mod)
    if hasattr(sol_mod, 'solve'):
        solver_solve = sol_mod.solve
    elif hasattr(sol_mod, 'Solver'):
        solver_solve = sol_mod.Solver().solve

def validate(problem, matching):
    prop_raw = problem['proposer_prefs']
    recv_raw = problem['receiver_prefs']
    if isinstance(prop_raw, dict):
        n = len(prop_raw)
        proposer_prefs = [prop_raw[i] for i in range(n)]
    else:
        proposer_prefs = list(prop_raw)
        n = len(proposer_prefs)
    if isinstance(recv_raw, dict):
        receiver_prefs = [recv_raw[i] for i in range(n)]
    else:
        receiver_prefs = list(recv_raw)
    if not (isinstance(matching, list) and len(matching) == n):
        return False
    if len(set(matching)) != n or not all(0 <= r < n for r in matching):
        return False
    receiver_to_proposer = [0] * n
    for p, r in enumerate(matching):
        receiver_to_proposer[r] = p
    for p in range(n):
        try:
            p_match_rank = proposer_prefs[p].index(matching[p])
        except ValueError:
            return False
        for better_r in proposer_prefs[p][:p_match_rank]:
            other_p = receiver_to_proposer[better_r]
            r_prefs = receiver_prefs[better_r]
            if r_prefs.index(p) < r_prefs.index(other_p):
                return False
    return True

tests = []

# Example n=3
tests.append({
    'name': 'example_n3',
    'problem': {
        'proposer_prefs': [[0,1,2],[1,0,2],[0,1,2]],
        'receiver_prefs': [[1,0,2],[0,1,2],[0,1,2]]
    }
})

# Edge n=0
tests.append({
    'name': 'edge_n0',
    'problem': {'proposer_prefs': [], 'receiver_prefs': []}
})

# Edge n=1
tests.append({
    'name': 'edge_n1',
    'problem': {'proposer_prefs': [[0]], 'receiver_prefs': [[0]]}
})

# Random small n=5
random.seed(0)
for i in range(5):
    n = 5
    proposer = [random.sample(range(n), n) for _ in range(n)]
    receiver = [random.sample(range(n), n) for _ in range(n)]
    tests.append({
        'name': f'random5_{i}',
        'problem': {'proposer_prefs': proposer, 'receiver_prefs': receiver}
    })

# Performance test n=500
n_big = 500
proposer_big = [random.sample(range(n_big), n_big) for _ in range(n_big)]
receiver_big = [random.sample(range(n_big), n_big) for _ in range(n_big)]
tests.append({
    'name': 'perf_500',
    'problem': {'proposer_prefs': proposer_big, 'receiver_prefs': receiver_big}
})

all_pass = True
for test in tests:
    name = test['name']
    prob = test['problem']
    print(f'Running test {name}...')
    # Reference
    start = time.time()
    ref_result = reference_solve(prob)
    ref_time = time.time() - start
    ref_match = ref_result['matching']
    if not validate(prob, ref_match):
        print(f'  Reference solution invalid for {name}')
        all_pass = False
        continue
    print(f'  Reference time: {ref_time:.4f}s')
    # Solver
    if solver_solve is not None:
        start = time.time()
        try:
            sol_result = solver_solve(prob)
        except Exception as e:
            print(f'  Solver raised exception on {name}: {e}')
            all_pass = False
            continue
        sol_time = time.time() - start
        sol_match = sol_result.get('matching')
        if sol_match is None:
            print(f'  Solver output missing matching for {name}')
            all_pass = False
        elif not validate(prob, sol_match):
            print(f'  Solver produced invalid matching for {name}')
            all_pass = False
        else:
            print(f'  Solver time: {sol_time:.4f}s (valid)')
print('All tests passed' if all_pass else 'Some tests failed')
