import importlib.util
import json
import sys
import time
import traceback

# Load reference implementation
spec_ref = importlib.util.spec_from_file_location('reference_impl', 'reference_impl.py')
ref_mod = importlib.util.module_from_spec(spec_ref)
spec_ref.loader.exec_module(ref_mod)
reference_solve = ref_mod.solve

# Load solver implementation (solver.py expected)
spec_sol = importlib.util.spec_from_file_location('solver', 'solver.py')
sol_mod = importlib.util.module_from_spec(spec_sol)
spec_sol.loader.exec_module(sol_mod)
# Solver may expose solve directly or via a class instance
if hasattr(sol_mod, 'solve'):
    solver_solve = sol_mod.solve
else:
    # try to find Solver class
    Solver = getattr(sol_mod, 'Solver', None)
    if Solver is None:
        raise RuntimeError('Solver module does not provide solve function')
    solver_solve = Solver().solve

# Helper to compare solutions
def compare_solutions(ref, cand):
    # both are dicts with 'mst_edges'
    if 'mst_edges' not in cand:
        return False, "Missing 'mst_edges' key"
    ref_edges = ref.get('mst_edges')
    cand_edges = cand.get('mst_edges')
    if len(ref_edges) != len(cand_edges):
        return False, f"Edge count mismatch: expected {len(ref_edges)} got {len(cand_edges)}"
    if ref_edges != cand_edges:
        return False, f"Edge list differs.\nRef: {ref_edges}\nCand: {cand_edges}"
    return True, ''

# Test case generator functions
def test_single_node():
    return {"num_nodes": 1, "edges": []}

def test_duplicate_and_zero_negative():
    # Graph with 4 nodes, duplicate edges, zero and negative weights
    return {
        "num_nodes": 4,
        "edges": [
            [0, 1, 0.0],
            [0, 1, 2.5],  # duplicate higher weight
            [1, 2, -1.2],
            [2, 3, 3.3],
            [3, 0, 0.0],
            [1, 3, 1.1],
            [2, 0, -0.5]
        ]
    }

def test_large_random():
    import random
    random.seed(42)
    n = 120
    edges = []
    # generate a connected graph first by chain
    for i in range(n-1):
        w = random.uniform(0.1, 10.0)
        edges.append([i, i+1, w])
    # add extra random edges
    extra = n * 3
    for _ in range(extra):
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        w = random.uniform(0.1, 10.0)
        edges.append([u, v, w])
    return {"num_nodes": n, "edges": edges}

tests = [
    ("single_node", test_single_node()),
    ("duplicate_zero_negative", test_duplicate_and_zero_negative()),
    ("large_random", test_large_random()),
]

all_pass = True
for name, prob in tests:
    print(f"Running test: {name}")
    try:
        start = time.time()
        ref_sol = reference_solve(prob)
        ref_time = time.time() - start
        start = time.time()
        cand_sol = solver_solve(prob)
        cand_time = time.time() - start
        ok, msg = compare_solutions(ref_sol, cand_sol)
        if ok:
            print(f"  PASS (ref {ref_time:.4f}s, solver {cand_time:.4f}s)")
        else:
            print(f"  FAIL: {msg}")
            all_pass = False
    except Exception as e:
        print(f"  EXCEPTION: {e}\n{traceback.format_exc()}")
        all_pass = False

if all_pass:
    print("All tests passed.")
    sys.exit(0)
else:
    print("Some tests failed.")
    sys.exit(1)
