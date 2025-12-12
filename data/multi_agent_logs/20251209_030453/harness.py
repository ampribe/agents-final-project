import importlib.util
import json
import time
import sys
import traceback
from pathlib import Path

# Load reference implementation
ref_spec = importlib.util.spec_from_file_location("reference_impl", "reference_impl.py")
ref_mod = importlib.util.module_from_spec(ref_spec)
ref_spec.loader.exec_module(ref_mod)
reference_solve = ref_mod.solve

# Load solver implementation (could be a function or class with solve method)
solver_path = Path("solver.py")
if not solver_path.is_file():
    print("Solver file solver.py not found", file=sys.stderr)
    sys.exit(1)
solver_spec = importlib.util.spec_from_file_location("solver_mod", str(solver_path))
solver_mod = importlib.util.module_from_spec(solver_spec)
solver_spec.loader.exec_module(solver_mod)
# Determine callable solve
if hasattr(solver_mod, "solve"):
    solver_solve = solver_mod.solve
else:
    # try to find a class with solve method
    solver_solve = None
    for attr_name in dir(solver_mod):
        attr = getattr(solver_mod, attr_name)
        if isinstance(attr, type):  # class
            if hasattr(attr, "solve"):
                solver_solve = attr().solve
                break
    if solver_solve is None:
        print("Solver does not expose a solve function", file=sys.stderr)
        sys.exit(1)

# Helper to run a single test case
def run_test(test_id, problem):
    try:
        ref_start = time.time()
        ref_out = reference_solve(problem)
        ref_time = time.time() - ref_start
    except Exception as e:
        print(f"[FAIL] Test {test_id}: Reference implementation raised exception: {e}")
        traceback.print_exc()
        return False
    try:
        solver_start = time.time()
        sol_out = solver_solve(problem)
        solver_time = time.time() - solver_start
    except Exception as e:
        print(f"[FAIL] Test {test_id}: Solver raised exception: {e}")
        traceback.print_exc()
        return False
    # Compare outputs directly
    if sol_out != ref_out:
        print(f"[FAIL] Test {test_id}: Output mismatch")
        print("Expected:", json.dumps(ref_out, sort_keys=True))
        print("Got     :", json.dumps(sol_out, sort_keys=True))
        return False
    print(f"[PASS] Test {test_id}: ref_time={ref_time:.4f}s solver_time={solver_time:.4f}s")
    return True

# Generate test cases
tests = []
# 1. Small graph
tests.append({
    "id": "small",
    "problem": {
        "num_nodes": 5,
        "edges": [
            [0, 1, 1.2],
            [0, 2, 2.3],
            [1, 2, 1.0],
            [2, 3, 3.4],
            [1, 4, 0.9]
        ]
    }
})
# 2. Dense graph (complete graph) with 8 nodes
import random
random.seed(0)
num_nodes_dense = 8
edges_dense = []
for i in range(num_nodes_dense):
    for j in range(i+1, num_nodes_dense):
        w = round(random.uniform(0.1, 10.0), 2)
        edges_dense.append([i, j, w])
tests.append({
    "id": "dense",
    "problem": {"num_nodes": num_nodes_dense, "edges": edges_dense}
})
# 3. Large sparse graph (n≈120, ~2n edges)
num_nodes_large = 120
edges_large = []
random.seed(1)
# ensure connectivity via a random spanning tree first
parents = list(range(num_nodes_large))
for i in range(1, num_nodes_large):
    p = random.randint(0, i-1)
    w = round(random.uniform(0.1, 20.0), 2)
    edges_large.append([i, p, w])
# add extra random edges (~n more)
extra_edges = num_nodes_large
while len(edges_large) < num_nodes_large + extra_edges:
    u = random.randint(0, num_nodes_large-1)
    v = random.randint(0, num_nodes_large-1)
    if u == v:
        continue
    if u > v:
        u, v = v, u
    # avoid duplicate
    if any(e[0]==u and e[1]==v for e in edges_large):
        continue
    w = round(random.uniform(0.1, 20.0), 2)
    edges_large.append([u, v, w])
tests.append({
    "id": "large",
    "problem": {"num_nodes": num_nodes_large, "edges": edges_large}
})

all_pass = True
for t in tests:
    ok = run_test(t["id"], t["problem"])
    if not ok:
        all_pass = False

if all_pass:
    print("All tests passed.")
    sys.exit(0)
else:
    print("Some tests failed.")
    sys.exit(1)
