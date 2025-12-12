import importlib
import json
import sys
import time
import random
from typing import Any, Dict, List

# Import reference implementation
import reference_impl
ref_solve = reference_impl.solve

# Try to import solver implementation
try:
    solver_module = importlib.import_module('solver')
    # If solver defines a class with solve method, expose as function
    if hasattr(solver_module, 'Solver'):
        solver_instance = solver_module.Solver()
        solve = solver_instance.solve
    elif hasattr(solver_module, 'solve'):
        solve = solver_module.solve
    else:
        raise AttributeError('No solve function or Solver class found')
except Exception as e:
    print('Failed to import solver module:', e)
    solve = None


def is_solution(problem: Dict[str, Any], solution: Dict[str, Any]) -> bool:
    if "matching" not in solution:
        return False
    prop_raw = problem["proposer_prefs"]
    recv_raw = problem["receiver_prefs"]
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

    matching = solution["matching"]
    if not (isinstance(matching, list) and len(matching) == n):
        return False
    if len(set(matching)) != n or not all(0 <= r < n for r in matching):
        return False

    # build inverse map
    proposer_to_receiver = matching
    receiver_to_proposer = [0] * n
    for p, r in enumerate(proposer_to_receiver):
        receiver_to_proposer[r] = p

    # stability check
    for p in range(n):
        current_r = proposer_to_receiver[p]
        # find rank of current partner
        pref_list = proposer_prefs[p]
        try:
            cur_rank = pref_list.index(current_r)
        except ValueError:
            return False
        for better_r in pref_list[:cur_rank]:
            other_p = receiver_to_proposer[better_r]
            r_prefs = receiver_prefs[better_r]
            # if receiver prefers p over its current partner
            if r_prefs.index(p) < r_prefs.index(other_p):
                return False
    return True


def generate_random_instance(n: int, seed: int = None) -> Dict[str, Any]:
    if seed is not None:
        random.seed(seed)
    proposers = []
    receivers = []
    for _ in range(n):
        lst = list(range(n))
        random.shuffle(lst)
        proposers.append(lst.copy())
    for _ in range(n):
        lst = list(range(n))
        random.shuffle(lst)
        receivers.append(lst.copy())
    return {"proposer_prefs": proposers, "receiver_prefs": receivers}


def generate_fixed_instance() -> Dict[str, Any]:
    # Example from description (n=3)
    return {
        "proposer_prefs": [[0,1,2],[1,0,2],[0,1,2]],
        "receiver_prefs": [[1,0,2],[0,1,2],[0,1,2]]
    }


def main():
    test_cases: List[Dict[str, Any]] = []
    # Small sizes
    for n in range(1,6):
        test_cases.append(generate_random_instance(n, seed=n))
    # Fixed example
    test_cases.append(generate_fixed_instance())
    # Larger case
    test_cases.append(generate_random_instance(100, seed=42))

    all_pass = True
    for idx, case in enumerate(test_cases):
        print(f"Test case {idx+1}/{len(test_cases)} (n={len(case['proposer_prefs'])})")
        # Reference solution
        start = time.time()
        ref_sol = ref_solve(case)
        ref_time = time.time() - start
        if not is_solution(case, ref_sol):
            print("  Reference solution invalid!")
            all_pass = False
            continue
        print(f"  Reference solution time: {ref_time:.4f}s")
        # Solver solution if available
        if solve is not None:
            try:
                start = time.time()
                sol = solve(case)
                sol_time = time.time() - start
                valid = is_solution(case, sol)
                print(f"  Solver solution time: {sol_time:.4f}s, valid: {valid}")
                if not valid:
                    all_pass = False
            except Exception as e:
                print(f"  Solver raised exception: {e}")
                all_pass = False
        else:
            print("  No solver to test.")
    print("All tests passed" if all_pass else "Some tests failed")

if __name__ == "__main__":
    main()
