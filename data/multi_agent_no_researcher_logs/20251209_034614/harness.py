#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib
import sys
import time
import traceback
import numpy as np
from typing import List, Dict

import reference_impl

# Load solver module
try:
    solver_module = importlib.import_module('solver')
except Exception as e:
    print('Failed to import solver module:', e)
    sys.exit(1)

if hasattr(solver_module, 'solve'):
    solve_solver = solver_module.solve
elif hasattr(solver_module, 'Solver'):
    solve_solver = solver_module.Solver().solve
else:
    print('Solver module does not expose solve function or Solver class')
    sys.exit(1)

solve_reference = reference_impl.solve


def is_square(matrix) -> bool:
    if not isinstance(matrix, (list, np.ndarray)):
        return False
    arr = np.asarray(matrix)
    return arr.ndim == 2 and arr.shape[0] == arr.shape[1]

def validate_solution(problem: Dict, solution: Dict) -> bool:
    """Validate an LU factorization solution according to the spec."""
    A = problem.get('matrix')
    if A is None:
        print('Problem does not contain "matrix"')
        return False
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        print('Input matrix A must be square.')
        return False
    if 'LU' not in solution:
        print('Solution missing LU key')
        return False
    lu_sol = solution['LU']
    for key in ('P', 'L', 'U'):
        if key not in lu_sol:
            print(f"Solution missing {key} key")
            return False
    try:
        P = np.asarray(lu_sol['P'], dtype=float)
        L = np.asarray(lu_sol['L'], dtype=float)
        U = np.asarray(lu_sol['U'], dtype=float)
    except Exception as e:
        print('Error converting solution matrices to numpy:', e)
        return False
    n = A.shape[0]
    if P.shape != (n, n) or L.shape != (n, n) or U.shape != (n, n):
        print('Dimension mismatch')
        return False
    # Finite entries
    for mat, name in ((P, 'P'), (L, 'L'), (U, 'U')):
        if not np.all(np.isfinite(mat)):
            print(f"{name} contains non-finite entries")
            return False
    atol = 1e-8
    rtol = 1e-6
    I = np.eye(n)
    # Check P permutation matrix
    if not np.all(np.isclose(P, 0.0, atol=atol) | np.isclose(P, 1.0, atol=atol)):
        print('P entries not 0/1')
        return False
    if not (np.allclose(P.sum(axis=1), 1.0, atol=atol) and np.allclose(P.sum(axis=0), 1.0, atol=atol)):
        print('P rows/cols do not sum to 1')
        return False
    if not (np.allclose(P @ P.T, I, rtol=rtol, atol=atol) and np.allclose(P.T @ P, I, rtol=rtol, atol=atol)):
        print('P not orthogonal')
        return False
    # L lower triangular
    if not np.allclose(L, np.tril(L), rtol=rtol, atol=atol):
        print('L not lower triangular')
        return False
    # U upper triangular
    if not np.allclose(U, np.triu(U), rtol=rtol, atol=atol):
        print('U not upper triangular')
        return False
    # Reconstruct
    A_rec = P @ L @ U
    if not np.allclose(A, A_rec, rtol=rtol, atol=1e-6):
        print('Reconstruction mismatch')
        return False
    return True


def generate_tests() -> List[Dict]:
    tests = []
    tests.append({"name": "basic_2x2", "matrix": [[2.0, 3.0], [5.0, 4.0]]})
    tests.append({"name": "singular", "matrix": [[1.0, 2.0], [2.0, 4.0]]})
    tests.append({"name": "non_square", "matrix": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]})
    tests.append({"name": "empty", "matrix": []})
    np.random.seed(0)
    tests.append({"name": "random_5", "matrix": np.random.randn(5,5).tolist()})
    np.random.seed(1)
    tests.append({"name": "random_10", "matrix": np.random.randn(10,10).tolist()})
    np.random.seed(2)
    tests.append({"name": "random_50", "matrix": np.random.randn(50,50).tolist()})
    return tests


def run_test(test):
    name = test['name']
    problem = {'matrix': test['matrix']}
    expected_valid = is_square(test['matrix'])
    results = {'name': name, 'expected_valid': expected_valid,
               'solver_valid': None, 'reference_valid': None,
               'solver_time': None, 'reference_time': None}
    # Solver
    try:
        start = time.time()
        sol = solve_solver(problem)
        results['solver_time'] = time.time() - start
        results['solver_valid'] = validate_solution(problem, sol)
    except Exception as e:
        results['solver_valid'] = False
        results['solver_time'] = None
        results['solver_exception'] = str(e)
    # Reference
    try:
        start = time.time()
        ref = solve_reference(problem)
        results['reference_time'] = time.time() - start
        results['reference_valid'] = validate_solution(problem, ref)
    except Exception as e:
        results['reference_valid'] = False
        results['reference_time'] = None
        results['reference_exception'] = str(e)
    return results


def fmt_time(t):
    return f"{t:.4f}s" if t is not None else "N/A"


def main():
    tests = generate_tests()
    all_pass = True
    for test in tests:
        res = run_test(test)
        print(f"Test {res['name']} (expected valid: {res['expected_valid']}):")
        print(f"  Solver valid: {res['solver_valid']}, time: {fmt_time(res['solver_time'])}")
        print(f"  Reference valid: {res['reference_valid']}, time: {fmt_time(res['reference_time'])}")
        # Check solver against expectation
        if res['expected_valid']:
            if not res['solver_valid']:
                all_pass = False
                print('  Solver failed on valid input.')
        else:
            if res['solver_valid']:
                all_pass = False
                print('  Solver incorrectly succeeded on invalid input.')
        # Reference should be valid only when expected_valid is True
        if res['expected_valid'] and not res['reference_valid']:
            all_pass = False
            print('  Reference validation failed (unexpected).')
        print()
    if all_pass:
        print('All tests passed.')
    else:
        print('Some tests failed.')

if __name__ == '__main__':
    main()
