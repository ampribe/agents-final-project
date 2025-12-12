#!/usr/bin/env python
import importlib.util, time, numpy as np, traceback

# Load reference implementation
ref_spec = importlib.util.spec_from_file_location('reference_impl', 'reference_impl.py')
ref_mod = importlib.util.module_from_spec(ref_spec)
ref_spec.loader.exec_module(ref_mod)
reference_solve = ref_mod.solve

# Load solver implementation (expects top-level solve function)
solver_spec = importlib.util.spec_from_file_location('solver', 'solver.py')
solver_mod = importlib.util.module_from_spec(solver_spec)
solver_spec.loader.exec_module(solver_mod)
if not hasattr(solver_mod, 'solve'):
    for name in dir(solver_mod):
        attr = getattr(solver_mod, name)
        if isinstance(attr, type) and hasattr(attr, 'solve'):
            solver_mod.solve = attr().solve
            break
if not hasattr(solver_mod, 'solve'):
    raise AttributeError('Solver module does not provide a solve function')
solver_solve = solver_mod.solve

def matmul_toeplitz(tup, x):
    c, r = tup
    n = len(c)
    x = np.asarray(x, dtype=complex)
    T = np.empty((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            T[i, j] = c[i - j] if i >= j else r[j - i]
    return T @ x

def is_solution(problem, sol):
    c = np.array(problem['c'], dtype=complex)
    r = np.array(problem['r'], dtype=complex)
    b = np.array(problem['b'], dtype=complex)
    x = np.array(sol, dtype=complex)
    if x.shape != b.shape:
        return False
    if not np.all(np.isfinite(x)):
        return False
    Tx = matmul_toeplitz((c, r), x)
    return np.allclose(Tx, b, atol=1e-6)

# Generators (ensure c[0] == r[0])

def generate_random_real(n, seed):
    rng = np.random.default_rng(seed)
    c = rng.uniform(-5, 5, size=n).tolist()
    r = rng.uniform(-5, 5, size=n).tolist()
    r[0] = c[0]  # enforce Toeplitz consistency
    x_true = rng.uniform(-3, 3, size=n)
    T = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            T[i, j] = c[i - j] if i >= j else r[j - i]
    b = (T @ x_true).tolist()
    return {'c': c, 'r': r, 'b': b}

def generate_random_complex(n, seed):
    rng = np.random.default_rng(seed)
    c = (rng.uniform(-5,5,size=n) + 1j * rng.uniform(-5,5,size=n)).tolist()
    r = (rng.uniform(-5,5,size=n) + 1j * rng.uniform(-5,5,size=n)).tolist()
    r[0] = c[0]
    x_true = (rng.uniform(-3,3,size=n) + 1j * rng.uniform(-3,3,size=n))
    T = np.empty((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            T[i, j] = c[i - j] if i >= j else r[j - i]
    b = (T @ x_true).tolist()
    return {'c': c, 'r': r, 'b': b}

# Build test suite
tests = []
# size 1 edge
tests.append(('size_one_real', generate_random_real(1, seed=0)))
# sizes 2..20 real
for n in range(2, 21):
    tests.append((f'random_real_n{n}', generate_random_real(n, seed=n)))
# sizes 2..20 complex
for n in range(2, 21):
    tests.append((f'random_complex_n{n}', generate_random_complex(n, seed=n+100)))
# large benchmark sizes
tests.append(('large_200', generate_random_real(200, seed=999)))
tests.append(('large_500', generate_random_real(500, seed=1000)))

# Execute tests
for name, prob in tests:
    n = len(prob['c'])
    print(f'=== Test {name} (n={n}) ===')
    # Reference solution
    try:
        t0 = time.perf_counter()
        ref_sol = reference_solve(prob)
        ref_time = time.perf_counter() - t0
        ref_ok = is_solution(prob, ref_sol)
    except Exception as e:
        ref_sol = None
        ref_time = None
        ref_ok = False
        print('Reference error:', e)
        traceback.print_exc()
    # Solver solution
    try:
        t0 = time.perf_counter()
        sol = solver_solve(prob)
        sol_time = time.perf_counter() - t0
        sol_ok = is_solution(prob, sol)
    except Exception as e:
        sol = None
        sol_time = None
        sol_ok = False
        print('Solver error:', e)
        traceback.print_exc()
    print(f'Ref: time {ref_time:.6f}s, ok={ref_ok}')
    if sol_time is None:
        print('Sol: error occurred, ok=False')
    else:
        print(f'Sol: time {sol_time:.6f}s, ok={sol_ok}')
    if not sol_ok:
        print('*** Solver failed validation ***')
    print()

print('All tests completed.')
