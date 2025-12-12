import importlib.util
import sys
import time
import numpy as np
import scipy.ndimage
import traceback

# Load reference implementation
ref_spec = importlib.util.spec_from_file_location("reference_impl", "reference_impl.py")
ref_mod = importlib.util.module_from_spec(ref_spec)
ref_spec.loader.exec_module(ref_mod)
ref_solve = ref_mod.solve

# Load solver implementation (solver.py)
solver_path = "solver.py"
solver_spec = importlib.util.spec_from_file_location("solver", solver_path)
solver_mod = importlib.util.module_from_spec(solver_spec)
solver_spec.loader.exec_module(solver_mod)
# Solver may expose a class with a solve method, or a top-level solve function
if hasattr(solver_mod, "solve"):
    solver_solve = solver_mod.solve
else:
    # Attempt to instantiate Solver class
    Solver = getattr(solver_mod, "Solver", None)
    if Solver is None:
        raise RuntimeError("Solver module does not expose solve function or Solver class")
    solver_instance = Solver()
    if not hasattr(solver_instance, "solve"):
        raise RuntimeError("Solver instance has no solve method")
    solver_solve = solver_instance.solve

# Tolerances matching validator
RTOL = 1e-5
ATOL = 1e-7

def compare_outputs(ref, cand):
    # Both should be dict with 'transformed_image'
    if not isinstance(cand, dict) or 'transformed_image' not in cand:
        return False, "Invalid output format"
    cand_arr = np.asarray(cand['transformed_image'], dtype=float)
    ref_arr = np.asarray(ref['transformed_image'], dtype=float)
    if cand_arr.shape != ref_arr.shape:
        return False, f"Shape mismatch {cand_arr.shape} vs {ref_arr.shape}"
    if not np.all(np.isfinite(cand_arr)):
        return False, "Non-finite values in candidate"
    if not np.allclose(cand_arr, ref_arr, rtol=RTOL, atol=ATOL):
        diff = np.abs(cand_arr - ref_arr)
        return False, f"Values differ, max abs error {np.max(diff):.6f}"
    return True, "OK"

# Define test cases

def generate_identity(n):
    img = np.random.rand(n, n) * 255
    matrix = [[1, 0, 0], [0, 1, 0]]
    return {'image': img.tolist(), 'matrix': matrix}

def generate_translation(n, tx, ty):
    img = np.random.rand(n, n) * 255
    matrix = [[1, 0, tx], [0, 1, ty]]
    return {'image': img.tolist(), 'matrix': matrix}

def generate_rotation(n, angle_deg):
    img = np.random.rand(n, n) * 255
    theta = np.deg2rad(angle_deg)
    cos, sin = np.cos(theta), np.sin(theta)
    matrix = [[cos, -sin, 0], [sin, cos, 0]]
    return {'image': img.tolist(), 'matrix': matrix}

def generate_scaling(n, sx, sy):
    img = np.random.rand(n, n) * 255
    matrix = [[sx, 0, 0], [0, sy, 0]]
    return {'image': img.tolist(), 'matrix': matrix}

def generate_random(n):
    img = np.random.rand(n, n) * 255
    # Random affine matrix (including translation)
    a = np.random.randn(2, 3) * 0.5
    # Ensure reasonable scale to keep image within view
    a[:, :2] += np.eye(2) * 0.9
    matrix = a.tolist()
    return {'image': img.tolist(), 'matrix': matrix}

test_cases = []
# Small deterministic cases
test_cases.append(('identity_3', generate_identity(3)))
test_cases.append(('translation_5', generate_translation(5, 1.2, -0.7)))
test_cases.append(('rotation_45', generate_rotation(6, 45)))
test_cases.append(('scaling_4', generate_scaling(4, 1.5, 0.8)))
# Random small
test_cases.append(('random_8', generate_random(8)))
# Larger benchmark case
test_cases.append(('large_256', generate_random(256)))

# Run tests
all_pass = True
for name, prob in test_cases:
    print(f"--- Test: {name} (size {len(prob['image'])}x{len(prob['image'])}) ---")
    # Reference
    start = time.time()
    ref_out = ref_solve(prob)
    ref_time = time.time() - start
    # Solver
    try:
        start = time.time()
        cand_out = solver_solve(prob)
        cand_time = time.time() - start
    except Exception as e:
        print(f"Solver raised exception: {e}\n{traceback.format_exc()}")
        all_pass = False
        continue
    # Compare
    ok, msg = compare_outputs(ref_out, cand_out)
    print(f"Reference time: {ref_time:.4f}s, Solver time: {cand_time:.4f}s")
    print(f"Result: {'PASS' if ok else 'FAIL'} - {msg}\n")
    if not ok:
        all_pass = False

print("ALL PASS" if all_pass else "SOME TESTS FAILED")
