"""Solver evaluation logic."""

import time
from pathlib import Path
from dataclasses import dataclass, field

from evaluation.algotune_loader import patch_algotune_imports, load_task_class, load_solver
from evaluation.benchmark import benchmark_solver, run_with_timeout, TimeoutError
from evaluation.display import print_evaluation_start, print_problem_result, print_summary


@dataclass
class EvaluationResult:
    """Result of evaluating a solver."""
    success: bool
    pass_rate: float = 0.0
    num_correct: int = 0
    num_total: int = 0
    median_speedup: float | None = None
    speedups: list[float] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)
    error: str | None = None


def evaluate_solver(
    solver_path: Path,
    task_name: str,
    algotune_repo: Path,
    num_problems: int = 5,
    timeout_seconds: int = 60
) -> EvaluationResult:
    """
    Evaluate a solver on AlgoTune task.

    Args:
        solver_path: Path to solver.py
        task_name: AlgoTune task name
        algotune_repo: Path to AlgoTune repository
        num_problems: Number of test problems
        timeout_seconds: Maximum time allowed per solver run (warmups + timed runs)

    Returns:
        EvaluationResult with metrics
    """
    print_evaluation_start(solver_path, task_name)

    t0 = time.time()
    patch_algotune_imports()
    print(f"- Patched imports: {time.time()-t0:.2f}s")

    try:
        t0 = time.time()
        task = load_task_class(task_name, algotune_repo)
        print(f"- Loaded task class: {time.time()-t0:.2f}s")

        t0 = time.time()
        solver = load_solver(solver_path)
        print(f"- Loaded solver: {time.time()-t0:.2f}s")
    except Exception as e:
        return EvaluationResult(success=False, error=str(e))

    results = []

    print(f"- Running {num_problems} test problems...\n")

    for i in range(num_problems):
        seed = 42 + i
        n = 50 * (i + 1)
        try:
            t0 = time.time()
            problem = task.generate_problem(n=n, random_seed=seed)
            gen_time = time.time() - t0
            print(f"  Problem {i+1}/{num_problems} (n={n}) - gen: {gen_time:.2f}s")

            # Benchmark reference solver (with warmup)
            reference_bench = None
            try:
                reference_bench = benchmark_solver(
                    task.solve, problem,
                    warmup=True,
                    timeout_seconds=timeout_seconds
                )
                print(f"    Reference: {reference_bench.time_ms:.2f}ms")
            except TimeoutError as e:
                print(f"    Reference timeout: {e}")
                return EvaluationResult(
                    success=False,
                    error=f"Reference solver timeout after {timeout_seconds}s on problem {i+1}/{num_problems} (n={n}). Problem too large."
                )
            except Exception as e:
                print(f"     Reference failed: {type(e).__name__}: {e}")
                reference_bench = None

            agent_bench = None
            try:
                agent_bench = benchmark_solver(
                    solver.solve, problem,
                    warmup=True,
                    timeout_seconds=timeout_seconds
                )
                speedup = (
                    reference_bench.time_ms / agent_bench.time_ms
                    if reference_bench and agent_bench
                    else None
                )
                print(f"    Agent: {agent_bench.time_ms:.2f}ms", end="")
                if speedup is not None:
                    print(f" - {speedup:.2f}x vs reference")
                else:
                    print()
            except TimeoutError as e:
                print(f"    Solver timeout: {e}")
                results.append({
                    "n": n,
                    "seed": seed,
                    "reference_time_ms": reference_bench.time_ms if reference_bench else None,
                    "agent_time_ms": None,
                    "speedup": None,
                    "error": f"Timeout after {timeout_seconds}s",
                    "correct": False
                })
                continue
            except Exception as e:
                print(f"    Solver error: {type(e).__name__}: {e}")
                results.append({
                    "n": n,
                    "seed": seed,
                    "reference_time_ms": reference_bench.time_ms if reference_bench else None,
                    "agent_time_ms": None,
                    "speedup": None,
                    "error": str(e),
                    "correct": False
                })
                continue

            is_correct = False
            val_error = None
            try:
                t0 = time.time()
                is_correct = run_with_timeout(
                    task.is_solution,
                    timeout_seconds=timeout_seconds,
                    problem=problem,
                    solution=agent_bench.result
                )
                val_time = time.time() - t0
                if is_correct:
                    print(f"    Correct! (validation: {val_time:.2f}s)")
                else:
                    print(f"    Incorrect (validation: {val_time:.2f}s)")
            except TimeoutError as e:
                val_time = time.time() - t0
                val_error = f"Validation timeout: {e}"
                print(f"    {val_error}")
                is_correct = False
            except Exception as e:
                val_time = time.time() - t0
                val_error = f"{type(e).__name__}: {e}"
                print(f"    Validation error: {val_error}")
                is_correct = False

            results.append({
                "n": n,
                "seed": seed,
                "reference_time_ms": reference_bench.time_ms if reference_bench else None,
                "agent_time_ms": agent_bench.time_ms if agent_bench else None,
                "speedup": speedup,
                "correct": is_correct,
                "validation_error": val_error
            })

        except Exception as e:
            print(f"  Problem {i+1}/{num_problems}: Error: {e}")
            results.append({"n": n, "seed": seed, "error": str(e), "correct": False})

    num_correct = sum(1 for r in results if r.get("correct", False))
    pass_rate = num_correct / len(results) * 100

    print_summary(num_correct, len(results), [])

    return EvaluationResult(
        success=True,
        pass_rate=pass_rate,
        num_correct=num_correct,
        num_total=len(results),
        median_speedup=None,
        speedups=[],
        results=results
    )
