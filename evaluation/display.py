"""Display and formatting utilities."""

import statistics
from dataclasses import dataclass


def print_header(title: str):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")


def print_evaluation_start(solver_path, task_name):
    """Print evaluation start message."""
    print_header(f"Evaluating: {task_name}")
    print(f"  Solver: {solver_path}\n")


def print_problem_result(i, total, n, seed, baseline_ms, agent_ms, correct, speedup=None):
    """Print individual problem result."""
    print(f"  Problem {i+1}/{total} (n={n}, seed={seed}):")
    print(f"    Baseline: {baseline_ms:.2f}ms | Agent: {agent_ms:.2f}ms", end="")
    if correct:
        print(f" | Correct, speedup: {speedup:.2f}x")
    else:
        print(f" | Incorrect")


def print_summary(num_correct, num_total, speedups):
    """Print evaluation summary."""
    print_header("Summary")
    pass_rate = num_correct / num_total * 100
    print(f"  Pass rate: {num_correct}/{num_total} ({pass_rate:.1f}%)")

    if speedups:
        median = statistics.median(speedups)
        mean = statistics.mean(speedups)
        print(f"  Speedup: {median:.2f}x median, {mean:.2f}x mean")
        print(f"  Range: {min(speedups):.2f}x - {max(speedups):.2f}x")
    print()
