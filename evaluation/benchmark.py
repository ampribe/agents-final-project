"""Code to benchmark solvers."""

import time
from dataclasses import dataclass


class TimeoutError(Exception):
    """Raised when solver exceeds timeout."""
    pass


@dataclass
class BenchmarkResult:
    """Result of benchmarking a solver."""
    time_ms: float
    result: any


def benchmark_solver(solve_fn, problem, warmup=True, timeout_seconds=None) -> BenchmarkResult:
    """
    Benchmark a solver function with simple timing.

    Args:
        solve_fn: Function to benchmark
        problem: Problem instance to solve
        warmup: Whether to run a warmup before timing (recommended to avoid cache effects)
        timeout_seconds: Maximum time allowed per run (uses signal.alarm - works for most Python code)

    Returns:
        BenchmarkResult with time in milliseconds and result

    Raises:
        TimeoutError: If solver exceeds timeout_seconds
        Exception: Any exception raised by solve_fn
    """
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Solver exceeded {timeout_seconds}s")

    old_handler = None
    if timeout_seconds is not None:
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)

    try:
        if warmup:
            if timeout_seconds is not None:
                signal.alarm(timeout_seconds)
            try:
                _ = solve_fn(problem)
            except Exception as e:
                raise
            finally:
                if timeout_seconds is not None:
                    signal.alarm(0)

        if timeout_seconds is not None:
            signal.alarm(timeout_seconds)

        start = time.perf_counter()
        result = solve_fn(problem)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return BenchmarkResult(elapsed_ms, result)
    finally:
        if timeout_seconds is not None:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)


def run_with_timeout(fn, timeout_seconds, *args, **kwargs):
    """
    Run function with a timeout.
    Args:
        fn: Function to run
        timeout_seconds: Maximum time allowed
        *args, **kwargs: Arguments to pass to fn

    Returns:
        Result of fn(*args, **kwargs)

    Raises:
        TimeoutError: If function exceeds timeout
        Exception: Any exception raised by fn
    """
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation exceeded {timeout_seconds}s")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        return fn(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
