"""Stub for AlgoTuneTasks.base to avoid dependencies."""

TASK_REGISTRY = {}


def register_task(name: str):
    """Decorator to register a Task subclass."""
    def decorator(cls):
        TASK_REGISTRY[name] = cls
        return cls
    return decorator


class Task:
    """Task base class stub."""

    def generate_problem(self, **kwargs):
        raise NotImplementedError

    def solve(self, problem):
        raise NotImplementedError

    def is_solution(self, problem, solution):
        raise NotImplementedError
