"""AlgoTune task and solver loading utilities."""

import sys
import importlib
import importlib.util
from pathlib import Path


def patch_algotune_imports():
    """
    Patch AlgoTune imports to avoid heavy dependencies.
    """
    # Import and inject minimal base module
    from evaluation import minimal_algotune_base
    sys.modules['AlgoTuneTasks.base'] = minimal_algotune_base

    # Stub out AlgoTuner package modules
    stub_modules = [
        'AlgoTuner',
        'AlgoTuner.config',
        'AlgoTuner.config.loader',
        'AlgoTuner.utils',
        'AlgoTuner.utils.serialization',
        'AlgoTuner.utils.streaming_json',
        'AlgoTuner.utils.casting',
        'AlgoTuner.utils.type_inspection',
        'AlgoTuner.utils.isolated_benchmark',
        'AlgoTuner.utils.multiprocessing_utils',
        'AlgoTuner.utils.k_search',
        'AlgoTuner.utils.dataset_manager',
    ]

    for module_name in stub_modules:
        if module_name not in sys.modules:
            stub = type('ModuleStub', (), {
                '__file__': f'<stub:{module_name}>',
                '__name__': module_name,
                '__package__': module_name.rsplit('.', 1)[0] if '.' in module_name else '',
            })()
            sys.modules[module_name] = stub



def load_task_class(task_name: str, algotune_repo: Path):
    """
    Load AlgoTune task class dynamically.

    Args:
        task_name: Task name (e.g., 'feedback_controller_design')
        algotune_repo: Path to AlgoTune repository

    Returns:
        Task class instance
    """
    # Add AlgoTune to path
    sys.path.insert(0, str(algotune_repo))

    # Import task module
    task_module_name = f"AlgoTuneTasks.{task_name}.{task_name}"
    task_module = importlib.import_module(task_module_name)

    # Find Task subclass
    from AlgoTuneTasks.base import Task
    for attr_name in dir(task_module):
        attr = getattr(task_module, attr_name)
        if isinstance(attr, type) and attr_name != 'Task':
            try:
                if issubclass(attr, Task) and attr is not Task:
                    return attr()
            except TypeError:
                continue

    raise ValueError(f"No Task class found in {task_module_name}")


def load_solver(solver_path: Path):
    """
    Load solver module from file.

    Args:
        solver_path: Path to solver.py file

    Returns:
        Loaded solver module
    """
    spec = importlib.util.spec_from_file_location("agent_solver", solver_path)
    solver_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solver_module)
    return solver_module
