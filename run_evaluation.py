#!/usr/bin/env python3
"""
AlgoTune agent evaluation runner.

Invokes an agent to produce a solver, then evaluates it using AlgoTune task classes.
"""

import argparse
import inspect
import json
import os
import sys
import yaml
from dataclasses import asdict
from pathlib import Path
from datetime import datetime

from evaluation import (
    load_config,
    import_agent,
    setup_venv,
    evaluate_solver,
    patch_algotune_imports,
    load_task_class
)
from evaluation.logging import Logger


def normalize_task_name(task_name: str) -> str:
    """Convert task name from terminal-bench format to AlgoTune format."""
    if task_name.startswith("algotune-"):
        task_name = task_name[9:]
    return task_name.replace("-", "_")


def load_tasks_file(path: Path) -> list[str]:
    """Load tasks from a YAML file with a top-level `tasks` list."""
    if not path.exists():
        raise SystemExit(f"Tasks file not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    tasks = data.get("tasks", [])
    if not tasks:
        raise SystemExit(f"No tasks found in {path}")
    if not isinstance(tasks, list):
        raise SystemExit(f"`tasks` in {path} must be a list")
    return [str(t) for t in tasks]


def main():
    parser = argparse.ArgumentParser(description="AlgoTune agent evaluation")
    parser.add_argument("--agent", required=True, help="Agent path MODULE:CLASS")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--task", help="Single AlgoTune task name (mutually exclusive with --tasks-file)")
    parser.add_argument("--tasks-file", type=Path, help="YAML file with tasks list (mutually exclusive with --task)")
    parser.add_argument("--src-repo", default="AlgoTune", help="AlgoTune repo path")
    parser.add_argument("--runs-dir", default="runs", help="Directory to store run outputs (default: runs)")
    parser.add_argument("--max-steps", type=int, default=20, help="Max agent steps")
    parser.add_argument("--num-problems", type=int, default=3, help="Number of evaluation problems per task")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout per solver run in seconds")
    args = parser.parse_args()

    # Validate mutually exclusive args
    if args.task and args.tasks_file:
        raise SystemExit("Cannot specify both --task and --tasks-file")
    if not args.task and not args.tasks_file:
        raise SystemExit("Must specify either --task or --tasks-file")

    # Get task list
    if args.task:
        tasks = [normalize_task_name(args.task)]
    else:
        tasks = [normalize_task_name(t) for t in load_tasks_file(args.tasks_file)]

    # Get absolute paths before changing directory
    project_root = Path.cwd().resolve()

    # Check source repo (resolve to absolute path)
    src_repo = (project_root / args.src_repo).resolve()
    if not src_repo.exists():
        print(f"Source repo not found: {src_repo}")
        sys.exit(1)

    # Load agent once
    agent_cls = import_agent(args.agent)
    agent_name = agent_cls.__name__

    # Load config once
    config = load_config()
    packages = config.get("packages", [])

    # Patch AlgoTune imports once
    patch_algotune_imports()

    # Run each task
    for task_name in tasks:
        # Load task-specific resources
        task_desc_path = src_repo / "AlgoTuneTasks" / task_name / "description.txt"
        task_description = task_desc_path.read_text() if task_desc_path.exists() else ""

        task_class = load_task_class(task_name, src_repo)
        reference_solve = inspect.getsource(task_class.solve)
        reference_is_solution = inspect.getsource(task_class.is_solution)

        # Create run directory
        runs_dir = (project_root / args.runs_dir).resolve()
        runs_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{timestamp}_{task_name}_{agent_name}"
        run_dir = runs_dir / run_id
        run_dir.mkdir()

        print(f"\n=== Running {agent_name} on {task_name} ({run_id}) ===")

        # Setup venv for agent isolation
        venv_path = run_dir / ".venv"
        setup_venv(venv_path, packages)

        # Change to run directory so agent sees it as root
        os.chdir(run_dir)

        # Create reference files in run directory
        Path("REFERENCE.md").write_text(f"# Reference Implementation\n\n```python\n{reference_solve}\n```\n")
        Path("VALIDATION.md").write_text(f"# Validation Method\n\n```python\n{reference_is_solution}\n```\n")

        # Initialize logger for run directory
        logger = Logger(agent=agent_name, task=task_name, run_dir=run_dir, echo=True)

        # Run agent with relative paths only, sharing the run logger
        agent = agent_cls(
            model=args.model,
            task=task_name,
            max_steps=args.max_steps,
            logger=logger,
        )

        solver_path = None
        run_error = None
        try:
            solver_path = agent.run(
                workdir=Path("."),
                task_description=task_description,
                python_path=Path(".venv/bin/python"),
                config=config
            )
        except Exception as exc:
            run_error = f"agent_error: {exc}"
            print(f"\nAgent error: {exc}")
        finally:
            # Change back to project root
            os.chdir(project_root)
            # Save logs even if agent fails
            logger.serialize()

        if not solver_path or not Path(run_dir / solver_path).exists():
            print("No solver produced")
            continue  # Continue to next task instead of exiting

        print(f"\nSolver created: {solver_path}")

        # Evaluate (no subprocess needed - same dependencies)
        eval_result = None
        try:
            eval_result = evaluate_solver(
                run_dir / solver_path,
                task_name,
                src_repo,
                num_problems=args.num_problems,
                timeout_seconds=args.timeout
            )
        except Exception as exc:
            run_error = run_error or f"evaluation_error: {exc}"
            print(f"\nEvaluation error: {exc}")

        # Persist results
        meta = {
            "run_id": run_id,
            "task": task_name,
            "agent": agent_name,
            "model": args.model,
            "max_steps": args.max_steps,
            "num_problems": args.num_problems,
            "run_dir": str(run_dir),
            "solver_path": str(solver_path) if solver_path else None,
            "error": run_error,
        }

        if eval_result:
            (run_dir / "evaluation_result.json").write_text(
                json.dumps(asdict(eval_result), indent=2)
            )
            meta.update(
                {
                    "pass_rate": eval_result.pass_rate,
                    "num_correct": eval_result.num_correct,
                    "num_total": eval_result.num_total,
                    "median_speedup": eval_result.median_speedup,
                    "success": eval_result.success,
                }
            )

        (run_dir / "summary.json").write_text(json.dumps(meta, indent=2))
        print(f"Saved results to {run_dir}")


if __name__ == "__main__":
    main()
