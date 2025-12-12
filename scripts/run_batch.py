"""Batch runner for evaluating multiple agents across multiple tasks."""

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml

from evaluation.config import import_agent, load_config
from evaluation.environment import setup_venv
from evaluation.evaluator import evaluate_solver


def parse_agent_specs(agent_specs: list[str]) -> dict[str, str]:
    """Convert name=module:Class specs into a dict."""
    parsed: dict[str, str] = {}
    for spec in agent_specs:
        if "=" not in spec:
            raise ValueError("Agent spec must be name=module:Class, e.g., simple=agents.simple_agent:SimpleAgent")
        name, path = spec.split("=", 1)
        parsed[name.strip()] = path.strip()
    return parsed


def load_batch_config(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise SystemExit(f"Batch config not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f) or {}


def load_tasks_file(path: Path | None) -> list[str]:
    """Load tasks from a YAML file with a top-level `tasks` list."""
    if path is None:
        return []
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
    parser = argparse.ArgumentParser(description="Run agents on tasks and evaluate their solvers.")
    parser.add_argument("--batch-config", type=Path, help="YAML file with tasks/agents and defaults")
    parser.add_argument("--tasks", nargs="+", help="Task names, e.g. feedback_controller_design")
    parser.add_argument(
        "--tasks-file",
        type=Path,
        default=Path("evaluation/tasks.yaml"),
        help="YAML file containing a top-level `tasks` list (default: evaluation/tasks.yaml)",
    )
    parser.add_argument("--agents", nargs="+", help="Agent specs as name=module:Class")
    parser.add_argument("--runs-dir", help="Directory to store run outputs")
    parser.add_argument("--model", help="Model name for agents")
    parser.add_argument("--max-steps", type=int, help="Max agent steps")
    parser.add_argument("--num-problems", type=int, help="Number of eval problems per task")
    parser.add_argument("--algotune-repo", help="Path to AlgoTune repository")
    args = parser.parse_args()

    cfg = load_batch_config(args.batch_config)

    def pick(key: str, cli_value, default=None):
        if cli_value is not None:
            return cli_value
        return cfg.get(key, default)

    tasks = pick("tasks", args.tasks)
    if not tasks:
        tasks = load_tasks_file(args.tasks_file)
    if not tasks:
        raise SystemExit("No tasks provided (use --tasks, --tasks-file, or batch config)")

    agents_cfg = pick("agents", args.agents)
    if not agents_cfg:
        raise SystemExit("No agents provided (use --agents or batch config)")

    # Default to a separate directory from the main `runs/` to keep batch outputs isolated.
    runs_root = Path(pick("runs_dir", args.runs_dir, "batch_runs")).resolve()
    model = pick("model", args.model, "openai/gpt-oss-20b")
    max_steps = pick("max_steps", args.max_steps, 30)
    num_problems = pick("num_problems", args.num_problems, 5)
    algotune_repo = Path(pick("algotune_repo", args.algotune_repo, "repos/AlgoTune")).resolve()

    config = load_config()
    runs_root.mkdir(parents=True, exist_ok=True)

    agent_specs = parse_agent_specs(agents_cfg if isinstance(agents_cfg, list) else [f"{k}={v}" for k, v in agents_cfg.items()])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for task in tasks:
        for agent_name, agent_path in agent_specs.items():
            run_id = f"{timestamp}_{task}_{agent_name}"
            run_dir = runs_root / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n=== Running {agent_name} on {task} ({run_id}) ===")

            venv_path = run_dir / ".venv"
            python_path = setup_venv(venv_path, config.get("packages", []))

            agent_cls = import_agent(agent_path)
            agent = agent_cls(model=model, task=task, max_steps=max_steps)

            solver_path = None
            run_error = None
            try:
                solver_path = agent.run(
                    run_dir,
                    task_description="",
                    python_path=python_path,
                    config=config,
                )
            except Exception as exc:  # noqa: BLE001
                run_error = f"agent_error: {exc}"

            eval_result = None
            if solver_path:
                try:
                    eval_result = evaluate_solver(
                        solver_path,
                        task_name=task,
                        algotune_repo=algotune_repo,
                        num_problems=num_problems,
                    )
                except Exception as exc:  # noqa: BLE001
                    run_error = run_error or f"evaluation_error: {exc}"

            # Persist results
            meta = {
                "run_id": run_id,
                "task": task,
                "agent": agent_name,
                "agent_path": agent_path,
                "model": model,
                "max_steps": max_steps,
                "num_problems": num_problems,
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

