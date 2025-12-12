#!/usr/bin/env python3
"""Compare agent performance on a fixed set of tasks.

Outputs a table with columns:
Task | Agent | Success | n-weighted benchmark ms | n-weighted agent ms | n-weighted speedup.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - fallback if PyYAML is unavailable
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
TASK_FILE = ROOT / "evaluation" / "tasks.yaml"

AGENT_SPECS: List[Tuple[str, Path]] = [
    ("SingleAgent", ROOT / "batch_runs_new" / "summary.csv"),
    ("MultiAgent", ROOT / "batch_runs_multi" / "summary.csv"),
    ("MultiAgentNoResearcher", ROOT / "batch_runs_multi_no_researcher" / "summary.csv"),
    ("MultiAgentNoResearcherOrTester", ROOT / "batch_runs_multi_no_researcher_or_tester" / "summary.csv"),
]

OUTPUT_CSV = ROOT / "data" / "agent_comparison.csv"


def load_tasks(task_path: Path) -> List[str]:
    """Load the first ten tasks from the YAML file, tolerating missing PyYAML."""
    if yaml:
        data = yaml.safe_load(task_path.read_text()) or {}
        tasks = data.get("tasks", [])
    else:
        tasks = []
        for line in task_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("- "):
                tasks.append(line[2:].strip())
    return tasks[:10]


def parse_float(val: str) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return math.nan


def load_rows(csv_path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "task": r.get("task"),
                    "n": parse_float(r.get("n")),
                    "reference_time_ms": parse_float(r.get("reference_time_ms")),
                    "agent_time_ms": parse_float(r.get("agent_time_ms")),
                    "correct": str(r.get("correct", "")).strip().lower() == "true",
                }
            )
    return rows


def weighted_average(rows: Iterable[Dict[str, object]], key: str) -> float:
    pairs = [
        (float(r[key]), float(r["n"]))
        for r in rows
        if not math.isnan(float(r.get(key, math.nan))) and not math.isnan(float(r.get("n", math.nan)))
    ]
    if not pairs:
        return math.nan
    total_weight = sum(w for _, w in pairs)
    if total_weight == 0:
        return math.nan
    return sum(v * w for v, w in pairs) / total_weight


def format_number(val: float) -> str:
    if val is None or not math.isfinite(val):
        return "N/A"
    if abs(val) >= 100:
        return f"{val:,.2f}"
    return f"{val:.3f}"


def build_table(tasks: List[str]) -> List[List[str]]:
    agent_rows: Dict[str, List[Dict[str, object]]] = {
        label: load_rows(path) for label, path in AGENT_SPECS
    }

    table_rows: List[List[str]] = []
    for task in tasks:
        for agent_label, _ in AGENT_SPECS:
            task_rows = [r for r in agent_rows[agent_label] if r["task"] == task]
            success = bool(task_rows) and all(
                r.get("correct") and not math.isnan(float(r.get("agent_time_ms", math.nan))) and not math.isnan(float(r.get("reference_time_ms", math.nan)))
                for r in task_rows
            )
            w_ref = weighted_average(task_rows, "reference_time_ms")
            w_agent = weighted_average(task_rows, "agent_time_ms")
            speedup = w_ref / w_agent if math.isfinite(w_ref) and math.isfinite(w_agent) and w_agent != 0 else math.nan

            table_rows.append(
                [
                    task,
                    agent_label,
                    str(bool(success)),
                    format_number(w_ref),
                    format_number(w_agent),
                    format_number(speedup),
                ]
            )
    return table_rows


def print_table(headers: List[str], rows: List[List[str]]) -> None:
    widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    def fmt_row(row: List[str]) -> str:
        return " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


def write_csv(headers: List[str], rows: List[List[str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def main() -> None:
    tasks = load_tasks(TASK_FILE)
    headers = [
        "Task",
        "Agent",
        "Success",
        "n-weighted benchmark ms",
        "n-weighted agent ms",
        "n-weighted speedup",
    ]
    table = build_table(tasks)
    write_csv(headers, table, OUTPUT_CSV)
    print(f"Wrote comparison to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()


