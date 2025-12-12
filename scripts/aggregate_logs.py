#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from log_utils import clean_text, load_json, parse_log_line, detect_agent_type


BASE_COLUMNS = [
    "run_id",
    "timestamp",
    "task",
    "event",
    "output",
    "action",
    "tool",
    "error",
    "message",
]

MULTI_AGENT_COLUMNS = [
    "agent",
    "subagent",
]


def get_columns(agent_type: str) -> List[str]:
    if agent_type == "multi":
        return MULTI_AGENT_COLUMNS + BASE_COLUMNS
    return BASE_COLUMNS


def normalize_row(
    raw: Dict,
    fallback_task: Optional[str],
    run_dir: Path,
    agent_type: str
) -> Optional[Dict]:
    event = raw.get("event")
    if not isinstance(event, str):
        return None

    def get_message() -> str:
        return clean_text(raw.get("message") or raw.get("msg"))

    row = {
        "run_id": clean_text(raw.get("run_id") or run_dir.name),
        "timestamp": clean_text(raw.get("timestamp")),
        "task": clean_text(raw.get("task") or fallback_task),
        "event": event,
        "output": "",
        "action": "",
        "tool": "",
        "error": "",
        "message": "",
    }

    if agent_type == "multi":
        row["agent"] = clean_text(raw.get("agent"))
        row["subagent"] = clean_text(raw.get("subagent"))

    if event == "llm_response":
        row["output"] = clean_text(raw.get("output"))
    elif event in {"tool_start", "tool_call", "start_subagent"}:
        row["action"] = clean_text(raw.get("action"))
        row["tool"] = clean_text(raw.get("tool"))
    elif event in {"tool_result", "tool_return"}:
        row["action"] = clean_text(raw.get("action"))
        row["tool"] = clean_text(raw.get("tool"))
        row["output"] = clean_text(raw.get("output"))
    elif event in {"tool_error", "researcher_mcp_error"}:
        row["tool"] = clean_text(raw.get("tool"))
        row["error"] = clean_text(raw.get("error"))
        row["message"] = get_message()
        if not row["message"]:
            row["message"] = clean_text(raw.get("output"))
    else:
        row["output"] = clean_text(raw.get("output"))
        row["action"] = clean_text(raw.get("action"))
        row["tool"] = clean_text(raw.get("tool"))
        row["error"] = clean_text(raw.get("error"))
        row["message"] = get_message()

    if not row["output"]:
        row["output"] = clean_text(raw.get("result"))

    if agent_type == "multi":
        if not row["agent"] and row["run_id"]:
            parts = row["run_id"].split("_")
            if len(parts) >= 3:
                row["agent"] = parts[2]

    return row


def iter_log_rows(run_dir: Path, agent_type: str) -> Iterable[Dict]:
    log_path = run_dir / "agent_log.jsonl"
    if not log_path.exists():
        return []

    summary = load_json(run_dir / "summary.json") or {}
    fallback_task = summary.get("task")

    rows: List[Dict] = []
    with log_path.open() as f:
        for line in f:
            raw = parse_log_line(line)
            if not raw:
                continue
            row = normalize_row(raw, fallback_task, run_dir, agent_type)
            if row:
                rows.append(row)
    return rows


def collect_rows(runs_dir: Path, agent_type: str) -> List[Dict]:
    rows: List[Dict] = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        rows.extend(iter_log_rows(run_dir, agent_type))
    return rows


def write_csv(rows: List[Dict], path: Path, columns: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate agent_log.jsonl files into a CSV."
    )
    parser.add_argument(
        "--runs-dir",
        required=True,
        help="Directory containing run subdirectories",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: <runs-dir>/aggregated_logs.csv)",
    )
    parser.add_argument(
        "--agent-type",
        choices=["auto", "single", "multi"],
        default="auto",
        help="Agent type (auto-detect by default)",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise SystemExit(f"Runs directory not found: {runs_dir}")

    agent_type = args.agent_type
    if agent_type == "auto":
        agent_type = detect_agent_type(runs_dir)
        print(f"Auto-detected agent type: {agent_type}")

    output_path = Path(args.output) if args.output else runs_dir / "aggregated_logs.csv"
    columns = get_columns(agent_type)
    rows = collect_rows(runs_dir, agent_type)
    write_csv(rows, output_path, columns)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
