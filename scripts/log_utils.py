#!/usr/bin/env python3

import json
from pathlib import Path
from typing import Any, Dict, Optional


def clean_text(value: Any, max_len: int = 100) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    text = " ".join(value.strip().split())
    if len(text) <= max_len:
        return text
    keep = max_len // 2
    return f"{text[:keep]}...{text[-keep:]}"


def load_json(path: Path) -> Optional[Dict]:
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return None


def parse_log_line(line: str) -> Optional[Dict]:
    try:
        return json.loads(line)
    except Exception:
        return None


def detect_agent_type(runs_dir: Path) -> str:
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        log_path = run_dir / "agent_log.jsonl"
        if not log_path.exists():
            continue
        with log_path.open() as f:
            for line in f:
                data = parse_log_line(line)
                if data:
                    subagent = data.get("subagent", "")
                    if subagent and subagent not in ["", "new_agent"]:
                        if subagent in ["coordinator", "researcher", "coder", "test_generator"]:
                            return "multi"
                    break
        break
    return "single"
