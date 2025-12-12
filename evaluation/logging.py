import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class Logger:
    """JSONL logger for agent trajectories and evaluation results."""

    def __init__(
        self,
        agent: str,
        task: str,
        run_dir: Path | None = None,
        run_id: str | None = None,
        echo: bool = False,
    ):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id or f"{self.timestamp}_{agent}_{task}"
        self.agent = agent
        self.task = task
        self.run_dir = run_dir or Path(".")
        self.logs: list[Dict[str, Any]] = []
        self.evaluation_result: Optional[dict] = None
        self.echo = echo
        self.start_time = datetime.now()
        self.step_start_times: dict[int, datetime] = {}
        self.tool_start_times: dict[Tuple[str | None, str | None, int | None], datetime] = {}

        if self.echo:
            run_dir_display = str(self.run_dir.resolve())
            print(
                f"[log] start agent={self.agent} task={self.task} run_dir={run_dir_display}"
            )

    def log_event(
        self,
        *,
        event: str,
        step: int | None = None,
        subagent: str | None = None,
        action: str | None = None,
        output: str | None = None,
        result: str | None = None,
        done: bool = False,
        error: str | None = None,
        extra: Dict[str, Any] | None = None,
    ):
        now = datetime.now()
        wall_ms = int((now - self.start_time).total_seconds() * 1000)
        entry = {
            "timestamp": now.isoformat(),
            "run_id": self.run_id,
            "agent": self.agent,
            "task": self.task,
            "event": event,
            "step": step,
            "subagent": subagent,
            "action": action,
            "output": output,
            "result": result,
            "done": done,
            "error": error,
            "wall_ms": wall_ms,
        }

        if event == "step_start" and step is not None:
            self.step_start_times[step] = now
        if step is not None and step in self.step_start_times:
            entry["step_ms"] = int((now - self.step_start_times[step]).total_seconds() * 1000)

        key = (subagent, action, step)
        if event == "tool_start":
            self.tool_start_times[key] = now
        if event in {"tool_result", "tool_return", "tool_error"}:
            start_ts = self.tool_start_times.get(key)
            if start_ts:
                entry["tool_ms"] = int((now - start_ts).total_seconds() * 1000)
        if event.startswith("tool_"):
            entry.setdefault("cwd", str(Path.cwd()))
            entry.setdefault("workdir", str(self.run_dir.resolve()))

        if extra:
            entry.update(extra)

        entry.setdefault("phase", extra.get("phase") if extra else None or subagent or "coordinator")
        if error:
            entry.setdefault("status", "error")
        elif result and isinstance(result, str) and "exit_code=0" in result:
            entry.setdefault("status", "ok")
        elif done:
            entry.setdefault("status", "ok")
        else:
            entry.setdefault("status", "info")

        entry.setdefault("msg", _shorten(output or action or result or "", 200))

        self.logs.append(entry)

        if self.echo:
            if event == "start_agent" and output:
                header = " | ".join(
                    part
                    for part in [
                        f"{entry['timestamp']}",
                        f"phase={entry.get('phase') or ''}".strip(),
                        f"event={event}",
                        f"done={done}",
                    ]
                    if part
                )
                print(f"[log] {header}")
                print(output)
                return

            preview_parts = [
                f"{now.strftime('%H:%M:%S.%f')[:-3]}",
                f"phase={entry.get('phase')}",
                f"event={event}",
                f"step={step}" if step is not None else "",
                f"status={entry.get('status')}",
                f"wall={wall_ms}ms",
                f"step_ms={entry.get('step_ms')}ms" if entry.get("step_ms") is not None else "",
                f"tool_ms={entry.get('tool_ms')}ms" if entry.get("tool_ms") is not None else "",
                f"action={_shorten(action)}" if action else "",
                f"tool={_shorten(entry.get('tool'))}" if entry.get("tool") else "",
                f"args={_shorten(entry.get('arguments'))}" if entry.get("arguments") else "",
                f"result={_shorten(result)}" if result else "",
                f"error={_shorten(error)}" if error else "",
                f"msg={entry.get('msg')}" if entry.get("msg") else "",
            ]
            preview = " | ".join([p for p in preview_parts if p])
            print(f"[log] {preview}")

    def log_evaluation(self, evaluation_result: dict):
        """Attach evaluation results to be written on serialize."""
        self.evaluation_result = evaluation_result

    def serialize(self) -> Path:
        """Save logs and evaluation results to run directory."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.run_dir / "agent_log.jsonl"
        with open(log_path, "w") as f:
            for log in self.logs:
                f.write(json.dumps(log) + "\n")
            if self.evaluation_result:
                f.write(json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "run_id": self.run_id,
                    "type": "evaluation",
                    "result": self.evaluation_result
                }) + "\n")

        summary_path = self.run_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump({
                "run_id": self.run_id,
                "agent": self.agent,
                "task": self.task,
                "timestamp": self.timestamp,
                "num_steps": len(self.logs),
                "evaluation": self.evaluation_result
            }, f, indent=2)

        return log_path


def _shorten(value: Any, max_len: int = 160) -> str:
    """Compact long text for log previews."""
    if value is None:
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ")
    while "  " in text:
        text = text.replace("  ", " ")
    if len(text) <= max_len:
        return text
    keep = max_len // 2
    return f"{text[:keep]}...{text[-keep:]}"