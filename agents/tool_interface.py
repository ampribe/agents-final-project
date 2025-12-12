import json
from pathlib import Path

from evaluation.logging import Logger

from agents.shell_interface import ShellInterface


def _shorten(value: str | None, max_len: int = 160) -> str:
    """Compact long text for log previews."""
    if value is None:
        return ""
    text = " ".join(str(value).split())
    if len(text) <= max_len:
        return text
    keep = max_len // 2
    return f"{text[:keep]}...{text[-keep:]}"


indicate_completion_tool = {
    "type": "function",
    "function": {
        "name": "indicate_completion",
        "description": "Indicate that the task is complete.",
        # Explicit empty-arg schema so the model can emit a proper
        # function call instead of plain text.
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
}

class RunShellTool:
    definition = {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Execute a shell command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "Shell command to execute.",
                    },
                    "action": {
                        "type": "string",
                        "description": "One sentence summary of what you're doing.",
                    },
                },
                "required": ["cmd", "action"],
            },
        }
    }
    def __init__(self, workdir: Path, logger: Logger):
        self.workdir = workdir
        self.logger = logger
        self.shell = ShellInterface(workdir, timeout=60.0)

    def execute(self, tool_call, step_num: int) -> str:
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as exc:
            self.logger.log_event(
                event="tool_error",
                step=step_num,
                subagent="new_agent",
                error="invalid_json",
                extra={"tool": "run_shell", "message": str(exc)},
            )
            return f"Error: Invalid JSON in run_shell arguments ({exc})"
        if not isinstance(args, dict):
            self.logger.log_event(
                event="tool_error",
                step=step_num,
                subagent="new_agent",
                error="invalid_args_type",
                extra={"tool": "run_shell", "message": str(type(args))},
            )
            return "Error: run_shell arguments must be a JSON object"
        cmd = args.get("cmd")
        action = args.get("action")
        if not cmd:
            self.logger.log_event(
                event="tool_error",
                step=step_num,
                subagent="new_agent",
                error="must_include_cmd",
                extra={"tool": "run_shell"},
            )
            return "Error: Must include cmd in call to run_shell"
        elif not action:
            self.logger.log_event(event="tool_error", step=step_num, subagent="new_agent", error="must_include_action", extra={"tool": "run_shell"})
            return "Error: Must include action in call to run_shell"
        action_display = f"[run_shell] {action} | cmd={_shorten(cmd, 140)}"
        self.logger.log_event(
            event="tool_start",
            step=step_num,
            subagent="new_agent",
            action=action_display,
            extra={"tool": "run_shell", "cmd": cmd},
        )
        output, return_code = self.shell.run_command(cmd)
        return output + f" (exit_code={return_code})"

class WriteFileTool:
    definition = {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Existing content will be overwritten.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to write content to. Enclosing directories must exist.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to file.",
                    },
                    "action": {
                        "type": "string",
                        "description": "One sentence summary of what you're doing.",
                    },
                },
                "required": ["path", "content", "action"],
            },
        }
    }

    def __init__(self, workdir: Path, logger: Logger):
        self.workdir = workdir
        self.logger = logger

    def execute(self, tool_call, step_num: int) -> str:
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as exc:
            self.logger.log_event(
                event="tool_error",
                step=step_num,
                subagent="new_agent",
                error="invalid_json",
                extra={"tool": "write_file", "message": str(exc)},
            )
            return f"Error: Invalid JSON in write_file arguments ({exc})"
        if not isinstance(args, dict):
            self.logger.log_event(
                event="tool_error",
                step=step_num,
                subagent="new_agent",
                error="invalid_args_type",
                extra={"tool": "write_file", "message": str(type(args))},
            )
            return "Error: write_file arguments must be a JSON object"
        path = args.get("path")
        content_arg = args.get("content")
        action = args.get("action")
        if not path:
            self.logger.log_event(event="tool_error", step=step_num, subagent="new_agent", error="must_include_path", extra={"tool": "write_file"})
            content = "Error: Must include path in call to write_file"
        elif not content_arg:
            self.logger.log_event(event="tool_error", step=step_num, subagent="new_agent", error="must_include_content", extra={"tool": "write_file"})
            content = "Error: Must include content in call to write_file"
        elif not action:
            self.logger.log_event(event="tool_error", step=step_num, subagent="new_agent", error="must_include_action", extra={"tool": "write_file"})
            content = "Error: Must include action in call to write_file"
        else:
            action_display = f"[write_file] {action} | path={_shorten(path, 80)}"
            self.logger.log_event(
                event="tool_start",
                step=step_num,
                subagent="new_agent",
                action=action_display,
                extra={"tool": "write_file", "path": path},
            )
            target = Path(path)
            if not target.is_absolute():
                target = self.workdir / target
            try:
                target = target.resolve()
                target.relative_to(self.workdir)
                target.parent.mkdir(parents=True, exist_ok=True)
                content = content_arg
                target.write_text(content_arg)
                rel = target.relative_to(self.workdir)
                content = f"Wrote {len(content)} bytes to {rel}"
            except Exception as exc:
                content = f"Error writing file: {exc}"
        return content

