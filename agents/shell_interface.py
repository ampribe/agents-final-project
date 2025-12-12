import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple


class ShellInterface:
    """Execute shell commands with timeout and custom command hooks."""

    def __init__(self, workdir: Path, timeout: float = 60.0):
        self.workdir = Path(workdir).resolve()
        self.timeout = timeout
        self._custom_commands: dict[str, Callable[[str, Optional[str]], Tuple[str, int]]] = {}
        # Built-in file writer to avoid brittle printf/heredoc flows from LLMs.
        self.register_custom_command("write_file", self._handle_write_file)

    def run_command(self, cmd: str) -> Tuple[str, int]:
        """
        Execute a shell command with timeout.

        Returns:
            (output, return_code)
        """
        if not cmd.strip():
            return ("Empty command", -1)

        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=self.workdir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=self.timeout,
            )
            return proc.stdout or "", proc.returncode
        except subprocess.TimeoutExpired:
            return (f"Command timed out after {self.timeout} seconds", -1)
        except Exception as exc:
            return (f"Command execution failed: {exc}", -1)

    def register_custom_command(
        self, name: str, handler: Callable[[str, Optional[str]], Tuple[str, int]]
    ) -> None:
        """Register a custom command handler.

        Handlers receive (arg_string, content) where `content` is an optional
        payload from the JSON request (used by write_file).
        """
        self._custom_commands[name] = handler

    def execute(self, cmd: str, content: Optional[str] = None) -> Tuple[str, int]:
        """
        Execute a command, checking for custom commands first.

        Custom handlers receive the arguments string after the command name.
        """
        stripped = cmd.strip()
        if not stripped:
            return ("Empty command", -1)

        try:
            tokens = shlex.split(stripped)
        except ValueError as exc:
            return (f"Command parse error: {exc}", -1)

        if tokens and tokens[0] in self._custom_commands:
            name = tokens[0]
            handler = self._custom_commands[name]
            arg = stripped[len(name) :].strip()
            try:
                return handler(arg, content)
            except Exception as exc:
                return (f"Custom command '{name}' failed: {exc}", -1)

        return self.run_command(stripped)

    # Internal handlers -------------------------------------------------
    def _handle_write_file(self, arg: str, content: Optional[str]) -> Tuple[str, int]:
        """Write `content` to the provided path (relative to workdir)."""
        if not arg:
            return ("Usage: write_file <path>", 1)
        if content is None:
            return ("write_file requires a 'content' field in the JSON payload", 1)

        target = Path(arg)
        if not target.is_absolute():
            target = self.workdir / target
        try:
            target = target.resolve()
            target.relative_to(self.workdir)
        except Exception:
            return ("write_file target must be within the working directory", 1)

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        rel = target.relative_to(self.workdir)
        return (f"Wrote {len(content)} bytes to {rel}", 0)

    # Envelope-based execution ------------------------------------------
    def execute_envelope(self, envelope: Dict[str, Any]) -> Tuple[str, int]:
        """
        Execute from a validated envelope (output of LLMInterface.validate_envelope).
        
        Envelope format:
        {
            "tool": str,
            "args": dict,
            "done": bool,
            "action": str,
            "raw_command": str,
        }
        
        Returns: (output, return_code)
        """
        tool = envelope.get("tool", "")
        args = envelope.get("args", {})
        
        if tool == "none":
            # No-op for done=true with no command
            return ("", 0)
        
        if tool == "write_file":
            path = args.get("path", "")
            content = args.get("content")
            return self._handle_write_file(path, content)
        
        if tool == "run_shell":
            cmd = args.get("cmd", "")
            return self.run_command(cmd)
        
        # Check for registered custom commands (research, coder, etc.)
        if tool in self._custom_commands:
            handler = self._custom_commands[tool]
            # Custom handlers expect (arg_string, content)
            # For new format, reconstruct arg string from args
            if "raw" in args:
                # Coder backward compat
                arg_str = args["raw"]
            elif "query" in args:
                # Research
                arg_str = args["query"]
            else:
                # Generic: pass args as JSON
                import json
                arg_str = json.dumps(args)
            try:
                return handler(arg_str, args.get("content"))
            except Exception as exc:
                return (f"Custom command '{tool}' failed: {exc}", 1)
        
        # Fallback: try to execute raw_command as shell
        raw_cmd = envelope.get("raw_command", "")
        if raw_cmd:
            return self.execute(raw_cmd, args.get("content"))
        
        return (f"Unknown tool: {tool}", 1)

