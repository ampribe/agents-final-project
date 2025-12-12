import json
from pathlib import Path

from agents.session import Session
from evaluation.logging import Logger

import agents.tool_interface


def _shorten(text: str | None, max_len: int = 200) -> str:
    """Compact long text for logging."""
    if text is None:
        return ""
    compact = " ".join(str(text).split())
    if len(compact) <= max_len:
        return compact
    keep = max_len // 2
    return f"{compact[:keep]}...{compact[-keep:]}"


def _summarize_tool_call(tool_call) -> str:
    """Readable one-line summary of a tool call."""
    name = getattr(tool_call.function, "name", "unknown")
    raw_args = getattr(tool_call.function, "arguments", "")
    try:
        parsed = json.loads(raw_args)
        arg_display = json.dumps(parsed, ensure_ascii=False)
    except Exception:
        arg_display = raw_args
    return f"{name}({ _shorten(arg_display, 160)})"


def _summarize_llm_message(message) -> str:
    """Readable one-line summary of LLM output."""
    parts = []
    if message.content:
        parts.append(f"content={_shorten(message.content, 160)}")
    if message.tool_calls:
        tool_bits = ", ".join(_summarize_tool_call(tc) for tc in message.tool_calls)
        parts.append(f"tools={tool_bits}")
    else:
        parts.append("tools=none")
    return " | ".join(parts) or "empty_message"


class AgentError(Exception):
    """Exception raised for agent error."""
    pass

class EnvironmentError(Exception):
    """Exception raised for environment error."""
    pass

SYSTEM_PROMPT_TEMPLATE = """# Role
You are a coding agent that writes Python solvers.

# Task
{task_description}

You have access to the following Python packages: {packages}.

Your goal: Write a Python solver that implements the solve() function according to the task specification above.
Save the solver at: {solver_path}

REFERENCE FILES:
- REFERENCE.md: Contains the reference solve() implementation showing expected signature and return format
- VALIDATION.md: Contains the is_solution() method that will validate your solution

FIRST STEP - READ THE REFERENCE:
Before writing any code, you MUST read the reference files using the run_shell tool.

AVAILABLE TOOLS:
1. run_shell - Execute any bash command
2. write_file - Write content to a file (preferred for creating/editing files)
3. indicate_completion - Call this when the task is complete and solver.py exists

INSTRUCTIONS:
- Use the available tools to complete the task
- Always test your solver before calling indicate_completion
- Use THIS Python interpreter: {python_cmd}
- Create the solver at EXACT path: {solver_path}
"""



def get_config_fields(config, workdir, python_path):
    if config and "output" in config:
        output_format = config["output"].get("format", "{task_name}/solver.py")
        base_dir = config["output"].get("base_dir", "results")
        packages = config["packages"]
    else:
        raise EnvironmentError("No output configuration found in config")

    # solver_filename = output_format.format(task_name=self.task)
    # solver_path = workdir / base_dir / solver_filename
    # solver_path.parent.mkdir(parents=True, exist_ok=True)
    solver_path = workdir / "solver.py"

    if python_path is None:
        raise ValueError("No python path provided")
    python_path = Path(python_path)
    if not python_path.is_absolute():
        python_path = (workdir / python_path).absolute()
    if not python_path.exists():
        raise FileNotFoundError(f"Python interpreter not found at {python_path}")
    try:
        python_cmd_path = python_path.relative_to(workdir)
        if not str(python_cmd_path).startswith("."):
            python_cmd_path = Path(".") / python_cmd_path
    except ValueError:
        python_cmd_path = python_path
    python_cmd = python_cmd_path.as_posix()

    try:
        solver_prompt_path = solver_path.relative_to(workdir)
        if not str(solver_prompt_path).startswith("."):
            solver_prompt_path = Path(".") / solver_prompt_path
    except ValueError:
        solver_prompt_path = Path(solver_path.name)
    solver_prompt = solver_prompt_path.as_posix()

    return python_cmd, packages, solver_path, solver_prompt
    
class SingleAgent:
    def __init__(
        self,
        model: str,
        task: str,
        max_steps: int = 20,
        logger: Logger = None,
    ):
        self.model = model
        self.task = task
        self.max_steps = max_steps
        self.logger = logger
    
    def run(
        self,
        workdir: Path,
        task_description: str = "",
        python_path: Path = None,
        config: dict = None
    ) -> Path | None:
        workdir = Path(workdir).resolve()
        own_logger = self.logger is None
        logger = self.logger or Logger(agent="single_agent", task=self.task, run_dir=workdir)
        self.write_file_tool = agents.tool_interface.WriteFileTool(workdir, logger)
        self.run_shell_tool = agents.tool_interface.RunShellTool(workdir, logger)
        python_cmd, packages, solver_path, solver_prompt = get_config_fields(config, workdir, python_path)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            task_description=task_description,
            solver_path=solver_prompt,
            python_cmd=python_cmd,
            packages=packages,
        )
        logger.log_event(
            event="start",
            subagent="single_agent",
            action="init",
            output="",
            result="",
            done=False,
            extra={
                "workdir": str(workdir),
                "solver_path": str(solver_path),
                "python_cmd": python_cmd,
                "packages": packages,
            }
        )
        logger.log_event(
            event="start_agent",
            subagent="single_agent",
            output=system_prompt,
            extra={
                "workdir": str(workdir),
                "solver_path": str(solver_path),
                "python_cmd": python_cmd,
                "packages": packages,
            },
        )

        session = Session(
            model=self.model,
            system_prompt=system_prompt,
            tools=[
                self.write_file_tool.definition,
                self.run_shell_tool.definition,
                agents.tool_interface.indicate_completion_tool,
            ],
        )

        for step_num in range(1, self.max_steps + 1):
            logger.log_event(event="step_start", step=step_num, subagent="single_agent")
            try:
                message = session.get_response().choices[0].message
                logger.log_event(
                    event="llm_response",
                    step=step_num,
                    subagent="single_agent",
                    output=_summarize_llm_message(message),
                )
            except Exception as e:
                logger.log_event(
                    event="close_agent",
                    subagent="single_agent",
                    result="llm_error",
                    done=False,
                )
                raise AgentError(f"Error getting response: {e}")
            if not message.tool_calls:
                session.add_messages(
                    [
                        {
                            "role": "assistant",
                            "content": message.content if message.content else "No content produced."
                        },
                        {
                            "role": "user",
                            "content": "ERROR: You must use the available tools (run_shell, write_file, indicate_completion). Do not output text or JSON - use the tool calling interface. If you want to indicate completion, call the indicate_completion tool."
                        }
                    ]
                )
                logger.log_event(event="output_content_not_tool_error", step=step_num, subagent="single_agent", output=message.content, error="output_content_not_tool_error")
                continue
            
            session.add_messages(
                [
                    {
                        "role": "assistant",
                        "tool_calls": message.tool_calls
                    }
                ]
            )
            for tool_call in message.tool_calls:
                    tool = tool_call.function.name
                    if tool == "write_file":
                        session.add_messages(
                            [
                                {
                                    "role": "tool",
                                    "content": self.write_file_tool.execute(tool_call, step_num),
                                    "tool_call_id": tool_call.id,
                                }
                            ]
                        )
                    elif tool == "run_shell":
                        session.add_messages(
                            [
                                {
                                    "role": "tool",
                                    "content": self.run_shell_tool.execute(tool_call, step_num),
                                    "tool_call_id": tool_call.id,
                                }
                            ]
                        )
                    elif tool == "indicate_completion":
                        if solver_path.exists():
                            logger.log_event(event="done", step=step_num, subagent="single_agent", result="solver_exists", done=True)
                            logger.log_event(
                                event="close_agent",
                                subagent="single_agent",
                                result="solver_exists",
                                done=True,
                            )
                            if own_logger:
                                logger.serialize()
                            return solver_path
                        else:
                            session.add_messages(
                                [
                                    {
                                        "role": "user",
                                        "content": f"ERROR: You set done=true but solver file does not exist at {solver_path}. You must create the solver.py file before setting done=true. Continue working."
                                    }
                                ]
                            )
                            logger.log_event(event="missing_solver", step=step_num, subagent="single_agent", error="solver_missing", done=False)
                            continue

                    else:
                        content = "Error: Unknown tool"
                        session.add_messages(
                            [
                                {
                                    "role": "tool",
                                    "content": content,
                                    "tool_call_id": tool_call.id,
                                }
                            ]
                        )
        logger.log_event(event="max_steps", subagent="single_agent", result="incomplete", done=False)
        logger.log_event(
            event="close_agent",
            subagent="single_agent",
            result="max_steps",
            done=False,
        )
        if own_logger:
            logger.serialize()
        return solver_path if solver_path.exists() else None
