import json
from pathlib import Path

from agents.multi_agent.prompts import TEST_GEN_PROMPT
from agents.multi_agent.tool_chat import ToolChatAgent
from agents.multi_agent.utils import rel_path
from agents.tool_interface import RunShellTool, WriteFileTool


class TestGenerator:
    definition = {
        "type": "function",
        "function": {
            "name": "call_testgen",
            "description": "Generate/refresh tests and benchmark the solver; returns a brief summary and the test command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "solver_path": {
                        "type": "string",
                        "description": "Path to the solver to test (required).",
                    },
                    "directions": {
                        "type": "string",
                        "description": "Optional guidance (scope, focus areas, benchmarks).",
                    },
                },
                "required": ["solver_path"],
            },
        },
    }

    def __init__(
        self,
        model: str,
        client,
        workdir: Path,
        python_cmd: str,
        solver_path: Path,
        reference_text: str,
        validation_text: str,
        write_file_tool: WriteFileTool,
        run_shell_tool: RunShellTool,
        logger,
        max_steps: int = 10,
        num_test_problems: int = 3,
    ):
        self.workdir = workdir
        self.python_cmd = python_cmd
        self.solver_path = solver_path
        self.reference_text = reference_text
        self.validation_text = validation_text
        self.write_file_tool = write_file_tool
        self.run_shell_tool = run_shell_tool
        self.num_test_problems = num_test_problems
        self.history: list = []
        self.agent = ToolChatAgent(
            name="test_generator",
            model=model,
            client=client,
            tools=[write_file_tool.definition, run_shell_tool.definition],
            tool_handlers={
                "write_file": self.write_file,
                "run_shell": self.run_shell,
            },
            logger=logger,
            max_steps=max_steps,
        )

    def write_file(self, tool_call, step_num: int) -> str:
        return self.write_file_tool.execute(tool_call, step_num)

    def run_shell(self, tool_call, step_num: int) -> str:
        return self.run_shell_tool.execute(tool_call, step_num)

    def execute(self, tool_call, step_num: int, task_name: str, task_description: str) -> str:
        args = tool_call.function.arguments or "{}"
        solver_path_arg = None
        directions = ""
        try:
            parsed = json.loads(args)
            solver_path_arg = parsed.get("solver_path")
            directions = (parsed.get("directions") or "").strip()
        except json.JSONDecodeError as exc:
            return f"Error: Invalid JSON in call_testgen arguments. {exc}\n\nMake sure solver_path is provided as a string."
        if not isinstance(solver_path_arg, str) or not solver_path_arg.strip():
            return f"Error: solver_path is required for call_testgen.\n\nReceived: {parsed}\n\nPlease provide solver_path as a non-empty string."

        target = Path(solver_path_arg)
        if not target.is_absolute():
            target = (self.workdir / target).resolve()
        self.solver_path = target

        prompt = TEST_GEN_PROMPT.format(
            task_name=task_name,
            task_description=task_description or "(no description provided)",
            python_cmd=self.python_cmd,
            solver_path=rel_path(self.workdir, self.solver_path),
            num_test_problems=self.num_test_problems,
            directions=directions or "Focus on realistic, scaling benchmarks.",
            reference_text=self.reference_text or "(reference unavailable)",
            validation_text=self.validation_text or "(validation unavailable)",
        )
        final_text = self.agent.run(prompt, step_num, messages=self.history)
        return final_text.strip() or "No summary produced."
