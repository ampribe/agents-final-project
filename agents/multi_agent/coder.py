import json
from pathlib import Path
from typing import Optional

from agents.multi_agent.prompts import CODER_PROMPT_BASE, CODER_RESEARCHER_INSTRUCTIONS
from agents.multi_agent.tool_chat import ToolChatAgent
from agents.multi_agent.utils import rel_path
from agents.tool_interface import RunShellTool, WriteFileTool


class CoderTool:
    definition = {
        "type": "function",
        "function": {
            "name": "call_coder",
            "description": "Ask the coder to generate or update the solver using the provided implementation strategy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "approach": {
                        "type": "string",
                        "description": "Specific implementation strategy from the coordinator (the coder should follow this).",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional path for the solver (defaults to configured solver_path).",
                    },
                },
                "required": ["approach"],
            },
        },
    }

    def __init__(
        self,
        model: str,
        client,
        workdir: Path,
        solver_path: Path,
        packages: list[str],
        python_cmd: str,
        reference_text: str,
        validation_text: str,
        write_file_tool: WriteFileTool,
        run_shell_tool: RunShellTool,
        logger,
        researcher_tool: Optional = None,
        max_steps: int = 14,
    ):
        self.workdir = workdir
        self.solver_path = solver_path
        self.packages = packages
        self.python_cmd = python_cmd
        self.reference_text = reference_text
        self.validation_text = validation_text
        self.write_file_tool = write_file_tool
        self.run_shell_tool = run_shell_tool
        self.researcher_tool = researcher_tool

        tools = [
            write_file_tool.definition,
            run_shell_tool.definition,
        ]
        tool_handlers = {
            "write_file": self.write_file,
            "run_shell": self.run_shell,
        }

        if researcher_tool is not None:
            tools.append(researcher_tool.definition)
            tool_handlers["call_researcher"] = self.call_researcher

        self.agent = ToolChatAgent(
            name="coder",
            model=model,
            client=client,
            tools=tools,
            tool_handlers=tool_handlers,
            logger=logger,
            max_steps=max_steps,
        )

    def write_file(self, tool_call, step_num: int) -> str:
        return self.write_file_tool.execute(tool_call, step_num)

    def run_shell(self, tool_call, step_num: int) -> str:
        return self.run_shell_tool.execute(tool_call, step_num)

    def call_researcher(self, tool_call, step_num: int) -> str:
        if self.researcher_tool is None:
            return "Error: Researcher not available in this configuration."
        return self.researcher_tool.execute(tool_call, step_num)

    def execute(self, tool_call, step_num: int, task_name: str, task_description: str) -> str:
        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError as exc:
            return f"Error: Invalid JSON in call_coder arguments. {exc}\n\nMake sure 'approach' is provided as a properly escaped string."

        approach = args.get("approach", "").strip()
        if not approach:
            return "Error: 'approach' parameter is required for call_coder.\n\nPlease provide a clear implementation strategy describing:\n- Which algorithm/library to use\n- Key implementation details\n- Expected input/output format"

        output_path_raw = args.get("output_path")
        target = Path(output_path_raw) if output_path_raw else self.solver_path
        if not target.is_absolute():
            target = (self.solver_path.parent / target).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        self.solver_path = target

        researcher_instructions = CODER_RESEARCHER_INSTRUCTIONS if self.researcher_tool else ""

        prompt = CODER_PROMPT_BASE.format(
            solver_path=rel_path(self.workdir, target),
            task_name=task_name,
            task_description=task_description or "(no description provided)",
            approach=approach or "(no approach provided)",
            packages=", ".join(self.packages) or "(none)",
            python_cmd=self.python_cmd,
            reference_text=self.reference_text or "(reference unavailable)",
            validation_text=self.validation_text or "(validation unavailable)",
            researcher_instructions=researcher_instructions,
        )
        summary = self.agent.run(prompt, step_num)
        exists = target.exists()
        result = f"Coder finished. solver_path={rel_path(self.workdir, target)} exists={exists}. Notes: {summary}"

        if not exists:
            result += "\n\nWARNING: The solver file was not created. The coder may have encountered an error or didn't use write_file."

        return result
