import importlib.util
import json
from pathlib import Path
from typing import Optional, Tuple

from openai import OpenAI

from inference_auth_token import get_access_token
from evaluation.logging import Logger
from agents.tool_interface import RunShellTool, WriteFileTool, indicate_completion_tool
from agents.multi_agent.prompts import get_coordinator_prompt
from agents.multi_agent.utils import rel_path
from agents.multi_agent.coder import CoderTool
from agents.multi_agent.researcher import ResearcherTool
from agents.multi_agent.test_generator import TestGenerator


class MultiAgent:
    def __init__(
        self,
        model: str,
        task: str,
        max_steps: int = 30,
        logger=None,
        enable_researcher: bool = True,
        enable_tester: bool = True,
    ):
        self.model = model
        self.task = task
        self.max_steps = max_steps
        self.logger = logger
        self.enable_researcher = enable_researcher
        self.enable_tester = enable_tester
        self.client = OpenAI(
            api_key=get_access_token(),
            base_url="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
        )

    @staticmethod
    def _build_python_cmd(workdir: Path, python_path: Path) -> str:
        python_path = Path(python_path)
        try:
            rel = python_path.relative_to(workdir)
            if not str(rel).startswith("."):
                rel = Path(".") / rel
            return rel.as_posix()
        except ValueError:
            return python_path.as_posix()

    def run(
        self,
        workdir: Path,
        task_description: str = "",
        python_path: Path = None,
        config: dict = None,
    ) -> Optional[Path]:
        workdir = Path(workdir).resolve()
        if config and "output" in config:
            output_format = config["output"].get("format", "{task_name}/solver.py")
            base_dir = config["output"].get("base_dir", "results")
            packages = config.get("packages", [])
        else:
            raise ValueError("No output configuration found in config")

        solver_filename = output_format.format(task_name=self.task)
        solver_path = workdir / base_dir / solver_filename
        solver_path.parent.mkdir(parents=True, exist_ok=True)

        if python_path is None:
            raise ValueError("No python path provided")
        python_path = Path(python_path)
        if not python_path.is_absolute():
            python_path = (workdir / python_path).absolute()
        if not python_path.exists():
            raise FileNotFoundError(f"Python interpreter not found at {python_path}")

        python_cmd = self._build_python_cmd(workdir, python_path)

        own_logger = self.logger is None
        logger = self.logger or Logger(agent="multi_agent", task=self.task, run_dir=workdir, echo=False)

        reference_text = (workdir / "REFERENCE.md").read_text() if (workdir / "REFERENCE.md").exists() else ""
        validation_text = (workdir / "VALIDATION.md").read_text() if (workdir / "VALIDATION.md").exists() else ""

        write_file_tool = WriteFileTool(workdir, logger)
        run_shell_tool = RunShellTool(workdir, logger)

        researcher_tool = None
        if self.enable_researcher:
            researcher_tool = ResearcherTool(
                model=self.model,
                client=self.client,
                workdir=workdir,
                task_description=task_description or "(no description provided)",
                logger=logger,
            )

        coder_tool = CoderTool(
            model=self.model,
            client=self.client,
            workdir=workdir,
            solver_path=solver_path,
            packages=packages,
            python_cmd=python_cmd,
            reference_text=reference_text,
            validation_text=validation_text,
            write_file_tool=write_file_tool,
            run_shell_tool=run_shell_tool,
            researcher_tool=researcher_tool,
            logger=logger,
        )

        test_generator = None
        if self.enable_tester:
            test_generator = TestGenerator(
                model=self.model,
                client=self.client,
                workdir=workdir,
                python_cmd=python_cmd,
                solver_path=solver_path,
                reference_text=reference_text,
                validation_text=validation_text,
                write_file_tool=write_file_tool,
                run_shell_tool=run_shell_tool,
                logger=logger,
            )

        coordinator_prompt_template = get_coordinator_prompt(self.enable_researcher, self.enable_tester)

        prompt_kwargs = {
            "task_name": self.task,
            "task_description": task_description or "(no description provided)",
            "solver_path": rel_path(workdir, solver_path),
            "python_cmd": python_cmd,
            "packages": ", ".join(packages) or "(none)",
        }

        if self.enable_tester:
            prompt_kwargs["test_command"] = f"{python_cmd} harness.py {{solver_path}}"

        system_prompt = coordinator_prompt_template.format(**prompt_kwargs)

        messages = [{"role": "developer", "content": system_prompt}]

        tools = [
            run_shell_tool.definition,
            indicate_completion_tool,
        ]

        if self.enable_researcher:
            tools.insert(1, researcher_tool.definition)

        tools.insert(1 if not self.enable_researcher else 2, coder_tool.definition)

        if self.enable_tester:
            tools.insert(-1, test_generator.definition)

        logger.log_event(
            event="start_agent",
            subagent="coordinator",
            output=system_prompt,
            extra={
                "config": {
                    "enable_researcher": self.enable_researcher,
                    "enable_tester": self.enable_tester,
                },
                "solver_path": rel_path(workdir, solver_path),
                "python_cmd": python_cmd,
                "packages": packages,
            },
        )

        def _snapshot_dir() -> str:
            try:
                entries = []
                for p in sorted(workdir.iterdir()):
                    entries.append(p.name + ("/" if p.is_dir() else ""))
                return " ".join(entries)
            except Exception as exc:
                return f"(ls error: {exc})"

        def _parse_tool_args(raw: str | None) -> Tuple[object, bool]:
            try:
                return json.loads(raw or "{}"), True
            except json.JSONDecodeError:
                return {"_raw": raw or "", "_error": "invalid_json"}, False

        tool_to_phase = {
            "run_shell": "run_shell",
            "write_file": "write_file",
            "call_coder": "coder",
            "indicate_completion": "coordinator",
        }

        if self.enable_researcher:
            tool_to_phase["call_researcher"] = "researcher"

        if self.enable_tester:
            tool_to_phase["call_testgen"] = "test_generator"

        for step_num in range(1, self.max_steps + 1):
            logger.log_event(event="step_start", subagent="coordinator", step=step_num)
            messages = [
                m
                for m in messages
                if not (
                    m.get("role") == "developer"
                    and isinstance(m.get("content"), str)
                    and m["content"].startswith("Tool calls remaining:")
                )
            ]
            remaining = self.max_steps - step_num + 1
            messages.append(
                {
                    "role": "developer",
                    "content": f"Tool calls remaining: {remaining}. Emit a tool call now; avoid free-form text.",
                }
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
            )
            message = response.choices[0].message
            logger.log_event(
                event="llm_response",
                subagent="coordinator",
                step=step_num,
                output=message.content or "",
                extra={"tool_calls": [tc.function.name for tc in message.tool_calls or []], "phase": "coordinator"},
            )

            if not message.tool_calls:
                text = message.content or "No content produced."
                messages.append({"role": "assistant", "content": text})
                looks_like_tool_json = False
                stripped = text.strip() if text else ""
                if stripped.startswith("{") and '"tool"' in stripped:
                    looks_like_tool_json = True

                tool_descriptions = []
                if self.enable_researcher:
                    tool_descriptions.append("- call_researcher(query, libraries) - get library/API documentation")
                tool_descriptions.append("- call_coder(approach, output_path) - generate/update the solver")
                if self.enable_tester:
                    tool_descriptions.append("- call_testgen(solver_path, directions) - generate tests and run benchmarks")
                tool_descriptions.append("- run_shell(cmd, action) - run shell commands for quick checks")
                tool_descriptions.append("- indicate_completion() - mark task as complete")

                error_prompt = (
                    "ERROR: You must emit real tool calls using the available tools.\n\n"
                    "Available tools:\n"
                    + "\n".join(tool_descriptions)
                    + "\n\n"
                    "Do NOT:\n"
                    "- Paste JSON that looks like a tool call\n"
                    "- Write code directly in your response\n"
                    "- Use apply_patch, tee, or cat > to edit files (use call_coder"
                    + (" or call_testgen" if self.enable_tester else "")
                    + " instead)\n\n"
                    "You must invoke the functions directly using the tool call mechanism."
                )
                if looks_like_tool_json:
                    error_prompt = f"ERROR: Plain JSON is not a valid tool call.\n\nYou wrote:\n{stripped[:200]}\n\n" + error_prompt
                logger.log_event(
                    event="warning_no_tool_call",
                    subagent="coordinator",
                    step=step_num,
                    extra={"phase": "coordinator", "looks_like_tool_json": looks_like_tool_json, "content_preview": stripped[:200]},
                )
                messages.append({"role": "user", "content": error_prompt})
                continue

            messages.append({"role": "assistant", "tool_calls": message.tool_calls})

            for tool_call in message.tool_calls:
                name = tool_call.function.name
                raw_arguments = tool_call.function.arguments
                parsed_args, args_valid = _parse_tool_args(raw_arguments)
                phase = tool_to_phase.get(name, name)

                logger.log_event(
                    event="tool_call",
                    subagent=phase,
                    step=step_num,
                    extra={
                        "tool": name,
                        "arguments": parsed_args,
                        "raw_arguments": raw_arguments,
                        "phase": phase,
                        "tool_call_id": tool_call.id,
                        "arguments_valid": args_valid,
                    },
                )

                if name in {"call_researcher", "call_coder", "call_testgen"}:
                    logger.log_event(
                        event="start_subagent",
                        subagent=phase,
                        step=step_num,
                        extra={
                            "tool": name,
                            "arguments": parsed_args,
                            "raw_arguments": raw_arguments,
                            "phase": phase,
                            "tool_call_id": tool_call.id,
                        },
                    )

                if name == "write_file":
                    content = write_file_tool.execute(tool_call, step_num)
                elif name == "run_shell":
                    content = run_shell_tool.execute(tool_call, step_num)
                elif name == "call_researcher":
                    if not self.enable_researcher or researcher_tool is None:
                        content = "Error: Researcher not available in this configuration"
                    else:
                        content = researcher_tool.execute(tool_call, step_num)
                elif name == "call_coder":
                    content = coder_tool.execute(
                        tool_call,
                        step_num,
                        task_name=self.task,
                        task_description=task_description,
                    )
                    solver_path = coder_tool.solver_path
                elif name == "call_testgen":
                    if not self.enable_tester or test_generator is None:
                        content = "Error: Test generator not available in this configuration"
                    else:
                        content = test_generator.execute(
                            tool_call,
                            step_num,
                            task_name=self.task,
                            task_description=task_description,
                        )
                elif name == "indicate_completion":
                    target = coder_tool.solver_path

                    def _has_top_level_solve(path: Path) -> tuple[bool, str]:
                        if not path.exists():
                            return False, f"Solver not found at {rel_path(workdir, path)}.\n\nThe file does not exist yet. Use call_coder to create it first."
                        try:
                            spec = importlib.util.spec_from_file_location("agent_solver_completion_check", path)
                            module = importlib.util.module_from_spec(spec)
                            assert spec and spec.loader
                            spec.loader.exec_module(module)
                        except ImportError as exc:
                            return False, f"Failed to import solver due to missing dependency: {exc}\n\nThe solver imports a module that cannot be found. Common issues:\n- Importing from a file that doesn't exist (e.g., 'from solver import ...' in reference_impl.py)\n- Missing package installation\n- Circular import\n\nCheck that all files are self-contained and don't have circular dependencies."
                        except Exception as exc:
                            return False, f"Failed to import solver: {exc}\n\nThere is a syntax error or runtime error in the solver code. Use call_coder to fix it."
                        solve_fn = getattr(module, "solve", None)
                        if callable(solve_fn):
                            return True, ""
                        return False, f"Solver must export a module-level callable solve(problem).\n\nThe solver file exists but doesn't have a callable 'solve' function at module level.\n\nIf you use a Solver class, add this line at the end: solve = Solver().solve"

                    ok, err_msg = _has_top_level_solve(target)
                    if ok:
                        logger.log_event(event="done", subagent="coordinator", step=step_num, result="solver_exists", done=True)
                        if own_logger:
                            logger.serialize()
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"Solver ready at {rel_path(workdir, target)}. Completing run.",
                            }
                        )
                        return target
                    content = f"Cannot complete: {err_msg}\n\nFix the issue before calling indicate_completion again."
                else:
                    content = f"Unknown tool: {name}"

                logger.log_event(
                    event="tool_return",
                    subagent=name,
                    step=step_num,
                    output=content,
                    extra={
                        "tool": name,
                        "arguments": parsed_args,
                        "raw_arguments": raw_arguments,
                        "dir_ls": _snapshot_dir(),
                        "phase": phase if name != "indicate_completion" else "coordinator",
                    },
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": content,
                    }
                )
            summary_tools = [tc.function.name for tc in message.tool_calls] if message.tool_calls else []
            logger.log_event(
                event="step_summary",
                subagent="coordinator",
                step=step_num,
                extra={
                    "phase": "coordinator",
                    "tools": summary_tools,
                    "status": "ok" if summary_tools else "no_tools",
                },
            )

        logger.log_event(event="max_steps", subagent="coordinator", result="incomplete", done=False)
        if own_logger:
            logger.serialize()
        return coder_tool.solver_path if coder_tool.solver_path.exists() else None
