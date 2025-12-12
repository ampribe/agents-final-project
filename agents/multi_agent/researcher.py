import asyncio
import json
from pathlib import Path

from fastmcp import Client

from agents.multi_agent.prompts import RESEARCHER_PROMPT


class ResearcherTool:
    definition = {
        "type": "function",
        "function": {
            "name": "call_researcher",
            "description": "Ask the researcher for package/API guidance using Context7 MCP tools; include the libraries to inspect for speedups.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look up"},
                    "libraries": {
                        "type": "array",
                        "description": "List of libraries/packages to inspect for performance-oriented usage.",
                        "items": {"type": "string"},
                    },
                },
                "required": ["query"],
            },
        },
    }

    def __init__(
        self,
        model: str,
        client,
        workdir: Path,
        task_description: str,
        logger,
        max_steps: int = 15,
    ):
        self.model = model
        self.client = client
        self.logger = logger
        self.workdir = workdir
        self.task_description = task_description
        self.max_steps = max_steps
        self.mcp_url = "https://mcp.context7.com/mcp"

    def execute(self, tool_call, step_num: int) -> str:
        raw_args = tool_call.function.arguments or ""
        args = self._safe_json_load(raw_args)

        if not args:
            return f"Error: Invalid JSON in call_researcher arguments.\n\nReceived: {raw_args[:200]}\n\nPlease provide 'query' as a string and optionally 'libraries' as an array of strings."

        query = args.get("query", "").strip()
        if not query:
            return "Error: 'query' parameter is required for call_researcher.\n\nPlease provide a clear question about what library/API information you need."

        libraries = args.get("libraries") or []
        libraries_text = ", ".join(libraries) if libraries else "(none provided)"
        prompt = RESEARCHER_PROMPT.format(
            query=query,
            libraries=libraries_text,
            task_description=self.task_description,
        )

        try:
            return self._run_async(self._run_researcher(prompt, step_num))
        except Exception as exc:
            if self.logger:
                self.logger.log_event(
                    event="researcher_mcp_error",
                    subagent="researcher",
                    result="error",
                    extra={"error": str(exc), "query": query, "libraries": libraries, "phase": "researcher"},
                )
            return f"Error running researcher: {exc}\n\nQuery: {query}\nLibraries: {libraries_text}\n\nThe MCP service may be unavailable or the query may have timed out. Consider rephrasing the query or trying fewer libraries at once."

    async def _run_researcher(self, prompt: str, step_num: int) -> str:
        messages = [{"role": "developer", "content": prompt}]
        final_text = ""

        async with Client(self.mcp_url) as client:
            try:
                tools = await client.list_tools()
            except Exception as exc:
                if self.logger:
                    self.logger.log_event(
                        event="researcher_mcp_discovery_failed",
                        subagent="researcher",
                        result="error",
                        extra={"error": str(exc), "phase": "researcher"},
                    )
                return f"Error listing MCP tools: {exc}"

            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                }
                for tool in tools
            ]

            if self.logger:
                self.logger.log_event(
                    event="researcher_mcp_tools_loaded",
                    subagent="researcher",
                    result="success",
                    extra={"tool_count": len(openai_tools), "phase": "researcher"},
                )

            for inner_step in range(1, self.max_steps + 1):
                if self.logger:
                    self.logger.log_event(
                        event="llm_call",
                        subagent="researcher",
                        step=f"{step_num}.{inner_step}",
                        extra={"phase": "researcher"},
                    )

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=openai_tools,
                )
                message = response.choices[0].message

                if self.logger:
                    self.logger.log_event(
                        event="llm_response",
                        subagent="researcher",
                        step=f"{step_num}.{inner_step}",
                        output=message.content or "",
                        extra={"tool_calls": [tc.function.name for tc in message.tool_calls or []], "phase": "researcher"},
                    )

                if not message.tool_calls:
                    final_text = message.content or ""
                    messages.append({"role": "assistant", "content": final_text})
                    break

                messages.append({"role": "assistant", "tool_calls": message.tool_calls})

                for tc in message.tool_calls:
                    tool_args = self._safe_json_load(tc.function.arguments)
                    try:
                        result = await client.call_tool(tc.function.name, tool_args)
                        tool_result = self._extract_tool_text(result)
                        if self.logger:
                            self.logger.log_event(
                                event="researcher_mcp_tool_called",
                                subagent="researcher",
                                result="success",
                                extra={"tool": tc.function.name, "phase": "researcher"},
                            )
                    except Exception as exc:
                        tool_result = f"Error calling MCP tool {tc.function.name}: {exc}"
                        if self.logger:
                            self.logger.log_event(
                                event="researcher_mcp_error",
                                subagent="researcher",
                                result="error",
                                extra={"tool": tc.function.name, "error": str(exc), "phase": "researcher"},
                            )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tool_result,
                        }
                    )

        if not final_text:
            messages.append(
                {
                    "role": "user",
                    "content": "Summarize your findings now in plain text (no tool calls).",
                }
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=None,
            )
            summary_msg = response.choices[0].message
            final_text = summary_msg.content or ""
            if self.logger:
                self.logger.log_event(
                    event="llm_response",
                    subagent="researcher",
                    step=f"{step_num}.final",
                    output=final_text,
                    extra={"tool_calls": []},
                )

        return final_text or "No researcher answer produced."

    @staticmethod
    def _safe_json_load(raw_json: str) -> dict:
        try:
            return json.loads(raw_json or "{}")
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _extract_tool_text(result) -> str:
        fragments = []
        for content in getattr(result, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                fragments.append(text)
        return "\n".join(fragments)

    def _run_async(self, coro):
        try:
            return asyncio.run(coro)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
