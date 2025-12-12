from typing import Callable


class ToolChatAgent:
    def __init__(
        self,
        name: str,
        model: str,
        client,
        tools: list,
        tool_handlers: dict[str, Callable],
        logger,
        max_steps: int,
    ):
        self.name = name
        self.model = model
        self.client = client
        self.tools = tools
        self.tool_handlers = tool_handlers
        self.logger = logger
        self.max_steps = max_steps

    def run(self, system_prompt: str, step_num: int, messages: list | None = None) -> str:
        if messages is None:
            messages = []
        if self.logger:
            self.logger.log_event(
                event="subagent_prompt",
                subagent=self.name,
                step=step_num,
                output=system_prompt,
                    extra={"phase": self.name},
            )
        messages.append({"role": "developer", "content": system_prompt})
        final_text = ""
        for inner_step in range(1, self.max_steps + 1):
            if self.logger:
                self.logger.log_event(
                    event="llm_call",
                    subagent=self.name,
                    step=f"{step_num}.{inner_step}",
                        extra={"phase": self.name},
                )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
            )
            message = response.choices[0].message
            if self.logger:
                self.logger.log_event(
                    event="llm_response",
                    subagent=self.name,
                    step=f"{step_num}.{inner_step}",
                    output=message.content or "",
                    extra={"tool_calls": [tc.function.name for tc in message.tool_calls or []], "phase": self.name},
                )
            if message.tool_calls:
                messages.append({"role": "assistant", "tool_calls": message.tool_calls})
                for tc in message.tool_calls:
                    handler = self.tool_handlers.get(tc.function.name)
                    result = handler(tc, step_num) if handler else f"Unknown tool: {tc.function.name}"
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        }
                    )
                continue
            final_text = message.content or ""
            messages.append({"role": "assistant", "content": final_text})
            break
        return final_text
