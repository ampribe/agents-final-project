from openai import OpenAI

from inference_auth_token import get_access_token


class Session:
    def __init__(self, model: str, system_prompt: str|None = None, tools: list|None = None):
        self.client = OpenAI(
            api_key=get_access_token(),
            base_url="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
        )
        self.model = model
        self.messages = [
            {"role": "developer", "content": system_prompt}
        ]
        self.tools = tools
    
    def add_messages(self, messages):
        self.messages.extend(messages)
    
    def get_response(self, **kwargs):
        if self.tools:
            return self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tools,
                **kwargs,
            )
        else:
            return self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                **kwargs,
            )