import abc
import json
import logging
import re

import requests

from llms.prompt_formatter import (
    ClaudePromptFormatter,
    GPTPromptFormatter,
    PromptFormatter,
)
from utils.helpers import get_key

logger = logging.getLogger(__name__)


def create_request_header(
    model_name: str, api_key: str, request_url: str, **kwargs
) -> dict:
    if model_name == "claude":
        request_header = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
            if "anthropic-version" not in kwargs
            else kwargs["anthropic-version"],
            "content-type": "application/json",
        }
    elif model_name == "gpt":
        request_header = {"Authorization": f"Bearer {api_key}"}
        # use api-key header for Azure deployments
        if "/deployments" in request_url:
            request_header = {"api-key": f"{api_key}"}
    else:
        raise ValueError(f"Model {model_name} not supported")
    return request_header


def api_endpoint_from_url(request_url) -> str | None:
    """Extract the API endpoint from the request URL."""
    match = re.search(r"^https://[^/]+/v\\d+/(.+)$", request_url)
    if match is None:
        # for Azure OpenAI deployment urls
        match = re.search(
            r"^https://[^/]+/openai/deployments/[^/]+/(.+?)(\?|$)", request_url
        )
    return match[1] if match else None


class AttemptTracker:
    def __init__(self, max_attempts: int = 5) -> None:
        self.max_attempts: int = max_attempts
        self.failure_count: int = 0

    def __repr__(self) -> str:
        return f"AttemptTracker(max_attempts:{self.max_attempts}, failure_count: {self.failure_count})"

    @property
    def attempts_left(self) -> int:
        return self.max_attempts - self.failure_count

    def record(self) -> None:
        self.failure_count += 1

    def reset(self) -> None:
        self.failure_count = 0

    def block(self) -> bool:
        return self.failure_count >= self.max_attempts


class LLMAgent(abc.ABC):
    def __init__(
        self,
        model_name: str,
        api_key: str,
        request_url: str,
        max_attempts: int = 5,
        **kwargs,
    ):
        self.model_name: str = model_name
        self.api_key: str = api_key
        self.request_url: str = request_url
        self.request_header: dict = create_request_header(
            model_name, api_key, request_url, **kwargs
        )
        self.counter: AttemptTracker = AttemptTracker(max_attempts=max_attempts)

    @staticmethod
    def is_json(data: str) -> bool:
        try:
            _ = json.loads(data)
        except ValueError:
            return False
        return True

    def to_json(self, s: str) -> dict:
        s = s[next(idx for idx, c in enumerate(s) if c in "{[") :]
        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            if self.is_json(s[: e.pos]):
                return json.loads(s[: e.pos])
            elif e.msg == "Expecting ',' delimiter":
                return self.to_json(s[: e.pos] + "," + s[e.pos :])
            elif e.msg == "Invalid control character at":
                return self.to_json(s[: e.pos] + '",\n' + s[e.pos :])
            else:
                logger.error("Invalid JSON format")
                logger.info(s)
                raise ValueError("Invalid JSON format")

    @abc.abstractmethod
    def get_response_text(self, response) -> str:
        pass

    @abc.abstractmethod
    def get_prompt_formatter(self) -> PromptFormatter:
        pass

    def generate(self, prompt: dict) -> str | None:
        try:
            if self.counter.block():
                logger.warning(
                    f"Request failed after {self.counter.max_attempts} attempts.\n{prompt}"
                )
            else:
                response = requests.post(
                    self.request_url,
                    headers=self.request_header,
                    json=prompt,
                )
                response = response.json()

                if "error" in response:
                    self.counter.record()
                    logger.warning(
                        f"Request failed with error {response['error']}, {self.counter.attempts_left} attempts remaining."
                    )
                    self.generate(prompt)
                return response
        except Exception as e:
            logger.warning(f"Request failed due to {e}")
        return None

    def call(self, prompt: dict) -> str:
        response = self.generate(prompt)
        response = self.get_response_text(response)
        return response if response else ""


class GptAgent(LLMAgent):
    def __init__(
        self,
        api_key: str,
        request_url: str = "https://api.openai.com/v1/chat/completions",
    ):
        super().__init__("gpt", api_key, request_url)

    def get_response_text(self, response) -> str:
        return response["choices"][0]["message"]["content"]

    def get_prompt_formatter(self) -> GPTPromptFormatter:
        return GPTPromptFormatter()


class ClaudeAgent(LLMAgent):
    def __init__(
        self,
        api_key: str,
        request_url: str = "https://api.anthropic.com/v1/messages",
    ):
        super().__init__("claude", api_key, request_url)

    def get_response_text(self, response) -> str:
        return response["content"][0]["text"]

    def get_prompt_formatter(self) -> ClaudePromptFormatter:
        return ClaudePromptFormatter()


if __name__ == "__main__":
    api_key = get_key(filename="claude", keyname="organization")
    # gpt_agent = GptAgent(
    #     model_name="gpt",
    #     api_key=api_key,
    #     request_url="https://api.openai.com/v1/chat/completions",
    # )
    claude_agent = ClaudeAgent(api_key=api_key)
    test_sample = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 100,
        "system": "You are a helpful assistant.",
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of France? Return the answer in a JSON object.",
            },
        ],
    }
    print(claude_agent.call(prompt=test_sample))
