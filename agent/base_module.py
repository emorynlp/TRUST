import abc
import logging
import sys


from llms.prompt_formatter import (
    Patterns,
    PromptData,
)
from llms.gpt_call import ClaudeAgent, GptAgent
from utils.helpers import get_key

logger = logging.getLogger(__name__)


class LLMModule(abc.ABC):
    def __init__(self, model_name: str) -> None:
        self.model_name: str = model_name
        self.model = self._load_model()
        self.prompt_formatter = self.model.get_prompt_formatter()

    def _load_model(self):
        if self.model_name.startswith("gpt"):
            return GptAgent(get_key(filename="openai", keyname="organization"))
        elif self.model_name.startswith("claude"):
            return ClaudeAgent(get_key(filename="claude", keyname="organization"))
        else:
            raise ValueError(f"Model {self.model_name} is not supported for now.")

    @abc.abstractmethod
    def generate(self, *args, **kwargs) -> dict | str | None:
        pass

    def format_prompt(
        self, prompt_templates: dict, params: dict, **pattern_pairs
    ) -> dict:
        """
        Format the prompt using the given template and prompt parameters

        Parameters:
            prompt_templates (dict): Conversation templates
            params (dict): Parameters for the prompt
            pattern_pairs (dict): Pattern pairs for the prompt

        Returns:
            dict: Formatted prompt
        """
        pattern_dict = Patterns()
        for role, template in prompt_templates.items():
            if "{" in template and role != "assistant":
                pattern_dict.fill_patterns(role, template, **pattern_pairs)
        prompt_data = PromptData().from_patterns(
            prompt_templates=prompt_templates, patterns=pattern_dict
        )
        prompts = self.prompt_formatter.format(prompt_data, **params)
        # logger.info(prompts)
        return prompts[0]
