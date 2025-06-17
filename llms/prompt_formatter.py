# ========================================================================
# Copyright 2024 Sichang TU
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================

__author__ = "Sichang TU"

import abc
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)


def get_patterns(string) -> list[str]:
    return re.findall(r"\{(.+?)\}", string)


@dataclass
class Params:
    model_name: str
    temperature: float | None = None
    top_p: float | None = None
    stream: bool | None = None

    def update(self, **kwargs):
        """Update the parameters with the given keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown parameter {key} for {self.__class__.__name__}")

    def filter_none(self) -> dict:
        """Filter out None values from the parameters."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class GPTParams(Params):
    max_completion_tokens: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    response_format: dict | None = None
    n: int | None = None
    seed: int | None = None


@dataclass
class ClaudeParams(Params):
    max_tokens: int = 1024
    top_k: int | None = None


@dataclass
class Patterns:
    """Dynamic patterns for the prompt templates.

    Atributes:
        system_patterns (dict): Key, value pairs for the replaceable patterns in the system template.
            For example, {"pattern1": "value1", "pattern2": "value2"}
        user_patterns (dict): Key, value pairs for the replaceable patterns in the user template.
            For example, {"pattern1": "value1", "pattern2": "value2"}
        assistant_patterns (dict): Key, value pairs for the replaceable patterns in the assistant template.
            For example, {"pattern1": "value1", "pattern2": "value2"}
    """

    system_patterns: dict = field(default_factory=dict)
    user_patterns: dict = field(default_factory=dict)
    assistant_patterns: dict = field(default_factory=dict)

    def fill_patterns(self, role: str, template: str, **patterns_pair):
        patterns_names = get_patterns(template)
        patterns = getattr(self, f"{role}_patterns")
        for pattern in patterns_names:
            if pattern in patterns_pair:
                patterns[pattern] = patterns_pair[pattern]
            else:
                raise ValueError(f"Pattern {pattern} not found in patterns pairs.")


@dataclass
class PromptSample:
    """Format for one sample of the prompt data.

    Atributes:
        user_patterns (dict): Key, value pairs for the replaceable patterns in the user template.
            For example, {"pattern1": "value1", "pattern2": "value2"}
        assistant_patterns (dict): Key, value pairs for the replaceable patterns in the assistant template.
            For example, {"pattern1": "value1", "pattern2": "value2"}
        examples (list): List of examples for the prompt sample.
            Each example is a list of dictionaries, where each dictionary is a message from the user or assistant.
    """

    user_patterns: dict = field(default_factory=dict)
    assistant_patterns: dict = field(default_factory=dict)
    examples: list = field(default_factory=list)


@dataclass
class PromptData:
    """Format prompt data with replaceable patterns.

    Atributes:
        system_template (str): The system template with replaceable patterns.
            For example, "Hello {pattern1}, how are you?"
        user_template (str): The user template with replaceable patterns.
        assistant_template (str): The assistant template with replaceable patterns.
        system_patterns (dict): Key, value pairs for the replaceable patterns in the system template.
            For example, {"pattern1": "value1", "pattern2": "value2"}
        samples (list[PromptSample]): List of prompt samples.
            Each sample is a PromptSample object.
    """

    system_template: str = field(default_factory=str)
    user_template: str = field(default_factory=str)
    assistant_template: str = field(default_factory=str)
    system_patterns: dict = field(default_factory=dict)
    samples: list[PromptSample] = field(default_factory=list)

    def from_patterns(
        self, prompt_templates: dict, patterns: Patterns, examples: list | None = None
    ):
        self.system_template = prompt_templates["system"]
        self.user_template = prompt_templates["user"]
        self.assistant_template = (
            prompt_templates["assistant"] if "assistant" in prompt_templates else ""
        )
        self.system_patterns = patterns.system_patterns
        prompt_sample = PromptSample(
            user_patterns=patterns.user_patterns,
            assistant_patterns=patterns.assistant_patterns,
        )
        if examples:
            prompt_sample.examples = examples
        self.samples = [prompt_sample]
        return self


class PromptFormatter(abc.ABC):
    """Base class for prompt formatters.
    This class defines the interface for formatting prompts for different models.
    It supports both conversational and instruction styles.

    Attributes:
        style (str): The style of the prompt, either "conversational" or "instruction".
        inference (bool): Whether the prompt is for inference or not.
            If True, the assistant message will not be included in the prompt.
            For training, it should be False.
    """

    def __init__(
        self,
        style: Literal["conversational", "instruction"] = "conversational",
        inference: bool = False,
    ):
        """Initialize the prompt formatter with the given style and inference flag.

        Args:
            style (str): The style of the prompt, either "conversational" or "instruction".
            inference (bool): Whether the prompt is for inference or not.
                If True, the assistant message will not be included in the prompt.
                For training, it should be False.
        """
        if style not in ["conversational", "instruction"]:
            raise ValueError(
                f"Unknown style: {style}, please use conversational or instruction"
            )

        self.style = style  # could be conversational or instruction
        self.inference = inference

    @abc.abstractmethod
    def to_text(self, prompt) -> str: ...

    def add_prompt_tokens(self, role, msg) -> str: ...

    def set_params(self, *args, **kwargs) -> dict: ...

    def update_funcs(self, **funcs):
        """Update the prompt formatter with new functions."""
        for key, value in funcs.items():
            if hasattr(self, key):
                logger.warning(f"Function {key} already exists, overwriting it.")
            setattr(self, key, value)

    def load_funcs_from_registry(self, registry: dict, *modules: str):
        """Load functions from a registry dictionary.

        Args:
            registry (dict): A dictionary containing function definitions.
            The registry should have function names as keys and function objects as values.
            *modules (str): Optional module names to load specific functions from the registry.
        """
        if modules:
            for module in modules:
                self.load_funcs_from_registry(registry[module])
        else:
            self.update_funcs(**registry)

    def form_pattern(self, pattern_value: str | list | dict, pattern_name: str) -> str:
        """Form the pattern value based on the pattern name and value.

        Args:
            pattern_value (str | list | dict): The value of the pattern to be formatted.
                It can be a string, a list, or a dictionary.
            pattern_name (str): The name of the pattern to be formatted.

        Returns:
            str: The formatted pattern value.
        """
        func = getattr(self, f"form_{pattern_name}", None)
        if func is not None:
            return func(pattern_value)
        else:
            if not pattern_value:
                return ""
            elif isinstance(pattern_value, str):
                return pattern_value
            else:
                return json.dumps(pattern_value)

    def form_msg(self, template: str, pattern_dict: dict) -> str:
        """Form the message by replacing patterns in the template with values from the pattern_dict.

        Args:
            template (str): The template string containing patterns to be replaced.
            pattern_dict (dict): A dictionary containing key-value pairs for the patterns.
                The keys should match the patterns in the template, and the values are the replacements.

        Returns:
            str: The formatted message with patterns replaced by their corresponding values.
        """
        patterns = get_patterns(template)
        if patterns:
            for pattern in patterns:
                if pattern in pattern_dict:
                    replace_str = self.form_pattern(pattern_dict[pattern], pattern)
                    template = template.replace(f"{{{pattern}}}", replace_str)
                else:
                    raise ValueError(f"Pattern {pattern} not found in pattern_dict")
        return template.strip()

    @abc.abstractmethod
    def conversational_msg(
        self, system_msg, user_msg, assistant_msg, examples=None, **kwargs
    ) -> dict: ...

    def form_examples(self, examples: list, func_name: str, **kwargs) -> list:
        """Form examples for few-shot learning.

        Args:
            examples (list): List of examples to be formatted.
            func_name (str): The name of the function to be used for formatting the examples.
                It should be one of the methods defined in this class.
            **kwargs: Additional keyword arguments to be passed to the formatting function.

        Returns:
            list: A list of formatted examples.
            Each example is a dictionary with keys "role" and "content".
        """
        if not examples:
            return []
        func = getattr(self, func_name, None)
        if func is None:
            raise ValueError(f"Unknown function name: {func_name}")
        output = []
        for i, example in enumerate(examples):
            role = "user" if i % 2 == 0 else "assistant"
            if func_name == "form_msg":
                output.append(
                    func(kwargs[f"{role}_template"], dict(zip(kwargs[role], example)))
                )
            elif func_name == "conversational":
                output.append({"role": role, "content": example})
            elif func_name == "instruction":
                output.append(self.add_prompt_tokens(role, example))
        return output

    def format(self, prompt_data: PromptData, **params) -> list:
        """Format the prompt data into a list of prompts.

        Args:
            prompt_data (PromptData): The prompt data containing templates and patterns.
            **params: Additional parameters to be passed to the formatting function.

        Returns:
            list: A list of formatted prompts.
            The structure depends on the style of the prompt formatter.
        """
        format_func = getattr(self, f"{self.style}_msg")
        sys_msg = self.form_msg(
            prompt_data.system_template, prompt_data.system_patterns
        )
        formatted = []
        for sample in prompt_data.samples:
            examples = sample.examples
            user_msg = self.form_msg(prompt_data.user_template, sample.user_patterns)
            assistant_msg = (
                self.form_msg(prompt_data.assistant_template, sample.assistant_patterns)
                if not self.inference
                else ""
            )
            examples = self.form_examples(
                examples,
                "form_msg",
                user_template=prompt_data.user_template,
                assistant_template=prompt_data.assistant_template,
                user=sample.user_patterns.keys(),
                assistant=sample.assistant_patterns.keys(),
            )
            messages = format_func(sys_msg, user_msg, assistant_msg, examples, **params)
            formatted.append(messages)
        return formatted


class GPTPromptFormatter(PromptFormatter):
    """Prompt formatter for GPT models, supporting conversational style."""

    def __init__(
        self,
        style: Literal["conversational"] = "conversational",
        inference=True,
        **funcs,
    ) -> None:
        super().__init__(style, inference)
        if style != "conversational":
            raise ValueError("GPTPromptFormatter only supports conversational style")
        if funcs:
            self.update_funcs(**funcs)

    def set_params(
        self,
        model_name: str,
        json_format: bool = False,
        temperature: int = 0,
        seed: int = 93,
        json_schema: dict | None = None,
        **kwargs,
    ) -> dict:
        params = GPTParams(
            model_name=model_name,
            temperature=temperature,
            seed=seed,
        )
        if json_format:
            params.response_format = {"type": "json_object"}
        if json_schema:
            params.response_format = {"type": "json_schema", "schema": json_schema}
        if kwargs:
            params.update(**kwargs)

        return params.filter_none()

    def to_text(self, prompt) -> str:
        params, content = "", ""
        for key, value in prompt.items():
            if key == "messages":
                for message in value:
                    content += f"<{message['role'].upper()}>\n{message['content']}\n"
            else:
                params += f"{key}: {value}\n"
        params = "<PARAMETERS>\n" + params
        return params + "\n" + content

    def conversational_msg(
        self, system_msg, user_msg, assistant_msg, examples=None, **kwargs
    ) -> dict:
        prompt = {}
        prompt["messages"] = []
        if system_msg:
            prompt["messages"].append({"role": "developer", "content": system_msg})
        if examples:
            prompt["messages"].extend(self.form_examples(examples, "conversational"))
        prompt["messages"].append({"role": "user", "content": user_msg})
        if not self.inference:
            prompt["messages"].append({"role": "assistant", "content": assistant_msg})
        if kwargs:
            prompt.update(kwargs)
        return prompt


class ClaudePromptFormatter(PromptFormatter):
    """Prompt formatter for Claude models, supporting conversational style."""

    def __init__(
        self,
        style: Literal["conversational"] = "conversational",
        inference=True,
        **funcs,
    ) -> None:
        super().__init__(style, inference)
        if style != "conversational":
            raise ValueError("ClaudePromptFormatter only supports conversational style")
        if funcs:
            self.update_funcs(**funcs)

    def set_params(
        self, model_name: str, max_tokens: int = 1024, temperature: int = 0, **kwargs
    ) -> dict:
        params = ClaudeParams(
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if kwargs:
            params.update(**kwargs)
        return params.filter_none()

    def to_text(self, prompt) -> str:
        params, content, system = "", "", "<SYSTEM>\n"
        for key, value in prompt.items():
            if key == "messages":
                for message in value:
                    content += f"<{message['role'].upper()}>\n{message['content']}\n"
            elif key == "system":
                system += value
            else:
                params += f"{key}: {value}\n"
        params = "<PARAMETERS>\n" + params
        return params + "\n" + system + "\n\n" + content

    def conversational_msg(
        self, system_msg, user_msg, assistant_msg, examples=None, **kwargs
    ) -> dict:
        prompt = {}
        if system_msg:
            prompt["system"] = system_msg
        prompt["messages"] = []
        if examples:
            prompt["messages"].extend(self.form_examples(examples, "conversational"))
        prompt["messages"].append({"role": "user", "content": user_msg})
        if not self.inference:
            prompt["messages"].append({"role": "assistant", "content": assistant_msg})
        if kwargs:
            prompt.update(kwargs)
        return prompt


class Llama3PromptFormatter(PromptFormatter):
    """Prompt formatter for Llama 3 models, supporting both conversational and instruction styles."""

    START = "<|begin_of_text|>"
    ID_START = "<|start_header_id|>"
    ID_END = "<|end_header_id|>"
    M_END = "<|eot_id|>"
    END = "<|end_of_text|>"

    def __init__(
        self,
        style: Literal["conversational", "instruction"] = "conversational",
        inference=False,
        add_bos=False,
        add_eos=False,
        **funcs,
    ) -> None:
        super().__init__(style, inference)
        self.add_bos = add_bos
        self.add_eos = add_eos
        if funcs:
            for key, value in funcs.items():
                setattr(self, key, value)

    def instruction_msg(
        self, system_msg, user_msg, assistant_msg, examples=None
    ) -> dict:
        prompt = {}
        prompt["prompt"] = (
            self.add_prompt_tokens("system", system_msg) if system_msg else ""
        )

        if examples:
            prompt["prompt"] += "".join(self.form_examples(examples, "instruction"))
        prompt["prompt"] += self.add_prompt_tokens("user", user_msg, prompt_end=True)
        if not self.inference:
            prompt["completion"] = self.add_prompt_tokens(
                "assistant", assistant_msg, completion_start=True
            )
        if self.add_bos:
            prompt["prompt"] = self.START + prompt["prompt"]
        if self.add_eos and not self.inference:
            prompt["completion"] = prompt["completion"] + self.END

        return prompt

    def add_prompt_tokens(
        self,
        role,
        msg,
        prompt_end=False,
        completion_start=False,
    ) -> str:
        if completion_start:
            new_msg = "\n\n" + msg + self.M_END
        else:
            new_msg = self.ID_START + role + self.ID_END + "\n\n" + msg + self.M_END

        if prompt_end:
            new_msg += self.ID_START + "assistant" + self.ID_END

        return new_msg

    def to_text(self, prompt) -> str: ...

    def conversational_msg(
        self, system_msg, user_msg, assistant_msg, examples=None, **kwargs
    ) -> dict:
        prompt = {}
        prompt["messages"] = []
        if system_msg:
            prompt["messages"].append({"role": "system", "content": system_msg})
        if examples:
            prompt["messages"].extend(self.form_examples(examples, "conversational"))
        prompt["messages"].append({"role": "user", "content": user_msg})
        if not self.inference:
            prompt["messages"].append({"role": "assistant", "content": assistant_msg})
        if kwargs:
            prompt.update(kwargs)
        return prompt


if __name__ == "__main__":
    # fmter = Llama3PromptFormatter(form_condition=form_condition)
    # fmter = GPTPromptFormatter()
    # print(fmter)
    gptparams = GPTParams(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        top_p=0.9,
        # top_k=50,
        # max_tokens=1500,
        stream=True,
    )
    print(gptparams)
