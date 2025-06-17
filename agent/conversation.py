import logging
import re

from agent.base_module import LLMModule
from agent.form_funcs import registered
from utils.file import File

logger = logging.getLogger(__name__)


class Conversation(LLMModule):
    """
    Conversation class to generate responses for the user input and generate the dialogue act tags

    Parameters:
        model_name (str): model name to use for conversation

    Attributes:
        model_name (str): model name to use for conversation
        prompt_formatter (GPTPromptFormatter | ClaudePromptFormatter): Prompt formatter for the model
        model (GptAgent | ClaudeAgent): The LLM model to use for conversation
        simulate_user (dict): Formatted original transcripts for simulating user input
        conv_templates (dict): Conversation templates for the model
    """

    def __init__(
        self, model_name: str, template_file: str, simulate_user: str | None = None
    ):
        super().__init__(model_name)
        self.prompt_formatter.load_funcs_from_registry(
            registered, "shared", "conversation"
        )

        self.simulate_user = File(simulate_user).load() if simulate_user else None
        self.conv_templates = File(template_file).load()

    def validate_tags(self, tags: list) -> list:
        """
        Validate the dialogue act tags and remove invalid tags

        Parameters:
            tags (list): List of dialogue act tags

        Returns:
            list: List of valid dialogue act tags
        """
        if "IS" in tags and "CQ" in tags:
            logger.info("IS and CQ tags found together")
            # tags.remove("CQ")

        for tag in tags:
            if tag not in ["GC", "IS", "CA", "CQ", "GI", "ACK", "EMP", "VAL"]:
                logger.info(f"Invalid tag: {tag}")
                tags.remove(tag)

        if tags == []:
            logger.info("No valid tags found, adding IS to proceed")
            tags.append("IS")
        return tags

    def da_tags(self, history: list, questions: list) -> list:
        """
        Generate dialogue act tags for the next agent response

        Parameters:
            history (list): List of previous user and agent messages
            questions (list): List of next available interview questions

        Returns:
            list: List of valid dialogue act tags
        """
        prompt = self.format_prompt(
            self.conv_templates["predict_tag"],
            params=self.prompt_formatter.set_params(self.model_name, json_format=True),
            # params=self.prompt_formatter.set_params(self.model_name),
            history=history,
            IS_questions=questions,
        )
        logger.info(f"DA_TAGS:\n{self.prompt_formatter.to_text(prompt)}")
        response = self.model.call(prompt)
        response = self.model.to_json(response)
        logger.info(f"GENERATED DA_TAGS: {response}")
        valid_tags = self.validate_tags(response["tags"])
        logger.info(f"Cleaned next tags: {valid_tags}")
        return valid_tags

    def conv_transition(self, history: list, variable) -> str:
        """
        Generate the agent response for the transition variable

        Parameters:
            history (list): List of previous user and agent messages
            var (Variable): Variable object

        Returns:
            dict: Agent response and user input
        """
        transition = " ".join([q.question for q in variable.queries])
        prompt = self.format_prompt(
            self.conv_templates["transition"],
            params=self.prompt_formatter.set_params(self.model_name),
            history=history,
            transition=transition,
        )
        logger.info(f"TRANSITION:\n{self.prompt_formatter.to_text(prompt)}")
        response = self.model.call(prompt)
        response = re.sub(r"\n+", " ", response)
        logger.info(f"GENERATED_TRANSITION: {response}")
        return response

    def conv_question(self, cur_query, cur_tags: list, history: list, variable) -> str:
        """
        Generate the agent response for the non transition variable

        Parameters:
            cur_query (Query): Current query object
            cur_tags (list): Current dialogue act tags
            history (list): List of previous user and agent messages
            variable (Variable): Variable object

        Returns:
            dict: Agent response and user input
        """
        prompt = self.format_prompt(
            self.conv_templates["assessment_question"],
            params=self.prompt_formatter.set_params(self.model_name),
            history=history,
            info_enough=variable.metadata,
            # questions=cur_query.question,
            da_tags={"tags": cur_tags, "question": cur_query.question},
        )
        logger.info(f"AGENT RESPONSE:\n{self.prompt_formatter.to_text(prompt)}")
        response = self.model.call(prompt)
        response = re.sub(r"\n+", " ", response)
        logger.info(f"GENERATED RESPONSE: {response}")
        return response

    def simulate_user_input(self, var, bot_message: str, history: list) -> str:
        """
        Simulate user input using the given agent message and formatted original transcripts for the variable

        Parameters:
            var (Variable): Variable object
            bot_message (str): Last agent message
            history (list): List of previous user and agent messages

        Returns:
            str: Simulated user input
        """
        if self.simulate_user is None:
            raise ValueError("User simulation file is not loaded.")
        reference = (
            self.simulate_user[var.vid] if var.vid in self.simulate_user else None
        )
        prompt = self.format_prompt(
            self.conv_templates["simulate_user"],
            params=self.prompt_formatter.set_params(self.model_name, json_format=True),
            # params=self.prompt_formatter.set_params(self.model_name),
            bot_message=bot_message,
            reference=reference,
            history=history,
        )
        logger.info(f"SIMULATE_USER:\n{self.prompt_formatter.to_text(prompt)}")
        response = self.model.call(prompt)
        response = re.sub(r"\n+", " ", response)
        logger.info(f"SIMULATION: {response}")
        return response

    def generate(self, cur_query, cur_tags, history, variable):
        """
        Generate the agent response for the current query

        Parameters:
            cur_query (Query): Current query object
            cur_tags (list): Current dialogue act tags
            history (list): List of previous user and agent messages
            variable (Variable): Variable object
            var_template (dict): Variable template

        Returns:
            dict: Agent response, user input and dialogue act tags
        """
        if cur_tags is None or "GC" in cur_tags:
            response = self.conv_transition(history, variable)

        else:
            response = self.conv_question(cur_query, cur_tags, history, variable)

        return response


# if __name__ == "__main__":
#     conv = Conversation("gpt-4o")
#     tmpleat = {
#         "system": "Suppose that you are a clinician and conduct a diagnostic interview with the patient about PTSD. Based on the given information, please generate appropriate responses. Return only the response.",
#         "user": "{history}\n{question}",
#         "assistant": "{answer}",
#     }
#     conv.format_prompt("transition", template, history=["history"], question="question")
