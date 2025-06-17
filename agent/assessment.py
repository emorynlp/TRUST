import logging

import utils.file as uf
from data_struct.variable_struct import VariableMeta

import agent.assessment_funcs as af
from agent.base_module import LLMModule
from agent.form_funcs import registered

logger = logging.getLogger(__name__)


class Assessment(LLMModule):
    """
    Assessment class to generate assessment for variables and generate the next question to ask

    Parameters:
        model_name (str): model name to use for assessment

    Attributes:
        model_name (str): model name to use for assessment
        prompt_formatter (GPTPromptFormatter | ClaudePromptFormatter): Prompt formatter for the model
        model (GptAgent | ClaudeAgent): The LLM model to use for assessment
    """

    def __init__(
        self,
        model_name: str,
        template_path: str = "agent/assessment_prompts.json",
    ):
        super().__init__(model_name)
        self.prompt_formatter.load_funcs_from_registry(
            registered, "shared", "assessment"
        )
        self.assessment_templates = uf.File(template_path).load()

    def info_enough(
        self, var_template: VariableMeta | None, history: list, questions: list
    ) -> bool:
        """
        Determine if the information is enough for assessing the given variable

        Parameters:
            var_template (dict): variable template for the given variable
            history (list): interview history

        Returns:
            bool: True if the information is enough, False otherwise
        """
        if var_template is None:
            # transitions, no more info needed
            return True

        prompt = self.format_prompt(
            self.assessment_templates["info_enough"],
            self.prompt_formatter.set_params(self.model_name, json_format=True),
            # self.prompt_formatter.set_params(self.model_name),
            info_enough=var_template,
            history=history,
            follow_up=questions,
        )
        logger.info(f"INFO_ENOUGH:\n{self.prompt_formatter.to_text(prompt)}")
        response = self.model.call(prompt)
        response = self.model.to_json(response)
        logger.info(f"INFO_ENOUGH RESPONSE: {response}")
        return (
            True
            if isinstance(response, dict) and response["info_enough"] == "yes"
            else False
        )

    def validate_next_question(self, response: dict) -> int | None:
        """
        Validate the next question index from the response

        Parameters:
            response (dict): response from the model

        Returns:
            int: the next question index
        """
        if "qid" in response:
            if isinstance(response["qid"], int):
                return response["qid"]
            else:
                logger.warning(f"Invalid qid: {response['qid']}")
        else:
            logger.warning(f"No qid found in response: {response}")
        return None

    def choose_next_question(self, history: list, questions: list) -> int:
        """
        Choose the next question to ask based on the given interview history and possible next questions

        Parameters:
            history (list): interview history
            questions (list): list of possible next questions

        Returns:
            int: the index of the next question to ask
        """
        if len(questions) == 1:
            # if only one possible next question, return the index
            return questions[0].qid

        prompt = self.format_prompt(
            self.assessment_templates["choose_next_question"],
            self.prompt_formatter.set_params(self.model_name, json_format=True),
            history=history,
            next_question=questions,
        )
        logger.info(f"CHOOSE_NEXT_QUESTION:\n{self.prompt_formatter.to_text(prompt)}")
        response = self.model.call(prompt)
        response = self.model.to_json(response)
        if not isinstance(response, dict):
            raise TypeError("invalid respnse value type")
        qid = self.validate_next_question(response)
        if qid is None:
            qid = questions[0].qid
        logger.info(f"Next question index: {qid}")
        return qid

    def generate(
        self, history: list, variable, prerequisites: dict | None
    ) -> dict | str | None:
        """
        Generate the assessment for the given variable based on the interview history

        Parameters:
            history (list): interview history
            variable (Variable): variable object
            var_template (list): list of variable templates
            prerequisites (dict): prerequisites for the variable

        Returns:
            dict: assessment for the variable
        """
        if variable.metadata.var_type == "rule":
            return af.CAPS5_rule(variable.metadata, prerequisites)
        elif variable.metadata.var_type == "IA":
            func = getattr(af, variable.vid, None)
            return func(prerequisites) if func else {"reason": "", "answer": -1}
        else:
            prompt = self.format_prompt(
                variable.metadata.template,
                self.prompt_formatter.set_params(self.model_name, json_format=True),
                history=history,
                choices=variable.metadata.patterns["range"],
                **variable.metadata.patterns,
            )
            logger.info(f"ASSESSMENT:\n{self.prompt_formatter.to_text(prompt)}")
            response = self.model.call(prompt)
            response = self.model.to_json(response)
            # logger.info(f"ASSESSMENT RESULTS: {response}")
            return response
