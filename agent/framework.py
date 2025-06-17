import logging
import os
import signal
import sys
from collections import deque
from dataclasses import asdict
from datetime import datetime

import fire

import utils.file as uf
from agent.assessment import Assessment
from agent.conversation import Conversation

# from compare_results import save_comparison
from data_struct.record_struct import Record, Score, State
from data_struct.variable_struct import Query, SectionVars, Variable
from utils.config import get_config
from utils.logger import Logger

logger = logging.getLogger()


class Tracker:
    """Tracker tracks the conversation history and assessment scores.
    It also loads and saves the conversation history and assessment scores to log files.

    Attributes:
        date_stamp (str): The date and time when the tracker was initialized.
        maxsize (int): The maximum size of the history deque.
        history (deque): A deque that stores the conversation history.
        last_state (dict | None): The last state of the conversation.
        log_path (str): The path to the log directory.
        history_file (File): A File object that stores the conversation history.
        assessment_file (File): A File object that stores the assessment scores.
    """

    def __init__(
        self,
        maxsize: int = 5,
        log_path: str | None = None,
        filename: str | None = None,
    ):
        """Initialize the Tracker with a maximum size for the history deque, log path, and log file name.

        Args:
            maxsize (int): The maximum size of the history deque.
            log_path (str|None): The path to the log directory.
                If None, the history and assessment files will not be saved.
            filename (str|None): The name of the log file.
                If None, the date stamp will be used.
        """
        self.date_stamp: str = datetime.now().strftime("%b%d_%H-%M-%S")
        self.maxsize: int = maxsize
        self.history = deque([], maxlen=maxsize)
        self.last_state: dict | None = None
        self.log_path: str | None = log_path
        self.filename: str | None = filename
        self.history_file, self.assessment_file = self._load_log_file(
            self.filename, self.date_stamp
        )

    def _load_log_file(
        self, filename: str | None = None, date_stamp: str | None = None
    ) -> tuple:
        """Load the history and assessment files based on the date stamp."""
        if self.log_path is None:
            return None, None

        history_fname = (
            f"history_{filename}.jsonl"
            if self.filename
            else f"history_{date_stamp}.jsonl"
        )
        assessment_fname = (
            f"assessment_{filename}.jsonl"
            if self.filename
            else f"assessment_{date_stamp}.jsonl"
        )
        history_file = uf.File(os.path.join(self.log_path, history_fname))
        assessment_file = uf.File(os.path.join(self.log_path, assessment_fname))
        return history_file, assessment_file

    def get_history(self) -> list[tuple]:
        """Get the conversation history in the deque."""
        return [(record.response, record.user_input) for record in self.history]

    def get_var_history(self, vid: str | list[str], full: bool = False) -> list:
        """Get the conversation history for a specific variable ID.

        Args:
            vid (str|list[str]): The string or list of string of variable ID/name to filter the history.
            full (bool): A flag to return the full history record.

        Returns:
            list: A list of conversation history records.
        """
        if not self.history_file.path.exists():
            return []

        if isinstance(vid, list):
            var_history = []
            for v in vid:
                var_history.extend(
                    [h for h in self.history_file.load() if h["vid"] == v]
                )
        else:
            var_history = [h for h in self.history_file.load() if h["vid"] == vid]
        if full:
            return var_history
        return [(h["response"], h["user_input"]) for h in var_history]

    def add_history(self, state: State):
        record = state.to_record()
        self.history.append(record)
        if self.history_file is not None:
            self.history_file.append(asdict(record))

    def add_assessment(self, **kwargs):
        """Add a score history record to the deque and log file if there is assessment file.

        Args:
            kwargs: A dictionary of score attributes.
        """
        score = Score(**kwargs)
        if self.assessment_file is not None:
            self.assessment_file.append(asdict(score))

    def get_assessment(self, *vars) -> dict | None:
        """Get the assessment scores for the specified variable IDs.

        Args:
            vars (str): variable IDs/names to filter the assessment scores.

        Returns:
            dict: A dictionary of variable ID and the corresponding assessment score.
        """
        if not self.assessment_file.path.exists():
            return None
        scores = self.assessment_file.load()
        if "IA" in vars:
            return {score["vid"]: score["answer"] for score in scores}
        return {
            score["vid"]: score["answer"] for score in scores if score["vid"] in vars
        }

    def _load_history(self, file: uf.BaseFile):
        """Load the conversation history from a file.

        Args:
            file (str|list|File): The file path, list of records, or File object to load the history.
        """
        try:
            data = file.load()
        except Exception:
            raise ValueError("Invalid file type")

        if "next_qid" in data[-1]:
            self.last_state = data[-1] if data else None
            data = data[:-1]  # remove the last state record
            file.save(data)

        self.history = deque(
            [Record(**record) for record in data[-self.maxsize :]], maxlen=self.maxsize
        )

    def from_checkpoint(
        self, checkpoint: str, load_path: str | None = None, remove_old: bool = False
    ):
        """Load the conversation history and assessment scores from a checkpoint.

        Args:
            checkpoint (str): The checkpoint name.
            load_path (str): The path to the log directory.
            remove_old (bool): A flag to remove the old history and assessment files.
        """
        if load_path:
            self.log_path = load_path

        # self.date_stamp = checkpoint
        prev_history_file, prev_assessment_file = self._load_log_file(
            filename=checkpoint
        )
        self._load_history(prev_history_file)

        if self.history_file is not None:
            try:
                prev_history_file.copy_to(self.history_file)
                prev_assessment_file.copy_to(self.assessment_file)
            except FileNotFoundError as e:
                logger.error(f"Error copying files {e}")

            if remove_old:
                prev_assessment_file.delete()
                prev_history_file.delete()

    def save(self, state: State):
        record = asdict(state.to_record())
        logger.info("exit save")
        print("exit save")
        next_action = state.next_qid or state.next_tags
        if next_action:
            record["next_qid"] = state.next_qid
            record["next_tags"] = state.next_tags
        if self.history_file is not None:
            self.history_file.append(record)


class Chatbot:
    """Chatbot is a class that controls the dialogue flow and generates responses.

    Args:
        config (dict): A dictionary of configuration parameters.
        simulation_file (str): The path to the simulation file that provide original transcript as the
            reference to simulate user input.
        log_history (bool): A flag to log the conversation history.

    Attributes:
        config (dict): A dictionary of configuration parameters.
        log_history (bool): A flag to log the conversation history.
        section_vars (SectionVars): A SectionVars object that stores the variables for the current section.
        tracker (Tracker): A Tracker object that stores the conversation history and assessment scores.
        assessment (Assessment): The Assessment model that generates the assessment scores.
        conversation (Conversation): The Conversation model that generates the conversation responses.
    """

    def __init__(
        self,
        config: dict,
        simulation_file: str | None = None,
        log_fname: str | None = None,
    ):
        self.config: dict = config
        # self.sections: Iterator = (sec for sec in config["sections"])
        self.sections: list[str] = config["sections"]
        self.simulation_file: str | None = simulation_file
        self.log_fname: str | None = log_fname
        self._load_components()

    # load models for assessment and conversation
    def _load_components(self):
        self.tracker = Tracker(
            maxsize=5,
            log_path=self.config["log_dir"],
            filename=self.log_fname,
            # assessment_path=self.config["log_dir"],
        )
        self.state: State = self.load_state()
        self.assessment = Assessment(self.config["assessment_model_name"])
        self.conversation = Conversation(
            self.config["conversation_model_name"],
            self.config["conv_templates"],
            simulate_user=self.simulation_file,
        )
        # self._register_exit_signals()

    def load_checkpoint(self, checkpoint: str):
        self.tracker.from_checkpoint(checkpoint)
        self.load_state()

    def _register_exit_signals(self):
        def handler(signum, frame):
            logger.info("Exiting gracefully...")
            self.tracker.save(self.state)
            sys.exit(0)

        signal.signal(signal.SIGINT, handler)  # Handle Ctrl+C
        signal.signal(signal.SIGTERM, handler)  # Handle termination signal

    def load_state(self) -> State:
        if self.tracker.last_state:
            section_vars = self.load_var_data(self.tracker.last_state["section"])
            state = State.from_tracker(self.tracker.last_state, section_vars)

        else:
            section = self.sections.pop(0)
            section_vars = self.load_var_data(section)
            variable = section_vars[0]
            query = self.get_cur_query(variable)
            da_tags = ["GC", "GI"]
            state = State(
                section,
                section_vars,
                variable,
                query,
                da_tags,
            )
        return state

    def update_state(self):
        next_section, next_section_vars = None, None

        if self.state.signal.next_variable:
            next_var_idx = self.state.variable.var_idx + 1
            if next_var_idx >= len(self.state.section_vars):
                # no more variable in the current section, go to next section
                self.state.signal.next_section = True

        if self.state.signal.next_section:
            # next_section = next(self.sections)
            next_section = self.sections.pop(0)
            if next_section is None:
                # no more section to continue
                self.state.signal.end = True
            else:
                next_section_vars = self.load_var_data(next_section)

        self.state.update(next_section, next_section_vars)
        if self.state.query is None:
            self.state.query = self.get_cur_query(self.state.variable)

    def get_history(self, vid: str | list[str] | None = None) -> list:
        cur_turn = self.state.to_turn()
        if vid:
            tracker_history = self.tracker.get_var_history(vid)
            if vid == self.state.variable.vid and cur_turn:
                # if the variable is the current one, append the current turn
                tracker_history.append(cur_turn)
        else:
            tracker_history = self.tracker.get_history()
            if cur_turn:
                tracker_history.append(cur_turn)
        return tracker_history

    def get_available_questions(
        self, cur_query, variable, include_next: bool = True
    ) -> list:
        """Get the available question list for the next question.

        Args:
            cur_query (Query): The current query object.
            variable (Variable): The current variable object.
            var_template (dict): The variable template dictionary.
            include_next (bool): A flag to include the next variable's questions.

        Returns:
            list: A list of available questions for the next turn.
        """
        # if cur_query is core: return level 1 questions (children)
        # if cur_query is level 1: return child and other level 1 questions
        # if cur_query is >level 1: return child (if any), siblings and other level 1 questions
        # if cur_query is None: return next variable's core question (if any)
        # if cur_query is None and no core question in next var: return next variable's level 1 questions

        questions = []
        if cur_query is not None and variable.metadata is not None:
            # exclude transition, instructions etc.
            questions.extend([variable[c] for c in cur_query.children])
            questions.extend(variable.get_sibling_queries(cur_query))

            if cur_query.level > 1:
                questions.extend(variable.get_parent_queries(cur_query))

        if include_next and (not questions or all([q.level > 1 for q in questions])):
            # if no questions or all questions are below level 1, add next var's questions
            questions.extend(
                self.state.section_vars.get_queries_in_next_var(variable.var_idx)
            )

        return questions

    def anti_infinite_loop(self, variable, max_len: int = 10):
        """incase of infinite loop if the variable is not ending"""
        if self.tracker.history_file.path.exists():
            cur_var_len = len(self.get_history(variable.vid))
            if cur_var_len > max_len:
                raise ValueError(
                    f"Variable {variable.vid} reached max length {max_len}"
                )

    def ask_dependent_var(self, prerequisites: list, has_queries: bool) -> bool:
        """Check if the dependent variable needs to be asked based on the prerequisites.

        Args:
            variable (Variable): The current variable object.
        """
        if has_queries:
            if "IA" in prerequisites:
                return False
            elif "calculation" in prerequisites:
                return False
            else:
                prerequisite_var = prerequisites[0]
                prev_assessment = self.tracker.get_assessment(prerequisite_var)
                if prev_assessment is None:
                    raise ValueError(
                        f"Dependent assessment {prerequisite_var} is None."
                    )
                prev_score = prev_assessment[prerequisite_var]
                if prev_score > 0:
                    return True
                else:
                    return False
        else:
            return False

    def load_var_data(self, section: str) -> SectionVars:
        var_file = os.path.join(self.config["var_dir"], f"{section}.json")
        template_file = os.path.join(
            os.path.join(self.config["template_dir"], f"{section}_var_template.json")
        )
        section_vars = SectionVars(query_file=var_file, meta_file=template_file)
        return section_vars

    def get_cur_query(
        self, variable: Variable, cur_query_idx: int | None = None
    ) -> Query | None:
        """Get the current query to ask based on the query hierarchy within the variable.

        Args:
            variable (Variable): The current variable object.

        Returns:
            Query: The current query to ask.
        """
        if cur_query_idx is not None:
            # resume from the previous session with a start query idx provided
            cur_query = variable.get_query_by_qid(cur_query_idx)
            # return cur_query if variable.metadata else None
            return cur_query

        core_queries = variable.get_queries_by_level()
        if core_queries:
            return core_queries[0]
        else:
            queries = variable.get_queries_by_level(level=1)
            if variable.prerequisites:
                if self.ask_dependent_var(
                    variable.prerequisites, True if variable.queries else False
                ):
                    return queries[0]
            else:
                return queries[0]
        return None

    def get_user_input(self, bot_message: str, variable: Variable, history: list):
        if self.simulation_file is None:
            print(f"Chatbot: {bot_message}")
            user_input = None
            while not user_input:
                user_input = input("User: ")
                if not user_input:
                    print("Please enter a valid input")
        else:
            user_input = self.conversation.simulate_user_input(
                variable, bot_message, history
            )
        return user_input

    def skip_assessment(self, variable: Variable) -> bool:
        # skip the assessment for dependent variables that do not need to be assessed

        no_current_query = self.state.query is None
        rule_or_IA = variable.metadata and variable.metadata.var_type not in [
            "rule",
            "IA",
        ]
        not_calculation = "calculation" not in variable.prerequisites
        return all([no_current_query, rule_or_IA, not_calculation])

    def assess_variable(self, variable: Variable):
        skip = self.state.variable.metadata is None or self.tracker.get_assessment(
            self.state.variable.vid
        )

        if self.state.to_assess() and not skip:
            if self.skip_assessment(variable):
                # skip the assessment for this variable
                score = {"reason": "", "answer": -1}
            else:
                vids = variable.vid if variable.queries else variable.prerequisites
                history = self.get_history(vids)
                score = self.assessment.generate(
                    history,
                    variable,
                    prerequisites=self.tracker.get_assessment(
                        *variable.prerequisites if variable.prerequisites else []
                    ),
                )
                if not isinstance(score, dict):
                    raise ValueError("Score is not returned properly")

            self.tracker.add_assessment(
                section=self.state.section, vid=self.state.variable.vid, **score
            )

    def generate_response(self):
        if not self.state.response:
            self.anti_infinite_loop(self.state.variable)
            response = self.conversation.generate(
                self.state.query,
                self.state.cur_tags,
                self.get_history(),
                self.state.variable,
            )
            self.state.response = response

    def generate_user_input(self, user_input: str | None = None):
        if not self.state.response:
            raise ValueError("Response is not generated yet")

        if user_input:
            self.state.user_input = user_input
        if not self.state.user_input:
            if self.simulation_file is None:
                print(f"Chatbot: {self.state.response}")
                user_input = None
                while not user_input:
                    user_input = input("User: ")
                    if not user_input:
                        print("Please enter a valid input")
            else:
                user_input = self.conversation.simulate_user_input(
                    self.state.variable, self.state.response, self.get_history()
                )
            self.state.user_input = user_input

    def generate_next_actions(self):
        # ? what if this is the last one in the section?
        next_queries = self.get_available_questions(
            self.state.query, self.state.variable
        )

        if not self.state.info_enough:
            info_enough = self.assessment.info_enough(
                self.state.variable.metadata,
                self.get_history(self.state.variable.vid),
                next_queries,
            )
            self.state.info_enough = info_enough

        if not self.state.next_tags:
            next_tags = self.generate_next_tags(next_queries, self.state.info_enough)
            self.state.next_tags = next_tags

    def generate_next_tags(self, next_queries: list, info_enough: bool):
        next_tags = self.conversation.da_tags(
            history=self.get_history(), questions=next_queries
        )

        if info_enough:
            # proceed to next variable
            if "IS" not in next_tags:
                next_tags.append("IS")
            self.state.signal.next_variable = True
        else:
            if "IS" in next_tags:
                next_qid = self.assessment.choose_next_question(
                    history=self.get_history(self.state.variable.vid),
                    questions=next_queries,
                )

                next_query = self.state.variable.get_query_by_qid(next_qid)
                if next_query:
                    self.state.signal.next_query = True
                    self.state.next_qid = next_qid
                else:
                    self.state.signal.next_variable = True
            else:  # ask open-end interview questions following da tags
                self.state.signal.next_turn = True
        return next_tags

    def chat(self, checkpoint: str | None = None):
        # get current state
        # if new, start from the first variable
        # if not new, get the last variable
        if checkpoint:
            self.load_checkpoint(checkpoint)

        # init the current state

        while True:
            if self.state.signal.any():
                # update the state based on the signal
                self.tracker.add_history(self.state)
                self.update_state()

            if self.state.signal.end:
                # the end of the interview, or the user wants to quit
                break

            if self.state.query is None:
                # rule variables, no questions to ask
                # proceed to variable assessment
                self.state.signal.next_variable = True
            else:
                self.generate_response()
                self.generate_user_input()
                self.generate_next_actions()

            # if next variable signal -> assess variable
            self.assess_variable(self.state.variable)


def main(
    simulation_file,
    config_name="claude_agent",
    checkpoint=None,
):
    # simulation_file, checkpoint = check_file(simulation_file)

    # if simulation_file is None:
    #     return

    config_file = "configs/agent.conf"
    config = get_config(
        config_name=config_name,
        config_file=config_file,
    )
    # config["log_dir"] = str(uf.File(simulation_file).path.parent)
    config["log_dir"] = "agent/test_log"
    LOGGER = Logger(
        logger,
        log_level=logging.INFO,
    ).config(
        # log_name=uf.File(simulation_file).path.stem.replace("_simulation", ""),
        log_name=config_name,
        log_dir=config["log_dir"],
        console_log=False,
        # time_suffix=False,
    )
    config["log_filename"] = LOGGER.log_name
    logger.info(f"CONFIG:\n{config}")
    chatbot = Chatbot(config, simulation_file)
    chatbot.chat(checkpoint=checkpoint)


def check_variable(section_vars):
    for variable in section_vars:
        core_queries = variable.get_queries_by_level()
        if not core_queries and not variable.prerequisites:
            print(variable.vid)


if __name__ == "__main__":
    main(
        # MOD: change the simulation file path
        # simulation_file="agent/samples/22018/audio_only_simulation.json",
        simulation_file=None
    )
    # fire.Fire(main)
