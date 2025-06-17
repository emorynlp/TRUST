import dataclasses as dc
import sys

from agent.framework import Chatbot, Tracker
from data_struct.record_struct import State
from utils.config import get_config


@dc.dataclass
class TraumaBot:
    agent: Chatbot = dc.field(init=False)
    tracker: Tracker = dc.field(init=False)
    bot_state: State = dc.field(init=False)
    checkpoint: str | None = None
    utter_id: int = 0

    def __post_init__(self):
        config_file = "configs/agent.conf"
        config = get_config(
            config_name="claude_agent",
            config_file=config_file,
        )
        self.agent = Chatbot(config=config)
        if self.checkpoint:
            self.agent.load_checkpoint(self.checkpoint)
        self.tracker = self.agent.tracker
        self.bot_state = self.agent.state

    def respond(self) -> str:
        if self.bot_state.user_input:
            self.agent.generate_next_actions()

            self.agent.assess_variable(self.bot_state.variable)

        while True:
            if self.bot_state.signal.any():
                # update the state based on the signal
                self.tracker.add_history(self.bot_state)
                self.agent.update_state()

            if self.bot_state.signal.end:
                # the end of the interview
                sys.exit(0)

            if self.bot_state.query is None:
                # rule variables, no questions to ask
                # proceed to variable assessment
                self.bot_state.signal.next_variable = True
                self.agent.assess_variable(self.bot_state.variable)
            else:
                self.agent.generate_response()
                break

        return self.bot_state.response
