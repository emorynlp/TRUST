import dataclasses as dc

import gradio as gr

from data_struct.record_struct import LECItem
from utils import file as uf


@dc.dataclass
class LECState:
    username: str = "Anonymous"
    var_idx: int = 0
    lec_vars: dict = dc.field(init=False, repr=False)
    responses: list[list[str]] = dc.field(init=False)

    def __post_init__(self):
        self.lec_vars = uf.File("CAPS/agent_vars/LEC.json").load()
        self.responses = [[] for _ in range(len(self.lec_vars))]

    @property
    def lec_var(self):
        lec_var = self.lec_vars.get(f"lec5_{self.var_idx + 1}", None)
        return lec_var


def click_next_btn(lec_state: LECState):
    if lec_state.var_idx < len(lec_state.lec_vars):
        lec_state.var_idx += 1
    return render_lec_section(lec_state)


def click_prev_btn(lec_state: LECState):
    if lec_state.var_idx > 0:
        lec_state.var_idx -= 1

    return render_lec_section(lec_state)


def render_lec_section(lec_state: LECState):
    visibility = lec_state.var_idx < len(lec_state.lec_vars)
    if visibility:
        mkd = gr.Markdown(f"## {lec_state.var_idx + 1}: {lec_state.lec_var}")
    else:
        mkd = gr.Markdown(
            "## LEC section finished! Click Submit to submit the checklist."
        )

    return (
        mkd,
        gr.update(
            visible=visibility,
            value=lec_state.responses[lec_state.var_idx] if visibility else [],
        ),
        gr.update(visible=lec_state.var_idx > 0),
        gr.update(visible=visibility),
        gr.update(visible=lec_state.var_idx == len(lec_state.lec_vars)),
    )


def on_select(lec_state, value, evt: gr.SelectData):
    lec_state.responses[lec_state.var_idx] = value
    return lec_state


def missing_response(lec_state: LECState):
    missing = any([not lec_state.responses[i] for i in range(len(lec_state.lec_vars))])
    return missing


def submit_lec(lec_state: LECState):
    if missing_response(lec_state):
        gr.Warning("Please complete all sections of the LEC before submitting!")
        visibility = False
    else:
        output_file = uf.File(f"APP/logs/{lec_state.username}_LEC_responses.json")
        output_file.save(lec_state.responses)
        gr.Info("LEC submitted successfully! You can now log out or close the window.")
        visibility = True
    return gr.update(visible=visibility)


bot_img = "APP/trustai.png"

with gr.Blocks() as demo:
    lec_state = gr.State(value=LECState())

    with gr.Row():
        with gr.Column(elem_id="banner", scale=1):
            gr.Image(bot_img, label="TRUST agent bg", interactive=False)
        with gr.Column(elem_id="page_title", scale=6):
            gr.Markdown("""
            # Life Events Checklist for DSM-5 (LEC-5)
            For each event check one or more of the boxes to the right to indicate that:<br>
            1. it happened to you personally;<br>
            2. you witnessed it happen to someone else;<br>
            3. you learned about it happening to a close family member or close friend;<br>
            4. you were exposed to it as part of your job (for example, paramedic, police, military, or other first responder);<br>
            5. you're not sure if it fits; or<br>
            6. it doesn't apply to you. <br><br>
            Be sure to consider your entire life (growing up as well as adulthood) as you go through the list of events.
            """)

    mkd = gr.Markdown("")
    check_group = gr.CheckboxGroup(
        [item.short_desc for item in LECItem],
        label="Check one or more boxes that apply",
        visible=False,
    )

    with gr.Row():
        prev_button = gr.Button("Previous", elem_id="prev_button")
        next_button = gr.Button(
            "Next", inputs=lec_state, elem_id="next_button", variant="primary"
        )
        submit_button = gr.Button(
            "Submit", elem_id="submit_button", variant="primary", visible=False
        )

    logout = gr.Button("Log Out", link="/logout", visible=False)

    demo.load(
        fn=render_lec_section,
        inputs=lec_state,
        outputs=[mkd, check_group, prev_button, next_button, submit_button],
    )
    check_group.select(on_select, inputs=[lec_state, check_group], outputs=lec_state)
    next_button.click(
        fn=click_next_btn,
        inputs=lec_state,
        outputs=[mkd, check_group, prev_button, next_button, submit_button],
    )
    prev_button.click(
        fn=click_prev_btn,
        inputs=lec_state,
        outputs=[mkd, check_group, prev_button, next_button, submit_button],
    )
    submit_button.click(fn=submit_lec, inputs=lec_state, outputs=logout)


if __name__ == "__main__":
    demo.launch(ssl_verify=False)
# lec_state = LECState()
# print(f"Initial LEC state: {lec_state}")
