import dataclasses as dc
import pathlib as pl
import re
from datetime import datetime

import gradio as gr
import numpy as np
import transformers as hf
from fastrtc import (
    AdditionalOutputs,
    AlgoOptions,
    KokoroTTSOptions,
    ReplyOnPause,
    WebRTC,
    audio_to_bytes,
    get_tts_model,
)
from gradio.context import Context

import utils.file as uf
from APP.trauma_bot import TraumaBot

whisper = hf.pipeline(  # type: ignore
    "automatic-speech-recognition",
    model="openai/whisper-base.en",
    # return_timestamps=True,
)


def transcribe(audio):
    sr, y = audio
    print(f"sample rate: {sr}, audio shape: {y.shape}")

    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)

    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    transcription = whisper(audio_to_bytes(audio))

    return transcription["text"]  # type: ignore


tts_client = get_tts_model("kokoro")
options = KokoroTTSOptions(voice="af_heart", speed=1.0, lang="en-us")


@dc.dataclass
class AppState(TraumaBot):
    user_needs_to_press_record: bool = True
    username: str = "anonymous"

    def save_state(self, **kwargs):
        exclude_keys = [
            "tracker",
            "bot_state",
            "agent",
            "checkpoint",
        ]
        state_dict = {k: v for k, v in dc.asdict(self).items() if k not in exclude_keys}
        state_file = uf.File(f"APP/logs/{self.username}/state.json")
        state_file.save(state_dict, **kwargs)

    def update_state(self, state_dict: dict):
        """Update the state with new values."""
        for key, value in state_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: {key} is not a valid attribute of AppState.")


def more_recent_checkpoint(
    fmt: str = "%b%d_%H-%M-%S", pattern: str = r"", **checkpoints
) -> str:
    regex = re.compile(pattern)
    return max(checkpoints, key=lambda d: datetime.strptime(regex.findall(d)[0], fmt))


def get_checkpoint(checkpoint_folder: pl.Path) -> str | None:
    """Retrieve the checkpoint for a given user folder."""
    files = [p for p in checkpoint_folder.rglob("*.jsonl") if p.is_file()]
    if files:
        files = sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)
        # return str(files[0])
        return re.findall(r"[A-Z]\w\w\d+_\d+-\d+-\d+_\w+", files[0].stem)[0]
    return None


def start_up(chat, state: AppState):
    # Initialize the state when the app starts
    # if not state.turns:
    if state.user_needs_to_press_record:
        state.user_needs_to_press_record = False
        text = "Hi there! I'm TRUST, a clinical AI. Nice to meet you!"
        chat.append({"role": "assistant", "content": text})
        yield AdditionalOutputs(chat, state)
        print("appended start message to chat")
        for chunk in tts_client.stream_tts_sync(
            text,
            options=options,
        ):
            yield chunk
            yield AdditionalOutputs(chat, state)


def respond(microphone: tuple[int, np.ndarray], chat: list[dict], state: AppState):
    user_input = transcribe(microphone)
    chat.append({"role": "user", "content": user_input})
    yield AdditionalOutputs(chat, state)
    if state.user_needs_to_press_record:
        state.user_needs_to_press_record = False
    else:
        state.bot_state.user_input = user_input
    response = state.respond()
    chat.append({"role": "assistant", "content": response})
    state.bot_state.response = response
    yield AdditionalOutputs(chat, state)

    for chunk in tts_client.stream_tts_sync(response, options=options):
        #     # print("chunk", i, time.time() - start)
        yield chunk
        #     # print("finished tts", time.time() - start)
        yield AdditionalOutputs(chat, state)


def save_on_session_end(state: AppState):
    """Save the state on session end."""
    if not state.username:
        return
    # Save the chat history to a file
    print(f"Saved state for {state.username}...")
    state.save_state(indent=2)
    state.tracker.save(state.bot_state)


bot_img = "APP/trustai.png"
# microphone_waveform = {"show_recording_waveform": True}

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(elem_id="banner", scale=1):
            gr.Image(bot_img, label="TRUST agent bg", interactive=False)
        with gr.Column(elem_id="page_title", scale=4):
            gr.Markdown("""
            # Welcome to TRUST AI 😊
            """)
    with gr.Row():
        with gr.Column(elem_id="main"):
            chat = gr.Chatbot(label="", elem_id="chat_box", type="messages")

            microphone = WebRTC(
                label="Voice Chat",
                modality="audio",
                mode="send-receive",
            )
            logout = gr.Button("Log Out", link="/logout")

    state = gr.State(value=AppState(), delete_callback=lambda v: print("STATE DELETED"))

    @Context.root_block.load(inputs=[state, chat], outputs=[state, chat])
    def on_load(state: AppState, chat: list[dict], request: gr.Request):
        username = request.username or "anonymous"
        user_path = state.tracker.log_path + f"/{username}"
        checkpoint = get_checkpoint(pl.Path(user_path))
        loaded_state = AppState(checkpoint=checkpoint, username=username)
        loaded_state.tracker.log_path = user_path
        saved_chat = uf.File(f"{loaded_state.tracker.log_path}/state.json")
        if saved_chat.path.exists():
            loaded_state.update_state(saved_chat.load())
        chat.clear()
        for record in loaded_state.tracker.history:
            if record.response:
                chat.append({"role": "assistant", "content": record.response})
            if record.user_input:
                chat.append({"role": "user", "content": record.user_input})
        return loaded_state, chat

    @demo.unload
    def on_unload():
        """Save the state when the page is unloaded."""
        print("Unloading the app, saving state...")
        save_on_session_end(state.value)

    microphone.stream(
        fn=ReplyOnPause(
            respond,
            output_sample_rate=16000,
            startup_fn=start_up,
            can_interrupt=False,
            algo_options=AlgoOptions(
                audio_chunk_duration=3,
                started_talking_threshold=1,
                speech_threshold=1,
            ),
        ),
        inputs=[microphone, chat, state],
        outputs=[microphone],
    )
    microphone.on_additional_outputs(
        lambda c, a: (c, a),
        outputs=[chat, state],
        # queue=False,  # show_progress="hidden"
    )


if __name__ == "__main__":
    auths = uf.File("APP/auth.json").load()
    demo.launch(auth=auths, ssl_verify=False)
    # demo.launch(ssl_verify=False)
