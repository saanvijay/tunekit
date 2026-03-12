"""TuneKit — Streamlit two-panel UI."""

import json
import threading
import time
from pathlib import Path

import streamlit as st

from tunekit.trainer import MODELS, TECHNIQUES, finetune, zip_model
from tunekit.bot import Bot

DATA_FILE = "data.json"

PRESETS: dict[str, dict] = {
    "Default      (epochs=3, lr=2e-4, batch=2, LoRA r=8)":  {"epochs": 3, "lr": 2e-4, "batch": 2, "lora_r": 8,  "lora_alpha": 32, "lora_dropout": 0.1,  "vt": 16},
    "Fast         (epochs=1, lr=3e-4, batch=4, LoRA r=4)":  {"epochs": 1, "lr": 3e-4, "batch": 4, "lora_r": 4,  "lora_alpha": 16, "lora_dropout": 0.1,  "vt": 8},
    "High Quality (epochs=5, lr=1e-4, batch=2, LoRA r=16)": {"epochs": 5, "lr": 1e-4, "batch": 2, "lora_r": 16, "lora_alpha": 64, "lora_dropout": 0.05, "vt": 32},
}

DATA_FORMAT = """**Expected JSON format:**
```json
[
  {"instruction": "What is the capital of France?", "response": "Paris."},
  {"instruction": "...", "response": "..."}
]
```
Minimum 2 examples required."""


def load_sample_data() -> tuple[list, str]:
    p = Path(DATA_FILE)
    if not p.exists():
        return [], "data.json not found."
    with open(p) as f:
        data = json.load(f)
    return data, f"{len(data)} examples loaded from data.json"


def _init_state():
    if "bot" not in st.session_state:
        st.session_state.bot = Bot()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of (user, assistant) tuples
    if "bot_status" not in st.session_state:
        st.session_state.bot_status = ""
    if "dataset" not in st.session_state:
        data, _ = load_sample_data()
        st.session_state.dataset = data
    if "train_logs" not in st.session_state:
        st.session_state.train_logs = ""
    if "train_status" not in st.session_state:
        st.session_state.train_status = "idle"  # idle | success | error
    if "zip_bytes" not in st.session_state:
        st.session_state.zip_bytes = None


def _finetune_panel():
    st.subheader("Fine-tune")

    model_choice = st.selectbox("Foundation Model", list(MODELS.keys()))
    technique    = st.selectbox("Fine-tuning Technique", TECHNIQUES)
    preset_label = st.selectbox("Hyperparameters", list(PRESETS.keys()))

    st.markdown("**Dataset**")
    dataset_choice = st.radio(
        "dataset_src",
        ["Use default data.json", "Upload your own"],
        label_visibility="collapsed",
    )

    if dataset_choice == "Upload your own":
        st.markdown(DATA_FORMAT)
        uploaded = st.file_uploader("Upload JSON", type=["json"])
        if uploaded is not None:
            data = json.load(uploaded)
            st.session_state.dataset = data
            st.info(f"{len(data)} examples loaded from uploaded file")
    else:
        data, msg = load_sample_data()
        st.session_state.dataset = data
        st.info(msg)

    if st.button("Fine-tune", type="primary", use_container_width=True):
        data = st.session_state.dataset
        if not data or len(data) < 2:
            st.error("Need at least 2 training examples.")
            return

        p = PRESETS[preset_label]
        logs: list[str] = []
        state = {"step": 0, "total": 1, "done": False, "error": None}

        def log_fn(msg):
            logs.append(msg)

        def progress_fn(step, total):
            state["step"] = step
            state["total"] = max(total, 1)

        def _train():
            try:
                finetune(
                    data=data,
                    model_choice=model_choice,
                    technique=technique,
                    epochs=p["epochs"],
                    learning_rate=p["lr"],
                    batch_size=p["batch"],
                    lora_r=p["lora_r"],
                    lora_alpha=p["lora_alpha"],
                    lora_dropout=p["lora_dropout"],
                    num_virtual_tokens=p["vt"],
                    log_fn=log_fn,
                    progress_fn=progress_fn,
                )
            except Exception as e:
                state["error"] = str(e)
            finally:
                state["done"] = True

        threading.Thread(target=_train, daemon=True).start()

        progress_bar  = st.progress(0, text="Starting…")
        log_container = st.empty()

        while not state["done"]:
            frac = state["step"] / state["total"]
            progress_bar.progress(frac, text=f"Step {state['step']} / {state['total']}")
            log_container.code("\n".join(logs), language=None)
            time.sleep(0.4)

        progress_bar.progress(1.0, text="Done")
        final_log = "\n".join(logs)
        log_container.code(final_log, language=None)

        # persist result in session state for re-renders
        st.session_state.train_logs = final_log

        if state["error"]:
            st.session_state.train_status = "error"
            st.session_state.zip_bytes = None
        else:
            zip_path = zip_model()
            st.session_state.train_status = "success"
            with open(zip_path, "rb") as zf:
                st.session_state.zip_bytes = zf.read()

    # ── post-training status & log ─────────────────────────────────────────────
    status = st.session_state.train_status
    if status == "success":
        st.success("Fine-tuning complete — model ready to download")
    elif status == "error":
        st.error("Training failed — see log above")

    if st.session_state.train_logs and status != "idle":
        st.text_area("Training Log", st.session_state.train_logs, height=220, key="persisted_log")

    if st.session_state.zip_bytes is not None:
        st.download_button(
            "Download fine-tuned model (.zip)",
            data=st.session_state.zip_bytes,
            file_name="finetuned-model.zip",
            mime="application/zip",
        )


def _bot_panel():
    st.subheader("Test Bot")

    col_btn, col_status = st.columns([2, 3])
    with col_btn:
        if st.button("Load Fine-tuned Model", use_container_width=True):
            msg = st.session_state.bot.load()
            st.session_state.bot_status = msg
    with col_status:
        if st.session_state.bot_status:
            st.info(st.session_state.bot_status)

    # Chat history
    chat_box = st.container(height=420)
    with chat_box:
        for user_msg, bot_msg in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(user_msg)
            with st.chat_message("assistant"):
                st.write(bot_msg)

    # Input
    prompt = st.chat_input("Enter your prompt…")
    if prompt:
        reply = st.session_state.bot.chat(prompt.strip())
        st.session_state.chat_history.append((prompt.strip(), reply))
        st.rerun()


def main():
    """Entry point for the `tunekit` CLI command."""
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "streamlit", "run", __file__], check=True)


def _app():
    """Called by Streamlit when running this file directly."""
    st.set_page_config(page_title="TuneKit", layout="wide")
    _init_state()

    st.title("TuneKit")

    left, right = st.columns(2, gap="large")
    with left:
        _finetune_panel()
    with right:
        _bot_panel()


# Streamlit executes the module top-level on each rerun
from streamlit.runtime.scriptrunner import get_script_run_ctx
if get_script_run_ctx() is not None:
    _app()
