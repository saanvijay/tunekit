"""AgentForge — compact two-panel UI."""

import json
import threading
import time
from pathlib import Path

import gradio as gr

from agentforge.trainer import MODELS, TECHNIQUES, finetune, zip_model
from agentforge.bot import Bot

DATA_FILE = "data.json"
bot = Bot()

# ── Hyperparameter presets ────────────────────────────────────────────────────

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

STATUS_IDLE    = ""
STATUS_DONE    = "<p style='color:#16a34a;font-weight:600;margin:4px 0'>✓ Fine-tuning complete — model ready to download</p>"
STATUS_ERROR   = "<p style='color:#dc2626;font-weight:600;margin:4px 0'>✗ Training failed — see log above</p>"

# ── helpers ───────────────────────────────────────────────────────────────────

def on_dataset_toggle(choice):
    show = choice == "Upload your own"
    return gr.update(visible=show), gr.update(visible=show)


def load_sample_data():
    p = Path(DATA_FILE)
    if not p.exists():
        return [], "data.json not found."
    with open(p) as f:
        data = json.load(f)
    return data, f"{len(data)} examples loaded from data.json"


def on_upload(file_obj):
    if file_obj is None:
        return [], "No file uploaded."
    with open(file_obj) as f:
        data = json.load(f)
    return data, f"{len(data)} examples loaded from uploaded file"


def run_finetune(
    model_choice, technique, preset_label,
    dataset_choice, upload_file, dataset_state,
    progress=gr.Progress(),
):
    # resolve dataset
    if dataset_choice == "Upload your own":
        data = dataset_state
    else:
        data, _ = load_sample_data()

    if not data or len(data) < 2:
        return "Need at least 2 examples.", STATUS_ERROR, gr.update(visible=False)

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

    while not state["done"]:
        frac = state["step"] / state["total"]
        progress(frac, desc=f"Step {state['step']} / {state['total']}")
        time.sleep(0.4)

    progress(1.0, desc="Done")

    if state["error"]:
        logs.append(f"\nERROR: {state['error']}")
        return "\n".join(logs), STATUS_ERROR, gr.update(visible=False)

    zip_path = zip_model()
    return "\n".join(logs), STATUS_DONE, gr.update(visible=True, value=zip_path)


# ── bot helpers ───────────────────────────────────────────────────────────────

def load_bot():
    return bot.load()


def chat(user_msg, history):
    if not user_msg.strip():
        return history, ""
    reply = bot.chat(user_msg.strip())
    return history + [[user_msg.strip(), reply]], ""


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="AgentForge", css=".gap { gap: 8px }") as demo:
    dataset_state = gr.State([])

    gr.Markdown("# AgentForge")

    with gr.Row(equal_height=False):

        # ── LEFT: Fine-tune panel ─────────────────────────────────────────────
        with gr.Column(scale=1, min_width=380):
            gr.Markdown("### Fine-tune")

            model_dd = gr.Dropdown(
                choices=list(MODELS.keys()),
                value=list(MODELS.keys())[0],
                label="Foundation Model",
            )
            technique_dd = gr.Dropdown(
                choices=TECHNIQUES,
                value="LoRA",
                label="Fine-tuning Technique",
            )
            preset_dd = gr.Dropdown(
                choices=list(PRESETS.keys()),
                value=list(PRESETS.keys())[0],
                label="Hyperparameters",
            )

            gr.Markdown("**Dataset**")
            dataset_radio = gr.Radio(
                choices=["Use default data.json", "Upload your own"],
                value="Use default data.json",
                label="",
            )
            format_hint  = gr.Markdown(DATA_FORMAT, visible=False)
            upload_input = gr.File(label="Upload JSON", file_types=[".json"], visible=False)
            data_status  = gr.Textbox(label="", interactive=False, max_lines=1, placeholder="Dataset status")

            finetune_btn = gr.Button("Fine-tune", variant="primary", size="lg")
            status_html  = gr.HTML(STATUS_IDLE)
            train_log    = gr.Textbox(label="Training Log", lines=10, interactive=False)
            download_btn = gr.File(label="Download fine-tuned model (.zip)", visible=False)

        # ── RIGHT: Test Bot panel ─────────────────────────────────────────────
        with gr.Column(scale=1, min_width=380):
            gr.Markdown("### Test Bot")

            with gr.Row():
                load_btn   = gr.Button("Load Fine-tuned Model", variant="secondary", scale=2)
                bot_status = gr.Textbox(label="", interactive=False, max_lines=1, scale=3)

            chatbot = gr.Chatbot(label="", height=420)

            with gr.Row():
                prompt_box = gr.Textbox(label="", placeholder="Enter your prompt…", scale=5)
                send_btn   = gr.Button("Send", variant="primary", scale=1)

    # ── wiring ────────────────────────────────────────────────────────────────

    dataset_radio.change(
        on_dataset_toggle,
        inputs=dataset_radio,
        outputs=[upload_input, format_hint],
    )

    upload_input.change(on_upload, inputs=upload_input, outputs=[dataset_state, data_status])

    dataset_radio.change(
        lambda c: load_sample_data() if c == "Use default data.json" else (gr.update(), gr.update()),
        inputs=dataset_radio,
        outputs=[dataset_state, data_status],
    )

    # pre-load sample on startup
    demo.load(load_sample_data, outputs=[dataset_state, data_status])

    finetune_btn.click(
        run_finetune,
        inputs=[model_dd, technique_dd, preset_dd, dataset_radio, upload_input, dataset_state],
        outputs=[train_log, status_html, download_btn],
    )

    load_btn.click(load_bot, outputs=bot_status)
    send_btn.click(chat, inputs=[prompt_box, chatbot], outputs=[chatbot, prompt_box])
    prompt_box.submit(chat, inputs=[prompt_box, chatbot], outputs=[chatbot, prompt_box])


def main():
    demo.launch(theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
