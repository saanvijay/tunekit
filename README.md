# TuneKit

Fine-tune a foundation model (GPT-2 + LoRA) and test it — all from a simple browser UI. No code required.

## Project Structure

```
tunekit/
├── pyproject.toml
├── data.json                   ← sample training data
└── src/tunekit/
    ├── trainer.py              ← LoRA fine-tuning logic
    ├── bot.py                  ← inference / chat logic
    └── app.py                  ← Streamlit UI (two panels)
```

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

**1. Install uv** (if not already installed):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# or via Homebrew
brew install uv
```

**2. Clone / enter the project directory:**

```bash
cd /path/to/tunekit
```

**3. Create a virtual environment and install dependencies:**

```bash
uv venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows

uv pip install -e .
```

## Running the App

```bash
tunekit
```

Then open your browser at **http://localhost:8501**.

---

## Using the UI

### Tab 1 — Fine-tune

1. **Add training data** — type an instruction and a response, then click **Add Row** (or press Enter).
2. **Load existing data** — click **Load data.json** to pre-populate the table from the bundled sample file.
3. **Edit the table** — rows are editable directly in the browser.
4. **Set epochs** — use the slider (1–10). Default is 3.
5. **Start training** — click **Start Fine-tuning**. The training log streams live below.

When training finishes, the model is saved to `./finetuned-model/`.

### Tab 2 — Test Bot

1. Click **Load Fine-tuned Model** to load the trained model into memory.
2. Type a prompt in the text box and press **Send** or hit Enter.
3. The bot responds using the fine-tuned model.

---

## How It Works

| Component | Detail |
|---|---|
| **Base model** | `gpt2` (117M params) — downloads automatically from Hugging Face |
| **Fine-tuning** | LoRA via [PEFT](https://github.com/huggingface/peft) — trains ~0.3% of parameters |
| **LoRA rank** | `r=8`, `alpha=32`, target module `c_attn` |
| **UI** | [Streamlit](https://streamlit.io/) |

## Customising the Base Model

To use a different model, edit `src/tunekit/trainer.py`:

```python
BASE_MODEL = "gpt2"          # change to e.g. "distilgpt2" or "facebook/opt-125m"
```

For 7B+ models (e.g. Mistral, LLaMA), also update `target_modules` in the `LoraConfig`:

```python
target_modules=["q_proj", "v_proj"],   # instead of ["c_attn"]
```

## Sample Data Format

`data.json` is a list of instruction/response pairs:

```json
[
  {
    "instruction": "What is the capital of France?",
    "response": "The capital of France is Paris."
  }
]
```

You can edit this file directly or manage all data through the UI.
