"""Fine-tuning logic supporting multiple models and techniques."""

import shutil
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    PrefixTuningConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

OUTPUT_DIR = "./finetuned-model"
MAX_LENGTH = 128

# ── Supported foundation models ───────────────────────────────────────────────

MODELS: dict[str, dict] = {
    "DeepSeek-R1 (1.5B)": {"id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "lora_targets": ["q_proj", "v_proj"], "stop_strings": ["### Instruction:"]},
    "Qwen2.5     (1.5B)": {"id": "Qwen/Qwen2.5-1.5B",                          "lora_targets": ["q_proj", "v_proj"], "stop_strings": ["### Instruction:"]},
    "SmolLM2     (1.7B)": {"id": "HuggingFaceTB/SmolLM2-1.7B",                 "lora_targets": ["q_proj", "v_proj"], "stop_strings": ["### Instruction:", "\n\n###"]},
}

TECHNIQUES = ["LoRA", "QLoRA", "Prefix Tuning", "Full Fine-tuning"]


# ── Format ────────────────────────────────────────────────────────────────────

def format_example(ex: dict) -> str:
    return f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['response']}"


# ── Main entry point ──────────────────────────────────────────────────────────

def finetune(
    data: list[dict],
    model_choice: str,
    technique: str,
    epochs: int = 3,
    learning_rate: float = 2e-4,
    batch_size: int = 2,
    # LoRA params
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    # Prefix Tuning params
    num_virtual_tokens: int = 16,
    # callbacks
    log_fn=None,
    progress_fn=None,        # progress_fn(step: int, total: int)
) -> str:
    """Run fine-tuning and return the output directory path."""

    def log(msg: str):
        if log_fn:
            log_fn(msg)
        print(msg)

    model_cfg = MODELS[model_choice]
    model_id  = model_cfg["id"]

    log(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts    = [format_example(ex) for ex in data]
    dataset  = Dataset.from_dict({"text": texts})

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")

    log(f"Loading base model: {model_id}")
    if technique == "QLoRA":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id)

    # ── Apply technique ───────────────────────────────────────────────────────
    if technique in ("LoRA", "QLoRA"):
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=model_cfg["lora_targets"],
            lora_dropout=lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, config)

    elif technique == "Prefix Tuning":
        config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=num_virtual_tokens,
        )
        model = get_peft_model(model, config)

    # "Full Fine-tuning" — no PEFT, all params train

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log(f"Technique: {technique} | Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # ── Progress callback ─────────────────────────────────────────────────────
    class _Callback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            if progress_fn:
                progress_fn(0, state.max_steps)

        def on_step_end(self, args, state, control, **kwargs):
            if progress_fn:
                progress_fn(state.global_step, state.max_steps)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                loss = logs.get("loss")
                if loss is not None:
                    log(f"  step {state.global_step:>4} | loss: {loss:.4f}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=1,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[_Callback()],
    )

    log(f"\nTraining: {epochs} epoch(s), {len(data)} examples, lr={learning_rate}, batch={batch_size}\n")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    log(f"\nModel saved → {OUTPUT_DIR}")
    return OUTPUT_DIR


def zip_model(output_dir: str = OUTPUT_DIR) -> str:
    """Zip the output directory and return the zip file path."""
    zip_path = shutil.make_archive("finetuned-model", "zip", output_dir)
    return zip_path
