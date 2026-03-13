"""Inference bot: loads the fine-tuned model and answers prompts."""

import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

OUTPUT_DIR = "./finetuned-model"


class Bot:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self, model_path: str = OUTPUT_DIR) -> str:
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            adapter_cfg = os.path.join(model_path, "adapter_config.json")
            if os.path.exists(adapter_cfg):
                from peft import PeftModel
                with open(adapter_cfg) as f:
                    base_model_id = json.load(f)["base_model_name_or_path"]
                base = AutoModelForCausalLM.from_pretrained(base_model_id)
                self.model = PeftModel.from_pretrained(base, model_path)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_path)

            self.model.eval()
            return "Model loaded successfully"
        except Exception as e:
            return f"Error loading model: {e}"

    def chat(self, instruction: str, max_new_tokens: int = 80) -> str:
        if not self.is_loaded:
            return "Model not loaded — load a fine-tuned model first."

        try:
            device = next(self.model.parameters()).device
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
            inputs = {k: v.to(device) for k, v in self.tokenizer(prompt, return_tensors="pt").items()}

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            full = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            marker = "### Response:\n"
            return full.split(marker)[-1].strip() if marker in full else full
        except Exception as e:
            return f"Error generating response: {e}"
