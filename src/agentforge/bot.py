"""Inference bot: loads the fine-tuned model and answers prompts."""

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
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.model.eval()
            return f"Model loaded from {model_path}"
        except Exception as e:
            return f"Error loading model: {e}"

    def chat(self, instruction: str, max_new_tokens: int = 80) -> str:
        if not self.is_loaded:
            return "Model not loaded — click 'Load Fine-tuned Model' first."

        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        full = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        marker = "### Response:\n"
        return full.split(marker)[-1].strip() if marker in full else full
