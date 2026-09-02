from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from .config import RAGConfig

class Generator:
    _instance = None

    def __new__(cls, config: RAGConfig):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: RAGConfig):
        if self._initialized:
            return
        self.config = config
        self.device = config.device
        self.tokenizer = AutoTokenizer.from_pretrained(config.generation_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.generation_model).to(self.device)
        self._initialized = True

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=self.config.max_output_tokens, do_sample=False)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer