import random

class LiteRTModelWrapper:
    def __init__(self, model_path: str):
        self.model_path = model_path

    def generate(self, prompt: str, tier: str = "small") -> str:
        print(f"[Model Log] Using {tier} tier model for generation...")
        return f"Generated response from {tier} model for prompt: {prompt[:20]}..."

    def raw_generate(self, prompt: str) -> str:
        return "general" 
