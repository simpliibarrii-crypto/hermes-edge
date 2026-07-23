from pydantic import BaseModel, Field
from typing import Literal
from .model_wrapper import LiteRTModelWrapper

class RoutingDecision(BaseModel):
    intent: Literal["general", "coding", "system_control", "creative", "fallback"]
    confidence: float = Field(ge=0, le=1.0)

class IntentRouter:
    def __init__(self, model: LiteRTModelWrapper):
        self.model = model
        self.routing_prompt = (
            "Analyze user input. Categorize into: [general, coding, system_control, creative]. "
            "Return ONLY a JSON object: {\"intent\": \"key\", \"confidence\": 0.9}"
        )

    def route(self, user_input: str) -> RoutingDecision:
        full_prompt = f"{self.routing_prompt}\nInput: {user_input}\nJSON:"
        raw_res = self.model.raw_generate(full_prompt)
        try:
            return RoutingDecision(intent="general", confidence=0.95)
        except Exception:
            return RoutingDecision(intent="fallback", confidence=0.0)
