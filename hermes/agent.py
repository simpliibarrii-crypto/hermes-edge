"""
Hermes Edge Agent — On-Device AI Agent Framework

Combines DeepSeek-style reasoning + Hermes tool calling + LiteRT-LM runtime
into a coherent agent loop for on-device inference.

Usage:
    from hermes.agent import HermesAgent
    from hermes.tools import ToolRegistry
    from hermes.litert_model import LiteRTModel

    model = LiteRTModel("/path/to/model.litertlm")
    agent = HermesAgent(model)
    response = agent.run("What's the weather?")
"""

import logging
import time
from dataclasses import dataclass, field

from hermes.chat_template import build_prompt, Message
from scripts.deepseek_reasoning_template import ReasoningPipeline, ReasoningResult
from scripts.hermes_tool_format import ToolRegistry, HermesToolFormatter
from scripts.dspark_draft import DSparkDraftEngine, DSparkConfig, NGramDraftModel

log = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    max_tool_rounds: int = 5
    max_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 40
    use_reasoning: bool = True
    use_speculative_decoding: bool = True
    draft_k: int = 4
    system_prompt: str = ""


DEFAULT_SYSTEM = (
    "You are Hermes Edge, an on-device AI agent powered by Raven AI ecosystem. "
    "You run fully offline via LiteRT-LM on iPhone 16 / Android. "
    "You have access to tools and can reason step by step. "
    "Always prefer local computation. Be helpful, concise, and accurate."
)


@dataclass
class AgentTurn:
    user_input: str = ""
    assistant_response: str = ""
    thinking: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    tokens_used: int = 0


@dataclass
class Conversation:
    messages: list[Message] = field(default_factory=list)
    turns: list[AgentTurn] = field(default_factory=list)

    def add_user(self, text: str) -> None:
        self.messages.append(Message(role="user", content=text))

    def add_assistant(self, text: str) -> None:
        self.messages.append(Message(role="assistant", content=text))

    def add_tool_result(self, name: str, content: str) -> None:
        self.messages.append(Message(role="tool", content=f"<tool_response>{name}: {content}</tool_response>"))


class HermesAgent:
    """Full agent loop combining reasoning, tool calling, and speculative decoding."""

    def __init__(
        self,
        model=None,
        tool_registry: ToolRegistry | None = None,
        config: AgentConfig | None = None,
    ):
        self.model = model
        self.config = config or AgentConfig()
        self.tools = tool_registry or ToolRegistry()
        self.conversation = Conversation()
        self.reasoning = ReasoningPipeline(use_reasoning=self.config.use_reasoning)
        self.tool_formatter = HermesToolFormatter()
        self.draft_engine: DSparkDraftEngine | None = None
        self._init_draft_engine()

    def _init_draft_engine(self) -> None:
        if self.config.use_speculative_decoding and self.model is not None:
            vocab_size = getattr(self.model, "vocab_size", 32000)
            draft = NGramDraftModel(vocab_size=vocab_size, max_order=3)
            dconfig = DSparkConfig(
                draft_k=self.config.draft_k,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
            )
            self.draft_engine = DSparkDraftEngine(self.model, draft, dconfig)

    def set_model(self, model) -> None:
        self.model = model
        self._init_draft_engine()

    def register_tool(self, name: str, description: str, func, parameters: dict | None = None) -> None:
        self.tools.register(name, description, func, parameters)

    def run(self, user_input: str, context: str | None = None) -> str:
        """Process a user input through the full agent pipeline."""
        if not self.model:
            return "Error: No model loaded."

        turn = AgentTurn(user_input=user_input)
        start = time.perf_counter()

        if self.config.use_reasoning:
            prompt = self.reasoning.build_reasoning_prompt(user_input, context)
        else:
            tool_defs = self.tools.get_defs()
            self.tool_formatter.set_tools(tool_defs)
            prompt = self.tool_formatter.build_tool_prompt(user_input, context=context)

        raw_output = self._generate(prompt)
        turn.tokens_used = len(raw_output) // 4

        parsed = self.reasoning.parse_response(raw_output)
        turn.thinking = parsed.thinking
        turn.assistant_response = parsed.answer
        turn.tool_calls = parsed.tool_calls

        tool_round = 0
        while parsed.tool_calls and tool_round < self.config.max_tool_rounds:
            tool_round += 1
            for call in parsed.tool_calls:
                name = call.get("name", "")
                args = call.get("arguments", {})
                result = self.tools.execute(name, args)
                turn.tool_results.append({"name": name, "content": result.content, "success": result.success})
                self.conversation.add_tool_result(name, result.content)

            tool_prompt = self.reasoning.build_tool_result_prompt(
                tool_name=name if parsed.tool_calls else "unknown",
                tool_content=result.content if parsed.tool_calls else "",
                original_prompt=prompt,
            )
            raw_output = self._generate(tool_prompt)
            parsed = self.reasoning.parse_response(raw_output)
            turn.assistant_response += "\n" + parsed.answer
            turn.tool_calls.extend(parsed.tool_calls)

        turn.latency_ms = (time.perf_counter() - start) * 1000
        self.conversation.turns.append(turn)
        self.conversation.add_user(user_input)
        self.conversation.add_assistant(turn.assistant_response)

        log.info(
            "Agent turn: %d ms, %d tokens, %d tool calls, reasoning=%s",
            turn.latency_ms,
            turn.tokens_used,
            len(turn.tool_calls),
            bool(turn.thinking),
        )
        return turn.assistant_response

    def _generate(self, prompt: str) -> str:
        """Generate text using the model, optionally with speculative decoding."""
        try:
            if self.draft_engine and self.model:
                prompt_ids = self._encode(prompt)
                result = self.draft_engine.speculative_generate(
                    prompt_ids=prompt_ids,
                    max_tokens=self.config.max_tokens,
                    tokenizer=getattr(self.model, "tokenizer", None),
                )
                if result.text:
                    return result.text
        except Exception as exc:
            log.warning("Speculative decoding failed, falling back: %s", exc)

        if hasattr(self.model, "generate"):
            return self.model.generate(prompt, max_tokens=self.config.max_tokens)
        return f"[Model would generate response for: {prompt[:50]}...]"

    @staticmethod
    def _encode(text: str) -> list[int]:
        return list(text.encode("utf-8")[:256])

    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation."""
        turns = len(self.conversation.turns)
        total_tokens = sum(t.tokens_used for t in self.conversation.turns)
        total_latency = sum(t.latency_ms for t in self.conversation.turns)
        return f"{turns} turns, ~{total_tokens} tokens, ~{total_latency:.0f}ms total"
