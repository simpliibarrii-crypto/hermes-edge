import logging
import time
from dataclasses import dataclass, field

from hermes.chat_template import build_prompt, Message, format_tool_call, parse_tool_call
from scripts.deepseek_reasoning_template import ReasoningPipeline, ReasoningResult
from scripts.hermes_tool_format import ToolRegistry, HermesToolFormatter, ToolDef

log = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    max_tool_rounds: int = 3
    max_tokens: int = 384
    temperature: float = 0.6
    top_k: int = 40
    use_reasoning: bool = True
    max_thinking_tokens: int = 128
    enable_mtp: bool = True
    system_prompt: str = ""


DEFAULT_SYSTEM = (
    "You are Hermes Edge, a fast on-device AI agent. "
    "Think briefly, answer naturally. Be concise and helpful."
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
        self.reasoning = ReasoningPipeline(
            use_reasoning=self.config.use_reasoning,
            max_thinking_tokens=self.config.max_thinking_tokens,
        )
        self.tool_formatter = HermesToolFormatter()

    def set_model(self, model) -> None:
        self.model = model

    def register_tool(self, name: str, description: str, func, parameters: dict | None = None) -> None:
        self.tools.register(name, description, func, parameters)

    def run(self, user_input: str, context: str | None = None) -> str:
        if not self.model:
            return "Error: No model loaded."

        turn = AgentTurn(user_input=user_input)
        start = time.perf_counter()

        history = self._build_history_prompt()
        prompt = self.reasoning.build_reasoning_prompt(user_input, context or history)
        tool_defs = self.tools.get_defs()
        self.tool_formatter.set_tools(tool_defs)

        raw_output = self._generate(prompt)
        turn.tokens_used = max(1, len(raw_output) // 4)

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
            if parsed.answer:
                turn.assistant_response += "\n" + parsed.answer
            turn.tool_calls.extend(parsed.tool_calls)

        turn.latency_ms = (time.perf_counter() - start) * 1000
        self.conversation.turns.append(turn)
        self.conversation.add_user(user_input)
        self.conversation.add_assistant(turn.assistant_response)

        if turn.thinking:
            log.debug("Thinking (%d chars): %s", len(turn.thinking), turn.thinking[:200])
        log.info(
            "Agent turn: %d ms, %d tokens, %d tool calls",
            turn.latency_ms,
            turn.tokens_used,
            len(turn.tool_calls),
        )
        return turn.assistant_response

    def _build_history_prompt(self) -> str:
        if len(self.conversation.turns) < 2:
            return ""
        recent = self.conversation.turns[-3:]
        parts = ["Previous conversation:"]
        for t in recent:
            parts.append(f"User: {t.user_input[:200]}")
            if t.assistant_response:
                parts.append(f"Assistant: {t.assistant_response[:200]}")
        return "\n".join(parts)

    def _generate(self, prompt: str) -> str:
        if self.config.enable_mtp and hasattr(self.model, "enable_mtp"):
            self.model.enable_mtp = True

        if hasattr(self.model, "generate"):
            return self.model.generate(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
            )
        return f"[Model would generate response for: {prompt[:50]}...]"

    def get_conversation_summary(self) -> str:
        turns = len(self.conversation.turns)
        total_tokens = sum(t.tokens_used for t in self.conversation.turns)
        total_latency = sum(t.latency_ms for t in self.conversation.turns)
        return f"{turns} turns, ~{total_tokens} tokens, ~{total_latency:.0f}ms total"
