import json
import logging
import re
from dataclasses import dataclass, field

from hermes.chat_template import IM_START, IM_END, TOOL_CALL_START, TOOL_CALL_END, TOOL_RESPONSE_START, TOOL_RESPONSE_END

log = logging.getLogger(__name__)

THINK_START = "<think>"
THINK_END = "</think>"


@dataclass
class ReasoningResult:
    thinking: str = ""
    answer: str = ""
    tool_calls: list[dict] = field(default_factory=list)


SYSTEM_PROMPT_REASONING = (
    "You are Hermes, a fast on-device AI. Think briefly in <think>, "
    "then answer naturally. No long chains unless the question requires math or logic. "
    "Keep thinking under 3 sentences. Use <tool_call> for tools."
)

SYSTEM_PROMPT_DIRECT = (
    "You are Hermes, a fast on-device AI. Answer naturally and concisely. "
    "Use <tool_call> for tools when needed."
)


class ReasoningPipeline:

    def __init__(self, use_reasoning: bool = True, max_thinking_tokens: int = 128):
        self.use_reasoning = use_reasoning
        self.max_thinking_tokens = max_thinking_tokens

    def build_reasoning_prompt(self, user_input: str, context: str | None = None) -> str:
        system = SYSTEM_PROMPT_REASONING if self.use_reasoning else SYSTEM_PROMPT_DIRECT
        messages = [{"role": "system", "content": system}]
        if context:
            messages.append({"role": "user", "content": context})
        messages.append({"role": "user", "content": user_input})
        return self._format_chatml(messages)

    def build_tool_result_prompt(
        self, tool_name: str, tool_content: str, original_prompt: str | None = None
    ) -> str:
        parts = []
        if original_prompt:
            parts.append(original_prompt.rstrip())
        parts.append(
            f"{TOOL_RESPONSE_START}{{{{\"name\": \"{tool_name}\", \"content\": {json.dumps(tool_content)}}}}}{TOOL_RESPONSE_END}"
        )
        return "\n".join(parts)

    def parse_response(self, text: str) -> ReasoningResult:
        result = ReasoningResult()

        think_pattern = re.compile(
            re.escape(THINK_START) + r"(.*?)" + re.escape(THINK_END), re.DOTALL
        )
        think_match = think_pattern.search(text)
        if think_match:
            result.thinking = think_match.group(1).strip()
            text = think_pattern.sub("", text).strip()

        tool_pattern = re.compile(
            re.escape(TOOL_CALL_START) + r"(.*?)" + re.escape(TOOL_CALL_END), re.DOTALL
        )
        for match in tool_pattern.finditer(text):
            try:
                result.tool_calls.append(json.loads(match.group(1).strip()))
            except json.JSONDecodeError:
                log.warning("Failed to parse tool call: %s", match.group(1))

        answer = tool_pattern.sub("", text).strip()
        result.answer = answer

        return result

    @staticmethod
    def _format_chatml(messages: list[dict]) -> str:
        parts = []
        for msg in messages:
            parts.append(f"{IM_START}{msg['role']}\n{msg['content']}{IM_END}\n")
        parts.append(f"{IM_START}assistant")
        last = parts[-2] if len(parts) >= 2 else ""
        if "<think>" not in last:
            parts.append("\n" + THINK_START + "\n")
        return "".join(parts)

    @staticmethod
    def extract_final_answer(text: str) -> str:
        result = ReasoningPipeline().parse_response(text)
        return result.answer or result.thinking or text
