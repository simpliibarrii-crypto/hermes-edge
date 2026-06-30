"""
DeepSeek-Style Chain-of-Thought Reasoning Templates

Implements the reasoning prompt pattern used by DeepSeek-R1 and DeepSeek-V4:
  - <think>...</think> tags to delimit the internal reasoning trace
  - The model generates reasoning first, then the final answer
  - Compatible with Hermes Agent tool-calling format

Usage:
    from deepseek_reasoning_template import ReasoningPipeline

    pipe = ReasoningPipeline()
    prompt = pipe.build_reasoning_prompt("Solve 2x + 5 = 13")
    # Assistant generates: <think>Let me solve this step by step...</think>\n\nx = 4
    result = pipe.parse_response(generated_text)
    # -> {"thinking": "Let me solve this step by step...", "answer": "x = 4"}
"""

import json
import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

THINK_START = "<think>"
THINK_END = "</think>"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
TOOL_RESPONSE_START = "<tool_response>"
TOOL_RESPONSE_END = "</tool_response>"


@dataclass
class ReasoningResult:
    thinking: str = ""
    answer: str = ""
    tool_calls: list[dict] = field(default_factory=list)


class ReasoningPipeline:
    """Builds prompts and parses responses for DeepSeek-style chain-of-thought."""

    SYSTEM_PROMPT_REASONING = """You are Hermes Edge, an on-device AI agent powered by Raven AI ecosystem. Think step by step before answering.

You MUST follow this format:
1. First, reason internally inside <think> tags
2. Then provide your final answer after </think>

If you need to use tools, emit:
<tool_call>{"name": "tool_name", "arguments": {"key": "value"}}</tool_call>
The tool result will be provided as:
<tool_response>{"name": "tool_name", "content": "result"}</tool_response>
Continue reasoning after receiving results.

DeepSeek reasoning principles:
- Break complex problems into steps
- Verify each step before proceeding
- Consider multiple approaches
- Be explicit about assumptions
- Show your work in <think> tags"""

    SYSTEM_PROMPT_DIRECT = (
        "You are Hermes Edge, an on-device AI agent powered by Raven AI ecosystem. "
        "Respond helpfully and concisely."
    )

    def __init__(self, use_reasoning: bool = True):
        self.use_reasoning = use_reasoning

    def build_reasoning_prompt(self, user_input: str, context: str | None = None) -> str:
        """Build a ChatML-formatted prompt with reasoning priming."""
        system = self.SYSTEM_PROMPT_REASONING if self.use_reasoning else self.SYSTEM_PROMPT_DIRECT
        messages = [{"role": "system", "content": system}]
        if context:
            messages.append({"role": "user", "content": context})
        messages.append({"role": "user", "content": user_input})
        return self._format_chatml(messages)

    def build_tool_result_prompt(
        self, tool_name: str, tool_content: str, original_prompt: str | None = None
    ) -> str:
        """Build prompt with tool result fed back for continued reasoning."""
        parts = []
        if original_prompt:
            parts.append(original_prompt.rstrip())
        parts.append(
            f"{TOOL_RESPONSE_START}{{{{\"name\": \"{tool_name}\", \"content\": {json.dumps(tool_content)}}}}}{TOOL_RESPONSE_END}"
        )
        return "\n".join(parts)

    def parse_response(self, text: str) -> ReasoningResult:
        """Parse a model response into thinking trace + answer + tool calls."""
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
        """Format messages as ChatML (compatible with Qwen3/Gemma/Hermes models)."""
        im_start = "<|im_start|>"
        im_end = "<|im_end|>"
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"{im_start}{role}\n{content}{im_end}\n")
        parts.append(f"{im_start}assistant")
        if "<think>" not in "\n".join(m.split("\n")[-1] for m in parts):
            parts.append("\n" + THINK_START + "\n")
        return "".join(parts)

    @staticmethod
    def extract_final_answer(text: str) -> str:
        """Get just the final answer, stripping thinking trace."""
        result = ReasoningPipeline().parse_response(text)
        return result.answer or result.thinking or text
