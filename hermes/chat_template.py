"""Hermes chat + tool-calling prompt format.

The format follows the ChatML-style convention used by the original Hermes
models (`<|im_start|>role ... <|im_end|>`) and adds explicit tool-call markers
so the on-device model can emit structured function calls that the Google AI
Edge Gallery Agent Skills runtime can parse and dispatch.

A tool call is emitted as::

    <tool_call>{"name": "calculator", "arguments": {"expression": "2+2"}}</tool_call>

Constrained decoding in LiteRT-LM can be anchored on the ``<tool_call>`` /
``</tool_call>`` sentinels to guarantee well-formed JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
TOOL_RESPONSE_START = "<tool_response>"
TOOL_RESPONSE_END = "</tool_response>"

DEFAULT_SYSTEM_PROMPT = (
    "You are Hermes, a helpful on-device AI agent. You can call tools when "
    "they help answer the user. To call a tool, respond ONLY with a "
    "<tool_call> block containing JSON: "
    '{"name": <tool_name>, "arguments": <json_args>}. '
    "After receiving a <tool_response>, use it to answer the user."
)


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str


def render_tools(tools: List[Dict[str, Any]]) -> str:
    """Render available tool schemas into the system context."""
    if not tools:
        return ""
    lines = ["You have access to the following tools:"]
    for tool in tools:
        lines.append(json.dumps(tool, ensure_ascii=False))
    return "\n".join(lines)


def build_prompt(
    messages: List[Message],
    tools: Optional[List[Dict[str, Any]]] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    add_generation_prompt: bool = True,
) -> str:
    """Render a list of messages into the Hermes ChatML training/inference string."""
    parts: List[str] = []

    system_content = system_prompt
    tool_block = render_tools(tools or [])
    if tool_block:
        system_content = f"{system_prompt}\n\n{tool_block}"
    parts.append(f"{IM_START}system\n{system_content}{IM_END}")

    for msg in messages:
        if msg.role == "tool":
            body = f"{TOOL_RESPONSE_START}\n{msg.content}\n{TOOL_RESPONSE_END}"
            parts.append(f"{IM_START}tool\n{body}{IM_END}")
        else:
            parts.append(f"{IM_START}{msg.role}\n{msg.content}{IM_END}")

    if add_generation_prompt:
        parts.append(f"{IM_START}assistant\n")
    return "\n".join(parts)


def format_tool_call(name: str, arguments: Dict[str, Any]) -> str:
    """Serialize a tool call into the sentinel-wrapped JSON the model emits."""
    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"{TOOL_CALL_START}{payload}{TOOL_CALL_END}"


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Extract a tool call from model output, or None if absent/malformed."""
    start = text.find(TOOL_CALL_START)
    if start == -1:
        return None
    start += len(TOOL_CALL_START)
    end = text.find(TOOL_CALL_END, start)
    snippet = text[start:end] if end != -1 else text[start:]
    try:
        call = json.loads(snippet.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(call, dict) or "name" not in call:
        return None
    call.setdefault("arguments", {})
    return call
