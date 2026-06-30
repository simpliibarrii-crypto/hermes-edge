"""
Hermes Agent-Style Tool Calling Format

Implements the NousResearch Hermes function-calling protocol:
  - Tool definitions in <tools> XML block in system message
  - Model emits <tool_call>{"name": "...", "arguments": {...}}</tool_call>
  - Results return as <tool_response>{"name": "...", "content": ...}</tool_response>
  - Supports multi-turn recursive tool calling

Compatible with LiteRT-LM constrained decoding anchors.

Usage:
    tools = [
        ToolDef(name="calculator", description="Math calculator",
                parameters={"type": "object", "properties": {"expr": {"type": "string"}}})
    ]
    formatter = HermesToolFormatter(tools)
    prompt = formatter.build_tool_prompt("What's 2+2?")
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict | None = None


@dataclass
class ToolResult:
    name: str
    content: str
    success: bool = True


from hermes.chat_template import (
    TOOL_CALL_START, TOOL_CALL_END, TOOL_RESPONSE_START, TOOL_RESPONSE_END,
    IM_START, IM_END, build_prompt as canonical_build_prompt,
)


class HermesToolFormatter:
    """Builds prompts and parses responses in Hermes function-calling format."""

    def __init__(self, tools: list[ToolDef] | None = None):
        self.tools = tools or []

    def set_tools(self, tools: list[ToolDef]) -> None:
        self.tools = tools

    def build_tools_block(self) -> str:
        """Build the <tools> XML block for the system message."""
        if not self.tools:
            return ""
        lines = ["<tools>"]
        for tool in self.tools:
            entry = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                },
            }
            if tool.parameters:
                entry["function"]["parameters"] = tool.parameters
            lines.append(json.dumps(entry))
        lines.append("</tools>")
        return "\n".join(lines)

    def build_system_message(self, base_system: str = "") -> str:
        """Build the full system message with tool definitions."""
        parts = [base_system] if base_system else []
        tools_block = self.build_tools_block()
        if tools_block:
            parts.append(tools_block)
        return "\n\n".join(parts) if parts else "You are a helpful AI assistant."

    def build_tool_prompt(
        self,
        user_input: str,
        system_override: str | None = None,
        context: str | None = None,
    ) -> str:
        """Build a full ChatML prompt with tool definitions in the system message."""
        system = system_override or self.build_system_message(
            "You are Hermes Edge, an on-device AI agent. Use tools when needed."
        )
        messages = [{"role": "system", "content": system}]
        if context:
            messages.append({"role": "user", "content": context})
        messages.append({"role": "user", "content": user_input})
        return self._format_chatml(messages)

    def parse_tool_calls(self, text: str) -> list[dict]:
        """Parse <tool_call>...</tool_call> blocks from model output."""
        import re

        pattern = re.compile(
            re.escape(TOOL_CALL_START) + r"(.*?)" + re.escape(TOOL_CALL_END), re.DOTALL
        )
        calls = []
        for match in pattern.finditer(text):
            try:
                parsed = json.loads(match.group(1).strip())
                calls.append(parsed)
            except json.JSONDecodeError:
                log.warning("Failed to parse tool call: %s", match.group(1))
        return calls

    def format_tool_result(self, name: str, content: str) -> str:
        """Format a tool result for feeding back into the prompt."""
        payload = json.dumps({"name": name, "content": content})
        return f"{TOOL_RESPONSE_START}{payload}{TOOL_RESPONSE_END}"

    @staticmethod
    def _format_chatml(messages: list[dict]) -> str:
        msgs = [Message(role=m["role"], content=m["content"]) for m in messages]
        return canonical_build_prompt(msgs, add_generation_prompt=True)


class ToolRegistry:
    """Registry of executable tools that the agent can call."""

    def __init__(self):
        self._tools: dict[str, tuple[ToolDef, Callable]] = {}

    def register(
        self,
        name: str,
        description: str,
        func: Callable,
        parameters: dict | None = None,
    ) -> ToolDef:
        tool = ToolDef(name=name, description=description, parameters=parameters)
        self._tools[name] = (tool, func)
        return tool

    def get_defs(self) -> list[ToolDef]:
        return [t for t, _ in self._tools.values()]

    def execute(self, name: str, arguments: dict | None = None) -> ToolResult:
        if name not in self._tools:
            return ToolResult(name=name, content=f"Unknown tool: {name}", success=False)
        _, func = self._tools[name]
        try:
            if arguments:
                result = func(**arguments)
            else:
                result = func()
            return ToolResult(name=name, content=str(result), success=True)
        except Exception as exc:
            log.error("Tool %s failed: %s", name, exc)
            return ToolResult(name=name, content=str(exc), success=False)
