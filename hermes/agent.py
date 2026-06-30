import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from hermes.chat_template import Message
from hermes.litert_model import LiteRTModel
from hermes.router import classify, get_intent, INTENT_CHAT, INTENT_REASONING, INTENT_TOOLS
from scripts.deepseek_reasoning_template import ReasoningPipeline
from scripts.hermes_tool_format import ToolRegistry, HermesToolFormatter
from hermes.mcp_client import MCPManager
from hermes.code_executor import CodeExecutor
from hermes.memory import AgentMemory
from hermes.rag import RAGEngine

log = logging.getLogger(__name__)

REASONING_EFFORT_LOW = "low"
REASONING_EFFORT_MEDIUM = "medium"
REASONING_EFFORT_HIGH = "high"
REASONING_EFFORTS = [REASONING_EFFORT_LOW, REASONING_EFFORT_MEDIUM, REASONING_EFFORT_HIGH]


class ResponseCache:
    """LRU response cache with TTL for instant re-query."""

    def __init__(self, capacity: int = 256, ttl_seconds: float = 120.0):
        self._cache: OrderedDict[tuple, tuple[str, float]] = OrderedDict()
        self._capacity = capacity
        self._ttl = ttl_seconds

    def _make_key(self, user_input: str, intent: str) -> tuple:
        return (user_input.lower().strip(), intent)

    def get(self, user_input: str, intent: str) -> str | None:
        key = self._make_key(user_input, intent)
        if key not in self._cache:
            return None
        response, timestamp = self._cache[key]
        if time.monotonic() - timestamp > self._ttl:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return response

    def put(self, user_input: str, intent: str, response: str) -> None:
        key = self._make_key(user_input, intent)
        self._cache[key] = (response, time.monotonic())
        while len(self._cache) > self._capacity:
            self._cache.popitem(last=False)

    def invalidate(self, user_input: str, intent: str | None = None) -> None:
        if intent:
            key = self._make_key(user_input, intent)
            self._cache.pop(key, None)
        else:
            self._cache = OrderedDict()

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        return 0.0  # tracked externally


# ── Model Manager ─────────────────────────────────────────────────


class ModelManager:
    """Multi-model lifecycle for intent-based routing.

    Keeps a hot model (270M, ~180 MB) always loaded for instant chat.
    Specialist models (reasoning, tools) load on demand.
    LiteRT-LM fast initialization makes on-demand loading practical.
    """

    def __init__(self, backend: str = "auto"):
        self._models: dict[str, LiteRTModel] = {}
        self._hot_key: str = ""
        self.current_key: str = ""
        self.backend = backend

    def load_hot(self, path: str) -> bool:
        p = Path(path).resolve()
        m = LiteRTModel(str(p), backend=self.backend)
        if m.load():
            self._models["_hot"] = m
            self._hot_key = "_hot"
            self.current_key = "_hot"
            log.info("Hot model loaded: %s (%.1f MB)", p.name, p.stat().st_size / 1_048_576)
            return True
        log.warning("Failed to load hot model: %s", path)
        return False

    def register(self, key: str, path: str) -> None:
        p = Path(path).resolve()
        if not p.exists():
            log.warning("Model path not found: %s (registering anyway)", path)
        self._models[key] = LiteRTModel(str(p), backend=self.backend)

    def select(self, key: str) -> LiteRTModel:
        if key not in self._models:
            return self._fallback()
        m = self._models[key]
        if not m._loaded:
            p = Path(m.model_path)
            if not p.exists():
                log.warning("Model file missing: %s", p)
                return self._fallback()
            m.load()
        self.current_key = key
        return m

    def get_current(self) -> LiteRTModel | None:
        if self.current_key and self.current_key in self._models:
            return self._models[self.current_key]
        return self._fallback() if self._hot_key else None

    def _fallback(self) -> LiteRTModel:
        if self._hot_key and self._hot_key in self._models:
            self.current_key = self._hot_key
            return self._models[self._hot_key]
        raise RuntimeError("No model available")

    def resolve(self, intent: str) -> LiteRTModel:
        key = intent if intent in (INTENT_REASONING, INTENT_TOOLS) else "_hot"
        return self.select(key)


# ── Agent Config ──────────────────────────────────────────────────


@dataclass
class AgentConfig:
    max_tool_rounds: int = 3
    max_tokens: int = 384
    temperature: float = 0.6
    top_k: int = 40
    use_reasoning: bool = True
    max_thinking_tokens: int = 128
    enable_mtp: bool = True
    enable_routing: bool = True

    mcp_servers: list[str] = field(default_factory=list)
    memory: bool = False
    rag_db: str = ""

    reasoning_effort: str = REASONING_EFFORT_MEDIUM
    enable_response_cache: bool = True
    enable_streaming: bool = False

    intent_temperature: dict = field(default_factory=lambda: {
        INTENT_CHAT: 0.7,
        INTENT_REASONING: 0.6,
        INTENT_TOOLS: 0.5,
    })
    intent_max_tokens: dict = field(default_factory=lambda: {
        INTENT_CHAT: 128,
        INTENT_REASONING: 384,
        INTENT_TOOLS: 256,
    })
    intent_mtp: dict = field(default_factory=lambda: {
        INTENT_CHAT: True,
        INTENT_REASONING: False,
        INTENT_TOOLS: True,
    })

    effort_map: dict = field(default_factory=lambda: {
        REASONING_EFFORT_LOW: {
            "max_tokens": 64,
            "temperature": 0.8,
            "max_thinking_tokens": 0,
            "use_reasoning": False,
            "enable_mtp": True,
            "model_intent": INTENT_CHAT,
        },
        REASONING_EFFORT_MEDIUM: {
            "max_tokens": 256,
            "temperature": 0.6,
            "max_thinking_tokens": 128,
            "use_reasoning": True,
            "enable_mtp": True,
            "model_intent": None,
        },
        REASONING_EFFORT_HIGH: {
            "max_tokens": 512,
            "temperature": 0.4,
            "max_thinking_tokens": 256,
            "use_reasoning": True,
            "enable_mtp": False,
            "model_intent": INTENT_REASONING,
        },
    })


@dataclass
class AgentTurn:
    user_input: str = ""
    assistant_response: str = ""
    thinking: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    intent: str = INTENT_CHAT
    latency_ms: float = 0.0
    tokens_used: int = 0
    model_used: str = ""


@dataclass
class Conversation:
    messages: list[Message] = field(default_factory=list)
    turns: list[AgentTurn] = field(default_factory=list)

    def add_user(self, text: str) -> None:
        self.messages.append(Message(role="user", content=text))

    def add_assistant(self, text: str) -> None:
        self.messages.append(Message(role="assistant", content=text))

    def add_tool_result(self, name: str, content: str) -> None:
        self.messages.append(
            Message(role="tool", content=f"<tool_response>{name}: {content}</tool_response>")
        )


# ── Agent ─────────────────────────────────────────────────────────


class HermesAgent:

    def __init__(
        self,
        model=None,
        tool_registry: ToolRegistry | None = None,
        config: AgentConfig | None = None,
        model_manager: ModelManager | None = None,
    ):
        self.model = model
        self.model_manager = model_manager
        self.config = config or AgentConfig()
        self.tools = tool_registry or ToolRegistry()
        self.conversation = Conversation()
        self.reasoning = ReasoningPipeline(
            use_reasoning=self.config.use_reasoning,
            max_thinking_tokens=self.config.max_thinking_tokens,
        )
        self.tool_formatter = HermesToolFormatter()
        self.mcp_manager: MCPManager | None = None
        self.code_executor: CodeExecutor | None = None
        self.agent_memory: AgentMemory | None = None
        self.rag: RAGEngine | None = None
        self.response_cache = ResponseCache()
        self._cache_hits = 0
        self._cache_misses = 0

        self._apply_effort_config()

        if self.config.mcp_servers:
            self._init_mcp(self.config.mcp_servers)
        if self.config.memory:
            self.agent_memory = AgentMemory()
        if self.config.rag_db:
            self.rag = RAGEngine(db_path=self.config.rag_db)

        if self.mcp_manager:
            self._register_mcp_tools()
        self._register_code_tool()
        self._register_knowledge_tool()

    def _apply_effort_config(self) -> None:
        """Apply reasoning_effort to agent config (like GPT-5.5 adaptive reasoning)."""
        effort = self.config.effort_map.get(
            self.config.reasoning_effort,
            self.config.effort_map[REASONING_EFFORT_MEDIUM],
        )
        self.config.max_tokens = effort["max_tokens"]
        self.config.temperature = effort["temperature"]
        self.config.max_thinking_tokens = effort["max_thinking_tokens"]
        self.config.use_reasoning = effort["use_reasoning"]
        self.config.enable_mtp = effort["enable_mtp"]
        self.reasoning.max_thinking_tokens = effort["max_thinking_tokens"]
        self.reasoning.use_reasoning = effort["use_reasoning"]

    def set_model(self, model) -> None:
        self.model = model

    def register_tool(
        self, name: str, description: str, func, parameters: dict | None = None
    ) -> None:
        self.tools.register(name, description, func, parameters)

    def register_default_tools(self) -> None:
        """Register built-in tools: web search, calculator."""
        try:
            from hermes.web_search import web_search as _ws
            self.register_tool(
                "web_search",
                "Search the web for current information. Use for news, facts, real-time data.",
                _ws,
                {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (be specific)",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Number of results (1-5)",
                            "default": 3,
                        },
                    },
                    "required": ["query"],
                },
            )
        except ImportError:
            log.info("web_search tool not available (hermes.web_search not found)")

        try:
            import math as _math

            def _calc(expr: str) -> str:
                safe = {"abs": abs, "round": round, "int": int, "float": float,
                        "min": min, "max": max, "sum": sum, "pow": pow}
                safe.update({k: getattr(_math, k) for k in dir(_math)
                            if not k.startswith("_") and callable(getattr(_math, k))})
                return str(eval(expr, {"__builtins__": {}}, safe))

            self.register_tool(
                "calculator",
                "Evaluate mathematical expressions. Supports +, -, *, /, sqrt, sin, cos, etc.",
                _calc,
                {
                    "type": "object",
                    "properties": {
                        "expr": {
                            "type": "string",
                            "description": "Math expression (e.g., 'sqrt(144) + 42')",
                        }
                    },
                    "required": ["expr"],
                },
            )
        except ImportError:
            pass

    def _init_mcp(self, servers: list[str]) -> None:
        """Connect to MCP servers."""
        self.mcp_manager = MCPManager()
        for server_cmd in servers:
            name = server_cmd.split()[-1] if " " in server_cmd else server_cmd
            self.mcp_manager.connect_stdio(name, server_cmd)

    def _register_mcp_tools(self) -> None:
        """Register MCP tools in ToolRegistry."""
        if not self.mcp_manager:
            return
        for tool in self.mcp_manager.get_all_tools():
            server_name = tool.get("_mcp_server", "mcp")
            tool_name = tool.get("name", "")
            desc = tool.get("description", f"MCP tool from {server_name}")
            input_schema = tool.get("inputSchema", {})
            params = tool.get("parameters", input_schema)

            def _make_mcp_call(srv: str, tname: str):
                def _call(**kwargs):
                    result = self.mcp_manager.call_tool(srv, tname, kwargs)
                    content = result.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(c.get("text", "") for c in content)
                    return str(content)
                return _call

            self.tools.register(tool_name, desc, _make_mcp_call(server_name, tool_name), params)

    def _register_code_tool(self) -> None:
        """Register execute_python tool with CodeExecutor."""
        self.code_executor = CodeExecutor()

        def _execute_python(code: str) -> str:
            result = self.code_executor.execute(code)
            if result.success:
                output = result.output
                if result.variables:
                    output += "\nVariables: " + str(result.variables)
                return output
            return f"Error: {result.error}"

        self.register_tool(
            "execute_python",
            "Execute Python code in a restricted sandbox. Use for calculations, data processing, automation.",
            _execute_python,
            {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute",
                    }
                },
                "required": ["code"],
            },
        )

    def _register_knowledge_tool(self) -> None:
        """Register knowledge_search tool using RAGEngine."""
        if not self.rag:
            return

        def _knowledge_search(query: str, top_k: int = 3) -> str:
            return self.rag.get_relevant_context(query, top_k)

        self.register_tool(
            "knowledge_search",
            "Search local knowledge base for relevant information. Use for facts, documentation, stored data.",
            _knowledge_search,
            {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results (1-10)",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        )

    def _inject_memory_context(self, messages: list[Message]) -> list[Message]:
        """Add memory summary to system prompt."""
        if not self.agent_memory:
            return messages
        summary = self.agent_memory.get_summary()
        if not summary:
            return messages
        return [Message(role="system", content=f"[Memory Context]\n{summary}")] + messages

    def run(self, user_input: str, context: str | None = None) -> str:
        if not self.model and not self.model_manager:
            return "Error: No model loaded."

        intent = classify(user_input).intent if self.config.enable_routing else INTENT_CHAT

        if self.config.enable_response_cache:
            cached = self.response_cache.get(user_input, intent)
            if cached is not None:
                self._cache_hits += 1
                log.debug("Response cache HIT: intent=%s", intent)
                return cached
            self._cache_misses += 1

        turn = AgentTurn(user_input=user_input)
        start = time.perf_counter()

        if self.agent_memory:
            self.conversation.messages = self._inject_memory_context(self.conversation.messages)
            self.agent_memory.add_entry("user", user_input)

        turn.intent = intent

        if self.model_manager:
            effort_override = self.config.effort_map.get(
                self.config.reasoning_effort, {}
            ).get("model_intent")
            resolve_intent = effort_override or intent
            active_model = self.model_manager.resolve(resolve_intent)
            turn.model_used = self.model_manager.current_key
        else:
            active_model = self.model
            turn.model_used = "default"

        temperature = self.config.intent_temperature.get(intent, self.config.temperature)
        max_tokens = self.config.intent_max_tokens.get(intent, self.config.max_tokens)
        enable_mtp = self.config.intent_mtp.get(intent, self.config.enable_mtp)

        if hasattr(active_model, "enable_mtp"):
            active_model.enable_mtp = enable_mtp

        self.reasoning.use_reasoning = intent == INTENT_REASONING
        history = self._build_history_prompt()
        prompt = self.reasoning.build_reasoning_prompt(user_input, context or history)

        tool_defs = self.tools.get_defs()
        self.tool_formatter.set_tools(tool_defs)

        raw_output = self._generate(active_model, prompt, max_tokens, temperature)
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
                turn.tool_results.append(
                    {"name": name, "content": result.content, "success": result.success}
                )
                self.conversation.add_tool_result(name, result.content)

            tool_prompt = self.reasoning.build_tool_result_prompt(
                tool_name=name if parsed.tool_calls else "unknown",
                tool_content=result.content if parsed.tool_calls else "",
                original_prompt=prompt,
            )
            raw_output = self._generate(active_model, tool_prompt, max_tokens, temperature)
            parsed = self.reasoning.parse_response(raw_output)
            if parsed.answer:
                turn.assistant_response += "\n" + parsed.answer
            turn.tool_calls.extend(parsed.tool_calls)

        turn.latency_ms = (time.perf_counter() - start) * 1000
        self.conversation.turns.append(turn)
        self.conversation.add_user(user_input)
        self.conversation.add_assistant(turn.assistant_response)
        if self.agent_memory:
            self.agent_memory.add_entry("assistant", turn.assistant_response)

        if self.config.enable_response_cache:
            self.response_cache.put(user_input, intent, turn.assistant_response)

        log.info(
            "Agent: %s ms, %d tok, intent=%s, model=%s",
            f"{turn.latency_ms:.0f}", turn.tokens_used, turn.intent, turn.model_used,
        )
        return turn.assistant_response

    def run_stream(self, user_input: str, context: str | None = None) -> Iterator[str]:
        """Streaming run: yields tokens progressively and returns full response at end.

        First yields a skeleton with the intent tag, then streams model tokens.
        """
        if not self.model and not self.model_manager:
            yield "Error: No model loaded."
            return

        intent = classify(user_input).intent if self.config.enable_routing else INTENT_CHAT

        if self.config.enable_response_cache:
            cached = self.response_cache.get(user_input, intent)
            if cached is not None:
                self._cache_hits += 1
                log.debug("Response cache HIT (stream): intent=%s", intent)
                yield cached
                return
            self._cache_misses += 1

        turn = AgentTurn(user_input=user_input)
        start = time.perf_counter()

        if self.agent_memory:
            self.conversation.messages = self._inject_memory_context(self.conversation.messages)
            self.agent_memory.add_entry("user", user_input)

        turn.intent = intent
        intent_tag = f"[{intent}] "

        if self.model_manager:
            effort_override = self.config.effort_map.get(
                self.config.reasoning_effort, {}
            ).get("model_intent")
            resolve_intent = effort_override or intent
            active_model = self.model_manager.resolve(resolve_intent)
            turn.model_used = self.model_manager.current_key
        else:
            active_model = self.model
            turn.model_used = "default"

        temperature = self.config.intent_temperature.get(intent, self.config.temperature)
        max_tokens = self.config.intent_max_tokens.get(intent, self.config.max_tokens)
        enable_mtp = self.config.intent_mtp.get(intent, self.config.enable_mtp)

        if hasattr(active_model, "enable_mtp"):
            active_model.enable_mtp = enable_mtp

        self.reasoning.use_reasoning = intent == INTENT_REASONING

        yield intent_tag

        history = self._build_history_prompt()
        prompt = self.reasoning.build_reasoning_prompt(user_input, context or history)

        tool_defs = self.tools.get_defs()
        self.tool_formatter.set_tools(tool_defs)

        full_output = ""
        for token in self._generate_stream(active_model, prompt, max_tokens, temperature):
            full_output += token
            yield token

        turn.tokens_used = max(1, len(full_output) // 4)

        parsed = self.reasoning.parse_response(full_output)
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
                turn.tool_results.append(
                    {"name": name, "content": result.content, "success": result.success}
                )
                self.conversation.add_tool_result(name, result.content)

            tool_prompt = self.reasoning.build_tool_result_prompt(
                tool_name=name if parsed.tool_calls else "unknown",
                tool_content=result.content if parsed.tool_calls else "",
                original_prompt=prompt,
            )
            for token in self._generate_stream(active_model, tool_prompt, max_tokens, temperature):
                full_output += token
                yield token
            parsed = self.reasoning.parse_response(full_output)
            if parsed.answer:
                turn.assistant_response += "\n" + parsed.answer
            turn.tool_calls.extend(parsed.tool_calls)

        turn.latency_ms = (time.perf_counter() - start) * 1000
        self.conversation.turns.append(turn)
        self.conversation.add_user(user_input)
        self.conversation.add_assistant(turn.assistant_response)
        if self.agent_memory:
            self.agent_memory.add_entry("assistant", turn.assistant_response)

        if self.config.enable_response_cache:
            self.response_cache.put(user_input, intent, turn.assistant_response)

        log.info(
            "Agent: %s ms, %d tok, intent=%s, model=%s",
            f"{turn.latency_ms:.0f}", turn.tokens_used, turn.intent, turn.model_used,
        )

    def _generate_stream(
        self, model, prompt: str, max_tokens: int, temperature: float
    ) -> Iterator[str]:
        """Stream tokens from the model generator."""
        if hasattr(model, "generate_stream"):
            yield from model.generate_stream(prompt, max_tokens=max_tokens, temperature=temperature)
        elif hasattr(model, "generate"):
            yield model.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        else:
            yield ""

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

    def _generate(
        self, model, prompt: str, max_tokens: int, temperature: float
    ) -> str:
        if hasattr(model, "generate"):
            return model.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=self.config.top_k,
            )
        return ""

    def get_conversation_summary(self) -> str:
        turns = len(self.conversation.turns)
        total_tokens = sum(t.tokens_used for t in self.conversation.turns)
        total_latency = sum(t.latency_ms for t in self.conversation.turns)
        return f"{turns} turns, ~{total_tokens} tokens, ~{total_latency:.0f}ms total"
