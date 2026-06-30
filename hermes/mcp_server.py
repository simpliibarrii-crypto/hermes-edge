"""
MCP Server — exposes Hermes Edge tools as MCP endpoints.
"""
import json
import logging
import math
from dataclasses import dataclass
from typing import Any, Callable

log = logging.getLogger(__name__)


@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: dict


TOOL_DEFINITIONS: list[MCPTool] = [
    MCPTool(
        name="web_search",
        description="Search the web for current information. Use for news, facts, real-time data.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query (be specific)"},
                "max_results": {"type": "integer", "description": "Number of results (1-5)", "default": 3},
            },
            "required": ["query"],
        },
    ),
    MCPTool(
        name="calculator",
        description="Evaluate mathematical expressions. Supports +, -, *, /, sqrt, sin, cos, etc.",
        input_schema={
            "type": "object",
            "properties": {
                "expr": {"type": "string", "description": "Math expression e.g. 'sqrt(144) + 42'"},
            },
            "required": ["expr"],
        },
    ),
    MCPTool(
        name="memory_write",
        description="Store a value in the agent's working memory.",
        input_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Memory key"},
                "value": {"type": "string", "description": "Value to store"},
            },
            "required": ["key", "value"],
        },
    ),
    MCPTool(
        name="memory_read",
        description="Read a value from the agent's working memory.",
        input_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Memory key"},
            },
            "required": ["key"],
        },
    ),
]


def _web_search_wrapper(query: str, max_results: int = 3) -> str:
    try:
        from hermes.web_search import web_search as _ws
        return _ws(query=query, max_results=max_results)
    except ImportError:
        return json.dumps({"error": "web_search module not available"})


def _calculator_wrapper(expr: str) -> str:
    safe: dict[str, Any] = {
        "abs": abs, "round": round, "int": int, "float": float,
        "min": min, "max": max, "sum": sum, "pow": pow,
    }
    safe.update(
        {k: getattr(math, k) for k in dir(math)
         if not k.startswith("_") and callable(getattr(math, k))}
    )
    try:
        return str(eval(expr, {"__builtins__": {}}, safe))
    except Exception as e:
        return json.dumps({"error": f"Calculator error: {e}"})


class MCPServer:
    """Minimal MCP server over stdio transport."""

    def __init__(self, tools: dict[str, Callable] | None = None):
        self._tools: dict[str, Callable] = {}
        self._memory: dict[str, str] = {}

        if tools:
            for name, fn in tools.items():
                self._tools[name] = fn

        self._tools.setdefault("web_search", _web_search_wrapper)
        self._tools.setdefault("calculator", _calculator_wrapper)
        self._tools.setdefault("memory_write", self._memory_write)
        self._tools.setdefault("memory_read", self._memory_read)

    def _memory_write(self, key: str, value: str) -> str:
        self._memory[key] = value
        return json.dumps({"ok": True, "key": key})

    def _memory_read(self, key: str) -> str:
        val = self._memory.get(key)
        if val is None:
            return json.dumps({"error": f"Key not found: {key}"})
        return json.dumps({"key": key, "value": val})

    def _build_tool_list(self) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema,
            }
            for t in TOOL_DEFINITIONS
        ]

    def handle_request(self, request: dict) -> dict:
        method = request.get("method", "")
        req_id = request.get("id", 0)

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": self._build_tool_list()},
            }
        elif method == "tools/call":
            return self._call_tool(request.get("params", {}), req_id)
        elif method == "resources/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"resources": []}}
        elif method == "prompts/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"prompts": []}}

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": "Method not found"},
        }

    def _call_tool(self, params: dict, req_id: int) -> dict:
        name = params.get("name", "")
        arguments = params.get("arguments", {})

        fn = self._tools.get(name)
        if fn is None:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": f"Unknown tool: {name}"},
            }

        try:
            result = fn(**arguments)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": str(result)}]},
            }
        except Exception as e:
            log.error("Tool call error: %s", e)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32603, "message": f"Internal error: {e}"},
            }

    def serve_stdio(self) -> None:
        """Read JSON-RPC requests from stdin, write responses to stdout."""
        import sys
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                resp = self.handle_request(req)
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
            except Exception as e:
                log.error("MCP error: %s", e)


def serve_stdio(tools: dict[str, Callable] | None = None) -> None:
    """Create an MCPServer and run it over stdio."""
    server = MCPServer(tools)
    server.serve_stdio()
