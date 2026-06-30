"""
MCP Client — connects to MCP servers over stdio or HTTP.
"""
import json
import logging
import subprocess
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class MCPConnection:
    """Connection to an MCP server."""
    name: str
    transport: str  # "stdio" or "http"
    command: str = ""
    url: str = ""
    tools: list[dict] = field(default_factory=list)


class MCPManager:
    """Manages connections to MCP servers and aggregates their tools."""

    def __init__(self) -> None:
        self.connections: list[MCPConnection] = []
        self._processes: dict[str, subprocess.Popen] = {}

    def connect_stdio(self, name: str, command: str) -> bool:
        """Connect to a stdio-based MCP server."""
        try:
            proc = subprocess.Popen(
                command.split(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            req = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
            proc.stdin.write(req + "\n")
            proc.stdin.flush()
            resp = json.loads(proc.stdout.readline())

            conn = MCPConnection(name=name, transport="stdio", command=command)
            conn.tools = resp.get("result", {}).get("tools", [])
            self.connections.append(conn)
            self._processes[name] = proc
            log.info("Connected to MCP server: %s (%d tools)", name, len(conn.tools))
            return True
        except Exception as e:
            log.warning("Failed to connect MCP server %s: %s", name, e)
            return False

    def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> Any:
        """Call a tool on a specific MCP server."""
        proc = self._processes.get(server_name)
        if not proc:
            raise ValueError(f"MCP server not connected: {server_name}")

        req = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        })
        proc.stdin.write(req + "\n")
        proc.stdin.flush()
        resp = json.loads(proc.stdout.readline())
        return resp.get("result", {})

    def get_all_tools(self) -> list[dict]:
        """Get aggregated tool list from all connected servers."""
        all_tools: list[dict] = []
        for conn in self.connections:
            for t in conn.tools:
                entry = dict(t)
                entry["_mcp_server"] = conn.name
                all_tools.append(entry)
        return all_tools

    def disconnect_all(self) -> None:
        for name, proc in self._processes.items():
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._processes.clear()
        self.connections.clear()
