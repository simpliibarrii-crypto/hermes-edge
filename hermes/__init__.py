"""
Hermes Edge — Package Init
"""

__version__ = "0.3.0"
__author__ = "Barry Clerjuste"
__email__ = "bclerjuste@gmail.com"

from hermes.config import HermesConfig, get_config, PRESETS
from hermes.chat_template import build_prompt, Message
from hermes.litert_model import LiteRTModel
from hermes.router import classify, get_intent, INTENT_CHAT, INTENT_REASONING, INTENT_TOOLS
from hermes.exceptions import HermesEdgeError, ModelLoadError, InferenceError, ToolExecutionError, RoutingError, WebSearchError, ConfigError
from hermes.web_search import web_search, SearchResult
from hermes.rag import RAGEngine
from hermes.memory import AgentMemory
from hermes.mcp_server import MCPServer
from hermes.mcp_client import MCPManager
from hermes.code_executor import CodeExecutor, ExecutionResult

try:
    from hermes.agent import (
        HermesAgent, AgentConfig, ModelManager, ModelPreloader,
        ResponseCache, AgentTurn, Conversation,
        REASONING_EFFORT_LOW, REASONING_EFFORT_MEDIUM, REASONING_EFFORT_HIGH,
        REASONING_EFFORTS,
    )
except ImportError:
    pass
