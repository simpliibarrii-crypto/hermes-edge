"""
Hermes Edge — Package Init
"""

__version__ = "0.2.0"
__author__ = "Barry Clerjuste"
__email__ = "bclerjuste@gmail.com"

from hermes.config import HermesConfig, get_config, PRESETS
from hermes.chat_template import build_prompt, Message
from hermes.litert_model import LiteRTModel
from hermes.router import classify, get_intent, INTENT_CHAT, INTENT_REASONING, INTENT_TOOLS

try:
    from hermes.agent import HermesAgent, AgentConfig, ModelManager
except ImportError:
    pass
