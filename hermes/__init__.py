"""Hermes mobile AI agent for Google AI Edge Gallery (LiteRT-LM).

A small, agentic decoder-only transformer designed to be converted to the
``.litertlm`` format and run on-device via the LiteRT-LM runtime.
"""

from hermes.config import HermesConfig, hermes_1b_config, hermes_270m_config

__all__ = [
    "HermesConfig",
    "hermes_1b_config",
    "hermes_270m_config",
]

__version__ = "0.1.0"
