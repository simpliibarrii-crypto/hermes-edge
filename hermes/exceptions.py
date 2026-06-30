class HermesEdgeError(Exception):
    """Base exception for all Hermes Edge errors."""


class ModelLoadError(HermesEdgeError):
    """Raised when a model fails to load."""


class InferenceError(HermesEdgeError):
    """Raised during inference failures."""


class ToolExecutionError(HermesEdgeError):
    """Raised when a tool execution fails."""


class RoutingError(HermesEdgeError):
    """Raised when intent routing fails."""


class WebSearchError(HermesEdgeError):
    """Raised when web search fails."""


class ConfigError(HermesEdgeError):
    """Raised for invalid configuration."""
