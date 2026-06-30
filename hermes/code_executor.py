"""
Code Executor — safe Python code execution for the code-as-tool pattern.
Inspired by smolagents and Anthropic's code-as-tool approach.
"""
import ast
import io
import logging
import sys
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

ALLOWED_MODULES = frozenset({
    "math", "json", "re", "random", "statistics",
    "itertools", "collections", "datetime", "typing",
})


@dataclass
class ExecutionResult:
    success: bool
    output: str = ""
    error: str = ""
    variables: dict = field(default_factory=dict)


class CodeExecutor:
    """Sandboxed Python code execution for agent-generated code."""

    def __init__(self, context: dict[str, Any] | None = None) -> None:
        self._context: dict[str, Any] = context or {}

    def execute(self, code: str) -> ExecutionResult:
        """Execute Python code in a restricted environment."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ExecutionResult(False, error=f"Syntax error: {e}")

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top not in ALLOWED_MODULES:
                        return ExecutionResult(
                            False, error=f"Blocked module: {alias.name}"
                        )
            elif isinstance(node, ast.Call):
                self._check_call_safety(node)

        safe_builtins: dict[str, Any] = {
            "abs": abs, "all": all, "any": any, "bool": bool,
            "dict": dict, "enumerate": enumerate, "float": float,
            "format": format, "frozenset": frozenset, "int": int,
            "isinstance": isinstance, "len": len, "list": list,
            "max": max, "min": min, "print": print, "range": range,
            "round": round, "set": set, "sorted": sorted, "str": str,
            "sum": sum, "tuple": tuple, "type": type, "zip": zip,
            "map": map, "filter": filter, "reversed": reversed,
            "slice": slice, "pow": pow, "divmod": divmod,
            "True": True, "False": False, "None": None,
        }

        safe_globals: dict[str, Any] = {
            "__builtins__": safe_builtins,
        }
        safe_globals.update(self._context)

        old_stdout = sys.stdout
        sys.stdout = captured = io.StringIO()

        try:
            exec(code, safe_globals)
            output = captured.getvalue()
            result_vars = {
                k: v for k, v in safe_globals.items()
                if not k.startswith("_")
                and k not in self._context
                and k != "__builtins__"
            }
            return ExecutionResult(
                success=True,
                output=output,
                variables=result_vars,
            )
        except Exception as e:
            return ExecutionResult(
                False, error=f"{type(e).__name__}: {e}"
            )
        finally:
            sys.stdout = old_stdout

    def _check_call_safety(self, node: ast.Call) -> None:
        blocked = frozenset({"exec", "eval", "compile", "open", "__import__"})
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in blocked:
                raise SecurityError(f"Blocked built-in: {node.func.attr}")
        elif isinstance(node.func, ast.Name):
            if node.func.id in blocked:
                raise SecurityError(f"Blocked built-in: {node.func.id}")


class SecurityError(Exception):
    pass
