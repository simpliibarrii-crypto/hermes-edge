"""
Code Executor — safe Python code execution for the code-as-tool pattern.
Inspired by smolagents and Anthropic's code-as-tool approach.

SECURITY NOTE: The sandbox here uses AST-level validation and restricted
builtins to prevent common escape patterns (exec/eval/compile/open/__import__).
However, Python's dynamic nature means a fully hardened sandbox is extremely
difficult to achieve. This executor is intended for agent-generated code from
a TRUSTED model, NOT for arbitrary user-supplied code. For full isolation,
consider using a container runtime (nsjail, Docker) or a subprocess with
seccomp/apparmor profiles.
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

# Functions that are blocked even when accessed via attribute chains like
# ().__class__.__bases__[0].__subclasses__()
_BLOCKED_FUNCTIONS = frozenset({
    "exec", "eval", "compile", "open", "__import__",
    "breakpoint", "input", "memoryview", "bytearray",
})

# Dunder methods that are commonly used in sandbox escapes via reflection
_BLOCKED_DUNDER_METHODS = frozenset({
    "__subclasses__", "__class__", "__bases__", "__base__",
    "__mro__", "__globals__", "__code__", "__closure__",
    "__self__", "__func__", "__dict__", "__reduce__",
    "__reduce_ex__", "__getattribute__", "__getattr__",
    "__init_subclass__", "__subclasshook__",
    "__build_class__",
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

        # Whitelist of allowed AST node types — blocks dangerous constructs
        # like class/function definitions.
        # Note: ast.Finally and ast.ExceptHandler are not separate types in
        # Python 3.13+; they are sub-nodes of ast.Try / ast.TryStar.
        _ALLOWED_NODES = frozenset({
            ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.Name, ast.Store,
            ast.Load, ast.Constant, ast.Num, ast.Str, ast.BinOp, ast.UnaryOp,
            ast.BoolOp, ast.Compare, ast.IfExp, ast.Call, ast.Attribute,
            ast.Subscript, ast.Slice, ast.List, ast.Tuple, ast.Dict,
            ast.Set, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
            ast.Pow, ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,
            ast.MatMult, ast.USub, ast.UAdd, ast.Not, ast.Invert, ast.And, ast.Or,
            ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot,
            ast.In, ast.NotIn, ast.If, ast.For, ast.While, ast.Break, ast.Continue,
            ast.Pass, ast.Raise, ast.Try, ast.ExceptHandler,
            ast.Return, ast.Yield, ast.YieldFrom, ast.Starred, ast.Delete,
            ast.With, ast.withitem, ast.comprehension, ast.ListComp, ast.SetComp,
            ast.DictComp, ast.GeneratorExp, ast.Lambda, ast.arg, ast.arguments,
            ast.keyword,
            ast.Import, ast.ImportFrom,
            ast.JoinedStr, ast.FormattedValue,
            ast.alias,
        })
        for node in ast.walk(tree):
            ntype = type(node)
            if ntype not in _ALLOWED_NODES:
                return ExecutionResult(
                    False, error=f"Disallowed syntax: {ntype.__name__}"
                )
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top not in ALLOWED_MODULES:
                        return ExecutionResult(
                            False, error=f"Blocked module: {alias.name}"
                        )
            elif isinstance(node, ast.Call):
                if self._is_blocked_call(node):
                    return ExecutionResult(
                        False, error="Blocked function call detected"
                    )
            elif isinstance(node, ast.Attribute):
                # Block access to dangerous dunder attributes on ALL types
                # This prevents: ().__class__, "".__class__, [].__class__, etc.
                if isinstance(node.ctx, ast.Load):
                    if node.attr in _BLOCKED_DUNDER_METHODS:
                        return ExecutionResult(
                            False,
                            error=f"Blocked dunder attribute access: {node.attr}"
                        )
                    # Block any __.*__ attribute access on literals/constants
                    # as these are exclusively used for sandbox escapes
                    if node.attr.startswith("__") and node.attr.endswith("__"):
                        # Allow limited safe dunders on Name nodes (variables)
                        # but only if they're known safe ones
                        if isinstance(node.value, (ast.Constant, ast.Num, ast.Str,
                                                    ast.List, ast.Tuple, ast.Dict,
                                                    ast.Set)):
                            return ExecutionResult(
                                False,
                                error=f"Blocked dunder attribute on literal: {node.attr}"
                            )

        log.warning(
            "Executing agent-generated code in restricted sandbox. "
            "This is NOT safe for untrusted user input."
        )

        safe_builtins: dict[str, Any] = {
            "abs": abs, "all": all, "any": any, "bool": bool,
            "dict": dict, "enumerate": enumerate, "float": float,
            "format": format, "frozenset": frozenset, "int": int,
            "isinstance": isinstance, "len": len, "list": list,
            "max": max, "min": min, "print": print, "range": range,
            "round": round, "set": set, "sorted": sorted, "str": str,
            "sum": sum, "tuple": tuple, "zip": zip,
            "map": map, "filter": filter, "reversed": reversed,
            "slice": slice, "pow": pow, "divmod": divmod,
            "type": type,
            "__import__": __import__,  # Needed for import statements; module safety is at AST level
            "True": True, "False": False, "None": None,
            # Common exception classes
            "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
            "KeyError": KeyError, "IndexError": IndexError, "AttributeError": AttributeError,
            "RuntimeError": RuntimeError, "StopIteration": StopIteration,
            "ArithmeticError": ArithmeticError, "ZeroDivisionError": ZeroDivisionError,
            "FileNotFoundError": FileNotFoundError, "ImportError": ImportError,
            "NameError": NameError, "SyntaxError": SyntaxError,
            "OSError": OSError, "SystemExit": SystemExit, "KeyboardInterrupt": KeyboardInterrupt,
            # NOTE: 'getattr', 'setattr', 'delattr', 'vars', 'globals', 'locals',
            # 'open', 'exec', 'eval', 'compile' are deliberately excluded.
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

    def _is_blocked_call(self, node: ast.Call) -> bool:
        """Check if a call node targets a blocked function.

        Handles direct calls (eval(...)), attribute calls (obj.eval(...)),
        and chains (obj.__class__.__bases__[0].__subclasses__()).
        """
        func = node.func

        # Direct name call: eval(...)
        if isinstance(func, ast.Name):
            if func.id in _BLOCKED_FUNCTIONS:
                return True

        # Attribute call: obj.eval(...) or obj.attr.eval(...)
        if isinstance(func, ast.Attribute):
            # Check the final attribute name
            if func.attr in _BLOCKED_FUNCTIONS:
                return True
            # Also block if the final attribute is a blocked dunder method
            if func.attr in _BLOCKED_DUNDER_METHODS:
                return True
            # Recursively check the chain for blocked names
            # (catches: ().__class__.__bases__[0].__subclasses__())
            if _has_blocked_chain(func):
                return True

        return False

    def _check_call_safety(self, node: ast.Call) -> None:
        """Legacy check — kept for backward compatibility but superseded by _is_blocked_call."""
        if self._is_blocked_call(node):
            raise SecurityError("Blocked function call detected")


def _has_blocked_chain(node: ast.AST) -> bool:
    """Recursively check an attribute chain for blocked dunder methods.

    Checks patterns like:
      obj.__class__.__bases__[0].__subclasses__()
    """
    if isinstance(node, ast.Attribute):
        if node.attr in _BLOCKED_DUNDER_METHODS:
            return True
        # Block any __X__ on attribute access (reflection pattern)
        if node.attr.startswith("__") and node.attr.endswith("__"):
            return True
        return _has_blocked_chain(node.value)
    if isinstance(node, ast.Call):
        return _has_blocked_chain(node.func)
    if isinstance(node, ast.Subscript):
        return _has_blocked_chain(node.value)
    return False


class SecurityError(Exception):
    pass
