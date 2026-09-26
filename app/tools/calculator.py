"""
Calculator tool.

Evaluates simple arithmetic expressions safely using Python's `ast` module
(no `eval` of arbitrary code). Supports +, -, *, /, //, %, ** and unary
minus, on numbers only.
"""

from __future__ import annotations

import ast
import operator

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class CalculatorError(ValueError):
    pass


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.operand))
    raise CalculatorError(f"Unsupported expression element: {ast.dump(node)}")


def calculate(expression: str) -> float:
    """Safely evaluate an arithmetic expression string and return the result."""
    try:
        tree = ast.parse(expression, mode="eval")
        return _eval_node(tree.body)
    except (SyntaxError, CalculatorError, ZeroDivisionError, TypeError) as e:
        raise CalculatorError(f"Could not evaluate '{expression}': {e}") from e
