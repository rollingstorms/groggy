"""
Simple expression parser for builder DSL.

Parses string expressions like "sum(ranks[neighbors(node)])" into
Expr JSON structures that Rust can understand.
"""

import re
from typing import Any, Dict, List


def parse_expression(expr_str: str) -> Dict[str, Any]:
    """
    Parse a string expression into an Expr JSON structure.

    Supports:
    - sum(var[neighbors(node)]) - sum of neighbor values
    - mean(var[neighbors(node)]) - mean of neighbor values
    - mode(var[neighbors(node)]) - most common neighbor value
    - var[neighbors(node)] - list of neighbor values
    - neighbors(node) - list of neighbor IDs
    - Arithmetic: var * 2, var + 1, etc.

    Args:
        expr_str: Expression string to parse

    Returns:
        Expr JSON structure

    Example:
        >>> parse_expression("sum(ranks[neighbors(node)])")
        {
            "type": "call",
            "func": "sum",
            "args": [{
                "type": "call",
                "func": "neighbor_values",
                "args": [{"type": "var", "name": "ranks"}]
            }]
        }
    """
    expr_str = expr_str.strip()

    # Handle aggregation functions: sum(...), mean(...), mode(...)
    agg_match = re.match(
        r"(sum|mean|mode|count)\s*\(\s*(.+)\s*\)", expr_str, re.IGNORECASE
    )
    if agg_match:
        func_name = agg_match.group(1).lower()
        inner_expr = agg_match.group(2).strip()

        # Parse the inner expression
        inner = _parse_inner_expression(inner_expr)

        return {"type": "call", "func": func_name, "args": [inner]}

    # Handle simple expressions
    return _parse_inner_expression(expr_str)


def _parse_inner_expression(expr_str: str) -> Dict[str, Any]:
    """Parse inner expression (variable access, neighbor access, etc.)"""
    expr_str = expr_str.strip()

    # Handle: var[neighbors(node)]
    neighbor_val_match = re.match(
        r"(\w+)\s*\[\s*neighbors\s*\(\s*node\s*\)\s*\]", expr_str
    )
    if neighbor_val_match:
        var_name = neighbor_val_match.group(1)
        return {
            "type": "call",
            "func": "neighbor_values",
            "args": [{"type": "var", "name": var_name}],
        }

    # Handle: neighbors(node)
    if re.match(r"neighbors\s*\(\s*node\s*\)", expr_str):
        return {"type": "call", "func": "neighbors", "args": []}

    # Handle arithmetic operations: var * 2, var + 1, etc.
    arith_match = re.match(r"(\w+)\s*([+\-*/])\s*(\d+\.?\d*)", expr_str)
    if arith_match:
        var_name = arith_match.group(1)
        op = arith_match.group(2)
        value = arith_match.group(3)

        op_map = {"+": "add", "-": "sub", "*": "mul", "/": "div"}

        return {
            "type": "binary_op",
            "op": op_map[op],
            "left": {"type": "var", "name": var_name},
            "right": {"type": "const", "value": float(value)},
        }

    # Handle plain variable reference
    if re.match(r"^\w+$", expr_str):
        return {"type": "var", "name": expr_str}

    # Handle constants
    try:
        value = float(expr_str)
        return {"type": "const", "value": value}
    except ValueError:
        pass

    raise ValueError(f"Unable to parse expression: {expr_str}")


def build_neighbor_aggregation_expr(var_name: str, agg_func: str) -> Dict[str, Any]:
    """
    Build an expression for neighbor aggregation.

    Args:
        var_name: Variable to aggregate from neighbors
        agg_func: Aggregation function (sum, mean, mode)

    Returns:
        Expr JSON structure

    Example:
        >>> build_neighbor_aggregation_expr("ranks", "sum")
        {
            "type": "call",
            "func": "sum",
            "args": [{
                "type": "call",
                "func": "neighbor_values",
                "args": [{"type": "var", "name": "ranks"}]
            }]
        }
    """
    return {
        "type": "call",
        "func": agg_func,
        "args": [
            {
                "type": "call",
                "func": "neighbor_values",
                "args": [{"type": "var", "name": var_name}],
            }
        ],
    }
