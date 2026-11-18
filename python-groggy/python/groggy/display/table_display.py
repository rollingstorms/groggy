"""
Rich display formatter for GraphTable structures.
"""

from typing import Any, Dict, Optional

from .._groggy import _display_dtype, _display_positions, _display_table
from ._presentation import _python_scalar, _resolve_option

from .unicode_chars import Symbols, has_color_support


class TableDisplayFormatter:
    """Formatter for GraphTable rich display with Polars-style formatting."""

    def __init__(
        self,
        max_rows: Optional[int] = None,
        max_cols: Optional[int] = None,
        max_width: Optional[int] = None,
        precision: Optional[int] = None,
    ):
        self.max_rows = _resolve_option("max_rows", max_rows, 10)
        self.max_cols = _resolve_option("max_cols", max_cols, 20)
        self.max_width = _resolve_option("max_width", max_width, 120)
        self.precision = _resolve_option("precision", precision, 2)

    def format(self, table_data: Dict[str, Any]) -> str:
        """Marshal only visible cells; native code owns rectangular rendering."""
        columns = table_data.get("columns", [])
        dtypes = table_data.get("dtypes", {})
        data = table_data.get("data", [])
        headers, types, rows = [], [], []
        empty = not columns or not data
        if not empty:
            positions = _display_positions(len(columns) + 1, self.max_cols)
            headers = [
                Symbols.ELLIPSIS
                if col is None
                else "#"
                if col == 0
                else columns[col - 1]
                for col in positions
            ]
            types = [
                Symbols.ELLIPSIS
                if col is None
                else ""
                if col == 0
                else self._format_dtype(dtypes.get(columns[col - 1], "object"))
                for col in positions
            ]
            for index in _display_positions(len(data), self.max_rows):
                row = []
                for col in positions:
                    if index is None:
                        row.append(Symbols.ELLIPSIS)
                    elif col is None:
                        row.append(Symbols.TRUNCATION_INDICATOR)
                    elif col == 0:
                        row.append(str(index))
                    elif col - 1 < len(data[index]):
                        row.append(
                            self._format_value(
                                data[index][col - 1],
                                dtypes.get(columns[col - 1], "object"),
                            )
                        )
                    else:
                        row.append("")
                rows.append(row)
        shape = table_data.get("shape", (0, 0))
        summary = [f"rows: {shape[0]:,}", f"cols: {shape[1]}"]
        nulls = table_data.get("nulls", {})
        if nulls:
            summary.append(
                "nulls: " + ", ".join(f"{key}={value}" for key, value in nulls.items())
            )
        summary.append("index: " + table_data.get("index_type", "int64"))
        if table_data.get("table_type") == "GraphTable":
            nodes = table_data.get("nodes_shape", (0, 0))
            edges = table_data.get("edges_shape", (0, 0))
            summary.extend(
                [f"nodes: {nodes[0]:,}×{nodes[1]}", f"edges: {edges[0]:,}×{edges[1]}"]
            )
        return _display_table(
            headers,
            types,
            rows,
            empty,
            f" {Symbols.DOT_SEPARATOR} ".join(summary),
            self.max_width,
            has_color_support(),
        )

    def _format_dtype(self, dtype: str) -> str:
        return _display_dtype(dtype, True)

    def _format_value(self, value: Any, dtype: str) -> str:
        return _python_scalar(value, "table", self.precision, dtype)


def format_table(table_data: Dict[str, Any], **kwargs) -> str:
    """Convenience function to format a GraphTable."""
    formatter = TableDisplayFormatter(**kwargs)
    return formatter.format(table_data)
