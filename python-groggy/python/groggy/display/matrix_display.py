"""
Rich display formatter for GraphMatrix structures.
"""

from typing import Any, Dict, Optional

from .._groggy import _display_matrix, _display_positions
from ._presentation import _python_scalar, _resolve_option

from .unicode_chars import Symbols


class MatrixDisplayFormatter:
    """Formatter for GraphMatrix rich display with shape and dtype information."""

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

    def format(self, matrix_data: Dict[str, Any]) -> str:
        """Marshal selected cells, preserving Python scalar representations."""
        data = matrix_data.get("data", [])
        shape = matrix_data.get("shape", (0, 0))
        empty = not data or shape[0] == 0 or shape[1] == 0
        rows = []
        if not empty:
            columns = _display_positions(len(data[0]), self.max_cols)
            for index in _display_positions(len(data), self.max_rows):
                rows.append(
                    [
                        Symbols.TRUNCATION_INDICATOR
                        if index is None or col is None
                        else self._format_matrix_value(data[index][col])
                        for col in columns
                        if index is None or col is None or col < len(data[index])
                    ]
                )
        return _display_matrix(
            rows, empty, str(shape), matrix_data.get("dtype", "object")
        )

    def _format_matrix_value(self, value: Any) -> str:
        return _python_scalar(value, "matrix", self.precision, "object")


def format_matrix(matrix_data: Dict[str, Any], **kwargs) -> str:
    """Convenience function to format a GraphMatrix."""
    formatter = MatrixDisplayFormatter(**kwargs)
    return formatter.format(matrix_data)
