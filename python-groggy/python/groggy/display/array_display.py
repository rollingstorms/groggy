"""
Rich display formatter for GraphArray structures.
"""

from typing import Any, Dict, Optional

from .._groggy import _display_array, _display_positions
from ._presentation import _python_scalar, _resolve_option

from .unicode_chars import Symbols


class ArrayDisplayFormatter:
    """Formatter for GraphArray rich display with column-style layout."""

    def __init__(
        self,
        max_rows: Optional[int] = None,
        max_width: Optional[int] = None,
        precision: Optional[int] = None,
    ):
        self.max_rows = _resolve_option("max_rows", max_rows, 10)
        self.max_width = _resolve_option("max_width", max_width, 60)
        self.precision = _resolve_option("precision", precision, 4)

    def format(self, array_data: Dict[str, Any]) -> str:
        """Marshal visible Python scalar spellings to the shared native layout."""
        data = array_data.get("data", [])
        cells = []
        for index in _display_positions(len(data), self.max_rows):
            if index is None:
                cells.append((None, Symbols.ELLIPSIS, False))
            else:
                value = data[index]
                cells.append(
                    (
                        index,
                        self._format_array_value(value),
                        isinstance(value, (int, float))
                        and str(value).lower() not in ("nan", "inf", "-inf"),
                    )
                )
        return _display_array(
            cells,
            len(data),
            array_data.get("name", "col1"),
            array_data.get("dtype", "object"),
            str(array_data.get("shape", (0,))),
        )

    def _format_array_value(self, value: Any) -> str:
        return _python_scalar(value, "array", self.precision, "object")


def format_array(array_data: Dict[str, Any], **kwargs) -> str:
    """Convenience function to format a GraphArray."""
    formatter = ArrayDisplayFormatter(**kwargs)
    return formatter.format(array_data)
