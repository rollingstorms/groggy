"""
Main formatters module providing high-level formatting functions.
"""

from typing import Any, Dict

from .._groggy import _display_kind

from .array_display import ArrayDisplayFormatter, format_array
from .matrix_display import MatrixDisplayFormatter, format_matrix
from .table_display import TableDisplayFormatter, format_table

# Re-export main formatting functions
__all__ = [
    "format_table",
    "format_matrix",
    "format_array",
    "TableDisplayFormatter",
    "MatrixDisplayFormatter",
    "ArrayDisplayFormatter",
]


# Convenience function that routes to appropriate formatter based on data type
def format_data_structure(
    data: Dict[str, Any], data_type: str = "auto", **kwargs
) -> str:
    """
    Format any groggy data structure for rich display.

    Args:
        data: The data structure to format
        data_type: Type hint ('table', 'matrix', 'array', or 'auto' for auto-detection)
        **kwargs: Additional formatting options

    Returns:
        Formatted string ready for display
    """
    if data_type == "auto":
        data_type = _detect_data_type(data)

    if data_type == "table":
        return format_table(data, **kwargs)
    elif data_type == "matrix":
        return format_matrix(data, **kwargs)
    elif data_type == "array":
        return format_array(data, **kwargs)
    else:
        raise ValueError(f"Unknown data type: {data_type}")


def _detect_data_type(data: Dict[str, Any]) -> str:
    """Auto-detect the data structure type based on contents."""
    rows = data.get("data")
    return _display_kind(
        "columns" in data and "dtypes" in data,
        isinstance(rows, list),
        isinstance(rows, list) and bool(rows) and isinstance(rows[0], list),
    )
