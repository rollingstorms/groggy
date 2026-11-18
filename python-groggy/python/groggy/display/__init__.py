"""
Rich Display Module for Groggy Data Structures

Provides beautiful, professional formatting for GraphTable, GraphMatrix, and GraphArray
using Unicode box-drawing characters and smart truncation.

Based on display_draft.txt requirements.
"""

from . import _presentation
from .array_display import ArrayDisplayFormatter
from .formatters import format_array, format_matrix, format_table
from .matrix_display import MatrixDisplayFormatter
from .table_display import TableDisplayFormatter

__all__ = [
    "format_table",
    "format_matrix",
    "format_array",
    "TableDisplayFormatter",
    "MatrixDisplayFormatter",
    "ArrayDisplayFormatter",
]

# Default display settings
DEFAULT_MAX_ROWS = 10
DEFAULT_MAX_COLS = 8
DEFAULT_MAX_WIDTH = 120
DEFAULT_PRECISION = 2


def configure_display(max_rows=None, max_cols=None, max_width=None, precision=None):
    """Set defaults for subsequently created dictionary display formatters.

    Explicit per-formatter arguments take precedence. Existing instances and the
    independent native typed table/HTML configuration retain their settings.
    """
    global DEFAULT_MAX_ROWS, DEFAULT_MAX_COLS, DEFAULT_MAX_WIDTH, DEFAULT_PRECISION

    supplied = {
        "max_rows": max_rows,
        "max_cols": max_cols,
        "max_width": max_width,
        "precision": precision,
    }
    # Validate everything before publishing any new defaults.
    updates = {
        key: _presentation._resolve_option(key, value, 0)
        for key, value in supplied.items()
        if value is not None
    }
    _presentation._DISPLAY_OVERRIDES.update(updates)
    DEFAULT_MAX_ROWS = updates.get("max_rows", DEFAULT_MAX_ROWS)
    DEFAULT_MAX_COLS = updates.get("max_cols", DEFAULT_MAX_COLS)
    DEFAULT_MAX_WIDTH = updates.get("max_width", DEFAULT_MAX_WIDTH)
    DEFAULT_PRECISION = updates.get("precision", DEFAULT_PRECISION)
