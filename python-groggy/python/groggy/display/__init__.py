"""
Rich Display Module for Groggy Data Structures

Provides beautiful, professional formatting for GraphTable, GraphMatrix, and GraphArray
using Unicode box-drawing characters and smart truncation.

Based on display_draft.txt requirements.
"""

from .formatters import format_table, format_matrix, format_array
from .table_display import TableDisplayFormatter
from .matrix_display import MatrixDisplayFormatter
from .array_display import ArrayDisplayFormatter

__all__ = [
    'format_table',
    'format_matrix', 
    'format_array',
    'TableDisplayFormatter',
    'MatrixDisplayFormatter',
    'ArrayDisplayFormatter'
]

# Default display settings
DEFAULT_MAX_ROWS = 10
DEFAULT_MAX_COLS = 8
DEFAULT_MAX_WIDTH = 120
DEFAULT_PRECISION = 2

def configure_display(max_rows=None, max_cols=None, max_width=None, precision=None):
    """Configure global display settings for all data structures."""
    global DEFAULT_MAX_ROWS, DEFAULT_MAX_COLS, DEFAULT_MAX_WIDTH, DEFAULT_PRECISION
    
    if max_rows is not None:
        DEFAULT_MAX_ROWS = max_rows
    if max_cols is not None:
        DEFAULT_MAX_COLS = max_cols  
    if max_width is not None:
        DEFAULT_MAX_WIDTH = max_width
    if precision is not None:
        DEFAULT_PRECISION = precision
