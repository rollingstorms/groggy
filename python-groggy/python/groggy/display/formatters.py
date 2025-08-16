"""
Main formatters module providing high-level formatting functions.
"""

from typing import Dict, Any
from .table_display import format_table, TableDisplayFormatter
from .matrix_display import format_matrix, MatrixDisplayFormatter
from .array_display import format_array, ArrayDisplayFormatter

# Re-export main formatting functions
__all__ = [
    'format_table',
    'format_matrix', 
    'format_array',
    'TableDisplayFormatter',
    'MatrixDisplayFormatter',
    'ArrayDisplayFormatter'
]

# Convenience function that routes to appropriate formatter based on data type
def format_data_structure(data: Dict[str, Any], data_type: str = 'auto', **kwargs) -> str:
    """
    Format any groggy data structure for rich display.
    
    Args:
        data: The data structure to format
        data_type: Type hint ('table', 'matrix', 'array', or 'auto' for auto-detection)
        **kwargs: Additional formatting options
    
    Returns:
        Formatted string ready for display
    """
    if data_type == 'auto':
        data_type = _detect_data_type(data)
    
    if data_type == 'table':
        return format_table(data, **kwargs)
    elif data_type == 'matrix':
        return format_matrix(data, **kwargs)
    elif data_type == 'array':
        return format_array(data, **kwargs)
    else:
        raise ValueError(f"Unknown data type: {data_type}")

def _detect_data_type(data: Dict[str, Any]) -> str:
    """Auto-detect the data structure type based on contents."""
    if 'columns' in data and 'dtypes' in data:
        return 'table'
    elif 'data' in data and isinstance(data.get('data'), list):
        first_item = data['data'][0] if data['data'] else None
        if isinstance(first_item, list):
            return 'matrix'
        else:
            return 'array'
    else:
        # Default fallback
        return 'table'
