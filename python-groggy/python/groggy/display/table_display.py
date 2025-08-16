"""
Rich display formatter for GraphTable structures.
"""

from typing import List, Dict, Any, Optional, Tuple
from .unicode_chars import BoxChars, Symbols, Colors, colorize
from .truncation import truncate_rows, truncate_columns, calculate_column_widths, truncate_string

class TableDisplayFormatter:
    """Formatter for GraphTable rich display with Polars-style formatting."""
    
    def __init__(self, max_rows: int = 10, max_cols: int = 8, max_width: int = 120, precision: int = 2):
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.max_width = max_width
        self.precision = precision
    
    def format(self, table_data: Dict[str, Any]) -> str:
        """
        Format a GraphTable for rich display.
        
        Expected table_data structure:
        {
            'columns': ['name', 'city', 'age', 'score', 'joined'],
            'dtypes': {'name': 'str', 'city': 'category', 'age': 'int', 'score': 'float', 'joined': 'date'},
            'data': [
                ['Alice', 'NYC', 25, 91.5, '2024-02-15'],
                ['Bob', 'Paris', 30, 87.0, '2023-11-20'],
                ...
            ],
            'shape': (1000, 5),
            'nulls': {'score': 12},
            'index_type': 'int64'
        }
        """
        columns = table_data.get('columns', [])
        dtypes = table_data.get('dtypes', {})
        data = table_data.get('data', [])
        shape = table_data.get('shape', (0, 0))
        nulls = table_data.get('nulls', {})
        index_type = table_data.get('index_type', 'int64')
        
        if not columns or not data:
            return self._format_empty_table()
        
        # Add index column
        headers = ['#'] + columns
        type_headers = [''] + [self._format_dtype(dtypes.get(col, 'object')) for col in columns]
        
        # Add row indices to data
        indexed_data = []
        for i, row in enumerate(data):
            indexed_row = [str(i)] + [self._format_value(val, dtypes.get(columns[j], 'object')) for j, val in enumerate(row)]
            indexed_data.append(indexed_row)
        
        # Truncate if necessary
        truncated_data, rows_truncated = truncate_rows(indexed_data, self.max_rows)
        truncated_headers, truncated_data, cols_truncated = truncate_columns(
            headers, truncated_data, self.max_cols
        )
        truncated_type_headers = type_headers[:len(truncated_headers)]
        
        # Calculate column widths
        col_widths = calculate_column_widths(truncated_headers, truncated_data, self.max_width)
        
        # Build the formatted table
        lines = []
        
        # Header with section indicator
        lines.append(f"{Symbols.HEADER_PREFIX} gr.table")
        
        # Top border
        lines.append(self._build_border_line(col_widths, 'top'))
        
        # Column headers
        lines.append(self._build_data_line(truncated_headers, col_widths, bold=True))
        lines.append(self._build_data_line(truncated_type_headers, col_widths, dim=True))
        
        # Header separator
        lines.append(self._build_border_line(col_widths, 'middle'))
        
        # Data rows
        for row in truncated_data:
            lines.append(self._build_data_line(row, col_widths))
        
        # Bottom border
        lines.append(self._build_border_line(col_widths, 'bottom'))
        
        # Summary statistics
        summary_parts = [
            f"rows: {shape[0]:,}",
            f"cols: {shape[1]}",
        ]
        
        if nulls:
            null_info = ', '.join(f"{col}={count}" for col, count in nulls.items())
            summary_parts.append(f"nulls: {null_info}")
        
        summary_parts.append(f"index: {index_type}")
        summary = f" {Symbols.DOT_SEPARATOR} ".join(summary_parts)
        lines.append(summary)
        
        return '\n'.join(lines)
    
    def _format_empty_table(self) -> str:
        """Format an empty table."""
        return f"{Symbols.HEADER_PREFIX} gr.table (empty)"
    
    def _format_dtype(self, dtype: str) -> str:
        """Format data type for display."""
        dtype_map = {
            'string': 'str',
            'category': 'cat',
            'int64': 'i64',
            'int32': 'i32', 
            'float64': 'f64',
            'float32': 'f32',
            'bool': 'bool',
            'datetime': 'date',
            'object': 'obj'
        }
        
        base_type = dtype_map.get(dtype, dtype)
        
        # Add size hints for string/category types
        if dtype in ['string', 'str']:
            return f"{base_type}[8]"  # Could be dynamic based on max length
        elif dtype in ['category', 'cat']:
            return f"{base_type}(12)"  # Could be dynamic based on category count
        else:
            return base_type
    
    def _format_value(self, value: Any, dtype: str) -> str:
        """Format a single value for display."""
        if value is None or (isinstance(value, float) and str(value).lower() in ['nan', 'inf', '-inf']):
            return Symbols.NULL_DISPLAY
        
        if dtype in ['float64', 'float32', 'float'] and isinstance(value, (int, float)):
            return f"{float(value):.{self.precision}f}"
        elif dtype in ['datetime', 'date'] and isinstance(value, str):
            # Truncate dates for display
            return truncate_string(value, 10) + (Symbols.ELLIPSIS if len(value) > 10 else '')
        elif isinstance(value, str):
            # Truncate long strings
            return truncate_string(value, 12)
        else:
            return str(value)
    
    def _build_border_line(self, col_widths: List[int], position: str) -> str:
        """Build a border line (top, middle, bottom)."""
        if position == 'top':
            left = BoxChars.TOP_LEFT
            right = BoxChars.TOP_RIGHT
            sep = BoxChars.T_TOP
        elif position == 'middle':
            left = BoxChars.T_LEFT
            right = BoxChars.T_RIGHT
            sep = BoxChars.CROSS
        else:  # bottom
            left = BoxChars.BOTTOM_LEFT
            right = BoxChars.BOTTOM_RIGHT
            sep = BoxChars.T_BOTTOM
        
        segments = []
        for i, width in enumerate(col_widths):
            segment = BoxChars.HORIZONTAL * (width + 2)  # +2 for padding
            segments.append(segment)
        
        return left + sep.join(segments) + right
    
    def _build_data_line(self, row_data: List[str], col_widths: List[int], bold: bool = False, dim: bool = False) -> str:
        """Build a data line with proper padding and alignment."""
        cells = []
        for i, (value, width) in enumerate(zip(row_data, col_widths)):
            # Truncate if value is too long
            display_value = truncate_string(str(value), width)
            
            # Pad to column width (left-align for most, right-align for numbers in index)
            if i == 0:  # Index column - right align
                padded = display_value.rjust(width)
            else:  # Data columns - left align
                padded = display_value.ljust(width)
            
            # Apply formatting
            if bold:
                padded = colorize(padded, bold=True)
            elif dim:
                padded = colorize(padded, dim=True)
            
            cells.append(f" {padded} ")  # Add padding around content
        
        return BoxChars.VERTICAL + BoxChars.VERTICAL.join(cells) + BoxChars.VERTICAL

def format_table(table_data: Dict[str, Any], **kwargs) -> str:
    """Convenience function to format a GraphTable."""
    formatter = TableDisplayFormatter(**kwargs)
    return formatter.format(table_data)
