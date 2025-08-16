"""
Rich display formatter for GraphMatrix structures.
"""

from typing import List, Dict, Any, Optional, Tuple
from .unicode_chars import BoxChars, Symbols, Colors, colorize
from .truncation import smart_matrix_truncation, truncate_string

class MatrixDisplayFormatter:
    """Formatter for GraphMatrix rich display with shape and dtype information."""
    
    def __init__(self, max_rows: int = 10, max_cols: int = 8, max_width: int = 120, precision: int = 2):
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.max_width = max_width
        self.precision = precision
    
    def format(self, matrix_data: Dict[str, Any]) -> str:
        """
        Format a GraphMatrix for rich display.
        
        Expected matrix_data structure:
        {
            'data': [
                [0.12, -1.50, 2.00, 0.00],
                [3.14, 0.00, float('nan'), 1.25],
                [-0.01, 4.50, -2.30, 8.00]
            ],
            'shape': (3, 4),
            'dtype': 'f32',
            'column_names': ['col1', 'col2', 'col3', 'col4']  # Optional
        }
        """
        data = matrix_data.get('data', [])
        shape = matrix_data.get('shape', (0, 0))
        dtype = matrix_data.get('dtype', 'object')
        column_names = matrix_data.get('column_names', None)
        
        if not data or shape[0] == 0 or shape[1] == 0:
            return self._format_empty_matrix(shape, dtype)
        
        # Smart truncation for large matrices
        truncated_data, rows_truncated, cols_truncated = smart_matrix_truncation(
            data, self.max_rows, self.max_cols
        )
        
        # Calculate column widths based on content
        col_widths = self._calculate_matrix_column_widths(truncated_data)
        
        # Build the formatted matrix
        lines = []
        
        # Header with section indicator
        lines.append(f"{Symbols.HEADER_PREFIX} gr.matrix")
        
        # Top border
        lines.append(self._build_border_line(col_widths, 'top'))
        
        # Data rows
        for row in truncated_data:
            lines.append(self._build_matrix_row(row, col_widths))
        
        # Bottom border
        lines.append(self._build_border_line(col_widths, 'bottom'))
        
        # Shape and dtype information
        shape_info = f"shape: {shape} • dtype: {dtype}"
        lines.append(shape_info)
        
        return '\n'.join(lines)
    
    def _format_empty_matrix(self, shape: Tuple[int, int], dtype: str) -> str:
        """Format an empty matrix."""
        return f"{Symbols.HEADER_PREFIX} gr.matrix (empty) • shape: {shape} • dtype: {dtype}"
    
    def _format_matrix_value(self, value: Any) -> str:
        """Format a single matrix value for display."""
        if value is None:
            return Symbols.NULL_DISPLAY
        elif isinstance(value, float):
            if str(value).lower() in ['nan', 'inf', '-inf']:
                return Symbols.NULL_DISPLAY
            else:
                return f"{value:.{self.precision}f}"
        elif value == Symbols.TRUNCATION_INDICATOR:
            return Symbols.TRUNCATION_INDICATOR
        else:
            return str(value)
    
    def _calculate_matrix_column_widths(self, data: List[List[Any]]) -> List[int]:
        """Calculate optimal column widths for matrix display."""
        if not data or not data[0]:
            return []
        
        num_cols = len(data[0])
        col_widths = []
        
        for col_idx in range(num_cols):
            # Get all values in this column
            col_values = [self._format_matrix_value(row[col_idx]) for row in data if col_idx < len(row)]
            
            # Calculate max width needed
            max_width = max(len(str(val)) for val in col_values) if col_values else 6
            
            # Ensure minimum width and maximum reasonable width
            width = max(6, min(max_width, 12))
            col_widths.append(width)
        
        return col_widths
    
    def _build_border_line(self, col_widths: List[int], position: str) -> str:
        """Build a border line for matrix display."""
        if position == 'top':
            left = BoxChars.TOP_LEFT
            right = BoxChars.TOP_RIGHT
            sep = BoxChars.T_TOP
        else:  # bottom
            left = BoxChars.BOTTOM_LEFT
            right = BoxChars.BOTTOM_RIGHT
            sep = BoxChars.T_BOTTOM
        
        segments = []
        for width in col_widths:
            segment = BoxChars.HORIZONTAL * (width + 2)  # +2 for padding
            segments.append(segment)
        
        return left + sep.join(segments) + right
    
    def _build_matrix_row(self, row_data: List[Any], col_widths: List[int]) -> str:
        """Build a matrix data row with proper alignment."""
        cells = []
        for value, width in zip(row_data, col_widths):
            formatted_value = self._format_matrix_value(value)
            
            # Right-align numeric values, center-align special symbols
            if formatted_value == Symbols.TRUNCATION_INDICATOR:
                padded = formatted_value.center(width)
            else:
                # Right-align for better number alignment
                padded = formatted_value.rjust(width)
            
            cells.append(f" {padded} ")
        
        return BoxChars.VERTICAL + BoxChars.VERTICAL.join(cells) + BoxChars.VERTICAL

def format_matrix(matrix_data: Dict[str, Any], **kwargs) -> str:
    """Convenience function to format a GraphMatrix."""
    formatter = MatrixDisplayFormatter(**kwargs)
    return formatter.format(matrix_data)
