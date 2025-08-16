"""
Rich display formatter for GraphArray structures.
"""

from typing import List, Dict, Any, Optional
from .unicode_chars import BoxChars, Symbols, Colors, colorize
from .truncation import truncate_rows, truncate_string

class ArrayDisplayFormatter:
    """Formatter for GraphArray rich display with column-style layout."""
    
    def __init__(self, max_rows: int = 10, max_width: int = 60, precision: int = 4):
        self.max_rows = max_rows
        self.max_width = max_width
        self.precision = precision
    
    def format(self, array_data: Dict[str, Any]) -> str:
        """
        Format a GraphArray for rich display.
        
        Expected array_data structure:
        {
            'data': [0.125, 3.1416, float('nan'), -2.75, 8.0, 34],
            'dtype': 'f32',
            'shape': (6,),
            'name': 'col1'  # Optional column name
        }
        """
        data = array_data.get('data', [])
        dtype = array_data.get('dtype', 'object')
        shape = array_data.get('shape', (0,))
        name = array_data.get('name', 'col1')
        
        if not data:
            return self._format_empty_array(shape, dtype, name)
        
        # Truncate data if too long
        truncated_data, was_truncated = truncate_rows(data, self.max_rows)
        
        # Calculate column widths
        index_width = len(str(len(data) - 1)) if data else 1
        index_width = max(index_width, 1)  # Minimum width
        
        # Format all values to determine value column width
        formatted_values = [self._format_array_value(val) for val in truncated_data]
        value_width = max(len(val) for val in formatted_values) if formatted_values else 6
        value_width = max(value_width, len(name))  # At least as wide as column name
        value_width = min(value_width, 20)  # Maximum reasonable width
        
        # Build the formatted array
        lines = []
        
        # Header with section indicator
        lines.append(f"{Symbols.HEADER_PREFIX} gr.array")
        lines.append("")  # Empty line for spacing
        
        # Top border
        lines.append(self._build_border_line(index_width, value_width, 'top'))
        
        # Column headers
        lines.append(self._build_header_line(index_width, value_width, name, dtype))
        
        # Header separator
        lines.append(self._build_border_line(index_width, value_width, 'middle'))
        
        # Data rows
        for i, value in enumerate(truncated_data):
            if isinstance(value, list) and len(value) == 1 and value[0] == Symbols.ELLIPSIS:
                # This is an ellipsis row
                lines.append(self._build_ellipsis_line(index_width, value_width))
            else:
                # Regular data row
                original_index = i if not was_truncated or i < self.max_rows // 2 else len(data) - (len(truncated_data) - i)
                lines.append(self._build_data_line(original_index, value, index_width, value_width))
        
        # Bottom border
        lines.append(self._build_border_line(index_width, value_width, 'bottom'))
        
        # Shape information
        shape_info = f"shape: {shape}"
        lines.append(shape_info)
        
        return '\n'.join(lines)
    
    def _format_empty_array(self, shape: tuple, dtype: str, name: str) -> str:
        """Format an empty array."""
        return f"{Symbols.HEADER_PREFIX} gr.array (empty) • shape: {shape} • dtype: {dtype}"
    
    def _format_array_value(self, value: Any) -> str:
        """Format a single array value for display."""
        if value is None:
            return Symbols.NULL_DISPLAY
        elif isinstance(value, float):
            if str(value).lower() in ['nan', 'inf', '-inf']:
                return Symbols.NULL_DISPLAY
            else:
                return f"{value:.{self.precision}g}"  # Use 'g' format for smart precision
        elif isinstance(value, int):
            return str(value)
        elif isinstance(value, str):
            return truncate_string(value, 15)
        elif value == Symbols.ELLIPSIS:
            return Symbols.ELLIPSIS
        else:
            return str(value)
    
    def _build_border_line(self, index_width: int, value_width: int, position: str) -> str:
        """Build a border line for array display."""
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
        
        index_segment = BoxChars.HORIZONTAL * (index_width + 2)
        value_segment = BoxChars.HORIZONTAL * (value_width + 2)
        
        return left + index_segment + sep + value_segment + right
    
    def _build_header_line(self, index_width: int, value_width: int, name: str, dtype: str) -> str:
        """Build the header line with column name and type."""
        # Index header
        index_header = "#".center(index_width)
        
        # Value header with name and type
        value_header = truncate_string(name, value_width)
        type_header = dtype
        
        # Build the line
        index_cell = f" {index_header} "
        value_cell = f" {value_header.ljust(value_width)} "
        
        line1 = BoxChars.VERTICAL + index_cell + BoxChars.VERTICAL + value_cell + BoxChars.VERTICAL
        
        # Second line with type info
        empty_index = " " * (index_width + 2)
        type_cell = f" {type_header.ljust(value_width)} "
        line2 = BoxChars.VERTICAL + empty_index + BoxChars.VERTICAL + type_cell + BoxChars.VERTICAL
        
        return line1 + '\n' + line2
    
    def _build_data_line(self, index: int, value: Any, index_width: int, value_width: int) -> str:
        """Build a data line with index and value."""
        formatted_index = str(index).rjust(index_width)
        formatted_value = self._format_array_value(value)
        
        # Right-align numbers, left-align strings
        if isinstance(value, (int, float)) and value is not None and str(value).lower() not in ['nan', 'inf', '-inf']:
            padded_value = formatted_value.rjust(value_width)
        else:
            padded_value = formatted_value.ljust(value_width)
        
        index_cell = f" {formatted_index} "
        value_cell = f" {padded_value} "
        
        return BoxChars.VERTICAL + index_cell + BoxChars.VERTICAL + value_cell + BoxChars.VERTICAL
    
    def _build_ellipsis_line(self, index_width: int, value_width: int) -> str:
        """Build an ellipsis line for truncated arrays."""
        ellipsis_index = Symbols.ELLIPSIS.center(index_width)
        ellipsis_value = Symbols.ELLIPSIS.center(value_width)
        
        index_cell = f" {ellipsis_index} "
        value_cell = f" {ellipsis_value} "
        
        return BoxChars.VERTICAL + index_cell + BoxChars.VERTICAL + value_cell + BoxChars.VERTICAL

def format_array(array_data: Dict[str, Any], **kwargs) -> str:
    """Convenience function to format a GraphArray."""
    formatter = ArrayDisplayFormatter(**kwargs)
    return formatter.format(array_data)
