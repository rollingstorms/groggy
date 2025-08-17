"""
Smart truncation algorithms for large datasets.
"""

from typing import List, Tuple, Any, Optional
from .unicode_chars import Symbols

def truncate_rows(data: List[Any], max_rows: int) -> Tuple[List[Any], bool]:
    """
    Truncate rows with smart first/last display.
    
    Returns (truncated_data, was_truncated)
    """
    if len(data) <= max_rows:
        return data, False
    
    if max_rows < 3:
        # Too few rows to show meaningful truncation
        return data[:max_rows], True
    
    # Show first half and last half with ellipsis in middle
    show_first = max_rows // 2
    show_last = max_rows - show_first - 1  # -1 for ellipsis row
    
    truncated = (
        data[:show_first] + 
        [create_ellipsis_row(data[0] if data else [])] +
        data[-show_last:] if show_last > 0 else []
    )
    
    return truncated, True

def truncate_columns(headers: List[str], data: List[List[Any]], max_cols: int) -> Tuple[List[str], List[List[Any]], bool]:
    """
    Truncate columns with smart first/last display.
    
    Returns (truncated_headers, truncated_data, was_truncated)
    """
    if len(headers) <= max_cols:
        return headers, data, False
    
    if max_cols < 3:
        # Too few columns to show meaningful truncation
        return headers[:max_cols], [row[:max_cols] for row in data], True
    
    # Show first half and last half with ellipsis in middle
    show_first = max_cols // 2
    show_last = max_cols - show_first - 1  # -1 for ellipsis column
    
    # Truncate headers
    truncated_headers = (
        headers[:show_first] + 
        [Symbols.ELLIPSIS] +
        (headers[-show_last:] if show_last > 0 else [])
    )
    
    # Truncate data rows
    truncated_data = []
    for row in data:
        truncated_row = (
            row[:show_first] + 
            [Symbols.TRUNCATION_INDICATOR] +
            (row[-show_last:] if show_last > 0 else [])
        )
        truncated_data.append(truncated_row)
    
    return truncated_headers, truncated_data, True

def create_ellipsis_row(sample_row: Any) -> Any:
    """Create an ellipsis row that matches the structure of a data row."""
    # Handle 1D arrays where sample_row is a scalar value
    if not hasattr(sample_row, '__len__') or isinstance(sample_row, str):
        return Symbols.ELLIPSIS
    
    # Handle 2D arrays where sample_row is a list/array
    return [Symbols.ELLIPSIS] * len(sample_row) if sample_row else [Symbols.ELLIPSIS]

def truncate_string(text: str, max_width: int, ellipsis: str = Symbols.ELLIPSIS) -> str:
    """Truncate a string to max_width with ellipsis if needed."""
    if len(text) <= max_width:
        return text
    
    if max_width < len(ellipsis):
        return text[:max_width]
    
    return text[:max_width - len(ellipsis)] + ellipsis

def calculate_column_widths(headers: List[str], data: List[List[Any]], max_total_width: int) -> List[int]:
    """
    Calculate optimal column widths that fit within max_total_width.
    
    Uses a fair distribution algorithm that prioritizes readability.
    """
    if not headers:
        return []
    
    num_cols = len(headers)
    if num_cols == 0:
        return []
    
    # Calculate minimum widths needed for each column
    min_widths = []
    for i, header in enumerate(headers):
        col_values = [str(row[i]) if i < len(row) else '' for row in data]
        col_values.append(header)  # Include header in width calculation
        
        min_width = max(len(str(val)) for val in col_values) if col_values else 0
        min_widths.append(max(min_width, 4))  # Minimum 4 chars per column for better readability
    
    # Account for borders and separators: | col1 | col2 | col3 |
    # Each column needs 2 extra chars for " |", plus 1 for initial "|"
    border_width = sum(min_widths) + (num_cols * 3) + 1
    
    if border_width <= max_total_width:
        # All columns fit comfortably
        return min_widths
    
    # Need to truncate - distribute available width fairly
    available_width = max_total_width - (num_cols * 3) - 1  # Account for borders
    if available_width < num_cols * 3:  # Not enough space even for minimum
        return [3] * num_cols
    
    # Distribute width proportionally but ensure minimums
    total_min = sum(min_widths)
    scale_factor = available_width / total_min
    
    scaled_widths = []
    for min_width in min_widths:
        scaled_width = max(3, int(min_width * scale_factor))
        scaled_widths.append(scaled_width)
    
    # Adjust if we went over/under
    total_scaled = sum(scaled_widths)
    if total_scaled != available_width:
        diff = available_width - total_scaled
        # Distribute the difference across columns
        for i in range(abs(diff)):
            col_idx = i % num_cols
            if diff > 0:
                scaled_widths[col_idx] += 1
            elif scaled_widths[col_idx] > 3:  # Don't go below minimum
                scaled_widths[col_idx] -= 1
    
    return scaled_widths

def smart_matrix_truncation(matrix_data: List[List[Any]], max_rows: int, max_cols: int) -> Tuple[List[List[Any]], bool, bool]:
    """
    Smart truncation for matrices showing corners and edges.
    
    Returns (truncated_matrix, rows_truncated, cols_truncated)
    """
    rows_truncated = False
    cols_truncated = False
    
    # First truncate rows
    if len(matrix_data) > max_rows:
        rows_truncated = True
        if max_rows < 3:
            matrix_data = matrix_data[:max_rows]
        else:
            show_first_rows = max_rows // 2
            show_last_rows = max_rows - show_first_rows - 1
            
            ellipsis_row = [Symbols.TRUNCATION_INDICATOR] * len(matrix_data[0]) if matrix_data else []
            matrix_data = (
                matrix_data[:show_first_rows] + 
                [ellipsis_row] +
                (matrix_data[-show_last_rows:] if show_last_rows > 0 else [])
            )
    
    # Then truncate columns
    if matrix_data and len(matrix_data[0]) > max_cols:
        cols_truncated = True
        if max_cols < 3:
            matrix_data = [row[:max_cols] for row in matrix_data]
        else:
            show_first_cols = max_cols // 2
            show_last_cols = max_cols - show_first_cols - 1
            
            truncated_matrix = []
            for row in matrix_data:
                truncated_row = (
                    row[:show_first_cols] + 
                    [Symbols.TRUNCATION_INDICATOR] +
                    (row[-show_last_cols:] if show_last_cols > 0 else [])
                )
                truncated_matrix.append(truncated_row)
            matrix_data = truncated_matrix
    
    return matrix_data, rows_truncated, cols_truncated
