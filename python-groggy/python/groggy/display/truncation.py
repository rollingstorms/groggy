"""Compatibility projections over native display selection and width plans.

Python objects keep their identity; only indices and formatted text cross FFI.
"""

from typing import Any, List, Tuple

from .._groggy import _display_clip, _display_positions, _display_widths
from .unicode_chars import Symbols


def truncate_rows(data: List[Any], max_rows: int) -> Tuple[List[Any], bool]:
    positions = _display_positions(len(data), max_rows)
    if len(data) <= max_rows:
        return data, False
    marker = create_ellipsis_row(data[0] if data else [])
    return [marker if index is None else data[index] for index in positions], True


def truncate_columns(headers: List[str], data: List[List[Any]], max_cols: int):
    positions = _display_positions(len(headers), max_cols)
    if len(headers) <= max_cols:
        return headers, data, False
    selected = [Symbols.ELLIPSIS if col is None else headers[col] for col in positions]
    rows = [
        [
            Symbols.TRUNCATION_INDICATOR if col is None else row[col]
            for col in positions
            if col is None or col < len(row)
        ]
        for row in data
    ]
    return selected, rows, True


def create_ellipsis_row(sample_row: Any) -> Any:
    """Project a native omission marker into the caller's Python row structure."""
    if not hasattr(sample_row, "__len__") or isinstance(sample_row, str):
        return Symbols.ELLIPSIS
    return [Symbols.ELLIPSIS] * len(sample_row) if sample_row else [Symbols.ELLIPSIS]


def truncate_string(text: str, max_width: int, ellipsis: str = Symbols.ELLIPSIS) -> str:
    return _display_clip(text, max_width, ellipsis)


def calculate_column_widths(
    headers: List[str], data: List[List[Any]], max_total_width: int
):
    return _display_widths(
        headers, [[str(cell) for cell in row] for row in data], max_total_width
    )


def smart_matrix_truncation(matrix_data: List[List[Any]], max_rows: int, max_cols: int):
    row_positions = _display_positions(len(matrix_data), max_rows)
    if not matrix_data:
        return matrix_data, False, False
    rows_truncated = len(matrix_data) > max_rows
    cols_truncated = len(matrix_data[0]) > max_cols
    if not rows_truncated and not cols_truncated:
        return matrix_data, False, False
    if not cols_truncated:
        rows = [
            ([Symbols.TRUNCATION_INDICATOR] * len(matrix_data[0]))
            if index is None
            else matrix_data[index]
            for index in row_positions
        ]
        return rows, rows_truncated, False
    columns = _display_positions(len(matrix_data[0]), max_cols)
    rows = [
        [
            Symbols.TRUNCATION_INDICATOR
            if index is None or col is None
            else matrix_data[index][col]
            for col in columns
            if index is None or col is None or col < len(matrix_data[index])
        ]
        for index in row_positions
    ]
    return rows, rows_truncated, cols_truncated
