"""Python-only scalar spelling and defaults for native dictionary render adapters.

No graph, numerical, truncation or layout algorithm lives here. Shared helpers
are imported once, avoiding a package initialization cycle and per-cell imports.
"""

from math import isfinite as _isfinite
from operator import index as _index

from .._groggy import _display_clip as _clip
from .unicode_chars import Symbols as _Symbols


_DISPLAY_OVERRIDES = {}


def _resolve_option(name, value, historical_default):
    value = _DISPLAY_OVERRIDES.get(name, historical_default) if value is None else value
    value = _index(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _python_scalar(value, presentation, precision, dtype):
    """Retain Python repr/format semantics at the language presentation boundary."""
    if value is None or isinstance(value, float) and not _isfinite(value):
        return _Symbols.NULL_DISPLAY
    if presentation == "array":
        if isinstance(value, float):
            return f"{value:.{precision}g}"
        if isinstance(value, str):
            return _clip(value, 15, _Symbols.ELLIPSIS)
    elif presentation == "matrix":
        if isinstance(value, float):
            return f"{value:.{precision}f}"
    else:
        if dtype in ("float64", "float32", "float") and isinstance(value, (int, float)):
            return f"{float(value):.{precision}f}"
        if isinstance(value, str):
            return _clip(
                value, 10 if dtype in ("datetime", "date") else 12, _Symbols.ELLIPSIS
            )
    return str(value)
