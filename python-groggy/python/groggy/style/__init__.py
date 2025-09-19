"""
Groggy Style System

Unified styling system for all visualization backends with theme support,
customization capabilities, and seamless integration with both basic rendering
and advanced Jupyter widgets.
"""

from .style_system import (
    GroggyStyleSystem,
    ThemeType, 
    NodeStyle,
    EdgeStyle,
    CanvasStyle
)

__all__ = [
    'GroggyStyleSystem',
    'ThemeType',
    'NodeStyle', 
    'EdgeStyle',
    'CanvasStyle'
]