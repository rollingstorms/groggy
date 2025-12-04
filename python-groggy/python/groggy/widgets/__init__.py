"""
Groggy Jupyter Widgets

Interactive visualization widgets for Jupyter notebooks with full bidirectional
communication, drag-and-drop functionality, and real-time Python callbacks.
"""

try:
    from .graph_widget import GroggyGraphWidget

    WIDGETS_AVAILABLE = True
except ImportError:
    # ipywidgets not available - create placeholder
    WIDGETS_AVAILABLE = False

    class GroggyGraphWidget:
        """Placeholder for when ipywidgets is not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Jupyter widgets not available. Install with: pip install ipywidgets\n"
                "Then enable in Jupyter: jupyter nbextension enable --py --sys-prefix ipywidgets"
            )


__all__ = ["GroggyGraphWidget", "WIDGETS_AVAILABLE"]
