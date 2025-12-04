"""
GroggyGraphWidget - Interactive Jupyter Widget for Graph Visualization

Provides full bidirectional communication between Python and JavaScript with
drag-and-drop nodes, real-time layout switching, hover effects, click callbacks,
and synchronized state management.
"""

try:
    import ipywidgets as widgets
    from traitlets import (Bool, Dict, Float, Int, List, Unicode, observe,
                           validate)

    IPYWIDGETS_AVAILABLE = True
except ImportError:
    IPYWIDGETS_AVAILABLE = False

    # Create mock classes for when ipywidgets is not available
    class widgets:
        class DOMWidget:
            pass

    def observe(func):
        return func

    def validate(func):
        return func

    class Unicode:
        def tag(self, **kwargs):
            return self

    class Dict:
        def tag(self, **kwargs):
            return self

    class List:
        def tag(self, **kwargs):
            return self

    class Int:
        def tag(self, **kwargs):
            return self

    class Bool:
        def tag(self, **kwargs):
            return self

    class Float:
        def tag(self, **kwargs):
            return self


import json
from typing import Any, Callable
from typing import Dict as TypeDict
from typing import List as TypeList
from typing import Optional

# Import widget loader for automatic JavaScript loading
from .widget_loader import auto_load_widget


class GroggyGraphWidget(widgets.DOMWidget):
    """
    Interactive graph widget with full Python-JavaScript synchronization.

    Provides drag-and-drop nodes, real-time layout switching, hover effects,
    click callbacks, and synchronized camera state for rich graph exploration.

    Examples:
        >>> # Basic usage
        >>> widget = GroggyGraphWidget(graph_data)
        >>> widget.on_node_click(lambda node_id, data: print(f"Clicked: {node_id}"))
        >>> display(widget)

        >>> # Advanced configuration
        >>> widget = GroggyGraphWidget(
        ...     graph_data,
        ...     width=1000,
        ...     height=600,
        ...     layout_algorithm='circular',
        ...     theme='dark',
        ...     enable_animations=True
        ... )
    """

    if not IPYWIDGETS_AVAILABLE:

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "ipywidgets not available. Install with: pip install ipywidgets\n"
                "Then enable in Jupyter: jupyter nbextension enable --py --sys-prefix ipywidgets"
            )

    # Widget metadata for Jupyter communication - point to our actual module
    _view_name = Unicode("GroggyGraphView").tag(sync=True)
    _model_name = Unicode("GroggyGraphModel").tag(sync=True)
    _view_module = Unicode("groggy-widgets").tag(sync=True)  # Our actual module name
    _view_module_version = Unicode("^0.1.0").tag(sync=True)  # Back to proper semver
    _model_module = Unicode("groggy-widgets").tag(sync=True)  # Our actual module name
    _model_module_version = Unicode("^0.1.0").tag(sync=True)  # Back to proper semver

    # Graph data synchronized with JavaScript
    nodes = List().tag(sync=True)
    edges = List().tag(sync=True)
    node_positions = Dict().tag(sync=True)

    # Visualization configuration
    layout_algorithm = Unicode("force-directed").tag(sync=True)
    theme = Unicode("light").tag(sync=True)
    width = Int(800).tag(sync=True)
    height = Int(600).tag(sync=True)
    title = Unicode("Graph Visualization").tag(sync=True)

    # Interactive state synchronized with JavaScript
    selected_nodes = List().tag(sync=True)
    hovered_node = Unicode("").tag(sync=True)
    camera_position = Dict({"x": 0.0, "y": 0.0, "zoom": 1.0}).tag(sync=True)
    is_dragging = Bool(False).tag(sync=True)

    # Feature configuration
    enable_drag = Bool(True).tag(sync=True)
    enable_pan = Bool(True).tag(sync=True)
    enable_zoom = Bool(True).tag(sync=True)
    enable_animations = Bool(True).tag(sync=True)
    animation_duration = Int(300).tag(sync=True)

    # Style configuration (will be populated by style system)
    style_config = Dict().tag(sync=True)

    def __init__(self, data_source=None, style_system=None, **kwargs):
        """
        Initialize the interactive graph widget.

        Args:
            data_source: Graph data source (Graph, Subgraph, etc.)
            style_system: GroggyStyleSystem instance for styling
            **kwargs: Additional widget configuration
        """
        if not IPYWIDGETS_AVAILABLE:
            raise ImportError("ipywidgets not available")

        # Initialize attributes first (before ipywidgets init which may trigger observers)
        self.data_source = data_source
        self.style_system = style_system

        # Callback registry for Python event handlers (must be set before super init)
        self._callbacks = {
            "node_click": [],
            "node_hover": [],
            "node_double_click": [],
            "layout_change": [],
            "camera_change": [],
            "selection_change": [],
            "drag_start": [],
            "drag_end": [],
        }

        # Ensure JavaScript widget is loaded before initializing
        auto_load_widget()

        # Initialize ipywidgets after setting up callbacks
        super().__init__(**kwargs)

        # Initialize widget data
        if data_source is not None:
            self._extract_and_sync_data()

        if style_system is not None:
            self._apply_style_system()

        # Set up custom message handling
        self.on_msg(self._handle_custom_message)

    def _extract_and_sync_data(self):
        """Extract data from the data source and sync to JavaScript."""
        if self.data_source is None:
            return

        try:
            # Extract nodes
            nodes_data = []
            if hasattr(self.data_source, "node_ids"):
                for node_id in self.data_source.node_ids:
                    attrs = {}
                    try:
                        if hasattr(self.data_source, "get_node_attr"):
                            attrs = {
                                "label": self.data_source.get_node_attr(
                                    node_id, "label"
                                ),
                                "color": self.data_source.get_node_attr(
                                    node_id, "color"
                                ),
                                "size": self.data_source.get_node_attr(node_id, "size"),
                                "type": self.data_source.get_node_attr(node_id, "type"),
                            }
                            # Remove None values
                            attrs = {k: v for k, v in attrs.items() if v is not None}
                    except:
                        pass

                    nodes_data.append(
                        {
                            "id": str(node_id),
                            "label": attrs.get("label", f"Node {node_id}"),
                            "color": attrs.get("color"),
                            "size": attrs.get("size", 8),
                            "type": attrs.get("type", "default"),
                            **attrs,
                        }
                    )

            # Extract edges
            edges_data = []
            if hasattr(self.data_source, "edge_ids"):
                for edge_id in self.data_source.edge_ids:
                    try:
                        src, dst = self.data_source.edge_endpoints(edge_id)
                        attrs = {}
                        try:
                            if hasattr(self.data_source, "get_edge_attr"):
                                attrs = {
                                    "weight": self.data_source.get_edge_attr(
                                        edge_id, "weight"
                                    ),
                                    "label": self.data_source.get_edge_attr(
                                        edge_id, "label"
                                    ),
                                    "type": self.data_source.get_edge_attr(
                                        edge_id, "type"
                                    ),
                                }
                                attrs = {
                                    k: v for k, v in attrs.items() if v is not None
                                }
                        except:
                            pass

                        edges_data.append(
                            {
                                "id": str(edge_id),
                                "source": str(src),
                                "target": str(dst),
                                "weight": attrs.get("weight", 1.0),
                                "label": attrs.get("label"),
                                "type": attrs.get("type", "default"),
                                **attrs,
                            }
                        )
                    except Exception as e:
                        print(f"Warning: Failed to process edge {edge_id}: {e}")
                        continue

            # Update synchronized data
            self.nodes = nodes_data
            self.edges = edges_data

            # Initialize positions (will be calculated by JavaScript)
            self.node_positions = {}

        except Exception as e:
            print(f"Warning: Failed to extract graph data: {e}")
            # Fallback to demo data
            self._set_demo_data()

    def _set_demo_data(self):
        """Set demo data for testing when data extraction fails."""
        self.nodes = [
            {"id": "A", "label": "Node A", "color": "#ff6b6b", "size": 10},
            {"id": "B", "label": "Node B", "color": "#4ecdc4", "size": 10},
            {"id": "C", "label": "Node C", "color": "#45b7d1", "size": 10},
        ]
        self.edges = [
            {"id": "e1", "source": "A", "target": "B", "weight": 1.0},
            {"id": "e2", "source": "B", "target": "C", "weight": 1.0},
        ]
        self.node_positions = {}

    def _apply_style_system(self):
        """Apply style system configuration to widget."""
        if self.style_system is None:
            return

        try:
            # Get style configuration for JavaScript
            style_config = self.style_system.to_js_config()
            self.style_config = style_config

            # Apply theme
            self.theme = self.style_system.theme.value

        except Exception as e:
            print(f"Warning: Failed to apply style system: {e}")

    # Callback registration methods
    def on_node_click(self, callback: Callable[[str, TypeDict], None]):
        """
        Register callback for node click events.

        Args:
            callback: Function(node_id: str, node_data: dict) -> None
        """
        self._callbacks["node_click"].append(callback)

    def on_node_hover(self, callback: Callable[[Optional[str]], None]):
        """
        Register callback for node hover events.

        Args:
            callback: Function(node_id: str | None) -> None
        """
        self._callbacks["node_hover"].append(callback)

    def on_node_double_click(self, callback: Callable[[str, TypeDict], None]):
        """
        Register callback for node double-click events.

        Args:
            callback: Function(node_id: str, node_data: dict) -> None
        """
        self._callbacks["node_double_click"].append(callback)

    def on_layout_change(self, callback: Callable[[str], None]):
        """
        Register callback for layout algorithm changes.

        Args:
            callback: Function(layout_name: str) -> None
        """
        self._callbacks["layout_change"].append(callback)

    def on_selection_change(self, callback: Callable[[TypeList[str]], None]):
        """
        Register callback for node selection changes.

        Args:
            callback: Function(selected_node_ids: List[str]) -> None
        """
        self._callbacks["selection_change"].append(callback)

    def on_camera_change(self, callback: Callable[[TypeDict], None]):
        """
        Register callback for camera position/zoom changes.

        Args:
            callback: Function(camera_state: dict) -> None
        """
        self._callbacks["camera_change"].append(callback)

    # State observation methods (triggered by JavaScript changes)
    @observe("selected_nodes")
    def _on_selection_change(self, change):
        """Handle selection changes from JavaScript."""
        for callback in self._callbacks["selection_change"]:
            try:
                callback(change["new"])
            except Exception as e:
                print(f"Error in selection change callback: {e}")

    @observe("hovered_node")
    def _on_hover_change(self, change):
        """Handle hover changes from JavaScript."""
        node_id = change["new"] if change["new"] else None
        for callback in self._callbacks["node_hover"]:
            try:
                callback(node_id)
            except Exception as e:
                print(f"Error in hover callback: {e}")

    @observe("layout_algorithm")
    def _on_layout_change(self, change):
        """Handle layout changes from JavaScript."""
        for callback in self._callbacks["layout_change"]:
            try:
                callback(change["new"])
            except Exception as e:
                print(f"Error in layout change callback: {e}")

    @observe("camera_position")
    def _on_camera_change(self, change):
        """Handle camera changes from JavaScript."""
        for callback in self._callbacks["camera_change"]:
            try:
                callback(change["new"])
            except Exception as e:
                print(f"Error in camera change callback: {e}")

    def _handle_custom_message(self, _, content, buffers):
        """Handle custom messages from JavaScript."""
        msg_type = content.get("type")

        if msg_type == "node_click":
            node_id = content.get("node_id")
            node_data = content.get("node_data", {})
            for callback in self._callbacks["node_click"]:
                try:
                    callback(node_id, node_data)
                except Exception as e:
                    print(f"Error in node click callback: {e}")

        elif msg_type == "node_double_click":
            node_id = content.get("node_id")
            node_data = content.get("node_data", {})
            for callback in self._callbacks["node_double_click"]:
                try:
                    callback(node_id, node_data)
                except Exception as e:
                    print(f"Error in node double-click callback: {e}")

        elif msg_type == "drag_start":
            for callback in self._callbacks["drag_start"]:
                try:
                    callback(content)
                except Exception as e:
                    print(f"Error in drag start callback: {e}")

        elif msg_type == "drag_end":
            for callback in self._callbacks["drag_end"]:
                try:
                    callback(content)
                except Exception as e:
                    print(f"Error in drag end callback: {e}")

    # Public API methods for programmatic control
    def set_layout(self, algorithm: str, animate: bool = True):
        """
        Change layout algorithm with optional animation.

        Args:
            algorithm: Layout algorithm ('force-directed', 'circular', 'grid')
            animate: Whether to animate the transition
        """
        self.send({"type": "set_layout", "algorithm": algorithm, "animate": animate})
        self.layout_algorithm = algorithm

    def select_nodes(self, node_ids: TypeList[str], clear_existing: bool = True):
        """
        Programmatically select nodes.

        Args:
            node_ids: List of node IDs to select
            clear_existing: Whether to clear existing selection
        """
        if clear_existing:
            self.selected_nodes = node_ids
        else:
            # Add to existing selection
            current = set(self.selected_nodes)
            current.update(node_ids)
            self.selected_nodes = list(current)

    def clear_selection(self):
        """Clear all selected nodes."""
        self.selected_nodes = []

    def focus_on_node(self, node_id: str, zoom_level: float = 2.0):
        """
        Focus camera on specific node.

        Args:
            node_id: Node ID to focus on
            zoom_level: Zoom level for focus
        """
        self.send({"type": "focus_node", "node_id": node_id, "zoom": zoom_level})

    def reset_camera(self):
        """Reset camera to default position and zoom."""
        self.send({"type": "reset_camera"})
        self.camera_position = {"x": 0.0, "y": 0.0, "zoom": 1.0}

    def set_theme(self, theme: str):
        """
        Change the visualization theme.

        Args:
            theme: Theme name ('light', 'dark', 'publication', 'minimal')
        """
        self.theme = theme
        if self.style_system:
            self.style_system.set_theme(theme)
            self._apply_style_system()

    def export_positions(self) -> TypeDict[str, TypeDict[str, float]]:
        """
        Export current node positions.

        Returns:
            Dictionary mapping node IDs to {x, y} positions
        """
        return dict(self.node_positions)

    def import_positions(self, positions: TypeDict[str, TypeDict[str, float]]):
        """
        Import node positions.

        Args:
            positions: Dictionary mapping node IDs to {x, y} positions
        """
        self.node_positions = positions
        self.send({"type": "update_positions", "positions": positions})

    def get_graph_stats(self) -> TypeDict[str, Any]:
        """
        Get basic statistics about the graph.

        Returns:
            Dictionary with node count, edge count, etc.
        """
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "selected_count": len(self.selected_nodes),
            "layout_algorithm": self.layout_algorithm,
            "theme": self.theme,
            "camera_position": dict(self.camera_position),
            "widget_size": {"width": self.width, "height": self.height},
        }

    def __repr__(self):
        """String representation of the widget."""
        stats = self.get_graph_stats()
        return (
            f"GroggyGraphWidget(nodes={stats['node_count']}, "
            f"edges={stats['edge_count']}, layout={stats['layout_algorithm']}, "
            f"theme={stats['theme']})"
        )


# For backwards compatibility and when ipywidgets is not available
if not IPYWIDGETS_AVAILABLE:

    class GroggyGraphWidget:
        """Placeholder for when ipywidgets is not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Jupyter widgets not available. Install with:\n"
                "  pip install ipywidgets\n"
                "Then enable in Jupyter:\n"
                "  jupyter nbextension enable --py --sys-prefix ipywidgets"
            )
