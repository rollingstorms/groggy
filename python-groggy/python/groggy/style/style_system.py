"""
Groggy Style System Core Implementation

Provides unified styling across all visualization backends with theme support,
customization capabilities, and conversion to CSS/JavaScript formats.
"""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Optional, Union


class ThemeType(Enum):
    """Available visualization themes."""

    LIGHT = "light"
    DARK = "dark"
    PUBLICATION = "publication"
    MINIMAL = "minimal"
    CUSTOM = "custom"


@dataclass
class NodeStyle:
    """Node styling configuration with all visual properties."""

    default_color: str = "#007bff"
    selected_color: str = "#ff0000"
    hovered_color: str = "#ffa500"
    default_size: float = 8.0
    size_multiplier_hover: float = 1.3
    size_multiplier_selected: float = 1.1
    border_width: float = 1.0
    border_width_selected: float = 3.0
    border_color: str = "#333333"
    border_color_selected: str = "#ff0000"
    shadow_blur: float = 10.0
    shadow_color: str = "rgba(0,0,0,0.3)"
    label_font: str = "12px Arial"
    label_color: str = "#000000"
    label_offset_y: float = -15.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class EdgeStyle:
    """Edge styling configuration with connection visual properties."""

    default_color: str = "#999999"
    selected_color: str = "#ff0000"
    hovered_color: str = "#ffa500"
    default_width: float = 1.0
    selected_width: float = 3.0
    hovered_width: float = 2.0
    opacity: float = 0.8
    opacity_selected: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class CanvasStyle:
    """Canvas and layout styling configuration."""

    background_color: str = "#ffffff"
    border_color: str = "#dddddd"
    border_width: str = "1px"
    padding: int = 20
    container_border_radius: str = "4px"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class GroggyStyleSystem:
    """
    Unified styling system for all visualization backends.

    Provides theme management, customization, and conversion to different
    output formats (CSS, JavaScript) for seamless integration across
    basic rendering and advanced widget systems.
    """

    # Predefined theme configurations
    THEMES = {
        ThemeType.LIGHT: {
            "nodes": NodeStyle(
                default_color="#007bff",
                selected_color="#dc3545",
                hovered_color="#ffc107",
                label_color="#000000",
                border_color="#333333",
                border_color_selected="#dc3545",
            ),
            "edges": EdgeStyle(
                default_color="#999999",
                selected_color="#dc3545",
                hovered_color="#ffc107",
            ),
            "canvas": CanvasStyle(background_color="#ffffff", border_color="#dddddd"),
        },
        ThemeType.DARK: {
            "nodes": NodeStyle(
                default_color="#4CAF50",
                selected_color="#ff6b6b",
                hovered_color="#ffd93d",
                label_color="#ffffff",
                border_color="#ffffff",
                border_color_selected="#ff6b6b",
                shadow_color="rgba(255,255,255,0.2)",
            ),
            "edges": EdgeStyle(
                default_color="#666666",
                selected_color="#ff6b6b",
                hovered_color="#ffd93d",
            ),
            "canvas": CanvasStyle(background_color="#2d2d2d", border_color="#555555"),
        },
        ThemeType.PUBLICATION: {
            "nodes": NodeStyle(
                default_color="#2E3440",
                selected_color="#BF616A",
                hovered_color="#D08770",
                default_size=6.0,
                label_font="10px 'Times New Roman', serif",
                label_color="#2E3440",
                border_width=0.5,
                border_color="#4C566A",
                size_multiplier_hover=1.2,
                shadow_blur=5.0,
            ),
            "edges": EdgeStyle(
                default_color="#4C566A",
                selected_color="#BF616A",
                hovered_color="#D08770",
                default_width=0.8,
                opacity=0.7,
            ),
            "canvas": CanvasStyle(
                background_color="#ECEFF4", border_color="#D8DEE9", padding=30
            ),
        },
        ThemeType.MINIMAL: {
            "nodes": NodeStyle(
                default_color="#000000",
                selected_color="#ff0000",
                hovered_color="#666666",
                default_size=4.0,
                label_font="10px Arial",
                label_color="#000000",
                border_width=0,
                shadow_blur=0,
                size_multiplier_hover=1.5,
            ),
            "edges": EdgeStyle(
                default_color="#cccccc",
                selected_color="#ff0000",
                hovered_color="#666666",
                default_width=0.5,
                opacity=0.6,
            ),
            "canvas": CanvasStyle(
                background_color="#ffffff",
                border_color="transparent",
                border_width="0px",
                padding=10,
            ),
        },
    }

    def __init__(self, theme: Union[ThemeType, str] = ThemeType.LIGHT):
        """
        Initialize style system with a theme.

        Args:
            theme: Theme type or string name
        """
        self.theme = ThemeType(theme) if isinstance(theme, str) else theme
        self._custom_styles = {}
        self._validate_theme()

    def _validate_theme(self):
        """Validate that the current theme exists."""
        if self.theme not in self.THEMES:
            raise ValueError(
                f"Unknown theme: {self.theme}. Available themes: {list(self.THEMES.keys())}"
            )

    def get_node_style(self) -> NodeStyle:
        """
        Get current node styling configuration with custom overrides.

        Returns:
            NodeStyle instance with theme and custom styling applied
        """
        base_style = self.THEMES[self.theme]["nodes"]
        if "nodes" in self._custom_styles:
            # Create new instance with custom overrides
            style_dict = base_style.to_dict()
            style_dict.update(self._custom_styles["nodes"])
            return NodeStyle(**style_dict)
        return base_style

    def get_edge_style(self) -> EdgeStyle:
        """
        Get current edge styling configuration with custom overrides.

        Returns:
            EdgeStyle instance with theme and custom styling applied
        """
        base_style = self.THEMES[self.theme]["edges"]
        if "edges" in self._custom_styles:
            style_dict = base_style.to_dict()
            style_dict.update(self._custom_styles["edges"])
            return EdgeStyle(**style_dict)
        return base_style

    def get_canvas_style(self) -> CanvasStyle:
        """
        Get current canvas styling configuration with custom overrides.

        Returns:
            CanvasStyle instance with theme and custom styling applied
        """
        base_style = self.THEMES[self.theme]["canvas"]
        if "canvas" in self._custom_styles:
            style_dict = base_style.to_dict()
            style_dict.update(self._custom_styles["canvas"])
            return CanvasStyle(**style_dict)
        return base_style

    def customize_nodes(self, **kwargs):
        """
        Customize node styling with specific overrides.

        Args:
            **kwargs: Any NodeStyle field to override

        Examples:
            >>> style.customize_nodes(default_color='#ff0000', default_size=12.0)
        """
        if "nodes" not in self._custom_styles:
            self._custom_styles["nodes"] = {}
        self._custom_styles["nodes"].update(kwargs)

    def customize_edges(self, **kwargs):
        """
        Customize edge styling with specific overrides.

        Args:
            **kwargs: Any EdgeStyle field to override
        """
        if "edges" not in self._custom_styles:
            self._custom_styles["edges"] = {}
        self._custom_styles["edges"].update(kwargs)

    def customize_canvas(self, **kwargs):
        """
        Customize canvas styling with specific overrides.

        Args:
            **kwargs: Any CanvasStyle field to override
        """
        if "canvas" not in self._custom_styles:
            self._custom_styles["canvas"] = {}
        self._custom_styles["canvas"].update(kwargs)

    def set_theme(self, theme: Union[ThemeType, str]):
        """
        Change the current theme, preserving custom overrides.

        Args:
            theme: New theme to apply
        """
        self.theme = ThemeType(theme) if isinstance(theme, str) else theme
        self._validate_theme()

    def clear_customizations(self):
        """Clear all custom style overrides, reverting to pure theme."""
        self._custom_styles = {}

    def to_css_dict(self) -> Dict[str, str]:
        """
        Convert styles to CSS custom properties dictionary.

        Returns:
            Dictionary of CSS custom properties for HTML/CSS integration

        Examples:
            >>> css_vars = style.to_css_dict()
            >>> # {'--groggy-node-color': '#007bff', ...}
        """
        node_style = self.get_node_style()
        edge_style = self.get_edge_style()
        canvas_style = self.get_canvas_style()

        return {
            # Node CSS variables
            "--groggy-node-color": node_style.default_color,
            "--groggy-node-selected-color": node_style.selected_color,
            "--groggy-node-hovered-color": node_style.hovered_color,
            "--groggy-node-size": f"{node_style.default_size}px",
            "--groggy-node-border-width": f"{node_style.border_width}px",
            "--groggy-node-border-color": node_style.border_color,
            "--groggy-node-label-font": node_style.label_font,
            "--groggy-node-label-color": node_style.label_color,
            # Edge CSS variables
            "--groggy-edge-color": edge_style.default_color,
            "--groggy-edge-selected-color": edge_style.selected_color,
            "--groggy-edge-width": f"{edge_style.default_width}px",
            "--groggy-edge-opacity": str(edge_style.opacity),
            # Canvas CSS variables
            "--groggy-canvas-bg": canvas_style.background_color,
            "--groggy-canvas-border": f"{canvas_style.border_width} solid {canvas_style.border_color}",
            "--groggy-canvas-border-radius": canvas_style.container_border_radius,
            "--groggy-canvas-padding": f"{canvas_style.padding}px",
        }

    def to_css_string(self) -> str:
        """
        Convert styles to CSS string for direct HTML embedding.

        Returns:
            CSS string with custom properties
        """
        css_dict = self.to_css_dict()
        css_rules = [f"  {prop}: {value};" for prop, value in css_dict.items()]
        return ":root {\n" + "\n".join(css_rules) + "\n}"

    def to_js_config(self) -> Dict[str, Any]:
        """
        Convert styles to JavaScript configuration object.

        Returns:
            JavaScript-compatible configuration object for widget integration
        """
        return {
            "nodes": self.get_node_style().to_dict(),
            "edges": self.get_edge_style().to_dict(),
            "canvas": self.get_canvas_style().to_dict(),
            "theme": self.theme.value,
            "css_vars": self.to_css_dict(),
        }

    def apply_to_html_template(self, html_template: str) -> str:
        """
        Apply styles to an HTML template by injecting CSS variables.

        Args:
            html_template: HTML template string

        Returns:
            HTML with injected CSS styling
        """
        css_style = self.to_css_string()

        # Insert CSS into template
        if "<head>" in html_template:
            # Insert into existing head
            css_block = f"    <style>\n{css_style}\n    </style>\n"
            html_template = html_template.replace("<head>", f"<head>\n{css_block}")
        elif "<style>" in html_template:
            # Append to existing style block
            html_template = html_template.replace("<style>", f"<style>\n{css_style}\n")
        else:
            # Add style block before any content
            css_block = f"<style>\n{css_style}\n</style>\n"
            html_template = css_block + html_template

        return html_template

    def __repr__(self):
        """String representation of the style system."""
        custom_count = len(self._custom_styles)
        return f"GroggyStyleSystem(theme={self.theme.value}, custom_overrides={custom_count})"
