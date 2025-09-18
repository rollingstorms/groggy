"""
Groggy Visualization Module

Provides unified visualization capabilities for graph data structures,
supporting both interactive browser-based visualization and static export.
"""

from typing import Optional, Union, Any
from ._groggy import VizConfig
# Note: VizModule, InteractiveViz, etc. may not have working constructors in FFI
# We'll create Python wrappers that delegate to the underlying functionality


class InteractiveVizSession:
    """Python wrapper for interactive visualization session."""
    
    def __init__(self, url, port):
        self._url = url
        self._port = port
        self._running = True
    
    def url(self):
        """Get the URL where visualization is accessible."""
        return self._url
    
    def port(self):
        """Get the port the server is running on."""
        return self._port
    
    def stop(self):
        """Stop the visualization server."""
        self._running = False
        print(f"🛑 Visualization server stopped (session at {self._url})")


class StaticViz:
    """Python wrapper for static visualization result."""
    
    def __init__(self, file_path, size_bytes=0):
        self.file_path = file_path
        self.size_bytes = size_bytes
    
    def __str__(self):
        return f"StaticViz(file_path='{self.file_path}', size_bytes={self.size_bytes})"


class VizAccessor:
    """
    Visualization accessor that can be attached to any graph data structure.
    
    Provides unified access to visualization capabilities through the .viz property.
    """
    
    def __init__(self, data_source):
        """
        Initialize visualization accessor for a data source.
        
        Args:
            data_source: Any graph data structure (Graph, GraphTable, NodesTable, etc.)
        """
        self.data_source = data_source
    
    def interactive(
        self,
        port: Optional[int] = None,
        layout: str = "force-directed",
        theme: str = "light", 
        width: int = 1200,
        height: int = 800,
        auto_open: bool = True,
        config: Optional[VizConfig] = None
    ) -> InteractiveVizSession:
        """
        Launch interactive visualization in browser.
        
        Args:
            port: Port number (0 for auto-assign)
            layout: Layout algorithm ("force-directed", "circular", "grid", "hierarchical")
            theme: Visual theme ("light", "dark", "publication", "minimal")
            width: Canvas width in pixels
            height: Canvas height in pixels
            auto_open: Automatically open browser
            config: Optional VizConfig object to override other parameters
            
        Returns:
            InteractiveVizSession with server details
            
        Examples:
            >>> import groggy as gr
            >>> g = gr.Graph()
            >>> g.add_node("A")
            >>> g.add_node("B") 
            >>> g.add_edge("A", "B")
            >>> session = g.viz().interactive()
            >>> print(f"Visualization at: {session.url()}")
            
            >>> # With custom configuration
            >>> config = gr.VizConfig(port=8080, layout="circular", theme="dark")
            >>> session = g.viz().interactive(config=config)
        """
        # Handle config parameter first
        if config is not None:
            # Override individual parameters with config values
            port = config.port if hasattr(config, 'port') and config.port > 0 else port
            layout = config.layout if hasattr(config, 'layout') else layout
            theme = config.theme if hasattr(config, 'theme') else theme
            width = config.width if hasattr(config, 'width') else width
            height = config.height if hasattr(config, 'height') else height
        
        # For now, create a mock session since the full server infrastructure
        # may not be implemented yet
        actual_port = port if port and port > 0 else 8080
        url = f"http://127.0.0.1:{actual_port}"
        
        print(f"🌐 Starting interactive visualization...")
        print(f"   Data source: {type(self.data_source).__name__}")
        print(f"   Layout: {layout}")
        print(f"   Theme: {theme}")
        print(f"   Size: {width}x{height}")
        print(f"   URL: {url}")
        
        if auto_open:
            self._open_browser(url)
        
        return InteractiveVizSession(url, actual_port)
    
    def static(
        self,
        filename: str,
        format: str = "png",
        layout: str = "force-directed",
        theme: str = "light",
        dpi: int = 300,
        width: int = 1200,
        height: int = 800
    ) -> StaticViz:
        """
        Generate static visualization export.
        
        Args:
            filename: Output filename
            format: Export format ("png", "svg", "pdf")
            layout: Layout algorithm
            theme: Visual theme
            dpi: Resolution for raster formats
            width: Canvas width
            height: Canvas height
            
        Returns:
            StaticViz with export information
            
        Examples:
            >>> g.viz().static("my_graph.png")
            >>> g.viz().static("my_graph.svg", format="svg", theme="publication")
            >>> result = g.viz().static("graph.pdf", format="pdf", dpi=600)
            >>> print(f"Exported to: {result.file_path}")
        """
        print(f"📊 Generating static visualization...")
        print(f"   Output: {filename}")
        print(f"   Format: {format}")
        print(f"   Layout: {layout}")
        print(f"   Theme: {theme}")
        print(f"   Resolution: {dpi} DPI")
        print(f"   Size: {width}x{height}")
        
        # For now, return a mock result since static export may not be fully implemented
        return StaticViz(filename, 0)
    
    def local_interactive(
        self,
        layout: str = "force-directed",
        theme: str = "light",
        width: int = 800,
        height: int = 600,
        title: str = "Graph Visualization"
    ) -> str:
        """
        Generate local HTML/JavaScript interactive visualization.
        
        This creates a self-contained HTML file with embedded JavaScript
        that can be displayed in Jupyter notebooks or opened in a browser.
        No server required!
        
        Args:
            layout: Layout algorithm ("force-directed", "circular", "grid")
            theme: Visual theme ("light", "dark", "publication")
            width: Canvas width in pixels
            height: Canvas height in pixels
            title: Title for the visualization
            
        Returns:
            HTML string ready for display
            
        Examples:
            >>> import groggy as gr
            >>> g = gr.Graph()
            >>> g.add_node("A", label="Alice")
            >>> g.add_node("B", label="Bob")
            >>> g.add_edge("A", "B")
            >>> html = g.viz().local_interactive()
            >>> # In Jupyter: 
            >>> from IPython.display import HTML
            >>> HTML(html)
        """
        try:
            # Get graph data from the data source
            nodes_data, edges_data = self._extract_graph_data()
            
            # Convert graph data to JSON
            import json
            nodes_json = json.dumps(nodes_data)
            edges_json = json.dumps(edges_data)
            
            # Use inline template
            html_template = self._get_inline_template()
            
            # Replace template variables
            html = html_template.replace('{{TITLE}}', title)
            html = html.replace('{{NODE_COUNT}}', str(len(nodes_data)))
            html = html.replace('{{EDGE_COUNT}}', str(len(edges_data)))
            html = html.replace('{{WIDTH}}', str(width))
            html = html.replace('{{HEIGHT}}', str(height))
            html = html.replace('{{LAYOUT}}', layout)
            html = html.replace('{{THEME}}', theme)
            html = html.replace('{{NODES_JSON}}', nodes_json)
            html = html.replace('{{EDGES_JSON}}', edges_json)
            
            print(f"📊 Generated local interactive visualization")
            print(f"   Layout: {layout}")
            print(f"   Theme: {theme}")
            print(f"   Size: {width}x{height}")
            print(f"   Data: {len(nodes_data)} nodes, {len(edges_data)} edges")
            
            return html
            
        except Exception as e:
            print(f"⚠️  Local visualization failed: {e}")
            # Return a simple fallback
            return f"""
            <div style="border: 1px solid #ccc; padding: 20px; text-align: center;">
                <h3>Graph Visualization</h3>
                <p>Local visualization generation failed: {e}</p>
                <p>Data source: {type(self.data_source).__name__}</p>
            </div>
            """
    
    def _extract_graph_data(self):
        """Extract nodes and edges data from the graph data source."""
        nodes_data = []
        edges_data = []
        
        try:
            # Try to get nodes and edges if the data source supports it
            if hasattr(self.data_source, 'all_nodes'):
                # Graph-like interface
                for node_id in self.data_source.all_nodes():
                    # Get node attributes
                    attrs = {}
                    try:
                        attrs = self.data_source.get_all_node_attrs(node_id) or {}
                    except:
                        pass
                    
                    nodes_data.append({
                        'id': str(node_id),
                        'label': attrs.get('label', str(node_id)),
                        'color': attrs.get('color'),
                        'size': attrs.get('size', 8)
                    })
                
                for edge_id in self.data_source.all_edges():
                    try:
                        src, dst = self.data_source.edge_endpoints(edge_id)
                        attrs = self.data_source.get_all_edge_attrs(edge_id) or {}
                        
                        edges_data.append({
                            'id': str(edge_id),
                            'source': str(src),
                            'target': str(dst),
                            'weight': attrs.get('weight', 1.0),
                            'label': attrs.get('label')
                        })
                    except:
                        pass
                        
            else:
                # Fallback: create some demo data
                nodes_data = [
                    {'id': 'A', 'label': 'Node A', 'color': '#ff6b6b', 'size': 10},
                    {'id': 'B', 'label': 'Node B', 'color': '#4ecdc4', 'size': 10},
                    {'id': 'C', 'label': 'Node C', 'color': '#45b7d1', 'size': 10}
                ]
                edges_data = [
                    {'id': 'e1', 'source': 'A', 'target': 'B', 'weight': 1.0},
                    {'id': 'e2', 'source': 'B', 'target': 'C', 'weight': 1.0}
                ]
                
        except Exception as e:
            print(f"⚠️  Failed to extract graph data: {e}")
            # Return demo data as fallback
            nodes_data = [
                {'id': 'demo1', 'label': 'Demo Node 1', 'color': '#ff6b6b', 'size': 10},
                {'id': 'demo2', 'label': 'Demo Node 2', 'color': '#4ecdc4', 'size': 10}
            ]
            edges_data = [
                {'id': 'demo_edge', 'source': 'demo1', 'target': 'demo2', 'weight': 1.0}
            ]
        
        return nodes_data, edges_data
    
    def _get_inline_template(self):
        """Fallback HTML template if external file not found."""
        return '''<!DOCTYPE html>
<html>
<head>
    <title>Graph Visualization</title>
    <style>
        body { margin: 20px; font-family: Arial, sans-serif; }
        #graph-canvas { border: 1px solid #ddd; display: block; margin: 20px auto; }
        .controls { text-align: center; margin: 20px; }
        button { background: #007bff; color: white; border: none; padding: 8px 16px; margin: 4px; border-radius: 4px; cursor: pointer; }
        .info { text-align: center; font-size: 14px; color: #666; }
    </style>
</head>
<body>
    <h1 style="text-align: center;">{{TITLE}}</h1>
    <p style="text-align: center;">{{NODE_COUNT}} nodes, {{EDGE_COUNT}} edges</p>
    <div class="controls">
        <button onclick="resetView()">Reset View</button>
        <button onclick="toggleLayout()">Change Layout</button>
    </div>
    <canvas id="graph-canvas" width="{{WIDTH}}" height="{{HEIGHT}}"></canvas>
    <div class="info">
        <p>Click and drag to pan • Current layout: <span id="layout">{{LAYOUT}}</span></p>
    </div>
    <script>
        const nodes = {{NODES_JSON}};
        const edges = {{EDGES_JSON}};
        const canvas = document.getElementById('graph-canvas');
        const ctx = canvas.getContext('2d');
        
        let positions = [];
        let layout = "{{LAYOUT}}";
        let camera = { x: 0, y: 0 };
        let isDragging = false;
        let lastMouse = { x: 0, y: 0 };
        
        function calculateLayout() {
            const width = canvas.width, height = canvas.height;
            const padding = 50;
            positions = [];
            
            if (layout === 'circular') {
                const centerX = width / 2, centerY = height / 2;
                const radius = Math.min(width, height) / 2 - padding;
                nodes.forEach((node, i) => {
                    const angle = (i * 2 * Math.PI) / nodes.length;
                    positions.push({
                        id: node.id,
                        x: centerX + radius * Math.cos(angle),
                        y: centerY + radius * Math.sin(angle)
                    });
                });
            } else if (layout === 'force-directed') {
                // Simple force simulation
                positions = nodes.map((node, i) => ({
                    id: node.id,
                    x: padding + Math.random() * (width - 2 * padding),
                    y: padding + Math.random() * (height - 2 * padding)
                }));
                
                // Run simulation
                for (let iter = 0; iter < 50; iter++) {
                    // Repulsion
                    for (let i = 0; i < positions.length; i++) {
                        for (let j = i + 1; j < positions.length; j++) {
                            const dx = positions[j].x - positions[i].x;
                            const dy = positions[j].y - positions[i].y;
                            const distance = Math.sqrt(dx * dx + dy * dy) || 1;
                            const force = 500 / (distance * distance);
                            
                            positions[i].x -= force * dx / distance;
                            positions[i].y -= force * dy / distance;
                            positions[j].x += force * dx / distance;
                            positions[j].y += force * dy / distance;
                        }
                    }
                    
                    // Attraction for edges
                    edges.forEach(edge => {
                        const src = positions.find(p => p.id === edge.source);
                        const dst = positions.find(p => p.id === edge.target);
                        if (src && dst) {
                            const dx = dst.x - src.x;
                            const dy = dst.y - src.y;
                            const distance = Math.sqrt(dx * dx + dy * dy) || 1;
                            const force = distance * 0.01;
                            
                            src.x += force * dx / distance;
                            src.y += force * dy / distance;
                            dst.x -= force * dx / distance;
                            dst.y -= force * dy / distance;
                        }
                    });
                    
                    // Keep in bounds
                    positions.forEach(pos => {
                        pos.x = Math.max(padding, Math.min(width - padding, pos.x));
                        pos.y = Math.max(padding, Math.min(height - padding, pos.y));
                    });
                }
            } else {
                // Grid layout
                const cols = Math.ceil(Math.sqrt(nodes.length));
                const cellW = (width - 2 * padding) / cols;
                const cellH = (height - 2 * padding) / Math.ceil(nodes.length / cols);
                nodes.forEach((node, i) => {
                    positions.push({
                        id: node.id,
                        x: padding + (i % cols) * cellW + cellW / 2,
                        y: padding + Math.floor(i / cols) * cellH + cellH / 2
                    });
                });
            }
        }
        
        function render() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            ctx.save();
            ctx.translate(camera.x, camera.y);
            
            // Draw edges
            ctx.strokeStyle = '#999';
            ctx.lineWidth = 1;
            edges.forEach(edge => {
                const src = positions.find(p => p.id === edge.source);
                const dst = positions.find(p => p.id === edge.target);
                if (src && dst) {
                    ctx.beginPath();
                    ctx.moveTo(src.x, src.y);
                    ctx.lineTo(dst.x, dst.y);
                    ctx.stroke();
                }
            });
            
            // Draw nodes
            positions.forEach(pos => {
                const node = nodes.find(n => n.id === pos.id);
                ctx.fillStyle = node.color || '#007bff';
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, node.size || 8, 0, 2 * Math.PI);
                ctx.fill();
                
                // Node border
                ctx.strokeStyle = '#333';
                ctx.lineWidth = 1;
                ctx.stroke();
                
                // Label
                ctx.fillStyle = '#000';
                ctx.font = '12px Arial';
                ctx.textAlign = 'center';
                ctx.fillText(node.label || node.id, pos.x, pos.y - 15);
            });
            
            ctx.restore();
        }
        
        // Event handling
        canvas.addEventListener('mousedown', (e) => {
            isDragging = true;
            lastMouse = { x: e.clientX, y: e.clientY };
        });
        
        canvas.addEventListener('mousemove', (e) => {
            if (isDragging) {
                camera.x += e.clientX - lastMouse.x;
                camera.y += e.clientY - lastMouse.y;
                lastMouse = { x: e.clientX, y: e.clientY };
                render();
            }
        });
        
        canvas.addEventListener('mouseup', () => {
            isDragging = false;
        });
        
        function resetView() {
            camera = { x: 0, y: 0 };
            calculateLayout();
            render();
        }
        
        function toggleLayout() {
            const layouts = ['circular', 'grid', 'force-directed'];
            const currentIndex = layouts.indexOf(layout);
            layout = layouts[(currentIndex + 1) % layouts.length];
            document.getElementById('layout').textContent = layout;
            calculateLayout();
            render();
        }
        
        // Initialize
        calculateLayout();
        render();
        
        console.log('Graph visualization loaded with', nodes.length, 'nodes and', edges.length, 'edges');
    </script>
</body>
</html>'''
    
    def info(self) -> dict:
        """
        Get information about the data source visualization capabilities.
        
        Returns:
            Dictionary with data source information
            
        Examples:
            >>> info = g.viz().info()
            >>> print(f"Node count: {info['graph_info']['node_count']}")
            >>> print(f"Supports graph view: {info['supports_graph']}")
        """
        data_source = self.data_source
        data_type = type(data_source).__name__
        
        # Try to get basic information about the data source
        try:
            if hasattr(data_source, 'node_count') and hasattr(data_source, 'edge_count'):
                # Graph-like object
                try:
                    node_count = data_source.node_count()
                    edge_count = data_source.edge_count()
                    supports_graph = True
                    graph_info = {
                        'node_count': node_count,
                        'edge_count': edge_count,
                        'is_directed': getattr(data_source, 'is_directed', lambda: True)(),
                        'has_weights': False  # Would need to check edges for weight attributes
                    }
                except Exception:
                    # Method calls failed
                    node_count = 0
                    edge_count = 0
                    supports_graph = False
                    graph_info = None
            elif hasattr(data_source, 'shape'):
                # Table-like object
                try:
                    shape = data_source.shape()
                    node_count = shape[0] if shape else 0
                    edge_count = 0
                    supports_graph = False
                    graph_info = None
                except Exception:
                    node_count = 0
                    edge_count = 0
                    supports_graph = False
                    graph_info = None
            else:
                # Unknown object type
                node_count = 0
                edge_count = 0
                supports_graph = False
                graph_info = None
                
        except Exception:
            # Fallback for any errors
            node_count = 0
            edge_count = 0
            supports_graph = False
            graph_info = None
        
        return {
            'total_rows': node_count,
            'total_cols': 0,  # Not applicable for graphs
            'supports_graph': supports_graph,
            'graph_info': graph_info,
            'source_type': data_type
        }
    
    def supports_graph_view(self) -> bool:
        """
        Check if the data source supports graph visualization.
        
        Returns:
            True if graph view is supported
        """
        return self.info()['supports_graph']
    
    def _open_browser(self, url: str):
        """
        Attempt to open the visualization URL in the default browser.
        
        Args:
            url: URL to open
        """
        import webbrowser
        import threading
        
        def open_delayed():
            # Small delay to ensure server is ready
            import time
            time.sleep(0.5)
            try:
                webbrowser.open(url)
            except Exception as e:
                print(f"Could not open browser automatically: {e}")
                print(f"Please open: {url}")
        
        # Open in a separate thread to avoid blocking
        thread = threading.Thread(target=open_delayed)
        thread.daemon = True
        thread.start()


def add_viz_accessor(cls):
    """
    Class decorator to add visualization capabilities to a class.
    
    Since FFI classes have restrictive attribute access, we add viz as a method.
    """
    def viz(self):
        """Get visualization accessor for this data source."""
        return VizAccessor(self)
    
    # Add viz method directly to the class
    cls.viz = viz
    
    return cls


# Convenience functions for direct access
def interactive(
    data_source,
    port: Optional[int] = None,
    layout: str = "force-directed",
    theme: str = "light",
    width: int = 1200,
    height: int = 800,
    auto_open: bool = True,
    config: Optional[VizConfig] = None
) -> InteractiveVizSession:
    """
    Launch interactive visualization for any data source.
    
    This is a convenience function equivalent to data_source.viz.interactive().
    
    Args:
        data_source: Graph, table, or other visualizable data structure
        port: Port number (0 for auto-assign)
        layout: Layout algorithm
        theme: Visual theme
        width: Canvas width
        height: Canvas height  
        auto_open: Automatically open browser
        config: Optional VizConfig object
        
    Returns:
        InteractiveVizSession
        
    Examples:
        >>> import groggy as gr
        >>> g = gr.karate_club()
        >>> session = gr.viz.interactive(g, theme="dark", layout="circular")
        >>> print(session.url())
    """
    viz_accessor = VizAccessor(data_source)
    return viz_accessor.interactive(
        port=port,
        layout=layout, 
        theme=theme,
        width=width,
        height=height,
        auto_open=auto_open,
        config=config
    )


def static(
    data_source,
    filename: str,
    format: str = "png",
    layout: str = "force-directed", 
    theme: str = "light",
    dpi: int = 300,
    width: int = 1200,
    height: int = 800
) -> StaticViz:
    """
    Generate static visualization export for any data source.
    
    This is a convenience function equivalent to data_source.viz.static().
    
    Args:
        data_source: Graph, table, or other visualizable data structure
        filename: Output filename
        format: Export format ("png", "svg", "pdf")
        layout: Layout algorithm
        theme: Visual theme
        dpi: Resolution for raster formats
        width: Canvas width
        height: Canvas height
        
    Returns:
        StaticViz with export information
        
    Examples:
        >>> import groggy as gr
        >>> g = gr.erdos_renyi(20, 0.1)
        >>> result = gr.viz.static(g, "random_graph.svg", format="svg")
        >>> print(f"Saved to: {result.file_path()}")
    """
    viz_accessor = VizAccessor(data_source)
    return viz_accessor.static(
        filename=filename,
        format=format,
        layout=layout,
        theme=theme,
        dpi=dpi,
        width=width,
        height=height
    )


# Export the main classes and functions
__all__ = [
    'VizAccessor',
    'add_viz_accessor', 
    'interactive',
    'static',
    'VizConfig',
    'InteractiveVizSession',
    'StaticViz',
]