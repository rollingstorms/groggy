"""
Groggy Visualization Module

Provides unified visualization capabilities for graph data structures,
supporting both interactive browser-based visualization and static export.

The new unified approach uses a single render() method with backend switching:
- render(backend='jupyter') - Jupyter notebook embedding
- render(backend='streaming') - Interactive WebSocket server
- render(backend='file') - Static file export (HTML/SVG/PNG)
- render(backend='local') - Self-contained HTML
"""

from typing import Optional, Union, Any, Dict, Literal
from enum import Enum
from ._groggy import VizConfig

# Import style system
try:
    from .style import GroggyStyleSystem, ThemeType
    STYLE_SYSTEM_AVAILABLE = True
except ImportError:
    STYLE_SYSTEM_AVAILABLE = False
    class GroggyStyleSystem:
        def __init__(self, *args, **kwargs):
            pass
        def to_js_config(self):
            return {}
        def apply_to_html_template(self, html):
            return html

# Import widgets (optional)
try:
    from .widgets import GroggyGraphWidget, WIDGETS_AVAILABLE
except ImportError:
    WIDGETS_AVAILABLE = False
    class GroggyGraphWidget:
        def __init__(self, *args, **kwargs):
            raise ImportError("Widgets not available")


class VizBackend(Enum):
    """Visualization backend options for unified rendering."""
    JUPYTER = "jupyter"      # Jupyter notebook embedding with optimized display
    STREAMING = "streaming"  # WebSocket interactive server with real-time updates
    FILE = "file"           # Static file export (HTML/SVG/PNG)
    LOCAL = "local"         # Self-contained HTML with embedded data
    
    @classmethod
    def from_string(cls, backend: Union[str, 'VizBackend']) -> 'VizBackend':
        """Convert string or enum to VizBackend, with validation."""
        if isinstance(backend, cls):
            return backend
        if isinstance(backend, str):
            try:
                return cls(backend.lower())
            except ValueError:
                valid_backends = [b.value for b in cls]
                raise ValueError(f"Invalid backend '{backend}'. Valid options: {valid_backends}")
        raise TypeError(f"Backend must be string or VizBackend enum, got {type(backend)}")


class VizTemplate:
    """
    Unified template engine for graph visualization across different backends.
    
    This class extracts graph data once and can render to multiple backends
    using a single HTML template with conditional sections.
    """
    
    def __init__(self, data_source):
        """
        Initialize template with graph data extraction.
        
        Args:
            data_source: Any graph data structure (Graph, Subgraph, etc.)
        """
        self.data_source = data_source
        self._nodes_data = None
        self._edges_data = None
        self._metadata = None
        self._extracted = False
    
    def _extract_data_once(self):
        """Extract graph data once and cache for multiple renders."""
        if self._extracted:
            return
            
        try:
            self._nodes_data, self._edges_data = self._extract_graph_data()
            self._metadata = {
                'node_count': len(self._nodes_data),
                'edge_count': len(self._edges_data),
                'is_directed': self._get_graph_direction(),
                'has_weights': any(edge.get('weight') is not None for edge in self._edges_data),
            }
            self._extracted = True
        except Exception as e:
            print(f"⚠️  Data extraction failed: {e}")
            # Use fallback demo data
            self._nodes_data = [
                {'id': 'demo1', 'label': 'Demo Node 1', 'color': '#ff6b6b', 'size': 10},
                {'id': 'demo2', 'label': 'Demo Node 2', 'color': '#4ecdc4', 'size': 10}
            ]
            self._edges_data = [
                {'id': 'demo_edge', 'source': 'demo1', 'target': 'demo2', 'weight': 1.0}
            ]
            self._metadata = {
                'node_count': 2, 'edge_count': 1, 'is_directed': True, 'has_weights': True
            }
            self._extracted = True
    
    def render(
        self,
        backend: Union[VizBackend, str],
        *,
        # Universal parameters
        layout: str = "force-directed",
        theme: str = "light",
        width: int = 800,
        height: int = 600,
        title: Optional[str] = None,
        # Backend-specific parameters (passed as **kwargs)
        **kwargs
    ) -> Union[None, 'StaticViz', 'InteractiveVizSession', str]:
        """
        Unified rendering method that adapts to different backends.
        
        Args:
            backend: Target backend (jupyter/streaming/file/local)
            layout: Layout algorithm ("force-directed", "circular", "grid")
            theme: Visual theme ("light", "dark", "publication")
            width: Canvas width in pixels
            height: Canvas height in pixels
            title: Custom title for visualization
            **kwargs: Backend-specific parameters
            
        Returns:
            - None for jupyter backend (displays directly)
            - StaticViz for file backend
            - InteractiveVizSession for streaming backend
            - str (HTML) for local backend
            
        Examples:
            >>> g.viz().render(backend='jupyter')
            >>> g.viz().render(backend='file', filename='graph.html')
            >>> session = g.viz().render(backend='streaming', port=8080)
            >>> html = g.viz().render(backend='local')
        """
        # Validate and normalize backend
        backend_enum = VizBackend.from_string(backend)
        
        # Extract data once for this render operation
        self._extract_data_once()
        
        # Set default title
        if title is None:
            title = f"Graph Visualization - {type(self.data_source).__name__}"
        
        # Route to appropriate backend handler
        if backend_enum == VizBackend.JUPYTER:
            return self._render_jupyter(layout, theme, width, height, title, **kwargs)
        elif backend_enum == VizBackend.STREAMING:
            return self._render_streaming(layout, theme, width, height, title, **kwargs)
        elif backend_enum == VizBackend.FILE:
            return self._render_file(layout, theme, width, height, title, **kwargs)
        elif backend_enum == VizBackend.LOCAL:
            return self._render_local(layout, theme, width, height, title, **kwargs)
        else:
            raise ValueError(f"Unsupported backend: {backend_enum}")
    
    def _render_jupyter(self, layout, theme, width, height, title, **kwargs):
        """Render for Jupyter notebook with optimized embedding."""
        try:
            from IPython.display import display, HTML
            
            # Generate unified HTML with Jupyter backend flag
            html_content = self._generate_unified_html(layout, theme, width, height, title, backend='jupyter')
            display(HTML(html_content))
            return None  # Jupyter displays directly
            
        except ImportError:
            print("IPython not available. Use render(backend='file') to save to file instead.")
            print("To enable Jupyter display: pip install ipython")
            return None
        except Exception as e:
            print(f"Jupyter rendering failed: {e}")
            return None
    
    def _render_streaming(self, layout, theme, width, height, title, **kwargs):
        """Render for streaming WebSocket server."""
        port = kwargs.get('port', 8080)
        auto_open = kwargs.get('auto_open', True)
        
        print(f"🌐 Starting interactive visualization server...")
        print(f"   Layout: {layout}, Theme: {theme}")
        print(f"   Size: {width}x{height}")
        print(f"   Port: {port}")
        
        # Generate unified HTML with streaming backend flag
        html_content = self._generate_unified_html(layout, theme, width, height, title, backend='streaming')
        
        # TODO: Start actual streaming server with this HTML content
        # For now, return a mock session
        url = f"http://127.0.0.1:{port}"
        print(f"   URL: {url}")
        
        if auto_open:
            self._open_browser(url)
        
        return InteractiveVizSession(url, port)
    
    def _render_file(self, layout, theme, width, height, title, **kwargs):
        """Render for static file export."""
        filename = kwargs.get('filename', 'graph_visualization.html')
        format_type = kwargs.get('format', 'html')
        
        print(f"📊 Generating static visualization...")
        print(f"   Output: {filename}")
        print(f"   Format: {format_type}")
        
        if format_type.lower() in ['html', 'htm']:
            # Generate unified HTML with file backend flag
            html_content = self._generate_unified_html(layout, theme, width, height, title, backend='file')
            
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                import os
                if os.path.exists(filename):
                    size = os.path.getsize(filename)
                    return StaticViz(filename, size, html_content)
                else:
                    print(f"⚠️  Warning: File {filename} was not created")
                    return StaticViz(filename, 0)
            except Exception as e:
                print(f"⚠️  Error writing file: {e}")
                return StaticViz(filename, 0)
        else:
            print(f"⚠️  Format {format_type} not yet implemented")
            return StaticViz(filename, 0)
    
    def _render_local(self, layout, theme, width, height, title, **kwargs):
        """Render for self-contained HTML."""
        print(f"📊 Generating local interactive visualization...")
        print(f"   Layout: {layout}, Theme: {theme}")
        print(f"   Size: {width}x{height}")
        
        return self._generate_unified_html(layout, theme, width, height, title, backend='local')
    
    def _generate_unified_html(self, layout, theme, width, height, title, backend='local'):
        """Generate unified HTML template that adapts to different backends."""
        import json
        
        nodes_json = json.dumps(self._nodes_data)
        edges_json = json.dumps(self._edges_data)
        canvas_id = f"groggy-viz-{id(self)}"
        
        # Backend-specific wrapper and styling
        if backend == 'jupyter':
            # Jupyter: Simple div wrapper, no full HTML structure
            wrapper_start = f'<div style="width: {width}px; height: {height}px; border: 1px solid #ddd; position: relative; background: #fafafa;">'
            wrapper_end = '</div>'
            include_head = False
            include_controls = False
        else:
            # File/Local/Streaming: Full HTML document
            wrapper_start = ''
            wrapper_end = ''
            include_head = True
            include_controls = True
        
        # Generate unified template
        html_parts = []
        
        # Conditional HEAD section
        if include_head:
            html_parts.append(f'''<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ margin: 20px; font-family: Arial, sans-serif; }}
        #groggy-viz-container {{ border: 1px solid #ddd; position: relative; background: #fafafa; margin: 20px auto; }}
        #groggy-viz-canvas {{ display: block; background: white; }}
        .groggy-controls {{ text-align: center; margin: 20px; }}
        .groggy-controls button {{ background: #007bff; color: white; border: none; padding: 8px 16px; margin: 4px; border-radius: 4px; cursor: pointer; }}
        .groggy-info {{ text-align: center; font-size: 14px; color: #666; }}
        .groggy-stats {{ position: absolute; top: 5px; right: 5px; font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <h1 style="text-align: center;">{title}</h1>
    <p style="text-align: center;">{self._metadata['node_count']} nodes, {self._metadata['edge_count']} edges</p>''')
        
        # Conditional controls
        if include_controls:
            html_parts.append('''    <div class="groggy-controls">
        <button onclick="resetView()">Reset View</button>
        <button onclick="toggleLayout()">Change Layout</button>
    </div>''')
        
        # Main visualization container (always included)
        html_parts.append(f'''{wrapper_start}
    <div id="groggy-viz-container" style="width: {width}px; height: {height}px;{' border: 1px solid #ddd; position: relative; background: #fafafa;' if backend == 'jupyter' else ''}">
        <canvas id="{canvas_id}" width="{width}" height="{height}" 
                style="display: block; background: white;"></canvas>
        <div class="groggy-stats">
            {self._metadata['node_count']} nodes, {self._metadata['edge_count']} edges
        </div>
    </div>''')
        
        # Conditional info section
        if include_controls:
            html_parts.append(f'''    <div class="groggy-info">
        <p>Click and drag to pan • Current layout: <span id="current-layout">{layout}</span></p>
    </div>''')
        
        # Unified JavaScript (always included)
        html_parts.append(f'''    <script>
    (function() {{
        // Unified Groggy Visualization Engine
        // Backend: {backend} | Layout: {layout} | Theme: {theme}
        
        const canvas = document.getElementById('{canvas_id}');
        if (!canvas) {{
            console.error('Groggy viz: Canvas not found:', '{canvas_id}');
            return;
        }}
        
        const ctx = canvas.getContext('2d');
        const width = {width};
        const height = {height};
        const backend = '{backend}';
        const layout = '{layout}';
        const theme = '{theme}';
        
        // Graph data (unified across all backends)
        const nodes = {nodes_json};
        const edges = {edges_json};
        
        console.log(`Groggy viz (${{backend}}): Loading ${{nodes.length}} nodes, ${{edges.length}} edges`);
        
        // Unified state management
        let positions = [];
        let currentLayout = layout;
        let camera = {{ x: 0, y: 0 }};
        let isDragging = false;
        let lastMouse = {{ x: 0, y: 0 }};
        
        // Unified layout algorithms
        function calculateLayout(layoutType = currentLayout) {{
            const padding = 50;
            positions = [];
            
            if (layoutType === 'circular' && nodes.length > 0) {{
                const centerX = width / 2;
                const centerY = height / 2;
                const radius = Math.min(width, height) / 2 - padding;
                
                nodes.forEach((node, i) => {{
                    const angle = (i * 2 * Math.PI) / nodes.length;
                    positions.push({{
                        id: node.id,
                        x: centerX + radius * Math.cos(angle),
                        y: centerY + radius * Math.sin(angle)
                    }});
                }});
            }} else if (layoutType === 'force-directed') {{
                // Initialize random positions
                positions = nodes.map((node, i) => ({{
                    id: node.id,
                    x: padding + Math.random() * (width - 2 * padding),
                    y: padding + Math.random() * (height - 2 * padding)
                }}));
                
                // Force-directed simulation
                for (let iter = 0; iter < (backend === 'jupyter' ? 30 : 50); iter++) {{
                    // Node repulsion
                    for (let i = 0; i < positions.length; i++) {{
                        for (let j = i + 1; j < positions.length; j++) {{
                            const dx = positions[j].x - positions[i].x;
                            const dy = positions[j].y - positions[i].y;
                            const distance = Math.sqrt(dx * dx + dy * dy) || 1;
                            const force = 500 / (distance * distance);
                            
                            positions[i].x -= force * dx / distance;
                            positions[i].y -= force * dy / distance;
                            positions[j].x += force * dx / distance;
                            positions[j].y += force * dy / distance;
                        }}
                    }}
                    
                    // Edge attraction
                    edges.forEach(edge => {{
                        const src = positions.find(p => p.id === edge.source);
                        const dst = positions.find(p => p.id === edge.target);
                        if (src && dst) {{
                            const dx = dst.x - src.x;
                            const dy = dst.y - src.y;
                            const distance = Math.sqrt(dx * dx + dy * dy) || 1;
                            const force = distance * 0.01;
                            
                            src.x += force * dx / distance;
                            src.y += force * dy / distance;
                            dst.x -= force * dx / distance;
                            dst.y -= force * dy / distance;
                        }}
                    }});
                    
                    // Keep in bounds
                    positions.forEach(pos => {{
                        pos.x = Math.max(padding, Math.min(width - padding, pos.x));
                        pos.y = Math.max(padding, Math.min(height - padding, pos.y));
                    }});
                }}
            }} else {{
                // Grid layout (default)
                const cols = Math.ceil(Math.sqrt(nodes.length));
                const cellW = (width - 2 * padding) / cols;
                const cellH = (height - 2 * padding) / Math.ceil(nodes.length / cols);
                
                nodes.forEach((node, i) => {{
                    positions.push({{
                        id: node.id,
                        x: padding + (i % cols) * cellW + cellW / 2,
                        y: padding + Math.floor(i / cols) * cellH + cellH / 2
                    }});
                }});
            }}
        }}
        
        // Unified rendering function
        function render() {{
            ctx.clearRect(0, 0, width, height);
            
            ctx.save();
            if (backend !== 'jupyter') {{
                ctx.translate(camera.x, camera.y);
            }}
            
            // Draw edges
            ctx.strokeStyle = theme === 'dark' ? '#666' : '#999';
            ctx.lineWidth = backend === 'jupyter' ? 2 : 1;
            edges.forEach(edge => {{
                const src = positions.find(p => p.id === edge.source);
                const dst = positions.find(p => p.id === edge.target);
                if (src && dst) {{
                    ctx.beginPath();
                    ctx.moveTo(src.x, src.y);
                    ctx.lineTo(dst.x, dst.y);
                    ctx.stroke();
                }}
            }});
            
            // Draw nodes
            nodes.forEach(node => {{
                const pos = positions.find(p => p.id === node.id);
                if (pos) {{
                    ctx.fillStyle = node.color || (theme === 'dark' ? '#4CAF50' : '#007bff');
                    ctx.beginPath();
                    ctx.arc(pos.x, pos.y, node.size || (backend === 'jupyter' ? 12 : 8), 0, 2 * Math.PI);
                    ctx.fill();
                    
                    ctx.strokeStyle = theme === 'dark' ? '#fff' : '#333';
                    ctx.lineWidth = backend === 'jupyter' ? 2 : 1;
                    ctx.stroke();
                    
                    // Label
                    ctx.fillStyle = theme === 'dark' ? '#fff' : '#000';
                    ctx.font = backend === 'jupyter' ? 'bold 12px Arial' : '12px Arial';
                    ctx.textAlign = 'center';
                    ctx.fillText(node.label || node.id, pos.x, pos.y + (backend === 'jupyter' ? 20 : -15));
                }}
            }});
            
            ctx.restore();
        }}
        
        // Backend-specific event handling
        if (backend === 'jupyter') {{
            // Jupyter: Simple click handling only
            canvas.onclick = function(e) {{
                console.log('Groggy viz: Canvas clicked at', e.offsetX, e.offsetY);
            }};
        }} else {{
            // File/Local/Streaming: Full interaction
            canvas.addEventListener('mousedown', (e) => {{
                isDragging = true;
                lastMouse = {{ x: e.clientX, y: e.clientY }};
            }});
            
            canvas.addEventListener('mousemove', (e) => {{
                if (isDragging) {{
                    camera.x += e.clientX - lastMouse.x;
                    camera.y += e.clientY - lastMouse.y;
                    lastMouse = {{ x: e.clientX, y: e.clientY }};
                    render();
                }}
            }});
            
            canvas.addEventListener('mouseup', () => {{
                isDragging = false;
            }});
            
            // Global functions for controls
            if (typeof window !== 'undefined') {{
                window.resetView = function() {{
                    camera = {{ x: 0, y: 0 }};
                    calculateLayout();
                    render();
                }};
                
                window.toggleLayout = function() {{
                    const layouts = ['circular', 'grid', 'force-directed'];
                    const currentIndex = layouts.indexOf(currentLayout);
                    currentLayout = layouts[(currentIndex + 1) % layouts.length];
                    
                    const layoutSpan = document.getElementById('current-layout');
                    if (layoutSpan) layoutSpan.textContent = currentLayout;
                    
                    calculateLayout(currentLayout);
                    render();
                }};
            }}
        }}
        
        // Initialize visualization
        calculateLayout();
        render();
        
        console.log(`Groggy viz (${{backend}}): Rendered ${{nodes.length}} nodes with ${{currentLayout}} layout`);
    }})();
    </script>''')
        
        # Close wrapper
        html_parts.append(wrapper_end)
        
        # Close HTML document if needed
        if include_head:
            html_parts.append('</body>\n</html>')
        
        return '\n'.join(html_parts)
    
    def _generate_static_html_deprecated(self, layout, theme, width, height, title):
        """Generate full HTML page with embedded graph data."""
        import json
        
        nodes_json = json.dumps(self._nodes_data)
        edges_json = json.dumps(self._edges_data)
        
        return f'''<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ margin: 20px; font-family: Arial, sans-serif; }}
        #graph-canvas {{ border: 1px solid #ddd; display: block; margin: 20px auto; }}
        .controls {{ text-align: center; margin: 20px; }}
        button {{ background: #007bff; color: white; border: none; padding: 8px 16px; margin: 4px; border-radius: 4px; cursor: pointer; }}
        .info {{ text-align: center; font-size: 14px; color: #666; }}
    </style>
</head>
<body>
    <h1 style="text-align: center;">{title}</h1>
    <p style="text-align: center;">{self._metadata['node_count']} nodes, {self._metadata['edge_count']} edges</p>
    <div class="controls">
        <button onclick="resetView()">Reset View</button>
        <button onclick="toggleLayout()">Change Layout</button>
    </div>
    <canvas id="graph-canvas" width="{width}" height="{height}"></canvas>
    <div class="info">
        <p>Click and drag to pan • Current layout: <span id="layout">{layout}</span></p>
    </div>
    <script>
        const nodes = {nodes_json};
        const edges = {edges_json};
        const canvas = document.getElementById('graph-canvas');
        const ctx = canvas.getContext('2d');
        
        let positions = [];
        let layout = "{layout}";
        let camera = {{ x: 0, y: 0 }};
        let isDragging = false;
        let lastMouse = {{ x: 0, y: 0 }};
        
        function calculateLayout() {{
            const width = canvas.width, height = canvas.height;
            const padding = 50;
            positions = [];
            
            if (layout === 'circular') {{
                const centerX = width / 2, centerY = height / 2;
                const radius = Math.min(width, height) / 2 - padding;
                nodes.forEach((node, i) => {{
                    const angle = (i * 2 * Math.PI) / nodes.length;
                    positions.push({{
                        id: node.id,
                        x: centerX + radius * Math.cos(angle),
                        y: centerY + radius * Math.sin(angle)
                    }});
                }});
            }} else if (layout === 'force-directed') {{
                positions = nodes.map((node, i) => ({{
                    id: node.id,
                    x: padding + Math.random() * (width - 2 * padding),
                    y: padding + Math.random() * (height - 2 * padding)
                }}));
                
                for (let iter = 0; iter < 50; iter++) {{
                    for (let i = 0; i < positions.length; i++) {{
                        for (let j = i + 1; j < positions.length; j++) {{
                            const dx = positions[j].x - positions[i].x;
                            const dy = positions[j].y - positions[i].y;
                            const distance = Math.sqrt(dx * dx + dy * dy) || 1;
                            const force = 500 / (distance * distance);
                            
                            positions[i].x -= force * dx / distance;
                            positions[i].y -= force * dy / distance;
                            positions[j].x += force * dx / distance;
                            positions[j].y += force * dy / distance;
                        }}
                    }}
                    
                    edges.forEach(edge => {{
                        const src = positions.find(p => p.id === edge.source);
                        const dst = positions.find(p => p.id === edge.target);
                        if (src && dst) {{
                            const dx = dst.x - src.x;
                            const dy = dst.y - src.y;
                            const distance = Math.sqrt(dx * dx + dy * dy) || 1;
                            const force = distance * 0.01;
                            
                            src.x += force * dx / distance;
                            src.y += force * dy / distance;
                            dst.x -= force * dx / distance;
                            dst.y -= force * dy / distance;
                        }}
                    }});
                    
                    positions.forEach(pos => {{
                        pos.x = Math.max(padding, Math.min(width - padding, pos.x));
                        pos.y = Math.max(padding, Math.min(height - padding, pos.y));
                    }});
                }}
            }} else {{
                const cols = Math.ceil(Math.sqrt(nodes.length));
                const cellW = (width - 2 * padding) / cols;
                const cellH = (height - 2 * padding) / Math.ceil(nodes.length / cols);
                nodes.forEach((node, i) => {{
                    positions.push({{
                        id: node.id,
                        x: padding + (i % cols) * cellW + cellW / 2,
                        y: padding + Math.floor(i / cols) * cellH + cellH / 2
                    }});
                }});
            }}
        }}
        
        function render() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            ctx.save();
            ctx.translate(camera.x, camera.y);
            
            ctx.strokeStyle = '#999';
            ctx.lineWidth = 1;
            edges.forEach(edge => {{
                const src = positions.find(p => p.id === edge.source);
                const dst = positions.find(p => p.id === edge.target);
                if (src && dst) {{
                    ctx.beginPath();
                    ctx.moveTo(src.x, src.y);
                    ctx.lineTo(dst.x, dst.y);
                    ctx.stroke();
                }}
            }});
            
            positions.forEach(pos => {{
                const node = nodes.find(n => n.id === pos.id);
                ctx.fillStyle = node.color || '#007bff';
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, node.size || 8, 0, 2 * Math.PI);
                ctx.fill();
                
                ctx.strokeStyle = '#333';
                ctx.lineWidth = 1;
                ctx.stroke();
                
                ctx.fillStyle = '#000';
                ctx.font = '12px Arial';
                ctx.textAlign = 'center';
                ctx.fillText(node.label || node.id, pos.x, pos.y - 15);
            }});
            
            ctx.restore();
        }}
        
        canvas.addEventListener('mousedown', (e) => {{
            isDragging = true;
            lastMouse = {{ x: e.clientX, y: e.clientY }};
        }});
        
        canvas.addEventListener('mousemove', (e) => {{
            if (isDragging) {{
                camera.x += e.clientX - lastMouse.x;
                camera.y += e.clientY - lastMouse.y;
                lastMouse = {{ x: e.clientX, y: e.clientY }};
                render();
            }}
        }});
        
        canvas.addEventListener('mouseup', () => {{
            isDragging = false;
        }});
        
        function resetView() {{
            camera = {{ x: 0, y: 0 }};
            calculateLayout();
            render();
        }}
        
        function toggleLayout() {{
            const layouts = ['circular', 'grid', 'force-directed'];
            const currentIndex = layouts.indexOf(layout);
            layout = layouts[(currentIndex + 1) % layouts.length];
            document.getElementById('layout').textContent = layout;
            calculateLayout();
            render();
        }}
        
        calculateLayout();
        render();
        
        console.log('Graph visualization loaded with', nodes.length, 'nodes and', edges.length, 'edges');
    </script>
</body>
</html>'''
    
    def _extract_graph_data(self):
        """Extract nodes and edges data from the graph data source."""
        nodes_data = []
        edges_data = []
        
        try:
            # Try to get nodes and edges from Graph or Subgraph
            if hasattr(self.data_source, 'node_ids') and hasattr(self.data_source, 'edge_ids'):
                # Graph-like interface (Graph, Subgraph, etc.)
                for node_id in self.data_source.node_ids:
                    # Get node attributes
                    attrs = {}
                    try:
                        # Try different methods to get node attributes
                        if hasattr(self.data_source, 'get_node_attrs'):
                            attrs = self.data_source.get_node_attrs(node_id) or {}
                        elif hasattr(self.data_source, 'get_all_node_attrs'):
                            attrs = self.data_source.get_all_node_attrs(node_id) or {}
                    except Exception as e:
                        # If attribute access fails, try individual attributes
                        try:
                            if hasattr(self.data_source, 'get_node_attr'):
                                attrs['label'] = self.data_source.get_node_attr(node_id, 'label')
                                attrs['color'] = self.data_source.get_node_attr(node_id, 'color')
                                attrs['age'] = self.data_source.get_node_attr(node_id, 'age')
                                attrs['size'] = self.data_source.get_node_attr(node_id, 'size')
                        except:
                            pass
                    
                    nodes_data.append({
                        'id': str(node_id),
                        'label': attrs.get('label', f'Node {node_id}'),
                        'color': attrs.get('color', '#4CAF50'),
                        'size': attrs.get('size', 8),
                        'age': attrs.get('age'),  # Include other attributes for info
                    })
                
                for edge_id in self.data_source.edge_ids:
                    try:
                        src, dst = self.data_source.edge_endpoints(edge_id)
                        attrs = {}
                        
                        # Try to get edge attributes
                        try:
                            if hasattr(self.data_source, 'get_edge_attrs'):
                                attrs = self.data_source.get_edge_attrs(edge_id) or {}
                            elif hasattr(self.data_source, 'get_all_edge_attrs'):
                                attrs = self.data_source.get_all_edge_attrs(edge_id) or {}
                        except:
                            # Try individual attributes
                            try:
                                if hasattr(self.data_source, 'get_edge_attr'):
                                    attrs['weight'] = self.data_source.get_edge_attr(edge_id, 'weight')
                                    attrs['label'] = self.data_source.get_edge_attr(edge_id, 'label')
                            except:
                                pass
                        
                        edges_data.append({
                            'id': str(edge_id),
                            'source': str(src),
                            'target': str(dst),
                            'weight': attrs.get('weight', 1.0),
                            'label': attrs.get('label')
                        })
                    except Exception as e:
                        print(f"⚠️  Failed to process edge {edge_id}: {e}")
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
    
    def _get_graph_direction(self):
        """Get graph direction (directed/undirected)."""
        try:
            if hasattr(self.data_source, 'is_directed'):
                directed_attr = getattr(self.data_source, 'is_directed')
                if callable(directed_attr):
                    return directed_attr()
                else:
                    return bool(directed_attr)
        except:
            pass
        return True  # Default to directed
    
    def _open_browser(self, url: str):
        """Open URL in default browser."""
        import webbrowser
        import threading
        
        def open_delayed():
            import time
            time.sleep(0.5)
            try:
                webbrowser.open(url)
            except Exception as e:
                print(f"Could not open browser automatically: {e}")
                print(f"Please open: {url}")
        
        thread = threading.Thread(target=open_delayed)
        thread.daemon = True
        thread.start()


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
    
    def __init__(self, file_path, size_bytes=0, html_content=None):
        self.file_path = file_path
        self.size_bytes = size_bytes
        self._html_content = html_content
    
    def __str__(self):
        return f"StaticViz(file_path='{self.file_path}', size_bytes={self.size_bytes})"
    
    def _repr_html_(self):
        """IPython/Jupyter display representation."""
        if self._html_content and self.file_path.endswith(('.html', '.htm')):
            # Embed the HTML directly in Jupyter
            return self._html_content
        elif self.file_path.endswith(('.png', '.jpg', '.jpeg')):
            # For images, show as embedded image
            return f'<img src="{self.file_path}" style="max-width: 100%; height: auto;">'
        elif self.file_path.endswith('.svg'):
            # For SVG, try to read and embed
            try:
                with open(self.file_path, 'r') as f:
                    return f.read()
            except:
                return f'<p>SVG file created: <a href="{self.file_path}">{self.file_path}</a></p>'
        else:
            return f'<p>Visualization exported to: <a href="{self.file_path}">{self.file_path}</a> ({self.size_bytes} bytes)</p>'
    
    def show(self):
        """Display the visualization in Jupyter notebook."""
        try:
            from IPython.display import display, HTML
            if self._html_content:
                display(HTML(self._html_content))
            else:
                display(HTML(f'<p>File created: {self.file_path} ({self.size_bytes} bytes)</p>'))
        except ImportError:
            print(f"Visualization saved to: {self.file_path} ({self.size_bytes} bytes)")
            print("To display in Jupyter, install IPython: pip install ipython")


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
        self._template = None  # Lazy-loaded template instance
    
    def _get_template(self) -> VizTemplate:
        """Get or create the VizTemplate instance for this data source."""
        if self._template is None:
            self._template = VizTemplate(self.data_source)
        return self._template
    
    def render(
        self,
        backend: Union[VizBackend, str],
        *,
        # Universal parameters
        layout: str = "force-directed",
        theme: str = "light",
        width: int = 800,
        height: int = 600,
        title: Optional[str] = None,
        # Widget enhancement
        interactive: bool = False,  # NEW: Enable widget mode for Jupyter
        style_theme: Optional[str] = None,  # NEW: Style system theme
        # Backend-specific parameters
        **kwargs
    ) -> Union[None, StaticViz, InteractiveVizSession, str, 'GroggyGraphWidget']:
        """
        🎯 UNIFIED VISUALIZATION METHOD - The One Command To Rule Them All
        
        This is the new unified approach that replaces 4 separate methods with
        a single command that adapts to different backends, with optional
        widget enhancement for full interactivity.
        
        Args:
            backend: Target backend ('jupyter', 'streaming', 'file', 'local')
            layout: Layout algorithm ("force-directed", "circular", "grid")
            theme: Visual theme ("light", "dark", "publication")
            width: Canvas width in pixels
            height: Canvas height in pixels
            title: Custom title for visualization
            interactive: Enable interactive widget mode (Jupyter only)
            style_theme: Style system theme (overrides theme if provided)
            **kwargs: Backend-specific parameters
            
        Returns:
            - None for jupyter backend (displays directly)
            - GroggyGraphWidget for jupyter + interactive=True
            - StaticViz for file backend
            - InteractiveVizSession for streaming backend
            - str (HTML) for local backend
            
        Examples:
            >>> # Basic Jupyter embedding
            >>> g.viz().render(backend='jupyter')
            
            >>> # Interactive Jupyter widget with full functionality
            >>> widget = g.viz().render(backend='jupyter', interactive=True)
            >>> widget.on_node_click(lambda node_id, data: print(f"Clicked: {node_id}"))
            
            >>> # Styled visualization with publication theme
            >>> g.viz().render('jupyter', style_theme='publication', interactive=True)
            
            >>> # Interactive server (replaces .interactive())
            >>> session = g.viz().render(backend='streaming', port=8080)
            
            >>> # Static file export (replaces .static())
            >>> result = g.viz().render(backend='file', filename='graph.html')
            
            >>> # Self-contained HTML (replaces .local_interactive())
            >>> html = g.viz().render(backend='local')
            
            >>> # Enhanced styling with custom theme
            >>> g.viz().render('local', style_theme='dark', width=1000)
        """
        # Validate and normalize backend
        backend_enum = VizBackend.from_string(backend)
        
        # Use style_theme if provided, otherwise use theme
        final_theme = style_theme if style_theme is not None else theme
        
        # Widget mode for Jupyter with interactive=True
        if backend_enum == VizBackend.JUPYTER and interactive:
            return self._render_jupyter_widget(
                layout=layout,
                theme=final_theme,
                width=width,
                height=height,
                title=title,
                **kwargs
            )
        
        # Standard unified rendering (possibly with style enhancement)
        template = self._get_template()
        
        # Apply style system if available and style_theme is specified
        if STYLE_SYSTEM_AVAILABLE and style_theme is not None:
            result = template.render(
                backend=backend,
                layout=layout,
                theme=final_theme,
                width=width,
                height=height,
                title=title,
                **kwargs
            )
            
            # Enhance with style system for HTML outputs
            if isinstance(result, str):
                style_system = GroggyStyleSystem(style_theme)
                result = style_system.apply_to_html_template(result)
            
            return result
        else:
            return template.render(
                backend=backend,
                layout=layout,
                theme=final_theme,
                width=width,
                height=height,
                title=title,
                **kwargs
            )
    
    def widget(
        self,
        style_theme: str = 'light',
        width: int = 800,
        height: int = 600,
        layout: str = 'force-directed',
        **kwargs
    ) -> 'GroggyGraphWidget':
        """
        Direct widget access for advanced use cases.
        
        Creates an interactive Jupyter widget with full Python-JavaScript
        communication capabilities.
        
        Args:
            style_theme: Style system theme
            width: Widget width in pixels
            height: Widget height in pixels
            layout: Initial layout algorithm
            **kwargs: Additional widget configuration
            
        Returns:
            GroggyGraphWidget instance
            
        Examples:
            >>> # Create widget with callbacks
            >>> widget = g.viz().widget(style_theme='dark')
            >>> widget.on_node_click(lambda node_id, data: print(f"Clicked: {node_id}"))
            >>> widget.on_layout_change(lambda layout: print(f"Layout: {layout}"))
            >>> display(widget)
            
            >>> # Advanced configuration
            >>> widget = g.viz().widget(
            ...     style_theme='publication',
            ...     width=1000,
            ...     height=700,
            ...     layout='circular',
            ...     enable_animations=True
            ... )
        """
        if not WIDGETS_AVAILABLE:
            raise ImportError(
                "Jupyter widgets not available. Install with:\\n"
                "  pip install ipywidgets\\n"
                "Then enable in Jupyter:\\n"
                "  jupyter nbextension enable --py --sys-prefix ipywidgets"
            )
        
        # Create style system
        style_system = None
        if STYLE_SYSTEM_AVAILABLE:
            style_system = GroggyStyleSystem(style_theme)
        
        # Create widget
        widget = GroggyGraphWidget(
            data_source=self.data_source,
            style_system=style_system,
            width=width,
            height=height,
            layout_algorithm=layout,
            theme=style_theme,
            **kwargs
        )
        
        return widget
    
    def _render_jupyter_widget(
        self,
        layout: str = 'force-directed',
        theme: str = 'light',
        width: int = 800,
        height: int = 600,
        title: Optional[str] = None,
        **kwargs
    ) -> 'GroggyGraphWidget':
        """
        Render enhanced Jupyter widget with full interactivity.
        
        This method creates and displays an interactive widget that provides
        drag-and-drop nodes, real-time callbacks, and bidirectional communication.
        """
        try:
            # Create widget
            widget = self.widget(
                style_theme=theme,
                width=width,
                height=height,
                layout=layout,
                **kwargs
            )
            
            # Display widget in Jupyter
            try:
                from IPython.display import display
                display(widget)
            except ImportError:
                print("IPython not available - widget created but not displayed")
                print("Use display(widget) to show the widget in Jupyter")
            
            return widget
            
        except ImportError as e:
            print("⚠️  Jupyter widgets not available. Falling back to basic rendering.")
            print("   To enable full interactivity: pip install ipywidgets")
            print("   Then enable: jupyter nbextension enable --py --sys-prefix ipywidgets")
            
            # Fallback to basic unified rendering
            template = self._get_template()
            template.render(
                backend='jupyter',
                layout=layout,
                theme=theme,
                width=width,
                height=height,
                title=title,
                **kwargs
            )
            return None
    
    # ========================================================================
    # UNIFIED VISUALIZATION SYSTEM ONLY
    # 
    # This module now uses only the unified render() method with optional
    # widget enhancement. Legacy methods have been removed to encourage
    # adoption of the new unified API.
    # ========================================================================
    
    
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
                    
                    # Handle is_directed - could be property or method
                    is_directed = True  # Default
                    if hasattr(data_source, 'is_directed'):
                        directed_attr = getattr(data_source, 'is_directed')
                        if callable(directed_attr):
                            is_directed = directed_attr()
                        else:
                            is_directed = bool(directed_attr)
                    
                    supports_graph = True
                    graph_info = {
                        'node_count': node_count,
                        'edge_count': edge_count,
                        'is_directed': is_directed,
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




# Export the main classes and functions
__all__ = [
    # Unified visualization system
    'VizBackend',
    'VizTemplate', 
    'VizAccessor',
    # Supporting classes
    'VizConfig',
    'InteractiveVizSession',
    'StaticViz',
    # Widget system (if available)
    'GroggyGraphWidget',
    'WIDGETS_AVAILABLE',
    # Style system (if available)  
    'GroggyStyleSystem',
    'STYLE_SYSTEM_AVAILABLE',
    # Utilities
    'add_viz_accessor',
]