#!/usr/bin/env python3
"""
Groggy Streaming Visualization Template Generator

This script extracts the HTML structure and CSS from the streaming graph/table visualizer
to create standalone HTML and CSS files for rapid prototyping and styling.

Usage:
    python streaming_template_generator.py

Generates:
    - streaming_template.html: Interactive graph/table template
    - css/graph_visualization.css: Extracted graph visualization CSS
    - css/sleek.css: Base table styling CSS
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

class StreamingTemplateGenerator:
    def __init__(self):
        self.src_path = Path("../../src/viz/streaming")
        self.css_path = self.src_path / "css"
        self.output_dir = Path("streaming_prototype")

    def generate_all(self):
        """Generate all template files"""
        print("🚀 Groggy Streaming Template Generator")
        print("=" * 50)

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "css").mkdir(exist_ok=True)

        # Extract current HTML structure from html.rs
        html_structure = self.extract_html_structure()

        # Copy CSS files
        self.copy_css_files()

        # Generate enhanced template
        self.generate_streaming_template(html_structure)

        # Generate CSS playground
        self.generate_css_playground()

        print(f"\n✅ Streaming templates generated in: {self.output_dir}")
        print(f"   📁 {self.output_dir}/streaming_template.html")
        print(f"   📁 {self.output_dir}/css_playground.html")
        print(f"   📁 {self.output_dir}/css/")

    def extract_html_structure(self):
        """Extract HTML structure from streaming html.rs"""
        print("\n📋 Extracting HTML structure from html.rs...")

        html_file = self.src_path / "html.rs"
        if not html_file.exists():
            print(f"   ⚠️ html.rs not found at {html_file}")
            return ""

        content = html_file.read_text()

        # Extract the HTML template from the Rust string
        # Look for the main HTML structure between the triple quotes
        html_pattern = r'r#"(.*?)"#'
        matches = re.findall(html_pattern, content, re.DOTALL)

        if matches:
            # Get the largest match (likely the main template)
            html_template = max(matches, key=len)
            print(f"   ✓ Extracted HTML structure ({len(html_template)} chars)")
            return html_template
        else:
            print("   ⚠️ Could not extract HTML template")
            return ""

    def copy_css_files(self):
        """Copy CSS files from streaming directory"""
        print("\n📋 Copying CSS files...")

        css_files = ["graph_visualization.css", "sleek.css"]

        for css_file in css_files:
            src_file = self.css_path / css_file
            dest_file = self.output_dir / "css" / css_file

            if src_file.exists():
                css_content = src_file.read_text()
                dest_file.write_text(css_content)
                print(f"   ✓ {css_file}")
            else:
                print(f"   ⚠️ {css_file} not found")

    def generate_streaming_template(self, html_structure):
        """Generate the streaming template with sample data"""
        print("\n🏗️ Generating streaming template...")

        # Create sample graph data
        sample_graph_data = self.create_sample_graph_data()

        # Create enhanced template
        template_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Groggy Streaming Visualizer Template</title>
    <link rel="stylesheet" href="css/sleek.css">
    <link rel="stylesheet" href="css/graph_visualization.css">
    <style>
        body {{
            font-family: system-ui, -apple-system, sans-serif;
            margin: 0;
            padding: 20px;
            background: var(--bg);
        }}

        .prototype-controls {{
            margin-bottom: 20px;
            padding: 15px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            gap: 20px;
            align-items: center;
            flex-wrap: wrap;
        }}

        .control-group {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .control-group label {{
            font-weight: 500;
        }}

        .control-group select, .control-group input {{
            padding: 4px 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}

        .style-panel {{
            position: fixed;
            top: 20px;
            right: 20px;
            width: 300px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            padding: 15px;
            z-index: 1000;
            max-height: 80vh;
            overflow-y: auto;
            display: none;
        }}

        .style-panel.active {{
            display: block;
        }}

        .style-section {{
            margin-bottom: 20px;
        }}

        .style-section h4 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            font-weight: 600;
            color: #333;
        }}

        .style-control {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}

        .style-control label {{
            font-size: 12px;
            color: #666;
        }}

        .style-control input {{
            width: 60px;
            font-size: 11px;
        }}
    </style>
</head>
<body>
    <h1>🎨 Groggy Streaming Visualizer Prototype</h1>

    <div class="prototype-controls">
        <div class="control-group">
            <label>View:</label>
            <select id="view-select" onchange="switchView(this.value)">
                <option value="table">📊 Table</option>
                <option value="graph">🕸️ Graph</option>
            </select>
        </div>

        <div class="control-group">
            <label>Node Size:</label>
            <input type="range" id="node-size" min="3" max="20" value="8" onchange="updateNodeSize(this.value)">
            <span id="node-size-value">8</span>
        </div>

        <div class="control-group">
            <label>Edge Curviness:</label>
            <input type="range" id="edge-curve" min="0" max="1" step="0.1" value="0.2" onchange="updateEdgeCurviness(this.value)">
            <span id="edge-curve-value">0.2</span>
        </div>

        <div class="control-group">
            <button onclick="toggleStylePanel()">🎨 Style Panel</button>
        </div>

        <div class="control-group">
            <button onclick="generateSampleData()">🔄 New Data</button>
        </div>
    </div>

    <!-- Style Panel -->
    <div id="style-panel" class="style-panel">
        <h3>Live CSS Editor</h3>

        <div class="style-section">
            <h4>Node Styling</h4>
            <div class="style-control">
                <label>Default Color:</label>
                <input type="color" id="node-color" value="#4dabf7" onchange="updateNodeColor(this.value)">
            </div>
            <div class="style-control">
                <label>Selected Color:</label>
                <input type="color" id="node-selected-color" value="#ff6b6b" onchange="updateNodeSelectedColor(this.value)">
            </div>
            <div class="style-control">
                <label>Hover Color:</label>
                <input type="color" id="node-hover-color" value="#339af0" onchange="updateNodeHoverColor(this.value)">
            </div>
        </div>

        <div class="style-section">
            <h4>Edge Styling</h4>
            <div class="style-control">
                <label>Default Color:</label>
                <input type="color" id="edge-color" value="#999999" onchange="updateEdgeColor(this.value)">
            </div>
            <div class="style-control">
                <label>Width:</label>
                <input type="range" id="edge-width" min="1" max="5" value="1" onchange="updateEdgeWidth(this.value)">
            </div>
        </div>

        <div class="style-section">
            <h4>Table Styling</h4>
            <div class="style-control">
                <label>Border Color:</label>
                <input type="color" id="table-border" value="#eee" onchange="updateTableBorder(this.value)">
            </div>
            <div class="style-control">
                <label>Hover Color:</label>
                <input type="color" id="table-hover" value="#f3f6ff" onchange="updateTableHover(this.value)">
            </div>
        </div>

        <div class="style-section">
            <button onclick="exportStyles()" style="width: 100%; padding: 8px; background: #4dabf7; color: white; border: none; border-radius: 4px; cursor: pointer;">
                📋 Export Styles
            </button>
        </div>
    </div>

    <div class="table-container">
        {self.create_template_structure()}
    </div>

    <script>
        // Sample data and configuration
        {sample_graph_data}

        {self.create_template_javascript()}
    </script>
</body>
</html>"""

        output_file = self.output_dir / "streaming_template.html"
        output_file.write_text(template_content)
        print(f"   ✓ streaming_template.html")

    def create_template_structure(self):
        """Create the HTML structure for table and graph views"""
        return '''
        <div class="table-header">
            <div class="table-title">🎨 Interactive Visualization</div>
            <div style="display: flex; align-items: center; gap: 16px;">
                <div class="view-controls">
                    <div class="view-toggle">
                        <button id="table-view-btn" class="view-toggle-btn active">📊 Table</button>
                        <button id="graph-view-btn" class="view-toggle-btn">🕸️ Graph</button>
                    </div>
                </div>
                <div class="table-stats">Sample data × 6 cols</div>
                <div id="connection-status" class="connection-status status-connected">Prototype Mode</div>
            </div>
        </div>

        <div class="viz-container">
            <!-- Table View -->
            <div id="table-view" class="table-view">
                <div class="groggy-display-container">
                    <table id="data-table" class="groggy-table">
                        <thead>
                            <tr>
                                <th data-col-priority="high">ID</th>
                                <th data-col-priority="high">Name</th>
                                <th data-col-priority="med">Type</th>
                                <th data-col-priority="med">Value</th>
                                <th data-col-priority="low">Connected</th>
                                <th data-col-priority="low">Weight</th>
                            </tr>
                        </thead>
                        <tbody id="table-tbody">
                            <!-- Data will be populated by JavaScript -->
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Graph View -->
            <div id="graph-view" class="graph-view" style="display: none;">
                <canvas id="graph-canvas" class="graph-canvas"></canvas>

                <!-- Graph Controls -->
                <div class="graph-controls">
                    <button class="graph-btn" onclick="zoomIn()" title="Zoom In">🔍+</button>
                    <button class="graph-btn" onclick="zoomOut()" title="Zoom Out">🔍-</button>
                    <button class="graph-btn" onclick="resetView()" title="Reset View">🎯</button>
                </div>

                <!-- Layout Controls -->
                <div class="layout-controls">
                    <label>Layout:</label>
                    <select class="layout-select" onchange="changeLayout(this.value)">
                        <option value="force">Force</option>
                        <option value="circle">Circle</option>
                        <option value="grid">Grid</option>
                    </select>
                </div>
            </div>
        </div>'''

    def create_sample_graph_data(self):
        """Create sample graph data in JavaScript format"""
        return '''
        // Sample graph data
        let graphData = {
            nodes: [
                {id: "node1", label: "Alice", x: -50, y: -30, size: 8, type: "person"},
                {id: "node2", label: "Bob", x: 50, y: -30, size: 12, type: "person"},
                {id: "node3", label: "Carol", x: 0, y: 30, size: 10, type: "person"},
                {id: "node4", label: "Data", x: -80, y: 0, size: 6, type: "data"},
                {id: "node5", label: "Process", x: 80, y: 0, size: 14, type: "process"},
                {id: "node6", label: "Output", x: 0, y: 60, size: 8, type: "output"}
            ],
            edges: [
                {id: "edge1", source: "node1", target: "node2", weight: 2},
                {id: "edge2", source: "node2", target: "node3", weight: 1},
                {id: "edge3", source: "node1", target: "node3", weight: 3},
                {id: "edge4", source: "node4", target: "node1", weight: 1},
                {id: "edge5", source: "node5", target: "node2", weight: 2},
                {id: "edge6", source: "node3", target: "node6", weight: 1}
            ]
        };

        // Sample table data
        let tableData = [
            ["node1", "Alice", "person", "active", "true", "1.5"],
            ["node2", "Bob", "person", "active", "true", "2.0"],
            ["node3", "Carol", "person", "inactive", "false", "1.8"],
            ["node4", "Data", "data", "source", "true", "0.5"],
            ["node5", "Process", "process", "running", "true", "3.2"],
            ["node6", "Output", "output", "ready", "false", "1.0"]
        ];'''

    def create_template_javascript(self):
        """Create the JavaScript for the template"""
        return '''
        // Canvas and rendering variables
        let canvas = null;
        let ctx = null;
        let scale = 1.0;
        let translateX = 0;
        let translateY = 0;
        let selectedNode = null;
        let hoveredNode = null;
        let isDragging = false;
        let lastMouseX = 0;
        let lastMouseY = 0;
        let selectedNodes = new Set();
        let currentView = 'table';

        // Canvas styling configuration
        const canvasStyles = {
            node: {
                defaultRadius: 8,
                minRadius: 3,
                maxRadius: 20,
                borderWidth: 2,
                borderHoverWidth: 3,
                colors: {
                    default: '#4dabf7',
                    selected: '#ff6b6b',
                    hover: '#339af0',
                    border: '#333',
                    label: '#333'
                },
                sizeProperty: 'size',
                sizeScale: 1.0
            },
            edge: {
                defaultWidth: 1,
                selectedWidth: 3,
                hoverWidth: 2,
                opacity: 0.8,
                hoverOpacity: 1.0,
                colors: {
                    default: '#999',
                    selected: '#ff8cc8',
                    hover: '#666'
                },
                curviness: 0.2,
                arrowSize: 6,
                animationSpeed: 0.02
            },
            animation: {
                nodeHoverGrow: 1.3,
                edgePulse: true,
                transitionSpeed: 200
            }
        };

        // Initialize everything
        window.addEventListener('load', function() {
            initializeTemplate();
            populateTable();
            setupEventListeners();
        });

        function initializeTemplate() {
            canvas = document.getElementById('graph-canvas');
            if (canvas) {
                ctx = canvas.getContext('2d');
                resizeCanvas();
                window.addEventListener('resize', resizeCanvas);
            }
        }

        function resizeCanvas() {
            if (canvas) {
                const container = canvas.parentElement;
                canvas.width = container.clientWidth;
                canvas.height = container.clientHeight;
                renderGraph();
            }
        }

        function populateTable() {
            const tbody = document.getElementById('table-tbody');
            if (tbody) {
                tbody.innerHTML = tableData.map(row =>
                    `<tr>${row.map(cell => `<td>${cell}</td>`).join('')}</tr>`
                ).join('');
            }
        }

        function setupEventListeners() {
            document.getElementById('table-view-btn')?.addEventListener('click', () => switchView('table'));
            document.getElementById('graph-view-btn')?.addEventListener('click', () => switchView('graph'));

            if (canvas) {
                canvas.addEventListener('mousedown', onCanvasMouseDown);
                canvas.addEventListener('mousemove', onCanvasMouseMove);
                canvas.addEventListener('mouseup', onCanvasMouseUp);
                canvas.addEventListener('wheel', onCanvasWheel);
            }
        }

        function switchView(view) {
            currentView = view;
            const tableView = document.getElementById('table-view');
            const graphView = document.getElementById('graph-view');

            document.querySelectorAll('.view-toggle-btn').forEach(btn => btn.classList.remove('active'));

            if (view === 'table') {
                tableView.style.display = 'block';
                graphView.style.display = 'none';
                document.getElementById('table-view-btn').classList.add('active');
            } else if (view === 'graph') {
                tableView.style.display = 'none';
                graphView.style.display = 'block';
                document.getElementById('graph-view-btn').classList.add('active');
                setTimeout(() => {
                    resizeCanvas();
                    renderGraph();
                }, 100);
            }

            // Update view selector
            const viewSelect = document.getElementById('view-select');
            if (viewSelect) viewSelect.value = view;
        }

        function getNodeRadius(node) {
            let baseRadius = canvasStyles.node.defaultRadius;
            if (node[canvasStyles.node.sizeProperty]) {
                baseRadius = Math.max(
                    canvasStyles.node.minRadius,
                    Math.min(
                        canvasStyles.node.maxRadius,
                        node[canvasStyles.node.sizeProperty] * canvasStyles.node.sizeScale
                    )
                );
            }
            if (hoveredNode === node.id) {
                baseRadius *= canvasStyles.animation.nodeHoverGrow;
            }
            return baseRadius;
        }

        function drawCurvedEdge(sourceNode, targetNode, curviness = 0) {
            if (curviness === 0) {
                ctx.beginPath();
                ctx.moveTo(sourceNode.x || 0, sourceNode.y || 0);
                ctx.lineTo(targetNode.x || 0, targetNode.y || 0);
                ctx.stroke();
                return;
            }

            const sx = sourceNode.x || 0;
            const sy = sourceNode.y || 0;
            const tx = targetNode.x || 0;
            const ty = targetNode.y || 0;

            const midX = (sx + tx) / 2;
            const midY = (sy + ty) / 2;

            const dx = tx - sx;
            const dy = ty - sy;
            const dist = Math.sqrt(dx * dx + dy * dy);
            const offsetX = -dy / dist * dist * curviness;
            const offsetY = dx / dist * dist * curviness;

            const cpX = midX + offsetX;
            const cpY = midY + offsetY;

            ctx.beginPath();
            ctx.moveTo(sx, sy);
            ctx.quadraticCurveTo(cpX, cpY, tx, ty);
            ctx.stroke();
        }

        function renderGraph() {
            if (!ctx || !canvas) return;

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.save();

            ctx.translate(canvas.width / 2 + translateX, canvas.height / 2 + translateY);
            ctx.scale(scale, scale);

            // Render edges
            graphData.edges.forEach(edge => {
                const sourceNode = graphData.nodes.find(n => n.id === edge.source);
                const targetNode = graphData.nodes.find(n => n.id === edge.target);

                if (sourceNode && targetNode) {
                    let edgeWidth = canvasStyles.edge.defaultWidth;
                    let edgeColor = canvasStyles.edge.colors.default;

                    if (edge.weight && edge.weight > 1) {
                        edgeWidth *= Math.min(3, Math.sqrt(edge.weight));
                    }

                    ctx.strokeStyle = edgeColor;
                    ctx.lineWidth = edgeWidth;
                    ctx.globalAlpha = canvasStyles.edge.opacity;

                    drawCurvedEdge(sourceNode, targetNode, canvasStyles.edge.curviness);
                    ctx.globalAlpha = 1.0;
                }
            });

            // Render nodes
            graphData.nodes.forEach(node => {
                const x = node.x || 0;
                const y = node.y || 0;
                const radius = getNodeRadius(node);

                const isSelected = selectedNodes.has(node.id);
                const isHovered = hoveredNode === node.id;

                let nodeColor = canvasStyles.node.colors.default;
                let borderWidth = canvasStyles.node.borderWidth;

                if (isSelected) {
                    nodeColor = canvasStyles.node.colors.selected;
                    borderWidth = canvasStyles.node.borderHoverWidth;
                } else if (isHovered) {
                    nodeColor = canvasStyles.node.colors.hover;
                    borderWidth = canvasStyles.node.borderHoverWidth;
                }

                // Node shadow
                ctx.shadowColor = 'rgba(0,0,0,0.2)';
                ctx.shadowBlur = isSelected || isHovered ? 8 : 4;
                ctx.shadowOffsetX = 1;
                ctx.shadowOffsetY = 1;

                ctx.beginPath();
                ctx.arc(x, y, radius, 0, 2 * Math.PI);
                ctx.fillStyle = nodeColor;
                ctx.fill();

                // Reset shadow
                ctx.shadowColor = 'transparent';
                ctx.shadowBlur = 0;
                ctx.shadowOffsetX = 0;
                ctx.shadowOffsetY = 0;

                // Node border
                ctx.strokeStyle = canvasStyles.node.colors.border;
                ctx.lineWidth = borderWidth;
                ctx.stroke();

                // Node label
                if (node.label) {
                    ctx.fillStyle = canvasStyles.node.colors.label;
                    ctx.font = '10px Arial';
                    ctx.textAlign = 'center';
                    ctx.fillText(node.label, x, y + radius + 12);
                }
            });

            ctx.restore();
        }

        // Mouse event handlers
        function onCanvasMouseDown(e) {
            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;

            lastMouseX = mouseX;
            lastMouseY = mouseY;
            isDragging = true;

            const worldX = (mouseX - canvas.width / 2 - translateX) / scale;
            const worldY = (mouseY - canvas.height / 2 - translateY) / scale;

            selectedNode = null;
            selectedNodes.clear();

            graphData.nodes.forEach(node => {
                const dx = worldX - (node.x || 0);
                const dy = worldY - (node.y || 0);
                const distance = Math.sqrt(dx * dx + dy * dy);
                const nodeRadius = getNodeRadius(node);
                if (distance < nodeRadius + 2) {
                    selectedNode = node.id;
                    selectedNodes.add(node.id);
                }
            });

            renderGraph();
        }

        function onCanvasMouseMove(e) {
            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;

            const worldX = (mouseX - canvas.width / 2 - translateX) / scale;
            const worldY = (mouseY - canvas.height / 2 - translateY) / scale;

            if (!isDragging) {
                // Handle hover
                let newHoveredNode = null;
                graphData.nodes.forEach(node => {
                    const dx = worldX - (node.x || 0);
                    const dy = worldY - (node.y || 0);
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    const nodeRadius = getNodeRadius(node);
                    if (distance < nodeRadius + 2) {
                        newHoveredNode = node.id;
                    }
                });

                if (newHoveredNode !== hoveredNode) {
                    hoveredNode = newHoveredNode;
                    canvas.style.cursor = hoveredNode ? 'pointer' : 'grab';
                    renderGraph();
                }
                return;
            }

            const deltaX = mouseX - lastMouseX;
            const deltaY = mouseY - lastMouseY;

            if (selectedNode !== null) {
                const node = graphData.nodes.find(n => n.id === selectedNode);
                if (node) {
                    node.x = (node.x || 0) + deltaX / scale;
                    node.y = (node.y || 0) + deltaY / scale;
                }
            } else {
                translateX += deltaX;
                translateY += deltaY;
            }

            lastMouseX = mouseX;
            lastMouseY = mouseY;
            renderGraph();
        }

        function onCanvasMouseUp(e) {
            isDragging = false;
            selectedNode = null;
            canvas.style.cursor = 'grab';
        }

        function onCanvasWheel(e) {
            e.preventDefault();
            const scaleFactor = e.deltaY > 0 ? 0.9 : 1.1;
            scale = Math.max(0.1, Math.min(5.0, scale * scaleFactor));
            renderGraph();
        }

        // Control functions
        function zoomIn() {
            scale = Math.min(5.0, scale * 1.2);
            renderGraph();
        }

        function zoomOut() {
            scale = Math.max(0.1, scale * 0.8);
            renderGraph();
        }

        function resetView() {
            scale = 1.0;
            translateX = 0;
            translateY = 0;
            renderGraph();
        }

        function changeLayout(layout) {
            // Simple layout algorithms
            const nodeCount = graphData.nodes.length;

            graphData.nodes.forEach((node, i) => {
                if (layout === 'circle') {
                    const angle = (i / nodeCount) * 2 * Math.PI;
                    const radius = 80;
                    node.x = Math.cos(angle) * radius;
                    node.y = Math.sin(angle) * radius;
                } else if (layout === 'grid') {
                    const cols = Math.ceil(Math.sqrt(nodeCount));
                    const spacing = 60;
                    node.x = (i % cols) * spacing - (cols * spacing / 2);
                    node.y = Math.floor(i / cols) * spacing - 40;
                } else if (layout === 'force') {
                    // Simple force layout
                    node.x = (Math.random() - 0.5) * 200;
                    node.y = (Math.random() - 0.5) * 200;
                }
            });

            renderGraph();
        }

        // Style control functions
        function updateNodeSize(value) {
            canvasStyles.node.defaultRadius = parseInt(value);
            document.getElementById('node-size-value').textContent = value;
            renderGraph();
        }

        function updateEdgeCurviness(value) {
            canvasStyles.edge.curviness = parseFloat(value);
            document.getElementById('edge-curve-value').textContent = value;
            renderGraph();
        }

        function updateNodeColor(color) {
            canvasStyles.node.colors.default = color;
            renderGraph();
        }

        function updateNodeSelectedColor(color) {
            canvasStyles.node.colors.selected = color;
            renderGraph();
        }

        function updateNodeHoverColor(color) {
            canvasStyles.node.colors.hover = color;
            renderGraph();
        }

        function updateEdgeColor(color) {
            canvasStyles.edge.colors.default = color;
            renderGraph();
        }

        function updateEdgeWidth(value) {
            canvasStyles.edge.defaultWidth = parseInt(value);
            renderGraph();
        }

        function updateTableBorder(color) {
            document.documentElement.style.setProperty('--line', color);
        }

        function updateTableHover(color) {
            document.documentElement.style.setProperty('--row-hover', color);
        }

        function toggleStylePanel() {
            const panel = document.getElementById('style-panel');
            panel.classList.toggle('active');
        }

        function generateSampleData() {
            // Generate new random graph data
            const nodeCount = 5 + Math.floor(Math.random() * 5);
            graphData.nodes = [];
            graphData.edges = [];

            for (let i = 0; i < nodeCount; i++) {
                graphData.nodes.push({
                    id: `node${i}`,
                    label: `Node ${i}`,
                    x: (Math.random() - 0.5) * 150,
                    y: (Math.random() - 0.5) * 150,
                    size: 5 + Math.random() * 10,
                    type: ['person', 'data', 'process'][Math.floor(Math.random() * 3)]
                });
            }

            // Generate random edges
            for (let i = 0; i < nodeCount - 1; i++) {
                const source = `node${i}`;
                const target = `node${(i + 1) % nodeCount}`;
                graphData.edges.push({
                    id: `edge${i}`,
                    source,
                    target,
                    weight: 1 + Math.random() * 2
                });
            }

            renderGraph();
            populateTable();
        }

        function exportStyles() {
            const styles = {
                nodeDefaultColor: canvasStyles.node.colors.default,
                nodeSelectedColor: canvasStyles.node.colors.selected,
                nodeHoverColor: canvasStyles.node.colors.hover,
                edgeDefaultColor: canvasStyles.edge.colors.default,
                edgeDefaultWidth: canvasStyles.edge.defaultWidth,
                edgeCurviness: canvasStyles.edge.curviness,
                nodeDefaultRadius: canvasStyles.node.defaultRadius
            };

            const cssCode = `
/* Generated CSS from Groggy Template */
:root {
  --node-default-color: ${styles.nodeDefaultColor};
  --node-selected-color: ${styles.nodeSelectedColor};
  --node-hover-color: ${styles.nodeHoverColor};
  --edge-default-color: ${styles.edgeDefaultColor};
  --edge-default-width: ${styles.edgeDefaultWidth}px;
  --edge-curviness: ${styles.edgeCurviness};
  --node-default-radius: ${styles.nodeDefaultRadius}px;
}`;

            // Copy to clipboard
            navigator.clipboard.writeText(cssCode).then(() => {
                alert('CSS styles copied to clipboard!');
            }).catch(() => {
                // Fallback
                const textarea = document.createElement('textarea');
                textarea.value = cssCode;
                document.body.appendChild(textarea);
                textarea.select();
                document.execCommand('copy');
                document.body.removeChild(textarea);
                alert('CSS styles copied to clipboard!');
            });
        }'''

    def generate_css_playground(self):
        """Generate a CSS playground for rapid style testing"""
        print("\n🎨 Generating CSS playground...")

        playground_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Groggy CSS Playground</title>
    <style>
        body {
            font-family: system-ui, -apple-system, sans-serif;
            margin: 0;
            padding: 0;
            height: 100vh;
            display: grid;
            grid-template-columns: 350px 1fr;
            background: #f5f5f5;
        }

        .editor-panel {
            background: #2d3748;
            color: #e2e8f0;
            padding: 20px;
            overflow-y: auto;
        }

        .editor-panel h3 {
            margin: 0 0 15px 0;
            color: #4fd1c7;
        }

        .css-editor {
            width: 100%;
            height: 60vh;
            background: #1a202c;
            color: #e2e8f0;
            border: 1px solid #4a5568;
            border-radius: 4px;
            padding: 10px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 13px;
            line-height: 1.4;
            resize: vertical;
        }

        .apply-btn {
            margin-top: 10px;
            padding: 8px 16px;
            background: #4fd1c7;
            color: #1a202c;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 500;
        }

        .apply-btn:hover {
            background: #38b2ac;
        }

        .preview-panel {
            background: white;
            padding: 20px;
            overflow-y: auto;
        }

        .preset-buttons {
            margin-bottom: 15px;
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
        }

        .preset-btn {
            padding: 4px 8px;
            background: #4a5568;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
        }

        .preset-btn:hover {
            background: #2d3748;
        }
    </style>
</head>
<body>
    <div class="editor-panel">
        <h3>🎨 CSS Playground</h3>

        <div class="preset-buttons">
            <button class="preset-btn" onclick="loadPreset('dark')">Dark Theme</button>
            <button class="preset-btn" onclick="loadPreset('minimal')">Minimal</button>
            <button class="preset-btn" onclick="loadPreset('colorful')">Colorful</button>
            <button class="preset-btn" onclick="loadPreset('reset')">Reset</button>
        </div>

        <textarea id="css-editor" class="css-editor" placeholder="Enter your CSS here...">
/* Edit CSS here and click Apply to see changes */
:root {
  --node-default-color: #4dabf7;
  --node-selected-color: #ff6b6b;
  --node-hover-color: #339af0;
  --edge-default-color: #999;
  --edge-default-width: 1px;
  --line: #eee;
  --row-hover: #f3f6ff;
}

.groggy-table {
  border-radius: 8px;
  overflow: hidden;
}

.graph-canvas {
  border: 2px solid var(--line);
  border-radius: 12px;
}
        </textarea>

        <button class="apply-btn" onclick="applyCSS()">Apply CSS</button>

        <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #4a5568;">
            <h4 style="margin: 0 0 10px 0; color: #cbd5e0;">Quick Actions</h4>
            <button class="preset-btn" onclick="exportCSS()" style="display: block; width: 100%; margin-bottom: 5px;">📋 Copy CSS</button>
            <button class="preset-btn" onclick="loadTemplate()" style="display: block; width: 100%;">🔄 Load from Template</button>
        </div>
    </div>

    <div class="preview-panel">
        <iframe id="preview-frame" src="streaming_template.html" style="width: 100%; height: 100%; border: none; border-radius: 8px;"></iframe>
    </div>

    <script>
        function applyCSS() {
            const css = document.getElementById('css-editor').value;
            const iframe = document.getElementById('preview-frame');

            iframe.onload = function() {
                try {
                    const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;

                    // Remove existing custom styles
                    const existingStyle = iframeDoc.getElementById('playground-styles');
                    if (existingStyle) {
                        existingStyle.remove();
                    }

                    // Add new styles
                    const styleElement = iframeDoc.createElement('style');
                    styleElement.id = 'playground-styles';
                    styleElement.textContent = css;
                    iframeDoc.head.appendChild(styleElement);
                } catch (e) {
                    console.error('Error applying CSS:', e);
                }
            };

            // If already loaded, apply immediately
            if (iframe.contentDocument) {
                iframe.onload();
            }
        }

        function loadPreset(preset) {
            const editor = document.getElementById('css-editor');
            const presets = {
                dark: `/* Dark Theme */
:root {
  --bg: #1a1a1a;
  --fg: #e2e8f0;
  --line: #4a5568;
  --hover: #2d3748;
  --row-hover: #2a4365;
  --node-default-color: #63b3ed;
  --node-selected-color: #f56565;
  --edge-default-color: #a0aec0;
}

body {
  background: var(--bg);
  color: var(--fg);
}

.graph-canvas {
  background: #2d3748;
}`,
                minimal: `/* Minimal Theme */
:root {
  --node-default-color: #718096;
  --node-selected-color: #2d3748;
  --edge-default-color: #e2e8f0;
  --line: #f7fafc;
  --row-hover: #edf2f7;
}

.groggy-table {
  border: none;
  box-shadow: none;
}

.graph-canvas {
  border: 1px solid #e2e8f0;
  border-radius: 4px;
}`,
                colorful: `/* Colorful Theme */
:root {
  --node-default-color: #ed8936;
  --node-selected-color: #d53f8c;
  --node-hover-color: #38b2ac;
  --edge-default-color: #805ad5;
  --line: #fbb6ce;
  --row-hover: #fed7d7;
  --bg: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.graph-canvas {
  border: 3px solid #d53f8c;
  border-radius: 20px;
  background: linear-gradient(45deg, #fbb6ce, #fed7d7);
}`,
                reset: `/* Reset to Default */
:root {
  --node-default-color: #4dabf7;
  --node-selected-color: #ff6b6b;
  --node-hover-color: #339af0;
  --edge-default-color: #999;
  --line: #eee;
  --row-hover: #f3f6ff;
}`
            };

            editor.value = presets[preset] || presets.reset;
        }

        function exportCSS() {
            const css = document.getElementById('css-editor').value;
            navigator.clipboard.writeText(css).then(() => {
                alert('CSS copied to clipboard!');
            }).catch(() => {
                const textarea = document.createElement('textarea');
                textarea.value = css;
                document.body.appendChild(textarea);
                textarea.select();
                document.execCommand('copy');
                document.body.removeChild(textarea);
                alert('CSS copied to clipboard!');
            });
        }

        function loadTemplate() {
            // This would load CSS from the current template
            alert('Load from template - connect this to your template file');
        }

        // Auto-apply CSS when typing (debounced)
        let cssTimeout;
        document.getElementById('css-editor').addEventListener('input', function() {
            clearTimeout(cssTimeout);
            cssTimeout = setTimeout(applyCSS, 500);
        });

        // Apply initial CSS
        setTimeout(applyCSS, 1000);
    </script>
</body>
</html>'''

        output_file = self.output_dir / "css_playground.html"
        output_file.write_text(playground_content)
        print(f"   ✓ css_playground.html")

def main():
    """Main function"""
    generator = StreamingTemplateGenerator()
    generator.generate_all()

    print("\n🎯 Next Steps:")
    print("   1. Open streaming_template.html to see the interactive template")
    print("   2. Use css_playground.html for rapid CSS prototyping")
    print("   3. Experiment with the style controls and live editing")
    print("   4. Export styles and sync back to the Rust source")
    print("   5. Use browser dev tools for advanced styling")

if __name__ == "__main__":
    main()