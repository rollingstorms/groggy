use super::server::StreamingServer;
use super::types::StreamingResult;

// --- HTML generation for the interactive page ---
impl StreamingServer {
    pub async fn generate_interactive_html(&self) -> StreamingResult<String> {
        // Use default port (fallback for compatibility)
        self.generate_interactive_html_with_port(self.config.port).await
    }

    pub async fn generate_interactive_html_with_port(&self, port: u16) -> StreamingResult<String> {
        // Get initial data window for the table
        let initial_window = self.virtual_scroller.get_visible_window(self.data_source.as_ref())?;
        let total_rows = self.data_source.total_rows();
        let total_cols = self.data_source.total_cols();

        // Generate column headers
        let column_names = self.data_source.get_column_names();
        let mut headers = Vec::new();
        for (col_idx, name) in column_names.iter().enumerate() {
            headers.push(format!(
                r#"<th class="col-header" data-col="{}">{}</th>"#,
                col_idx, Self::html_escape(name)
            ));
        }
        let headers_html = headers.join("\n                        ");

        // Generate initial rows - Use AttrValue directly for HTML display
        let mut rows = Vec::new();
        for (row_idx, row) in initial_window.rows.iter().enumerate() {
            let mut cells = Vec::new();
            for (col_idx, cell_data) in row.iter().enumerate() {
                let display_value = super::util::attr_value_to_display_text(cell_data);
                cells.push(format!(
                    r#"<td class="cell" data-row="{}" data-col="{}">{}</td>"#,
                    initial_window.start_offset + row_idx,
                    col_idx,
                    Self::html_escape(&display_value)
                ));
            }
            let row_html = format!(
                r#"<tr class="data-row" data-row="{}">{}</tr>"#,
                initial_window.start_offset + row_idx,
                cells.join("")
            );
            rows.push(row_html);
        }
        let rows_html = rows.join("\n                        ");

        // Use the actual port the server is running on for WebSocket connection
        let ws_port = port;
        
        let html = format!(r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="groggy-run-id" content="{run_id}">
    <title>Groggy Interactive Visualization</title>
    <link rel="stylesheet" href="/css/sleek.css">
    <link rel="stylesheet" href="/css/graph_visualization.css">
</head>
<body>
    <div class="groggy-display-container" data-theme="sleek">
        <div class="table-container">
            <div class="table-header">
                <div class="table-title">🎨 Interactive Visualization</div>
                <div style="display: flex; align-items: center; gap: 16px;">
                    <div class="view-controls">
                        <div class="view-toggle">
                            <button id="table-view-btn" class="view-toggle-btn active">📊 Table</button>
                            <button id="graph-view-btn" class="view-toggle-btn">🕸️ Graph</button>
                        </div>
                    </div>
                    <div class="table-stats">{total_rows} rows × {total_cols} cols</div>
                    <div id="connection-status" class="connection-status status-disconnected">Connecting...</div>
                </div>
            </div>
            
            <div id="error-container"></div>
            
            <div class="viz-container">
                <!-- Table View -->
                <div id="table-view" class="table-view">
                    <table id="data-table" class="groggy-table theme-sleek">
                        <thead>
                            <tr>
                                {headers_html}
                            </tr>
                        </thead>
                        <tbody id="table-body">
                            {rows_html}
                        </tbody>
                    </table>
                    
                    <div id="loading" class="loading" style="display: none;">
                        Loading more data...
                    </div>
                </div>
                
                <!-- Graph View -->
                <div id="graph-view" class="graph-view" style="display: none;">
                    <canvas id="graph-canvas" class="graph-canvas"></canvas>
                    
                    <!-- Graph Controls -->
                    <div class="graph-controls">
                        <button id="zoom-in-btn" class="graph-btn" title="Zoom In">🔍+</button>
                        <button id="zoom-out-btn" class="graph-btn" title="Zoom Out">🔍-</button>
                        <button id="center-btn" class="graph-btn" title="Center Graph">🎯</button>
                        <button id="screenshot-btn" class="graph-btn" title="Screenshot">📷</button>
                    </div>
                    
                    <!-- Layout Controls -->
                    <div class="layout-controls">
                        <label for="layout-select">Layout:</label>
                        <select id="layout-select" class="layout-select">
                            <option value="force-directed">Force</option>
                            <option value="circular">Circular</option>
                            <option value="grid">Grid</option>
                            <option value="hierarchical">Tree</option>
                            <option value="honeycomb">Honeycomb</option>
                        </select>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let currentOffset = 0;
        let totalRows = {total_rows};
        let isConnected = false;
        
        // Graph visualization variables
        let graphData = {{ nodes: [], edges: [], metadata: {{}} }};
        let currentView = 'table';
        let canvas = null;
        let ctx = null;
        let scale = 1.0;
        let translateX = 0;
        let translateY = 0;
        let selectedNode = null;
        let hoveredNode = null;
        let hoveredEdge = null;
        let isDragging = false;
        let lastMouseX = 0;
        let lastMouseY = 0;
        let selectedNodes = new Set();
        let selectedEdges = new Set();

        // Canvas styling configuration with customizable parameters
        const canvasStyles = {{
            // Node styling
            node: {{
                defaultRadius: 8,
                minRadius: 3,
                maxRadius: 20,
                borderWidth: 2,
                borderHoverWidth: 3,
                colors: {{
                    default: '#4dabf7',
                    selected: '#ff6b6b',
                    hover: '#339af0',
                    border: '#333',
                    label: '#333'
                }},
                // Node size can be controlled by data attributes
                sizeProperty: 'size', // which property controls node size
                sizeScale: 1.0        // multiplier for node sizes
            }},
            // Edge styling
            edge: {{
                defaultWidth: 1,
                selectedWidth: 3,
                hoverWidth: 2,
                opacity: 0.8,
                hoverOpacity: 1.0,
                colors: {{
                    default: '#999',
                    selected: '#ff8cc8',
                    hover: '#666'
                }},
                // Edge curviness control
                curviness: 0.2,       // 0 = straight, 1 = very curved
                arrowSize: 6,         // size of arrow heads for directed edges
                animationSpeed: 0.02  // speed of edge animations
            }},
            // Animation and interaction
            animation: {{
                nodeHoverGrow: 1.3,   // how much nodes grow on hover
                edgePulse: true,      // whether edges pulse when selected
                transitionSpeed: 200  // ms for transitions
            }}
        }};
        
        function updateConnectionStatus(connected) {{
            const statusEl = document.getElementById('connection-status');
            isConnected = connected;
            
            if (connected) {{
                statusEl.textContent = 'Connected';
                statusEl.className = 'connection-status status-connected';
            }} else {{
                statusEl.textContent = 'Disconnected';
                statusEl.className = 'connection-status status-disconnected';
            }}
        }}
        
        function showError(message) {{
            const errorContainer = document.getElementById('error-container');
            errorContainer.innerHTML = `<div class="error">Error: ${{message}}</div>`;
        }}
        
        // Graph rendering functions
        function initGraphCanvas() {{
            canvas = document.getElementById('graph-canvas');
            if (!canvas) return;
            
            ctx = canvas.getContext('2d');
            
            // Set canvas size
            const container = canvas.parentElement;
            canvas.width = container.clientWidth;
            canvas.height = container.clientHeight;
            
            // Add event listeners
            canvas.addEventListener('mousedown', onCanvasMouseDown);
            canvas.addEventListener('mousemove', onCanvasMouseMove);
            canvas.addEventListener('mouseup', onCanvasMouseUp);
            canvas.addEventListener('wheel', onCanvasWheel);
            
            // Initial render
            renderGraph();
        }}
        
        function getNodeRadius(node) {{
            // Calculate node radius based on size property and styling config
            let baseRadius = canvasStyles.node.defaultRadius;

            if (node[canvasStyles.node.sizeProperty]) {{
                baseRadius = Math.max(
                    canvasStyles.node.minRadius,
                    Math.min(
                        canvasStyles.node.maxRadius,
                        node[canvasStyles.node.sizeProperty] * canvasStyles.node.sizeScale
                    )
                );
            }}

            // Apply hover growth
            if (hoveredNode === node.id) {{
                baseRadius *= canvasStyles.animation.nodeHoverGrow;
            }}

            return baseRadius;
        }}

        function drawCurvedEdge(sourceNode, targetNode, curviness = 0) {{
            if (curviness === 0) {{
                // Straight line
                ctx.beginPath();
                ctx.moveTo(sourceNode.x || 0, sourceNode.y || 0);
                ctx.lineTo(targetNode.x || 0, targetNode.y || 0);
                ctx.stroke();
                return;
            }}

            const sx = sourceNode.x || 0;
            const sy = sourceNode.y || 0;
            const tx = targetNode.x || 0;
            const ty = targetNode.y || 0;

            // Calculate control point for curved edge
            const midX = (sx + tx) / 2;
            const midY = (sy + ty) / 2;

            // Perpendicular offset for curve
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
        }}

        function renderGraph() {{
            if (!ctx || !canvas) return;

            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Save transform state
            ctx.save();

            // Apply zoom and pan
            ctx.translate(canvas.width / 2 + translateX, canvas.height / 2 + translateY);
            ctx.scale(scale, scale);

            // Render edges first (behind nodes)
            graphData.edges.forEach(edge => {{
                const sourceNode = graphData.nodes.find(n => n.id === edge.source);
                const targetNode = graphData.nodes.find(n => n.id === edge.target);

                if (sourceNode && targetNode) {{
                    // Determine edge styling
                    const isSelected = selectedEdges.has(edge.id);
                    const isHovered = hoveredEdge === edge.id;

                    let edgeWidth = canvasStyles.edge.defaultWidth;
                    let edgeColor = canvasStyles.edge.colors.default;
                    let edgeOpacity = canvasStyles.edge.opacity;

                    if (isSelected) {{
                        edgeWidth = canvasStyles.edge.selectedWidth;
                        edgeColor = canvasStyles.edge.colors.selected;
                        edgeOpacity = 1.0;
                    }} else if (isHovered) {{
                        edgeWidth = canvasStyles.edge.hoverWidth;
                        edgeColor = canvasStyles.edge.colors.hover;
                        edgeOpacity = canvasStyles.edge.hoverOpacity;
                    }}

                    // Apply edge weight if available
                    if (edge.weight && edge.weight > 1) {{
                        edgeWidth *= Math.min(3, Math.sqrt(edge.weight));
                    }}

                    // Check if this is a meta edge
                    const isMetaEdge = edge.entity_type === 'meta' || edge.contains_subgraph !== undefined;
                    if (isMetaEdge) {{
                        edgeColor = '#9775fa'; // Purple color for meta edges
                        edgeWidth = Math.max(edgeWidth, 3); // Thicker width
                        edgeOpacity = Math.min(1.0, edgeOpacity + 0.1); // More opaque

                        // Set up dashed line for meta edges
                        ctx.setLineDash([8, 4]);
                    }} else {{
                        // Reset to solid line for regular edges
                        ctx.setLineDash([]);
                    }}

                    ctx.strokeStyle = edgeColor;
                    ctx.lineWidth = edgeWidth;
                    ctx.globalAlpha = edgeOpacity;

                    // Draw curved or straight edge based on curviness setting
                    drawCurvedEdge(sourceNode, targetNode, canvasStyles.edge.curviness);

                    // Reset drawing state
                    ctx.setLineDash([]); // Reset line dash
                    ctx.globalAlpha = 1.0; // Reset alpha
                }}
            }});

            // Render nodes
            graphData.nodes.forEach(node => {{
                const x = node.x || 0;
                const y = node.y || 0;
                const radius = getNodeRadius(node);

                // Determine node styling
                const isSelected = selectedNodes.has(node.id);
                const isHovered = hoveredNode === node.id;

                let nodeColor = canvasStyles.node.colors.default;
                let borderWidth = canvasStyles.node.borderWidth;

                if (isSelected) {{
                    nodeColor = canvasStyles.node.colors.selected;
                    borderWidth = canvasStyles.node.borderHoverWidth;
                }} else if (isHovered) {{
                    nodeColor = canvasStyles.node.colors.hover;
                    borderWidth = canvasStyles.node.borderHoverWidth;
                }}

                // Apply node type coloring if available
                if (node.type) {{
                    const typeColor = canvasStyles.node.colors[node.type];
                    if (typeColor) nodeColor = typeColor;
                }}

                // Check if this is a meta node (contains a subgraph)
                const isMeta = node.entity_type === 'meta' || node.contains_subgraph !== undefined;
                let nodeRadius = radius;
                let shadowColor = 'rgba(0,0,0,0.2)';
                let shadowBlur = isSelected || isHovered ? 8 : 4;

                if (isMeta) {{
                    // Meta nodes get special styling
                    nodeColor = '#9775fa'; // Purple color for meta nodes
                    nodeRadius = Math.max(radius, 14); // Larger radius
                    borderWidth = Math.max(borderWidth, 3); // Thicker border
                    shadowColor = 'rgba(144, 117, 250, 0.3)'; // Purple shadow
                    shadowBlur = isSelected || isHovered ? 12 : 8; // More prominent shadow
                }}

                // Node circle with shadow for depth
                ctx.shadowColor = shadowColor;
                ctx.shadowBlur = shadowBlur;
                ctx.shadowOffsetX = 1;
                ctx.shadowOffsetY = 1;

                ctx.beginPath();
                ctx.arc(x, y, nodeRadius, 0, 2 * Math.PI);
                ctx.fillStyle = nodeColor;
                ctx.fill();

                // Reset shadow
                ctx.shadowColor = 'transparent';
                ctx.shadowBlur = 0;
                ctx.shadowOffsetX = 0;
                ctx.shadowOffsetY = 0;

                // Node border
                if (isMeta) {{
                    // Meta nodes get a special purple border
                    ctx.strokeStyle = '#7048e8';
                    ctx.lineWidth = borderWidth;
                    ctx.stroke();

                    // Add an inner ring to indicate it's a meta node
                    ctx.beginPath();
                    ctx.arc(x, y, nodeRadius - 2, 0, 2 * Math.PI);
                    ctx.strokeStyle = '#9775fa';
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }} else {{
                    ctx.strokeStyle = canvasStyles.node.colors.border;
                    ctx.lineWidth = borderWidth;
                    ctx.stroke();
                }}

                // Node label
                if (node.label || node.id) {{
                    if (isMeta) {{
                        // Meta nodes get bold purple labels
                        ctx.fillStyle = '#6741d9';
                        ctx.font = 'bold 10px Arial';
                    }} else {{
                        ctx.fillStyle = canvasStyles.node.colors.label;
                        ctx.font = '10px Arial';
                    }}
                    ctx.textAlign = 'center';
                    ctx.fillText(node.label || node.id, x, y + nodeRadius + 12);
                }}
            }});

            // Restore transform state
            ctx.restore();
        }}
        
        function updateGraphData(data) {{
            graphData = data;
            
            // If no positions, generate simple circular layout
            if (graphData.nodes.length > 0 && !graphData.nodes[0].x) {{
                const centerX = 0;
                const centerY = 0;
                const radius = 100;
                const angleStep = (2 * Math.PI) / graphData.nodes.length;
                
                graphData.nodes.forEach((node, i) => {{
                    const angle = i * angleStep;
                    node.x = centerX + radius * Math.cos(angle);
                    node.y = centerY + radius * Math.sin(angle);
                }});
            }}
            
            renderGraph();
        }}
        
        function getNodeAtPosition(worldX, worldY) {{
            for (const node of graphData.nodes) {{
                const dx = worldX - (node.x || 0);
                const dy = worldY - (node.y || 0);
                const distance = Math.sqrt(dx * dx + dy * dy);
                const nodeRadius = getNodeRadius(node);
                if (distance < nodeRadius + 2) {{
                    return node;
                }}
            }}
            return null;
        }}

        function getEdgeAtPosition(worldX, worldY) {{
            const threshold = 5; // Distance threshold for edge selection
            for (const edge of graphData.edges) {{
                const sourceNode = graphData.nodes.find(n => n.id === edge.source);
                const targetNode = graphData.nodes.find(n => n.id === edge.target);

                if (sourceNode && targetNode) {{
                    const sx = sourceNode.x || 0;
                    const sy = sourceNode.y || 0;
                    const tx = targetNode.x || 0;
                    const ty = targetNode.y || 0;

                    // Calculate distance from point to line segment
                    const A = worldX - sx;
                    const B = worldY - sy;
                    const C = tx - sx;
                    const D = ty - sy;

                    const dot = A * C + B * D;
                    const lenSq = C * C + D * D;
                    let param = -1;
                    if (lenSq !== 0) param = dot / lenSq;

                    let xx, yy;
                    if (param < 0) {{
                        xx = sx;
                        yy = sy;
                    }} else if (param > 1) {{
                        xx = tx;
                        yy = ty;
                    }} else {{
                        xx = sx + param * C;
                        yy = sy + param * D;
                    }}

                    const dx = worldX - xx;
                    const dy = worldY - yy;
                    const distance = Math.sqrt(dx * dx + dy * dy);

                    if (distance < threshold) {{
                        return edge;
                    }}
                }}
            }}
            return null;
        }}

        function onCanvasMouseDown(e) {{
            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;

            lastMouseX = mouseX;
            lastMouseY = mouseY;
            isDragging = true;

            // Convert to world coordinates
            const worldX = (mouseX - canvas.width / 2 - translateX) / scale;
            const worldY = (mouseY - canvas.height / 2 - translateY) / scale;

            // Check for node selection
            const clickedNode = getNodeAtPosition(worldX, worldY);
            const clickedEdge = getEdgeAtPosition(worldX, worldY);

            if (clickedNode) {{
                // Handle node selection
                if (e.ctrlKey || e.metaKey) {{
                    // Multi-select with Ctrl/Cmd
                    if (selectedNodes.has(clickedNode.id)) {{
                        selectedNodes.delete(clickedNode.id);
                    }} else {{
                        selectedNodes.add(clickedNode.id);
                    }}
                }} else {{
                    // Single select
                    selectedNodes.clear();
                    selectedNodes.add(clickedNode.id);
                }}
                selectedNode = clickedNode.id;
            }} else if (clickedEdge) {{
                // Handle edge selection
                if (e.ctrlKey || e.metaKey) {{
                    if (selectedEdges.has(clickedEdge.id)) {{
                        selectedEdges.delete(clickedEdge.id);
                    }} else {{
                        selectedEdges.add(clickedEdge.id);
                    }}
                }} else {{
                    selectedEdges.clear();
                    selectedEdges.add(clickedEdge.id);
                }}
                selectedNode = null;
            }} else {{
                // Clear selection
                if (!(e.ctrlKey || e.metaKey)) {{
                    selectedNodes.clear();
                    selectedEdges.clear();
                }}
                selectedNode = null;
            }}

            canvas.classList.add('is-dragging');
            renderGraph();
        }}
        
        function onCanvasMouseMove(e) {{
            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;

            // Convert to world coordinates for hover detection
            const worldX = (mouseX - canvas.width / 2 - translateX) / scale;
            const worldY = (mouseY - canvas.height / 2 - translateY) / scale;

            if (!isDragging) {{
                // Handle hover effects when not dragging
                const hoveredNodeCandidate = getNodeAtPosition(worldX, worldY);
                const hoveredEdgeCandidate = getEdgeAtPosition(worldX, worldY);

                let cursorStyle = 'grab';
                let needsRedraw = false;

                // Update hovered node
                if (hoveredNodeCandidate && hoveredNodeCandidate.id !== hoveredNode) {{
                    hoveredNode = hoveredNodeCandidate.id;
                    hoveredEdge = null;
                    cursorStyle = 'pointer';
                    needsRedraw = true;
                }} else if (!hoveredNodeCandidate && hoveredNode !== null) {{
                    hoveredNode = null;
                    needsRedraw = true;
                }}

                // Update hovered edge (only if no node is hovered)
                if (!hoveredNodeCandidate) {{
                    if (hoveredEdgeCandidate && hoveredEdgeCandidate.id !== hoveredEdge) {{
                        hoveredEdge = hoveredEdgeCandidate.id;
                        cursorStyle = 'pointer';
                        needsRedraw = true;
                    }} else if (!hoveredEdgeCandidate && hoveredEdge !== null) {{
                        hoveredEdge = null;
                        needsRedraw = true;
                    }}
                }}

                canvas.style.cursor = cursorStyle;

                if (needsRedraw) {{
                    renderGraph();
                }}
                return;
            }}

            // Handle dragging
            const deltaX = mouseX - lastMouseX;
            const deltaY = mouseY - lastMouseY;

            if (selectedNode !== null) {{
                // Drag the selected node
                const node = graphData.nodes.find(n => n.id === selectedNode);
                if (node) {{
                    // Convert screen space delta to world space
                    node.x = (node.x || 0) + deltaX / scale;
                    node.y = (node.y || 0) + deltaY / scale;
                }}
            }} else {{
                // Drag the viewport
                translateX += deltaX;
                translateY += deltaY;
            }}

            lastMouseX = mouseX;
            lastMouseY = mouseY;

            renderGraph();
        }}
        
        function onCanvasMouseUp(e) {{
            isDragging = false;
            selectedNode = null; // Clear selection after dragging
            canvas.classList.remove('is-dragging');
            canvas.style.cursor = 'grab';
        }}
        
        function onCanvasWheel(e) {{
            e.preventDefault();
            
            const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
            scale = Math.max(0.1, Math.min(5.0, scale * zoomFactor));
            
            renderGraph();
        }}
        
        function requestGraphData() {{
            if (!ws || !isConnected) return;
            
            const message = {{
                type: 'GraphDataRequest',
                layout_algorithm: document.getElementById('layout-select').value,
                theme: 'sleek'
            }};
            
            ws.send(JSON.stringify(message));
        }}
        
        function connectWebSocket() {{
            const wsUrl = `ws://127.0.0.1:{ws_port}`;
            console.log('Connecting to WebSocket:', wsUrl);
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function(event) {{
                console.log('WebSocket connected');
                updateConnectionStatus(true);
                document.getElementById('error-container').innerHTML = '';
            }};
            
            ws.onmessage = function(event) {{
                try {{
                    const message = JSON.parse(event.data);
                    console.log('Received message:', message);
                    
                    if (message.type === 'InitialData') {{
                        updateTable(message.window, message.total_rows);
                        totalRows = message.total_rows;
                    }} else if (message.type === 'DataUpdate') {{
                        updateTable(message.new_window, totalRows);
                        currentOffset = message.offset;
                    }} else if (message.type === 'GraphDataResponse') {{
                        // Handle graph data from WebSocket
                        const graphDataFormatted = {{
                            nodes: message.nodes || [],
                            edges: message.edges || [],
                            metadata: message.metadata || {{}}
                        }};
                        
                        // Apply layout positions if provided
                        if (message.layout_positions) {{
                            message.layout_positions.forEach(pos => {{
                                const node = graphDataFormatted.nodes.find(n => n.id === pos.node_id);
                                if (node) {{
                                    node.x = pos.position.x;
                                    node.y = pos.position.y;
                                }}
                            }});
                        }}
                        
                        updateGraphData(graphDataFormatted);
                        console.log('Updated graph with', graphDataFormatted.nodes.length, 'nodes and', graphDataFormatted.edges.length, 'edges');
                    }} else if (message.type === 'LayoutResponse') {{
                        // Handle layout update from WebSocket
                        if (message.positions) {{
                            message.positions.forEach(pos => {{
                                const node = graphData.nodes.find(n => n.id === pos.node_id);
                                if (node) {{
                                    node.x = pos.position.x;
                                    node.y = pos.position.y;
                                }}
                            }});
                            renderGraph();
                        }}
                    }}
                }} catch (e) {{
                    console.error('Failed to parse WebSocket message:', e);
                    showError('Failed to parse server message');
                }}
            }};
            
            ws.onclose = function(event) {{
                console.log('WebSocket disconnected');
                updateConnectionStatus(false);
                
                // Attempt to reconnect after 3 seconds
                setTimeout(() => {{
                    if (!isConnected) {{
                        connectWebSocket();
                    }}
                }}, 3000);
            }};
            
            ws.onerror = function(error) {{
                console.error('WebSocket error:', error);
                showError('WebSocket connection failed');
                updateConnectionStatus(false);
            }};
        }}
        
        function updateTable(dataWindow, total) {{
            const tbody = document.getElementById('table-body');
            tbody.innerHTML = '';
            
            dataWindow.rows.forEach((row, rowIdx) => {{
                const tr = document.createElement('tr');
                tr.className = 'data-row';
                tr.dataset.row = dataWindow.offset + rowIdx;
                
                row.forEach((cell, colIdx) => {{
                    const td = document.createElement('td');
                    td.className = 'cell';
                    td.dataset.row = dataWindow.offset + rowIdx;
                    td.dataset.col = colIdx;
                    td.textContent = cell;
                    tr.appendChild(td);
                }});
                
                tbody.appendChild(tr);
            }});
            
            // Update stats
            const statsEl = document.querySelector('.table-stats');
            statsEl.textContent = `${{total}} rows × {total_cols} cols`;
        }}
        
        function requestScroll(offset, windowSize = 50) {{
            if (ws && ws.readyState === WebSocket.OPEN) {{
                const message = {{
                    type: 'ScrollRequest',
                    offset: offset,
                    window_size: windowSize
                }};
                ws.send(JSON.stringify(message));
            }}
        }}
        
        // Virtual scrolling support
        const tableContainer = document.querySelector('.table-container');
        let scrollTimeout = null;
        
        tableContainer.addEventListener('scroll', function() {{
            clearTimeout(scrollTimeout);
            scrollTimeout = setTimeout(() => {{
                const scrollTop = tableContainer.scrollTop;
                const rowHeight = 45; // Approximate row height
                const newOffset = Math.floor(scrollTop / rowHeight);
                
                if (Math.abs(newOffset - currentOffset) > 10) {{
                    requestScroll(newOffset);
                }}
            }}, 150);
        }});
        
        // View toggle functionality
        function switchView(view) {{
            currentView = view;
            const tableView = document.getElementById('table-view');
            const graphView = document.getElementById('graph-view');
            const graphControls = document.querySelector('.graph-controls');
            const layoutControls = document.querySelector('.layout-controls');

            // Update button states
            document.querySelectorAll('.view-toggle-btn').forEach(btn => btn.classList.remove('active'));

            if (view === 'table') {{
                // Show table, hide graph
                tableView.style.display = 'block';
                graphView.style.display = 'none';
                document.getElementById('table-view-btn').classList.add('active');

                // Freeze graph controls
                if (graphControls) graphControls.style.pointerEvents = 'none';
                if (layoutControls) layoutControls.style.pointerEvents = 'none';

                // Remove canvas event listeners to prevent interaction
                if (canvas) {{
                    canvas.style.pointerEvents = 'none';
                }}

            }} else if (view === 'graph') {{
                // Show graph, hide table
                tableView.style.display = 'none';
                graphView.style.display = 'block';
                document.getElementById('graph-view-btn').classList.add('active');

                // Unfreeze graph controls
                if (graphControls) graphControls.style.pointerEvents = 'auto';
                if (layoutControls) layoutControls.style.pointerEvents = 'auto';

                // Re-enable canvas interaction
                if (canvas) {{
                    canvas.style.pointerEvents = 'auto';
                }}

                // Initialize canvas when switching to graph view
                setTimeout(initGraphCanvas, 100);
                // Request graph data if we don't have it
                if (graphData.nodes.length === 0) {{
                    requestGraphData();
                }}
            }}
        }}
        
        // Graph controls functionality
        function setupGraphControls() {{
            // Zoom controls
            document.getElementById('zoom-in-btn')?.addEventListener('click', () => {{
                scale = Math.min(5.0, scale * 1.2);
                renderGraph();
            }});
            
            document.getElementById('zoom-out-btn')?.addEventListener('click', () => {{
                scale = Math.max(0.1, scale * 0.8);
                renderGraph();
            }});
            
            document.getElementById('center-btn')?.addEventListener('click', () => {{
                translateX = 0;
                translateY = 0;
                scale = 1.0;
                renderGraph();
            }});
            
            // Layout change
            document.getElementById('layout-select')?.addEventListener('change', (e) => {{
                requestGraphData();
            }});
            
            // View toggle buttons
            document.getElementById('table-view-btn')?.addEventListener('click', () => switchView('table'));
            document.getElementById('graph-view-btn')?.addEventListener('click', () => switchView('graph'));
        }}
        
        // Initialize everything when page loads
        window.addEventListener('load', () => {{
            connectWebSocket();
            setupGraphControls();
            // Start with table view
            switchView('table');
        }});
    </script>
</body>
</html>"#,
            total_rows = total_rows,
            total_cols = total_cols,
            headers_html = headers_html,
            rows_html = rows_html,
            ws_port = ws_port,
            run_id = self.run_id
        );

        Ok(html)
    }


    /// HTML escape utility function
    fn html_escape(input: &str) -> String {
        input
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&#39;")
    }
}