//! Client HTML and JavaScript for realtime visualization
//!
//! Provides the minimal HTML/JS canvas and controls shell for Phase 2.

/// Generate the client HTML page with WebSocket connectivity and canvas
pub fn generate_client_html(port: u16) -> String {
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Groggy Realtime Visualization</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f8f9fa;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .controls {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .controls-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .control-panel {{
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 15px;
            background: #f8f9fa;
        }}
        .panel-title {{
            font-weight: 600;
            margin-bottom: 15px;
            color: #495057;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .search-box {{
            width: 100%;
            padding: 8px 12px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            font-size: 14px;
            margin-bottom: 10px;
        }}
        .filter-group {{
            margin-bottom: 12px;
        }}
        .filter-label {{
            display: block;
            font-size: 12px;
            font-weight: 500;
            color: #6c757d;
            margin-bottom: 4px;
        }}
        .range-input {{
            width: 100%;
            margin: 5px 0;
        }}
        .range-values {{
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: #6c757d;
        }}
        .performance-stats {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-top: 10px;
        }}
        .stat-box {{
            background: white;
            padding: 8px;
            border-radius: 4px;
            text-align: center;
            border: 1px solid #e9ecef;
        }}
        .stat-value {{
            font-size: 18px;
            font-weight: bold;
            color: #007bff;
        }}
        .stat-label {{
            font-size: 11px;
            color: #6c757d;
            text-transform: uppercase;
        }}
        .attribute-panel {{
            position: absolute;
            top: 10px;
            right: 10px;
            width: 300px;
            background: white;
            border: 1px solid #ddd;
            border-radius: 6px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            padding: 15px;
            display: none;
            z-index: 1000;
        }}
        .attribute-panel.visible {{
            display: block;
        }}
        .attribute-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }}
        .attribute-table th,
        .attribute-table td {{
            padding: 4px 8px;
            text-align: left;
            border-bottom: 1px solid #e9ecef;
        }}
        .attribute-table th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        .play-pause-btn {{
            background: #28a745;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: 500;
            cursor: pointer;
            margin-right: 10px;
        }}
        .play-pause-btn.paused {{
            background: #ffc107;
        }}
        .highlighted {{
            stroke: #ff6b6b !important;
            stroke-width: 3px !important;
        }}
        .selected {{
            fill: #ff6b6b !important;
        }}
        .canvas-container {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        #canvas {{
            width: 100%;
            height: 600px;
            border: none;
            display: block;
        }}
        .status {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }}
        .status-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #dc3545;
        }}
        .status-dot.connected {{
            background: #28a745;
        }}
        .control-group {{
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 15px;
        }}
        .control-group label {{
            font-weight: 500;
            min-width: 120px;
        }}
        select, input[type="range"] {{
            padding: 5px 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .honeycomb-badge {{
            background: linear-gradient(45deg, #ffd700, #ffed4a);
            color: #333;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }}
        .stats {{
            display: flex;
            gap: 20px;
            font-size: 14px;
            color: #666;
        }}
        #messages {{
            height: 150px;
            overflow-y: auto;
            background: #f8f9fa;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            font-family: monospace;
            font-size: 12px;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1 style="margin: 0; color: #333;">🍯 Honeycomb N-Dimensional Visualization</h1>
                    <p style="margin: 5px 0 0 0; color: #666;">Realtime backend with advanced rotation controls</p>
                </div>
                <div class="honeycomb-badge">REALTIME</div>
            </div>

            <div class="status">
                <div class="status-dot" id="status-dot"></div>
                <span id="status-text">Connecting...</span>
                <div class="stats">
                    <span>Port: {port}</span>
                    <span id="node-count">Nodes: 0</span>
                    <span id="edge-count">Edges: 0</span>
                    <span id="client-count">Clients: 0</span>
                </div>
            </div>
        </div>

        <div class="controls">
            <div class="controls-grid">
                <!-- Embedding & Layout Controls -->
                <div class="control-panel">
                    <div class="panel-title">🧠 Embedding & Layout</div>

                    <div class="control-group">
                        <label>Embedding Method:</label>
                        <select id="embedding-method">
                            <option value="default">Default</option>
                            <option value="pca">PCA</option>
                            <option value="umap">UMAP</option>
                            <option value="tsne">t-SNE</option>
                            <option value="force_directed">Force Directed</option>
                        </select>
                    </div>

                    <div class="control-group">
                        <label>Dimensions:</label>
                        <select id="dimensions">
                            <option value="2">2D</option>
                            <option value="3">3D</option>
                            <option value="4">4D</option>
                            <option value="5" selected>5D</option>
                            <option value="6">6D</option>
                            <option value="8">8D</option>
                            <option value="10">10D</option>
                        </select>
                    </div>

                    <div class="control-group">
                        <label>Layout Algorithm:</label>
                        <select id="layout-algorithm">
                            <option value="honeycomb" selected>Honeycomb</option>
                            <option value="force_directed">Force Directed</option>
                            <option value="circular">Circular</option>
                            <option value="grid">Grid</option>
                        </select>
                    </div>

                    <div class="control-group">
                        <button id="play-pause-btn" class="play-pause-btn">⏸️ Pause Simulation</button>
                        <span style="font-size: 12px; color: #666;">Toggle live updates</span>
                    </div>
                </div>

                <!-- Filtering & Search -->
                <div class="control-panel">
                    <div class="panel-title">🔍 Filtering & Search</div>

                    <input type="text" id="search-box" class="search-box" placeholder="Search nodes by ID or attributes...">

                    <div class="filter-group">
                        <label class="filter-label">Degree Range</label>
                        <input type="range" id="degree-min" class="range-input" min="0" max="100" value="0">
                        <input type="range" id="degree-max" class="range-input" min="0" max="100" value="100">
                        <div class="range-values">
                            <span id="degree-min-val">0</span>
                            <span id="degree-max-val">100</span>
                        </div>
                    </div>

                    <div class="filter-group">
                        <label class="filter-label">Attribute Filter</label>
                        <select id="attribute-name">
                            <option value="">Select attribute...</option>
                        </select>
                        <input type="text" id="attribute-value" placeholder="Filter value..." style="margin-top: 5px; width: 100%; padding: 4px;">
                    </div>

                    <div class="filter-group">
                        <label class="filter-label">Node Visibility</label>
                        <input type="range" id="node-opacity" class="range-input" min="0" max="100" value="100">
                        <div class="range-values">
                            <span>Transparent</span>
                            <span id="opacity-val">100%</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Performance Stats -->
            <div class="performance-stats">
                <div class="stat-box">
                    <div class="stat-value" id="fps-display">60</div>
                    <div class="stat-label">FPS</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="update-rate">30</div>
                    <div class="stat-label">Updates/sec</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="latency">5</div>
                    <div class="stat-label">Latency (ms)</div>
                </div>
            </div>

            <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #eee;">
                <h4 style="margin: 0 0 10px 0;">🎮 N-Dimensional Interaction</h4>
                <div style="font-size: 12px; color: #666; line-height: 1.4;">
                    <div><strong>Left Mouse + Drag:</strong> Rotate dimensions 0-1 (primary plane)</div>
                    <div><strong>Ctrl + Drag:</strong> Rotate higher dimensions (2-3, 4-5, etc.)</div>
                    <div><strong>Right Mouse + Drag:</strong> Multi-dimensional rotation</div>
                    <div><strong>Shift + Drag:</strong> Cross-dimensional rotation</div>
                    <div><strong>Click Node:</strong> Show attributes and neighbors</div>
                </div>
            </div>
        </div>

        <div class="canvas-container" style="position: relative;">
            <canvas id="canvas"></canvas>

            <!-- Floating Attribute Panel -->
            <div id="attribute-panel" class="attribute-panel">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <h5 style="margin: 0; color: #495057;">Node Details</h5>
                    <button onclick="hideAttributePanel()" style="background: none; border: none; font-size: 16px; cursor: pointer;">✕</button>
                </div>
                <table class="attribute-table">
                    <thead>
                        <tr>
                            <th>Property</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody id="attribute-tbody">
                        <tr><td colspan="2">Click a node to view details</td></tr>
                    </tbody>
                </table>
                <div style="margin-top: 10px; font-size: 11px; color: #6c757d;">
                    <div><strong>Neighbors:</strong> <span id="neighbor-count">0</span></div>
                    <div><strong>Degree:</strong> <span id="node-degree">0</span></div>
                </div>
            </div>
        </div>

        <div id="messages"></div>
    </div>

    <script>
        class RealtimeViz {{
            constructor() {{
                this.ws = null;
                this.canvas = document.getElementById('canvas');
                this.ctx = this.canvas.getContext('2d');
                this.nodes = [];
                this.edges = [];
                this.positions = [];

                // State management
                this.isPlaying = true;
                this.selectedNode = null;
                this.highlightedNodes = new Set();
                this.filteredNodes = new Set();
                this.searchResults = new Set();

                // Performance tracking
                this.frameCount = 0;
                this.lastFrameTime = performance.now();
                this.fps = 60;
                this.updateCount = 0;
                this.lastUpdateTime = performance.now();
                this.updateRate = 0;
                this.latency = 0;

                // Filter settings
                this.filters = {{
                    degreeMin: 0,
                    degreeMax: 100,
                    attributeName: '',
                    attributeValue: '',
                    opacity: 1.0,
                    searchQuery: ''
                }};

                this.initCanvas();
                this.initWebSocket();
                this.initControls();
                this.startRenderLoop();
                this.startPerformanceMonitoring();
            }}

            initCanvas() {{
                // Set canvas size
                const container = this.canvas.parentElement;
                this.canvas.width = container.clientWidth;
                this.canvas.height = 600;

                // Handle resize
                window.addEventListener('resize', () => {{
                    this.canvas.width = container.clientWidth;
                    this.canvas.height = 600;
                    this.render(true);
                }});

                // Mouse event handlers for n-dimensional controls
                this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
                this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
                this.canvas.addEventListener('mouseup', this.onMouseUp.bind(this));
                this.canvas.addEventListener('click', this.onCanvasClick.bind(this));
                this.canvas.addEventListener('contextmenu', e => e.preventDefault());

                this.mouseDown = false;
                this.isDragging = false;
                this.dragStart = {{ x: 0, y: 0 }};
                this.dragMode = 'none';
                this.pendingDragMode = 'none';
                this.dragThreshold = 5; // pixels
            }}

            initWebSocket() {{
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${{protocol}}//${{window.location.hostname}}:{port}/realtime/ws`;

                this.log(`Connecting to ${{wsUrl}}`);
                this.ws = new WebSocket(wsUrl);

                this.ws.onopen = () => {{
                    this.log('🔗 WebSocket connected');
                    this.updateStatus(true, 'Connected');
                }};

                this.ws.onmessage = (event) => {{
                    try {{
                        const message = JSON.parse(event.data);
                        this.handleMessage(message);
                    }} catch (e) {{
                        this.log(`❌ Failed to parse message: ${{e}}`);
                    }}
                }};

                this.ws.onclose = () => {{
                    this.log('🔌 WebSocket disconnected');
                    this.updateStatus(false, 'Disconnected');

                    // Auto-reconnect after 3 seconds
                    setTimeout(() => this.initWebSocket(), 3000);
                }};

                this.ws.onerror = (error) => {{
                    this.log(`❌ WebSocket error: ${{error}}`);
                    this.updateStatus(false, 'Error');
                }};
            }}

            initControls() {{
                // Embedding controls
                document.getElementById('embedding-method').addEventListener('change', (e) => {{
                    this.sendControl('ChangeEmbedding', {{
                        method: e.target.value,
                        k: parseInt(document.getElementById('dimensions').value),
                        params: {{}}
                    }});
                }});

                document.getElementById('dimensions').addEventListener('change', (e) => {{
                    this.sendControl('ChangeEmbedding', {{
                        method: document.getElementById('embedding-method').value,
                        k: parseInt(e.target.value),
                        params: {{}}
                    }});
                }});

                document.getElementById('layout-algorithm').addEventListener('change', (e) => {{
                    this.sendControl('ChangeLayout', {{
                        algorithm: e.target.value,
                        params: {{}}
                    }});
                }});

                // Play/pause control
                document.getElementById('play-pause-btn').addEventListener('click', () => {{
                    this.togglePlayPause();
                }});

                // Search and filter controls
                document.getElementById('search-box').addEventListener('input', (e) => {{
                    this.filters.searchQuery = e.target.value;
                    this.applyFilters();
                }});

                // Degree range filters
                document.getElementById('degree-min').addEventListener('input', (e) => {{
                    this.filters.degreeMin = parseInt(e.target.value);
                    document.getElementById('degree-min-val').textContent = e.target.value;
                    this.applyFilters();
                }});

                document.getElementById('degree-max').addEventListener('input', (e) => {{
                    this.filters.degreeMax = parseInt(e.target.value);
                    document.getElementById('degree-max-val').textContent = e.target.value;
                    this.applyFilters();
                }});

                // Attribute filter
                document.getElementById('attribute-name').addEventListener('change', (e) => {{
                    this.filters.attributeName = e.target.value;
                    this.applyFilters();
                }});

                document.getElementById('attribute-value').addEventListener('input', (e) => {{
                    this.filters.attributeValue = e.target.value;
                    this.applyFilters();
                }});

                // Opacity control
                document.getElementById('node-opacity').addEventListener('input', (e) => {{
                    this.filters.opacity = parseInt(e.target.value) / 100;
                    document.getElementById('opacity-val').textContent = e.target.value + '%';
                }});
            }}

            handleMessage(message) {{
                this.log(`📨 ${{message.type}}: ${{message.payload ? 'data received' : 'no payload'}}`);

                switch (message.type) {{
                    case 'snapshot':
                        this.handleSnapshot(message.payload);
                        break;
                    case 'update':
                        this.handleUpdate(message.payload);
                        break;
                    case 'control_ack':
                        // Handle control acknowledgment - success and message are at top level, not in payload
                        const success = message.success !== undefined ? message.success : false;
                        const ackMessage = message.message || 'No message';
                        this.log(`📨 control_ack: ${{success ? 'success' : 'failed'}} - ${{ackMessage}}`);
                        break;
                }}
            }}

            handleSnapshot(snapshot) {{
                this.nodes = snapshot.nodes || [];
                this.edges = snapshot.edges || [];
                this.positions = snapshot.positions || [];
                // Ensure numeric coords (defensive if server sends strings)
                this.positions.forEach(p => {{ p.coords = (p.coords || []).map(Number); }});

                this.log(`📊 Snapshot: ${{this.nodes.length}} nodes, ${{this.edges.length}} edges`);

                // Initialize all nodes as visible by default
                this.filteredNodes.clear();
                this.nodes.forEach(node => this.filteredNodes.add(node.id));

                // Update UI components
                this.updateAttributeDropdown();
                this.updateStats();
                this.applyFilters();
                this.render(true);
            }}

            handleUpdate(update) {{
                this.updateCount++;
                const updateType = Object.keys(update)[0];
                
                console.log('📨 CLIENT: Received update type:', updateType, 'with payload:', update[updateType]);

                switch (updateType) {{
                    case 'NodeAdded':
                        this.nodes.push(update.NodeAdded);
                        this.log(`➕ Node added: ${{update.NodeAdded.id}}`);
                        this.updateAttributeDropdown();
                        break;

                    case 'NodeRemoved':
                        const nodeId = update.NodeRemoved;
                        this.nodes = this.nodes.filter(n => n.id !== nodeId);
                        this.positions = this.positions.filter(p => p.node_id !== nodeId);
                        this.log(`➖ Node removed: ${{nodeId}}`);
                        break;

                    case 'NodeChanged':
                        const nodeChange = update.NodeChanged;
                        const node = this.nodes.find(n => n.id === nodeChange.id);
                        if (node) {{
                            Object.assign(node.attributes, nodeChange.attributes);
                            this.log(`📝 Node updated: ${{nodeChange.id}}`);
                        }}
                        break;

                    case 'EdgeAdded':
                        this.edges.push(update.EdgeAdded);
                        this.log(`🔗 Edge added: ${{update.EdgeAdded.source}}→${{update.EdgeAdded.target}}`);
                        break;

                    case 'EdgeRemoved':
                        const edgeId = update.EdgeRemoved;
                        this.edges = this.edges.filter(e => e.id !== edgeId);
                        this.log(`💥 Edge removed: ${{edgeId}}`);
                        break;

                    case 'PositionDelta':
                        const positionDelta = update.PositionDelta;
                        const pos = this.positions.find(p => p.node_id === positionDelta.node_id);
                        if (pos && positionDelta.delta) {{
                            for (let i = 0; i < positionDelta.delta.length && i < pos.coords.length; i++) {{
                                pos.coords[i] += positionDelta.delta[i];
                            }}
                        }}
                        break;

                    case 'PositionsBatch':
                        const batchPositions = update.PositionsBatch;
                        console.log('🔄 CLIENT: Received PositionsBatch with', batchPositions.length, 'positions');
                        console.log('🔄 CLIENT: First position before update:', this.positions[0]);
                        
                        batchPositions.forEach(newPos => {{
                            newPos.coords = (newPos.coords || []).map(Number);
                            const existingPos = this.positions.find(p => p.node_id === newPos.node_id);
                            if (existingPos) {{
                                console.log(`🔄 CLIENT: Updating node ${{newPos.node_id}} position from [${{existingPos.coords[0]?.toFixed(2)}}, ${{existingPos.coords[1]?.toFixed(2)}}] to [${{newPos.coords[0]?.toFixed(2)}}, ${{newPos.coords[1]?.toFixed(2)}}]`);
                                existingPos.coords = [...newPos.coords];
                            }} else {{
                                console.log(`🔄 CLIENT: Adding new position for node ${{newPos.node_id}}:`, newPos.coords);
                                this.positions.push(newPos);
                            }}
                        }});
                        
                        console.log('🔄 CLIENT: First position after update:', this.positions[0]);
                        this.log(`📍 Batch position update: ${{batchPositions.length}} nodes`);
                        // Force a render after position updates to ensure visual changes are shown
                        console.log('🔄 CLIENT: Forcing render after position update');
                        this.render(true);
                        break;

                    case 'SelectionChanged':
                        const selection = update.SelectionChanged;
                        this.highlightedNodes.clear();
                        selection.selected.forEach(id => this.highlightedNodes.add(id));
                        this.log(`🎯 Selection changed: ${{selection.selected.length}} selected`);
                        break;

                    case 'EmbeddingChanged':
                        const embedding = update.EmbeddingChanged;
                        this.log(`🧠 Embedding changed: ${{embedding.method}} (${{embedding.dimensions}}D)`);
                        // Make sure the new layout/embedding becomes visible immediately
                        this.isPlaying = true;
                        this.updatePlayPauseUI();
                        this.render(true);
                        break;

                    case 'LayoutChanged':
                        const layout = update.LayoutChanged;
                        this.log(`📐 Layout changed: ${{layout.algorithm}}`);
                        // Make sure the new layout/embedding becomes visible immediately
                        this.isPlaying = true;
                        this.updatePlayPauseUI();
                        this.render(true);
                        break;

                    default:
                        this.log(`📈 Unknown update: ${{updateType}}`);
                }}

                this.applyFilters();
                this.updateStats();
                
                // Ensure visual updates after any position/data changes
                if (this.isPlaying) {{
                    this.render();
                }}
            }}

            onMouseDown(e) {{
                this.mouseDown = true;
                this.dragStart = {{ x: e.clientX, y: e.clientY }};

                // Determine potential drag mode based on mouse button and modifiers
                if (e.button === 0) {{ // Left mouse
                    if (e.ctrlKey) {{
                        this.pendingDragMode = 'rotate_higher_dims';
                    }} else {{
                        this.pendingDragMode = 'rotate_primary';
                    }}
                }} else if (e.button === 2) {{ // Right mouse
                    this.pendingDragMode = 'rotate_multi';
                }} else if (e.button === 1) {{ // Middle mouse
                    this.pendingDragMode = 'rotate_all_pairs';
                }}

                console.log(`🎮 CLIENT: Mouse down - pending mode: ${{this.pendingDragMode}}`);
            }}

            onMouseMove(e) {{
                if (!this.mouseDown) return;

                const dx = e.clientX - this.dragStart.x;
                const dy = e.clientY - this.dragStart.y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                // Only start dragging if mouse has moved beyond threshold
                if (!this.isDragging && distance > this.dragThreshold) {{
                    this.isDragging = true;
                    this.dragMode = this.pendingDragMode;
                    console.log(`🎮 CLIENT: Started dragging - mode: ${{this.dragMode}}`);
                    this.log(`🎮 ${{this.dragMode.replace('_', ' ')}} mode`);
                }}

                if (this.isDragging) {{
                    // Apply rotation based on drag mode
                    this.applyRotation(dx, dy, this.dragMode);
                    this.dragStart = {{ x: e.clientX, y: e.clientY }};
                }}
            }}

            onMouseUp(e) {{
                if (this.isDragging) {{
                    console.log(`🎮 CLIENT: Finished dragging - mode: ${{this.dragMode}}`);
                    this.log(`🎮 Released ${{this.dragMode}} mode`);
                }}

                this.mouseDown = false;
                this.isDragging = false;
                this.dragMode = 'none';
                this.pendingDragMode = 'none';
            }}

            applyRotation(dx, dy, mode) {{
                // Convert mouse movement to rotation parameters and send to server
                const rotationStrength = 0.01; // Scale factor for rotation sensitivity
                const rotationX = dx * rotationStrength;
                const rotationY = dy * rotationStrength;
                
                // Send embedding change command based on drag mode
                const payload = {{
                    method: "rotation",
                    k: 2, // 2D projection
                    params: {{
                        "rotation_x": rotationX.toString(),
                        "rotation_y": rotationY.toString(),
                        "mode": mode,
                        "timestamp": Date.now().toString()
                    }}
                }};
                
                this.sendControl("ChangeEmbedding", payload);
                this.log(`🔄 Rotation: dx=${{dx.toFixed(1)}}, dy=${{dy.toFixed(1)}}, mode=${{mode}}`);
            }}

            sendControl(kind, payload) {{
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {{
                    const message = {{
                        type: 'control',
                        version: 1,
                        payload: {{ [kind]: payload }}
                    }};
                    this.ws.send(JSON.stringify(message));
                    this.log(`🎮 Sent control: ${{kind}}`);
                }}
            }}

            render(force = false) {{
                if (!force && !this.isPlaying) return; // Skip rendering if paused unless forced

                const {{ width, height }} = this.canvas;
                this.ctx.clearRect(0, 0, width, height);

                console.log('🎨 CLIENT: Starting render - canvas size:', width, 'x', height);
                console.log('🎨 CLIENT: Have', this.positions.length, 'positions to render');
                if (this.positions.length > 0) {{
                    console.log('🎨 CLIENT: First 3 node positions:', this.positions.slice(0, 3));
                }}

                // Draw background
                this.ctx.fillStyle = '#f8f9fa';
                this.ctx.fillRect(0, 0, width, height);

                // Calculate position bounds for proper scaling
                let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
                this.positions.forEach(pos => {{
                    if (pos.coords[0] < minX) minX = pos.coords[0];
                    if (pos.coords[0] > maxX) maxX = pos.coords[0];
                    if (pos.coords[1] < minY) minY = pos.coords[1];
                    if (pos.coords[1] > maxY) maxY = pos.coords[1];
                }});

                // Calculate scale to fit nodes in canvas with padding
                const padding = 50;
                const rangeX = maxX - minX || 1;
                const rangeY = maxY - minY || 1;
                const scaleX = (width - 2 * padding) / rangeX;
                const scaleY = (height - 2 * padding) / rangeY;
                const scale = Math.min(scaleX, scaleY);

                console.log(`🎨 CLIENT: Position bounds: X[${{minX.toFixed(2)}}, ${{maxX.toFixed(2)}}] Y[${{minY.toFixed(2)}}, ${{maxY.toFixed(2)}}] scale=${{scale.toFixed(2)}}`);

                // Draw edges first (so nodes appear on top)
                this.edges.forEach(edge => {{
                    const sourcePos = this.positions.find(p => p.node_id === edge.source);
                    const targetPos = this.positions.find(p => p.node_id === edge.target);

                    if (sourcePos && targetPos) {{
                        // Check if both nodes are visible
                        const sourceVisible = this.filteredNodes.has(edge.source);
                        const targetVisible = this.filteredNodes.has(edge.target);
                        if (!sourceVisible || !targetVisible) return;

                        // Apply same scaling as nodes
                        const sx = (sourcePos.coords[0] - minX) * scale + padding;
                        const sy = (sourcePos.coords[1] - minY) * scale + padding;
                        const tx = (targetPos.coords[0] - minX) * scale + padding;
                        const ty = (targetPos.coords[1] - minY) * scale + padding;

                        // Highlight edges connected to selected/highlighted nodes
                        const isHighlighted = this.highlightedNodes.has(edge.source) || this.highlightedNodes.has(edge.target);
                        this.ctx.strokeStyle = isHighlighted ? '#ff6b6b' : '#6c757d';
                        this.ctx.lineWidth = isHighlighted ? 2 : 1;
                        this.ctx.globalAlpha = this.filters.opacity;

                        this.ctx.beginPath();
                        this.ctx.moveTo(sx, sy);
                        this.ctx.lineTo(tx, ty);
                        this.ctx.stroke();
                    }}
                }});

                // Reset alpha for nodes
                this.ctx.globalAlpha = 1.0;

                // Draw nodes (with filtering, highlighting, and search results)
                let nodesRendered = 0;
                this.positions.forEach(pos => {{
                    // Skip if node is filtered out
                    if (!this.filteredNodes.has(pos.node_id)) return;

                    // Apply proper scaling and centering
                    const x = (pos.coords[0] - minX) * scale + padding;
                    const y = (pos.coords[1] - minY) * scale + padding;

                    if (nodesRendered < 3) {{
                        console.log(`🎨 CLIENT: Rendering node ${{pos.node_id}} at canvas [${{x.toFixed(2)}}, ${{y.toFixed(2)}}] from world coords [${{pos.coords[0].toFixed(2)}}, ${{pos.coords[1].toFixed(2)}}]`);
                    }}
                    nodesRendered++;

                    // Determine node appearance based on state
                    let fillColor = '#007bff';
                    let strokeColor = null;
                    let strokeWidth = 0;
                    let radius = 5;

                    if (this.selectedNode && this.selectedNode.id === pos.node_id) {{
                        fillColor = '#ff6b6b';
                        strokeColor = '#dc3545';
                        strokeWidth = 3;
                        radius = 7;
                    }} else if (this.highlightedNodes.has(pos.node_id)) {{
                        fillColor = '#ffc107';
                        strokeColor = '#fd7e14';
                        strokeWidth = 2;
                        radius = 6;
                    }} else if (this.searchResults.has(pos.node_id)) {{
                        fillColor = '#28a745';
                        strokeColor = '#20c997';
                        strokeWidth = 2;
                    }}

                    // Apply opacity filter
                    this.ctx.globalAlpha = this.filters.opacity;

                    // Draw node
                    this.ctx.fillStyle = fillColor;
                    this.ctx.beginPath();
                    this.ctx.arc(x, y, radius, 0, 2 * Math.PI);
                    this.ctx.fill();

                    // Draw stroke if needed
                    if (strokeColor && strokeWidth > 0) {{
                        this.ctx.strokeStyle = strokeColor;
                        this.ctx.lineWidth = strokeWidth;
                        this.ctx.stroke();
                    }}
                }});

                console.log(`🎨 CLIENT: Render complete - rendered ${{nodesRendered}} nodes out of ${{this.positions.length}} total positions`);

                // Reset alpha
                this.ctx.globalAlpha = 1.0;

                // Draw honeycomb grid indicator
                this.ctx.strokeStyle = '#ffd700';
                this.ctx.lineWidth = 1;
                this.ctx.setLineDash([5, 5]);
                this.ctx.strokeRect(50, 50, width - 100, height - 100);
                this.ctx.setLineDash([]);

                // Update frame count for FPS calculation
                this.frameCount++;
            }}

            startRenderLoop() {{
                const render = () => {{
                    this.render();
                    requestAnimationFrame(render);
                }};
                render();
            }}

            updateStatus(connected, text) {{
                const dot = document.getElementById('status-dot');
                const statusText = document.getElementById('status-text');

                dot.classList.toggle('connected', connected);
                statusText.textContent = text;
            }}

            updateStats() {{
                document.getElementById('node-count').textContent = `Nodes: ${{this.nodes.length}}`;
                document.getElementById('edge-count').textContent = `Edges: ${{this.edges.length}}`;
            }}

            log(message) {{
                const messages = document.getElementById('messages');
                const time = new Date().toLocaleTimeString();
                messages.innerHTML += `[${{time}}] ${{message}}\\n`;
                messages.scrollTop = messages.scrollHeight;
            }}

            // New Phase 4 methods
            togglePlayPause() {{
                this.isPlaying = !this.isPlaying;
                this.updatePlayPauseUI();
                this.log(`${{this.isPlaying ? '▶️ Resumed' : '⏸️ Paused'}} simulation`);
            }}

            updatePlayPauseUI() {{
                const btn = document.getElementById('play-pause-btn');
                if (this.isPlaying) {{
                    btn.textContent = '⏸️ Pause Simulation';
                    btn.classList.remove('paused');
                }} else {{
                    btn.textContent = '▶️ Play Simulation';
                    btn.classList.add('paused');
                }}
            }}

            applyFilters() {{
                this.filteredNodes.clear();
                this.searchResults.clear();

                this.nodes.forEach(node => {{
                    let shouldShow = true;

                    // Search filter
                    if (this.filters.searchQuery) {{
                        const query = this.filters.searchQuery.toLowerCase();
                        const searchMatch = (
                            node.id.toString().includes(query) ||
                            Object.values(node.attributes || {{}}).some(val =>
                                val.toString().toLowerCase().includes(query)
                            )
                        );
                        if (searchMatch) {{
                            this.searchResults.add(node.id);
                        }} else {{
                            shouldShow = false;
                        }}
                    }}

                    // Degree filter
                    const degree = this.getNodeDegree(node.id);
                    if (degree < this.filters.degreeMin || degree > this.filters.degreeMax) {{
                        shouldShow = false;
                    }}

                    // Attribute filter
                    if (this.filters.attributeName && this.filters.attributeValue) {{
                        const attrValue = node.attributes?.[this.filters.attributeName];
                        if (!attrValue || !attrValue.toString().includes(this.filters.attributeValue)) {{
                            shouldShow = false;
                        }}
                    }}

                    if (shouldShow) {{
                        this.filteredNodes.add(node.id);
                    }}
                }});

                this.log(`🔍 Filters applied: ${{this.filteredNodes.size}}/${{this.nodes.length}} nodes visible`);
            }}

            getNodeDegree(nodeId) {{
                return this.edges.filter(e => e.source === nodeId || e.target === nodeId).length;
            }}

            updateAttributeDropdown() {{
                const select = document.getElementById('attribute-name');
                const attributes = new Set();

                this.nodes.forEach(node => {{
                    if (node.attributes) {{
                        Object.keys(node.attributes).forEach(attr => attributes.add(attr));
                    }}
                }});

                // Clear and repopulate
                select.innerHTML = '<option value="">Select attribute...</option>';
                [...attributes].sort().forEach(attr => {{
                    const option = document.createElement('option');
                    option.value = attr;
                    option.textContent = attr;
                    select.appendChild(option);
                }});
            }}

            onCanvasClick(e) {{
                console.log(`🎯 CLIENT: Canvas click - isDragging: ${{this.isDragging}}`);
                if (this.isDragging) {{
                    console.log(`🎯 CLIENT: Ignoring click because isDragging=true`);
                    return;
                }}

                const rect = this.canvas.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                console.log(`🎯 CLIENT: Click at canvas coords [${{x.toFixed(2)}}, ${{y.toFixed(2)}}]`);

                // Find clicked node
                const clickedNode = this.findNodeAt(x, y);
                if (clickedNode) {{
                    console.log(`🎯 CLIENT: Found clicked node:`, clickedNode);
                    this.selectedNode = clickedNode;
                    this.showAttributePanel(clickedNode);
                    this.highlightNeighbors(clickedNode.id);
                    this.sendControl('SelectNodes', [clickedNode.id]);
                    this.log(`🎯 Selected node: ${{clickedNode.id}}`);
                }} else {{
                    console.log(`🎯 CLIENT: No node found at click position`);
                    this.selectedNode = null;
                    this.hideAttributePanel();
                    this.highlightedNodes.clear();
                    this.sendControl('ClearSelection', {{}});
                }}
            }}

            findNodeAt(x, y) {{
                const {{ width, height }} = this.canvas;
                const nodeRadius = 7; // Slightly larger for easier clicking

                // Calculate the same scaling used in render()
                let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
                this.positions.forEach(pos => {{
                    if (pos.coords[0] < minX) minX = pos.coords[0];
                    if (pos.coords[0] > maxX) maxX = pos.coords[0];
                    if (pos.coords[1] < minY) minY = pos.coords[1];
                    if (pos.coords[1] > maxY) maxY = pos.coords[1];
                }});

                const padding = 50;
                const rangeX = maxX - minX || 1;
                const rangeY = maxY - minY || 1;
                const scaleX = (width - 2 * padding) / rangeX;
                const scaleY = (height - 2 * padding) / rangeY;
                const scale = Math.min(scaleX, scaleY);

                for (const pos of this.positions) {{
                    // Apply same scaling as render()
                    const nodeX = (pos.coords[0] - minX) * scale + padding;
                    const nodeY = (pos.coords[1] - minY) * scale + padding;
                    const dist = Math.sqrt((x - nodeX) ** 2 + (y - nodeY) ** 2);

                    if (dist <= nodeRadius) {{
                        console.log(`🎯 CLIENT: Clicked on node ${{pos.node_id}} at [${{nodeX.toFixed(2)}}, ${{nodeY.toFixed(2)}}]`);
                        return this.nodes.find(n => n.id === pos.node_id);
                    }}
                }}
                console.log(`🎯 CLIENT: No node found at click position [${{x.toFixed(2)}}, ${{y.toFixed(2)}}]`);
                return null;
            }}

            showAttributePanel(node) {{
                const panel = document.getElementById('attribute-panel');
                const tbody = document.getElementById('attribute-tbody');

                // Clear existing rows
                tbody.innerHTML = '';

                // Add node ID
                const idRow = tbody.insertRow();
                idRow.insertCell(0).textContent = 'ID';
                idRow.insertCell(1).textContent = node.id;

                // Add attributes
                if (node.attributes) {{
                    Object.entries(node.attributes).forEach(([key, value]) => {{
                        const row = tbody.insertRow();
                        row.insertCell(0).textContent = key;
                        row.insertCell(1).textContent = value;
                    }});
                }}

                // Update neighbor info
                const degree = this.getNodeDegree(node.id);
                document.getElementById('neighbor-count').textContent = degree;
                document.getElementById('node-degree').textContent = degree;

                panel.classList.add('visible');
            }}

            hideAttributePanel() {{
                document.getElementById('attribute-panel').classList.remove('visible');
            }}

            highlightNeighbors(nodeId) {{
                this.highlightedNodes.clear();
                this.highlightedNodes.add(nodeId);

                // Add neighbors
                this.edges.forEach(edge => {{
                    if (edge.source === nodeId) {{
                        this.highlightedNodes.add(edge.target);
                    }} else if (edge.target === nodeId) {{
                        this.highlightedNodes.add(edge.source);
                    }}
                }});
            }}

            startPerformanceMonitoring() {{
                setInterval(() => {{
                    // Calculate FPS
                    const now = performance.now();
                    if (now - this.lastFrameTime >= 1000) {{
                        this.fps = this.frameCount;
                        this.frameCount = 0;
                        this.lastFrameTime = now;
                        document.getElementById('fps-display').textContent = this.fps;
                    }}

                    // Calculate update rate
                    if (now - this.lastUpdateTime >= 1000) {{
                        this.updateRate = this.updateCount;
                        this.updateCount = 0;
                        this.lastUpdateTime = now;
                        document.getElementById('update-rate').textContent = this.updateRate;
                    }}

                    // Mock latency (would be calculated from message timestamps in real implementation)
                    this.latency = Math.random() * 10 + 2;
                    document.getElementById('latency').textContent = Math.round(this.latency);
                }}, 1000);
            }}
        }}

        // Global functions for attribute panel
        function hideAttributePanel() {{
            document.getElementById('attribute-panel').classList.remove('visible');
        }}

        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', () => {{
            new RealtimeViz();
        }});
    </script>
</body>
</html>"#,
        port = port
    )
}
