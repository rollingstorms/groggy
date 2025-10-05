/**
 * Groggy Realtime Visualization JavaScript Application
 *
 * This module provides the RealtimeViz class for interactive graph visualization
 * with WebSocket communication, canvas rendering, and dynamic controls.
 */

export class RealtimeViz {
    constructor(config) {
        this.config = config;
        this.ws = null;
        this.canvas = null;
        this.ctx = null;
        this.isConnected = false;
        this.isPaused = false;
        this.animationId = null;

        // State
        this.nodes = [];
        this.edges = [];
        this.visibleNodes = [];
        this.selectedNode = null;
        this.selectedEdge = null;
        this.hoveredNode = null;
        this.hoveredEdge = null;
        this.lastUpdateTime = Date.now();
        this.frameCount = 0;
        this.updateCount = 0;
        this.currentEmbeddingMethod = null;
        this.currentEmbeddingDimensions = 2;
        this.opacity = 1.0;
        this.curvatureMultiplier = 1.0;  // Default multiplier (1.0 = no change)
        this.filters = {
            degreeMin: 0,
            degreeMax: Number.POSITIVE_INFINITY,
            attributeName: '',
            attributeValue: ''
        };
        this.nodeDegrees = {};

        // Camera/viewport
        this.camera = {
            x: 0,
            y: 0,
            zoom: 1.0,
            targetZoom: 1.0,
            rotation: 0.0,
        };

        this.lastPointer = null;
        this.isDraggingCanvas = false;
        this.draggedNode = null;
        this.view3D = null;

        // Performance tracking
        this.stats = {
            fps: 60,
            updateRate: 0,
            latency: 0,
            nodeCount: 0,
            edgeCount: 0,
            visibleCount: 0
        };

        this.init();
    }

    init() {
        console.log('🚀 Initializing Groggy Realtime Visualization');
        this.setupCanvas();
        this.setupControls();
        this.setupWebSocket();
        this.startAnimationLoop();
        this.startStatsUpdater();

        // Initialize view manager for table/graph switching
        this.viewManager = new ViewManager(this);
    }

    setupCanvas() {
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');

        // Set canvas size
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());

        // Mouse event handlers
        this.canvas.addEventListener('click', (e) => this.handleCanvasClick(e));
        this.canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.handleMouseUp(e));
        this.canvas.addEventListener('mouseleave', (e) => this.handleMouseLeave(e));
        this.canvas.addEventListener('wheel', (e) => this.handleWheel(e), { passive: false });

        console.log('✅ Canvas initialized');
    }

    setupControls() {
        // Embedding method
        const embeddingSelect = document.getElementById('embedding-method');
        if (embeddingSelect) {
            this.currentEmbeddingMethod = embeddingSelect.value || 'pca';
            embeddingSelect.addEventListener('change', () => {
                const method = embeddingSelect.value;
                this.currentEmbeddingMethod = method;
                this.sendControlMessage('embedding', method);
            });
        }

        // Dimensions
        const dimensionsSelect = document.getElementById('dimensions');
        if (dimensionsSelect) {
            const initialDimensions = parseInt(dimensionsSelect.value, 10);
            if (!Number.isNaN(initialDimensions)) {
                this.currentEmbeddingDimensions = initialDimensions;
            }
            dimensionsSelect.addEventListener('change', () => {
                const value = parseInt(dimensionsSelect.value, 10);
                if (!Number.isNaN(value)) {
                    this.currentEmbeddingDimensions = value;
                }
                this.sendControlMessage('dimensions', this.currentEmbeddingDimensions);
            });
        }

        // Layout algorithm
        const layoutSelect = document.getElementById('layout-algorithm');
        const honeyCell = document.getElementById('honey-cell');
        const honeyScale = document.getElementById('honey-scale');
        const circleRadius = document.getElementById('circle-radius');
        const forceDist = document.getElementById('force-distance');
        const forceCharge = document.getElementById('force-charge');

        if (layoutSelect) {
            const sendLayout = () => {
                const layout = layoutSelect.value;
                const params = {};
                if (layout === 'honeycomb') {
                    if (honeyCell && honeyCell.value) params.cell_size = parseFloat(honeyCell.value);
                    if (honeyScale && honeyScale.value) params.scale = parseFloat(honeyScale.value);
                } else if (layout === 'circular') {
                    if (circleRadius && circleRadius.value) params.radius = parseFloat(circleRadius.value);
                } else if (layout === 'force_directed' || layout === 'force-directed') {
                    if (forceDist && forceDist.value) params.distance = parseFloat(forceDist.value);
                    if (forceCharge && forceCharge.value) params.charge = parseFloat(forceCharge.value);
                }
                this.sendControlMessage('layout', { algorithm: layout, params });

                const controllerMode = this.mapLayoutToController(layout);
                if (controllerMode) {
                    this.sendControlMessage('controller', { mode: controllerMode });
                    if (controllerMode === 'pan-2d') this.camera.rotation = 0;
                }
            };

            layoutSelect.addEventListener('change', sendLayout);
            [honeyCell, honeyScale, circleRadius, forceDist, forceCharge].forEach(el => {
                if (el) el.addEventListener('change', sendLayout);
            });
        }

        // Play/pause button
        const playPauseBtn = document.getElementById('play-pause-btn');
        if (playPauseBtn) {
            playPauseBtn.addEventListener('click', () => {
                this.isPaused = !this.isPaused;
                playPauseBtn.textContent = this.isPaused ? '▶️ Play' : '⏸️ Pause';
                if (!this.isPaused) {
                    this.draw();
                }
            });
        }

        // Degree filter sliders
        const degreeMin = document.getElementById('degree-min');
        const degreeMax = document.getElementById('degree-max');
        const degreeMinValue = document.getElementById('degree-min-value');
        const degreeMaxValue = document.getElementById('degree-max-value');

        if (degreeMin && degreeMax && degreeMinValue && degreeMaxValue) {
            const applyDegreeFilter = () => {
                const min = parseInt(degreeMin.value, 10);
                const max = parseInt(degreeMax.value, 10);
                this.filters.degreeMin = Number.isNaN(min) ? 0 : min;
                this.filters.degreeMax = Number.isNaN(max)
                    ? Number.POSITIVE_INFINITY
                    : max;
                degreeMinValue.textContent = degreeMin.value;
                degreeMaxValue.textContent = degreeMax.value;
                this.applyFilters();
            };

            degreeMin.addEventListener('input', applyDegreeFilter);
            degreeMax.addEventListener('input', applyDegreeFilter);
            degreeMinValue.textContent = degreeMin.value;
            degreeMaxValue.textContent = degreeMax.value;
        }

        // Search functionality
        const searchInput = document.getElementById('search-input');
        const searchBtn = document.getElementById('search-btn');
        const clearSearchBtn = document.getElementById('clear-search-btn');

        if (searchBtn && clearSearchBtn && searchInput) {
            searchBtn.addEventListener('click', () => {
                const query = searchInput.value.trim();
                if (query) {
                    this.handleSearch(query);
                }
            });

            clearSearchBtn.addEventListener('click', () => {
                searchInput.value = '';
                this.selectedNode = null;
                this.hideAttributePanel();
                this.draw();
            });

            searchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    searchBtn.click();
                }
            });
        }

        // Attribute filter
        const attributeName = document.getElementById('attribute-name');
        const attributeValue = document.getElementById('attribute-value');
        const applyFilterBtn = document.getElementById('apply-filter-btn');

        if (applyFilterBtn && attributeName && attributeValue) {
            applyFilterBtn.addEventListener('click', () => {
                this.filters.attributeName = attributeName.value || '';
                this.filters.attributeValue = attributeValue.value.trim();
                this.applyFilters();
            });
        }

        // Opacity slider
        const opacitySlider = document.getElementById('opacity-slider');
        const opacityValue = document.getElementById('opacity-value');

        if (opacitySlider && opacityValue) {
            opacitySlider.addEventListener('input', () => {
                const value = parseFloat(opacitySlider.value);
                this.opacity = Number.isNaN(value) ? 1.0 : value;
                opacityValue.textContent = this.opacity.toFixed(1);
                this.draw();
            });
            opacityValue.textContent = parseFloat(opacitySlider.value).toFixed(1);
        }

        // Curvature slider
        const curvatureSlider = document.getElementById('curvature-slider');
        const curvatureValue = document.getElementById('curvature-value');

        if (curvatureSlider && curvatureValue) {
            curvatureSlider.addEventListener('input', () => {
                const value = parseFloat(curvatureSlider.value);
                this.curvatureMultiplier = Number.isNaN(value) ? 1.0 : value;
                curvatureValue.textContent = this.curvatureMultiplier.toFixed(1);
                this.draw();
            });
            curvatureValue.textContent = parseFloat(curvatureSlider.value).toFixed(1);
        }

        // Autofit button
        const autofitBtn = document.getElementById('autofit-btn');
        if (autofitBtn) {
            autofitBtn.addEventListener('click', () => this.autoFitGraph());
        }

        console.log('✅ Controls initialized');

        if (layoutSelect) {
            const initialController = this.mapLayoutToController(layoutSelect.value);
            if (initialController) {
                this.sendControlMessage('controller', { mode: initialController });
            }
        }
    }

    setupWebSocket() {
        console.log(`🔌 Connecting to WebSocket: ${this.config.wsUrl}`);

        this.ws = new WebSocket(this.config.wsUrl);

        this.ws.onopen = () => {
            console.log('✅ WebSocket connected');
            this.isConnected = true;
            this.updateConnectionStatus('connected');

            const layoutSelect = document.getElementById('layout-algorithm');
            if (layoutSelect) {
                const controllerMode = this.mapLayoutToController(layoutSelect.value);
                if (controllerMode) {
                    this.sendControlMessage('controller', { mode: controllerMode });
                }
            }
        };

        this.ws.onclose = () => {
            console.log('❌ WebSocket disconnected');
            this.isConnected = false;
            this.updateConnectionStatus('disconnected');

            // Attempt to reconnect after 3 seconds
            setTimeout(() => this.setupWebSocket(), 3000);
        };

        this.ws.onerror = (error) => {
            console.error('❌ WebSocket error:', error);
            this.updateConnectionStatus('error');
        };

        this.ws.onmessage = (event) => {
            this.handleWebSocketMessage(event.data);
        };
    }

    handleWebSocketMessage(data) {
        try {
            const message = JSON.parse(data);
            const messageTime = Date.now();

            // Update latency
            if (message.timestamp) {
                this.stats.latency = messageTime - message.timestamp;
            }

            switch (message.type) {
                case 'snapshot':
                    console.log('📊 Received initial snapshot:', message);
                    this.handleSnapshot(message.payload);
                    break;

                case 'update':
                    console.log('📡 Received update:', message);
                    this.handleEngineUpdate(message.payload);
                    this.updateCount++;
                    break;

                case 'table_data':
                    console.log('📋 Received table_data message:', message);
                    console.log('  - ViewManager exists?', !!this.viewManager);
                    console.log('  - TableRenderer exists?', !!this.viewManager?.tableRenderer);
                    if (this.viewManager && this.viewManager.tableRenderer) {
                        console.log('  - Calling handleTableData...');
                        this.viewManager.tableRenderer.handleTableData(message.payload);
                    } else {
                        console.error('  - ❌ ViewManager or TableRenderer not available!');
                    }
                    break;

                case 'control_ack':
                    console.log('✅ Control ack:', message);
                    break;

                default:
                    console.warn('Unknown message type:', message.type, message);
            }
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    }

    handleSnapshot(snapshot) {
        console.log('📊 Processing initial snapshot:', snapshot);

        // Load nodes
        if (snapshot.nodes) {
            this.nodes = snapshot.nodes.map(node => ({
                id: node.id,
                attributes: node.attributes,
                label: node.label || node.attributes?.label || node.id.toString(),
                x: 0,
                y: 0,
                // Copy VizConfig styling fields
                color: node.color,
                size: node.size,
                shape: node.shape,
                opacity: node.opacity,
                border_color: node.border_color,
                border_width: node.border_width,
                label_color: node.label_color,
                label_size: node.label_size
            }));
            this.stats.nodeCount = this.nodes.length;
            console.log(`📊 Loaded ${this.nodes.length} nodes`);
            console.log('🎨 First node styling:', this.nodes[0]);
        }

        // Load edges
        if (snapshot.edges) {
            // Debug: check raw snapshot edges
            console.log('🔍 Raw snapshot edges sample:', snapshot.edges.slice(126, 128));
            console.log('🔍 First edge from snapshot:', JSON.stringify(snapshot.edges[0]));

            this.edges = snapshot.edges.map(edge => ({
                id: edge.id,
                source: edge.source,
                target: edge.target,
                attributes: edge.attributes,
                // Copy VizConfig styling fields
                color: edge.color,
                width: edge.width,
                opacity: edge.opacity,
                style: edge.style,
                curvature: edge.curvature,
                label: edge.label,
                label_size: edge.label_size,
                label_color: edge.label_color
            }));
            this.stats.edgeCount = this.edges.length;
            console.log(`📊 Loaded ${this.edges.length} edges`);

            // Log edges with curvature for debugging
            const curvedEdges = this.edges.filter(e => e.curvature);
            if (curvedEdges.length > 0) {
                console.log(`🌊 Found ${curvedEdges.length} edges with curvature:`, curvedEdges.map(e => ({id: e.id, curvature: e.curvature})));
            }

            // Log edges with labels for debugging
            console.log('🔍 UPDATED CODE LOADED - Checking first 3 edges for labels:', this.edges.slice(0, 3).map(e => ({
                id: e.id,
                label: e.label,
                hasLabel: !!e.label
            })));

            const labeledEdges = this.edges.filter(e => e.label);
            if (labeledEdges.length > 0) {
                console.log(`🏷️  Found ${labeledEdges.length} edges with labels:`, labeledEdges.slice(0, 5).map(e => ({
                    id: e.id,
                    label: e.label,
                    label_size: e.label_size,
                    label_color: e.label_color
                })));
            } else {
                console.log('⚠️  No edges with labels found! Checking raw snapshot first edge:', snapshot.edges[0]);
            }
        }

        // Apply positions if available
        if (snapshot.positions && snapshot.positions.length > 0) {
            console.log(`📍 Applying ${snapshot.positions.length} node positions`);
            const positionedNodeIds = new Set();
            
            for (const position of snapshot.positions) {
                const node = this.nodes.find(n => n.id === position.node_id);
                if (node && position.coords && position.coords.length >= 2) {
                    node.x = position.coords[0] || 0;
                    node.y = position.coords[1] || 0;
                    positionedNodeIds.add(node.id);
                }
            }
            
            // Generate random positions for nodes that didn't get positions from snapshot
            const nodesWithoutPositions = this.nodes.filter(n => !positionedNodeIds.has(n.id));
            if (nodesWithoutPositions.length > 0) {
                console.log(`📍 Generating random positions for ${nodesWithoutPositions.length} nodes without positions`);
                const centerX = 400;
                const centerY = 300;
                const radius = 200;
                
                nodesWithoutPositions.forEach((node, index) => {
                    const angle = (index / nodesWithoutPositions.length) * 2 * Math.PI;
                    node.x = centerX + Math.cos(angle) * radius + (Math.random() - 0.5) * 100;
                    node.y = centerY + Math.sin(angle) * radius + (Math.random() - 0.5) * 100;
                });
            }
        } else {
            // Generate random positions if none provided
            this.generateRandomPositions();
        }

        // Update metadata
        if (snapshot.meta) {
            console.log('📊 Graph metadata:', snapshot.meta);
            // Could use this for additional info display
        }

        // Update attribute dropdown
        if (this.nodes.length > 0) {
            const attributes = new Set();
            this.nodes.forEach(node => {
                Object.keys(node.attributes || {}).forEach(attr => attributes.add(attr));
            });
            this.updateAttributeDropdown(Array.from(attributes));
        }

        // Reset derived state for filters
        this.filters.degreeMin = 0;
        this.filters.degreeMax = Number.POSITIVE_INFINITY;
        this.filters.attributeName = '';
        this.filters.attributeValue = '';

        const degreeMinInput = document.getElementById('degree-min');
        if (degreeMinInput) {
            const min = parseInt(degreeMinInput.value, 10);
            if (!Number.isNaN(min)) {
                this.filters.degreeMin = min;
            }
        }
        
        // Don't read degreeMax from slider here - computeNodeDegrees will set it
        // to the actual max degree of the graph

        const attributeNameInput = document.getElementById('attribute-name');
        const attributeValueInput = document.getElementById('attribute-value');
        if (attributeNameInput && attributeValueInput) {
            this.filters.attributeName = attributeNameInput.value || '';
            this.filters.attributeValue = attributeValueInput.value.trim();
        }

        // Compute degrees and auto-set max degree filter
        this.computeNodeDegrees();
        this.applyFilters();

        console.log('✅ Snapshot processed successfully');
    }

    handleEngineUpdate(update) {
        console.log('📡 Processing engine update:', update);
        console.log('📡 Update keys:', Object.keys(update));
        console.log('📡 Update type:', update.type || Object.keys(update)[0]);

        // Handle different update types
        switch (update.type || Object.keys(update)[0]) {
            case 'UpdateEnvelope':
                {
                    const env = update.UpdateEnvelope;
                    if (!env) {
                        console.warn('⚠️  Missing UpdateEnvelope payload', update);
                        break;
                    }

                    if (env.params_changed) {
                        console.log('🎛️ Params changed:', env.params_changed);
                    }

                    if (env.graph_patch) {
                        const gp = env.graph_patch;
                        if (gp.nodes_removed && gp.nodes_removed.length) {
                            this.nodes = this.nodes.filter(n => !gp.nodes_removed.includes(n.id));
                            this.stats.nodeCount = this.nodes.length;
                        }
                        if (gp.nodes_added && gp.nodes_added.length) {
                            gp.nodes_added.forEach(n => {
                                this.nodes.push({
                                    id: n.id,
                                    attributes: n.attributes || {},
                                    label: (n.attributes && n.attributes.label) || String(n.id),
                                    x: undefined,
                                    y: undefined,
                                });
                            });
                            this.stats.nodeCount = this.nodes.length;
                        }
                        // TODO: apply nodes_changed / edges_* when UI requires it
                    }

                    if (env.positions && env.positions.positions) {
                        const positions = env.positions.positions;
                        for (const p of positions) {
                            const node = this.nodes.find(n => n.id === p.node_id);
                            if (node && p.coords && p.coords.length >= 2) {
                                node.x = p.coords[0];
                                node.y = p.coords[1];
                            }
                        }
                        if (positions.length) {
                            this.draw();
                        }
                    }

                    if (env.view_changed) {
                        if (env.view_changed.view_2d) {
                            const v = env.view_changed.view_2d;
                            if (typeof v.x === 'number') this.camera.x = v.x;
                            if (typeof v.y === 'number') this.camera.y = v.y;
                            if (typeof v.zoom === 'number') this.camera.zoom = v.zoom;
                            if (typeof v.rotation === 'number') this.camera.rotation = v.rotation;
                        }
                        if (env.view_changed.view_3d) {
                            this.view3D = env.view_changed.view_3d;
                        }
                    }

                    this.visibleNodes = this.nodes.map(n => n.id);
                    this.stats.visibleCount = this.visibleNodes.length;
                    this.draw();
                }
                break;

            case 'NodeAdded':
                if (update.NodeAdded) {
                    const node = update.NodeAdded;
                    this.nodes.push({
                        id: node.id,
                        attributes: node.attributes,
                        label: node.attributes.label || node.id.toString(),
                        x: Math.random() * 400,
                        y: Math.random() * 400
                    });
                    this.stats.nodeCount = this.nodes.length;
                }
                break;

            case 'NodeRemoved':
                if (update.NodeRemoved !== undefined) {
                    const nodeId = update.NodeRemoved;
                    this.nodes = this.nodes.filter(n => n.id !== nodeId);
                    this.stats.nodeCount = this.nodes.length;
                }
                break;

            case 'EdgeAdded':
                if (update.EdgeAdded) {
                    const edge = update.EdgeAdded;
                    this.edges.push({
                        id: edge.id,
                        source: edge.source,
                        target: edge.target,
                        attributes: edge.attributes
                    });
                    this.stats.edgeCount = this.edges.length;
                }
                break;

            case 'PositionDelta':
                if (update.PositionDelta) {
                    const { node_id, delta } = update.PositionDelta;
                    const node = this.nodes.find(n => n.id === node_id);
                    if (node && delta.length >= 2) {
                        node.x += delta[0];
                        node.y += delta[1];
                    }
                }
                break;

            case 'PositionsBatch':
                if (update.PositionsBatch) {
                    console.log(`📍 DEBUG: Received PositionsBatch with ${update.PositionsBatch.length} positions`);
                    let updated = 0;
                    for (const position of update.PositionsBatch) {
                        const node = this.nodes.find(n => n.id === position.node_id);
                        if (node && position.coords && position.coords.length >= 2) {
                            const oldX = node.x, oldY = node.y;
                            node.x = position.coords[0];
                            node.y = position.coords[1];
                            updated++;
                            console.log(`📍 DEBUG: Updated node ${position.node_id}: (${oldX.toFixed(1)}, ${oldY.toFixed(1)}) → (${node.x.toFixed(1)}, ${node.y.toFixed(1)})`);
                        } else {
                            console.log(`❌ DEBUG: Node ${position.node_id} not found or invalid coords:`, position);
                        }
                    }
                    console.log(`📍 DEBUG: Updated ${updated} node positions, triggering render`);
                    this.draw(); // Force redraw after position updates
                }
                break;

            case 'SelectionChanged':
                if (update.SelectionChanged) {
                    // Could update selection state here
                    console.log('Selection changed:', update.SelectionChanged);
                }
                break;

            case 'LayoutChanged':
                if (update.LayoutChanged) {
                    console.log('🎨 DEBUG: Layout changed to:', update.LayoutChanged.algorithm);
                    console.log('🎨 DEBUG: Layout params:', update.LayoutChanged.params);
                    // Layout changes are processed by the backend - we just log them
                    // The actual position updates should come via PositionsBatch
                }
                break;

            default:
                console.log('❌ DEBUG: Unhandled engine update type:', update);
        }

        this.computeNodeDegrees();
        this.applyFilters();
    }

    generateRandomPositions() {
        console.log('🎲 Generating random positions for nodes');
        const centerX = 400;
        const centerY = 300;
        const radius = 200;

        this.nodes.forEach((node, index) => {
            const angle = (index / this.nodes.length) * 2 * Math.PI;
            node.x = centerX + Math.cos(angle) * radius + (Math.random() - 0.5) * 100;
            node.y = centerY + Math.sin(angle) * radius + (Math.random() - 0.5) * 100;
        });
    }

    sendControlMessage(type, data) {
        if (!this.ws || !this.isConnected) {
            console.warn('⚠️  Control message ignored (WebSocket not ready)', type, data);
            return;
        }

        if (type === 'embedding' && typeof data === 'string') {
            this.currentEmbeddingMethod = data;
        } else if (type === 'dimensions' && typeof data === 'number') {
            this.currentEmbeddingDimensions = data;
        }

        const payload = this.buildControlPayload(type, data);
        if (!payload) {
            console.warn('⚠️  Unsupported control message type', type, data);
            return;
        }

        const message = {
            type: 'control',
            version: 1,
            payload,
        };

        this.ws.send(JSON.stringify(message));
        console.log('📤 Sent control message', message);
    }

    buildControlPayload(type, data) {
        switch (type) {
            case 'embedding':
            case 'dimensions':
                return {
                    ChangeEmbedding: {
                        method: this.currentEmbeddingMethod || 'pca',
                        k: this.currentEmbeddingDimensions || 2,
                        params: {},
                    },
                };
            case 'layout': {
                if (typeof data === 'string') {
                    return { ChangeLayout: { algorithm: data, params: {} } };
                }
                // expect { algorithm, params? }
                const algorithm = data?.algorithm ?? 'honeycomb';
                const params = data?.params ?? {};
                return { ChangeLayout: { algorithm, params } };
                }
            case 'controller':
                return {
                    SetInteractionController: {
                        mode: data.mode,
                    },
                };
            case 'pointer':
                return {
                    Pointer: {
                        event: data,
                    },
                };
            case 'wheel':
                return {
                    Wheel: {
                        event: data,
                    },
                };
            case 'node_drag':
                return {
                    NodeDrag: {
                        event: data,
                    },
                };
            default:
                return null;
        }
    }

    mapLayoutToController(layout) {
        switch (layout) {
            case 'globe':
            case 'sphere':
                return 'globe-3d';
            case 'honeycomb':
                return 'honeycomb-nd';
            default:
                return 'pan-2d';
        }
    }

    updateConnectionStatus(status) {
        const statusIndicator = document.getElementById('connection-status');
        const statusText = document.getElementById('connection-text');

        statusIndicator.className = `status-${status}`;

        switch (status) {
            case 'connected':
                statusText.textContent = 'Connected';
                break;
            case 'disconnected':
                statusText.textContent = 'Disconnected';
                break;
            case 'connecting':
                statusText.textContent = 'Connecting...';
                break;
            case 'error':
                statusText.textContent = 'Error';
                break;
        }
    }

    resizeCanvas() {
        const container = this.canvas.parentElement;
        this.canvas.width = container.clientWidth;
        this.canvas.height = container.clientHeight;
    }

    screenToWorld(event) {
        const rect = this.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        // Canvas transformation order:
        // 1. translate(width/2, height/2) - center
        // 2. scale(zoom, zoom)
        // 3. rotate(rotation)
        // 4. translate(-camera.x, -camera.y)

        // Inverse transformation (reverse order):
        // Start with screen coords, subtract canvas center
        let relX = x - this.canvas.width / 2;
        let relY = y - this.canvas.height / 2;

        // Inverse: divide by zoom (undo scale)
        relX /= this.camera.zoom;
        relY /= this.camera.zoom;

        // Inverse: rotate backwards (undo rotate)
        if (this.camera.rotation) {
            const cos = Math.cos(-this.camera.rotation);
            const sin = Math.sin(-this.camera.rotation);
            const rotX = relX * cos - relY * sin;
            const rotY = relX * sin + relY * cos;
            relX = rotX;
            relY = rotY;
        }

        // Inverse: add camera position (undo translate(-camera))
        const worldX = relX + this.camera.x;
        const worldY = relY + this.camera.y;

        return { screenX: x, screenY: y, worldX, worldY };
    }

    handleCanvasClick(event) {
        const { worldX, worldY } = this.screenToWorld(event);

        // Try to click node first (higher priority)
        const clickedNode = this.findNodeAtPosition(worldX, worldY);
        if (clickedNode) {
            this.selectedNode = clickedNode;
            this.selectedEdge = null;
            this.showAttributePanel(clickedNode, 'node');
            console.log('🎯 Selected node:', clickedNode.id);
            this.draw();
            return;
        }

        // Try to click edge
        const clickedEdge = this.findEdgeAtPosition(worldX, worldY);
        if (clickedEdge) {
            this.selectedEdge = clickedEdge;
            this.selectedNode = null;
            this.showAttributePanel(clickedEdge, 'edge');
            console.log('🎯 Selected edge:', clickedEdge.id);
            this.draw();
            return;
        }

        // Clicked empty space
        this.selectedNode = null;
        this.selectedEdge = null;
        this.hideAttributePanel();
        this.draw();
    }

    handleMouseDown(event) {
        event.preventDefault();
        const coords = this.screenToWorld(event);
        const node = this.findNodeAtPosition(coords.worldX, coords.worldY);
        this.lastPointer = { x: event.clientX, y: event.clientY };

        if (node) {
            this.draggedNode = node;
            this.sendControlMessage('node_drag', {
                Start: { node_id: node.id, x: coords.worldX, y: coords.worldY },
            });
        } else {
            this.isDraggingCanvas = true;
            this.sendControlMessage('pointer', {
                phase: 'Start',
                dx: 0,
                dy: 0,
                ctrl: event.ctrlKey,
                shift: event.shiftKey,
                alt: event.altKey,
            });
        }
    }

    handleMouseMove(event) {
        const coords = this.screenToWorld(event);

        if (!this.lastPointer) {
            // Not dragging, update hover state
            const hoveredNode = this.findNodeAtPosition(coords.worldX, coords.worldY);
            const hoveredEdge = hoveredNode ? null : this.findEdgeAtPosition(coords.worldX, coords.worldY);

            if (hoveredNode !== this.hoveredNode || hoveredEdge !== this.hoveredEdge) {
                this.hoveredNode = hoveredNode;
                this.hoveredEdge = hoveredEdge;
                this.canvas.style.cursor = (hoveredNode || hoveredEdge) ? 'pointer' : 'default';
                this.draw();
            }
            return;
        }

        const dx = event.clientX - this.lastPointer.x;
        const dy = event.clientY - this.lastPointer.y;
        this.lastPointer = { x: event.clientX, y: event.clientY };

        if (this.draggedNode) {
            this.draggedNode.x = coords.worldX;
            this.draggedNode.y = coords.worldY;
            this.sendControlMessage('node_drag', {
                Move: {
                    node_id: this.draggedNode.id,
                    x: coords.worldX,
                    y: coords.worldY,
                },
            });
            this.draw();
        } else if (this.isDraggingCanvas) {
            if (event.shiftKey) {
                this.camera.rotation += dx * 0.005;
            } else {
                this.camera.x -= dx / this.camera.zoom;
                this.camera.y -= dy / this.camera.zoom;
            }

            this.sendControlMessage('pointer', {
                phase: 'Move',
                dx,
                dy,
                ctrl: event.ctrlKey,
                shift: event.shiftKey,
                alt: event.altKey,
            });
            this.draw();
        }
    }

    handleMouseUp(event) {
        if (this.draggedNode) {
            this.sendControlMessage('node_drag', {
                End: { node_id: this.draggedNode.id },
            });
            this.draggedNode = null;
        }

        if (this.isDraggingCanvas) {
            this.sendControlMessage('pointer', {
                phase: 'End',
                dx: 0,
                dy: 0,
                ctrl: event.ctrlKey,
                shift: event.shiftKey,
                alt: event.altKey,
            });
        }

        this.isDraggingCanvas = false;
        this.lastPointer = null;
        this.draw();
    }

    handleMouseLeave() {
        if (this.isDraggingCanvas) {
            this.sendControlMessage('pointer', {
                phase: 'End',
                dx: 0,
                dy: 0,
                ctrl: false,
                shift: false,
                alt: false,
            });
        }

        if (this.draggedNode) {
            this.sendControlMessage('node_drag', {
                End: { node_id: this.draggedNode.id },
            });
        }

        this.isDraggingCanvas = false;
        this.draggedNode = null;
        this.lastPointer = null;
        this.draw();
    }

    handleWheel(event) {
        event.preventDefault();

        const coords = this.screenToWorld(event);
        const zoomFactor = event.deltaY < 0 ? 1.1 : 0.9;
        const newZoom = Math.max(0.1, Math.min(5.0, this.camera.zoom * zoomFactor));

        this.camera.zoom = newZoom;
        this.camera.x = coords.worldX - (coords.screenX - this.canvas.width / 2) / this.camera.zoom;
        this.camera.y = coords.worldY - (coords.screenY - this.canvas.height / 2) / this.camera.zoom;

        this.sendControlMessage('wheel', {
            Zoom: { delta: event.deltaY },
        });
    }

    findNodeAtPosition(x, y) {
        // Larger hit detection radius for easier hovering (15px in screen space)
        const nodeRadius = 15 / this.camera.zoom;
        return this.nodes.find(node => {
            if (node.x === undefined || node.y === undefined) return false;
            if (this.visibleNodes.length && !this.visibleNodes.includes(node.id)) return false;
            const dx = node.x - x;
            const dy = node.y - y;
            return Math.sqrt(dx * dx + dy * dy) <= nodeRadius;
        });
    }

    findEdgeAtPosition(x, y) {
        // More forgiving hit detection radius (15px in screen space)
        const clickRadius = 15 / this.camera.zoom;

        let closestEdge = null;
        let closestDistance = Infinity;

        for (const edge of this.edges) {
            const sourceNode = this.nodes.find(n => n.id === edge.source);
            const targetNode = this.nodes.find(n => n.id === edge.target);

            if (!sourceNode || !targetNode) continue;
            if (sourceNode.x === undefined || targetNode.x === undefined) continue;

            const edgeCurvature = edge.curvature || 0;
            const curvature = edgeCurvature * this.curvatureMultiplier;

            let edgeDistance;

            if (curvature !== 0) {
                // For curved edges, sample points along the Bezier curve
                const dx = targetNode.x - sourceNode.x;
                const dy = targetNode.y - sourceNode.y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                const midX = (sourceNode.x + targetNode.x) / 2;
                const midY = (sourceNode.y + targetNode.y) / 2;

                const perpX = -dy / dist;
                const perpY = dx / dist;

                const offset = curvature * dist * 0.2;
                const controlX = midX + perpX * offset;
                const controlY = midY + perpY * offset;

                // Sample points along the curve with finer resolution
                let minDistance = Infinity;
                for (let t = 0; t <= 1; t += 0.02) {
                    const px = (1 - t) * (1 - t) * sourceNode.x +
                               2 * (1 - t) * t * controlX +
                               t * t * targetNode.x;
                    const py = (1 - t) * (1 - t) * sourceNode.y +
                               2 * (1 - t) * t * controlY +
                               t * t * targetNode.y;

                    const d = Math.sqrt((x - px) * (x - px) + (y - py) * (y - py));
                    if (d < minDistance) minDistance = d;
                }

                edgeDistance = minDistance;
            } else {
                // Straight edge - calculate distance from point to line segment
                const x1 = sourceNode.x, y1 = sourceNode.y;
                const x2 = targetNode.x, y2 = targetNode.y;

                const A = x - x1;
                const B = y - y1;
                const C = x2 - x1;
                const D = y2 - y1;

                const dot = A * C + B * D;
                const lenSq = C * C + D * D;
                let param = -1;

                if (lenSq != 0) param = dot / lenSq;

                let xx, yy;

                if (param < 0) {
                    xx = x1;
                    yy = y1;
                } else if (param > 1) {
                    xx = x2;
                    yy = y2;
                } else {
                    xx = x1 + param * C;
                    yy = y1 + param * D;
                }

                const dx = x - xx;
                const dy = y - yy;
                edgeDistance = Math.sqrt(dx * dx + dy * dy);
            }

            // Track the closest edge within the click radius
            if (edgeDistance <= clickRadius && edgeDistance < closestDistance) {
                closestDistance = edgeDistance;
                closestEdge = edge;
            }
        }

        return closestEdge;
    }

    showAttributePanel(item, type) {
        const panel = document.getElementById('attribute-panel');
        const content = document.getElementById('attribute-content');

        let html = '';

        if (type === 'edge') {
            // Show edge information
            html += `<div class="attribute-row"><strong>Edge ${item.id}</strong></div>`;
            html += `<div class="attribute-row"><span class="attribute-key">Source:</span> <span class="attribute-value">${item.source}</span></div>`;
            html += `<div class="attribute-row"><span class="attribute-key">Target:</span> <span class="attribute-value">${item.target}</span></div>`;

            // Add curvature slider
            const currentCurvature = item.curvature || 0;
            html += `
                <div class="attribute-row" style="margin-top: 10px;">
                    <strong>Edge Curvature:</strong>
                </div>
                <div class="attribute-row" style="display: flex; align-items: center; gap: 8px;">
                    <input type="range" id="edge-curvature-slider" min="-2" max="2" step="0.1" value="${currentCurvature}" style="flex: 1;">
                    <span id="edge-curvature-value" style="min-width: 40px;">${currentCurvature.toFixed(1)}</span>
                </div>
            `;

            if (item.attributes) {
                html += `<div class="attribute-row" style="margin-top: 10px;"><strong>Attributes:</strong></div>`;
                for (const [key, value] of Object.entries(item.attributes)) {
                    html += `
                        <div class="attribute-row">
                            <span class="attribute-key">${key}:</span>
                            <span class="attribute-value">${value}</span>
                        </div>
                    `;
                }
            }
        } else {
            // Show node information (existing code)
            html += `<div class="attribute-row"><strong>Node ${item.id}</strong></div>`;
            if (item.attributes) {
                for (const [key, value] of Object.entries(item.attributes)) {
                    html += `
                        <div class="attribute-row">
                            <span class="attribute-key">${key}:</span>
                        <span class="attribute-value">${value}</span>
                    </div>
                `;
                }
            }
        }

        content.innerHTML = html || '<p>No attributes available</p>';
        panel.classList.add('visible');

        // Add event listener for edge curvature slider
        if (type === 'edge') {
            const slider = document.getElementById('edge-curvature-slider');
            const valueDisplay = document.getElementById('edge-curvature-value');

            if (slider && valueDisplay) {
                slider.addEventListener('input', (e) => {
                    const newCurvature = parseFloat(e.target.value);
                    valueDisplay.textContent = newCurvature.toFixed(1);

                    // Update the edge's curvature in the local data
                    item.curvature = newCurvature;

                    // Redraw the canvas to show the new curvature
                    this.draw();
                });
            }
        }
    }

    hideAttributePanel() {
        const panel = document.getElementById('attribute-panel');
        panel.classList.remove('visible');
    }

    updateAttributePanel(nodeId, attributes) {
        if (this.selectedNode && this.selectedNode.id === nodeId) {
            const content = document.getElementById('attribute-content');
            let html = '';
            for (const [key, value] of Object.entries(attributes)) {
                html += `
                    <div class="attribute-row">
                        <span class="attribute-key">${key}:</span>
                        <span class="attribute-value">${value}</span>
                    </div>
                `;
            }
            content.innerHTML = html;
        }
    }

    updateAttributeDropdown(attributes) {
        const dropdown = document.getElementById('attribute-name');
        const currentValue = dropdown.value;

        // Clear current options except "None"
        dropdown.innerHTML = '<option value="">None</option>';

        // Add attribute options
        attributes.forEach(attr => {
            const option = document.createElement('option');
            option.value = attr;
            option.textContent = attr;
            if (attr === currentValue) {
                option.selected = true;
            }
            dropdown.appendChild(option);
        });
    }

    computeNodeDegrees() {
        const degrees = {};
        this.nodes.forEach(node => {
            degrees[node.id] = 0;
        });

        this.edges.forEach(edge => {
            if (degrees[edge.source] !== undefined) {
                degrees[edge.source] += 1;
            }
            if (degrees[edge.target] !== undefined) {
                degrees[edge.target] += 1;
            }
        });

        this.nodeDegrees = degrees;
        
        // Calculate max degree in the graph
        const maxDegree = Math.max(0, ...Object.values(degrees));
        
        // Update degree filter slider to match graph's max degree
        const degreeMaxInput = document.getElementById('degree-max');
        const degreeMaxValue = document.getElementById('degree-max-value');
        
        if (degreeMaxInput && degreeMaxValue) {
            // Set slider max and value to actual max degree (with minimum of 10 for usability)
            const sliderMax = Math.max(10, maxDegree);
            degreeMaxInput.max = sliderMax;
            degreeMaxInput.value = sliderMax;
            degreeMaxValue.textContent = sliderMax;
            
            // Update filter to include all nodes by default
            this.filters.degreeMax = sliderMax;
            
            console.log(`📊 Degree range: 0 to ${maxDegree}, slider max set to ${sliderMax}`);
        }
    }

    applyFilters() {
        if (!this.nodes.length) {
            this.visibleNodes = [];
            this.stats.visibleCount = 0;
            this.updateStatsDisplay();
            this.draw();
            return;
        }

        const visible = [];
        const degrees = this.nodeDegrees || {};
        const attributeName = this.filters.attributeName;
        const attributeValue = this.filters.attributeValue.toLowerCase();

        this.nodes.forEach(node => {
            const degree = degrees[node.id] ?? 0;
            if (degree < this.filters.degreeMin) {
                return;
            }
            if (degree > this.filters.degreeMax) {
                return;
            }

            if (attributeName && attributeValue) {
                const rawValue = node.attributes?.[attributeName];
                const valueString = rawValue === undefined || rawValue === null
                    ? ''
                    : String(rawValue).toLowerCase();

                if (!valueString.includes(attributeValue)) {
                    return;
                }
            }

            visible.push(node.id);
        });

        this.visibleNodes = visible;

        if (this.selectedNode && !this.visibleNodes.includes(this.selectedNode.id)) {
            this.selectedNode = null;
            this.hideAttributePanel();
        }
        this.stats.visibleCount = this.visibleNodes.length;
        this.updateStatsDisplay();
        this.draw();
    }

    handleSearch(query) {
        const lowerQuery = query.toLowerCase();

        const match = this.nodes.find(node => {
            if (node.id.toString() === query) {
                return true;
            }
            if (node.label && node.label.toLowerCase().includes(lowerQuery)) {
                return true;
            }
            return Object.values(node.attributes || {}).some(value => {
                if (value === undefined || value === null) {
                    return false;
                }
                return String(value).toLowerCase().includes(lowerQuery);
            });
        });

        if (match) {
            this.selectedNode = match;
            this.showAttributePanel(match);
            this.centerOnNode(match);
        } else {
            this.selectedNode = null;
            this.hideAttributePanel();
            this.draw();
        }
    }

    centerOnNode(node) {
        if (node.x === undefined || node.y === undefined) {
            return;
        }
        this.camera.x = node.x;
        this.camera.y = node.y;
        this.draw();
    }

    startAnimationLoop() {
        const animate = () => {
            if (!this.isPaused) {
                this.render();
                this.frameCount++;
            }
            this.animationId = requestAnimationFrame(animate);
        };
        this.animationId = requestAnimationFrame(animate);
    }

    draw() {
        this.render();
    }

    pause() {
        this.isPaused = true;
        const playPauseBtn = document.getElementById('play-pause-btn');
        if (playPauseBtn) {
            playPauseBtn.textContent = '▶️ Play';
        }
    }

    resume() {
        this.isPaused = false;
        const playPauseBtn = document.getElementById('play-pause-btn');
        if (playPauseBtn) {
            playPauseBtn.textContent = '⏸️ Pause';
        }
        this.draw();
    }

    autoFitGraph() {
        // Calculate bounding box of all nodes
        if (this.nodes.length === 0) return;

        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;

        this.nodes.forEach(node => {
            const x = node.coords?.[0] ?? 0;
            const y = node.coords?.[1] ?? 0;
            minX = Math.min(minX, x);
            maxX = Math.max(maxX, x);
            minY = Math.min(minY, y);
            maxY = Math.max(maxY, y);
        });

        // Calculate center and size
        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;
        const width = maxX - minX;
        const height = maxY - minY;

        // Calculate zoom to fit with 10% padding
        const canvasWidth = this.canvas.width;
        const canvasHeight = this.canvas.height;
        const zoomX = (canvasWidth * 0.9) / width;
        const zoomY = (canvasHeight * 0.9) / height;
        const zoom = Math.min(zoomX, zoomY, 5); // Max zoom of 5

        // Apply camera transform
        this.camera.x = centerX;
        this.camera.y = centerY;
        this.camera.zoom = zoom || 1;
        this.camera.targetZoom = zoom || 1;

        this.draw();
    }

    render() {
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;

        // Clear canvas
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, width, height);

        // Save context for transformations
        ctx.save();

        // Apply camera transform
        ctx.translate(width / 2, height / 2);
        ctx.scale(this.camera.zoom, this.camera.zoom);
        if (this.camera.rotation) {
            ctx.rotate(this.camera.rotation);
        }
        ctx.translate(-this.camera.x, -this.camera.y);

        const visibleSet = this.visibleNodes.length
            ? new Set(this.visibleNodes)
            : null;
        const nodeIndex = new Map(this.nodes.map(node => [node.id, node]));

        // Draw edges
        ctx.globalAlpha = this.opacity;

        this.edges.forEach((edge, idx) => {
            if (visibleSet && (!visibleSet.has(edge.source) || !visibleSet.has(edge.target))) {
                return;
            }

            const sourceNode = nodeIndex.get(edge.source);
            const targetNode = nodeIndex.get(edge.target);

            if (!sourceNode || !targetNode) {
                return;
            }
            if (
                sourceNode.x === undefined ||
                sourceNode.y === undefined ||
                targetNode.x === undefined ||
                targetNode.y === undefined
            ) {
                return;
            }

            // Check if edge is selected, hovered, or connected to selected node
            const isSelected = this.selectedEdge && this.selectedEdge.id === edge.id;
            const isHovered = this.hoveredEdge && this.hoveredEdge.id === edge.id;
            const isConnectedToSelected = this.selectedNode && 
                (edge.source === this.selectedNode.id || edge.target === this.selectedNode.id);

            // Apply edge styling from VizConfig with selection/hover highlight
            if (isSelected) {
                ctx.strokeStyle = '#ff6b6b'; // Red for selection
                ctx.lineWidth = 3 / this.camera.zoom;
                ctx.globalAlpha = this.opacity;
            } else if (isHovered) {
                ctx.strokeStyle = '#ffa500'; // Orange for hover
                ctx.lineWidth = 2 / this.camera.zoom;
                ctx.globalAlpha = this.opacity;
            } else if (isConnectedToSelected) {
                // Highlight edges connected to selected node
                ctx.strokeStyle = edge.color || '#4a90e2'; // Use edge color or blue
                ctx.lineWidth = (edge.width || 1) * 1.5 / this.camera.zoom; // Slightly thicker
                ctx.globalAlpha = this.opacity * 0.9; // Nearly full opacity
            } else if (this.selectedNode) {
                // Dim edges not connected to selected node
                ctx.strokeStyle = edge.color || '#cccccc';
                ctx.lineWidth = (edge.width || 1) / this.camera.zoom;
                ctx.globalAlpha = this.opacity * 0.2; // Dimmed
            } else {
                ctx.strokeStyle = edge.color || '#cccccc';
                ctx.lineWidth = (edge.width || 1) / this.camera.zoom;
                ctx.globalAlpha = this.opacity;
            }

            if (edge.opacity !== undefined && !isConnectedToSelected) {
                ctx.globalAlpha = edge.opacity * this.opacity;
            }

            ctx.beginPath();

            // Handle edge styles (solid, dashed, dotted)
            if (edge.style === 'dashed') {
                ctx.setLineDash([5 / this.camera.zoom, 5 / this.camera.zoom]);
            } else if (edge.style === 'dotted') {
                ctx.setLineDash([2 / this.camera.zoom, 3 / this.camera.zoom]);
            } else {
                ctx.setLineDash([]);
            }

            // Draw edge with curvature support
            // Edge curvature: multiply individual curvature by global multiplier
            // This allows scaling all curvatures up/down while preserving relative differences
            const edgeCurvature = edge.curvature || 0;
            const curvature = edgeCurvature * this.curvatureMultiplier;

            // Check if this is a self-loop
            const isSelfLoop = edge.source === edge.target;

            if (isSelfLoop) {
                // Draw self-loop as a circle, positioned based on curvature for multi-self-loops
                const baseRadius = 20 / this.camera.zoom;
                const loopRadius = baseRadius + Math.abs(edgeCurvature) * 5 / this.camera.zoom;

                // Angle offset for multiple self-loops (use curvature to determine angle)
                const angleOffset = edgeCurvature * Math.PI / 4; // Up to 45 degrees per curvature unit
                const angle = -Math.PI / 2 + angleOffset; // Start at top, offset by curvature

                const loopCenterX = sourceNode.x + Math.cos(angle) * (loopRadius + 5 / this.camera.zoom);
                const loopCenterY = sourceNode.y + Math.sin(angle) * (loopRadius + 5 / this.camera.zoom);

                ctx.arc(loopCenterX, loopCenterY, loopRadius, 0, Math.PI * 2);
            } else if (curvature !== 0) {
                // Draw curved edge using quadratic Bezier curve
                const dx = targetNode.x - sourceNode.x;
                const dy = targetNode.y - sourceNode.y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                // Control point perpendicular to the edge
                const midX = (sourceNode.x + targetNode.x) / 2;
                const midY = (sourceNode.y + targetNode.y) / 2;

                // Perpendicular vector
                const perpX = -dy / dist;
                const perpY = dx / dist;

                // Offset control point by curvature amount
                const offset = curvature * dist * 0.2; // 0.2 is curvature strength factor
                const controlX = midX + perpX * offset;
                const controlY = midY + perpY * offset;

                ctx.moveTo(sourceNode.x, sourceNode.y);
                ctx.quadraticCurveTo(controlX, controlY, targetNode.x, targetNode.y);
            } else {
                // Draw straight edge
                ctx.moveTo(sourceNode.x, sourceNode.y);
                ctx.lineTo(targetNode.x, targetNode.y);
            }
            ctx.stroke();

            // Draw edge label
            const edgeLabel = edge.label;
            if (edgeLabel) {
                ctx.save();
                ctx.globalAlpha = 1.0;

                const edgeFontSize = edge.label_size || 10;
                ctx.font = `${edgeFontSize / this.camera.zoom}px Arial`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';

                // Calculate label position and rotation
                let labelX, labelY, labelAngle = 0;

                if (isSelfLoop) {
                    // For self-loops, place label on the loop circle
                    const baseRadius = 20 / this.camera.zoom;
                    const loopRadius = baseRadius + Math.abs(edgeCurvature) * 5 / this.camera.zoom;
                    const angleOffset = edgeCurvature * Math.PI / 4;
                    const angle = -Math.PI / 2 + angleOffset;

                    const loopCenterX = sourceNode.x + Math.cos(angle) * (loopRadius + 5 / this.camera.zoom);
                    const loopCenterY = sourceNode.y + Math.sin(angle) * (loopRadius + 5 / this.camera.zoom);

                    // Place label at top of loop circle
                    labelX = loopCenterX;
                    labelY = loopCenterY - loopRadius - 3 / this.camera.zoom;
                    labelAngle = 0; // Horizontal text for self-loops
                } else if (curvature !== 0) {
                    // For curved edges, calculate exact position on the Bezier curve
                    const dx = targetNode.x - sourceNode.x;
                    const dy = targetNode.y - sourceNode.y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    const midX = (sourceNode.x + targetNode.x) / 2;
                    const midY = (sourceNode.y + targetNode.y) / 2;
                    const perpX = -dy / dist;
                    const perpY = dx / dist;

                    // Calculate control point (same as edge drawing)
                    const offset = curvature * dist * 0.2;
                    const controlX = midX + perpX * offset;
                    const controlY = midY + perpY * offset;

                    // Quadratic Bezier at t=0.5 (midpoint of curve)
                    // B(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2
                    // At t=0.5: B(0.5) = 0.25*P0 + 0.5*P1 + 0.25*P2
                    const t = 0.5;
                    labelX = (1-t)*(1-t)*sourceNode.x + 2*(1-t)*t*controlX + t*t*targetNode.x;
                    labelY = (1-t)*(1-t)*sourceNode.y + 2*(1-t)*t*controlY + t*t*targetNode.y;

                    // Calculate tangent at t=0.5 for rotation
                    // B'(t) = 2(1-t)(P1-P0) + 2t(P2-P1)
                    const tangentX = 2*(1-t)*(controlX - sourceNode.x) + 2*t*(targetNode.x - controlX);
                    const tangentY = 2*(1-t)*(controlY - sourceNode.y) + 2*t*(targetNode.y - controlY);
                    labelAngle = Math.atan2(tangentY, tangentX);

                    // Keep text upright (don't flip it upside down)
                    if (labelAngle > Math.PI / 2 || labelAngle < -Math.PI / 2) {
                        labelAngle += Math.PI;
                    }
                } else {
                    // For straight edges, place label at midpoint
                    labelX = (sourceNode.x + targetNode.x) / 2;
                    labelY = (sourceNode.y + targetNode.y) / 2;

                    // Calculate angle along the edge
                    const dx = targetNode.x - sourceNode.x;
                    const dy = targetNode.y - sourceNode.y;
                    labelAngle = Math.atan2(dy, dx);

                    // Keep text upright (don't flip it upside down)
                    if (labelAngle > Math.PI / 2 || labelAngle < -Math.PI / 2) {
                        labelAngle += Math.PI;
                    }
                }

                // Rotate text to align with edge
                ctx.translate(labelX, labelY);
                ctx.rotate(labelAngle);

                // Draw label with white background for readability
                const textMetrics = ctx.measureText(edgeLabel);
                const textWidth = textMetrics.width;
                const textHeight = edgeFontSize / this.camera.zoom;
                const padding = 2 / this.camera.zoom;

                ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
                ctx.fillRect(
                    -textWidth / 2 - padding,
                    -textHeight / 2 - padding,
                    textWidth + padding * 2,
                    textHeight + padding * 2
                );

                // Draw label text
                ctx.fillStyle = edge.label_color || '#333333';
                ctx.fillText(edgeLabel, 0, 0);
                ctx.restore();
            }

            // Reset line dash and alpha
            ctx.setLineDash([]);
            ctx.globalAlpha = this.opacity;
        });

        // Draw nodes
        this.nodes.forEach(node => {
            if (node.x === undefined || node.y === undefined) {
                return;
            }
            if (visibleSet && !visibleSet.has(node.id)) {
                return;
            }

            // Check if node is selected or hovered
            const isSelected = this.selectedNode && this.selectedNode.id === node.id;
            const isHovered = this.hoveredNode && this.hoveredNode.id === node.id;

            // Apply node styling from VizConfig with selection/hover highlight
            const nodeColor = node.color || '#007bff';
            const nodeSize = (node.size || 6) / this.camera.zoom;
            const nodeOpacity = node.opacity !== undefined ? node.opacity : 1.0;
            const nodeShape = node.shape || 'circle';

            if (isSelected) {
                ctx.fillStyle = '#ff6b6b'; // Red for selection
            } else if (isHovered) {
                ctx.fillStyle = '#ffa500'; // Orange for hover
            } else {
                ctx.fillStyle = nodeColor;
            }
            ctx.globalAlpha = nodeOpacity * this.opacity;

            ctx.beginPath();

            // Draw different shapes
            if (nodeShape === 'square' || nodeShape === 'rectangle') {
                const size = nodeSize * 1.5;
                ctx.rect(node.x - size, node.y - size, size * 2, size * 2);
            } else if (nodeShape === 'triangle') {
                const size = nodeSize * 1.8;
                ctx.moveTo(node.x, node.y - size);
                ctx.lineTo(node.x - size, node.y + size);
                ctx.lineTo(node.x + size, node.y + size);
                ctx.closePath();
            } else if (nodeShape === 'diamond') {
                const size = nodeSize * 1.5;
                ctx.moveTo(node.x, node.y - size);
                ctx.lineTo(node.x + size, node.y);
                ctx.lineTo(node.x, node.y + size);
                ctx.lineTo(node.x - size, node.y);
                ctx.closePath();
            } else {
                // Default circle
                ctx.arc(node.x, node.y, nodeSize, 0, 2 * Math.PI);
            }

            ctx.fill();

            // Draw border if specified
            if (node.border_width && node.border_width > 0) {
                ctx.strokeStyle = node.border_color || '#000000';
                ctx.lineWidth = node.border_width / this.camera.zoom;
                ctx.stroke();
            }

            // Draw node label
            const label = node.label !== undefined ? node.label : (this.camera.zoom > 0.5 ? node.attributes?.label : null);
            if (label) {
                ctx.save();
                ctx.globalAlpha = 1.0;
                ctx.fillStyle = node.label_color || '#333333';
                const fontSize = node.label_size || 12;
                ctx.font = `${fontSize / this.camera.zoom}px Arial`;
                ctx.textAlign = 'center';
                ctx.fillText(label, node.x, node.y - (nodeSize + 4 / this.camera.zoom));
                ctx.restore();
            }

            // Reset alpha
            ctx.globalAlpha = this.opacity;
        });

        ctx.globalAlpha = 1.0;

        // Restore context
        ctx.restore();
    }

    startStatsUpdater() {
        let lastTime = Date.now();
        let lastFrameCount = 0;
        let lastUpdateCount = 0;

        setInterval(() => {
            const now = Date.now();
            const deltaTime = (now - lastTime) / 1000;

            // Calculate FPS
            const frameDelta = this.frameCount - lastFrameCount;
            this.stats.fps = Math.round(frameDelta / deltaTime);

            // Calculate update rate
            const updateDelta = this.updateCount - lastUpdateCount;
            this.stats.updateRate = Math.round(updateDelta / deltaTime);

            // Update UI
            this.updateStatsDisplay();

            // Reset counters
            lastTime = now;
            lastFrameCount = this.frameCount;
            lastUpdateCount = this.updateCount;
        }, 1000);
    }

    updateStatsDisplay() {
        document.getElementById('node-count').textContent = this.stats.nodeCount;
        document.getElementById('edge-count').textContent = this.stats.edgeCount;
        document.getElementById('visible-count').textContent = this.stats.visibleCount;
        document.getElementById('fps-counter').textContent = this.stats.fps;
        document.getElementById('update-rate').textContent = this.stats.updateRate;
        document.getElementById('latency').textContent = `${this.stats.latency}ms`;
    }
}

// ============================================================================
// View Switching & Table Renderer
// ============================================================================

class ViewManager {
    constructor(app) {
        this.app = app;
        this.currentView = 'graph'; // 'graph' or 'table'
        this.tableRenderer = new TableRenderer(app);
        this.initViewToggle();
    }

    initViewToggle() {
        const toggleBtn = document.getElementById('view-toggle');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => this.toggleView());
        }
    }

    toggleView() {
        if (this.currentView === 'graph') {
            this.switchToTable();
        } else {
            this.switchToGraph();
        }
    }

    switchToTable() {
        this.currentView = 'table';

        // Pause graph animation/physics
        this.app.pause();

        // Hide graph view
        const canvasView = document.getElementById('canvas-view');
        if (canvasView) canvasView.style.display = 'none';

        // Show table view
        const tableView = document.getElementById('table-view');
        if (tableView) tableView.style.display = 'flex';

        // Update toggle button
        const toggleBtn = document.getElementById('view-toggle');
        if (toggleBtn) {
            toggleBtn.querySelector('.view-icon').textContent = '📈';
            toggleBtn.querySelector('.view-label').textContent = 'Graph';
            toggleBtn.title = 'Switch to graph view';
        }

        // Request table data from server
        this.tableRenderer.requestTableData();
    }

    switchToGraph() {
        this.currentView = 'graph';

        // Show graph view first
        const canvasView = document.getElementById('canvas-view');
        if (canvasView) canvasView.style.display = 'block';

        // Hide table view
        const tableView = document.getElementById('table-view');
        if (tableView) tableView.style.display = 'none';

        // Resume graph animation/physics
        this.app.resume();
        
        // Force a canvas resize and redraw
        this.app.resizeCanvas();
        this.app.render();

        // Update toggle button
        const toggleBtn = document.getElementById('view-toggle');
        if (toggleBtn) {
            toggleBtn.querySelector('.view-icon').textContent = '📊';
            toggleBtn.querySelector('.view-label').textContent = 'Table';
            toggleBtn.title = 'Switch to table view';
        }
    }
}

class TableRenderer {
    constructor(app) {
        this.app = app;
        this.currentData = null;
        this.currentOffset = 0;
        this.windowSize = 100; // Number of rows to load at once
        this.bufferSize = 50; // Load more when this many rows from edge
        this.totalRows = 0;
        this.currentDataType = 'nodes'; // 'nodes' or 'edges'
        this.isLoading = false;
        this.loadedRanges = []; // Track what ranges we've loaded: [{start, end}]
        this.allRows = new Map(); // Cache of all loaded rows by index

        // Sorting state: array of {column: string, direction: 'asc'|'desc'}
        this.sortColumns = [];

        this.initInfiniteScroll();
        this.initTableTypeToggle();
    }

    initInfiniteScroll() {
        const tableScrollWrapper = document.querySelector('.table-scroll-wrapper');
        if (!tableScrollWrapper) return;

        // Set up scroll event listener for infinite scroll
        tableScrollWrapper.addEventListener('scroll', () => this.handleScroll());
    }

    handleScroll() {
        const tableScrollWrapper = document.querySelector('.table-scroll-wrapper');
        if (!tableScrollWrapper || this.isLoading) return;

        const scrollTop = tableScrollWrapper.scrollTop;
        const scrollHeight = tableScrollWrapper.scrollHeight;
        const clientHeight = tableScrollWrapper.clientHeight;

        // Calculate approximate row that's at the top of viewport
        const estimatedRowHeight = 30; // Approximate height per row
        const topRowIndex = Math.floor(scrollTop / estimatedRowHeight);
        const bottomRowIndex = Math.floor((scrollTop + clientHeight) / estimatedRowHeight);

        // Check if we need to load more data near the top
        if (topRowIndex < this.bufferSize && this.currentOffset > 0) {
            const newOffset = Math.max(0, this.currentOffset - this.windowSize);
            this.loadMoreData(newOffset);
        }
        // Check if we need to load more data near the bottom
        else if (bottomRowIndex > this.currentOffset + this.windowSize - this.bufferSize 
                 && this.currentOffset + this.windowSize < this.totalRows) {
            const newOffset = this.currentOffset + this.windowSize;
            this.loadMoreData(newOffset);
        }
    }

    loadMoreData(offset) {
        if (this.isLoading) return;
        
        this.isLoading = true;
        this.requestTableData(offset);
    }

    initTableTypeToggle() {
        const nodesBtn = document.getElementById('table-nodes-btn');
        const edgesBtn = document.getElementById('table-edges-btn');

        if (nodesBtn) {
            nodesBtn.addEventListener('click', () => this.switchToNodes());
        }

        if (edgesBtn) {
            edgesBtn.addEventListener('click', () => this.switchToEdges());
        }
    }

    switchToNodes() {
        this.currentDataType = 'nodes';
        this.currentOffset = 0;
        this.allRows.clear();
        this.loadedRanges = [];

        // Update button states
        document.getElementById('table-nodes-btn')?.classList.add('active');
        document.getElementById('table-edges-btn')?.classList.remove('active');

        // Clear table body and header for fresh load
        const tableBody = document.getElementById('table-body');
        const tableHeader = document.getElementById('table-header').querySelector('tr');
        if (tableBody) tableBody.innerHTML = '';
        if (tableHeader) tableHeader.innerHTML = '';

        this.requestTableData();
    }

    switchToEdges() {
        this.currentDataType = 'edges';
        this.currentOffset = 0;
        this.allRows.clear();
        this.loadedRanges = [];

        // Update button states
        document.getElementById('table-nodes-btn')?.classList.remove('active');
        document.getElementById('table-edges-btn')?.classList.add('active');

        // Clear table body and header for fresh load
        const tableBody = document.getElementById('table-body');
        const tableHeader = document.getElementById('table-header').querySelector('tr');
        if (tableBody) tableBody.innerHTML = '';
        if (tableHeader) tableHeader.innerHTML = '';

        this.requestTableData();
    }

    requestTableData(offset = 0) {
        this.currentOffset = offset;

        // Send WebSocket request for table data
        if (this.app.ws && this.app.ws.readyState === WebSocket.OPEN) {
            const message = {
                type: "RequestTableData",
                offset: offset,
                window_size: this.windowSize,
                data_type: this.currentDataType,
                sort_columns: this.sortColumns // Add sorting information
            };

            console.log('📤 Requesting table data:', message);
            this.app.ws.send(JSON.stringify(message));
        } else {
            console.warn('WebSocket not connected, showing placeholder');
            this.renderPlaceholder();
            this.isLoading = false;
        }
    }

    handleTableData(data) {
        // Handle incoming table data from WebSocket
        console.log('🎯 TableRenderer.handleTableData called with:', data);
        console.log('  - Headers:', data.headers);
        console.log('  - Rows:', data.rows?.length || 0);
        console.log('  - Total rows:', data.total_rows);
        console.log('  - Start offset:', data.start_offset || 0);

        this.currentData = data;
        this.totalRows = data.total_rows;
        
        // Cache the loaded rows
        const startOffset = data.start_offset || 0;
        data.rows.forEach((row, idx) => {
            this.allRows.set(startOffset + idx, row);
        });
        
        this.renderTableData(data);
        this.isLoading = false;
    }

    renderPlaceholder() {
        const tableBody = document.getElementById('table-body');
        const tableHeader = document.getElementById('table-header').querySelector('tr');
        
        if (!tableBody || !tableHeader) return;
        
        // Clear existing content
        tableBody.innerHTML = '';
        tableHeader.innerHTML = '';
        
        // Add placeholder headers
        const headers = ['ID', 'Name', 'Type', 'Degree', 'Attributes'];
        headers.forEach(header => {
            const th = document.createElement('th');
            th.textContent = header;
            tableHeader.appendChild(th);
        });
        
        // Add placeholder rows
        for (let i = 0; i < 10; i++) {
            const tr = document.createElement('tr');
            headers.forEach(() => {
                const td = document.createElement('td');
                td.textContent = '...';
                tr.appendChild(td);
            });
            tableBody.appendChild(tr);
        }
        
        // Update footer
        const rowInfo = document.getElementById('table-row-info');
        if (rowInfo) {
            rowInfo.textContent = 'Waiting for data...';
        }
    }

    renderTableData(data) {
        this.currentData = data;
        this.totalRows = data.total_rows;
        
        const tableBody = document.getElementById('table-body');
        const tableHeader = document.getElementById('table-header').querySelector('tr');
        
        if (!tableBody || !tableHeader) return;
        
        // Only update header if it's empty (first load)
        if (tableHeader.children.length === 0) {
            // Render sortable headers
            data.headers.forEach(header => {
                const th = document.createElement('th');
                th.className = 'sortable-header';
                th.style.cursor = 'pointer';
                th.style.userSelect = 'none';

                const headerContent = document.createElement('span');
                headerContent.textContent = header;

                const sortIcon = document.createElement('span');
                sortIcon.className = 'sort-icon';
                sortIcon.style.marginLeft = '5px';

                // Check current sort state
                const sortState = this.getSortStateForColumn(header);
                if (sortState) {
                    sortIcon.textContent = sortState.direction === 'asc' ? '↑' : '↓';
                    th.classList.add('sorted-' + sortState.direction);
                } else {
                    sortIcon.textContent = '↕';
                    sortIcon.style.opacity = '0.3';
                }

                th.appendChild(headerContent);
                th.appendChild(sortIcon);

                th.addEventListener('click', () => this.handleColumnSort(header));
                tableHeader.appendChild(th);
            });

            // Add sort reset button if any sorting is active
            if (this.sortColumns.length > 0) {
                const resetTh = document.createElement('th');
                resetTh.className = 'sort-reset-header';
                resetTh.style.width = '30px';
                resetTh.style.textAlign = 'center';

                const resetBtn = document.createElement('button');
                resetBtn.textContent = '✕';
                resetBtn.title = 'Reset all sorting';
                resetBtn.style.border = 'none';
                resetBtn.style.background = 'transparent';
                resetBtn.style.cursor = 'pointer';
                resetBtn.style.fontSize = '16px';
                resetBtn.style.color = '#999';

                resetBtn.addEventListener('click', () => this.resetSorting());
                resetTh.appendChild(resetBtn);
                tableHeader.appendChild(resetTh);
            }
        }
        
        // Append new rows (don't clear existing)
        const startOffset = data.start_offset || 0;
        data.rows.forEach((row, idx) => {
            const globalIndex = startOffset + idx;
            
            // Check if this row already exists
            const existingRow = Array.from(tableBody.children).find(
                tr => tr.dataset.rowIndex == globalIndex
            );
            
            if (!existingRow) {
                const tr = document.createElement('tr');
                tr.dataset.rowIndex = globalIndex;
                
                row.forEach(cell => {
                    const td = document.createElement('td');
                    td.textContent = this.formatCellValue(cell);
                    tr.appendChild(td);
                });
                
                // Insert row in the correct position
                const insertBefore = Array.from(tableBody.children).find(
                    tr => parseInt(tr.dataset.rowIndex) > globalIndex
                );
                
                if (insertBefore) {
                    tableBody.insertBefore(tr, insertBefore);
                } else {
                    tableBody.appendChild(tr);
                }
            }
        });
        
        // Update footer
        this.updateTableFooter();
    }

    formatCellValue(value) {
        if (value === null || value === undefined) return '';
        if (typeof value === 'object') return JSON.stringify(value);
        return String(value);
    }

    updateTableFooter() {
        const rowInfo = document.getElementById('table-row-info');
        
        if (!rowInfo) return;
        
        // Show total count and scroll info
        const cachedCount = this.allRows.size;
        rowInfo.textContent = `Loaded ${cachedCount} of ${this.totalRows} rows (scroll for more)`;
    }

    handleTableDataMessage(data) {
        // Handle incoming table data from WebSocket
        this.renderTableData(data.window);
    }

    // Sorting helper methods
    getSortStateForColumn(column) {
        return this.sortColumns.find(sort => sort.column === column);
    }

    handleColumnSort(column) {
        const existingSort = this.getSortStateForColumn(column);

        if (!existingSort) {
            // First click: add ASC sort
            this.sortColumns.push({ column, direction: 'asc' });
        } else if (existingSort.direction === 'asc') {
            // Second click: change to DESC
            existingSort.direction = 'desc';
        } else {
            // Third click: remove from sort
            this.sortColumns = this.sortColumns.filter(sort => sort.column !== column);
        }

        // Reset cache and reload when sorting changes
        this.allRows.clear();
        this.loadedRanges = [];
        this.currentOffset = 0;
        
        // Clear table body and header for fresh sort
        const tableBody = document.getElementById('table-body');
        const tableHeader = document.getElementById('table-header').querySelector('tr');
        if (tableBody) tableBody.innerHTML = '';
        if (tableHeader) tableHeader.innerHTML = '';
        
        this.requestTableData();
    }

    resetSorting() {
        this.sortColumns = [];
        this.allRows.clear();
        this.loadedRanges = [];
        this.currentOffset = 0;
        
        // Clear table for fresh load
        const tableBody = document.getElementById('table-body');
        const tableHeader = document.getElementById('table-header').querySelector('tr');
        if (tableBody) tableBody.innerHTML = '';
        if (tableHeader) tableHeader.innerHTML = '';
        
        this.requestTableData();
    }
}

// Initialize view manager when app is ready
// This will be called from the existing initialization code
