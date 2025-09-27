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
        this.lastUpdateTime = Date.now();
        this.frameCount = 0;
        this.updateCount = 0;
        this.currentEmbeddingMethod = null;
        this.currentEmbeddingDimensions = 2;
        this.opacity = 1.0;
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
                label: node.attributes.label || node.id.toString(),
                x: 0,
                y: 0
            }));
            this.stats.nodeCount = this.nodes.length;
            console.log(`📊 Loaded ${this.nodes.length} nodes`);
        }

        // Load edges
        if (snapshot.edges) {
            this.edges = snapshot.edges.map(edge => ({
                id: edge.id,
                source: edge.source,
                target: edge.target,
                attributes: edge.attributes
            }));
            this.stats.edgeCount = this.edges.length;
            console.log(`📊 Loaded ${this.edges.length} edges`);
        }

        // Apply positions if available
        if (snapshot.positions && snapshot.positions.length > 0) {
            console.log(`📍 Applying ${snapshot.positions.length} node positions`);
            for (const position of snapshot.positions) {
                const node = this.nodes.find(n => n.id === position.node_id);
                if (node && position.coords && position.coords.length >= 2) {
                    node.x = position.coords[0] || 0;
                    node.y = position.coords[1] || 0;
                }
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

        // Reset derived state for filters and recompute visibility
        this.filters.degreeMin = 0;
        this.filters.degreeMax = Number.POSITIVE_INFINITY;
        this.filters.attributeName = '';
        this.filters.attributeValue = '';

        const degreeMinInput = document.getElementById('degree-min');
        const degreeMaxInput = document.getElementById('degree-max');
        if (degreeMinInput) {
            const min = parseInt(degreeMinInput.value, 10);
            if (!Number.isNaN(min)) {
                this.filters.degreeMin = min;
            }
        }
        if (degreeMaxInput) {
            const max = parseInt(degreeMaxInput.value, 10);
            if (!Number.isNaN(max)) {
                this.filters.degreeMax = max;
            }
        }

        const attributeNameInput = document.getElementById('attribute-name');
        const attributeValueInput = document.getElementById('attribute-value');
        if (attributeNameInput && attributeValueInput) {
            this.filters.attributeName = attributeNameInput.value || '';
            this.filters.attributeValue = attributeValueInput.value.trim();
        }

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
        const worldX = (x - this.canvas.width / 2) / this.camera.zoom + this.camera.x;
        const worldY = (y - this.canvas.height / 2) / this.camera.zoom + this.camera.y;
        return { screenX: x, screenY: y, worldX, worldY };
    }

    handleCanvasClick(event) {
        const { worldX, worldY } = this.screenToWorld(event);
        const clickedNode = this.findNodeAtPosition(worldX, worldY);

        if (clickedNode) {
            this.selectedNode = clickedNode;
            this.showAttributePanel(clickedNode);
            console.log('🎯 Selected node:', clickedNode.id);
        } else {
            this.selectedNode = null;
            this.hideAttributePanel();
        }
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
        if (!this.lastPointer) {
            return;
        }

        const dx = event.clientX - this.lastPointer.x;
        const dy = event.clientY - this.lastPointer.y;
        this.lastPointer = { x: event.clientX, y: event.clientY };

        if (this.draggedNode) {
            const coords = this.screenToWorld(event);
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
        const nodeRadius = 8;
        return this.nodes.find(node => {
            if (node.x === undefined || node.y === undefined) return false;
            if (this.visibleNodes.length && !this.visibleNodes.includes(node.id)) return false;
            const dx = node.x - x;
            const dy = node.y - y;
            return Math.sqrt(dx * dx + dy * dy) <= nodeRadius;
        });
    }

    showAttributePanel(node) {
        const panel = document.getElementById('attribute-panel');
        const content = document.getElementById('attribute-content');

        if (node.attributes) {
            let html = '';
            for (const [key, value] of Object.entries(node.attributes)) {
                html += `
                    <div class="attribute-row">
                        <span class="attribute-key">${key}:</span>
                        <span class="attribute-value">${value}</span>
                    </div>
                `;
            }
            content.innerHTML = html;
        } else {
            content.innerHTML = '<p>No attributes available</p>';
        }

        panel.classList.add('visible');
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
        ctx.strokeStyle = '#cccccc';
        ctx.lineWidth = 1 / this.camera.zoom;
        ctx.globalAlpha = this.opacity;

        this.edges.forEach(edge => {
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

            ctx.beginPath();
            ctx.moveTo(sourceNode.x, sourceNode.y);
            ctx.lineTo(targetNode.x, targetNode.y);
            ctx.stroke();
        });

        // Draw nodes
        this.nodes.forEach(node => {
            if (node.x === undefined || node.y === undefined) {
                return;
            }
            if (visibleSet && !visibleSet.has(node.id)) {
                return;
            }

            ctx.fillStyle = '#007bff';
            if (this.selectedNode && this.selectedNode.id === node.id) {
                ctx.fillStyle = '#ff6b6b';
            }

            ctx.beginPath();
            ctx.arc(node.x, node.y, 6 / this.camera.zoom, 0, 2 * Math.PI);
            ctx.fill();

            // Draw node label if zoomed in enough
            if (this.camera.zoom > 0.5 && node.label) {
                ctx.save();
                ctx.globalAlpha = 1.0;
                ctx.fillStyle = '#333333';
                ctx.font = `${12 / this.camera.zoom}px Arial`;
                ctx.textAlign = 'center';
                ctx.fillText(node.label, node.x, node.y - 10 / this.camera.zoom);
                ctx.restore();
            }
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
