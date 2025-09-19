define("groggy-widgets", ["@jupyter-widgets/base"], (__WEBPACK_EXTERNAL_MODULE__jupyter_widgets_base__) => { return /******/ (() => { // webpackBootstrap
/******/ 	var __webpack_modules__ = ({

/***/ "../python/groggy/widgets/widget_view.js":
/*!***********************************************!*\
  !*** ../python/groggy/widgets/widget_view.js ***!
  \***********************************************/
/***/ ((module) => {

/**
 * GroggyGraphView - JavaScript Implementation for Widget
 * 
 * This is a simplified JavaScript view that can be embedded in HTML
 * for testing the widget functionality. In a full Jupyter widget,
 * this would be built as a proper npm package with webpack.
 */

class GroggyGraphView {
    constructor(container, config = {}) {
        this.container = container;
        this.config = {
            width: 800,
            height: 600,
            layout: 'force-directed',
            theme: 'light',
            enableDrag: true,
            enablePan: true,
            enableZoom: true,
            animationDuration: 300,
            ...config
        };
        
        // State
        this.nodes = [];
        this.edges = [];
        this.positions = [];
        this.selectedNodes = [];
        this.hoveredNode = null;
        this.camera = { x: 0, y: 0, zoom: 1.0 };
        
        // Interaction state
        this.isDragging = false;
        this.dragNode = null;
        this.isPanning = false;
        this.lastMouse = { x: 0, y: 0 };
        
        // Callbacks for Python communication
        this.callbacks = {
            nodeClick: [],
            nodeHover: [],
            nodeDoubleClick: [],
            layoutChange: [],
            cameraChange: [],
            selectionChange: []
        };
        
        this.init();
    }
    
    init() {
        // Create canvas
        this.canvas = document.createElement('canvas');
        this.canvas.width = this.config.width;
        this.canvas.height = this.config.height;
        this.canvas.style.border = '1px solid #ddd';
        this.canvas.style.cursor = 'default';
        this.canvas.style.display = 'block';
        
        this.ctx = this.canvas.getContext('2d');
        this.container.appendChild(this.canvas);
        
        // Create controls
        this.createControls();
        
        // Set up event listeners
        this.setupEventListeners();
        
        console.log('GroggyGraphView initialized');
    }
    
    createControls() {
        const controlsDiv = document.createElement('div');
        controlsDiv.style.cssText = `
            margin: 10px 0;
            text-align: center;
        `;
        
        // Layout button
        const layoutBtn = document.createElement('button');
        layoutBtn.textContent = 'Toggle Layout';
        layoutBtn.style.cssText = `
            background: #007bff;
            color: white;
            border: none;
            padding: 8px 16px;
            margin: 4px;
            border-radius: 4px;
            cursor: pointer;
        `;
        layoutBtn.onclick = () => this.toggleLayout();
        
        // Reset button
        const resetBtn = document.createElement('button');
        resetBtn.textContent = 'Reset View';
        resetBtn.style.cssText = layoutBtn.style.cssText.replace('#007bff', '#28a745');
        resetBtn.onclick = () => this.resetCamera();
        
        // Theme button
        const themeBtn = document.createElement('button');
        themeBtn.textContent = 'Toggle Theme';
        themeBtn.style.cssText = layoutBtn.style.cssText.replace('#007bff', '#6c757d');
        themeBtn.onclick = () => this.toggleTheme();
        
        controlsDiv.appendChild(layoutBtn);
        controlsDiv.appendChild(resetBtn);
        controlsDiv.appendChild(themeBtn);
        
        this.container.insertBefore(controlsDiv, this.canvas);
        
        // Info panel
        this.infoPanel = document.createElement('div');
        this.infoPanel.style.cssText = `
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 8px;
            border-radius: 4px;
            font-size: 12px;
            display: none;
        `;
        this.container.style.position = 'relative';
        this.container.appendChild(this.infoPanel);
    }
    
    setupEventListeners() {
        // Mouse events
        this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
        this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
        this.canvas.addEventListener('mouseup', this.onMouseUp.bind(this));
        this.canvas.addEventListener('click', this.onClick.bind(this));
        this.canvas.addEventListener('dblclick', this.onDoubleClick.bind(this));
        this.canvas.addEventListener('wheel', this.onWheel.bind(this));
        this.canvas.addEventListener('mouseleave', this.onMouseLeave.bind(this));
    }
    
    // Public API
    setData(nodes, edges) {
        this.nodes = nodes || [];
        this.edges = edges || [];
        this.calculateLayout();
        this.render();
        console.log(`Data updated: ${this.nodes.length} nodes, ${this.edges.length} edges`);
    }
    
    setLayout(algorithm, animate = true) {
        const oldLayout = this.config.layout;
        this.config.layout = algorithm;
        
        if (animate) {
            this.animateLayoutChange();
        } else {
            this.calculateLayout();
            this.render();
        }
        
        this.trigger('layoutChange', { from: oldLayout, to: algorithm });
        console.log(`Layout changed to: ${algorithm}`);
    }
    
    setTheme(theme) {
        this.config.theme = theme;
        this.render();
        console.log(`Theme changed to: ${theme}`);
    }
    
    selectNodes(nodeIds, clearExisting = true) {
        if (clearExisting) {
            this.selectedNodes = [...nodeIds];
        } else {
            this.selectedNodes = [...new Set([...this.selectedNodes, ...nodeIds])];
        }
        this.render();
        this.trigger('selectionChange', this.selectedNodes);
    }
    
    focusOnNode(nodeId, zoomLevel = 2.0) {
        const node = this.nodes.find(n => n.id === nodeId);
        const pos = this.positions.find(p => p.id === nodeId);
        
        if (node && pos) {
            // Center camera on node
            this.camera.x = this.config.width / 2 - pos.x * this.camera.zoom;
            this.camera.y = this.config.height / 2 - pos.y * this.camera.zoom;
            this.camera.zoom = zoomLevel;
            
            this.render();
            this.trigger('cameraChange', this.camera);
        }
    }
    
    resetCamera() {
        this.camera = { x: 0, y: 0, zoom: 1.0 };
        this.calculateLayout();
        this.render();
        this.trigger('cameraChange', this.camera);
    }
    
    // Event handling
    onMouseDown(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left - this.camera.x) / this.camera.zoom;
        const y = (e.clientY - rect.top - this.camera.y) / this.camera.zoom;
        
        const clickedNode = this.findNodeAt(x, y);
        
        if (clickedNode && this.config.enableDrag) {
            this.isDragging = true;
            this.dragNode = clickedNode;
            this.canvas.style.cursor = 'grabbing';
        } else if (this.config.enablePan) {
            this.isPanning = true;
            this.lastMouse = { x: e.clientX, y: e.clientY };
            this.canvas.style.cursor = 'grabbing';
        }
    }
    
    onMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left - this.camera.x) / this.camera.zoom;
        const y = (e.clientY - rect.top - this.camera.y) / this.camera.zoom;
        
        if (this.isDragging && this.dragNode) {
            // Update node position
            const pos = this.positions.find(p => p.id === this.dragNode.id);
            if (pos) {
                pos.x = x;
                pos.y = y;
                this.render();
            }
        } else if (this.isPanning) {
            // Update camera position
            this.camera.x += e.clientX - this.lastMouse.x;
            this.camera.y += e.clientY - this.lastMouse.y;
            this.lastMouse = { x: e.clientX, y: e.clientY };
            this.render();
        } else {
            // Hover detection
            const hoveredNode = this.findNodeAt(x, y);
            if (hoveredNode !== this.hoveredNode) {
                this.hoveredNode = hoveredNode;
                this.canvas.style.cursor = hoveredNode ? 'pointer' : 'default';
                
                if (hoveredNode) {
                    this.infoPanel.textContent = `${hoveredNode.label || hoveredNode.id} (${hoveredNode.id})`;
                    this.infoPanel.style.display = 'block';
                } else {
                    this.infoPanel.style.display = 'none';
                }
                
                this.render();
                this.trigger('nodeHover', hoveredNode ? hoveredNode.id : null);
            }
        }
    }
    
    onMouseUp() {
        this.isDragging = false;
        this.isPanning = false;
        this.dragNode = null;
        this.canvas.style.cursor = 'default';
    }
    
    onClick(e) {
        if (!this.isDragging) {
            const rect = this.canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left - this.camera.x) / this.camera.zoom;
            const y = (e.clientY - rect.top - this.camera.y) / this.camera.zoom;
            
            const clickedNode = this.findNodeAt(x, y);
            if (clickedNode) {
                // Handle selection
                if (e.ctrlKey || e.metaKey) {
                    // Add to selection
                    if (this.selectedNodes.includes(clickedNode.id)) {
                        this.selectedNodes = this.selectedNodes.filter(id => id !== clickedNode.id);
                    } else {
                        this.selectedNodes.push(clickedNode.id);
                    }
                } else {
                    // Replace selection
                    this.selectedNodes = [clickedNode.id];
                }
                
                this.render();
                this.trigger('nodeClick', { nodeId: clickedNode.id, nodeData: clickedNode });
                this.trigger('selectionChange', this.selectedNodes);
            } else {
                // Clear selection
                if (!e.ctrlKey && !e.metaKey) {
                    this.selectedNodes = [];
                    this.render();
                    this.trigger('selectionChange', this.selectedNodes);
                }
            }
        }
    }
    
    onDoubleClick(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left - this.camera.x) / this.camera.zoom;
        const y = (e.clientY - rect.top - this.camera.y) / this.camera.zoom;
        
        const clickedNode = this.findNodeAt(x, y);
        if (clickedNode) {
            this.trigger('nodeDoubleClick', { nodeId: clickedNode.id, nodeData: clickedNode });
            this.focusOnNode(clickedNode.id, 2.0);
        }
    }
    
    onWheel(e) {
        if (this.config.enableZoom) {
            e.preventDefault();
            const rect = this.canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            
            const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
            const newZoom = Math.max(0.1, Math.min(5.0, this.camera.zoom * zoomFactor));
            
            // Zoom toward mouse position
            this.camera.x = mouseX - (mouseX - this.camera.x) * (newZoom / this.camera.zoom);
            this.camera.y = mouseY - (mouseY - this.camera.y) * (newZoom / this.camera.zoom);
            this.camera.zoom = newZoom;
            
            this.render();
            this.trigger('cameraChange', this.camera);
        }
    }
    
    onMouseLeave() {
        this.hoveredNode = null;
        this.infoPanel.style.display = 'none';
        this.render();
    }
    
    // Utility methods
    findNodeAt(x, y) {
        return this.nodes.find(node => {
            const pos = this.positions.find(p => p.id === node.id);
            if (pos) {
                const radius = node.size || 8;
                const dx = x - pos.x;
                const dy = y - pos.y;
                return (dx * dx + dy * dy) <= (radius * radius);
            }
            return false;
        });
    }
    
    toggleLayout() {
        const layouts = ['circular', 'grid', 'force-directed'];
        const currentIndex = layouts.indexOf(this.config.layout);
        const nextLayout = layouts[(currentIndex + 1) % layouts.length];
        this.setLayout(nextLayout, true);
    }
    
    toggleTheme() {
        const themes = ['light', 'dark'];
        const currentIndex = themes.indexOf(this.config.theme);
        const nextTheme = themes[(currentIndex + 1) % themes.length];
        this.setTheme(nextTheme);
    }
    
    // Layout algorithms
    calculateLayout() {
        const { width, height, layout } = this.config;
        const padding = 50;
        this.positions = [];
        
        if (layout === 'circular' && this.nodes.length > 0) {
            const centerX = width / 2;
            const centerY = height / 2;
            const radius = Math.min(width, height) / 2 - padding;
            
            this.nodes.forEach((node, i) => {
                const angle = (i * 2 * Math.PI) / this.nodes.length;
                this.positions.push({
                    id: node.id,
                    x: centerX + radius * Math.cos(angle),
                    y: centerY + radius * Math.sin(angle)
                });
            });
        } else if (layout === 'force-directed') {
            // Initialize random positions
            this.positions = this.nodes.map(node => ({
                id: node.id,
                x: padding + Math.random() * (width - 2 * padding),
                y: padding + Math.random() * (height - 2 * padding)
            }));
            
            // Force simulation
            for (let iter = 0; iter < 50; iter++) {
                // Repulsion
                for (let i = 0; i < this.positions.length; i++) {
                    for (let j = i + 1; j < this.positions.length; j++) {
                        const dx = this.positions[j].x - this.positions[i].x;
                        const dy = this.positions[j].y - this.positions[i].y;
                        const distance = Math.sqrt(dx * dx + dy * dy) || 1;
                        const force = 500 / (distance * distance);
                        
                        this.positions[i].x -= force * dx / distance;
                        this.positions[i].y -= force * dy / distance;
                        this.positions[j].x += force * dx / distance;
                        this.positions[j].y += force * dy / distance;
                    }
                }
                
                // Attraction for edges
                this.edges.forEach(edge => {
                    const src = this.positions.find(p => p.id === edge.source);
                    const dst = this.positions.find(p => p.id === edge.target);
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
                this.positions.forEach(pos => {
                    pos.x = Math.max(padding, Math.min(width - padding, pos.x));
                    pos.y = Math.max(padding, Math.min(height - padding, pos.y));
                });
            }
        } else {
            // Grid layout
            const cols = Math.ceil(Math.sqrt(this.nodes.length));
            const cellW = (width - 2 * padding) / cols;
            const cellH = (height - 2 * padding) / Math.ceil(this.nodes.length / cols);
            
            this.nodes.forEach((node, i) => {
                this.positions.push({
                    id: node.id,
                    x: padding + (i % cols) * cellW + cellW / 2,
                    y: padding + Math.floor(i / cols) * cellH + cellH / 2
                });
            });
        }
    }
    
    animateLayoutChange() {
        const oldPositions = [...this.positions];
        this.calculateLayout();
        const newPositions = [...this.positions];
        
        // Simple animation (could be enhanced with proper easing)
        const startTime = Date.now();
        const duration = this.config.animationDuration;
        
        const animate = () => {
            const progress = Math.min(1, (Date.now() - startTime) / duration);
            
            this.positions = oldPositions.map((oldPos, i) => {
                const newPos = newPositions[i];
                return {
                    id: oldPos.id,
                    x: oldPos.x + (newPos.x - oldPos.x) * progress,
                    y: oldPos.y + (newPos.y - oldPos.y) * progress
                };
            });
            
            this.render();
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        
        animate();
    }
    
    // Rendering
    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        this.ctx.save();
        this.ctx.translate(this.camera.x, this.camera.y);
        this.ctx.scale(this.camera.zoom, this.camera.zoom);
        
        this.drawEdges();
        this.drawNodes();
        
        this.ctx.restore();
        
        // Draw UI elements (not affected by camera)
        this.drawStats();
    }
    
    drawEdges() {
        const { theme } = this.config;
        
        this.edges.forEach(edge => {
            const src = this.positions.find(p => p.id === edge.source);
            const dst = this.positions.find(p => p.id === edge.target);
            if (src && dst) {
                this.ctx.strokeStyle = theme === 'dark' ? '#666' : '#999';
                this.ctx.lineWidth = 1;
                this.ctx.beginPath();
                this.ctx.moveTo(src.x, src.y);
                this.ctx.lineTo(dst.x, dst.y);
                this.ctx.stroke();
            }
        });
    }
    
    drawNodes() {
        const { theme } = this.config;
        
        this.nodes.forEach(node => {
            const pos = this.positions.find(p => p.id === node.id);
            if (pos) {
                const isSelected = this.selectedNodes.includes(node.id);
                const isHovered = this.hoveredNode && this.hoveredNode.id === node.id;
                const radius = (node.size || 8) * (isHovered ? 1.3 : 1);
                
                // Shadow for hover
                if (isHovered) {
                    this.ctx.shadowColor = 'rgba(0,0,0,0.3)';
                    this.ctx.shadowBlur = 10;
                    this.ctx.shadowOffsetX = 2;
                    this.ctx.shadowOffsetY = 2;
                }
                
                // Node circle
                this.ctx.fillStyle = node.color || (theme === 'dark' ? '#4CAF50' : '#007bff');
                this.ctx.beginPath();
                this.ctx.arc(pos.x, pos.y, radius, 0, 2 * Math.PI);
                this.ctx.fill();
                
                this.ctx.shadowColor = 'transparent';
                this.ctx.shadowBlur = 0;
                
                // Border
                if (isSelected || isHovered) {
                    this.ctx.strokeStyle = isSelected ? '#ff0000' : '#ffa500';
                    this.ctx.lineWidth = 3;
                    this.ctx.stroke();
                }
                
                // Label
                this.ctx.fillStyle = theme === 'dark' ? '#fff' : '#000';
                this.ctx.font = '12px Arial';
                this.ctx.textAlign = 'center';
                this.ctx.fillText(node.label || node.id, pos.x, pos.y - radius - 5);
            }
        });
    }
    
    drawStats() {
        const stats = `${this.nodes.length} nodes | ${this.edges.length} edges | ${this.config.layout} | zoom: ${this.camera.zoom.toFixed(1)}x`;
        this.ctx.fillStyle = this.config.theme === 'dark' ? '#fff' : '#666';
        this.ctx.font = '12px Arial';
        this.ctx.textAlign = 'left';
        this.ctx.fillText(stats, 10, this.canvas.height - 10);
    }
    
    // Event system
    on(event, callback) {
        if (!this.callbacks[event]) {
            this.callbacks[event] = [];
        }
        this.callbacks[event].push(callback);
    }
    
    trigger(event, data) {
        if (this.callbacks[event]) {
            this.callbacks[event].forEach(callback => {
                try {
                    callback(data);
                } catch (e) {
                    console.error(`Error in ${event} callback:`, e);
                }
            });
        }
    }
}

// Export for module systems
if ( true && module.exports) {
    module.exports = GroggyGraphView;
}

/***/ }),

/***/ "./src/core.ts":
/*!*********************!*\
  !*** ./src/core.ts ***!
  \*********************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   JupyterGroggyView: () => (/* binding */ JupyterGroggyView)
/* harmony export */ });
/**
 * Elegant Core Bridge - Single Source of Truth
 *
 * This module provides the elegant abstraction layer that bridges
 * our existing GroggyGraphView with Jupyter's widget system.
 *
 * Philosophy: Extend, don't duplicate. One codebase, multiple interfaces.
 */
// Elegant import of our core visualization engine
// Using relative path for maximum elegance and portability
const GroggyGraphView = (__webpack_require__(/*! ../../python/groggy/widgets/widget_view.js */ "../python/groggy/widgets/widget_view.js").GroggyGraphView);
/**
 * Enhanced GroggyGraphView with Jupyter widget synchronization.
 *
 * This elegant wrapper extends our core visualization engine with
 * bidirectional communication capabilities for Jupyter widgets.
 */
class JupyterGroggyView {
    constructor(element, model) {
        this.syncCallbacks = new Map();
        this.element = element;
        this.model = model;
        // Initialize our elegant core engine
        this.coreView = new GroggyGraphView(element, this.extractCoreConfig());
        // Set up elegant bidirectional sync
        this.setupElegantSync();
    }
    /**
     * Extract core configuration from Jupyter model traits.
     * Elegant transformation: Jupyter traits → Core config
     */
    extractCoreConfig() {
        return {
            width: this.model.get('width') || 800,
            height: this.model.get('height') || 600,
            layout: this.model.get('layout_algorithm') || 'force-directed',
            theme: this.model.get('theme') || 'light',
            enableDrag: this.model.get('enable_drag') !== false,
            enablePan: this.model.get('enable_pan') !== false,
            enableZoom: this.model.get('enable_zoom') !== false,
            animationDuration: this.model.get('animation_duration') || 300
        };
    }
    /**
     * Elegant bidirectional synchronization setup.
     * Core events → Jupyter model updates
     * Jupyter model changes → Core engine updates
     */
    setupElegantSync() {
        // Elegant: Core → Jupyter synchronization
        this.syncFromCore();
        // Elegant: Jupyter → Core synchronization  
        this.syncFromJupyter();
    }
    /**
     * Sync core engine events to Jupyter model (Core → Jupyter)
     */
    syncFromCore() {
        // Node interactions
        this.coreView.on('nodeClick', (data) => {
            this.model.send({
                type: 'node_click',
                node_id: data.nodeId,
                node_data: data.nodeData,
                position: data.position
            });
        });
        this.coreView.on('nodeDoubleClick', (data) => {
            this.model.send({
                type: 'node_double_click',
                node_id: data.nodeId,
                node_data: data.nodeData
            });
        });
        this.coreView.on('nodeHover', (nodeId) => {
            this.model.set('hovered_node', nodeId || '');
            this.model.save_changes();
        });
        // Selection changes
        this.coreView.on('selectionChange', (selectedIds) => {
            this.model.set('selected_nodes', selectedIds);
            this.model.save_changes();
        });
        // Layout changes
        this.coreView.on('layoutChange', (data) => {
            this.model.set('layout_algorithm', data.to);
            this.model.save_changes();
        });
        // Camera changes (elegant throttling)
        let cameraThrottle = null;
        this.coreView.on('cameraChange', (camera) => {
            if (cameraThrottle)
                clearTimeout(cameraThrottle);
            cameraThrottle = setTimeout(() => {
                this.model.set('camera_position', camera);
                this.model.save_changes();
            }, 100); // Elegant 100ms throttling
        });
        // Drag state
        this.coreView.on('dragStart', () => {
            this.model.set('is_dragging', true);
            this.model.save_changes();
        });
        this.coreView.on('dragEnd', () => {
            this.model.set('is_dragging', false);
            this.model.save_changes();
        });
    }
    /**
     * Sync Jupyter model changes to core engine (Jupyter → Core)
     */
    syncFromJupyter() {
        // Data updates
        this.model.on('change:nodes change:edges', () => {
            const nodes = this.model.get('nodes') || [];
            const edges = this.model.get('edges') || [];
            this.coreView.setData(nodes, edges);
        });
        // Layout changes from Python
        this.model.on('change:layout_algorithm', () => {
            const layout = this.model.get('layout_algorithm');
            this.coreView.setLayout(layout, true);
        });
        // Theme changes from Python
        this.model.on('change:theme', () => {
            const theme = this.model.get('theme');
            this.coreView.setTheme(theme);
        });
        // Selection changes from Python
        this.model.on('change:selected_nodes', () => {
            const selectedNodes = this.model.get('selected_nodes') || [];
            this.coreView.selectNodes(selectedNodes, true);
        });
        // Node positions from Python
        this.model.on('change:node_positions', () => {
            const positions = this.model.get('node_positions') || {};
            this.updateCorePositions(positions);
        });
        // Configuration changes
        this.model.on('change:width change:height', () => {
            this.updateCoreSize();
        });
    }
    /**
     * Handle custom messages from Python (elegant command pattern)
     */
    handleCustomMessage(content) {
        const { type } = content;
        switch (type) {
            case 'set_layout':
                this.coreView.setLayout(content.algorithm, content.animate !== false);
                break;
            case 'focus_node':
                this.coreView.focusOnNode(content.node_id, content.zoom || 2.0);
                break;
            case 'reset_camera':
                this.coreView.resetCamera();
                break;
            case 'update_positions':
                this.updateCorePositions(content.positions);
                break;
            default:
                console.warn(`Unknown command type: ${type}`);
        }
    }
    /**
     * Elegant position update with smooth synchronization
     */
    updateCorePositions(positions) {
        // Update core engine positions elegantly
        Object.entries(positions).forEach(([nodeId, pos]) => {
            var _a, _b;
            const nodeElement = (_b = (_a = this.coreView).findNodeElement) === null || _b === void 0 ? void 0 : _b.call(_a, nodeId);
            if (nodeElement && pos) {
                // Elegant smooth position update
                this.animateNodePosition(nodeId, pos);
            }
        });
    }
    /**
     * Elegant smooth node position animation
     */
    animateNodePosition(nodeId, targetPos) {
        // Use core engine's animation system for elegance
        if (this.coreView.animateNodeTo) {
            this.coreView.animateNodeTo(nodeId, targetPos.x, targetPos.y);
        }
    }
    /**
     * Update core engine size elegantly
     */
    updateCoreSize() {
        const width = this.model.get('width');
        const height = this.model.get('height');
        if (this.coreView.resize) {
            this.coreView.resize(width, height);
        }
    }
    /**
     * Elegant initialization with data
     */
    initialize() {
        const nodes = this.model.get('nodes') || [];
        const edges = this.model.get('edges') || [];
        const positions = this.model.get('node_positions') || {};
        // Initialize core engine with data
        this.coreView.setData(nodes, edges);
        if (Object.keys(positions).length > 0) {
            this.updateCorePositions(positions);
        }
        console.log('🎨 Elegant Jupyter-Groggy bridge initialized');
    }
    /**
     * Elegant cleanup
     */
    destroy() {
        if (this.coreView.destroy) {
            this.coreView.destroy();
        }
    }
}


/***/ }),

/***/ "./src/plugin.ts":
/*!***********************!*\
  !*** ./src/plugin.ts ***!
  \***********************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "default": () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _jupyter_widgets_base__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @jupyter-widgets/base */ "@jupyter-widgets/base");
/* harmony import */ var _jupyter_widgets_base__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_jupyter_widgets_base__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _widget__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./widget */ "./src/widget.ts");
/**
 * Elegant JupyterLab Extension Plugin
 *
 * Minimal, beautiful integration with JupyterLab's extension system.
 */


/**
 * Elegant plugin definition
 */
const plugin = {
    id: 'groggy-widgets:plugin',
    requires: [_jupyter_widgets_base__WEBPACK_IMPORTED_MODULE_0__.IJupyterWidgetRegistry],
    activate: activateWidgetExtension,
    autoStart: true
};
/**
 * Elegant activation function
 */
function activateWidgetExtension(app, registry) {
    console.log('[groggy-widgets] registering', { GroggyGraphModel: _widget__WEBPACK_IMPORTED_MODULE_1__.GroggyGraphModel, GroggyGraphView: _widget__WEBPACK_IMPORTED_MODULE_1__.GroggyGraphView });
    // Sanity: these must be functions (constructors)
    console.log('types:', typeof _widget__WEBPACK_IMPORTED_MODULE_1__.GroggyGraphModel, typeof _widget__WEBPACK_IMPORTED_MODULE_1__.GroggyGraphView);
    // Register our elegant widget with JupyterLab
    registry.registerWidget({
        name: _widget__WEBPACK_IMPORTED_MODULE_1__.MODULE_NAME,
        version: _widget__WEBPACK_IMPORTED_MODULE_1__.MODULE_VERSION,
        exports: {
            GroggyGraphModel: _widget__WEBPACK_IMPORTED_MODULE_1__.GroggyGraphModel,
            GroggyGraphView: _widget__WEBPACK_IMPORTED_MODULE_1__.GroggyGraphView
        }
    });
    console.log('✨ Elegant Groggy widget extension activated in JupyterLab');
}
/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (plugin);


/***/ }),

/***/ "./src/widget.ts":
/*!***********************!*\
  !*** ./src/widget.ts ***!
  \***********************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   GroggyGraphModel: () => (/* binding */ GroggyGraphModel),
/* harmony export */   GroggyGraphView: () => (/* binding */ GroggyGraphView),
/* harmony export */   MODULE_NAME: () => (/* binding */ MODULE_NAME),
/* harmony export */   MODULE_VERSION: () => (/* binding */ MODULE_VERSION)
/* harmony export */ });
/* harmony import */ var _jupyter_widgets_base__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @jupyter-widgets/base */ "@jupyter-widgets/base");
/* harmony import */ var _jupyter_widgets_base__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_jupyter_widgets_base__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./core */ "./src/core.ts");
/**
 * Elegant Jupyter Widget Implementation
 *
 * Pure, minimal wrapper around our core visualization engine.
 * Philosophy: Thin interface, maximum elegance, zero duplication.
 */


const MODULE_NAME = 'groggy-widgets';
const MODULE_VERSION = '0.1.0';
/**
 * Elegant Graph Model - Pure data synchronization
 */
class GroggyGraphModel extends _jupyter_widgets_base__WEBPACK_IMPORTED_MODULE_0__.DOMWidgetModel {
    defaults() {
        return Object.assign(Object.assign({}, super.defaults()), { _model_name: GroggyGraphModel.model_name, _model_module: GroggyGraphModel.model_module, _model_module_version: GroggyGraphModel.model_module_version, _view_name: GroggyGraphModel.view_name, _view_module: GroggyGraphModel.view_module, _view_module_version: GroggyGraphModel.view_module_version, 
            // Graph data (elegant sync with Python)
            nodes: [], edges: [], node_positions: {}, 
            // Visualization state (elegant configuration)
            layout_algorithm: 'force-directed', theme: 'light', width: 800, height: 600, title: 'Graph Visualization', 
            // Interactive state (elegant real-time sync)
            selected_nodes: [], hovered_node: '', camera_position: { x: 0, y: 0, zoom: 1.0 }, is_dragging: false, 
            // Feature configuration (elegant control)
            enable_drag: true, enable_pan: true, enable_zoom: true, enable_animations: true, animation_duration: 300, 
            // Style configuration (elegant theming)
            style_config: {} });
    }
}
GroggyGraphModel.model_name = 'GroggyGraphModel';
GroggyGraphModel.model_module = MODULE_NAME;
GroggyGraphModel.model_module_version = MODULE_VERSION;
GroggyGraphModel.view_name = 'GroggyGraphView';
GroggyGraphModel.view_module = MODULE_NAME;
GroggyGraphModel.view_module_version = MODULE_VERSION;
GroggyGraphModel.serializers = Object.assign({}, _jupyter_widgets_base__WEBPACK_IMPORTED_MODULE_0__.DOMWidgetModel.serializers);
/**
 * Elegant Graph View - Pure visual interface
 */
class GroggyGraphView extends _jupyter_widgets_base__WEBPACK_IMPORTED_MODULE_0__.DOMWidgetView {
    constructor() {
        super(...arguments);
        this.groggyView = null;
    }
    /**
     * Elegant rendering - minimal setup, maximum power
     */
    render() {
        // Create elegant container
        this.el.classList.add('groggy-widget-container');
        this.el.style.cssText = `
            border: 1px solid #ddd;
            border-radius: 4px;
            background: #fafafa;
            position: relative;
            overflow: hidden;
        `;
        // Initialize our elegant core bridge
        this.groggyView = new _core__WEBPACK_IMPORTED_MODULE_1__.JupyterGroggyView(this.el, this.model);
        // Set up elegant message handling
        this.model.on('msg:custom', this.handleCustomMessage, this);
        // Elegant initialization
        this.groggyView.initialize();
        console.log('✨ Elegant Groggy widget rendered');
    }
    /**
     * Elegant custom message handling
     */
    handleCustomMessage(content) {
        if (this.groggyView) {
            this.groggyView.handleCustomMessage(content);
        }
    }
    /**
     * Elegant cleanup
     */
    remove() {
        if (this.groggyView) {
            this.groggyView.destroy();
            this.groggyView = null;
        }
        super.remove();
    }
    /**
     * Elegant resize handling
     */
    onResize() {
        // Automatic resize handling through model sync - elegant!
        if (this.groggyView) {
            const width = this.el.clientWidth;
            const height = this.el.clientHeight;
            if (width > 0 && height > 0) {
                this.model.set('width', width);
                this.model.set('height', height);
                this.model.save_changes();
            }
        }
    }
}
GroggyGraphView.view_name = 'GroggyGraphView';
GroggyGraphView.view_module = MODULE_NAME;
GroggyGraphView.view_module_version = MODULE_VERSION;


/***/ }),

/***/ "@jupyter-widgets/base":
/*!****************************************!*\
  !*** external "@jupyter-widgets/base" ***!
  \****************************************/
/***/ ((module) => {

"use strict";
module.exports = __WEBPACK_EXTERNAL_MODULE__jupyter_widgets_base__;

/***/ })

/******/ 	});
/************************************************************************/
/******/ 	// The module cache
/******/ 	var __webpack_module_cache__ = {};
/******/ 	
/******/ 	// The require function
/******/ 	function __webpack_require__(moduleId) {
/******/ 		// Check if module is in cache
/******/ 		var cachedModule = __webpack_module_cache__[moduleId];
/******/ 		if (cachedModule !== undefined) {
/******/ 			return cachedModule.exports;
/******/ 		}
/******/ 		// Create a new module (and put it into the cache)
/******/ 		var module = __webpack_module_cache__[moduleId] = {
/******/ 			// no module.id needed
/******/ 			// no module.loaded needed
/******/ 			exports: {}
/******/ 		};
/******/ 	
/******/ 		// Execute the module function
/******/ 		__webpack_modules__[moduleId](module, module.exports, __webpack_require__);
/******/ 	
/******/ 		// Return the exports of the module
/******/ 		return module.exports;
/******/ 	}
/******/ 	
/************************************************************************/
/******/ 	/* webpack/runtime/compat get default export */
/******/ 	(() => {
/******/ 		// getDefaultExport function for compatibility with non-harmony modules
/******/ 		__webpack_require__.n = (module) => {
/******/ 			var getter = module && module.__esModule ?
/******/ 				() => (module['default']) :
/******/ 				() => (module);
/******/ 			__webpack_require__.d(getter, { a: getter });
/******/ 			return getter;
/******/ 		};
/******/ 	})();
/******/ 	
/******/ 	/* webpack/runtime/define property getters */
/******/ 	(() => {
/******/ 		// define getter functions for harmony exports
/******/ 		__webpack_require__.d = (exports, definition) => {
/******/ 			for(var key in definition) {
/******/ 				if(__webpack_require__.o(definition, key) && !__webpack_require__.o(exports, key)) {
/******/ 					Object.defineProperty(exports, key, { enumerable: true, get: definition[key] });
/******/ 				}
/******/ 			}
/******/ 		};
/******/ 	})();
/******/ 	
/******/ 	/* webpack/runtime/hasOwnProperty shorthand */
/******/ 	(() => {
/******/ 		__webpack_require__.o = (obj, prop) => (Object.prototype.hasOwnProperty.call(obj, prop))
/******/ 	})();
/******/ 	
/******/ 	/* webpack/runtime/make namespace object */
/******/ 	(() => {
/******/ 		// define __esModule on exports
/******/ 		__webpack_require__.r = (exports) => {
/******/ 			if(typeof Symbol !== 'undefined' && Symbol.toStringTag) {
/******/ 				Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' });
/******/ 			}
/******/ 			Object.defineProperty(exports, '__esModule', { value: true });
/******/ 		};
/******/ 	})();
/******/ 	
/************************************************************************/
var __webpack_exports__ = {};
// This entry needs to be wrapped in an IIFE because it needs to be in strict mode.
(() => {
"use strict";
/*!**********************!*\
  !*** ./src/index.ts ***!
  \**********************/
__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   GroggyGraphModel: () => (/* reexport safe */ _widget__WEBPACK_IMPORTED_MODULE_0__.GroggyGraphModel),
/* harmony export */   GroggyGraphView: () => (/* reexport safe */ _widget__WEBPACK_IMPORTED_MODULE_0__.GroggyGraphView),
/* harmony export */   MODULE_NAME: () => (/* reexport safe */ _widget__WEBPACK_IMPORTED_MODULE_0__.MODULE_NAME),
/* harmony export */   MODULE_VERSION: () => (/* reexport safe */ _widget__WEBPACK_IMPORTED_MODULE_0__.MODULE_VERSION),
/* harmony export */   "default": () => (/* reexport safe */ _plugin__WEBPACK_IMPORTED_MODULE_1__["default"]),
/* harmony export */   version: () => (/* binding */ version)
/* harmony export */ });
/* harmony import */ var _widget__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./widget */ "./src/widget.ts");
/* harmony import */ var _plugin__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./plugin */ "./src/plugin.ts");
/**
 * Elegant Main Export - Groggy Widget Module
 *
 * This is the primary entry point for the groggy-widget npm package.
 * Exports all widget classes for Jupyter extension discovery.
 */
// Core widget classes that Jupyter needs to find

// Export the JupyterLab plugin as default

// Module metadata for Jupyter
const version = '0.1.0';

})();

/******/ 	return __webpack_exports__;
/******/ })()
;
});;
//# sourceMappingURL=index.js.map