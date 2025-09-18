/**
 * 🚀 Phase 10: Performance Optimization - Level-of-Detail Renderer
 * 
 * Advanced level-of-detail rendering system for large graph visualization
 * Dynamically adjusts rendering complexity based on zoom level, viewport, and performance
 * 
 * Features:
 * - Multi-level LOD system (Far, Medium, Near, Detail views)
 * - Viewport culling for off-screen elements
 * - Dynamic quality adjustment based on performance
 * - Adaptive node/edge simplification
 * - Intelligent label management
 * - Memory-efficient rendering pipelines
 * - Performance monitoring and auto-adjustment
 */

class LODRenderer {
    constructor(canvas, graphRenderer) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.graphRenderer = graphRenderer;
        
        // LOD Configuration
        this.lodLevels = {
            DETAIL: {
                minZoom: 2.0,
                maxNodes: 100,
                renderLabels: true,
                renderEdgeLabels: true,
                nodeDetail: 'full',
                edgeDetail: 'full',
                animationsEnabled: true,
                shadowsEnabled: true
            },
            NEAR: {
                minZoom: 1.0,
                maxNodes: 500,
                renderLabels: true,
                renderEdgeLabels: false,
                nodeDetail: 'simplified',
                edgeDetail: 'simplified',
                animationsEnabled: true,
                shadowsEnabled: false
            },
            MEDIUM: {
                minZoom: 0.3,
                maxNodes: 2000,
                renderLabels: false,
                renderEdgeLabels: false,
                nodeDetail: 'basic',
                edgeDetail: 'basic',
                animationsEnabled: false,
                shadowsEnabled: false
            },
            FAR: {
                minZoom: 0.0,
                maxNodes: 10000,
                renderLabels: false,
                renderEdgeLabels: false,
                nodeDetail: 'minimal',
                edgeDetail: 'minimal',
                animationsEnabled: false,
                shadowsEnabled: false
            }
        };
        
        // Current rendering state
        this.currentLOD = 'NEAR';
        this.currentZoom = 1.0;
        this.viewport = { x: 0, y: 0, width: canvas.width, height: canvas.height };
        this.visibleNodes = new Set();
        this.visibleEdges = new Set();
        
        // Performance tracking
        this.frameTime = 0;
        this.targetFPS = 60;
        this.performanceHistory = [];
        this.adaptiveQuality = true;
        this.qualityFactor = 1.0;
        
        // Rendering caches
        this.nodeCache = new Map();
        this.edgeCache = new Map();
        this.labelCache = new Map();
        this.cacheInvalidated = true;
        
        // Culling and spatial optimization
        this.spatialGrid = new Map();
        this.gridSize = 100; // pixels
        this.cullPadding = 50; // pixels outside viewport to still render
        
        // Performance optimization flags
        this.useInstancing = true;
        this.useOffscreenCanvas = true;
        this.useBatching = true;
        this.useWebGL = false; // Future: WebGL acceleration
        
        this.init();
    }
    
    init() {
        console.log('🚀 Initializing LOD Renderer for Phase 10');
        
        this.setupEventListeners();
        this.initializeOffscreenCanvas();
        this.buildSpatialGrid();
        
        console.log('✅ LOD Renderer initialized');
    }
    
    /**
     * Set up event listeners for zoom and viewport changes
     */
    setupEventListeners() {
        // Listen for zoom changes
        this.canvas.addEventListener('wheel', (e) => {
            this.handleZoomChange(e);
        });
        
        // Listen for pan changes
        this.canvas.addEventListener('mousemove', (e) => {
            if (e.buttons === 1) { // Left mouse button
                this.handleViewportChange(e);
            }
        });
        
        // Listen for resize
        window.addEventListener('resize', () => {
            this.handleResize();
        });
        
        // Listen for graph data changes
        document.addEventListener('graphDataChanged', (e) => {
            this.invalidateCache();
            this.buildSpatialGrid();
        });
    }
    
    /**
     * Initialize offscreen canvas for pre-rendering
     */
    initializeOffscreenCanvas() {
        if (!this.useOffscreenCanvas) return;
        
        try {
            this.offscreenCanvas = new OffscreenCanvas(this.canvas.width, this.canvas.height);
            this.offscreenCtx = this.offscreenCanvas.getContext('2d');
            console.log('📱 Offscreen canvas initialized');
        } catch (e) {
            console.warn('⚠️  Offscreen canvas not supported, falling back to standard canvas');
            this.useOffscreenCanvas = false;
        }
    }
    
    /**
     * Build spatial grid for efficient culling
     */
    buildSpatialGrid() {
        this.spatialGrid.clear();
        
        if (!this.graphRenderer.nodes) return;
        
        for (const [nodeId, node] of this.graphRenderer.nodes.entries()) {
            const gridX = Math.floor(node.x / this.gridSize);
            const gridY = Math.floor(node.y / this.gridSize);
            const gridKey = `${gridX},${gridY}`;
            
            if (!this.spatialGrid.has(gridKey)) {
                this.spatialGrid.set(gridKey, { nodes: [], edges: [] });
            }
            
            this.spatialGrid.get(gridKey).nodes.push(nodeId);
        }
        
        // Add edges to grid cells they pass through
        if (this.graphRenderer.edges) {
            for (const [edgeId, edge] of this.graphRenderer.edges.entries()) {
                const sourceNode = this.graphRenderer.nodes.get(edge.source);
                const targetNode = this.graphRenderer.nodes.get(edge.target);
                
                if (sourceNode && targetNode) {
                    const gridCells = this.getEdgeGridCells(sourceNode, targetNode);
                    for (const gridKey of gridCells) {
                        if (!this.spatialGrid.has(gridKey)) {
                            this.spatialGrid.set(gridKey, { nodes: [], edges: [] });
                        }
                        this.spatialGrid.get(gridKey).edges.push(edgeId);
                    }
                }
            }
        }
        
        console.log(`🗂️  Built spatial grid with ${this.spatialGrid.size} cells`);
    }
    
    /**
     * Get grid cells that an edge passes through
     */
    getEdgeGridCells(sourceNode, targetNode) {
        const cells = new Set();
        
        // Use Bresenham-like algorithm to find cells along edge path
        const dx = Math.abs(targetNode.x - sourceNode.x);
        const dy = Math.abs(targetNode.y - sourceNode.y);
        const steps = Math.max(dx, dy) / this.gridSize;
        
        for (let i = 0; i <= steps; i++) {
            const t = steps === 0 ? 0 : i / steps;
            const x = sourceNode.x + t * (targetNode.x - sourceNode.x);
            const y = sourceNode.y + t * (targetNode.y - sourceNode.y);
            
            const gridX = Math.floor(x / this.gridSize);
            const gridY = Math.floor(y / this.gridSize);
            cells.add(`${gridX},${gridY}`);
        }
        
        return cells;
    }
    
    /**
     * Determine appropriate LOD level based on zoom and node count
     */
    determineLODLevel(zoom, nodeCount) {
        const levels = Object.keys(this.lodLevels);
        
        for (const level of levels) {
            const config = this.lodLevels[level];
            if (zoom >= config.minZoom && nodeCount <= config.maxNodes) {
                return level;
            }
        }
        
        // Fallback to most restrictive level
        return 'FAR';
    }
    
    /**
     * Update viewport culling
     */
    updateViewportCulling(zoom, panX, panY) {
        this.currentZoom = zoom;
        this.viewport = {
            x: panX - this.cullPadding,
            y: panY - this.cullPadding,
            width: this.canvas.width / zoom + 2 * this.cullPadding,
            height: this.canvas.height / zoom + 2 * this.cullPadding
        };
        
        // Find visible grid cells
        const minGridX = Math.floor(this.viewport.x / this.gridSize);
        const maxGridX = Math.floor((this.viewport.x + this.viewport.width) / this.gridSize);
        const minGridY = Math.floor(this.viewport.y / this.gridSize);
        const maxGridY = Math.floor((this.viewport.y + this.viewport.height) / this.gridSize);
        
        this.visibleNodes.clear();
        this.visibleEdges.clear();
        
        for (let gridX = minGridX; gridX <= maxGridX; gridX++) {
            for (let gridY = minGridY; gridY <= maxGridY; gridY++) {
                const gridKey = `${gridX},${gridY}`;
                const cell = this.spatialGrid.get(gridKey);
                
                if (cell) {
                    for (const nodeId of cell.nodes) {
                        const node = this.graphRenderer.nodes.get(nodeId);
                        if (node && this.isNodeInViewport(node)) {
                            this.visibleNodes.add(nodeId);
                        }
                    }
                    
                    for (const edgeId of cell.edges) {
                        this.visibleEdges.add(edgeId);
                    }
                }
            }
        }
        
        console.log(`👁️  Culled to ${this.visibleNodes.size} nodes, ${this.visibleEdges.size} edges`);
    }
    
    /**
     * Check if node is within viewport
     */
    isNodeInViewport(node) {
        return node.x >= this.viewport.x && 
               node.x <= this.viewport.x + this.viewport.width &&
               node.y >= this.viewport.y && 
               node.y <= this.viewport.y + this.viewport.height;
    }
    
    /**
     * Render with level-of-detail optimization
     */
    render(zoom, panX, panY, nodes, edges) {
        const startTime = performance.now();
        
        // Update LOD and culling
        const nodeCount = this.visibleNodes.size || nodes.size;
        this.currentLOD = this.determineLODLevel(zoom, nodeCount);
        this.updateViewportCulling(zoom, panX, panY);
        
        const lodConfig = this.lodLevels[this.currentLOD];
        
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Set up rendering context
        this.ctx.save();
        this.ctx.scale(zoom, zoom);
        this.ctx.translate(-panX, -panY);
        
        // Apply quality factor for adaptive performance
        if (this.adaptiveQuality) {
            this.ctx.globalAlpha = Math.max(0.5, this.qualityFactor);
        }
        
        // Render based on LOD level
        this.renderEdges(lodConfig, edges);
        this.renderNodes(lodConfig, nodes);
        
        if (lodConfig.renderLabels) {
            this.renderLabels(lodConfig, nodes);
        }
        
        this.ctx.restore();
        
        // Track performance
        this.frameTime = performance.now() - startTime;
        this.updatePerformanceMetrics();
        
        // Update UI indicators
        this.updateLODIndicator();
    }
    
    /**
     * Render edges with LOD optimization
     */
    renderEdges(lodConfig, edges) {
        if (!edges || this.visibleEdges.size === 0) return;
        
        const edgeDetail = lodConfig.edgeDetail;
        
        // Batch rendering for performance
        this.ctx.beginPath();
        
        for (const edgeId of this.visibleEdges) {
            const edge = edges.get(edgeId);
            if (!edge) continue;
            
            const sourceNode = this.graphRenderer.nodes.get(edge.source);
            const targetNode = this.graphRenderer.nodes.get(edge.target);
            
            if (!sourceNode || !targetNode) continue;
            
            switch (edgeDetail) {
                case 'full':
                    this.renderFullEdge(edge, sourceNode, targetNode);
                    break;
                case 'simplified':
                    this.renderSimplifiedEdge(edge, sourceNode, targetNode);
                    break;
                case 'basic':
                    this.renderBasicEdge(sourceNode, targetNode);
                    break;
                case 'minimal':
                    this.renderMinimalEdge(sourceNode, targetNode);
                    break;
            }
        }
        
        // Apply edge styling
        this.ctx.strokeStyle = this.getEdgeColor(edgeDetail);
        this.ctx.lineWidth = this.getEdgeWidth(edgeDetail) * this.qualityFactor;
        this.ctx.stroke();
    }
    
    /**
     * Render nodes with LOD optimization
     */
    renderNodes(lodConfig, nodes) {
        if (!nodes || this.visibleNodes.size === 0) return;
        
        const nodeDetail = lodConfig.nodeDetail;
        
        for (const nodeId of this.visibleNodes) {
            const node = nodes.get(nodeId);
            if (!node) continue;
            
            switch (nodeDetail) {
                case 'full':
                    this.renderFullNode(node);
                    break;
                case 'simplified':
                    this.renderSimplifiedNode(node);
                    break;
                case 'basic':
                    this.renderBasicNode(node);
                    break;
                case 'minimal':
                    this.renderMinimalNode(node);
                    break;
            }
        }
    }
    
    /**
     * Render full-detail edge
     */
    renderFullEdge(edge, sourceNode, targetNode) {
        this.ctx.moveTo(sourceNode.x, sourceNode.y);
        this.ctx.lineTo(targetNode.x, targetNode.y);
        
        // Add arrow markers if needed
        if (edge.directed) {
            this.renderArrowMarker(sourceNode, targetNode);
        }
        
        // Add edge effects
        if (edge.selected) {
            this.ctx.shadowColor = '#0066FF';
            this.ctx.shadowBlur = 4;
        }
    }
    
    /**
     * Render simplified edge
     */
    renderSimplifiedEdge(edge, sourceNode, targetNode) {
        this.ctx.moveTo(sourceNode.x, sourceNode.y);
        this.ctx.lineTo(targetNode.x, targetNode.y);
    }
    
    /**
     * Render basic edge
     */
    renderBasicEdge(sourceNode, targetNode) {
        this.ctx.moveTo(sourceNode.x, sourceNode.y);
        this.ctx.lineTo(targetNode.x, targetNode.y);
    }
    
    /**
     * Render minimal edge (just a line)
     */
    renderMinimalEdge(sourceNode, targetNode) {
        this.ctx.moveTo(sourceNode.x, sourceNode.y);
        this.ctx.lineTo(targetNode.x, targetNode.y);
    }
    
    /**
     * Render full-detail node
     */
    renderFullNode(node) {
        const radius = this.getNodeRadius(node) * this.qualityFactor;
        
        // Node shadow
        this.ctx.shadowColor = 'rgba(0, 0, 0, 0.2)';
        this.ctx.shadowBlur = 4;
        this.ctx.shadowOffsetX = 2;
        this.ctx.shadowOffsetY = 2;
        
        // Node fill
        this.ctx.beginPath();
        this.ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI);
        this.ctx.fillStyle = this.getNodeColor(node);
        this.ctx.fill();
        
        // Node stroke
        this.ctx.strokeStyle = this.getNodeStrokeColor(node);
        this.ctx.lineWidth = 2 * this.qualityFactor;
        this.ctx.stroke();
        
        // Reset shadow
        this.ctx.shadowColor = 'transparent';
        this.ctx.shadowBlur = 0;
        this.ctx.shadowOffsetX = 0;
        this.ctx.shadowOffsetY = 0;
        
        // Node icon or text
        if (node.icon) {
            this.renderNodeIcon(node, radius);
        }
    }
    
    /**
     * Render simplified node
     */
    renderSimplifiedNode(node) {
        const radius = this.getNodeRadius(node) * this.qualityFactor;
        
        this.ctx.beginPath();
        this.ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI);
        this.ctx.fillStyle = this.getNodeColor(node);
        this.ctx.fill();
        
        this.ctx.strokeStyle = this.getNodeStrokeColor(node);
        this.ctx.lineWidth = 1;
        this.ctx.stroke();
    }
    
    /**
     * Render basic node
     */
    renderBasicNode(node) {
        const radius = Math.max(2, this.getNodeRadius(node) * this.qualityFactor * 0.8);
        
        this.ctx.beginPath();
        this.ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI);
        this.ctx.fillStyle = this.getNodeColor(node);
        this.ctx.fill();
    }
    
    /**
     * Render minimal node (just a small dot)
     */
    renderMinimalNode(node) {
        const radius = Math.max(1, 3 * this.qualityFactor);
        
        this.ctx.beginPath();
        this.ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI);
        this.ctx.fillStyle = this.getNodeColor(node);
        this.ctx.fill();
    }
    
    /**
     * Render labels with intelligent culling
     */
    renderLabels(lodConfig, nodes) {
        if (!lodConfig.renderLabels) return;
        
        const maxLabels = this.getMaxLabelsForLOD();
        let labelCount = 0;
        
        // Sort nodes by importance (size, selection, etc.)
        const importantNodes = Array.from(this.visibleNodes)
            .map(id => nodes.get(id))
            .filter(node => node && node.label)
            .sort((a, b) => this.getNodeImportance(b) - this.getNodeImportance(a))
            .slice(0, maxLabels);
        
        this.ctx.font = `${12 * this.qualityFactor}px Arial`;
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        
        for (const node of importantNodes) {
            if (labelCount >= maxLabels) break;
            
            const radius = this.getNodeRadius(node);
            const labelY = node.y + radius + 15;
            
            // Label background
            this.ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
            const textWidth = this.ctx.measureText(node.label).width;
            this.ctx.fillRect(node.x - textWidth/2 - 4, labelY - 8, textWidth + 8, 16);
            
            // Label text
            this.ctx.fillStyle = '#333';
            this.ctx.fillText(node.label, node.x, labelY);
            
            labelCount++;
        }
    }
    
    /**
     * Get maximum labels for current LOD
     */
    getMaxLabelsForLOD() {
        switch (this.currentLOD) {
            case 'DETAIL': return 100;
            case 'NEAR': return 50;
            case 'MEDIUM': return 20;
            case 'FAR': return 0;
            default: return 50;
        }
    }
    
    /**
     * Calculate node importance for label priority
     */
    getNodeImportance(node) {
        let importance = 0;
        
        // Selected nodes are always important
        if (node.selected) importance += 1000;
        
        // Larger nodes are more important
        importance += this.getNodeRadius(node) * 10;
        
        // Nodes with more connections are more important
        if (node.degree) importance += node.degree * 5;
        
        // Center of viewport is more important
        const centerDistance = Math.sqrt(
            Math.pow(node.x - (this.viewport.x + this.viewport.width/2), 2) +
            Math.pow(node.y - (this.viewport.y + this.viewport.height/2), 2)
        );
        importance -= centerDistance * 0.1;
        
        return importance;
    }
    
    /**
     * Update performance metrics and adaptive quality
     */
    updatePerformanceMetrics() {
        this.performanceHistory.push(this.frameTime);
        if (this.performanceHistory.length > 60) {
            this.performanceHistory.shift();
        }
        
        const avgFrameTime = this.performanceHistory.reduce((a, b) => a + b, 0) / this.performanceHistory.length;
        const currentFPS = 1000 / avgFrameTime;
        
        if (this.adaptiveQuality) {
            if (currentFPS < this.targetFPS * 0.8) {
                // Performance is poor, reduce quality
                this.qualityFactor = Math.max(0.3, this.qualityFactor * 0.95);
            } else if (currentFPS > this.targetFPS * 1.1) {
                // Performance is good, increase quality
                this.qualityFactor = Math.min(1.0, this.qualityFactor * 1.02);
            }
        }
        
        // Dispatch performance event
        document.dispatchEvent(new CustomEvent('lodPerformanceUpdate', {
            detail: {
                frameTime: this.frameTime,
                fps: currentFPS,
                qualityFactor: this.qualityFactor,
                lodLevel: this.currentLOD,
                visibleNodes: this.visibleNodes.size,
                visibleEdges: this.visibleEdges.size
            }
        }));
    }
    
    /**
     * Update LOD indicator in UI
     */
    updateLODIndicator() {
        const indicator = document.getElementById('lod-indicator');
        if (indicator) {
            indicator.textContent = `LOD: ${this.currentLOD} | Quality: ${Math.round(this.qualityFactor * 100)}%`;
            indicator.className = `lod-indicator lod-${this.currentLOD.toLowerCase()}`;
        }
    }
    
    /**
     * Handle zoom change events
     */
    handleZoomChange(event) {
        // This would be called by the main graph renderer
        this.invalidateCache();
    }
    
    /**
     * Handle viewport change events
     */
    handleViewportChange(event) {
        // This would be called by the main graph renderer
        this.cacheInvalidated = true;
    }
    
    /**
     * Handle canvas resize
     */
    handleResize() {
        this.viewport.width = this.canvas.width;
        this.viewport.height = this.canvas.height;
        
        if (this.useOffscreenCanvas && this.offscreenCanvas) {
            this.offscreenCanvas.width = this.canvas.width;
            this.offscreenCanvas.height = this.canvas.height;
        }
        
        this.invalidateCache();
    }
    
    /**
     * Invalidate rendering caches
     */
    invalidateCache() {
        this.cacheInvalidated = true;
        this.nodeCache.clear();
        this.edgeCache.clear();
        this.labelCache.clear();
    }
    
    /**
     * Helper methods for styling
     */
    getNodeRadius(node) {
        return node.radius || 8;
    }
    
    getNodeColor(node) {
        if (node.selected) return '#FF3B30';
        if (node.highlighted) return '#34C759';
        return node.color || '#007AFF';
    }
    
    getNodeStrokeColor(node) {
        return node.strokeColor || '#FFFFFF';
    }
    
    getEdgeColor(detail) {
        switch (detail) {
            case 'full': return '#333333';
            case 'simplified': return '#666666';
            case 'basic': return '#999999';
            case 'minimal': return '#CCCCCC';
            default: return '#999999';
        }
    }
    
    getEdgeWidth(detail) {
        switch (detail) {
            case 'full': return 2;
            case 'simplified': return 1.5;
            case 'basic': return 1;
            case 'minimal': return 0.5;
            default: return 1;
        }
    }
    
    /**
     * Render arrow marker for directed edges
     */
    renderArrowMarker(sourceNode, targetNode) {
        const angle = Math.atan2(targetNode.y - sourceNode.y, targetNode.x - sourceNode.x);
        const arrowLength = 10;
        const arrowAngle = Math.PI / 6;
        
        const endX = targetNode.x - Math.cos(angle) * this.getNodeRadius(targetNode);
        const endY = targetNode.y - Math.sin(angle) * this.getNodeRadius(targetNode);
        
        this.ctx.beginPath();
        this.ctx.moveTo(endX, endY);
        this.ctx.lineTo(
            endX - arrowLength * Math.cos(angle - arrowAngle),
            endY - arrowLength * Math.sin(angle - arrowAngle)
        );
        this.ctx.moveTo(endX, endY);
        this.ctx.lineTo(
            endX - arrowLength * Math.cos(angle + arrowAngle),
            endY - arrowLength * Math.sin(angle + arrowAngle)
        );
        this.ctx.stroke();
    }
    
    /**
     * Render node icon
     */
    renderNodeIcon(node, radius) {
        this.ctx.fillStyle = '#FFFFFF';
        this.ctx.font = `${radius}px Arial`;
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText(node.icon, node.x, node.y);
    }
    
    /**
     * Get current performance stats
     */
    getPerformanceStats() {
        return {
            frameTime: this.frameTime,
            fps: 1000 / this.frameTime,
            qualityFactor: this.qualityFactor,
            lodLevel: this.currentLOD,
            visibleNodes: this.visibleNodes.size,
            visibleEdges: this.visibleEdges.size,
            totalGridCells: this.spatialGrid.size
        };
    }
    
    /**
     * Configure LOD settings
     */
    configureLOD(settings) {
        if (settings.levels) {
            this.lodLevels = { ...this.lodLevels, ...settings.levels };
        }
        
        if (settings.adaptiveQuality !== undefined) {
            this.adaptiveQuality = settings.adaptiveQuality;
        }
        
        if (settings.targetFPS) {
            this.targetFPS = settings.targetFPS;
        }
        
        if (settings.gridSize) {
            this.gridSize = settings.gridSize;
            this.buildSpatialGrid();
        }
        
        console.log('🎛️  LOD configuration updated');
    }
    
    /**
     * Cleanup method
     */
    destroy() {
        this.invalidateCache();
        this.spatialGrid.clear();
        this.performanceHistory = [];
        
        console.log('🧹 LOD Renderer cleaned up');
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = LODRenderer;
}