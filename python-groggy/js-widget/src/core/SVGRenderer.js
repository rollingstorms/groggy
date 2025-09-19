/**
 * 🎨 Unified SVG Renderer
 * 
 * Efficient SVG rendering with Level-of-Detail (LOD) optimization
 * Consistent rendering pipeline across all visualization environments
 */

export class SVGRenderer {
    constructor(config = {}) {
        this.config = {
            width: 800,
            height: 600,
            nodeRadius: 8,
            edgeWidth: 1,
            nodeColor: '#007bff',
            edgeColor: '#999',
            backgroundColor: '#ffffff',
            enableLOD: true,        // Level-of-detail rendering
            lodThreshold: 1000,     // Switch to LOD when > 1000 nodes
            enableLabels: true,
            labelThreshold: 100,    // Hide labels when > 100 nodes
            ...config
        };
        
        this.element = null;
        this.svg = null;
        this.nodesGroup = null;
        this.edgesGroup = null;
        this.labelsGroup = null;
        
        // Performance tracking
        this.frameCount = 0;
        this.lastRenderTime = 0;
        this.renderTimes = [];
        
        console.log('🎨 SVGRenderer initialized:', this.config);
    }
    
    /**
     * Attach renderer to DOM element
     */
    attachToElement(element) {
        this.element = element;
        this.createSVGStructure();
        console.log('🔗 SVGRenderer attached to element');
    }
    
    /**
     * Create SVG structure
     */
    createSVGStructure() {
        if (!this.element) return;
        
        // Clear existing content
        this.element.innerHTML = '';
        
        // Create SVG
        this.svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        this.svg.setAttribute('width', this.config.width);
        this.svg.setAttribute('height', this.config.height);
        this.svg.style.backgroundColor = this.config.backgroundColor;
        this.svg.style.border = '1px solid #ddd';
        
        // Create groups for layered rendering (edges behind nodes)
        this.edgesGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.edgesGroup.setAttribute('class', 'edges');
        
        this.nodesGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.nodesGroup.setAttribute('class', 'nodes');
        
        this.labelsGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.labelsGroup.setAttribute('class', 'labels');
        
        // Add groups to SVG in correct order
        this.svg.appendChild(this.edgesGroup);
        this.svg.appendChild(this.nodesGroup);
        this.svg.appendChild(this.labelsGroup);
        
        // Add to DOM
        this.element.appendChild(this.svg);
    }
    
    /**
     * 🎯 Main render method - used by all environments
     */
    render(renderData) {
        if (!this.svg) return;
        
        const startTime = performance.now();
        
        const { nodes, edges, positions, frameCount } = renderData;
        
        // Choose rendering strategy based on data size
        const shouldUseLOD = this.config.enableLOD && nodes.length > this.config.lodThreshold;
        const shouldShowLabels = this.config.enableLabels && nodes.length <= this.config.labelThreshold;
        
        if (shouldUseLOD) {
            this.renderLOD(nodes, edges, positions);
        } else {
            this.renderFull(nodes, edges, positions, shouldShowLabels);
        }
        
        // Performance tracking
        const renderTime = performance.now() - startTime;
        this.recordRenderTime(renderTime);
        this.frameCount = frameCount || this.frameCount + 1;
    }
    
    /**
     * Full quality rendering
     */
    renderFull(nodes, edges, positions, showLabels = true) {
        // Render edges first (behind nodes)
        this.renderEdges(edges, positions);
        
        // Render nodes
        this.renderNodes(nodes, positions);
        
        // Render labels if enabled
        if (showLabels) {
            this.renderLabels(nodes, positions);
        } else {
            this.labelsGroup.innerHTML = '';
        }
    }
    
    /**
     * Level-of-detail rendering for large graphs
     */
    renderLOD(nodes, edges, positions) {
        // Simplified rendering for performance
        // - Smaller nodes
        // - No labels
        // - Thinner edges
        // - Possible node clustering
        
        this.renderEdgesLOD(edges, positions);
        this.renderNodesLOD(nodes, positions);
        this.labelsGroup.innerHTML = '';
    }
    
    /**
     * Render edges
     */
    renderEdges(edges, positions) {
        // Clear existing edges
        this.edgesGroup.innerHTML = '';
        
        edges.forEach(edge => {
            const sourcePos = positions.get ? positions.get(edge.source) : positions[edge.source];
            const targetPos = positions.get ? positions.get(edge.target) : positions[edge.target];
            
            if (!sourcePos || !targetPos) return;
            
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', sourcePos.x);
            line.setAttribute('y1', sourcePos.y);
            line.setAttribute('x2', targetPos.x);
            line.setAttribute('y2', targetPos.y);
            line.setAttribute('stroke', edge.color || this.config.edgeColor);
            line.setAttribute('stroke-width', edge.width || this.config.edgeWidth);
            line.setAttribute('opacity', '0.8');
            
            // Add edge ID for interaction
            line.setAttribute('data-edge-id', edge.id);
            
            this.edgesGroup.appendChild(line);
        });
    }
    
    /**
     * Render edges (LOD version)
     */
    renderEdgesLOD(edges, positions) {
        this.edgesGroup.innerHTML = '';
        
        edges.forEach(edge => {
            const sourcePos = positions.get ? positions.get(edge.source) : positions[edge.source];
            const targetPos = positions.get ? positions.get(edge.target) : positions[edge.target];
            
            if (!sourcePos || !targetPos) return;
            
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', sourcePos.x);
            line.setAttribute('y1', sourcePos.y);
            line.setAttribute('x2', targetPos.x);
            line.setAttribute('y2', targetPos.y);
            line.setAttribute('stroke', this.config.edgeColor);
            line.setAttribute('stroke-width', '0.5'); // Thinner for LOD
            line.setAttribute('opacity', '0.6');
            
            this.edgesGroup.appendChild(line);
        });
    }
    
    /**
     * Render nodes
     */
    renderNodes(nodes, positions) {
        // Clear existing nodes
        this.nodesGroup.innerHTML = '';
        
        nodes.forEach(node => {
            const position = positions.get ? positions.get(node.id) : positions[node.id];
            if (!position) return;
            
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', position.x);
            circle.setAttribute('cy', position.y);
            circle.setAttribute('r', node.size || this.config.nodeRadius);
            circle.setAttribute('fill', node.color || this.config.nodeColor);
            circle.setAttribute('stroke', '#ffffff');
            circle.setAttribute('stroke-width', '2');
            circle.setAttribute('opacity', '0.9');
            
            // Add node ID for interaction
            circle.setAttribute('data-node-id', node.id);
            
            // Make interactive
            circle.style.cursor = 'pointer';
            
            this.nodesGroup.appendChild(circle);
        });
    }
    
    /**
     * Render nodes (LOD version)
     */
    renderNodesLOD(nodes, positions) {
        this.nodesGroup.innerHTML = '';
        
        nodes.forEach(node => {
            const position = positions.get ? positions.get(node.id) : positions[node.id];
            if (!position) return;
            
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', position.x);
            circle.setAttribute('cy', position.y);
            circle.setAttribute('r', '3'); // Smaller for LOD
            circle.setAttribute('fill', this.config.nodeColor);
            circle.setAttribute('stroke', 'none');
            circle.setAttribute('opacity', '0.8');
            
            this.nodesGroup.appendChild(circle);
        });
    }
    
    /**
     * Render labels
     */
    renderLabels(nodes, positions) {
        this.labelsGroup.innerHTML = '';
        
        nodes.forEach(node => {
            if (!node.label) return;
            
            const position = positions.get ? positions.get(node.id) : positions[node.id];
            if (!position) return;
            
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', position.x);
            text.setAttribute('y', position.y - (node.size || this.config.nodeRadius) - 5);
            text.setAttribute('text-anchor', 'middle');
            text.setAttribute('font-family', 'Arial, sans-serif');
            text.setAttribute('font-size', '12');
            text.setAttribute('fill', '#333');
            text.textContent = node.label;
            
            this.labelsGroup.appendChild(text);
        });
    }
    
    /**
     * Record render time for performance monitoring
     */
    recordRenderTime(time) {
        this.renderTimes.push(time);
        if (this.renderTimes.length > 60) { // Keep last 60 frames
            this.renderTimes.shift();
        }
        this.lastRenderTime = time;
    }
    
    /**
     * Get performance statistics
     */
    getPerformanceStats() {
        if (this.renderTimes.length === 0) return null;
        
        const avg = this.renderTimes.reduce((a, b) => a + b, 0) / this.renderTimes.length;
        const max = Math.max(...this.renderTimes);
        const min = Math.min(...this.renderTimes);
        
        return {
            averageRenderTime: avg.toFixed(2),
            maxRenderTime: max.toFixed(2),
            minRenderTime: min.toFixed(2),
            frameCount: this.frameCount,
            fps: (1000 / avg).toFixed(1)
        };
    }
    
    /**
     * Update configuration
     */
    updateConfig(newConfig) {
        this.config = { ...this.config, ...newConfig };
        
        // Update SVG if size changed
        if (this.svg && (newConfig.width || newConfig.height)) {
            this.svg.setAttribute('width', this.config.width);
            this.svg.setAttribute('height', this.config.height);
        }
        
        console.log('⚙️ SVGRenderer config updated:', this.config);
    }
    
    /**
     * Clear renderer
     */
    clear() {
        if (this.nodesGroup) this.nodesGroup.innerHTML = '';
        if (this.edgesGroup) this.edgesGroup.innerHTML = '';
        if (this.labelsGroup) this.labelsGroup.innerHTML = '';
    }
    
    /**
     * Export current SVG as string
     */
    exportSVG() {
        if (!this.svg) return null;
        
        const serializer = new XMLSerializer();
        return serializer.serializeToString(this.svg);
    }
}

export default SVGRenderer;