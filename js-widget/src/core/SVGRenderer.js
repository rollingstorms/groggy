/**
 * 🎨 SVGRenderer - Efficient SVG Rendering Engine
 * 
 * Unified SVG rendering for all visualization environments.
 * Provides consistent, optimized rendering across Jupyter widgets,
 * WebSocket streaming, and static exports.
 * 
 * Features:
 * - Efficient DOM manipulation with minimal redraws
 * - Level-of-detail (LOD) rendering for performance
 * - Smooth animations and transitions
 * - Customizable themes and styling
 * - Responsive scaling and viewBox management
 * - Accessibility support
 */

export class SVGRenderer {
    constructor(config = {}) {
        this.config = {
            // Node styling
            nodeRadius: config.nodeRadius || 20,
            nodeStroke: config.nodeStroke || '#333',
            nodeStrokeWidth: config.nodeStrokeWidth || 2,
            nodeFill: config.nodeFill || '#4CAF50',
            nodeOpacity: config.nodeOpacity || 1.0,
            
            // Edge styling
            edgeStroke: config.edgeStroke || '#999',
            edgeStrokeWidth: config.edgeStrokeWidth || 2,
            edgeOpacity: config.edgeOpacity || 0.6,
            
            // Selection styling
            selectedNodeStroke: config.selectedNodeStroke || '#FF4444',
            selectedNodeStrokeWidth: config.selectedNodeStrokeWidth || 3,
            hoveredNodeOpacity: config.hoveredNodeOpacity || 0.8,
            
            // Effects
            enableShadows: config.enableShadows !== false,
            enableAnimations: config.enableAnimations !== false,
            animationDuration: config.animationDuration || 300,
            
            // Performance
            lodThreshold: config.lodThreshold || 1000,
            enableLOD: config.enableLOD !== false,
            renderBatchSize: config.renderBatchSize || 100,
            
            // Colors
            backgroundColor: config.backgroundColor || '#ffffff',
            nodeColorScheme: config.nodeColorScheme || 'default',
            
            ...config
        };
        
        // SVG elements
        this.svg = null;
        this.mainGroup = null;
        this.edgesGroup = null;
        this.nodesGroup = null;
        this.labelsGroup = null;
        
        // Element maps for efficient updates
        this.nodeElements = new Map();
        this.edgeElements = new Map();
        this.labelElements = new Map();
        
        // Rendering state
        this.currentScale = 1;
        this.currentTranslate = { x: 0, y: 0 };
        this.isRendering = false;
        this.renderQueue = [];
        
        // Performance tracking
        this.renderTimes = [];
        this.lastRenderTime = 0;
        
        // Color schemes
        this.colorSchemes = {
            default: {
                'person': '#FF6B6B',
                'team': '#4ECDC4', 
                'project': '#45B7D1',
                'executive': '#96CEB4',
                'manager': '#FFEAA7',
                'lead': '#DDA0DD',
                'developer': '#98D8C8',
                'product': '#F7DC6F',
                'application': '#85C1E9',
                'service': '#F8C471',
                'storage': '#82E0AA',
                'infrastructure': '#D7BDE2',
                'core': '#F1948A',
                'satellite': '#AED6F1'
            },
            dark: {
                'person': '#E74C3C',
                'team': '#1ABC9C', 
                'project': '#3498DB',
                'executive': '#27AE60',
                'manager': '#F39C12',
                'lead': '#9B59B6',
                'developer': '#16A085',
                'product': '#F1C40F',
                'application': '#2980B9',
                'service': '#E67E22',
                'storage': '#27AE60',
                'infrastructure': '#8E44AD',
                'core': '#C0392B',
                'satellite': '#2C3E50'
            }
        };
        
        console.log('🎨 SVGRenderer initialized with efficient rendering pipeline');
    }
    
    /**
     * 🏗️ Create SVG element with proper structure
     */
    createSVG(width, height) {
        this.svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        this.svg.setAttribute('width', width);
        this.svg.setAttribute('height', height);
        this.svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
        this.svg.style.background = this.config.backgroundColor;
        this.svg.style.cursor = 'grab';
        
        // Create main group for transformations
        this.mainGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.mainGroup.setAttribute('class', 'groggy-main-group');
        this.svg.appendChild(this.mainGroup);
        
        // Create sub-groups for organized rendering
        this.edgesGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.edgesGroup.setAttribute('class', 'groggy-edges');
        this.mainGroup.appendChild(this.edgesGroup);
        
        this.nodesGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.nodesGroup.setAttribute('class', 'groggy-nodes');
        this.mainGroup.appendChild(this.nodesGroup);
        
        this.labelsGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.labelsGroup.setAttribute('class', 'groggy-labels');
        this.mainGroup.appendChild(this.labelsGroup);
        
        // Add CSS filters for effects
        this.createSVGFilters();
        
        return this.svg;
    }
    
    /**
     * 🎭 Create SVG filters for visual effects
     */
    createSVGFilters() {
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        
        // Drop shadow filter
        if (this.config.enableShadows) {
            const shadowFilter = document.createElementNS('http://www.w3.org/2000/svg', 'filter');
            shadowFilter.setAttribute('id', 'groggy-shadow');
            shadowFilter.setAttribute('x', '-50%');
            shadowFilter.setAttribute('y', '-50%');
            shadowFilter.setAttribute('width', '200%');
            shadowFilter.setAttribute('height', '200%');
            
            const feDropShadow = document.createElementNS('http://www.w3.org/2000/svg', 'feDropShadow');
            feDropShadow.setAttribute('dx', '2');
            feDropShadow.setAttribute('dy', '2');
            feDropShadow.setAttribute('stdDeviation', '3');
            feDropShadow.setAttribute('flood-opacity', '0.3');
            
            shadowFilter.appendChild(feDropShadow);
            defs.appendChild(shadowFilter);
        }
        
        // Glow filter for selected nodes
        const glowFilter = document.createElementNS('http://www.w3.org/2000/svg', 'filter');
        glowFilter.setAttribute('id', 'groggy-glow');
        glowFilter.setAttribute('x', '-50%');
        glowFilter.setAttribute('y', '-50%');
        glowFilter.setAttribute('width', '200%');
        glowFilter.setAttribute('height', '200%');
        
        const feGaussianBlur = document.createElementNS('http://www.w3.org/2000/svg', 'feGaussianBlur');
        feGaussianBlur.setAttribute('stdDeviation', '3');
        feGaussianBlur.setAttribute('result', 'coloredBlur');
        
        const feMerge = document.createElementNS('http://www.w3.org/2000/svg', 'feMerge');
        const feMergeNode1 = document.createElementNS('http://www.w3.org/2000/svg', 'feMergeNode');
        feMergeNode1.setAttribute('in', 'coloredBlur');
        const feMergeNode2 = document.createElementNS('http://www.w3.org/2000/svg', 'feMergeNode');
        feMergeNode2.setAttribute('in', 'SourceGraphic');
        
        feMerge.appendChild(feMergeNode1);
        feMerge.appendChild(feMergeNode2);
        glowFilter.appendChild(feGaussianBlur);
        glowFilter.appendChild(feMerge);
        defs.appendChild(glowFilter);
        
        this.svg.appendChild(defs);
    }
    
    /**
     * 🎯 Set rendering data
     */
    setData(nodes, edges) {
        this.nodes = nodes;
        this.edges = edges;
        
        // Clear existing elements
        this.clearAllElements();
        
        console.log(`🎨 Renderer set with ${nodes.length} nodes, ${edges.length} edges`);
    }
    
    /**
     * 🖼️ Main render method
     */
    render(renderData) {
        if (this.isRendering) {
            this.renderQueue.push(renderData);
            return;
        }
        
        this.isRendering = true;
        const startTime = performance.now();
        
        const { nodes, edges, positions, selectedNodes, hoveredNode, width, height } = renderData;
        
        // Update SVG dimensions if needed
        if (width && height) {
            this.updateSVGDimensions(width, height);
        }
        
        // Determine if we should use LOD rendering
        const useLOD = this.config.enableLOD && nodes.length > this.config.lodThreshold;
        
        // Render elements
        this.renderEdges(edges, positions, useLOD);
        this.renderNodes(nodes, positions, selectedNodes, hoveredNode, useLOD);
        this.renderLabels(nodes, positions, useLOD);
        
        // Performance tracking
        const renderTime = performance.now() - startTime;
        this.updateRenderMetrics(renderTime);
        
        this.isRendering = false;
        
        // Process queued renders
        if (this.renderQueue.length > 0) {
            const nextRender = this.renderQueue.shift();
            this.render(nextRender);
        }
    }
    
    /**
     * 🔗 Render edges
     */
    renderEdges(edges, positions, useLOD) {
        edges.forEach(edge => {
            const sourcePos = positions.get(edge.source);
            const targetPos = positions.get(edge.target);
            
            if (!sourcePos || !targetPos) return;
            
            let edgeElement = this.edgeElements.get(edge.id);
            
            if (!edgeElement) {
                edgeElement = this.createEdgeElement(edge);
                this.edgeElements.set(edge.id, edgeElement);
                this.edgesGroup.appendChild(edgeElement);
            }
            
            // Update position
            edgeElement.setAttribute('x1', sourcePos.x);
            edgeElement.setAttribute('y1', sourcePos.y);
            edgeElement.setAttribute('x2', targetPos.x);
            edgeElement.setAttribute('y2', targetPos.y);
            
            // Apply LOD if needed
            if (useLOD) {
                this.applyEdgeLOD(edgeElement);
            }
        });
        
        // Remove edges that no longer exist
        this.cleanupElements(this.edgeElements, edges.map(e => e.id));
    }
    
    /**
     * ⭕ Render nodes
     */
    renderNodes(nodes, positions, selectedNodes, hoveredNode, useLOD) {
        nodes.forEach(node => {
            const pos = positions.get(node.id);
            if (!pos) return;
            
            let nodeElement = this.nodeElements.get(node.id);
            
            if (!nodeElement) {
                nodeElement = this.createNodeElement(node);
                this.nodeElements.set(node.id, nodeElement);
                this.nodesGroup.appendChild(nodeElement);
            }
            
            // Update position
            nodeElement.setAttribute('cx', pos.x);
            nodeElement.setAttribute('cy', pos.y);
            
            // Update selection state
            const isSelected = selectedNodes && selectedNodes.has(node.id);
            const isHovered = hoveredNode === node.id;
            
            this.updateNodeAppearance(nodeElement, node, isSelected, isHovered);
            
            // Apply LOD if needed
            if (useLOD) {
                this.applyNodeLOD(nodeElement, node);
            }
        });
        
        // Remove nodes that no longer exist
        this.cleanupElements(this.nodeElements, nodes.map(n => n.id));
    }
    
    /**
     * 🏷️ Render labels
     */
    renderLabels(nodes, positions, useLOD) {
        // Only render labels if scale is sufficient or LOD is disabled
        const shouldRenderLabels = !useLOD || this.currentScale > 0.8;
        
        if (!shouldRenderLabels) {
            this.labelsGroup.style.display = 'none';
            return;
        }
        
        this.labelsGroup.style.display = 'block';
        
        nodes.forEach(node => {
            const pos = positions.get(node.id);
            if (!pos || !node.label) return;
            
            let labelElement = this.labelElements.get(node.id);
            
            if (!labelElement) {
                labelElement = this.createLabelElement(node);
                this.labelElements.set(node.id, labelElement);
                this.labelsGroup.appendChild(labelElement);
            }
            
            // Update position
            labelElement.setAttribute('x', pos.x);
            labelElement.setAttribute('y', pos.y + 5);
        });
        
        // Remove labels that no longer exist
        this.cleanupElements(this.labelElements, 
            nodes.filter(n => n.label).map(n => n.id));
    }
    
    /**
     * 🏗️ Create edge element
     */
    createEdgeElement(edge) {
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('id', `edge-${edge.id}`);
        line.setAttribute('stroke', this.config.edgeStroke);
        line.setAttribute('stroke-width', this.config.edgeStrokeWidth);
        line.setAttribute('opacity', this.config.edgeOpacity);
        line.setAttribute('class', 'groggy-edge');
        
        // Add custom styling if provided
        if (edge.style) {
            Object.entries(edge.style).forEach(([key, value]) => {
                line.setAttribute(key, value);
            });
        }
        
        return line;
    }
    
    /**
     * ⭕ Create node element
     */
    createNodeElement(node) {
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('id', `node-${node.id}`);
        circle.setAttribute('r', node.size || this.config.nodeRadius);
        circle.setAttribute('fill', this.getNodeColor(node));
        circle.setAttribute('stroke', this.config.nodeStroke);
        circle.setAttribute('stroke-width', this.config.nodeStrokeWidth);
        circle.setAttribute('opacity', this.config.nodeOpacity);
        circle.setAttribute('class', 'groggy-node');
        circle.style.cursor = 'pointer';
        
        // Add shadow filter if enabled
        if (this.config.enableShadows) {
            circle.style.filter = 'url(#groggy-shadow)';
        }
        
        // Add custom styling if provided
        if (node.style) {
            Object.entries(node.style).forEach(([key, value]) => {
                circle.setAttribute(key, value);
            });
        }
        
        return circle;
    }
    
    /**
     * 🏷️ Create label element
     */
    createLabelElement(node) {
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('id', `label-${node.id}`);
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('font-family', 'Arial, sans-serif');
        text.setAttribute('font-size', '12');
        text.setAttribute('fill', '#333');
        text.setAttribute('pointer-events', 'none');
        text.setAttribute('class', 'groggy-label');
        text.textContent = node.label || node.id;
        
        return text;
    }
    
    /**
     * 🎨 Update node appearance based on state
     */
    updateNodeAppearance(nodeElement, node, isSelected, isHovered) {
        if (isSelected) {
            nodeElement.setAttribute('stroke', this.config.selectedNodeStroke);
            nodeElement.setAttribute('stroke-width', this.config.selectedNodeStrokeWidth);
            nodeElement.style.filter = 'url(#groggy-glow)';
        } else {
            nodeElement.setAttribute('stroke', this.config.nodeStroke);
            nodeElement.setAttribute('stroke-width', this.config.nodeStrokeWidth);
            if (this.config.enableShadows) {
                nodeElement.style.filter = 'url(#groggy-shadow)';
            } else {
                nodeElement.style.filter = 'none';
            }
        }
        
        if (isHovered) {
            nodeElement.setAttribute('opacity', this.config.hoveredNodeOpacity);
        } else {
            nodeElement.setAttribute('opacity', this.config.nodeOpacity);
        }
    }
    
    /**
     * 🎯 Get node color based on group/type
     */
    getNodeColor(node) {
        const scheme = this.colorSchemes[this.config.nodeColorScheme] || this.colorSchemes.default;
        return node.color || scheme[node.group] || scheme[node.type] || this.config.nodeFill;
    }
    
    /**
     * 📏 Apply Level-of-Detail (LOD) rendering for edges
     */
    applyEdgeLOD(edgeElement) {
        if (this.currentScale < 0.5) {
            edgeElement.style.display = 'none';
        } else if (this.currentScale < 0.8) {
            edgeElement.setAttribute('stroke-width', '1');
            edgeElement.setAttribute('opacity', '0.3');
            edgeElement.style.display = 'block';
        } else {
            edgeElement.setAttribute('stroke-width', this.config.edgeStrokeWidth);
            edgeElement.setAttribute('opacity', this.config.edgeOpacity);
            edgeElement.style.display = 'block';
        }
    }
    
    /**
     * 📏 Apply Level-of-Detail (LOD) rendering for nodes
     */
    applyNodeLOD(nodeElement, node) {
        if (this.currentScale < 0.3) {
            nodeElement.setAttribute('r', '3');
            nodeElement.setAttribute('stroke-width', '0');
        } else if (this.currentScale < 0.6) {
            nodeElement.setAttribute('r', '6');
            nodeElement.setAttribute('stroke-width', '1');
        } else {
            nodeElement.setAttribute('r', node.size || this.config.nodeRadius);
            nodeElement.setAttribute('stroke-width', this.config.nodeStrokeWidth);
        }
    }
    
    /**
     * 📐 Update SVG dimensions
     */
    updateSVGDimensions(width, height) {
        if (this.svg) {
            this.svg.setAttribute('width', width);
            this.svg.setAttribute('height', height);
            this.svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
        }
    }
    
    /**
     * 🔄 Update scale and translation for zoom/pan
     */
    updateTransform(scale, translate) {
        this.currentScale = scale;
        this.currentTranslate = translate;
        
        if (this.mainGroup) {
            this.mainGroup.setAttribute('transform', 
                `translate(${translate.x}, ${translate.y}) scale(${scale})`);
        }
    }
    
    /**
     * 🧹 Clean up removed elements
     */
    cleanupElements(elementMap, currentIds) {
        const currentIdSet = new Set(currentIds);
        
        for (const [id, element] of elementMap.entries()) {
            if (!currentIdSet.has(id)) {
                if (element.parentNode) {
                    element.parentNode.removeChild(element);
                }
                elementMap.delete(id);
            }
        }
    }
    
    /**
     * 🗑️ Clear all elements
     */
    clearAllElements() {
        if (this.edgesGroup) this.edgesGroup.innerHTML = '';
        if (this.nodesGroup) this.nodesGroup.innerHTML = '';
        if (this.labelsGroup) this.labelsGroup.innerHTML = '';
        
        this.nodeElements.clear();
        this.edgeElements.clear();
        this.labelElements.clear();
    }
    
    /**
     * 📊 Update render performance metrics
     */
    updateRenderMetrics(renderTime) {
        this.lastRenderTime = renderTime;
        this.renderTimes.push(renderTime);
        
        // Keep only last 60 renders for average
        if (this.renderTimes.length > 60) {
            this.renderTimes.shift();
        }
    }
    
    /**
     * ⚙️ Update configuration
     */
    updateConfig(newConfig) {
        this.config = { ...this.config, ...newConfig };
        
        // Re-apply styles to existing elements if needed
        if (this.svg) {
            this.svg.style.background = this.config.backgroundColor;
        }
    }
    
    /**
     * 📈 Get performance metrics
     */
    getPerformanceMetrics() {
        const avgRenderTime = this.renderTimes.length > 0 
            ? this.renderTimes.reduce((a, b) => a + b, 0) / this.renderTimes.length 
            : 0;
        
        return {
            lastRenderTime: this.lastRenderTime,
            averageRenderTime: avgRenderTime,
            renderFPS: avgRenderTime > 0 ? 1000 / avgRenderTime : 0,
            nodeCount: this.nodeElements.size,
            edgeCount: this.edgeElements.size,
            currentScale: this.currentScale
        };
    }
    
    /**
     * 🧹 Cleanup and destroy
     */
    destroy() {
        this.clearAllElements();
        
        if (this.svg && this.svg.parentNode) {
            this.svg.parentNode.removeChild(this.svg);
        }
        
        this.renderTimes = [];
        this.renderQueue = [];
        
        console.log('🎨 SVGRenderer destroyed');
    }
}